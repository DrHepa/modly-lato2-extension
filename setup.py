"""Provision the complete, isolated LATO.2 runtime for Modly.

The current Modly host passes one JSON object. Modly 0.2.x used the legacy
``<python_exe> <ext_dir> <gpu_sm> [cuda_version]`` form, which remains
supported so Repair also works for users upgrading an existing installation.

All immutable model/code assets live below Modly's configured ``models_dir``.
Only the extension virtual environment and the final, generated runtime
configuration live beside this file.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import BinaryIO, Iterator, Mapping, Sequence

# Electron invokes setup through ``runpy.run_path`` from Modly's own working
# directory, so the extension root is not guaranteed to be on sys.path.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lato2_modly import dependencies as deps
from lato2_modly.assets import ensure_snapshot, verify_asset, verify_snapshot
from lato2_modly.constants import (
    ASSETS,
    EXTENSION_ID,
    EXTENSION_VERSION,
    READY_MARKER_FILENAME,
    READY_SCHEMA_VERSION,
    REVISION_ID,
    RUNTIME_CONFIG_FILENAME,
    SETUP_LOCK_FILENAME,
)
from lato2_modly.ovoxel_cpu import materialize_ovoxel_cpu_build
from lato2_modly.paths import (
    SETUP_MODELS_PAYLOAD_KEYS,
    current_platform_name,
    normalize_platform_name,
    owned_snapshot_directory,
    resolve_models_root,
    safe_snapshot_directory,
    snapshot_paths,
)
from lato2_modly.portable import materialize_portable_runtime
from lato2_modly.state import write_runtime_config


SETUP_STATE_FILENAME = "setup-state.json"
VENV_NAME = "venv"
VENV_STAGING_NAME = "venv.__modly_staging"
VENV_BACKUP_NAME = "venv.__modly_backup"
STATE_STAGING_FILENAME = f"{SETUP_STATE_FILENAME}.__modly_staging"
STATE_BACKUP_FILENAME = f"{SETUP_STATE_FILENAME}.__modly_backup"
CONFIG_BACKUP_FILENAME = f"{RUNTIME_CONFIG_FILENAME}.__modly_backup"
SETUP_LOCK_TIMEOUT_SECONDS = 30.0
SETUP_LOCK_POLL_SECONDS = 0.25
COMMAND_TIMEOUT_SECONDS = 4 * 60 * 60
GIB = 1024**3
ASSET_HEADROOM_BYTES = 2 * GIB
PORTABLE_ENVIRONMENT_FREE_BYTES = 12 * GIB
EXACT_ENVIRONMENT_FREE_BYTES = 16 * GIB
PORTABLE_BUILD_CACHE_FREE_BYTES = 8 * GIB
EXACT_BUILD_CACHE_FREE_BYTES = 12 * GIB
WINDOWS_REPARSE_ATTRIBUTE = 0x400

INTERPRETER_PROBE = r"""
import json
import platform
import struct
import sys
import sysconfig

print(json.dumps({
    "implementation": sys.implementation.name,
    "version": list(sys.version_info[:2]),
    "cache_tag": sys.implementation.cache_tag,
    "abiflags": getattr(sys, "abiflags", ""),
    "soabi": sysconfig.get_config_var("SOABI"),
    "platform": sysconfig.get_platform().lower(),
    "machine": platform.machine().lower(),
    "pointer_bits": struct.calcsize("P") * 8,
}, sort_keys=True))
"""


class SetupFailure(RuntimeError):
    """An actionable, stable setup failure safe to show in Modly's UI."""

    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code}: {public_message}")


@dataclass(frozen=True)
class SetupContext:
    python_exe: Path
    ext_dir: Path
    gpu_sm: int
    cuda_version: int
    accelerator: str
    platform_name: str
    arch: str
    payload: Mapping[str, object]
    host_fingerprint: Mapping[str, object]


@dataclass(frozen=True)
class EnvironmentResult:
    python: Path
    reused: bool
    dependency_smoke: Mapping[str, object]
    portable_cpu_smoke: Mapping[str, object]
    promotion: "EnvironmentPromotion | None" = None


@dataclass(frozen=True)
class EnvironmentPromotion:
    """Previous-generation components retained until config publication."""

    had_previous_venv: bool
    had_previous_state: bool
    had_previous_config: bool


def _lock_would_block(exc: OSError) -> bool:
    return (
        exc.errno
        in {
            errno.EACCES,
            errno.EAGAIN,
            getattr(errno, "EDEADLK", -1),
        }
        or getattr(exc, "winerror", None) in {33, 36}
    )


def _try_setup_lock(handle: BinaryIO, platform_name: str) -> bool:
    """Attempt one non-blocking one-byte lock without hiding real I/O errors."""

    handle.seek(0)
    try:
        if platform_name == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if _lock_would_block(exc):
            return False
        raise
    return True


def _release_setup_lock(handle: BinaryIO, platform_name: str) -> None:
    handle.seek(0)
    if platform_name == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def setup_lock(
    extension_dir: Path,
    *,
    timeout: float = SETUP_LOCK_TIMEOUT_SECONDS,
    poll_interval: float = SETUP_LOCK_POLL_SECONDS,
    platform_name: str | None = None,
) -> Iterator[None]:
    """Serialize Install/Repair across processes on Windows and POSIX."""

    if timeout < 0 or poll_interval <= 0:
        raise ValueError("setup lock timeout must be non-negative and poll interval positive")
    system = normalize_platform_name(platform_name or current_platform_name())
    if system not in {"linux", "win32"}:
        raise SetupFailure("SETUP_LOCK_UNSUPPORTED", "setup locking requires Windows or Linux")
    lock_path = extension_dir / SETUP_LOCK_FILENAME
    handle: BinaryIO | None = None
    try:
        if lock_path.exists() or lock_path.is_symlink():
            info = lock_path.lstat()
            if (
                _is_alias(info)
                or not stat.S_ISREG(info.st_mode)
                or getattr(info, "st_nlink", 1) != 1
            ):
                raise SetupFailure("SETUP_LOCK_UNSAFE", "the setup lock path is unsafe")
        handle = lock_path.open("a+b")
        opened_info = os.fstat(handle.fileno())
        path_info = lock_path.lstat()
        if (
            _is_alias(path_info)
            or not stat.S_ISREG(path_info.st_mode)
            or not stat.S_ISREG(opened_info.st_mode)
            or getattr(path_info, "st_nlink", 1) != 1
            or getattr(opened_info, "st_nlink", 1) != 1
            or (
                getattr(path_info, "st_ino", 0)
                and getattr(opened_info, "st_ino", 0)
                and (
                    path_info.st_ino != opened_info.st_ino
                    or path_info.st_dev != opened_info.st_dev
                )
            )
        ):
            handle.close()
            raise SetupFailure("SETUP_LOCK_UNSAFE", "the setup lock path is unsafe")
        if opened_info.st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
            written_info = os.fstat(handle.fileno())
            written_path_info = lock_path.lstat()
            if (
                _is_alias(written_path_info)
                or not stat.S_ISREG(written_path_info.st_mode)
                or not stat.S_ISREG(written_info.st_mode)
                or getattr(written_path_info, "st_nlink", 1) != 1
                or getattr(written_info, "st_nlink", 1) != 1
                or (
                    getattr(written_path_info, "st_ino", 0)
                    and getattr(written_info, "st_ino", 0)
                    and (
                        written_path_info.st_ino != written_info.st_ino
                        or written_path_info.st_dev != written_info.st_dev
                    )
                )
            ):
                raise SetupFailure("SETUP_LOCK_UNSAFE", "the setup lock path is unsafe")
    except SetupFailure:
        if handle is not None and not handle.closed:
            handle.close()
        raise
    except OSError as exc:
        if handle is not None and not handle.closed:
            handle.close()
        raise SetupFailure("SETUP_LOCK_OPEN_FAILED", "the setup lock could not be opened") from exc

    if handle is None:
        raise SetupFailure("SETUP_LOCK_OPEN_FAILED", "the setup lock could not be opened")
    deadline = time.monotonic() + timeout
    acquired = False
    try:
        while True:
            try:
                acquired = _try_setup_lock(handle, system)
            except OSError as exc:
                raise SetupFailure(
                    "SETUP_LOCK_FAILED", "the setup lock could not be acquired"
                ) from exc
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise SetupFailure(
                    "SETUP_BUSY",
                    "another LATO.2 Install or Repair is still running; wait for it to finish and retry",
                )
            time.sleep(poll_interval)
        try:
            final_path_info = lock_path.lstat()
            final_opened_info = os.fstat(handle.fileno())
        except OSError as exc:
            raise SetupFailure(
                "SETUP_LOCK_UNSAFE", "the setup lock identity could not be revalidated"
            ) from exc
        if (
            _is_alias(final_path_info)
            or not stat.S_ISREG(final_path_info.st_mode)
            or not stat.S_ISREG(final_opened_info.st_mode)
            or getattr(final_path_info, "st_nlink", 1) != 1
            or getattr(final_opened_info, "st_nlink", 1) != 1
            or (
                getattr(final_path_info, "st_ino", 0)
                and getattr(final_opened_info, "st_ino", 0)
                and (
                    final_path_info.st_ino != final_opened_info.st_ino
                    or final_path_info.st_dev != final_opened_info.st_dev
                )
            )
        ):
            raise SetupFailure(
                "SETUP_LOCK_UNSAFE", "the setup lock identity changed while waiting"
            )
        yield
    finally:
        if acquired:
            try:
                _release_setup_lock(handle, system)
            except OSError:
                # Closing the descriptor also releases the OS lock. Avoid
                # masking a more useful setup failure from inside the context.
                pass
        handle.close()


def log(message: str) -> None:
    print(f"[LATO.2 setup] {message}", flush=True)


def error_log(message: str) -> None:
    # Modly includes the stderr tail in the installation error dialog.
    print(f"[LATO.2 setup] {message}", file=sys.stderr, flush=True)


def _integer(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise SetupFailure("SETUP_ARGUMENT_INVALID", f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SetupFailure("SETUP_ARGUMENT_INVALID", f"{label} must be an integer") from exc
    if str(value).strip() not in {str(parsed), f"{parsed}.0"}:
        raise SetupFailure("SETUP_ARGUMENT_INVALID", f"{label} must be an integer")
    if parsed < minimum or parsed > maximum:
        raise SetupFailure(
            "SETUP_ARGUMENT_INVALID",
            f"{label} must be between {minimum} and {maximum}",
        )
    return parsed


def parse_args(argv: Sequence[str]) -> dict[str, object]:
    """Parse the current Modly JSON contract or the historical positional one."""

    if len(argv) == 2:
        try:
            payload = json.loads(argv[1])
        except json.JSONDecodeError as exc:
            raise SetupFailure(
                "SETUP_JSON_INVALID", "Modly supplied malformed setup metadata"
            ) from exc
        if not isinstance(payload, dict):
            raise SetupFailure(
                "SETUP_JSON_INVALID", "Modly setup metadata must be a JSON object"
            )
        return payload
    if len(argv) >= 4:
        if len(argv) not in {4, 5}:
            raise SetupFailure(
                "SETUP_ARGUMENTS_INVALID",
                "legacy setup accepts python_exe, ext_dir, gpu_sm and optional cuda_version only",
            )
        gpu_sm = _integer(argv[3], "gpu_sm", minimum=0, maximum=999)
        cuda_version = (
            _integer(argv[4], "cuda_version", minimum=0, maximum=999)
            if len(argv) == 5
            else 0
        )
        return {
            "python_exe": argv[1],
            "ext_dir": argv[2],
            "gpu_sm": gpu_sm,
            "cuda_version": cuda_version,
            "accelerator": "cuda" if gpu_sm > 0 else "cpu",
            "platform": sys.platform,
            "arch": platform.machine(),
        }
    raise SetupFailure(
        "SETUP_ARGUMENTS_INVALID",
        "expected one Modly JSON argument or legacy python/ext_dir/gpu_sm arguments",
    )


def _normalize_arch(value: object) -> str:
    raw = str(value or "").strip().casefold().replace("-", "_")
    if raw in {"x86_64", "amd64", "x64"}:
        return "x64"
    if raw in {"aarch64", "arm64"}:
        return "arm64"
    return raw


def _is_alias(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )


def interpreter_fingerprint(python: Path) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [str(python), "-I", "-S", "-c", INTERPRETER_PROBE],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
            env=deps.sanitize_subprocess_environment(),
        )
        fingerprint = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise SetupFailure(
            "PYTHON_PROBE_FAILED", "Modly's Python interpreter could not be inspected"
        ) from exc
    if not isinstance(fingerprint, dict):
        raise SetupFailure("PYTHON_PROBE_INVALID", "Python returned invalid ABI metadata")
    return fingerprint


def validate_context(payload: Mapping[str, object], root: Path = ROOT) -> SetupContext:
    python_raw = payload.get("python_exe")
    if not isinstance(python_raw, str) or not python_raw.strip():
        raise SetupFailure("PYTHON_MISSING", "Modly did not provide its Python executable")
    python_exe = Path(python_raw).expanduser().resolve()
    if not python_exe.is_file():
        raise SetupFailure("PYTHON_MISSING", f"Modly Python is unavailable: {python_exe}")

    ext_raw = payload.get("ext_dir") or str(root)
    if not isinstance(ext_raw, str) or not ext_raw.strip():
        raise SetupFailure("EXTENSION_PATH_INVALID", "Modly supplied an invalid ext_dir")
    ext_dir = Path(ext_raw).expanduser().resolve()
    expected_root = root.resolve(strict=True)
    if ext_dir != expected_root:
        raise SetupFailure(
            "EXTENSION_PATH_MISMATCH",
            f"ext_dir does not identify this extension ({ext_dir})",
        )

    platform_name = normalize_platform_name(payload.get("platform") or sys.platform)
    if platform_name != current_platform_name():
        raise SetupFailure(
            "PLATFORM_MISMATCH", "Modly's platform metadata does not match this host"
        )
    arch = _normalize_arch(payload.get("arch") or platform.machine())
    host_arch = _normalize_arch(platform.machine())
    if arch != host_arch:
        raise SetupFailure(
            "ARCH_MISMATCH", "Modly's architecture metadata does not match this host"
        )

    gpu_sm = _integer(payload.get("gpu_sm", 0), "gpu_sm", minimum=0, maximum=999)
    cuda_version = _integer(
        payload.get("cuda_version", 0), "cuda_version", minimum=0, maximum=999
    )
    accelerator = str(
        payload.get("accelerator") or ("cuda" if gpu_sm > 0 else "cpu")
    ).strip().casefold()
    host_fingerprint = interpreter_fingerprint(python_exe)
    try:
        deps.python_abi_from_fingerprint(host_fingerprint)
    except deps.DependencyError as exc:
        raise SetupFailure(exc.code, exc.public_message) from exc

    normalized = dict(payload)
    normalized.update(
        {
            "python_exe": str(python_exe),
            "ext_dir": str(ext_dir),
            "gpu_sm": gpu_sm,
            "cuda_version": cuda_version,
            "accelerator": accelerator,
            "platform": platform_name,
            "arch": arch,
        }
    )
    return SetupContext(
        python_exe=python_exe,
        ext_dir=ext_dir,
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
        accelerator=accelerator,
        platform_name=platform_name,
        arch=arch,
        payload=normalized,
        host_fingerprint=host_fingerprint,
    )


def venv_python(venv: Path, platform_name: str | None = None) -> Path:
    system = platform_name or current_platform_name()
    return venv / ("Scripts/python.exe" if system == "win32" else "bin/python")


def _remove_venv(venv: Path, extension_dir: Path) -> None:
    if (
        venv.parent.resolve(strict=True) != extension_dir.resolve(strict=True)
        or venv.name not in {VENV_NAME, VENV_STAGING_NAME, VENV_BACKUP_NAME}
    ):
        raise SetupFailure("VENV_PATH_INVALID", "refusing to remove an unexpected path")
    try:
        info = venv.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SetupFailure("VENV_INSPECT_FAILED", "the extension venv cannot be inspected") from exc
    try:
        if _is_alias(info):
            # Windows directory junctions are reparse points but must be
            # removed with rmdir; never recurse through either alias kind.
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                venv.rmdir()
            else:
                venv.unlink()
        elif not stat.S_ISDIR(info.st_mode):
            venv.unlink()
        else:
            shutil.rmtree(venv)
    except OSError as exc:
        raise SetupFailure(
            "VENV_REMOVE_FAILED",
            "the stale extension venv could not be removed; close active LATO.2 jobs and retry Repair",
        ) from exc


def _create_venv(context: SetupContext, venv: Path) -> Path:
    log("Creating the isolated extension virtual environment")
    try:
        subprocess.run(
            [str(context.python_exe), "-m", "venv", str(venv)],
            check=True,
            stdin=subprocess.DEVNULL,
            timeout=15 * 60,
            env=deps.sanitize_subprocess_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SetupFailure(
            "VENV_CREATE_FAILED", "Modly's Python could not create the extension venv"
        ) from exc
    python = venv_python(venv, context.platform_name)
    if not python.is_file():
        raise SetupFailure("VENV_CREATE_INCOMPLETE", "venv creation produced no Python executable")
    if interpreter_fingerprint(python) != dict(context.host_fingerprint):
        _remove_venv(venv, context.ext_dir)
        raise SetupFailure(
            "VENV_ABI_MISMATCH", "the new venv does not match Modly's Python ABI"
        )
    return python


def _replace_owned_venv(source: Path, destination: Path, extension_dir: Path) -> None:
    allowed = {VENV_NAME, VENV_STAGING_NAME, VENV_BACKUP_NAME}
    extension = extension_dir.resolve(strict=True)
    if (
        source.parent.resolve(strict=True) != extension
        or destination.parent.resolve(strict=True) != extension
        or source.name not in allowed
        or destination.name not in allowed
    ):
        raise SetupFailure("VENV_PATH_INVALID", "refusing to move an unexpected venv path")
    if destination.exists() or destination.is_symlink():
        raise SetupFailure("VENV_PROMOTION_CONFLICT", "a venv transaction path already exists")
    try:
        source_info = source.lstat()
    except OSError as exc:
        raise SetupFailure("VENV_PROMOTION_FAILED", "the source venv is unavailable") from exc
    if _is_alias(source_info) or not stat.S_ISDIR(source_info.st_mode):
        raise SetupFailure("VENV_PATH_INVALID", "the source venv is not a regular directory")
    try:
        os.replace(source, destination)
    except OSError as exc:
        raise SetupFailure(
            "VENV_PROMOTION_FAILED",
            "the verified staging environment could not be promoted atomically",
        ) from exc


def _run_checked(
    command: Sequence[str],
    *,
    stage: str,
    env: Mapping[str, str] | None = None,
    timeout: int = COMMAND_TIMEOUT_SECONDS,
) -> None:
    log(stage)
    pip_command = len(command) >= 3 and list(command[1:3]) == ["-m", "pip"]
    try:
        subprocess.run(
            list(command),
            check=True,
            env=deps.sanitize_subprocess_environment(
                os.environ if env is None else env,
                for_pip=pip_command,
            ),
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SetupFailure(
            "COMMAND_TIMEOUT", f"{stage} exceeded its safe time limit; run Repair to resume"
        ) from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SetupFailure(
            "COMMAND_FAILED", f"{stage} failed; review the preceding installer output and run Repair"
        ) from exc


def _available_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError as exc:
        raise SetupFailure("DISK_CHECK_FAILED", f"free space could not be checked for {path}") from exc


def _require_free_space(path: Path, required: int, purpose: str) -> None:
    available = _available_bytes(path)
    if available < required:
        needed_gib = required / GIB
        free_gib = available / GIB
        raise SetupFailure(
            "DISK_SPACE_INSUFFICIENT",
            f"{purpose} needs about {needed_gib:.1f} GiB free at {path}; only {free_gib:.1f} GiB is available",
        )


def _remaining_asset_bytes(revision: Path) -> int:
    """Measure required growth from authenticated assets and safe partials.

    A readiness marker plus matching byte count is not enough for preflight:
    a corrupt same-sized checkpoint still needs a complete temporary download
    beside it. Hash before reserving space so setup cannot run out midway.
    """

    remaining = 0
    for spec in ASSETS:
        destination = revision.joinpath(*spec.relative_path.split("/"))
        part = destination.with_name(destination.name + ".part")
        valid, _reason = verify_asset(destination, spec)
        if valid:
            continue
        try:
            part_info = part.lstat()
        except OSError:
            part_info = None
        if (
            part_info is not None
            and not _is_alias(part_info)
            and stat.S_ISREG(part_info.st_mode)
            and getattr(part_info, "st_nlink", 1) == 1
            and 0 < part_info.st_size < spec.size
        ):
            remaining += spec.size - part_info.st_size
        else:
            remaining += spec.size
    return remaining


def _preflight_assets(revision: Path, models_root: Path) -> None:
    remaining = _remaining_asset_bytes(revision)
    if remaining:
        _require_free_space(
            models_root,
            remaining + ASSET_HEADROOM_BYTES,
            "the pinned 4.49 GiB LATO.2/DINO snapshot and safe extraction",
        )


def _preflight_plan(plan: deps.DependencyPlan, cache_root: Path) -> None:
    """Fail before multi-gigabyte installs; never substitute another profile."""

    if plan.install_native_stack:
        deps.native_build_environment(plan, cache_root)
    deps.cpu_build_environment(plan, cache_root)


def _volume_key(path: Path) -> tuple[int, str]:
    """Return a stable local volume identity for free-space aggregation."""

    try:
        resolved = path.resolve(strict=True)
        info = resolved.stat()
    except OSError as exc:
        raise SetupFailure(
            "DISK_CHECK_FAILED", f"storage volume could not be identified for {path}"
        ) from exc
    return info.st_dev, resolved.anchor.casefold()


def _preflight_install_storage(
    context: SetupContext,
    plan: deps.DependencyPlan,
    cache_root: Path,
) -> None:
    """Reserve venv and build/cache bytes on their actual destination volumes."""

    environment_required = (
        EXACT_ENVIRONMENT_FREE_BYTES
        if plan.install_native_stack
        else PORTABLE_ENVIRONMENT_FREE_BYTES
    )
    cache_required = (
        EXACT_BUILD_CACHE_FREE_BYTES
        if plan.install_native_stack
        else PORTABLE_BUILD_CACHE_FREE_BYTES
    )
    requirements = (
        (
            context.ext_dir,
            environment_required,
            f"the transactional {plan.profile} Python environment",
        ),
        (
            cache_root,
            cache_required,
            f"the {plan.profile} dependency downloads and native build cache",
        ),
    )
    grouped: dict[tuple[int, str], tuple[Path, int, list[str]]] = {}
    for path, required, purpose in requirements:
        key = _volume_key(path)
        if key in grouped:
            representative, total, purposes = grouped[key]
            grouped[key] = (representative, total + required, [*purposes, purpose])
        else:
            grouped[key] = (path, required, [purpose])
    for representative, required, purposes in grouped.values():
        _require_free_space(
            representative,
            required,
            " and ".join(purposes),
        )


def _invalidate_generated_file(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SetupFailure("STATE_INSPECT_FAILED", f"{label} cannot be inspected") from exc
    if _is_alias(info) or not stat.S_ISREG(info.st_mode):
        raise SetupFailure(
            "STATE_FILE_INVALID", f"{label} must be a regular local file"
        )
    try:
        path.unlink()
    except OSError as exc:
        raise SetupFailure(
            "STATE_INVALIDATE_FAILED", f"{label} could not be invalidated before Repair"
        ) from exc


def _invalidate_runtime_config(extension_dir: Path) -> None:
    _invalidate_generated_file(extension_dir / RUNTIME_CONFIG_FILENAME, RUNTIME_CONFIG_FILENAME)


def _invalidate_setup_state(extension_dir: Path) -> None:
    _invalidate_generated_file(extension_dir / SETUP_STATE_FILENAME, SETUP_STATE_FILENAME)


def _move_generated_file(source: Path, destination: Path, label: str) -> bool:
    """Move one regular extension-generated file, returning False if absent."""

    try:
        info = source.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SetupFailure("STATE_INSPECT_FAILED", f"{label} cannot be inspected") from exc
    if _is_alias(info) or not stat.S_ISREG(info.st_mode):
        raise SetupFailure("STATE_FILE_INVALID", f"{label} must be a regular local file")
    if destination.exists() or destination.is_symlink():
        raise SetupFailure("STATE_TRANSACTION_CONFLICT", f"a stale {label} backup exists")
    try:
        os.replace(source, destination)
    except OSError as exc:
        raise SetupFailure("STATE_MOVE_FAILED", f"{label} could not be moved atomically") from exc
    return True


def _restore_generated_file(backup: Path, destination: Path, label: str) -> None:
    if destination.exists() or destination.is_symlink():
        _invalidate_generated_file(destination, label)
    if backup.exists() or backup.is_symlink():
        _move_generated_file(backup, destination, label)


def _recover_environment_transaction(context: SetupContext) -> None:
    """Conservatively recover a setup interrupted between atomic renames."""

    extension = context.ext_dir
    venv = extension / VENV_NAME
    staging = extension / VENV_STAGING_NAME
    backup = extension / VENV_BACKUP_NAME
    config = extension / RUNTIME_CONFIG_FILENAME
    config_backup = extension / CONFIG_BACKUP_FILENAME
    state = extension / SETUP_STATE_FILENAME
    state_staging = extension / STATE_STAGING_FILENAME
    state_backup = extension / STATE_BACKUP_FILENAME

    interrupted_swap = (
        (backup.exists() or backup.is_symlink())
        and not (config.exists() or config.is_symlink())
        and (config_backup.exists() or config_backup.is_symlink())
    )
    if interrupted_swap:
        if venv.exists() or venv.is_symlink():
            _remove_venv(venv, extension)
        _replace_owned_venv(backup, venv, extension)
        _restore_generated_file(config_backup, config, RUNTIME_CONFIG_FILENAME)
        if state_backup.exists() or state_backup.is_symlink():
            _restore_generated_file(state_backup, state, SETUP_STATE_FILENAME)
        log("Recovered the previous environment after an interrupted promotion")
    elif backup.exists() or backup.is_symlink():
        if venv.exists() or venv.is_symlink():
            _remove_venv(backup, extension)
        else:
            _replace_owned_venv(backup, venv, extension)
        if config_backup.exists() or config_backup.is_symlink():
            if config.exists() or config.is_symlink():
                _invalidate_generated_file(config_backup, CONFIG_BACKUP_FILENAME)
            else:
                _restore_generated_file(config_backup, config, RUNTIME_CONFIG_FILENAME)

    if staging.exists() or staging.is_symlink():
        _remove_venv(staging, extension)
    for stale, label in (
        (state_staging, STATE_STAGING_FILENAME),
        (state_backup, STATE_BACKUP_FILENAME),
        (config_backup, CONFIG_BACKUP_FILENAME),
    ):
        if stale.exists() or stale.is_symlink():
            _invalidate_generated_file(stale, label)


def _reusable_environment(
    context: SetupContext,
    plan: deps.DependencyPlan,
    cache_root: Path,
    expected_state: Mapping[str, object],
) -> EnvironmentResult | None:
    venv = context.ext_dir / "venv"
    python = venv_python(venv, context.platform_name)
    state_path = context.ext_dir / SETUP_STATE_FILENAME
    try:
        venv_info = venv.lstat()
    except FileNotFoundError:
        _invalidate_runtime_config(context.ext_dir)
        return None
    except OSError:
        _invalidate_runtime_config(context.ext_dir)
        return None
    if _is_alias(venv_info) or not stat.S_ISDIR(venv_info.st_mode):
        _invalidate_runtime_config(context.ext_dir)
        return None
    if not python.is_file():
        _invalidate_runtime_config(context.ext_dir)
        return None
    if not deps.state_matches(state_path, expected_state):
        return None
    try:
        if interpreter_fingerprint(python) != dict(context.host_fingerprint):
            _invalidate_runtime_config(context.ext_dir)
            return None
        dependency_smoke = deps.verify_dependencies(python, plan, cache_root)
        portable_smoke = deps.verify_portable_cpu_extension(python, plan, cache_root)
    except Exception as exc:
        code = getattr(exc, "code", type(exc).__name__)
        log(f"Existing environment failed {code}; it will be rebuilt")
        _invalidate_runtime_config(context.ext_dir)
        return None
    log("Verified the existing dependency environment; skipped installation")
    return EnvironmentResult(python, True, dependency_smoke, portable_smoke)


def _install_environment(
    context: SetupContext,
    plan: deps.DependencyPlan,
    cache_root: Path,
    cpu_build: object,
    expected_state: Mapping[str, object],
) -> EnvironmentResult:
    extension = context.ext_dir
    venv = extension / VENV_NAME
    staging = extension / VENV_STAGING_NAME
    state_staging = extension / STATE_STAGING_FILENAME
    if staging.exists() or staging.is_symlink():
        _remove_venv(staging, extension)
    if state_staging.exists() or state_staging.is_symlink():
        _invalidate_generated_file(state_staging, STATE_STAGING_FILENAME)
    _preflight_install_storage(context, plan, cache_root)
    try:
        python = _create_venv(context, staging)
        constraint_file = deps.install_dependencies(
            python, plan, cache_root, log=log
        )
        if not isinstance(constraint_file, Path) or not constraint_file.is_absolute():
            raise SetupFailure(
                "DEPENDENCY_CONSTRAINTS_INVALID",
                "dependency installation did not return its locked constraints file",
            )
        constraint_file = deps.validate_dependency_constraints_file(
            constraint_file, plan
        )

        pip_args = getattr(cpu_build, "pip_install_args", None)
        expected_cpu_prefix = (
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
        )
        if (
            not isinstance(pip_args, tuple)
            or len(pip_args) != len(expected_cpu_prefix) + 1
            or pip_args[:-1] != expected_cpu_prefix
        ):
            raise SetupFailure(
                "PORTABLE_BUILD_INVALID",
                "the portable CPU build returned invalid install arguments",
            )
        source_argument = pip_args[-1]
        if not isinstance(source_argument, (str, os.PathLike)):
            raise SetupFailure(
                "PORTABLE_BUILD_INVALID",
                "the portable CPU build source path is invalid",
            )
        build_source = Path(source_argument)
        if not build_source.is_absolute():
            raise SetupFailure(
                "PORTABLE_BUILD_INVALID",
                "the portable CPU build source path must be absolute",
            )
        cpu_workspace = deps.prepare_build_workspace(
            build_source,
            cache_root,
            plan,
            "ovoxel-cpu",
        )
        cpu_env = deps.cpu_build_environment(plan, cache_root)
        cpu_install = deps.isolated_pip_command(
            python,
            [
                "install",
                "--constraint",
                str(constraint_file),
                "--no-build-isolation",
                "--no-deps",
                "--no-index",
                str(cpu_workspace),
            ],
            cpu_env,
        )
        _run_checked(
            cpu_install,
            stage="Building and installing the pinned o-voxel CPU operator",
            env=cpu_env,
        )
        _run_checked(
            deps.isolated_pip_command(python, ["check"], cpu_env),
            stage="Checking the staging dependency graph after the CPU operator",
            env=cpu_env,
            timeout=10 * 60,
        )
        dependency_smoke = deps.verify_dependencies(python, plan, cache_root)
        portable_smoke = deps.verify_portable_cpu_extension(python, plan, cache_root)
        deps.write_state(state_staging, expected_state)
        return _promote_environment(
            context,
            plan,
            cache_root,
            staging,
            state_staging,
            dependency_smoke,
            portable_smoke,
        )
    except BaseException:
        if staging.exists() or staging.is_symlink():
            _remove_venv(staging, extension)
        if state_staging.exists() or state_staging.is_symlink():
            _invalidate_generated_file(state_staging, STATE_STAGING_FILENAME)
        raise


def _promote_environment(
    context: SetupContext,
    plan: deps.DependencyPlan,
    cache_root: Path,
    staging: Path,
    state_staging: Path,
    dependency_smoke: Mapping[str, object],
    portable_smoke: Mapping[str, object],
) -> EnvironmentResult:
    """Swap a verified staging venv while retaining the prior generation.

    The previous venv, dependency state, and runtime configuration remain in
    transaction backups until ``_run_setup_locked`` has published the new
    runtime configuration.  This keeps the final configuration write inside
    the same logical transaction as the environment promotion.
    """

    extension = context.ext_dir
    venv = extension / VENV_NAME
    backup = extension / VENV_BACKUP_NAME
    state = extension / SETUP_STATE_FILENAME
    state_backup = extension / STATE_BACKUP_FILENAME
    config = extension / RUNTIME_CONFIG_FILENAME
    config_backup = extension / CONFIG_BACKUP_FILENAME
    for stale, label in (
        (state_backup, STATE_BACKUP_FILENAME),
        (config_backup, CONFIG_BACKUP_FILENAME),
    ):
        if stale.exists() or stale.is_symlink():
            _invalidate_generated_file(stale, label)
    if backup.exists() or backup.is_symlink():
        _remove_venv(backup, extension)

    config_moved = False
    old_venv_moved = False
    new_venv_moved = False
    old_state_moved = False
    new_state_moved = False
    try:
        config_moved = _move_generated_file(
            config, config_backup, RUNTIME_CONFIG_FILENAME
        )
        if venv.exists() or venv.is_symlink():
            existing_info = venv.lstat()
            if _is_alias(existing_info) or not stat.S_ISDIR(existing_info.st_mode):
                _remove_venv(venv, extension)
            else:
                _replace_owned_venv(venv, backup, extension)
                old_venv_moved = True
        _replace_owned_venv(staging, venv, extension)
        new_venv_moved = True
        python = venv_python(venv, context.platform_name)
        if interpreter_fingerprint(python) != dict(context.host_fingerprint):
            raise SetupFailure(
                "VENV_PROMOTION_ABI_MISMATCH",
                "the promoted environment no longer matches Modly's Python ABI",
            )
        cpu_env = deps.cpu_build_environment(plan, cache_root)
        _run_checked(
            deps.isolated_pip_command(python, ["check"], cpu_env),
            stage="Checking the promoted dependency graph",
            env=cpu_env,
            timeout=10 * 60,
        )
        dependency_smoke = deps.verify_dependencies(python, plan, cache_root)
        portable_smoke = deps.verify_portable_cpu_extension(python, plan, cache_root)

        old_state_moved = _move_generated_file(
            state, state_backup, SETUP_STATE_FILENAME
        )
        _move_generated_file(state_staging, state, SETUP_STATE_FILENAME)
        new_state_moved = True
    except BaseException as exc:
        rollback_error: BaseException | None = None
        try:
            if new_state_moved and (state.exists() or state.is_symlink()):
                _invalidate_generated_file(state, SETUP_STATE_FILENAME)
            if old_state_moved:
                _restore_generated_file(state_backup, state, SETUP_STATE_FILENAME)
            if new_venv_moved and (venv.exists() or venv.is_symlink()):
                _remove_venv(venv, extension)
            if old_venv_moved:
                _replace_owned_venv(backup, venv, extension)
            if config_moved and old_venv_moved:
                _restore_generated_file(
                    config_backup, config, RUNTIME_CONFIG_FILENAME
                )
            elif config_moved and (config_backup.exists() or config_backup.is_symlink()):
                _invalidate_generated_file(config_backup, CONFIG_BACKUP_FILENAME)
        except BaseException as rollback_exc:
            rollback_error = rollback_exc
        if rollback_error is not None:
            raise SetupFailure(
                "VENV_ROLLBACK_FAILED",
                "the new environment failed and the previous venv could not be restored; close active jobs and run Repair",
            ) from rollback_error
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, SetupFailure):
            raise
        raise SetupFailure(
            "VENV_PROMOTION_FAILED",
            "the promoted environment failed validation; the previous venv was restored",
        ) from exc

    return EnvironmentResult(
        python,
        False,
        dependency_smoke,
        portable_smoke,
        EnvironmentPromotion(
            had_previous_venv=old_venv_moved,
            had_previous_state=old_state_moved,
            had_previous_config=config_moved,
        ),
    )


def _rollback_environment_promotion(
    context: SetupContext, promotion: EnvironmentPromotion
) -> None:
    """Undo a promoted Repair generation after final publication failed.

    A first installation has no previous venv to restore.  Its fully verified
    venv and setup state are deliberately retained for an idempotent Repair,
    but no runtime configuration is allowed to survive the failed publication.
    """

    extension = context.ext_dir
    venv = extension / VENV_NAME
    backup = extension / VENV_BACKUP_NAME
    state = extension / SETUP_STATE_FILENAME
    state_backup = extension / STATE_BACKUP_FILENAME
    config = extension / RUNTIME_CONFIG_FILENAME
    config_backup = extension / CONFIG_BACKUP_FILENAME

    if config.exists() or config.is_symlink():
        _invalidate_generated_file(config, RUNTIME_CONFIG_FILENAME)

    if promotion.had_previous_venv:
        if state.exists() or state.is_symlink():
            _invalidate_generated_file(state, SETUP_STATE_FILENAME)
        if venv.exists() or venv.is_symlink():
            _remove_venv(venv, extension)
        _replace_owned_venv(backup, venv, extension)
        if promotion.had_previous_state:
            _restore_generated_file(state_backup, state, SETUP_STATE_FILENAME)
        if promotion.had_previous_config:
            _restore_generated_file(
                config_backup, config, RUNTIME_CONFIG_FILENAME
            )
    else:
        # There was no usable generation to restore.  Keep the already
        # validated environment/state so Repair can reuse them, but never
        # expose them through a configuration whose publication failed.
        if backup.exists() or backup.is_symlink():
            _remove_venv(backup, extension)

    for obsolete, label in (
        (state_backup, STATE_BACKUP_FILENAME),
        (config_backup, CONFIG_BACKUP_FILENAME),
    ):
        if obsolete.exists() or obsolete.is_symlink():
            _invalidate_generated_file(obsolete, label)


def _commit_environment_promotion(
    context: SetupContext, promotion: EnvironmentPromotion
) -> None:
    """Discard transaction backups after runtime_config.json is durable."""

    extension = context.ext_dir
    backup = extension / VENV_BACKUP_NAME
    if backup.exists() or backup.is_symlink():
        _remove_venv(backup, extension)
    for obsolete, label in (
        (extension / STATE_BACKUP_FILENAME, STATE_BACKUP_FILENAME),
        (extension / CONFIG_BACKUP_FILENAME, CONFIG_BACKUP_FILENAME),
    ):
        if obsolete.exists() or obsolete.is_symlink():
            _invalidate_generated_file(obsolete, label)


def install_or_reuse_environment(
    context: SetupContext,
    plan: deps.DependencyPlan,
    cache_root: Path,
    cpu_build: object,
) -> EnvironmentResult:
    _recover_environment_transaction(context)
    expected_state = deps.dependency_state_payload(plan, context.host_fingerprint)
    reusable = _reusable_environment(context, plan, cache_root, expected_state)
    if reusable is not None:
        return reusable
    return _install_environment(context, plan, cache_root, cpu_build, expected_state)


def _run_setup_locked(context: SetupContext) -> Path:
    log(
        f"host={context.platform_name}/{context.arch} accelerator={context.accelerator} "
        f"SM={context.gpu_sm} driver_cuda={context.cuda_version or 'unknown'}"
    )

    models_root = resolve_models_root(
        context.payload,
        context.ext_dir,
        context.platform_name,
        payload_keys=SETUP_MODELS_PAYLOAD_KEYS,
        require_existing=True,
    ).resolve(strict=True)
    revision = owned_snapshot_directory(models_root, create=True).resolve(strict=True)
    cache_root = safe_snapshot_directory(revision, "runtime-cache", create=True).resolve(
        strict=True
    )

    requested_profile = str(
        context.payload.get("dependency_profile")
        or os.environ.get("MODLY_LATO2_DEPENDENCY_PROFILE")
        or "auto"
    )
    plan = deps.select_dependency_plan(
        context.payload,
        requested_profile=requested_profile,
        interpreter_fingerprint=context.host_fingerprint,
    )
    log(
        f"selected dependency profile={plan.profile}, torch={plan.torch_lane}, "
        f"attention={plan.attention_backend}"
    )
    _preflight_plan(plan, cache_root)
    _preflight_assets(revision, models_root)

    ready_revision = ensure_snapshot(revision, log=log).resolve(strict=True)
    if ready_revision != revision:
        raise SetupFailure(
            "ASSET_ROOT_MISMATCH", "the asset manager returned an unexpected revision directory"
        )
    asset_failures = verify_snapshot(revision, require_ready=True)
    if asset_failures:
        raise SetupFailure(
            "ASSET_VERIFY_FAILED",
            "the pinned snapshot is incomplete after setup: " + "; ".join(asset_failures[:3]),
        )

    paths = snapshot_paths(revision)
    portable_report = materialize_portable_runtime(
        upstream_root=paths.lato_source,
        portable_root=revision / "source" / "LATO.2-portable",
    )
    cpu_sources = deps.prepare_portable_cpu_sources(cache_root, log=log)
    cpu_build = materialize_ovoxel_cpu_build(
        ovoxel_source_root=cpu_sources.ovoxel,
        eigen_source_root=cpu_sources.eigen,
        build_root=revision / "native" / "ovoxel_cpu-build",
    )

    environment = install_or_reuse_environment(context, plan, cache_root, cpu_build)
    # Configuration publication is the transaction's commit point.  A Repair
    # keeps its previous generation in backups through this write; any error
    # while building or publishing the final payload restores that generation.
    try:
        available_backends = (
            ["upstream", "portable"] if plan.install_native_stack else ["portable"]
        )
        default_backend = "upstream" if plan.install_native_stack else "portable"
        config = write_runtime_config(
            context.ext_dir,
            models_root,
            revision,
            extra={
                "extension_id": EXTENSION_ID,
                "extension_version": EXTENSION_VERSION,
                "revision_id": REVISION_ID,
                "default_backend": default_backend,
                "available_backends": available_backends,
                "attention_backend": plan.attention_backend,
                "portable_precision_env": True,
                "portable_precisions": (
                    ["auto", "bfloat16", "float16"]
                    if context.gpu_sm >= 80
                    else ["auto", "float16"]
                ),
                "dependency_profile": plan.profile,
                "python_lane": plan.python_abi.lane,
                "python_abi": plan.python_abi.payload(),
                "dependency_lock_digest": deps.dependency_lock_digest(plan),
                "dependency_support_level": plan.support_level,
                "torch_lane": plan.torch_lane,
                "runtime_cache_dir": str(cache_root),
                "ready_marker": str(revision / READY_MARKER_FILENAME),
                "portable_build": portable_report.to_dict(),
                "portable_cpu_build": cpu_build.to_dict(),
                "environment_reused": environment.reused,
                "platform": context.platform_name,
                "arch": context.arch,
                "gpu_sm": context.gpu_sm,
                "cuda_version": context.cuda_version,
            },
        )
    except BaseException:
        if environment.promotion is not None:
            try:
                _rollback_environment_promotion(context, environment.promotion)
            except BaseException as rollback_exc:
                raise SetupFailure(
                    "VENV_ROLLBACK_FAILED",
                    "runtime configuration publication failed and the previous environment could not be restored; close active jobs and run Repair",
                ) from rollback_exc
        raise
    if environment.promotion is not None:
        _commit_environment_promotion(context, environment.promotion)
    log(
        f"Setup complete: {plan.profile}; backends={','.join(available_backends)}; "
        "all seven LATO.2 checkpoints and offline DINOv2 are verified"
    )
    return config


def run_setup(payload: Mapping[str, object], root: Path = ROOT) -> Path:
    context = validate_context(payload, root)
    with setup_lock(context.ext_dir, platform_name=context.platform_name):
        return _run_setup_locked(context)


def _known_failure(exc: BaseException) -> tuple[str, str] | None:
    code = getattr(exc, "code", None)
    message = getattr(exc, "public_message", None)
    if isinstance(code, str) and isinstance(message, str):
        return code, message
    return None


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = parse_args(list(sys.argv if argv is None else argv))
        run_setup(payload)
        return 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        known = _known_failure(exc)
        if known is not None:
            code, message = known
            error_log(f"ERROR [{code}] {message}")
        else:
            error_log(
                f"ERROR [SETUP_UNEXPECTED] {type(exc).__name__}: {exc}. "
                "Review the preceding output and run Repair."
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
