"""Offline Modly runtime that dispatches the pinned LATO.2 inference scripts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
from typing import Any, BinaryIO, Callable, Iterator, Mapping, TextIO
from urllib.parse import unquote, urlsplit
import uuid

from .assets import verify_asset, verify_source_tree
from .constants import (
    ASSETS,
    AssetSpec,
    DINO_CHECKPOINT_SPEC,
    DINO_NODE_IDS,
    DINO_SOURCE_PATH,
    EXTENSION_ID,
    EXTENSION_VERSION,
    LATO_CHECKPOINT_PATHS,
    LATO_SOURCE_PATH,
    NODE_LATO_CHECKPOINTS,
    READY_MARKER_FILENAME,
    READY_SCHEMA_VERSION,
    REVISION_ID,
    RUNTIME_CONFIG_FILENAME,
    RUNTIME_CONFIG_SCHEMA_VERSION,
    SETUP_LOCK_FILENAME,
    SOURCE_ARCHIVES,
    SourceArchiveSpec,
)
from .portable import validate_portable_runtime
from .integrity import TreeIntegrityError, read_owned_regular_bytes
from .paths import (
    PathContractError,
    RUNTIME_MODELS_PAYLOAD_KEYS,
    current_platform_name,
    owned_snapshot_directory,
    resolve_models_root,
    safe_snapshot_file,
    safe_snapshot_directory,
    snapshot_paths,
)


# Apply offline policy before any optional ML package can be imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_INPUT_SUFFIXES = frozenset({".obj", ".glb", ".gltf", ".ply", ".stl", ".off"})
NODE_IDS = tuple(NODE_LATO_CHECKPOINTS)
BACKENDS = frozenset({"upstream", "portable"})
PRECISIONS = frozenset({"auto", "bfloat16", "float16"})
WINDOWS_REPARSE_ATTRIBUTE = 0x400
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_LOG_BYTES = 2 * 1024 * 1024
MAX_LOG_OVERHEAD_BYTES = 100
MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024
MAX_GLTF_JSON_BYTES = 64 * 1024 * 1024
MAX_MESH_REFERENCES = 4096
SETUP_READ_LOCK_TIMEOUT_SECONDS = 30.0
SETUP_READ_LOCK_POLL_SECONDS = 0.25
SENSITIVE_ENVIRONMENT = re.compile(
    r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|PASSWD|COOKIE|AUTHORIZATION|API_KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIALS?)(?:_|$)",
    re.IGNORECASE,
)
BLOCKED_INFERENCE_ENVIRONMENT = frozenset(
    {
        "PIP_CONFIG_FILE",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_TRUSTED_HOST",
        "PYTHONPATH",
        "PYTHONHOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "FTP_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    }
)
PASSTHROUGH_INFERENCE_ENVIRONMENT = frozenset(
    {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "INCLUDE",
        "LIB",
        "CC",
        "CXX",
        "CUDAHOSTCXX",
        "CMAKE_GENERATOR",
        "CMAKE_PREFIX_PATH",
        "MAX_JOBS",
        "TRITON_PTXAS_PATH",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
    }
)
PASSTHROUGH_INFERENCE_PREFIXES = (
    "CUDA_",
    "NCCL_",
    "NVIDIA_",
    "ROCR_",
    "HIP_",
    "HSA_",
)
WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{value}" for value in range(1, 10)}
    | {f"LPT{value}" for value in range(1, 10)}
)


ERRORS: dict[str, tuple[str, str]] = {
    "REQUEST_EMPTY": ("request validation", "Run one LATO.2 node from Modly."),
    "REQUEST_COUNT": ("request validation", "Run one process request at a time."),
    "REQUEST_JSON": ("request validation", "Run the node again from Modly."),
    "REQUEST_TYPE": ("request validation", "Run the node again from Modly."),
    "REQUEST_NODE": ("node dispatch", "Reinstall or update the extension and retry."),
    "REQUEST_INPUT": ("mesh input validation", "Connect one supported mesh and try again."),
    "REQUEST_PARAMS": ("parameter validation", "Reset this node's parameters and try again."),
    "REQUEST_PATHS": ("storage validation", "Restart Modly and verify its storage paths."),
    "MODELS_UNAVAILABLE": (
        "model storage discovery",
        "Keep Modly running, verify models_dir, then run Repair for this extension.",
    ),
    "SETUP_INVALID": (
        "local setup validation",
        "Run Repair to restore the pinned sources, weights, and dependencies.",
    ),
    "SETUP_BUSY": (
        "setup coordination",
        "Wait for Install or Repair to finish, then run the workflow again.",
    ),
    "BACKEND_UNAVAILABLE": (
        "backend selection",
        "Select an installed backend or run Repair to provision it.",
    ),
    "PRECISION_UNAVAILABLE": (
        "precision selection",
        "Use Auto/BFloat16 for upstream, or Repair the portable backend.",
    ),
    "INPUT_CONVERSION": (
        "input geometry conversion",
        "Verify that the mesh contains finite vertices and triangle faces.",
    ),
    "INFERENCE_FAILED": (
        "upstream inference",
        "Review the sanitized diagnostic under Workflows/LATO2 when available, "
        "then run Repair if the failure persists.",
    ),
    "OUTPUT_MISSING": (
        "upstream output validation",
        "Review the sanitized diagnostic under Workflows/LATO2 when available; "
        "then try the default parameters.",
    ),
    "OUTPUT_CONVERSION": (
        "Modly GLB conversion",
        "Review the sanitized diagnostic under Workflows/LATO2 when available; "
        "then run Repair to restore trimesh.",
    ),
    "OUTPUT_INVALID": (
        "result validation",
        "Review the sanitized diagnostic under Workflows/LATO2 when available; "
        "then verify workspace permissions.",
    ),
    "UNEXPECTED": ("processing", "Run Repair and try again."),
}


class ProcessFailure(RuntimeError):
    """A stable, allowlisted public failure."""

    def __init__(self, code: str) -> None:
        self.code = code if code in ERRORS else "UNEXPECTED"
        super().__init__(self.code)

    def public_message(self) -> str:
        stage, action = ERRORS[self.code]
        return f"[{self.code}] LATO.2 {stage} failed. {action}"


class _ChildCancellation(BaseException):
    """Internal signal used to unwind into process-group cleanup."""


class ProtocolEmitter:
    """Write complete NDJSON records and exactly one terminal record."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._terminal = False
        self._failed = False
        self._last_progress = -1

    def _write(self, value: dict[str, Any], *, terminal: bool = False) -> None:
        if self._terminal or self._failed:
            raise RuntimeError("protocol channel is closed")
        if terminal:
            self._terminal = True
        try:
            line = json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n"
            written = self._stream.write(line)
            if not isinstance(written, int) or written != len(line):
                raise OSError("short protocol write")
            self._stream.flush()
        except BaseException:
            self._failed = True
            raise

    @property
    def terminal_attempted(self) -> bool:
        return self._terminal

    @property
    def channel_failed(self) -> bool:
        return self._failed

    def progress(self, percent: int, label: str) -> None:
        value = max(self._last_progress, min(100, max(0, int(percent))))
        self._last_progress = value
        self._write({"type": "progress", "percent": value, "label": label})

    def log(self, message: str) -> None:
        self._write({"type": "log", "message": str(message)[:1000]})

    def done(self, output: Path) -> None:
        self._write(
            {"type": "done", "result": {"filePath": str(output)}},
            terminal=True,
        )

    def error(self, message: str) -> None:
        self._write({"type": "error", "message": message}, terminal=True)


@dataclass(frozen=True)
class RuntimeState:
    models_root: Path
    revision_root: Path
    source_root: Path
    checkpoints: Path
    dino_hub: Path
    default_backend: str
    available_backends: frozenset[str]
    attention_backend: str | None
    portable_precision_env: bool
    portable_precisions: frozenset[str]


@dataclass(frozen=True)
class ValidatedRequest:
    node_id: str
    input_path: Path
    params: dict[str, Any]
    workspace_dir: Path
    temp_dir: Path
    state: RuntimeState
    backend: str
    precision: str


COMMON_DEFAULTS: dict[str, Any] = {
    "backend": "auto",
    "precision": "auto",
    "seed": 42,
    "num_workers": 4,
}
VERTEX_FLOW_DEFAULTS: dict[str, Any] = {
    "inference_threshold": 0.5,
    "steps": 24,
    "cfg_strength": 3.0,
    "rescale_t": 1.0,
    "vert_num": 2000,
    "use_gt_vert_count": "false",
    "scaler": 1.0,
    "min_verts": 200.0,
    "max_verts": 5000.0,
    "render_azimuth": 45.0,
    "render_elevation": 30.0,
    "img_res": 518,
}
POINT_ENCODER_DEFAULTS: dict[str, Any] = {
    "pc_sample_number": 819200,
    "sample_type": "dora",
    "sample_posterior": "true",
}
NODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "lato2-e2e": {
        **COMMON_DEFAULTS,
        **{("vflow_steps" if key == "steps" else key): value for key, value in VERTEX_FLOW_DEFAULTS.items()},
        "tflow_steps": 50,
        "edge_threshold": 0.0,
        "chunk_size": 20000,
        "fill_quad_rings": "true",
    },
    "lato2-vflow": {
        **COMMON_DEFAULTS,
        **{("vflow_steps" if key == "steps" else key): value for key, value in VERTEX_FLOW_DEFAULTS.items()},
        "pc_sample_number": 819200,
        "sample_type": "dora",
        "reconstruct": "false",
        "sample_posterior": "true",
    },
    "lato2-vvae": {
        **COMMON_DEFAULTS,
        **POINT_ENCODER_DEFAULTS,
        "inference_threshold": 0.5,
    },
    "lato2-tflow": {
        **COMMON_DEFAULTS,
        "tflow_steps": 50,
        "use_cond": "true",
        "edge_threshold": 0.0,
        "chunk_size": 20000,
        "fill_quad_rings": "true",
        "save_voxel_field": "true",
    },
}

BOOL_PARAMETERS = frozenset(
    {
        "use_gt_vert_count",
        "reconstruct",
        "sample_posterior",
        "use_cond",
        "fill_quad_rings",
        "save_voxel_field",
    }
)
POSITIVE_INTS = frozenset(
    {"steps", "vflow_steps", "tflow_steps", "vert_num", "pc_sample_number", "img_res", "chunk_size"}
)
FLOAT_PARAMETERS = frozenset(
    {
        "inference_threshold",
        "cfg_strength",
        "rescale_t",
        "scaler",
        "min_verts",
        "max_verts",
        "render_azimuth",
        "render_elevation",
        "edge_threshold",
    }
)


def _is_alias(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )


def _lock_would_block(exc: OSError) -> bool:
    return exc.errno in {
        errno.EACCES,
        errno.EAGAIN,
        getattr(errno, "EDEADLK", -1),
    } or getattr(exc, "winerror", None) in {33, 36}


def _windows_kernel32() -> Any:
    import ctypes

    return ctypes.WinDLL("kernel32", use_last_error=True)


def _windows_overlapped_type() -> type[Any]:
    import ctypes
    from ctypes import wintypes

    class Overlapped(ctypes.Structure):
        _fields_ = (
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        )

    return Overlapped


def _try_windows_shared_lock(handle: BinaryIO) -> bool:
    """Acquire a true shared Win32 byte-range lock with LockFileEx."""

    import ctypes
    from ctypes import wintypes
    import msvcrt

    overlapped_type = _windows_overlapped_type()
    overlapped = overlapped_type()
    lock_file = _windows_kernel32().LockFileEx
    lock_file.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(overlapped_type),
    )
    lock_file.restype = wintypes.BOOL
    # LOCKFILE_FAIL_IMMEDIATELY without LOCKFILE_EXCLUSIVE_LOCK is shared.
    set_last_error = getattr(ctypes, "set_last_error", None)
    if callable(set_last_error):
        set_last_error(0)
    acquired = lock_file(
        wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno())),
        0x00000001,
        0,
        1,
        0,
        ctypes.byref(overlapped),
    )
    if acquired:
        return True
    get_last_error = getattr(ctypes, "get_last_error", None)
    error = get_last_error() if callable(get_last_error) else 1
    if error in {33, 36}:
        return False
    raise OSError(error, "LockFileEx failed")


def _release_windows_shared_lock(handle: BinaryIO) -> None:
    import ctypes
    from ctypes import wintypes
    import msvcrt

    overlapped_type = _windows_overlapped_type()
    overlapped = overlapped_type()
    unlock_file = _windows_kernel32().UnlockFileEx
    unlock_file.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(overlapped_type),
    )
    unlock_file.restype = wintypes.BOOL
    set_last_error = getattr(ctypes, "set_last_error", None)
    if callable(set_last_error):
        set_last_error(0)
    released = unlock_file(
        wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno())),
        0,
        1,
        0,
        ctypes.byref(overlapped),
    )
    if not released:
        get_last_error = getattr(ctypes, "get_last_error", None)
        error = get_last_error() if callable(get_last_error) else 1
        raise OSError(error, "UnlockFileEx failed")


def _try_setup_read_lock(handle: BinaryIO, platform_name: str) -> bool:
    """Acquire one shared setup-lock byte without blocking."""

    handle.seek(0)
    try:
        if platform_name == "win32":
            return _try_windows_shared_lock(handle)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError as exc:
        if _lock_would_block(exc):
            return False
        raise
    return True


def _release_setup_read_lock(handle: BinaryIO, platform_name: str) -> None:
    handle.seek(0)
    if platform_name == "win32":
        _release_windows_shared_lock(handle)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _same_lock_identity(left: os.stat_result, right: os.stat_result) -> bool:
    if left.st_size != right.st_size:
        return False
    left_ino = getattr(left, "st_ino", 0)
    right_ino = getattr(right, "st_ino", 0)
    if left_ino and right_ino:
        return left.st_dev == right.st_dev and left_ino == right_ino
    return True


@contextmanager
def _setup_read_lock(
    extension_dir: Path | None = None,
    *,
    timeout: float = SETUP_READ_LOCK_TIMEOUT_SECONDS,
    poll_interval: float = SETUP_READ_LOCK_POLL_SECONDS,
    platform_name: str | None = None,
) -> Iterator[None]:
    """Hold a shared lease so Repair cannot mutate a running inference."""

    if timeout < 0 or poll_interval <= 0:
        raise ValueError("setup read-lock timeout must be non-negative")
    system = platform_name or current_platform_name()
    if system not in {"linux", "win32"}:
        raise ProcessFailure("SETUP_INVALID")
    lock_path = (extension_dir or ROOT) / SETUP_LOCK_FILENAME
    handle: BinaryIO | None = None
    acquired = False
    try:
        before = lock_path.lstat()
        if (
            _is_alias(before)
            or not stat.S_ISREG(before.st_mode)
            or getattr(before, "st_nlink", 1) != 1
            or before.st_size < 1
        ):
            raise ProcessFailure("SETUP_INVALID")
        handle = lock_path.open("r+b")
        opened = os.fstat(handle.fileno())
        path_info = lock_path.lstat()
        if (
            _is_alias(path_info)
            or not stat.S_ISREG(path_info.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or getattr(path_info, "st_nlink", 1) != 1
            or getattr(opened, "st_nlink", 1) != 1
            or not _same_lock_identity(before, opened)
            or not _same_lock_identity(path_info, opened)
        ):
            raise ProcessFailure("SETUP_INVALID")
        deadline = time.monotonic() + timeout
        while not acquired:
            acquired = _try_setup_read_lock(handle, system)
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise ProcessFailure("SETUP_BUSY")
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
        final_path = lock_path.lstat()
        final_open = os.fstat(handle.fileno())
        if (
            _is_alias(final_path)
            or not stat.S_ISREG(final_path.st_mode)
            or getattr(final_path, "st_nlink", 1) != 1
            or getattr(final_open, "st_nlink", 1) != 1
            or not _same_lock_identity(final_path, final_open)
        ):
            raise ProcessFailure("SETUP_INVALID")
        yield
    except ProcessFailure:
        raise
    except OSError as exc:
        raise ProcessFailure("SETUP_INVALID") from exc
    finally:
        if handle is not None:
            if acquired:
                try:
                    _release_setup_read_lock(handle, system)
                except OSError:
                    pass
            handle.close()


def _regular_file(path: Path, *, nonempty: bool = True) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ProcessFailure("SETUP_INVALID") from exc
    if (
        _is_alias(info)
        or not stat.S_ISREG(info.st_mode)
        or getattr(info, "st_nlink", 1) != 1
        or (nonempty and info.st_size <= 0)
    ):
        raise ProcessFailure("SETUP_INVALID")
    return path


def _read_json_file(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    try:
        encoded = read_owned_regular_bytes(path, max_bytes=max_bytes)
        value = json.loads(encoded.decode("utf-8"))
    except (TreeIntegrityError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProcessFailure("SETUP_INVALID") from exc
    if not isinstance(value, dict):
        raise ProcessFailure("SETUP_INVALID")
    return value


def _canonical(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ProcessFailure("REQUEST_PATHS") from exc


def _runtime_config() -> dict[str, Any]:
    config = _read_json_file(ROOT / RUNTIME_CONFIG_FILENAME, max_bytes=64 * 1024)
    if config.get("schema_version") != RUNTIME_CONFIG_SCHEMA_VERSION:
        raise ProcessFailure("SETUP_INVALID")
    return config


def _validate_runtime_config_identity(
    config: Mapping[str, Any], revision_root: Path
) -> None:
    """Bind mutable runtime state to this exact extension release."""

    if (
        config.get("extension_id") != EXTENSION_ID
        or config.get("extension_version") != EXTENSION_VERSION
        or config.get("revision_id") != REVISION_ID
    ):
        raise ProcessFailure("SETUP_INVALID")

    expected_paths = {
        "ready_marker": (revision_root / READY_MARKER_FILENAME, False),
        "runtime_cache_dir": (revision_root / "runtime-cache", True),
    }
    for key, (expected, directory) in expected_paths.items():
        if key not in config:
            raise ProcessFailure("SETUP_INVALID")
        value = config.get(key)
        if not isinstance(value, str) or not value or "\0" in value:
            raise ProcessFailure("SETUP_INVALID")
        provided = Path(value)
        if not provided.is_absolute():
            raise ProcessFailure("SETUP_INVALID")
        try:
            info = provided.lstat()
            actual = provided.resolve(strict=True)
            expected_resolved = expected.resolve(strict=True)
        except OSError as exc:
            raise ProcessFailure("SETUP_INVALID") from exc
        if (
            _is_alias(info)
            or actual != expected_resolved
            or (directory and not stat.S_ISDIR(info.st_mode))
            or (not directory and not stat.S_ISREG(info.st_mode))
        ):
            raise ProcessFailure("SETUP_INVALID")


def _models_from_config(config: Mapping[str, Any]) -> Path:
    try:
        return resolve_models_root(
            {"models_dir": config.get("models_dir")},
            ROOT,
            current_platform_name(),
            payload_keys=("models_dir",),
            environ={},
            require_existing=True,
        )
    except PathContractError as exc:
        raise ProcessFailure("SETUP_INVALID") from exc


def _resolve_models(payload: Mapping[str, Any], config: Mapping[str, Any]) -> Path:
    try:
        current = resolve_models_root(
            payload,
            ROOT,
            current_platform_name(),
            payload_keys=RUNTIME_MODELS_PAYLOAD_KEYS,
            require_existing=True,
        )
    except PathContractError as exc:
        if exc.code != "PATH_MODELS_UNAVAILABLE":
            raise ProcessFailure("MODELS_UNAVAILABLE") from exc
        current = _models_from_config(config)
    configured = _models_from_config(config)
    if _canonical(current) != _canonical(configured):
        raise ProcessFailure("SETUP_INVALID")
    return current


def _validate_marker(revision_root: Path) -> None:
    marker = _read_json_file(revision_root / READY_MARKER_FILENAME)
    if not (
        marker.get("schema_version") == READY_SCHEMA_VERSION
        and marker.get("extension_id") == EXTENSION_ID
        and marker.get("extension_version") == EXTENSION_VERSION
        and marker.get("revision_id") == REVISION_ID
    ):
        raise ProcessFailure("SETUP_INVALID")
    inventory = marker.get("inventory")
    if not isinstance(inventory, list):
        raise ProcessFailure("SETUP_INVALID")
    declared: dict[str, tuple[int, str]] = {}
    for entry in inventory:
        if not isinstance(entry, dict):
            raise ProcessFailure("SETUP_INVALID")
        path = entry.get("path")
        size = entry.get("size")
        digest = entry.get("sha256")
        if not isinstance(path, str) or not isinstance(size, int) or not isinstance(digest, str):
            raise ProcessFailure("SETUP_INVALID")
        if path in declared:
            raise ProcessFailure("SETUP_INVALID")
        declared[path] = (size, digest.casefold())
    for spec in ASSETS:
        if declared.get(spec.relative_path) != (spec.size, spec.sha256):
            raise ProcessFailure("SETUP_INVALID")
        asset = revision_root.joinpath(*spec.relative_path.split("/"))
        _regular_file(asset)
        if asset.stat().st_size != spec.size:
            raise ProcessFailure("SETUP_INVALID")


def _pinned_asset_spec(relative_path: str) -> AssetSpec:
    matches = tuple(spec for spec in ASSETS if spec.relative_path == relative_path)
    if len(matches) != 1:
        raise ProcessFailure("SETUP_INVALID")
    return matches[0]


def _pinned_source_spec(destination: str) -> SourceArchiveSpec:
    matches = tuple(spec for spec in SOURCE_ARCHIVES if spec.destination == destination)
    if len(matches) != 1:
        raise ProcessFailure("SETUP_INVALID")
    return matches[0]


def _authenticate_asset(revision_root: Path, spec: AssetSpec) -> None:
    """Hash one exact regular file against the compile-time inventory."""

    try:
        path = safe_snapshot_file(
            revision_root,
            spec.relative_path,
            create_parent=False,
        )
    except PathContractError as exc:
        raise ProcessFailure("SETUP_INVALID") from exc
    valid, _reason = verify_asset(path, spec)
    if not valid:
        raise ProcessFailure("SETUP_INVALID")


def _authenticate_source_tree(revision_root: Path, spec: SourceArchiveSpec) -> None:
    """Authenticate both a pinned ZIP and the exact tree extracted from it."""

    _authenticate_asset(revision_root, _pinned_asset_spec(spec.asset_path))
    valid, _reason = verify_source_tree(revision_root, spec)
    if not valid:
        raise ProcessFailure("SETUP_INVALID")


def _authenticate_request(request: ValidatedRequest) -> None:
    """Authenticate every file that this request can load before inference."""

    _authenticate_source_tree(
        request.state.revision_root,
        _pinned_source_spec(LATO_SOURCE_PATH),
    )
    for name in NODE_LATO_CHECKPOINTS[request.node_id]:
        relative_path = LATO_CHECKPOINT_PATHS.get(name)
        if relative_path is None:
            raise ProcessFailure("SETUP_INVALID")
        _authenticate_asset(
            request.state.revision_root,
            _pinned_asset_spec(relative_path),
        )

    if request.node_id in DINO_NODE_IDS:
        _authenticate_source_tree(
            request.state.revision_root,
            _pinned_source_spec(DINO_SOURCE_PATH),
        )
        _authenticate_asset(request.state.revision_root, DINO_CHECKPOINT_SPEC)

    if request.backend == "portable":
        portable_root = request.state.revision_root / "source" / "LATO.2-portable"
        if not validate_portable_runtime(portable_root):
            raise ProcessFailure("SETUP_INVALID")


def _validate_state(payload: Mapping[str, Any]) -> RuntimeState:
    config = _runtime_config()
    models_root = _resolve_models(payload, config)
    try:
        revision = owned_snapshot_directory(models_root, create=False)
    except PathContractError as exc:
        raise ProcessFailure("SETUP_INVALID") from exc
    configured_revision = config.get("revision_dir")
    if not isinstance(configured_revision, str) or not Path(configured_revision).is_absolute():
        raise ProcessFailure("SETUP_INVALID")
    if _canonical(revision) != _canonical(Path(configured_revision)):
        raise ProcessFailure("SETUP_INVALID")
    _validate_runtime_config_identity(config, revision)
    _validate_marker(revision)
    for source_spec in SOURCE_ARCHIVES:
        valid, _reason = verify_source_tree(revision, source_spec)
        if not valid:
            raise ProcessFailure("SETUP_INVALID")
    paths = snapshot_paths(revision)
    required_files = (
        paths.lato_source / "scripts" / "e2e_inference.py",
        paths.lato_source / "scripts" / "vflow_inference.py",
        paths.lato_source / "scripts" / "vvae_inference.py",
        paths.lato_source / "scripts" / "tflow_inference.py",
        paths.dino_source / "hubconf.py",
        paths.dino_checkpoint,
    )
    for required in required_files:
        _regular_file(required)
    available = config.get("available_backends")
    default = config.get("default_backend")
    if (
        not isinstance(available, list)
        or not available
        or any(value not in BACKENDS for value in available)
        or len(set(available)) != len(available)
        or default not in available
    ):
        raise ProcessFailure("SETUP_INVALID")
    if "portable" in available:
        portable_root = revision / "source" / "LATO.2-portable"
        for name in ("e2e_inference.py", "vflow_inference.py", "vvae_inference.py", "tflow_inference.py"):
            _regular_file(portable_root / "scripts" / name)
        if importlib.util.find_spec("lato2_ovoxel_cpu") is None:
            raise ProcessFailure("SETUP_INVALID")
        if not validate_portable_runtime(portable_root):
            raise ProcessFailure("SETUP_INVALID")
    attention = config.get("attention_backend")
    if attention not in {"flash_attn", "xformers", "sdpa"}:
        raise ProcessFailure("SETUP_INVALID")
    if "upstream" in available and attention not in {"flash_attn", "xformers"}:
        raise ProcessFailure("SETUP_INVALID")
    if "upstream" not in available and attention != "sdpa":
        raise ProcessFailure("SETUP_INVALID")
    portable_precision = config.get("portable_precision_env", False)
    if not isinstance(portable_precision, bool):
        raise ProcessFailure("SETUP_INVALID")
    portable_precisions_raw = config.get("portable_precisions")
    if (
        not isinstance(portable_precisions_raw, list)
        or not portable_precisions_raw
        or any(value not in PRECISIONS for value in portable_precisions_raw)
        or len(set(portable_precisions_raw)) != len(portable_precisions_raw)
        or "auto" not in portable_precisions_raw
    ):
        raise ProcessFailure("SETUP_INVALID")
    return RuntimeState(
        models_root=_canonical(models_root),
        revision_root=_canonical(revision),
        source_root=_canonical(paths.lato_source),
        checkpoints=_canonical(paths.checkpoints),
        dino_hub=_canonical(paths.dino_hub),
        default_backend=str(default),
        available_backends=frozenset(available),
        attention_backend=attention,
        portable_precision_env=portable_precision,
        portable_precisions=frozenset(portable_precisions_raw),
    )


def _read_one_payload(stream: TextIO) -> dict[str, Any]:
    line = stream.readline(MAX_JSON_BYTES + 1)
    if not line:
        raise ProcessFailure("REQUEST_EMPTY")
    if len(line.encode("utf-8", errors="replace")) > MAX_JSON_BYTES:
        raise ProcessFailure("REQUEST_JSON")
    if stream.read(1):
        raise ProcessFailure("REQUEST_COUNT")
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProcessFailure("REQUEST_JSON") from exc
    if not isinstance(value, dict):
        raise ProcessFailure("REQUEST_TYPE")
    return value


def _directory(value: object) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ProcessFailure("REQUEST_PATHS")
    candidate = Path(value)
    if not candidate.is_absolute() or not candidate.is_dir():
        raise ProcessFailure("REQUEST_PATHS")
    return _canonical(candidate)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _is_filesystem_root(path: Path) -> bool:
    return path.parent == path


def _input_mesh(value: object) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ProcessFailure("REQUEST_INPUT")
    candidate = Path(value)
    if not candidate.is_absolute() or candidate.suffix.casefold() not in SUPPORTED_INPUT_SUFFIXES:
        raise ProcessFailure("REQUEST_INPUT")
    try:
        info = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProcessFailure("REQUEST_INPUT") from exc
    if (
        _is_alias(info)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
        or info.st_size > MAX_INPUT_BYTES
    ):
        raise ProcessFailure("REQUEST_INPUT")
    return resolved


def _fail_input() -> None:
    raise ProcessFailure("REQUEST_INPUT")


def _windows_unsafe_path_component(value: str) -> bool:
    if (
        not value
        or value.rstrip(" .") != value
        or any(ord(character) < 32 or character in '<>:"/\\|?*' for character in value)
    ):
        return True
    stem = value.split(".", 1)[0].upper()
    return stem in WINDOWS_RESERVED_NAMES


def _relative_reference_parts(
    reference: str,
    *,
    uri: bool,
) -> tuple[str, ...] | None:
    if not reference or "\0" in reference:
        _fail_input()
    value = reference
    if uri:
        if value[:5].casefold() == "data:":
            header, separator, payload = value.partition(",")
            if (
                not separator
                or not header.casefold().endswith(";base64")
                or len(payload) % 4 != 0
                or re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", payload) is None
            ):
                _fail_input()
            return None
        if re.search(r"%(?:25)*(?:2f|5c)", value, re.IGNORECASE) or re.search(
            r"(?:^|/)(?:\.|%(?:25)*2e){2}(?:$|/)", value, re.IGNORECASE
        ):
            _fail_input()
        if re.search(r"%(?![0-9a-f]{2})", value, re.IGNORECASE):
            _fail_input()
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            _fail_input()
        try:
            value = unquote(parsed.path, encoding="utf-8", errors="strict")
        except (UnicodeError, ValueError):
            _fail_input()
        if "\\" in value:
            _fail_input()
    else:
        value = value.strip()
        if not value or value.startswith(("/", "\\")):
            _fail_input()
        normalized_for_url = value.replace("\\", "/")
        parsed = urlsplit(normalized_for_url)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            _fail_input()
        value = normalized_for_url

    if (
        not value
        or "\0" in value
        or value.startswith("/")
        or ":" in value
        or re.match(r"^[a-zA-Z]:", value)
    ):
        _fail_input()
    raw_parts = value.split("/")
    if any(part in {"", "."} for part in raw_parts):
        _fail_input()
    if ".." in raw_parts:
        _fail_input()
    if any(
        part != ".." and _windows_unsafe_path_component(part) for part in raw_parts
    ):
        _fail_input()
    pure = PurePosixPath(*raw_parts)
    if pure.is_absolute() or tuple(pure.parts) != tuple(raw_parts):
        _fail_input()
    return tuple(raw_parts)


def _literal_uri_parts(reference: str) -> tuple[str, ...]:
    """Return the safe literal path spelling used by trimesh's GLB resolver."""

    parsed = urlsplit(reference)
    value = parsed.path
    if not value or "\\" in value:
        _fail_input()
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        _fail_input()
    if any(_windows_unsafe_path_component(part) for part in parts):
        _fail_input()
    return tuple(parts)


def _resolve_confined_sidecar(
    source_root: Path,
    reference_base: Path,
    parts: tuple[str, ...],
) -> tuple[Path, Path]:
    try:
        base_relative = reference_base.relative_to(source_root)
    except ValueError:
        _fail_input()
    normalized_parts = list(base_relative.parts)
    for component in parts:
        if component == "..":
            if not normalized_parts:
                _fail_input()
            normalized_parts.pop()
        else:
            normalized_parts.append(component)
    if not normalized_parts:
        _fail_input()
    relative = Path(*normalized_parts)
    current = source_root
    components = relative.parts
    try:
        for index, component in enumerate(components):
            current = current / component
            info = current.lstat()
            if _is_alias(info):
                _fail_input()
            final = index == len(components) - 1
            if final and (
                not stat.S_ISREG(info.st_mode)
                or getattr(info, "st_nlink", 1) != 1
            ):
                _fail_input()
            if not final and not stat.S_ISDIR(info.st_mode):
                _fail_input()
        resolved = current.resolve(strict=True)
        canonical_root = source_root.resolve(strict=True)
    except OSError:
        _fail_input()
    if canonical_root not in resolved.parents or resolved == canonical_root:
        _fail_input()
    return resolved, relative


def _copy_verified_input_file(
    source: Path,
    destination: Path,
    *,
    accumulated: int,
    limit: int,
) -> int:
    try:
        source_info = source.lstat()
        if (
            _is_alias(source_info)
            or not stat.S_ISREG(source_info.st_mode)
            or getattr(source_info, "st_nlink", 1) != 1
        ):
            _fail_input()
        with source.open("rb") as source_handle:
            opened = os.fstat(source_handle.fileno())
            identity = (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
            )
            if (
                not stat.S_ISREG(opened.st_mode)
                or getattr(opened, "st_nlink", 1) != 1
                or source_info.st_dev != opened.st_dev
                or source_info.st_ino != opened.st_ino
                or source_info.st_size != opened.st_size
                or accumulated + opened.st_size > limit
            ):
                _fail_input()
            destination.parent.mkdir(parents=True, exist_ok=True)
            copied = 0
            with destination.open("xb") as destination_handle:
                while block := source_handle.read(1024 * 1024):
                    copied += len(block)
                    if accumulated + copied > limit or copied > opened.st_size:
                        _fail_input()
                    destination_handle.write(block)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
            final = os.fstat(source_handle.fileno())
            if (
                copied != opened.st_size
                or (
                    final.st_dev,
                    final.st_ino,
                    final.st_size,
                    final.st_mtime_ns,
                )
                != identity
                or getattr(final, "st_nlink", 1) != 1
            ):
                destination.unlink(missing_ok=True)
                _fail_input()
    except ProcessFailure:
        raise
    except (OSError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        raise ProcessFailure("REQUEST_INPUT") from exc
    return accumulated + copied


def _gltf_reference_entries(
    document: object,
    section_names: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(document, dict):
        _fail_input()
    entries: list[dict[str, Any]] = []
    for section_name in section_names:
        section = document.get(section_name, [])
        if not isinstance(section, list):
            _fail_input()
        for entry in section:
            if not isinstance(entry, dict):
                _fail_input()
            if "uri" not in entry:
                continue
            uri_value = entry["uri"]
            if not isinstance(uri_value, str):
                _fail_input()
            if _relative_reference_parts(uri_value, uri=True) is None:
                continue
            entries.append(entry)
            if len(entries) > MAX_MESH_REFERENCES:
                _fail_input()
    return tuple(entries)


def _gltf_document(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_GLTF_JSON_BYTES:
            _fail_input()
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except ProcessFailure:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProcessFailure("REQUEST_INPUT") from exc
    if not isinstance(document, dict):
        _fail_input()
    _gltf_reference_entries(document, ("buffers",))
    return document


def _gltf_references(path: Path) -> tuple[str, ...]:
    return tuple(
        str(entry["uri"])
        for entry in _gltf_reference_entries(_gltf_document(path), ("buffers",))
    )


def _glb_document(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            header = handle.read(12)
            if len(header) != 12:
                _fail_input()
            magic, version, declared_size = struct.unpack("<4sII", header)
            if magic != b"glTF" or version != 2 or declared_size != size:
                _fail_input()
            chunk_header = handle.read(8)
            if len(chunk_header) != 8:
                _fail_input()
            chunk_size, chunk_type = struct.unpack("<II", chunk_header)
            if (
                chunk_type != 0x4E4F534A
                or chunk_size <= 0
                or chunk_size > MAX_GLTF_JSON_BYTES
                or 20 + chunk_size > size
            ):
                _fail_input()
            raw = handle.read(chunk_size)
        document = json.loads(raw.rstrip(b" \t\r\n\0").decode("utf-8-sig"))
    except ProcessFailure:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, struct.error) as exc:
        raise ProcessFailure("REQUEST_INPUT") from exc
    if not isinstance(document, dict):
        _fail_input()
    _gltf_reference_entries(document, ("buffers",))
    return document


def _rewrite_gltf_uris(path: Path, rewrites: Mapping[str, str]) -> None:
    if not rewrites:
        return
    document = _gltf_document(path)
    for entry in _gltf_reference_entries(document, ("buffers",)):
        uri_value = str(entry["uri"])
        replacement = rewrites.get(uri_value)
        if replacement is None:
            _fail_input()
        entry["uri"] = replacement
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if not encoded or len(encoded) > MAX_GLTF_JSON_BYTES:
        _fail_input()
    temporary = path.with_name(f".{path.name}.rewrite-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ProcessFailure("REQUEST_INPUT") from exc


def _stage_input_bundle(source: Path, staging_root: Path) -> Path:
    source_root = source.parent.resolve(strict=True)
    staged_by_source: dict[tuple[Path, Path], Path] = {}
    accumulated = 0
    try:
        staging_root.mkdir()

        def stage_file(
            original: Path,
            relative: Path,
            *,
            source_relative: Path | None = None,
        ) -> Path:
            nonlocal accumulated
            key = (original, relative)
            if key in staged_by_source:
                return staged_by_source[key]
            verification_relative = source_relative or relative
            confirmed, confirmed_relative = _resolve_confined_sidecar(
                source_root,
                source_root,
                tuple(verification_relative.parts),
            )
            if confirmed != original or confirmed_relative != verification_relative:
                _fail_input()
            destination = staging_root.joinpath(*relative.parts)
            accumulated = _copy_verified_input_file(
                original,
                destination,
                accumulated=accumulated,
                limit=MAX_INPUT_BYTES,
            )
            confirmed_after, relative_after = _resolve_confined_sidecar(
                source_root,
                source_root,
                tuple(verification_relative.parts),
            )
            if (
                confirmed_after != original
                or relative_after != verification_relative
            ):
                _fail_input()
            staged_by_source[key] = destination
            return destination

        main = stage_file(source, Path(source.name))

        def stage_reference(
            base: Path,
            reference: str,
            *,
            uri: bool,
            literal_destination: bool = False,
        ) -> tuple[Path, Path]:
            parts = _relative_reference_parts(
                reference,
                uri=uri,
            )
            if parts is None:
                _fail_input()
            original, relative = _resolve_confined_sidecar(source_root, base, parts)
            destination_relative = (
                Path(*_literal_uri_parts(reference))
                if literal_destination
                else relative
            )
            return original, stage_file(
                original,
                destination_relative,
                source_relative=relative,
            )

        suffix = source.suffix.casefold()
        if suffix == ".gltf":
            rewrites: dict[str, str] = {}
            for reference in _gltf_references(main):
                _original, staged = stage_reference(source_root, reference, uri=True)
                rewrites[reference] = staged.relative_to(staging_root).as_posix()
            _rewrite_gltf_uris(main, rewrites)
        elif suffix == ".glb":
            document = _glb_document(main)
            for entry in _gltf_reference_entries(document, ("buffers",)):
                reference = str(entry["uri"])
                stage_reference(
                    source_root,
                    reference,
                    uri=True,
                    literal_destination=True,
                )
        # OBJ/MTL and PLY materials are deliberately not opened: _prepare_input
        # loads geometry with skip_materials=True and exports a geometry-only GLB.
        return main
    except ProcessFailure:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    except (OSError, ValueError) as exc:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise ProcessFailure("REQUEST_INPUT") from exc


def _parse_int(value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ProcessFailure("REQUEST_PARAMS")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        parsed = int(value)
    else:
        raise ProcessFailure("REQUEST_PARAMS")
    if parsed < minimum or parsed > maximum:
        raise ProcessFailure("REQUEST_PARAMS")
    return parsed


def _parse_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ProcessFailure("REQUEST_PARAMS")
    if isinstance(value, str) and (not value.strip() or len(value) > 64):
        raise ProcessFailure("REQUEST_PARAMS")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProcessFailure("REQUEST_PARAMS") from exc
    if not math.isfinite(parsed):
        raise ProcessFailure("REQUEST_PARAMS")
    return parsed


def _parse_parameters(node_id: str, values: Mapping[str, Any]) -> dict[str, Any]:
    defaults = NODE_DEFAULTS[node_id]
    if set(values) - set(defaults):
        raise ProcessFailure("REQUEST_PARAMS")
    parsed = dict(defaults)
    parsed.update(values)
    for name in BOOL_PARAMETERS & parsed.keys():
        if parsed[name] not in {"true", "false"}:
            raise ProcessFailure("REQUEST_PARAMS")
        parsed[name] = parsed[name] == "true"
    if parsed["backend"] not in {"auto", *BACKENDS} or parsed["precision"] not in PRECISIONS:
        raise ProcessFailure("REQUEST_PARAMS")
    if "sample_type" in parsed and parsed["sample_type"] not in {"dora", "uniform"}:
        raise ProcessFailure("REQUEST_PARAMS")
    # NumPy's legacy RandomState, used verbatim by all four upstream scripts,
    # accepts only unsigned 32-bit seeds.
    parsed["seed"] = _parse_int(parsed["seed"], minimum=0, maximum=2**32 - 1)
    parsed["num_workers"] = _parse_int(parsed["num_workers"], minimum=0, maximum=64)
    for name in POSITIVE_INTS & parsed.keys():
        parsed[name] = _parse_int(parsed[name], minimum=1, maximum=2**31 - 1)
    for name in FLOAT_PARAMETERS & parsed.keys():
        parsed[name] = _parse_float(parsed[name])
    if "inference_threshold" in parsed and not 0.0 <= parsed["inference_threshold"] <= 1.0:
        raise ProcessFailure("REQUEST_PARAMS")
    if "img_res" in parsed and parsed["img_res"] > 4096:
        raise ProcessFailure("REQUEST_PARAMS")
    if "render_azimuth" in parsed and not -360.0 <= parsed["render_azimuth"] <= 360.0:
        raise ProcessFailure("REQUEST_PARAMS")
    if "render_elevation" in parsed and not -90.0 <= parsed["render_elevation"] <= 90.0:
        raise ProcessFailure("REQUEST_PARAMS")
    if "min_verts" in parsed and parsed["min_verts"] > parsed["max_verts"]:
        raise ProcessFailure("REQUEST_PARAMS")
    return parsed


def validate_request(payload: Mapping[str, Any]) -> ValidatedRequest:
    node_id = payload.get("nodeId")
    if not isinstance(node_id, str) or node_id not in NODE_IDS:
        raise ProcessFailure("REQUEST_NODE")
    input_data = payload.get("input")
    params = payload.get("params", {})
    if not isinstance(input_data, dict):
        raise ProcessFailure("REQUEST_INPUT")
    if not isinstance(params, dict):
        raise ProcessFailure("REQUEST_PARAMS")
    workspace = _directory(payload.get("workspaceDir"))
    temporary = _directory(payload.get("tempDir"))
    extension_root = _canonical(ROOT)
    if (
        _is_filesystem_root(workspace)
        or _is_filesystem_root(temporary)
        or _paths_overlap(workspace, extension_root)
        or _paths_overlap(temporary, extension_root)
    ):
        raise ProcessFailure("REQUEST_PATHS")
    mesh = _input_mesh(input_data.get("filePath"))
    state = _validate_state(payload)
    normalized = _parse_parameters(node_id, params)
    requested_backend = normalized["backend"]
    backend = state.default_backend if requested_backend == "auto" else requested_backend
    if backend not in state.available_backends:
        raise ProcessFailure("BACKEND_UNAVAILABLE")
    precision = normalized["precision"]
    if backend == "upstream" and precision not in {"auto", "bfloat16"}:
        raise ProcessFailure("PRECISION_UNAVAILABLE")
    if backend == "portable" and precision != "auto" and not state.portable_precision_env:
        raise ProcessFailure("PRECISION_UNAVAILABLE")
    if backend == "portable" and precision not in state.portable_precisions:
        raise ProcessFailure("PRECISION_UNAVAILABLE")
    for mutable in (workspace, temporary, extension_root):
        if _paths_overlap(state.models_root, mutable):
            raise ProcessFailure("REQUEST_PATHS")
    return ValidatedRequest(
        node_id=node_id,
        input_path=mesh,
        params=normalized,
        workspace_dir=workspace,
        temp_dir=temporary,
        state=state,
        backend=backend,
        precision=precision,
    )


def _owned_output_parent(workspace: Path) -> Path:
    workflows = workspace / "Workflows"
    try:
        workflows.mkdir(exist_ok=True)
        workflows_info = workflows.lstat()
        if _is_alias(workflows_info) or not stat.S_ISDIR(workflows_info.st_mode):
            raise OSError("unsafe Workflows directory")
        if _canonical(workflows).parent != workspace:
            raise OSError("Workflows escapes workspace")
        owned = workflows / "LATO2"
        owned.mkdir(exist_ok=True)
        if _canonical(owned).parent != _canonical(workflows):
            raise OSError("LATO2 output escapes Workflows")
        info = owned.lstat()
        if _is_alias(info) or not stat.S_ISDIR(info.st_mode):
            raise OSError("unsafe output directory")
    except OSError as exc:
        raise ProcessFailure("OUTPUT_INVALID") from exc
    return _canonical(owned)


def _prepare_input(source: Path, temp_root: Path) -> tuple[Path, Path]:
    run_root = temp_root / f"modly-lato2-{uuid.uuid4().hex}"
    input_dir = run_root / "input"
    source_dir = run_root / "source"
    try:
        input_dir.mkdir(parents=True)
        staged_source = _stage_input_bundle(source, source_dir)
        import numpy as np
        import trimesh

        # The normalized GLB intentionally carries geometry only.  Disabling
        # material loaders prevents format-specific resolvers (notably OBJ/PLY)
        # from reopening unvalidated paths after the controlled bundle copy.
        loaded = trimesh.load(
            str(staged_source),
            process=False,
            force="scene",
            skip_materials=True,
        )
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.dump(concatenate=True)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise ValueError("unsupported geometry container")
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if (
            vertices.ndim != 2
            or vertices.shape[0] < 3
            or vertices.shape[1] != 3
            or faces.ndim != 2
            or faces.shape[0] < 1
            or faces.shape[1] != 3
            or not bool(np.isfinite(vertices).all())
            or not bool(np.issubdtype(faces.dtype, np.integer))
            or int(faces.min()) < 0
            or int(faces.max()) >= vertices.shape[0]
        ):
            raise ValueError("mesh must contain finite triangle geometry")
        normalized = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        encoded = normalized.export(file_type="glb")
        if not isinstance(encoded, (bytes, bytearray)) or encoded[:4] != b"glTF":
            raise ValueError("GLB export failed")
        target = input_dir / "input.glb"
        with target.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException as exc:
        shutil.rmtree(run_root, ignore_errors=True)
        if isinstance(exc, ProcessFailure):
            raise
        raise ProcessFailure("INPUT_CONVERSION") from exc
    return run_root, target


def _ckpt(state: RuntimeState, name: str) -> str:
    relative = LATO_CHECKPOINT_PATHS[name]
    path = state.revision_root.joinpath(*relative.split("/"))
    _regular_file(path)
    return str(path)


def _boolean_flag(name: str, value: bool) -> str:
    return f"--{name}" if value else f"--no-{name}"


def _base_command(request: ValidatedRequest, input_dir: Path, output_dir: Path) -> list[str]:
    source = (
        request.state.source_root
        if request.backend == "upstream"
        else request.state.revision_root / "source" / "LATO.2-portable"
    )
    scripts = {
        "lato2-e2e": "e2e_inference.py",
        "lato2-vflow": "vflow_inference.py",
        "lato2-vvae": "vvae_inference.py",
        "lato2-tflow": "tflow_inference.py",
    }
    script = source / "scripts" / scripts[request.node_id]
    _regular_file(script)
    return [
        sys.executable,
        "-I",
        "-B",
        "-X",
        "utf8",
        str(script),
        "--mesh_dir",
        str(input_dir),
        "--out_dir",
        str(output_dir),
        "--batch_size",
        "1",
        "--num_samples",
        "1",
        "--num_workers",
        str(request.params["num_workers"]),
        "--seed",
        str(request.params["seed"]),
    ]


def _command(request: ValidatedRequest, input_dir: Path, output_dir: Path) -> list[str]:
    p = request.params
    command = _base_command(request, input_dir, output_dir)
    if request.node_id == "lato2-e2e":
        command += [
            "--vflow_ckpt", _ckpt(request.state, "vflow"),
            "--vvae_ckpt", _ckpt(request.state, "vvae"),
            "--offset_head_ckpt", _ckpt(request.state, "offset_head"),
            "--tflow_ckpt", _ckpt(request.state, "tflow"),
            "--tvae_ckpt", _ckpt(request.state, "tvae"),
            "--voxel_encoder_ckpt", _ckpt(request.state, "voxel_encoder"),
            "--dino_hub_dir", str(request.state.dino_hub),
            "--inference_threshold", str(p["inference_threshold"]),
            "--vflow_steps", str(p["vflow_steps"]),
            "--cfg_strength", str(p["cfg_strength"]),
            "--rescale_t", str(p["rescale_t"]),
            "--vert_num", str(p["vert_num"]),
            _boolean_flag("use_gt_vert_count", p["use_gt_vert_count"]),
            "--scaler", str(p["scaler"]),
            "--min_verts", str(p["min_verts"]),
            "--max_verts", str(p["max_verts"]),
            "--tflow_steps", str(p["tflow_steps"]),
            "--edge_threshold", str(p["edge_threshold"]),
            "--chunk_size", str(p["chunk_size"]),
            _boolean_flag("fill_quad_rings", p["fill_quad_rings"]),
            "--render_azimuth", str(p["render_azimuth"]),
            "--render_elevation", str(p["render_elevation"]),
            "--img_res", str(p["img_res"]),
        ]
    elif request.node_id == "lato2-vflow":
        command += [
            "--vflow_ckpt", _ckpt(request.state, "vflow"),
            "--vvae_ckpt", _ckpt(request.state, "vvae"),
            "--vdf_encoder_ckpt", _ckpt(request.state, "vdf_encoder"),
            "--offset_head_ckpt", _ckpt(request.state, "offset_head"),
            "--dino_hub_dir", str(request.state.dino_hub),
            "--pc_sample_number", str(p["pc_sample_number"]),
            "--sample_type", p["sample_type"],
            "--inference_threshold", str(p["inference_threshold"]),
            _boolean_flag("reconstruct", p["reconstruct"]),
            _boolean_flag("sample_posterior", p["sample_posterior"]),
            "--steps", str(p["vflow_steps"]),
            "--cfg_strength", str(p["cfg_strength"]),
            "--rescale_t", str(p["rescale_t"]),
            "--vert_num", str(p["vert_num"]),
            _boolean_flag("use_gt_vert_count", p["use_gt_vert_count"]),
            "--scaler", str(p["scaler"]),
            "--min_verts", str(p["min_verts"]),
            "--max_verts", str(p["max_verts"]),
            "--render_azimuth", str(p["render_azimuth"]),
            "--render_elevation", str(p["render_elevation"]),
            "--img_res", str(p["img_res"]),
        ]
    elif request.node_id == "lato2-vvae":
        command += [
            "--vvae_ckpt", _ckpt(request.state, "vvae"),
            "--vdf_encoder_ckpt", _ckpt(request.state, "vdf_encoder"),
            "--offset_head_ckpt", _ckpt(request.state, "offset_head"),
            "--pc_sample_number", str(p["pc_sample_number"]),
            "--sample_type", p["sample_type"],
            "--inference_threshold", str(p["inference_threshold"]),
            _boolean_flag("sample_posterior", p["sample_posterior"]),
        ]
    else:
        command += [
            "--tflow_ckpt", _ckpt(request.state, "tflow"),
            "--tvae_ckpt", _ckpt(request.state, "tvae"),
            "--voxel_encoder_ckpt", _ckpt(request.state, "voxel_encoder"),
            "--steps", str(p["tflow_steps"]),
            _boolean_flag("use_cond", p["use_cond"]),
            "--edge_threshold", str(p["edge_threshold"]),
            "--chunk_size", str(p["chunk_size"]),
            _boolean_flag("fill_quad_rings", p["fill_quad_rings"]),
            _boolean_flag("save_voxel_field", p["save_voxel_field"]),
        ]
    return command


def _runtime_environment(request: ValidatedRequest) -> dict[str, str]:
    renderer_override = os.environ.get("LATO2_RENDERER")
    env = {
        key: value
        for key, value in os.environ.items()
        if not SENSITIVE_ENVIRONMENT.search(key)
        and key.upper() not in BLOCKED_INFERENCE_ENVIRONMENT
        and (
            key.upper() in PASSTHROUGH_INFERENCE_ENVIRONMENT
            or key.upper().startswith(PASSTHROUGH_INFERENCE_PREFIXES)
        )
    }
    try:
        cache = safe_snapshot_directory(
            request.state.revision_root, "runtime-cache", create=True
        )
        owned_cache = {
            name: safe_snapshot_directory(
                request.state.revision_root, f"runtime-cache/{name}", create=True
            )
            for name in (
                "huggingface",
                "triton",
                "torch-extensions",
                "flex_gemm",
                "cuda",
                "torchinductor",
                "xdg-runtime",
                "home",
                "temporary",
                "xdg-cache",
                "xdg-config",
            )
        }
    except PathContractError as exc:
        raise ProcessFailure("SETUP_INVALID") from exc
    try:
        owned_cache["xdg-runtime"].chmod(0o700)
    except OSError:
        pass
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HOME": str(owned_cache["huggingface"]),
            "TORCH_HOME": str(request.state.dino_hub),
            "TRITON_CACHE_DIR": str(owned_cache["triton"]),
            "TORCH_EXTENSIONS_DIR": str(owned_cache["torch-extensions"]),
            "CUDA_CACHE_PATH": str(owned_cache["cuda"]),
            "TORCHINDUCTOR_CACHE_DIR": str(owned_cache["torchinductor"]),
            "MODLY_LATO2_CACHE_DIR": str(cache),
            "FLEX_GEMM_AUTOTUNE_CACHE_PATH": str(
                owned_cache["flex_gemm"] / "autotune_cache.json"
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
            "LATO2_PRECISION": request.precision,
            "HOME": str(owned_cache["home"]),
            "USERPROFILE": str(owned_cache["home"]),
            "TEMP": str(owned_cache["temporary"]),
            "TMP": str(owned_cache["temporary"]),
            "TMPDIR": str(owned_cache["temporary"]),
            "XDG_CACHE_HOME": str(owned_cache["xdg-cache"]),
            "XDG_CONFIG_HOME": str(owned_cache["xdg-config"]),
            "MPLCONFIGDIR": str(owned_cache["xdg-config"]),
        }
    )
    env["XDG_RUNTIME_DIR"] = str(owned_cache["xdg-runtime"])
    if sys.platform.startswith("linux"):
        env["EGL_PLATFORM"] = "surfaceless"
    if request.backend == "portable":
        if renderer_override is not None:
            env["LATO2_RENDERER"] = renderer_override
        env["SPARSE_BACKEND"] = "torch"
        env["SPARSE_ATTN_BACKEND"] = "sdpa"
        env.pop("ATTN_BACKEND", None)
        return env
    # This switch belongs only to the portable adapter. Do not let an inherited
    # diagnostic override imply that the exact upstream renderer was software.
    env.pop("LATO2_RENDERER", None)
    env["SPARSE_BACKEND"] = "spconv"
    attention = request.state.attention_backend
    if attention not in {"flash_attn", "xformers"}:
        raise ProcessFailure("SETUP_INVALID")
    env["SPARSE_ATTN_BACKEND"] = attention
    return env


def _log_tail(path: Path, emitter: ProtocolEmitter, replacements: Mapping[str, str]) -> None:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - MAX_LOG_BYTES))
            raw = handle.read(MAX_LOG_BYTES)
        text = raw.decode("utf-8", errors="replace")
    except OSError:
        return
    text = _sanitize_log_text(text, replacements)
    lines = [line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip()]
    for line in lines[-8:]:
        emitter.log("LATO.2: " + line[:700])


def _sanitize_log_text(text: str, replacements: Mapping[str, str]) -> str:
    """Remove local paths and common credential forms before persistence/UI."""

    for source, label in replacements.items():
        if not source:
            continue
        text = text.replace(source, label)
        alternate = source.replace("\\", "/")
        if alternate != source:
            text = text.replace(alternate, label)
    text = re.sub(
        r"(?i)\b(?:https?|ftp|socks[45]?)://[^\s/@]+@",
        lambda match: match.group(0).split("://", 1)[0] + "://[redacted]@",
        text,
    )
    text = re.sub(
        r"(?i)\b(?:proxy-authorization|authorization|set-cookie|cookie)\s*:\s*[^\r\n]+",
        "[redacted credential header]",
        text,
    )
    text = re.sub(
        r"(?i)\b(token|api[_-]?key|password|passwd|secret|cookie)\s*[:=]\s*"
        r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;]+)",
        lambda match: f"{match.group(1)}=[redacted]",
        text,
    )
    text = re.sub(r"(?i)\bhf_[A-Za-z0-9._-]{10,}\b", "[redacted]", text)
    text = re.sub(
        r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}",
        "Bearer [redacted]",
        text,
    )
    return text


def _bounded_sanitized_log(
    raw: bytes,
    replacements: Mapping[str, str],
    *,
    maximum: int,
) -> bytes:
    text = _sanitize_log_text(raw.decode("utf-8", errors="replace"), replacements)
    encoded = text.encode("utf-8")
    if len(encoded) <= maximum:
        return encoded

    prefix = b""
    if encoded.startswith(b"[Modly retained "):
        newline = encoded.find(b"\n")
        if 0 <= newline < MAX_LOG_OVERHEAD_BYTES:
            prefix = encoded[: newline + 1]
            encoded = encoded[newline + 1 :]

    footer = b""
    footer_match = re.search(
        rb"(?:^|\n)(\[Modly effective renderer: (?:software|open3d|no-render)\]\n?)$",
        encoded,
    )
    if footer_match is not None:
        footer = footer_match.group(1)
        encoded = encoded[: footer_match.start(1)]
    separator = b"\n" if footer and encoded and not encoded.endswith(b"\n") else b""
    budget = maximum - len(prefix) - len(separator) - len(footer)
    if budget < 0:
        raise ProcessFailure("INFERENCE_FAILED")
    tail = encoded[-budget:] if budget else b""
    # Never persist a partial UTF-8 sequence after front truncation.
    tail = tail.decode("utf-8", errors="ignore").encode("utf-8")
    return prefix + tail + separator + footer


def _sanitize_persistent_log(
    path: Path,
    replacements: Mapping[str, str],
) -> None:
    """Atomically sanitize one already bounded child log in place."""

    maximum = MAX_LOG_BYTES + MAX_LOG_OVERHEAD_BYTES
    temporary = path.with_name(f".{path.name}.sanitize-{uuid.uuid4().hex}")
    try:
        info = path.lstat()
        if (
            _is_alias(info)
            or not stat.S_ISREG(info.st_mode)
            or getattr(info, "st_nlink", 1) != 1
            or info.st_size > maximum
        ):
            raise OSError("unsafe or oversized upstream log")
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or getattr(opened, "st_nlink", 1) != 1
                or opened.st_dev != info.st_dev
                or opened.st_ino != info.st_ino
                or opened.st_size != info.st_size
            ):
                raise OSError("upstream log identity changed")
            raw = handle.read(maximum + 1)
        if len(raw) != info.st_size:
            raise OSError("upstream log changed while sanitizing")
        sanitized = _bounded_sanitized_log(raw, replacements, maximum=maximum)
        with temporary.open("xb") as handle:
            handle.write(sanitized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        final = path.lstat()
        if (
            _is_alias(final)
            or not stat.S_ISREG(final.st_mode)
            or getattr(final, "st_nlink", 1) != 1
            or final.st_size > maximum
        ):
            raise OSError("sanitized log publication failed")
    except ProcessFailure:
        temporary.unlink(missing_ok=True)
        raise
    except (OSError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise ProcessFailure("INFERENCE_FAILED") from exc


def _capture_bounded_output(
    source: Any,
    destination: Path,
    result: dict[str, Any],
) -> None:
    """Drain a child pipe without allowing its persistent log to grow forever."""

    tail = bytearray()
    total = 0
    fallback_needle = b"portable software fallback"
    overlap = b""
    try:
        while True:
            block = source.read(64 * 1024)
            if not block:
                break
            total += len(block)
            combined = overlap + block
            if fallback_needle in combined:
                result["renderer_fallback"] = True
            overlap = combined[-(len(fallback_needle) - 1) :]
            tail.extend(block)
            if len(tail) > MAX_LOG_BYTES:
                del tail[: len(tail) - MAX_LOG_BYTES]
        prefix = (
            f"[Modly retained the final {MAX_LOG_BYTES} of {total} log bytes.]\n".encode(
                "utf-8"
            )
            if total > MAX_LOG_BYTES
            else b""
        )
        with destination.open("xb") as handle:
            handle.write(prefix)
            handle.write(tail)
            handle.flush()
            os.fsync(handle.fileno())
            persisted = os.fstat(handle.fileno())
            result["log_identity"] = (
                persisted.st_dev,
                persisted.st_ino,
                persisted.st_size,
                persisted.st_mtime_ns,
            )
        result["total"] = total
    except BaseException as exc:
        result["error"] = exc
    finally:
        try:
            source.close()
        except BaseException:
            pass


def _record_effective_renderer(log_path: Path, renderer: str) -> None:
    marker = f"[Modly effective renderer: {renderer}]\n".encode("utf-8")
    maximum = MAX_LOG_BYTES + MAX_LOG_OVERHEAD_BYTES
    try:
        current = log_path.read_bytes()
        separator_size = int(bool(current) and not current.endswith(b"\n"))
        if len(current) + separator_size + len(marker) > maximum:
            prefix = b""
            body = current
            if current.startswith(b"[Modly retained "):
                newline = current.find(b"\n")
                if newline >= 0:
                    prefix = current[: newline + 1]
                    body = current[newline + 1 :]
            # Reserve one byte for a newline even when the retained body happens
            # to end with one; the persistent bound stays exact in both cases.
            keep = max(0, maximum - len(prefix) - len(marker) - 1)
            current = prefix + body[-keep:] if keep else prefix
        with log_path.open("wb") as handle:
            handle.write(current)
            if current and not current.endswith(b"\n"):
                handle.write(b"\n")
            handle.write(marker)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ProcessFailure("INFERENCE_FAILED") from exc


def _process_group_spawn_options(platform_name: str | None = None) -> dict[str, Any]:
    """Return shell-free child isolation flags for the current OS."""

    platform_value = sys.platform if platform_name is None else platform_name
    if platform_value.startswith("win"):
        creation_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", None)
        if not isinstance(creation_flag, int) or creation_flag <= 0:
            raise OSError("Windows process-group support is unavailable")
        return {"creationflags": creation_flag}
    return {"start_new_session": True}


def _install_child_cancel_handlers() -> dict[int, Any]:
    if threading.current_thread() is not threading.main_thread():
        return {}
    installed: dict[int, Any] = {}

    def cancel(signum: int, _frame: Any) -> None:
        raise _ChildCancellation(f"received signal {signum}")

    candidates = [signal.SIGTERM]
    sigbreak = getattr(signal, "SIGBREAK", None)
    if isinstance(sigbreak, int):
        candidates.append(sigbreak)
    for candidate in candidates:
        try:
            previous = signal.getsignal(candidate)
            signal.signal(candidate, cancel)
        except (OSError, RuntimeError, ValueError):
            continue
        installed[int(candidate)] = previous
    return installed


def _restore_child_cancel_handlers(installed: Mapping[int, Any]) -> None:
    for candidate, previous in installed.items():
        try:
            signal.signal(candidate, previous)
        except (OSError, RuntimeError, ValueError):
            pass


def _wait_for_child(process: subprocess.Popen[bytes], timeout: float) -> bool:
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    except BaseException:
        return process.poll() is not None
    return True


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    platform_name: str | None = None,
) -> None:
    """Best-effort termination of the isolated child and all descendants."""

    platform_value = sys.platform if platform_name is None else platform_name
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 1:
        return

    if platform_value.startswith("win"):
        if process.poll() is None:
            ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
            if ctrl_break is not None:
                try:
                    process.send_signal(ctrl_break)
                except BaseException:
                    pass
                _wait_for_child(process, 5)
        # taskkill /T addresses descendants that ignored CTRL_BREAK.  Arguments
        # stay in a list and shell=False so spaces or Unicode never need quoting.
        try:
            subprocess.run(
                ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                check=False,
                timeout=15,
            )
        except BaseException:
            pass
        if process.poll() is None:
            try:
                process.kill()
            except BaseException:
                pass
            _wait_for_child(process, 5)
        return

    try:
        current_group = os.getpgrp()
    except (AttributeError, OSError):
        current_group = None
    if pid == current_group:
        # This should be unreachable because start_new_session=True is required;
        # never signal Modly's own process group if isolation failed.
        try:
            process.terminate()
        except BaseException:
            pass
        if not _wait_for_child(process, 5):
            try:
                process.kill()
            except BaseException:
                pass
        return

    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    _wait_for_child(process, 10)
    try:
        os.killpg(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        group_alive = False
    else:
        group_alive = True
    if group_alive:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if process.poll() is None:
        _wait_for_child(process, 5)


def _run_upstream(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    emitter: ProtocolEmitter,
    replacements: Mapping[str, str],
    renderer_applicable: bool,
) -> str:
    process: subprocess.Popen[bytes] | None = None
    capture: threading.Thread | None = None
    capture_result: dict[str, Any] = {}
    cancel_handlers: dict[int, Any] = {}
    try:
        cancel_handlers = _install_child_cancel_handlers()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            **_process_group_spawn_options(),
        )
        if process.stdout is None:
            raise OSError("child output pipe was not created")
        capture = threading.Thread(
            target=_capture_bounded_output,
            args=(process.stdout, log_path, capture_result),
            name="lato2-log-capture",
            daemon=True,
        )
        capture.start()
        started = time.monotonic()
        next_update = started + 30.0
        while process.poll() is None:
            time.sleep(0.5)
            now = time.monotonic()
            if now >= next_update:
                minutes = max(1, int((now - started) // 60) + 1)
                emitter.progress(min(82, 34 + minutes * 2), "LATO.2 inference is still running")
                next_update = now + 30.0
        capture.join(timeout=30)
        if capture.is_alive() or capture_result.get("error") is not None:
            raise OSError("child output could not be captured safely")
        captured_identity = capture_result.get("log_identity")
        log_info = log_path.lstat()
        if (
            not isinstance(captured_identity, tuple)
            or len(captured_identity) != 4
            or _is_alias(log_info)
            or not stat.S_ISREG(log_info.st_mode)
            or getattr(log_info, "st_nlink", 1) != 1
            or (
                log_info.st_dev,
                log_info.st_ino,
                log_info.st_size,
                log_info.st_mtime_ns,
            )
            != captured_identity
        ):
            raise OSError("child log identity changed")
        if process.returncode != 0:
            _log_tail(log_path, emitter, replacements)
            _sanitize_persistent_log(log_path, replacements)
            raise ProcessFailure("INFERENCE_FAILED")
        renderer_fallback = capture_result.get("renderer_fallback") is True
        renderer_explicit = (
            renderer_applicable
            and env.get("LATO2_RENDERER", "auto").strip().casefold() == "software"
        )
        renderer = (
            "no-render"
            if not renderer_applicable
            else "software" if renderer_fallback or renderer_explicit else "open3d"
        )
        if renderer_fallback:
            emitter.log(
                "LATO.2 portable backend used the software conditioning renderer because Open3D was unavailable."
            )
        elif renderer_explicit:
            emitter.log(
                "LATO.2 used the explicitly selected software conditioning renderer."
            )
        _record_effective_renderer(log_path, renderer)
        _sanitize_persistent_log(log_path, replacements)
        return renderer
    except BaseException as exc:
        if process is not None:
            _terminate_process_group(process)
        if capture is not None:
            capture.join(timeout=5)
        if isinstance(exc, ProcessFailure):
            raise
        raise ProcessFailure("INFERENCE_FAILED") from exc
    finally:
        _restore_child_cancel_handlers(cancel_handlers)


def _upstream_result(node_id: str, output_dir: Path) -> tuple[Path, bool]:
    choices: dict[str, tuple[tuple[str, bool], ...]] = {
        "lato2-e2e": (("input_pred.obj", False), ("input_pred.ply", True)),
        "lato2-vflow": (("input_pred.ply", True),),
        "lato2-vvae": (("input_recon.ply", True),),
        "lato2-tflow": (("input_pred.obj", False), ("input_pred.ply", True)),
    }
    for name, points in choices[node_id]:
        candidate = output_dir / name
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ProcessFailure("OUTPUT_INVALID") from exc
        if (
            _is_alias(info)
            or not stat.S_ISREG(info.st_mode)
            or getattr(info, "st_nlink", 1) != 1
            or info.st_size <= 0
        ):
            raise ProcessFailure("OUTPUT_INVALID")
        return candidate, points
    raise ProcessFailure("OUTPUT_MISSING")


def _convert_to_glb(source: Path, destination: Path, *, points: bool) -> None:
    try:
        import numpy as np
        import trimesh

        loaded = trimesh.load(str(source), process=False)
        if isinstance(loaded, trimesh.Scene):
            geometry = loaded.dump(concatenate=True)
        else:
            geometry = loaded
        vertices = np.asarray(geometry.vertices)
        if vertices.ndim != 2 or vertices.shape[0] < 1 or vertices.shape[1] != 3:
            raise ValueError("generated geometry has no vertices")
        if points:
            colors = getattr(geometry, "colors", None)
            point_args: dict[str, Any] = {"vertices": vertices}
            if colors is not None:
                color_array = np.asarray(colors)
                if (
                    color_array.ndim == 2
                    and color_array.shape[0] == vertices.shape[0]
                    and color_array.shape[1] in {3, 4}
                ):
                    point_args["colors"] = color_array
            exportable = trimesh.points.PointCloud(**point_args)
        else:
            faces = np.asarray(geometry.faces)
            if faces.ndim != 2 or faces.shape[0] < 1 or faces.shape[1] != 3:
                raise ValueError("generated mesh has no triangle faces")
            exportable = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        encoded = trimesh.Scene(exportable).export(file_type="glb")
        if not isinstance(encoded, (bytes, bytearray)) or len(encoded) < 20 or encoded[:4] != b"glTF":
            raise ValueError("invalid GLB payload")
        with destination.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException as exc:
        destination.unlink(missing_ok=True)
        raise ProcessFailure("OUTPUT_CONVERSION") from exc


def _validate_result(path: Path, run_dir: Path, workspace: Path) -> Path:
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
        run_root = run_dir.resolve(strict=True)
        workspace_root = workspace.resolve(strict=True)
        with resolved.open("rb") as handle:
            magic = handle.read(4)
    except OSError as exc:
        raise ProcessFailure("OUTPUT_INVALID") from exc
    if (
        _is_alias(info)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size < 20
        or magic != b"glTF"
        or resolved.parent != run_root
        or workspace_root not in resolved.parents
    ):
        raise ProcessFailure("OUTPUT_INVALID")
    return resolved


def _publish_failure_diagnostic(
    *,
    source_log: Path,
    output_parent: Path,
    request: ValidatedRequest,
    stamp: str,
    token: str,
    replacements: Mapping[str, str],
    failure_code: str,
) -> bool:
    """Best-effort atomic publication of only sanitized failure metadata/logs."""

    destination = output_parent / f"failed-{request.node_id}-{stamp}-{token}"
    part = output_parent / f".{destination.name}.part-{uuid.uuid4().hex}"
    published = False
    try:
        parent_info = output_parent.lstat()
        if _is_alias(parent_info) or not stat.S_ISDIR(parent_info.st_mode):
            return False
        if output_parent.resolve(strict=True) != output_parent:
            return False
        try:
            destination.lstat()
        except FileNotFoundError:
            pass
        else:
            return False

        _sanitize_persistent_log(source_log, replacements)
        log_info = source_log.lstat()
        maximum = MAX_LOG_BYTES + MAX_LOG_OVERHEAD_BYTES
        if (
            _is_alias(log_info)
            or not stat.S_ISREG(log_info.st_mode)
            or log_info.st_size > maximum
        ):
            return False

        part.mkdir(mode=0o700)
        diagnostic_log = part / "upstream.log"
        with source_log.open("rb") as source_handle:
            opened = os.fstat(source_handle.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != log_info.st_dev
                or opened.st_ino != log_info.st_ino
                or opened.st_size != log_info.st_size
            ):
                return False
            with diagnostic_log.open("xb") as destination_handle:
                copied = 0
                while block := source_handle.read(64 * 1024):
                    copied += len(block)
                    if copied > maximum:
                        return False
                    destination_handle.write(block)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
            final_source = os.fstat(source_handle.fileno())
            if (
                copied != opened.st_size
                or final_source.st_dev != opened.st_dev
                or final_source.st_ino != opened.st_ino
                or final_source.st_size != opened.st_size
                or final_source.st_mtime_ns != opened.st_mtime_ns
            ):
                return False

        metadata = {
            "schema": "modly.lato2.failure.v1",
            "nodeId": request.node_id,
            "backend": request.backend,
            "precision": request.precision,
            "errorCode": failure_code,
            "stage": ERRORS.get(failure_code, ERRORS["UNEXPECTED"])[0],
        }
        encoded = (json.dumps(metadata, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        if len(encoded) > MAX_JSON_BYTES:
            return False
        with (part / "run-failure.json").open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

        for name in ("upstream.log", "run-failure.json"):
            info = (part / name).lstat()
            if _is_alias(info) or not stat.S_ISREG(info.st_mode):
                return False
        os.rename(part, destination)
        published = True
        return True
    except BaseException:
        return False
    finally:
        if not published:
            shutil.rmtree(part, ignore_errors=True)


def handle_request(payload: Mapping[str, Any], emitter: ProtocolEmitter) -> Path:
    emitter.progress(5, "Validating the LATO.2 request")
    request = validate_request(payload)
    emitter.progress(8, "Authenticating pinned LATO.2 sources and model assets")
    _authenticate_request(request)
    emitter.progress(15, "Authenticated pinned LATO.2 sources and model assets")
    temp_run: Path | None = None
    staging: Path | None = None
    output_parent: Path | None = None
    diagnostic_replacements: dict[str, str] = {}
    stamp = ""
    token = ""
    try:
        temp_run, normalized_input = _prepare_input(request.input_path, request.temp_dir)
        emitter.progress(25, "Prepared one self-contained geometry input")
        output_parent = _owned_output_parent(request.workspace_dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        token = uuid.uuid4().hex[:12]
        final_dir = output_parent / f"{request.node_id}-{stamp}-{token}"
        staging = output_parent / f".{final_dir.name}.part-{uuid.uuid4().hex}"
        staging.mkdir()
        command = _command(request, normalized_input.parent, staging)
        source_root = Path(command[5]).parent.parent
        attention = (
            "sdpa" if request.backend == "portable" else request.state.attention_backend
        )
        emitter.log(
            f"Running the {request.backend} LATO.2 backend with {request.precision} "
            f"precision policy and {attention} attention"
        )
        if request.backend == "portable" and request.node_id in {"lato2-e2e", "lato2-vflow"}:
            emitter.log(
                "Portable conditioning renderer policy: Open3D first, with a reported software fallback if unavailable."
            )
        emitter.progress(34, "Running pinned LATO.2 inference")
        diagnostic_replacements = {
            str(request.state.revision_root): "[model snapshot]",
            str(temp_run): "[temporary input]",
            str(request.workspace_dir): "[workspace]",
            str(ROOT): "[extension]",
        }
        renderer = _run_upstream(
            command,
            cwd=source_root,
            env=_runtime_environment(request),
            log_path=staging / "upstream.log",
            emitter=emitter,
            replacements=diagnostic_replacements,
            renderer_applicable=request.node_id in DINO_NODE_IDS,
        )
        emitter.progress(86, "Validating upstream artifacts")
        generated, points = _upstream_result(request.node_id, staging)
        if points and request.node_id in {"lato2-e2e", "lato2-tflow"}:
            emitter.log(
                "LATO.2 decoded no topology faces; returning the upstream point-cloud fallback."
            )
        emitter.progress(92, "Creating the Modly-compatible GLB result")
        _convert_to_glb(generated, staging / "result.glb", points=points)
        metadata = {
            "schema": "modly.lato2.run.v1",
            "nodeId": request.node_id,
            "backend": request.backend,
            "precision": request.precision,
            "attention": attention,
            "renderer": renderer,
            "resultKind": "points" if points else "mesh",
            "parameters": request.params,
        }
        if request.node_id in {"lato2-e2e", "lato2-tflow"}:
            metadata["topologyDecoded"] = not points
        with (staging / "run.json").open("xb") as handle:
            encoded = (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode(
                "utf-8"
            )
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _validate_result(staging / "result.glb", staging, request.workspace_dir)
        try:
            os.replace(staging, final_dir)
        except OSError as exc:
            raise ProcessFailure("OUTPUT_INVALID") from exc
        staging = None
        output = _validate_result(final_dir / "result.glb", final_dir, request.workspace_dir)
        emitter.progress(100, "LATO.2 processing complete")
        return output
    except ProcessFailure as failure:
        if (
            failure.code
            in {
                "INFERENCE_FAILED",
                "OUTPUT_MISSING",
                "OUTPUT_CONVERSION",
                "OUTPUT_INVALID",
            }
            and staging is not None
            and output_parent is not None
            and stamp
            and token
        ):
            try:
                published = _publish_failure_diagnostic(
                    source_log=staging / "upstream.log",
                    output_parent=output_parent,
                    request=request,
                    stamp=stamp,
                    token=token,
                    replacements=diagnostic_replacements,
                    failure_code=failure.code,
                )
            except BaseException:
                published = False
            if published:
                try:
                    emitter.log(
                        "A sanitized failure diagnostic was saved under Workflows/LATO2."
                    )
                except BaseException:
                    pass
        raise
    finally:
        if temp_run is not None:
            shutil.rmtree(temp_run, ignore_errors=True)
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


Handler = Callable[[Mapping[str, Any], ProtocolEmitter], Path]


def run_protocol(
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    handler: Handler = handle_request,
) -> int:
    emitter = ProtocolEmitter(output_stream)
    try:
        payload = _read_one_payload(input_stream)
        if handler is handle_request:
            with _setup_read_lock():
                output = handler(payload, emitter)
        else:
            output = handler(payload, emitter)
        emitter.done(output)
        return 0
    except BaseException as exc:
        if emitter.terminal_attempted or emitter.channel_failed:
            return 1
        failure = exc if type(exc) is ProcessFailure else ProcessFailure("UNEXPECTED")
        try:
            emitter.error(failure.public_message())
        except BaseException:
            return 1
        return 1
