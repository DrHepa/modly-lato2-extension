"""Pinned, resumable and offline-ready LATO.2 snapshot provisioning."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
from typing import Callable, Iterable
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid
from zipfile import BadZipFile, ZipFile, ZipInfo

from .constants import (
    ASSETS,
    DINO_CHECKPOINT_SPEC,
    DINO_REPO,
    DINO_SOURCE_REVISION,
    EXTENSION_ID,
    EXTENSION_VERSION,
    LATO_CHECKPOINT_SPECS,
    LATO_MODEL_REPO,
    LATO_MODEL_REVISION,
    LATO_REPO,
    LATO_SOURCE_REVISION,
    READY_MARKER_FILENAME,
    READY_SCHEMA_VERSION,
    REVISION_ID,
    SOURCE_ARCHIVES,
    SOURCE_ARCHIVE_ASSETS,
    AssetSpec,
    SourceArchiveSpec,
)
from .paths import (
    PathContractError,
    safe_snapshot_directory,
    safe_snapshot_file,
)


LogFunction = Callable[[str], None]
OpenFunction = Callable[..., object]
CHUNK_SIZE = 1024 * 1024
MAX_MARKER_BYTES = 64 * 1024
MAX_ZIP_ENTRIES = 4096
MAX_ZIP_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ZIP_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
WINDOWS_REPARSE_ATTRIBUTE = 0x400


class AssetError(RuntimeError):
    """A snapshot-provisioning failure with a stable diagnostic code."""

    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code}: {public_message}")


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    before = path.lstat()
    if (
        _is_alias(before)
        or not stat.S_ISREG(before.st_mode)
        or getattr(before, "st_nlink", 1) != 1
    ):
        raise OSError("asset is not an owned regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or getattr(opened, "st_nlink", 1) != 1
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise OSError("asset identity changed before hashing")
        while block := handle.read(chunk_size):
            digest.update(block)
        final_open = os.fstat(handle.fileno())
    after = path.lstat()
    identities = {
        (
            item.st_dev,
            item.st_ino,
            item.st_size,
            getattr(item, "st_mtime_ns", None),
            getattr(item, "st_nlink", 1),
        )
        for item in (before, opened, final_open, after)
    }
    if (
        len(identities) != 1
        or _is_alias(after)
        or not stat.S_ISREG(after.st_mode)
        or getattr(after, "st_nlink", 1) != 1
    ):
        raise OSError("asset changed while hashing")
    return digest.hexdigest()


def verify_asset(path: Path, spec: AssetSpec) -> tuple[bool, str]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False, "file is missing"
    except OSError as exc:
        return False, f"metadata is unavailable ({exc})"
    if (
        _is_alias(info)
        or not stat.S_ISREG(info.st_mode)
        or getattr(info, "st_nlink", 1) != 1
    ):
        return False, "path is not a regular local file"
    if info.st_size != spec.size:
        return False, f"size is {info.st_size}; expected {spec.size}"
    try:
        digest = sha256_file(path)
    except OSError as exc:
        return False, f"file could not be hashed ({exc})"
    if digest != spec.sha256:
        return False, "SHA-256 does not match the pinned asset"
    return True, "valid"


def inventory_payload() -> list[dict[str, object]]:
    return [
        {
            "path": spec.relative_path,
            "role": spec.role,
            "size": spec.size,
            "sha256": spec.sha256,
        }
        for spec in ASSETS
    ]


def ready_payload() -> dict[str, object]:
    return {
        "schema_version": READY_SCHEMA_VERSION,
        "extension_id": EXTENSION_ID,
        "extension_version": EXTENSION_VERSION,
        "revision_id": REVISION_ID,
        "lato": {
            "source_repo": LATO_REPO,
            "source_revision": LATO_SOURCE_REVISION,
            "model_repo": LATO_MODEL_REPO,
            "model_revision": LATO_MODEL_REVISION,
        },
        "dinov2": {
            "source_repo": DINO_REPO,
            "source_revision": DINO_SOURCE_REVISION,
            "checkpoint_sha256": DINO_CHECKPOINT_SPEC.sha256,
        },
        "inventory": inventory_payload(),
    }


def _is_alias(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )


def _atomic_json(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    if len(encoded) > MAX_MARKER_BYTES:
        raise AssetError("ASSET_MARKER_TOO_LARGE", "the readiness marker exceeds its limit")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise AssetError("ASSET_MARKER_WRITE_FAILED", "the readiness marker could not be written") from exc


def _read_ready_marker(snapshot_dir: Path) -> tuple[bool, str]:
    marker = snapshot_dir / READY_MARKER_FILENAME
    try:
        info = marker.lstat()
    except FileNotFoundError:
        return False, "readiness marker is missing"
    except OSError as exc:
        return False, f"readiness marker metadata failed ({exc})"
    if _is_alias(info) or not stat.S_ISREG(info.st_mode) or info.st_size > MAX_MARKER_BYTES:
        return False, "readiness marker is unsafe"
    try:
        parsed = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"readiness marker is invalid ({exc})"
    if parsed != ready_payload():
        return False, "readiness marker does not match this immutable revision"
    return True, "valid"


def _response_status(response: object) -> int:
    status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    getter = getattr(response, "getcode", None)
    value = getter() if callable(getter) else None
    if isinstance(value, int):
        return value
    raise AssetError("ASSET_HTTP_STATUS_MISSING", "the response has no HTTP status")


def _header(response: object, name: str) -> str:
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    return str(getter(name, "")) if callable(getter) else ""


def _validate_response(response: object, spec: AssetSpec, existing_size: int) -> tuple[str, int]:
    status = _response_status(response)
    if existing_size:
        if status == 200:
            mode, expected_body = "wb", spec.size
        elif status == 206:
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", _header(response, "Content-Range").strip()
            )
            if not match:
                raise AssetError("ASSET_RANGE_INVALID", "resume returned an invalid Content-Range")
            start, end, total = (int(value) for value in match.groups())
            if start != existing_size or total != spec.size or end < start or end >= total:
                raise AssetError("ASSET_RANGE_INVALID", "resume range does not match the pinned file")
            mode, expected_body = "ab", spec.size - existing_size
        else:
            raise AssetError("ASSET_HTTP_STATUS", f"download returned HTTP {status}")
    else:
        if status != 200:
            raise AssetError("ASSET_HTTP_STATUS", f"full download returned HTTP {status}")
        mode, expected_body = "wb", spec.size
    length = _header(response, "Content-Length").strip()
    if length:
        try:
            declared = int(length)
        except ValueError as exc:
            raise AssetError("ASSET_LENGTH_INVALID", "download returned an invalid length") from exc
        if declared != expected_body:
            raise AssetError("ASSET_LENGTH_INVALID", "download length does not match the pinned file")
    return mode, 0 if mode == "wb" else existing_size


def _stream_download(
    spec: AssetSpec,
    part_path: Path,
    *,
    opener: OpenFunction,
    log: LogFunction,
    timeout: float,
) -> None:
    try:
        initial_info = part_path.lstat()
    except FileNotFoundError:
        initial_info = None
    except OSError as exc:
        raise AssetError(
            "ASSET_PART_INVALID", "a partial path cannot be inspected"
        ) from exc
    if initial_info is not None and (
        _is_alias(initial_info)
        or not stat.S_ISREG(initial_info.st_mode)
        or getattr(initial_info, "st_nlink", 1) != 1
    ):
        raise AssetError("ASSET_PART_INVALID", "a partial path is unsafe")
    existing_size = initial_info.st_size if initial_info is not None else 0
    if existing_size >= spec.size:
        part_path.unlink(missing_ok=True)
        initial_info = None
        existing_size = 0
    headers = {"User-Agent": f"Modly-LATO2/{EXTENSION_VERSION}"}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"
        log(f"Resuming {spec.relative_path} at {existing_size} bytes")
    # All pinned assets are public. Never attach Hub tokens or Authorization:
    # urllib may follow redirects to a separate CDN host.
    request = Request(spec.url, headers=headers)
    with opener(request, timeout=timeout) as response:
        mode, downloaded = _validate_response(response, spec, existing_size)
        if existing_size and mode == "wb":
            log(f"Restarting {spec.relative_path}; server ignored Range")
        last_bucket = -1
        # Never open an existing pathname with ``wb``: truncation happens
        # before Python can compare the descriptor with lstat metadata.  A
        # verified owned partial is unlinked first and recreated exclusively;
        # a concurrent replacement then makes ``xb`` fail without modifying it.
        if mode == "wb":
            try:
                current = part_path.lstat()
            except FileNotFoundError:
                current = None
            if current is not None:
                if (
                    _is_alias(current)
                    or not stat.S_ISREG(current.st_mode)
                    or getattr(current, "st_nlink", 1) != 1
                    or initial_info is None
                    or current.st_dev != initial_info.st_dev
                    or current.st_ino != initial_info.st_ino
                ):
                    raise AssetError("ASSET_PART_INVALID", "a partial path is unsafe")
                part_path.unlink()
            file_mode = "xb"
        else:
            file_mode = "r+b"

        with part_path.open(file_mode) as handle:
            opened = os.fstat(handle.fileno())
            try:
                path_info = part_path.lstat()
            except OSError as exc:
                raise AssetError(
                    "ASSET_PART_INVALID", "a partial path cannot be verified"
                ) from exc
            if (
                _is_alias(path_info)
                or not stat.S_ISREG(path_info.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or getattr(path_info, "st_nlink", 1) != 1
                or getattr(opened, "st_nlink", 1) != 1
                or path_info.st_dev != opened.st_dev
                or path_info.st_ino != opened.st_ino
                or opened.st_size != (0 if mode == "wb" else existing_size)
                or (
                    mode == "ab"
                    and (
                        initial_info is None
                        or initial_info.st_dev != opened.st_dev
                        or initial_info.st_ino != opened.st_ino
                    )
                )
            ):
                raise AssetError("ASSET_PART_INVALID", "a partial path is unsafe")
            if mode == "ab":
                handle.seek(0, os.SEEK_END)
            while True:
                block = response.read(CHUNK_SIZE)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                if downloaded > spec.size:
                    raise AssetError("ASSET_SIZE_EXCEEDED", f"{spec.relative_path} exceeded its size")
                bucket = int(downloaded * 20 / spec.size)
                if bucket != last_bucket:
                    log(
                        f"Downloading {spec.relative_path}: "
                        f"{min(100, int(downloaded * 100 / spec.size))}%"
                    )
                    last_bucket = bucket
            handle.flush()
            os.fsync(handle.fileno())
            final = os.fstat(handle.fileno())
            final_path = part_path.lstat()
            if (
                not stat.S_ISREG(final.st_mode)
                or _is_alias(final_path)
                or not stat.S_ISREG(final_path.st_mode)
                or getattr(final, "st_nlink", 1) != 1
                or getattr(final_path, "st_nlink", 1) != 1
                or final.st_dev != final_path.st_dev
                or final.st_ino != final_path.st_ino
                or final.st_size != downloaded
            ):
                raise AssetError("ASSET_PART_INVALID", "a partial path changed during download")
    if downloaded != spec.size:
        raise AssetError(
            "ASSET_SIZE_INCOMPLETE",
            f"{spec.relative_path} stopped at {downloaded} of {spec.size} bytes",
        )


def ensure_asset(
    snapshot_dir: Path,
    spec: AssetSpec,
    *,
    opener: OpenFunction = urlopen,
    log: LogFunction = print,
    retries: int = 3,
    timeout: float = 90.0,
    retry_delay: float = 0.5,
) -> Path:
    """Verify or resumably download one asset, promoting it with os.replace."""

    if retries < 1:
        raise ValueError("retries must be at least one")
    destination = safe_snapshot_file(snapshot_dir, spec.relative_path, create_parent=True)
    part = destination.with_name(destination.name + ".part")
    valid, _ = verify_asset(destination, spec)
    if valid:
        if part.exists() or part.is_symlink():
            info = part.lstat()
            if (
                _is_alias(info)
                or not stat.S_ISREG(info.st_mode)
                or getattr(info, "st_nlink", 1) != 1
            ):
                raise AssetError("ASSET_PART_INVALID", "a partial path is unsafe")
            part.unlink()
        log(f"Verified {spec.relative_path}; skipped download")
        return destination

    if part.exists() or part.is_symlink():
        info = part.lstat()
        if (
            _is_alias(info)
            or not stat.S_ISREG(info.st_mode)
            or getattr(info, "st_nlink", 1) != 1
        ):
            raise AssetError("ASSET_PART_INVALID", "a partial path is unsafe")
        recovered, _ = verify_asset(part, spec)
        if recovered:
            os.replace(part, destination)
            log(f"Recovered completed {spec.relative_path} partial")
            return destination
        if info.st_size >= spec.size:
            part.unlink()

    last_error: BaseException | None = None
    range_restart_used = False
    for attempt in range(1, retries + 1):
        try:
            try:
                _stream_download(spec, part, opener=opener, log=log, timeout=timeout)
            except HTTPError as exc:
                partial_size = part.stat().st_size if part.is_file() else 0
                if exc.code != 416 or partial_size <= 0 or range_restart_used:
                    raise
                range_restart_used = True
                part.unlink(missing_ok=True)
                log(f"Restarting {spec.relative_path} after rejected Range")
                _stream_download(spec, part, opener=opener, log=log, timeout=timeout)
            part_valid, reason = verify_asset(part, spec)
            if not part_valid:
                if part.is_file() and part.stat().st_size >= spec.size:
                    part.unlink()
                raise AssetError("ASSET_VERIFY_FAILED", f"{spec.relative_path}: {reason}")
            os.replace(part, destination)
            final_valid, reason = verify_asset(destination, spec)
            if not final_valid:
                raise AssetError("ASSET_PROMOTION_FAILED", f"{spec.relative_path}: {reason}")
            log(f"Installed {spec.relative_path}")
            return destination
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            last_error = exc
            log(f"Attempt {attempt}/{retries} failed for {spec.relative_path}: {type(exc).__name__}")
            if attempt < retries:
                time.sleep(retry_delay)
    raise AssetError(
        "ASSET_DOWNLOAD_FAILED",
        f"could not download and verify {spec.relative_path}; check network and storage, then run Repair",
    ) from last_error


def _zip_entries(archive: Path, source: SourceArchiveSpec) -> list[tuple[ZipInfo, PurePosixPath]]:
    try:
        with ZipFile(archive) as bundle:
            infos = bundle.infolist()
    except (OSError, BadZipFile) as exc:
        raise AssetError("ASSET_ARCHIVE_INVALID", "a pinned source archive is unreadable") from exc
    if len(infos) > MAX_ZIP_ENTRIES:
        raise AssetError("ASSET_ARCHIVE_LIMIT", "a source archive contains too many entries")
    total = 0
    result: list[tuple[ZipInfo, PurePosixPath]] = []
    seen: set[str] = set()
    for info in infos:
        if "\\" in info.filename or "\0" in info.filename or ":" in info.filename:
            raise AssetError("ASSET_ARCHIVE_PATH", "a source archive contains an unsafe path")
        path = PurePosixPath(info.filename)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise AssetError("ASSET_ARCHIVE_PATH", "a source archive contains path traversal")
        if not path.parts or path.parts[0] != source.expected_archive_root:
            raise AssetError("ASSET_ARCHIVE_ROOT", "a source archive has an unexpected root")
        if len(path.parts) == 1:
            if not info.is_dir():
                raise AssetError("ASSET_ARCHIVE_ROOT", "a source archive root is not a directory")
            continue
        relative = PurePosixPath(*path.parts[1:])
        mode = (info.external_attr >> 16) & 0xFFFF
        kind = stat.S_IFMT(mode)
        if kind not in {0, stat.S_IFREG, stat.S_IFDIR} or stat.S_ISLNK(mode):
            raise AssetError("ASSET_ARCHIVE_SPECIAL", "a source archive contains a special entry")
        if info.file_size > MAX_ZIP_MEMBER_BYTES:
            raise AssetError("ASSET_ARCHIVE_LIMIT", "a source archive member exceeds its limit")
        total += info.file_size
        if total > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise AssetError("ASSET_ARCHIVE_LIMIT", "a source archive exceeds its extraction limit")
        key = relative.as_posix().rstrip("/")
        if key in seen:
            raise AssetError("ASSET_ARCHIVE_DUPLICATE", "a source archive repeats a path")
        seen.add(key)
        result.append((info, relative))
    if not result:
        raise AssetError("ASSET_ARCHIVE_EMPTY", "a source archive has no payload")
    return result


def _actual_tree(destination: Path) -> tuple[set[str], set[str], str | None]:
    files: set[str] = set()
    directories: set[str] = set()
    try:
        root_info = destination.lstat()
    except FileNotFoundError:
        return files, directories, "source tree is missing"
    except OSError as exc:
        return files, directories, f"source tree metadata failed ({exc})"
    if _is_alias(root_info) or not stat.S_ISDIR(root_info.st_mode):
        return files, directories, "source tree is not a regular directory"
    try:
        for root, dirnames, filenames in os.walk(destination, followlinks=False):
            root_path = Path(root)
            for name in dirnames:
                path = root_path / name
                info = path.lstat()
                if _is_alias(info) or not stat.S_ISDIR(info.st_mode):
                    return files, directories, "source tree contains an alias or special directory"
                directories.add(path.relative_to(destination).as_posix())
            for name in filenames:
                path = root_path / name
                info = path.lstat()
                if (
                    _is_alias(info)
                    or not stat.S_ISREG(info.st_mode)
                    or getattr(info, "st_nlink", 1) != 1
                ):
                    return files, directories, "source tree contains an alias or special file"
                files.add(path.relative_to(destination).as_posix())
    except OSError as exc:
        return files, directories, f"source tree could not be enumerated ({exc})"
    return files, directories, None


def verify_source_tree(
    snapshot_dir: Path,
    source: SourceArchiveSpec,
) -> tuple[bool, str]:
    try:
        archive = safe_snapshot_file(snapshot_dir, source.asset_path, create_parent=False)
        destination = snapshot_dir.joinpath(*PurePosixPath(source.destination).parts)
        entries = _zip_entries(archive, source)
    except (PathContractError, AssetError) as exc:
        return False, str(exc)

    expected_files = {
        relative.as_posix() for info, relative in entries if not info.is_dir()
    }
    expected_directories = {
        relative.as_posix().rstrip("/") for info, relative in entries if info.is_dir()
    }
    expected_directories.update(
        parent.as_posix()
        for _info, relative in entries
        for parent in relative.parents
        if parent != PurePosixPath(".")
    )
    actual_files, actual_directories, error = _actual_tree(destination)
    if error:
        return False, error
    if actual_files != expected_files or actual_directories != expected_directories:
        return False, "source tree inventory does not match its pinned archive"

    try:
        with ZipFile(archive) as bundle:
            for info, relative in entries:
                if info.is_dir():
                    continue
                disk_path = destination.joinpath(*relative.parts)
                if disk_path.stat().st_size != info.file_size:
                    return False, f"{relative.as_posix()}: size mismatch"
                archive_hash = hashlib.sha256()
                disk_hash = hashlib.sha256()
                with bundle.open(info, "r") as packed, disk_path.open("rb") as unpacked:
                    while block := packed.read(CHUNK_SIZE):
                        archive_hash.update(block)
                    while block := unpacked.read(CHUNK_SIZE):
                        disk_hash.update(block)
                if archive_hash.digest() != disk_hash.digest():
                    return False, f"{relative.as_posix()}: content mismatch"
    except (OSError, BadZipFile) as exc:
        return False, f"source tree verification failed ({exc})"
    return True, "valid"


def _validate_removal(path: Path, snapshot_dir: Path) -> None:
    try:
        path.relative_to(snapshot_dir)
        parent = path.parent.resolve(strict=True)
        root = snapshot_dir.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise AssetError("ASSET_REPAIR_CONTAINMENT", "a repair path is unsafe") from exc
    if parent != root and root not in parent.parents:
        raise AssetError("ASSET_REPAIR_CONTAINMENT", "a repair path escapes the snapshot")


def _remove_entry(path: Path, snapshot_dir: Path) -> None:
    _validate_removal(path, snapshot_dir)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise AssetError("ASSET_REPAIR_FAILED", "an owned entry cannot be inspected") from exc
    if _is_alias(info):
        try:
            path.rmdir() if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) else path.unlink()
        except IsADirectoryError:
            path.rmdir()
        except OSError as exc:
            raise AssetError("ASSET_REPAIR_FAILED", "an owned alias cannot be removed") from exc
        return
    if stat.S_ISDIR(info.st_mode):
        try:
            entries = list(os.scandir(path))
        except OSError as exc:
            raise AssetError("ASSET_REPAIR_FAILED", "an owned directory cannot be read") from exc
        for entry in entries:
            _remove_entry(Path(entry.path), snapshot_dir)
        try:
            path.rmdir()
        except OSError as exc:
            raise AssetError("ASSET_REPAIR_FAILED", "an owned directory cannot be removed") from exc
    else:
        try:
            path.unlink()
        except OSError as exc:
            raise AssetError("ASSET_REPAIR_FAILED", "an owned file cannot be removed") from exc


def _extract_source_tree(snapshot_dir: Path, source: SourceArchiveSpec) -> Path:
    archive = safe_snapshot_file(snapshot_dir, source.asset_path, create_parent=False)
    destination_path = PurePosixPath(source.destination)
    parent_relative = PurePosixPath(*destination_path.parts[:-1]).as_posix()
    parent = safe_snapshot_directory(snapshot_dir, parent_relative, create=True)
    destination = parent / destination_path.name
    staging = parent / f".{destination.name}.extract-{uuid.uuid4().hex}"
    backup = parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
    entries = _zip_entries(archive, source)
    try:
        staging.mkdir()
        staging_root = staging.resolve(strict=True)
        with ZipFile(archive) as bundle:
            for info, relative in entries:
                target = staging.joinpath(*relative.parts)
                resolved = target.resolve(strict=False)
                if resolved != staging_root and staging_root not in resolved.parents:
                    raise AssetError("ASSET_ARCHIVE_PATH", "archive extraction would escape staging")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info, "r") as source_handle, target.open("xb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle, length=CHUNK_SIZE)
                    target_handle.flush()
                    os.fsync(target_handle.fileno())
                mode = (info.external_attr >> 16) & 0o777
                if mode:
                    try:
                        target.chmod(mode)
                    except OSError:
                        pass

        if destination.exists() or destination.is_symlink():
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException:
            if backup.exists() or backup.is_symlink():
                os.replace(backup, destination)
            raise
        if backup.exists() or backup.is_symlink():
            _remove_entry(backup, snapshot_dir)
    except (KeyboardInterrupt, SystemExit):
        if staging.exists() or staging.is_symlink():
            _remove_entry(staging, snapshot_dir)
        raise
    except BaseException as exc:
        if staging.exists() or staging.is_symlink():
            _remove_entry(staging, snapshot_dir)
        if isinstance(exc, AssetError):
            raise
        raise AssetError("ASSET_EXTRACT_FAILED", "a pinned source tree could not be installed") from exc
    return destination


def ensure_source_tree(
    snapshot_dir: Path,
    source: SourceArchiveSpec,
    *,
    log: LogFunction = print,
) -> Path:
    valid, _ = verify_source_tree(snapshot_dir, source)
    destination = snapshot_dir.joinpath(*PurePosixPath(source.destination).parts)
    if valid:
        log(f"Verified {source.destination}; skipped extraction")
        return destination
    installed = _extract_source_tree(snapshot_dir, source)
    valid, reason = verify_source_tree(snapshot_dir, source)
    if not valid:
        raise AssetError("ASSET_SOURCE_VERIFY_FAILED", f"{source.destination}: {reason}")
    log(f"Installed pinned source tree {source.destination}")
    return installed


def _directory_children(path: Path) -> tuple[dict[str, bool], str | None]:
    try:
        info = path.lstat()
        if _is_alias(info) or not stat.S_ISDIR(info.st_mode):
            return {}, "is not a regular directory"
        entries = list(os.scandir(path))
    except FileNotFoundError:
        return {}, "is missing"
    except OSError as exc:
        return {}, f"cannot be inspected ({exc})"
    result: dict[str, bool] = {}
    for entry in entries:
        entry_info = Path(entry.path).lstat()
        if _is_alias(entry_info):
            return {}, f"contains alias {entry.name}"
        if stat.S_ISDIR(entry_info.st_mode):
            result[entry.name] = True
        elif stat.S_ISREG(entry_info.st_mode):
            result[entry.name] = False
        else:
            return {}, f"contains special entry {entry.name}"
    return result, None


def _expect_children(
    path: Path,
    expected: dict[str, bool],
    label: str,
    *,
    optional: dict[str, bool] | None = None,
) -> str | None:
    actual, error = _directory_children(path)
    if error:
        return f"{label} {error}"
    optional = optional or {}
    if any(actual.get(name) != kind for name, kind in expected.items()):
        return f"{label} inventory differs from the pinned layout"
    if any(name in actual and actual[name] != kind for name, kind in optional.items()):
        return f"{label} contains a derived path of the wrong type"
    if set(actual) - set(expected) - set(optional):
        return f"{label} inventory differs from the pinned layout"
    return None


def verify_snapshot(snapshot_dir: Path, *, require_ready: bool = True) -> list[str]:
    """Return bounded failures for the exact weights, code and offline DINO cache."""

    failures: list[str] = []
    for spec in ASSETS:
        try:
            path = safe_snapshot_file(snapshot_dir, spec.relative_path, create_parent=False)
        except PathContractError as exc:
            failures.append(f"{spec.relative_path}: {exc}")
            continue
        valid, reason = verify_asset(path, spec)
        if not valid:
            failures.append(f"{spec.relative_path}: {reason}")
    for source in SOURCE_ARCHIVES:
        valid, reason = verify_source_tree(snapshot_dir, source)
        if not valid:
            failures.append(f"{source.destination}: {reason}")

    expected_root = {
        "_archives": True,
        "ckpt": True,
        "source": True,
        "dinov2": True,
    }
    if require_ready:
        expected_root[READY_MARKER_FILENAME] = False
    checks = (
        (
            snapshot_dir,
            expected_root,
            "snapshot root",
            {"runtime-cache": True, "native": True},
        ),
        (
            snapshot_dir / "_archives",
            {Path(spec.relative_path).name: False for spec in SOURCE_ARCHIVE_ASSETS},
            "archive directory",
            {},
        ),
        (
            snapshot_dir / "ckpt",
            {Path(spec.relative_path).name: False for spec in LATO_CHECKPOINT_SPECS},
            "checkpoint directory",
            {},
        ),
        (
            snapshot_dir / "source",
            {"LATO.2": True},
            "source directory",
            {"LATO.2-portable": True},
        ),
        (
            snapshot_dir / "dinov2",
            {"facebookresearch_dinov2_main": True, "checkpoints": True},
            "DINO hub directory",
            {},
        ),
        (
            snapshot_dir / "dinov2" / "checkpoints",
            {Path(DINO_CHECKPOINT_SPEC.relative_path).name: False},
            "DINO checkpoint directory",
            {},
        ),
    )
    for path, expected, label, optional in checks:
        error = _expect_children(path, expected, label, optional=optional)
        if error:
            failures.append(error)
    if require_ready:
        valid, reason = _read_ready_marker(snapshot_dir)
        if not valid:
            failures.append(reason)
    return failures[:32]


def _prune_children(
    directory: Path,
    allowed: Iterable[str],
    snapshot_dir: Path,
) -> None:
    if not directory.exists():
        return
    allowed_names = set(allowed)
    try:
        entries = list(os.scandir(directory))
    except OSError as exc:
        raise AssetError("ASSET_REPAIR_FAILED", "an owned directory cannot be reconciled") from exc
    for entry in entries:
        if entry.name not in allowed_names:
            _remove_entry(Path(entry.path), snapshot_dir)


def _reconcile_fixed_layout(snapshot_dir: Path) -> None:
    marker = snapshot_dir / READY_MARKER_FILENAME
    if marker.exists() or marker.is_symlink():
        _remove_entry(marker, snapshot_dir)
    # An interrupted or older Repair may have left a file/alias where one of
    # the extension's known directories belongs. Remove only those exact owned
    # entries so the normal creation path can rebuild them safely.
    for relative in (
        "_archives",
        "ckpt",
        "source",
        "dinov2",
        "runtime-cache",
        "native",
        "source/LATO.2-portable",
        "dinov2/checkpoints",
    ):
        path = snapshot_dir.joinpath(*PurePosixPath(relative).parts)
        if not (path.exists() or path.is_symlink()):
            continue
        info = path.lstat()
        if _is_alias(info) or not stat.S_ISDIR(info.st_mode):
            _remove_entry(path, snapshot_dir)

    _prune_children(
        snapshot_dir,
        {"_archives", "ckpt", "source", "dinov2", "runtime-cache", "native"},
        snapshot_dir,
    )
    _prune_children(
        snapshot_dir / "_archives",
        {
            Path(spec.relative_path).name
            for spec in SOURCE_ARCHIVE_ASSETS
        }
        | {
            Path(spec.relative_path).name + ".part"
            for spec in SOURCE_ARCHIVE_ASSETS
        },
        snapshot_dir,
    )
    _prune_children(
        snapshot_dir / "ckpt",
        {Path(spec.relative_path).name for spec in LATO_CHECKPOINT_SPECS}
        | {Path(spec.relative_path).name + ".part" for spec in LATO_CHECKPOINT_SPECS},
        snapshot_dir,
    )
    _prune_children(
        snapshot_dir / "source", {"LATO.2", "LATO.2-portable"}, snapshot_dir
    )
    _prune_children(
        snapshot_dir / "dinov2",
        {"facebookresearch_dinov2_main", "checkpoints"},
        snapshot_dir,
    )
    _prune_children(
        snapshot_dir / "dinov2" / "checkpoints",
        {
            Path(DINO_CHECKPOINT_SPEC.relative_path).name,
            Path(DINO_CHECKPOINT_SPEC.relative_path).name + ".part",
        },
        snapshot_dir,
    )


def ensure_snapshot(
    snapshot_dir: Path,
    *,
    opener: OpenFunction = urlopen,
    log: LogFunction = print,
) -> Path:
    """Provision the exact full upstream snapshot and mark it ready last."""

    try:
        root_info = snapshot_dir.lstat()
    except OSError as exc:
        raise AssetError("ASSET_ROOT_MISSING", "the owned revision directory is unavailable") from exc
    if _is_alias(root_info) or not stat.S_ISDIR(root_info.st_mode):
        raise AssetError("ASSET_ROOT_INVALID", "the owned revision path is not a regular directory")
    current_failures = verify_snapshot(snapshot_dir, require_ready=True)
    if not current_failures:
        log("Pinned LATO.2 snapshot is already ready; no network access required")
        return snapshot_dir.resolve(strict=True)

    _reconcile_fixed_layout(snapshot_dir)
    for spec in ASSETS:
        ensure_asset(snapshot_dir, spec, opener=opener, log=log)
    for source in SOURCE_ARCHIVES:
        ensure_source_tree(snapshot_dir, source, log=log)

    # Successful downloads leave no .part files. Reconcile only known fixed
    # areas before validating the exact final inventory.
    _reconcile_fixed_layout(snapshot_dir)
    failures = verify_snapshot(snapshot_dir, require_ready=False)
    if failures:
        raise AssetError(
            "ASSET_INVENTORY_INVALID",
            "the pinned snapshot is incomplete: " + "; ".join(failures[:8]),
        )
    marker = safe_snapshot_file(snapshot_dir, READY_MARKER_FILENAME, create_parent=False)
    _atomic_json(marker, ready_payload())
    final_failures = verify_snapshot(snapshot_dir, require_ready=True)
    if final_failures:
        marker.unlink(missing_ok=True)
        raise AssetError(
            "ASSET_READINESS_INVALID",
            "snapshot readiness verification failed: " + "; ".join(final_failures[:8]),
        )
    log("Pinned LATO.2 weights, source and offline DINOv2 cache are ready")
    return snapshot_dir.resolve(strict=True)
