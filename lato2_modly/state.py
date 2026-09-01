"""Atomic path configuration shared by setup and the process runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from typing import Mapping
import uuid

from .constants import (
    EXTENSION_ID,
    REVISION_ID,
    RUNTIME_CONFIG_FILENAME,
    RUNTIME_CONFIG_SCHEMA_VERSION,
)
from .integrity import TreeIntegrityError, read_owned_regular_bytes


MAX_RUNTIME_CONFIG_BYTES = 64 * 1024
WINDOWS_REPARSE_ATTRIBUTE = 0x400
RESERVED_FIELDS = frozenset({"schema_version", "models_dir", "revision_dir"})


class StateError(RuntimeError):
    """A runtime-state failure with a stable diagnostic code."""

    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code}: {public_message}")


@dataclass(frozen=True)
class RuntimeConfig:
    models_dir: Path
    revision_dir: Path
    payload: Mapping[str, object]


def runtime_config_path(extension_dir: Path) -> Path:
    return extension_dir / RUNTIME_CONFIG_FILENAME


def _is_alias(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )


def _forbidden_key(key: object) -> bool:
    lowered = str(key).casefold()
    return any(fragment in lowered for fragment in ("token", "secret", "password", "authorization"))


def _contains_secret(value: object, key: object = "") -> bool:
    if _forbidden_key(key):
        return True
    if isinstance(value, dict):
        return any(_contains_secret(item, name) for name, item in value.items())
    if isinstance(value, list):
        return any(_contains_secret(item) for item in value)
    return False


def runtime_config_payload(
    models_dir: Path,
    revision_dir: Path,
    *,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    try:
        models = models_dir.resolve(strict=True)
        revision = revision_dir.resolve(strict=True)
    except OSError as exc:
        raise StateError("STATE_PATH_MISSING", "configured model paths must already exist") from exc
    if not models.is_dir() or not revision.is_dir() or models not in revision.parents:
        raise StateError("STATE_PATH_INVALID", "the revision directory must be below models_dir")
    expected_suffix = Path(EXTENSION_ID) / "lato2" / "revisions" / REVISION_ID
    try:
        relative = revision.relative_to(models)
    except ValueError as exc:
        raise StateError("STATE_PATH_INVALID", "the revision directory is outside models_dir") from exc
    if relative != expected_suffix:
        raise StateError("STATE_REVISION_INVALID", "the revision directory does not match this release")
    additions = dict(extra or {})
    conflict = RESERVED_FIELDS.intersection(additions)
    if conflict:
        raise StateError("STATE_FIELD_CONFLICT", "extra runtime state overrides a reserved field")
    if _contains_secret(additions):
        raise StateError("STATE_SECRET_REJECTED", "runtime state must not contain credentials")
    payload: dict[str, object] = {
        "schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
        "models_dir": str(models),
        "revision_dir": str(revision),
    }
    payload.update(additions)
    try:
        json.dumps(payload, ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise StateError("STATE_JSON_INVALID", "runtime state is not JSON serializable") from exc
    return payload


def write_runtime_config(
    extension_dir: Path,
    models_dir: Path,
    revision_dir: Path,
    *,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write runtime_config.json atomically after setup has fully succeeded."""

    try:
        extension = extension_dir.resolve(strict=True)
        extension_info = extension.lstat()
    except OSError as exc:
        raise StateError("STATE_EXTENSION_MISSING", "the extension directory is unavailable") from exc
    if _is_alias(extension_info) or not stat.S_ISDIR(extension_info.st_mode):
        raise StateError("STATE_EXTENSION_INVALID", "the extension directory is unsafe")
    payload = runtime_config_payload(models_dir, revision_dir, extra=extra)
    encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    if len(encoded) > MAX_RUNTIME_CONFIG_BYTES:
        raise StateError("STATE_TOO_LARGE", "runtime state exceeds its size limit")
    destination = extension / RUNTIME_CONFIG_FILENAME
    if destination.exists() or destination.is_symlink():
        info = destination.lstat()
        if (
            _is_alias(info)
            or not stat.S_ISREG(info.st_mode)
            or getattr(info, "st_nlink", 1) != 1
        ):
            raise StateError("STATE_FILE_INVALID", "existing runtime state is unsafe")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise StateError("STATE_WRITE_FAILED", "runtime state could not be written") from exc
    return destination


def read_runtime_config(extension_dir: Path) -> RuntimeConfig:
    path = runtime_config_path(extension_dir)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise StateError("STATE_MISSING", "runtime state is missing; run Repair") from exc
    except OSError as exc:
        raise StateError("STATE_READ_FAILED", "runtime state cannot be inspected") from exc
    if (
        _is_alias(info)
        or not stat.S_ISREG(info.st_mode)
        or getattr(info, "st_nlink", 1) != 1
        or info.st_size > MAX_RUNTIME_CONFIG_BYTES
    ):
        raise StateError("STATE_FILE_INVALID", "runtime state is unsafe; run Repair")
    try:
        encoded = read_owned_regular_bytes(path, max_bytes=MAX_RUNTIME_CONFIG_BYTES)
        parsed = json.loads(encoded.decode("utf-8"))
    except (TreeIntegrityError, UnicodeError, json.JSONDecodeError) as exc:
        raise StateError("STATE_JSON_INVALID", "runtime state is unreadable; run Repair") from exc
    if not isinstance(parsed, dict) or _contains_secret(parsed):
        raise StateError("STATE_JSON_INVALID", "runtime state is invalid; run Repair")
    if parsed.get("schema_version") != RUNTIME_CONFIG_SCHEMA_VERSION:
        raise StateError("STATE_SCHEMA_MISMATCH", "runtime state is stale; run Repair")
    models_raw = parsed.get("models_dir")
    revision_raw = parsed.get("revision_dir")
    if not isinstance(models_raw, str) or not isinstance(revision_raw, str):
        raise StateError("STATE_PATH_INVALID", "runtime state lacks model paths; run Repair")
    models = Path(models_raw)
    revision = Path(revision_raw)
    if not models.is_absolute() or not revision.is_absolute():
        raise StateError("STATE_PATH_INVALID", "runtime model paths must be absolute; run Repair")
    # Reuse the exact writer validation without discarding forward-compatible
    # backend metadata supplied by setup.
    extras = {key: value for key, value in parsed.items() if key not in RESERVED_FIELDS}
    expected = runtime_config_payload(models, revision, extra=extras)
    if parsed != expected:
        raise StateError("STATE_PATH_INVALID", "runtime model paths are stale; run Repair")
    return RuntimeConfig(
        models_dir=Path(str(expected["models_dir"])),
        revision_dir=Path(str(expected["revision_dir"])),
        payload=expected,
    )
