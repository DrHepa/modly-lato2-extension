"""Strict, cross-platform inventory helpers for extension-owned source trees."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Callable, Mapping


WINDOWS_REPARSE_ATTRIBUTE = 0x400


class TreeIntegrityError(RuntimeError):
    """Raised when a tree contains an alias, special entry, or changes while read."""


@dataclass(frozen=True)
class TreeInventory:
    files: Mapping[str, str]
    directories: tuple[str, ...]

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        for relative in self.directories:
            digest.update(b"D\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
        for relative, file_digest in sorted(self.files.items()):
            digest.update(b"F\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(file_digest))
        return digest.hexdigest()


def is_alias(info: os.stat_result) -> bool:
    """Recognize POSIX symlinks and Windows reparse points/junctions."""

    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )


def read_owned_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read one bounded non-alias, single-link file with stable identity."""

    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    try:
        before = path.lstat()
    except OSError as exc:
        raise TreeIntegrityError(f"cannot inspect file: {path}") from exc
    if (
        is_alias(before)
        or not stat.S_ISREG(before.st_mode)
        or getattr(before, "st_nlink", 1) != 1
        or before.st_size > max_bytes
    ):
        raise TreeIntegrityError(f"file is not a bounded owned regular file: {path}")
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or getattr(opened, "st_nlink", 1) != 1
                or opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
                or opened.st_size != before.st_size
            ):
                raise TreeIntegrityError(f"file identity changed before reading: {path}")
            encoded = handle.read(max_bytes + 1)
            final_open = os.fstat(handle.fileno())
        after = path.lstat()
    except TreeIntegrityError:
        raise
    except OSError as exc:
        raise TreeIntegrityError(f"cannot read file: {path}") from exc
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
        len(encoded) > max_bytes
        or len(identities) != 1
        or is_alias(after)
        or not stat.S_ISREG(after.st_mode)
        or getattr(after, "st_nlink", 1) != 1
    ):
        raise TreeIntegrityError(f"file changed while it was read: {path}")
    return encoded


def sha256_regular_file(path: Path) -> str:
    """Hash one regular non-alias file and fail closed if it changes while read."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise TreeIntegrityError(f"cannot inspect file: {path}") from exc
    if (
        is_alias(before)
        or not stat.S_ISREG(before.st_mode)
        or getattr(before, "st_nlink", 1) != 1
    ):
        raise TreeIntegrityError(f"tree entry is not a regular local file: {path}")

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (
                is_alias(opened)
                or not stat.S_ISREG(opened.st_mode)
                or getattr(opened, "st_nlink", 1) != 1
            ):
                raise TreeIntegrityError(f"opened entry is not a regular file: {path}")
            while block := handle.read(1024 * 1024):
                digest.update(block)
        after = path.lstat()
    except TreeIntegrityError:
        raise
    except OSError as exc:
        raise TreeIntegrityError(f"cannot hash file: {path}") from exc

    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        getattr(before, "st_mtime_ns", None),
        getattr(before, "st_nlink", 1),
    )
    identity_opened = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        getattr(opened, "st_mtime_ns", None),
        getattr(opened, "st_nlink", 1),
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        getattr(after, "st_mtime_ns", None),
        getattr(after, "st_nlink", 1),
    )
    if (
        is_alias(after)
        or getattr(after, "st_nlink", 1) != 1
        or identity_before != identity_opened
        or identity_before != identity_after
    ):
        raise TreeIntegrityError(f"file changed while its integrity was checked: {path}")
    return digest.hexdigest()


IgnorePredicate = Callable[[str, bool], bool]


def inventory_tree(
    root: Path,
    *,
    ignore: IgnorePredicate | None = None,
) -> TreeInventory:
    """Inventory a tree without following aliases.

    Ignored entries are still inspected and traversed so an ignored cache
    directory cannot conceal a symlink, junction, reparse point, or device.
    """

    raw_root = Path(root).absolute()
    try:
        root_info = raw_root.lstat()
    except OSError as exc:
        raise TreeIntegrityError(f"tree root is unavailable: {raw_root}") from exc
    if is_alias(root_info) or not stat.S_ISDIR(root_info.st_mode):
        raise TreeIntegrityError(f"tree root is not a regular local directory: {raw_root}")

    files: dict[str, str] = {}
    directories: set[str] = set()
    try:
        for current, dirnames, filenames in os.walk(
            raw_root, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            relative_current = current_path.relative_to(raw_root)
            current_ignored = any(
                ignore(parent.as_posix(), True)
                for parent in (relative_current, *relative_current.parents)
                if ignore is not None and parent != Path(".")
            )

            for name in list(dirnames):
                path = current_path / name
                info = path.lstat()
                relative = path.relative_to(raw_root).as_posix()
                if is_alias(info) or not stat.S_ISDIR(info.st_mode):
                    raise TreeIntegrityError(
                        f"tree contains an alias or special directory: {relative}"
                    )
                if not current_ignored and not (ignore and ignore(relative, True)):
                    directories.add(relative)

            for name in filenames:
                path = current_path / name
                info = path.lstat()
                relative = path.relative_to(raw_root).as_posix()
                if (
                    is_alias(info)
                    or not stat.S_ISREG(info.st_mode)
                    or getattr(info, "st_nlink", 1) != 1
                ):
                    raise TreeIntegrityError(
                        f"tree contains an alias, hardlink, or special file: {relative}"
                    )
                if current_ignored or (ignore and ignore(relative, False)):
                    continue
                files[relative] = sha256_regular_file(path)
    except TreeIntegrityError:
        raise
    except OSError as exc:
        raise TreeIntegrityError(f"tree could not be enumerated: {raw_root}") from exc

    return TreeInventory(
        files=dict(sorted(files.items())), directories=tuple(sorted(directories))
    )


def inventories_equal(actual: TreeInventory, expected: TreeInventory) -> bool:
    return actual.directories == expected.directories and dict(actual.files) == dict(
        expected.files
    )


def entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise TreeIntegrityError(f"cannot inspect tree entry: {path}") from exc
    return True


def remove_owned_entry(path: Path, owned_parent: Path) -> None:
    """Delete exactly one owned child without traversing an alias target."""

    target = Path(path).absolute()
    parent = Path(owned_parent).absolute()
    if target.parent != parent:
        raise TreeIntegrityError(
            f"refusing to remove a path outside its owned parent: {target}"
        )
    try:
        parent_info = parent.lstat()
        canonical_parent = parent.resolve(strict=True)
        canonical_target_parent = target.parent.resolve(strict=True)
    except OSError as exc:
        raise TreeIntegrityError(
            f"cannot verify the owned parent before removal: {parent}"
        ) from exc
    if (
        is_alias(parent_info)
        or not stat.S_ISDIR(parent_info.st_mode)
        or canonical_target_parent != canonical_parent
    ):
        raise TreeIntegrityError(
            f"refusing to remove through an unsafe owned parent: {parent}"
        )

    def remove(current: Path) -> None:
        try:
            info = current.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise TreeIntegrityError(f"cannot inspect owned entry: {current}") from exc
        if is_alias(info):
            try:
                if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                    current.rmdir()
                else:
                    current.unlink()
            except IsADirectoryError:
                current.rmdir()
            except OSError as exc:
                raise TreeIntegrityError(f"cannot remove owned alias: {current}") from exc
            return
        if stat.S_ISDIR(info.st_mode):
            try:
                children = list(os.scandir(current))
            except OSError as exc:
                raise TreeIntegrityError(
                    f"cannot enumerate owned directory: {current}"
                ) from exc
            for child in children:
                remove(Path(child.path))
            try:
                current.rmdir()
            except OSError as exc:
                raise TreeIntegrityError(f"cannot remove owned directory: {current}") from exc
            return
        try:
            current.unlink()
        except OSError as exc:
            raise TreeIntegrityError(f"cannot remove owned file: {current}") from exc

    remove(target)
