"""Materialize the minimal pinned o-voxel CPU extension build tree."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .integrity import (
    TreeIntegrityError,
    TreeInventory,
    entry_exists,
    inventories_equal,
    inventory_tree,
    remove_owned_entry,
    sha256_regular_file,
)


TRELLIS_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
EIGEN_REVISION = "21e4582d1739107337a03460c81412981130373e"
EIGEN_ARCHIVE_URL = (
    "https://gitlab.com/libeigen/eigen/-/archive/"
    f"{EIGEN_REVISION}/eigen-{EIGEN_REVISION}.zip"
)
FLEXIBLE_DUAL_GRID_SHA256 = (
    "95ebcdec3818539c52504cd4a89f409287afc51b051086b9092e11fc7308063f"
)
CONVERT_API_SHA256 = (
    "7391a6e6bb0a94ab928261a6b4f727c7507cd637d178e33a880d2f699a2a715d"
)
BUILD_MARKER = ".modly-ovoxel-cpu-build.json"
BUILD_SCHEMA_VERSION = 3
EIGEN_TREE_SHA256 = "a619ab288a4da17edd0caf0a3f2335d37ea26df14dd5eedb1591d0d1d860ddb7"
OVOXEL_CPU_DISTRIBUTION = "modly-lato2-ovoxel-cpu"
OVOXEL_CPU_VERSION = "0.0.1.post2"
TEMPLATE_TREE_SHA256 = "930378bb300fde0e8e99d68750e3e2ba1937995901c7e1634a7ba59deb928691"

# The locally built wheel contains copied TRELLIS.2 and Eigen code.  Keep the
# license payload anchored to repository-controlled hashes so a stale or
# modified models cache cannot silently publish different notices.
LICENSE_SOURCE_SPECS = {
    "Modly-LATO2-Wrapper-MIT.txt": (
        "LICENSE",
        "2e55c53ff294650e049d844f2544fec947c3516440aeffca4b2334cf94b13eeb",
    ),
    "TRELLIS.2-MIT.txt": (
        "LICENSES/TRELLIS.2-MIT.txt",
        "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383",
    ),
    "Eigen-MPL-2.0.txt": (
        "LICENSES/Eigen-MPL-2.0.txt",
        "66a3107d5ad6a058aab753eaac2047ccb2ed0e39465dd0fe5844da3e300d5172",
    ),
    "Eigen-Apache-2.0-notice.txt": (
        "LICENSES/Eigen-Apache-2.0-notice.txt",
        "8fae83faf0810de83f32636b2c97aaddaf7ae95008456e1e7f87cc6748214e95",
    ),
    "Eigen-BSD-notice.txt": (
        "LICENSES/Eigen-BSD-notice.txt",
        "51928dce36213c5333ba3172e847d735d4c6e9b7ff2722a326c49067155b82eb",
    ),
    "Eigen-MINPACK-notice.txt": (
        "LICENSES/Eigen-MINPACK-notice.txt",
        "c87b7f8ee88f6195e91743820c00354833583aef091b72e2d4a49c8e28e798a0",
    ),
    "Eigen-COPYING-README.txt": (
        "LICENSES/Eigen-COPYING-README.txt",
        "72fe0574781a5838d62b9abc804991489e52ad590e03514c79cdc65b4ed68403",
    ),
}


def _build_identity_payload() -> dict[str, object]:
    """Return the extension-controlled identity of every wheel build input."""

    return {
        "schema_version": BUILD_SCHEMA_VERSION,
        "distribution": OVOXEL_CPU_DISTRIBUTION,
        "version": OVOXEL_CPU_VERSION,
        "template_tree_sha256": TEMPLATE_TREE_SHA256,
        "trellis_revision": TRELLIS_REVISION,
        "flexible_dual_grid_sha256": FLEXIBLE_DUAL_GRID_SHA256,
        "convert_api_sha256": CONVERT_API_SHA256,
        "eigen_revision": EIGEN_REVISION,
        "eigen_tree_sha256": EIGEN_TREE_SHA256,
        "license_sha256": {
            name: digest
            for name, (_relative, digest) in sorted(LICENSE_SOURCE_SPECS.items())
        },
    }


OVOXEL_CPU_BUILD_IDENTITY = hashlib.sha256(
    json.dumps(
        _build_identity_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class OVoxelCpuBuildError(RuntimeError):
    """Raised when the minimal native build cannot be assembled safely."""


@dataclass(frozen=True)
class OVoxelCpuBuildReport:
    build_root: str
    distribution: str
    version: str
    build_identity: str
    trellis_revision: str
    eigen_revision: str
    eigen_tree_sha256: str
    reused: bool
    pip_install_args: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _file_sha256(path: Path) -> str:
    try:
        return sha256_regular_file(path)
    except TreeIntegrityError as exc:
        raise OVoxelCpuBuildError(str(exc)) from exc


def _template_ignore(relative: str, is_directory: bool) -> bool:
    parts = Path(relative).parts
    return "__pycache__" in parts or (not is_directory and Path(relative).suffix == ".pyc")


def _inventory(root: Path, *, template: bool = False) -> TreeInventory:
    try:
        return inventory_tree(root, ignore=_template_ignore if template else None)
    except TreeIntegrityError as exc:
        raise OVoxelCpuBuildError(str(exc)) from exc


def _tree_hash(root: Path) -> str:
    return _inventory(root).digest


def _directories_for_files(files: dict[str, str]) -> tuple[str, ...]:
    directories: set[str] = set()
    for relative in files:
        directories.update(
            parent.as_posix()
            for parent in Path(relative).parents
            if parent != Path(".")
        )
    return tuple(sorted(directories))


def _expected_license_hashes() -> dict[str, str]:
    return {name: digest for name, (_relative, digest) in LICENSE_SOURCE_SPECS.items()}


def _validate_bundled_licenses(build_root: Path) -> bool:
    license_root = build_root / "LICENSES"
    try:
        if not license_root.is_dir() or license_root.is_symlink():
            return False
        actual = {
            path.name: _file_sha256(path)
            for path in license_root.iterdir()
            if path.is_file() and not path.is_symlink()
        }
    except OSError:
        return False
    return actual == _expected_license_hashes()


def _copy_bundled_licenses(project_root: Path, staging: Path) -> None:
    destination = staging / "LICENSES"
    destination.mkdir()
    for output_name, (relative, expected_digest) in LICENSE_SOURCE_SPECS.items():
        source = project_root.joinpath(*Path(relative).parts)
        try:
            if not source.is_file() or source.is_symlink():
                raise OVoxelCpuBuildError(f"license source is unsafe: {relative}")
            if _file_sha256(source) != expected_digest:
                raise OVoxelCpuBuildError(f"license source does not match its pin: {relative}")
            shutil.copyfile(source, destination / output_name)
        except OSError as exc:
            raise OVoxelCpuBuildError(f"could not copy pinned license: {relative}") from exc
    if not _validate_bundled_licenses(staging):
        raise OVoxelCpuBuildError("materialized license payload failed validation")


def _resolve_ovoxel_root(path: Path) -> Path:
    direct = path / "src" / "convert" / "flexible_dual_grid.cpp"
    if direct.is_file():
        return path
    nested = path / "o-voxel" / "src" / "convert" / "flexible_dual_grid.cpp"
    if nested.is_file():
        return path / "o-voxel"
    raise OVoxelCpuBuildError(
        f"could not find o-voxel/src/convert under pinned TRELLIS source: {path}"
    )


def _validate_inputs(ovoxel_root: Path, eigen_root: Path) -> tuple[Path, str]:
    ov = _resolve_ovoxel_root(ovoxel_root)
    flexible = ov / "src" / "convert" / "flexible_dual_grid.cpp"
    api = ov / "src" / "convert" / "api.h"
    actual_flexible = _file_sha256(flexible)
    actual_api = _file_sha256(api)
    if actual_flexible != FLEXIBLE_DUAL_GRID_SHA256:
        raise OVoxelCpuBuildError(
            "flexible_dual_grid.cpp is not the pinned TRELLIS.2 source "
            f"(got {actual_flexible})"
        )
    if actual_api != CONVERT_API_SHA256:
        raise OVoxelCpuBuildError(
            f"o-voxel convert/api.h is not pinned (got {actual_api})"
        )
    eigen_inventory = _inventory(eigen_root)
    for sentinel in ("Eigen/Core", "Eigen/Dense"):
        if sentinel not in eigen_inventory.files:
            raise OVoxelCpuBuildError(
                f"Eigen {EIGEN_REVISION} source is incomplete: missing {sentinel}"
            )
    if eigen_inventory.digest != EIGEN_TREE_SHA256:
        raise OVoxelCpuBuildError(
            "Eigen source does not match the complete tree extracted from the "
            f"pinned {EIGEN_REVISION} archive"
        )
    return ov, eigen_inventory.digest


def _read_marker(build_root: Path) -> dict | None:
    try:
        value = json.loads((build_root / BUILD_MARKER).read_text("utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _build_ignore(relative: str, is_directory: bool) -> bool:
    return not is_directory and relative == BUILD_MARKER


def _build_inventory(build_root: Path) -> TreeInventory:
    try:
        return inventory_tree(build_root, ignore=_build_ignore)
    except TreeIntegrityError as exc:
        raise OVoxelCpuBuildError(str(exc)) from exc


def _expected_build_inventory(
    template_root: Path,
    eigen_root: Path,
) -> tuple[TreeInventory, TreeInventory, TreeInventory]:
    template_inventory = _inventory(template_root, template=True)
    if template_inventory.digest != TEMPLATE_TREE_SHA256:
        raise OVoxelCpuBuildError(
            "o-voxel CPU build template does not match its extension-controlled pin"
        )
    eigen_inventory = _inventory(eigen_root)
    files = dict(template_inventory.files)
    files["src/flexible_dual_grid.cpp"] = FLEXIBLE_DUAL_GRID_SHA256
    files["src/api.h"] = CONVERT_API_SHA256
    files.update(
        {
            f"third_party/eigen/{relative}": digest
            for relative, digest in eigen_inventory.files.items()
        }
    )
    files.update(
        {
            f"LICENSES/{name}": digest
            for name, digest in _expected_license_hashes().items()
        }
    )
    directories = set(_directories_for_files(files))
    directories.update(template_inventory.directories)
    directories.update(
        f"third_party/eigen/{relative}"
        for relative in eigen_inventory.directories
    )
    output = TreeInventory(
        files=dict(sorted(files.items())),
        directories=tuple(sorted(directories)),
    )
    return template_inventory, eigen_inventory, output


def _build_marker_payload(
    template: TreeInventory,
    eigen: TreeInventory,
    output: TreeInventory,
) -> dict[str, object]:
    return {
        "schema_version": BUILD_SCHEMA_VERSION,
        "distribution": OVOXEL_CPU_DISTRIBUTION,
        "version": OVOXEL_CPU_VERSION,
        "build_identity": OVOXEL_CPU_BUILD_IDENTITY,
        "trellis_revision": TRELLIS_REVISION,
        "eigen_revision": EIGEN_REVISION,
        "template_tree_sha256": template.digest,
        "eigen_tree_sha256": eigen.digest,
        "flexible_dual_grid_sha256": FLEXIBLE_DUAL_GRID_SHA256,
        "convert_api_sha256": CONVERT_API_SHA256,
        "license_sha256": _expected_license_hashes(),
        "build_tree_sha256": output.digest,
        "build_source_sha256": dict(output.files),
    }


def validate_ovoxel_cpu_build(
    build_root: os.PathLike[str] | str,
    *,
    eigen_tree_sha256: str | None = None,
    template_root: os.PathLike[str] | str | None = None,
) -> bool:
    root = Path(build_root).absolute()
    bundled_template = template_root is None
    template = (
        Path(template_root).absolute()
        if template_root is not None
        else Path(__file__).resolve().parent.parent / "native" / "ovoxel_cpu_template"
    )
    try:
        eigen_root = root / "third_party" / "eigen"
        template_inventory, eigen_inventory, expected_output = _expected_build_inventory(
            template, eigen_root
        )
        if bundled_template and template_inventory.digest != TEMPLATE_TREE_SHA256:
            return False
        if eigen_inventory.digest != EIGEN_TREE_SHA256:
            return False
        if eigen_tree_sha256 is not None and eigen_inventory.digest != eigen_tree_sha256:
            return False
        actual_output = _build_inventory(root)
        if not inventories_equal(actual_output, expected_output):
            return False
        marker = _read_marker(root)
        return marker == _build_marker_payload(
            template_inventory, eigen_inventory, expected_output
        )
    except (OSError, OVoxelCpuBuildError):
        return False


def materialize_ovoxel_cpu_build(
    ovoxel_source_root: os.PathLike[str] | str,
    eigen_source_root: os.PathLike[str] | str,
    build_root: os.PathLike[str] | str,
    *,
    template_root: os.PathLike[str] | str | None = None,
) -> OVoxelCpuBuildReport:
    """Create a build tree for the exact CPU function used by LATO.2.

    ``eigen_source_root`` must be the already verified extraction of the pinned
    Eigen submodule archive. Setup remains responsible for asset URL/size/SHA
    validation before calling this function.
    """
    ovoxel_input = Path(ovoxel_source_root).absolute()
    eigen = Path(eigen_source_root).absolute()
    destination = Path(build_root).absolute()
    bundled_template = template_root is None
    template = (
        Path(template_root).absolute()
        if template_root is not None
        else Path(__file__).resolve().parent.parent / "native" / "ovoxel_cpu_template"
    )
    project_root = Path(__file__).resolve().parent.parent
    if not template.is_dir():
        raise OVoxelCpuBuildError(f"minimal o-voxel build template is missing: {template}")
    ovoxel, eigen_hash = _validate_inputs(ovoxel_input, eigen)
    template_inventory, eigen_inventory, output_inventory = _expected_build_inventory(
        template, eigen
    )
    if bundled_template and template_inventory.digest != TEMPLATE_TREE_SHA256:
        raise OVoxelCpuBuildError(
            "bundled o-voxel CPU template does not match its extension pin"
        )
    if eigen_inventory.digest != eigen_hash:
        raise OVoxelCpuBuildError(
            "validated Eigen tree changed before the build could be materialized"
        )
    if validate_ovoxel_cpu_build(
        destination,
        eigen_tree_sha256=eigen_hash,
        template_root=template,
    ):
        return OVoxelCpuBuildReport(
            build_root=str(destination),
            distribution=OVOXEL_CPU_DISTRIBUTION,
            version=OVOXEL_CPU_VERSION,
            build_identity=OVOXEL_CPU_BUILD_IDENTITY,
            trellis_revision=TRELLIS_REVISION,
            eigen_revision=EIGEN_REVISION,
            eigen_tree_sha256=eigen_hash,
            reused=True,
            pip_install_args=(
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-deps",
                str(destination),
            ),
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staging = destination.parent / f".{destination.name}.staging-{token}"
    backup = destination.parent / f".{destination.name}.backup-{token}"
    try:
        shutil.copytree(
            template,
            staging,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(
            ovoxel / "src" / "convert" / "flexible_dual_grid.cpp",
            staging / "src" / "flexible_dual_grid.cpp",
        )
        shutil.copy2(
            ovoxel / "src" / "convert" / "api.h", staging / "src" / "api.h"
        )
        shutil.copytree(eigen, staging / "third_party" / "eigen")
        _copy_bundled_licenses(project_root, staging)
        actual_staging = _build_inventory(staging)
        if not inventories_equal(actual_staging, output_inventory):
            raise OVoxelCpuBuildError(
                "staged o-voxel CPU build differs from its trusted inputs"
            )
        marker = _build_marker_payload(
            template_inventory, eigen_inventory, output_inventory
        )
        (staging / BUILD_MARKER).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="",
        )
        if entry_exists(destination):
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException:
            if entry_exists(backup) and not entry_exists(destination):
                os.replace(backup, destination)
            raise
        if entry_exists(backup):
            remove_owned_entry(backup, destination.parent)
    finally:
        if entry_exists(staging):
            remove_owned_entry(staging, destination.parent)

    if not validate_ovoxel_cpu_build(
        destination,
        eigen_tree_sha256=eigen_hash,
        template_root=template,
    ):
        raise OVoxelCpuBuildError("minimal o-voxel build failed validation")
    return OVoxelCpuBuildReport(
        build_root=str(destination),
        distribution=OVOXEL_CPU_DISTRIBUTION,
        version=OVOXEL_CPU_VERSION,
        build_identity=OVOXEL_CPU_BUILD_IDENTITY,
        trellis_revision=TRELLIS_REVISION,
        eigen_revision=EIGEN_REVISION,
        eigen_tree_sha256=eigen_hash,
        reused=False,
        pip_install_args=(
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            str(destination),
        ),
    )
