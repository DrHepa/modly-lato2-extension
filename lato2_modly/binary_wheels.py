"""Pinned binary distribution for the portable operator; never build on users' PCs.

The committed inventory, not a downloaded manifest or pip cache, is the trust
root. Wheel identity includes the exact Torch build as well as Python and OS.
"""
from __future__ import annotations

from dataclasses import dataclass
import base64
import csv
from email.parser import BytesParser
import io
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Mapping
from zipfile import ZipFile, BadZipFile

from .assets import ensure_asset, verify_asset
from .constants import AssetSpec
from .integrity import read_owned_regular_bytes
from .ovoxel_cpu import (
    OVOXEL_CPU_BUILD_IDENTITY, OVOXEL_CPU_DISTRIBUTION, OVOXEL_CPU_VERSION,
    TEMPLATE_TREE_SHA256, LICENSE_SOURCE_SPECS,
)

SCHEMA = "modly.lato2.binary-wheels.v1"
INDEX = Path(__file__).with_name("binary-wheels.json")
RELEASE_BASE = "https://github.com/DrHepa/modly-lato2-extension/releases/download/"
MAX_WHEEL_BYTES = 128 * 1024 * 1024


class BinaryWheelError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code, self.public_message = code, message
        super().__init__(f"{code}: {message}")


def _error(message: str):
    raise BinaryWheelError("BINARY_INVENTORY_INVALID", message)


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None and value != "0" * 64


def _relative(name: str) -> bool:
    return (isinstance(name, str) and bool(name) and "\\" not in name and ":" not in name
            and "\x00" not in name and not PurePosixPath(name).is_absolute()
            and all(p not in {"", ".", ".."} for p in name.split("/")))


@dataclass(frozen=True)
class WheelSpec:
    key: str
    filename: str
    release: str
    sha256: str
    size: int
    build_identity: str
    template_sha256: str
    source_commit: str
    files: Mapping[str, str]
    glibc_min: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return {"provisioning": "prebuilt-wheel", **self.__dict__}

    @property
    def asset(self) -> AssetSpec:
        return AssetSpec(f"binary-wheels/{self.sha256}/{self.filename}", self.size,
                         self.sha256, f"{RELEASE_BASE}{self.release}/{self.filename}", "portable-cpu-wheel")


def wheel_key(plan) -> str:
    return f"{plan.system}/{plan.arch}/{plan.python_abi.lane}/{plan.torch_requirements[0].split('==', 1)[1]}"


def parse_spec(value: Mapping) -> WheelSpec:
    required = {"key", "filename", "release", "sha256", "size", "build_identity", "template_sha256", "source_commit", "files", "glibc_min"}
    if not isinstance(value, dict) or set(value) != required:
        _error("wheel record has missing or unknown fields")
    key = value["key"]
    if not isinstance(key, str) or not re.fullmatch(r"(win32/x64|linux/(x64|arm64))/cp3(11|12)/(2\.6\.0\+cu12[46]|2\.9\.1\+cu128)", key):
        _error("unsupported binary key")
    system, arch, python, torch = key.split("/")
    cuda = torch.split("+", 1)[1]
    if (cuda == "cu126") != (system == "linux" and arch == "arm64" and torch.startswith("2.6.0")):
        _error("incoherent CUDA/platform combination")
    platform_tag = {("win32", "x64"): "win_amd64", ("linux", "x64"): "linux_x86_64", ("linux", "arm64"): "linux_aarch64"}[(system, arch)]
    build = "1torch" + torch.replace(".", "").replace("+", "")
    filename = f"modly_lato2_ovoxel_cpu-{OVOXEL_CPU_VERSION}-{build}-{python}-{python}-{platform_tag}.whl"
    if value["filename"] != filename or not re.fullmatch(r"ovoxel-cpu-[A-Za-z0-9._-]+", str(value["release"])):
        _error("wheel name or release does not match its key")
    if not _sha(value["sha256"]) or type(value["size"]) is not int or not 0 < value["size"] <= MAX_WHEEL_BYTES:
        _error("invalid wheel size or digest")
    if value["build_identity"] != OVOXEL_CPU_BUILD_IDENTITY or value["template_sha256"] != TEMPLATE_TREE_SHA256:
        _error("binary built from different operator sources or licenses")
    if not re.fullmatch(r"[a-f0-9]{40}", str(value["source_commit"])):
        _error("missing source commit provenance")
    files = value["files"]
    dist = f"modly_lato2_ovoxel_cpu-{OVOXEL_CPU_VERSION}.dist-info"
    if not isinstance(files, dict) or not 4 <= len(files) <= 64:
        _error("invalid installed-file inventory")
    for name, digest in files.items():
        if not _relative(name) or not _sha(digest) or not name.startswith(("lato2_ovoxel_cpu/", dist + "/")) or name.endswith("/RECORD"):
            _error("unsafe installed-file inventory")
    binaries = [name for name in files if name.endswith((".pyd", ".so"))]
    if len(binaries) != 1 or not binaries[0].startswith("lato2_ovoxel_cpu/_C."):
        _error("the operator binary is missing or ambiguous")
    if any(name.endswith(".pth") for name in files) or f"{dist}/METADATA" not in files or f"{dist}/WHEEL" not in files:
        _error("unexpected executable path hook or missing wheel metadata")
    for name, (_, digest) in LICENSE_SOURCE_SPECS.items():
        if files.get(f"{dist}/licenses/LICENSES/{name}") != digest:
            _error("pinned license is missing from wheel")
    minimum = value["glibc_min"]
    if system == "linux":
        if not isinstance(minimum, (tuple, list)) or len(minimum) != 2 or any(type(x) is not int or x < 0 for x in minimum):
            _error("Linux binary must declare its tested glibc baseline")
        minimum = tuple(minimum)
    elif minimum is not None:
        _error("Windows wheel cannot declare a glibc baseline")
    return WheelSpec(**{**value, "files": dict(files), "glibc_min": minimum})


def load_inventory(path: Path = INDEX) -> list[WheelSpec]:
    try:
        data = json.loads(read_owned_regular_bytes(path, max_bytes=512 * 1024))
        if set(data) != {"schema", "artifacts"} or data["schema"] != SCHEMA or not isinstance(data["artifacts"], list):
            _error("unknown binary inventory schema")
        specs = [parse_spec(value) for value in data["artifacts"]]
        if len({s.key for s in specs}) != len(specs):
            _error("duplicate binary keys")
        return specs
    except BinaryWheelError:
        raise
    except Exception as exc:
        raise BinaryWheelError("BINARY_INVENTORY_INVALID", "committed binary inventory cannot be read") from exc


def select_wheel(plan, *, inventory: list[WheelSpec] | None = None) -> WheelSpec:
    specs = load_inventory() if inventory is None else inventory
    key = wheel_key(plan)
    matches = [spec for spec in specs if spec.key == key]
    if len(matches) != 1:
        raise BinaryWheelError("BINARY_WHEEL_UNAVAILABLE", f"no published, verified portable operator for {key}; update the extension. No local compilation was attempted")
    spec = matches[0]
    if spec.glibc_min is not None:
        import platform
        libc, version = platform.libc_ver()
        try:
            actual = tuple(int(x) for x in version.split(".")[:2])
        except ValueError:
            actual = ()
        if libc != "glibc" or actual < spec.glibc_min:
            raise BinaryWheelError("BINARY_GLIBC_UNSUPPORTED", f"{key} needs glibc {'.'.join(map(str, spec.glibc_min))} or newer")
    return spec


def verify_wheel(path: Path, spec: WheelSpec) -> None:
    valid, reason = verify_asset(path, spec.asset)
    if not valid:
        raise BinaryWheelError("BINARY_WHEEL_CORRUPT", f"wheel integrity failed: {reason}")
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [i.filename for i in infos if not i.is_dir()]
            if len(names) != len(set(names)) or len(infos) > 128 or sum(i.file_size for i in infos) > MAX_WHEEL_BYTES:
                _error("wheel archive is ambiguous or too large")
            records = [n for n in names if n.endswith(".dist-info/RECORD")]
            if len(records) != 1 or set(names) != set(spec.files) | set(records):
                _error("wheel contents differ from committed inventory")
            for info in infos:
                if not _relative(info.filename.rstrip("/")) or stat.S_ISLNK(info.external_attr >> 16):
                    _error("unsafe wheel member")
            for name, digest in spec.files.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    _error("wheel member differs from committed hash")
            dist = f"modly_lato2_ovoxel_cpu-{OVOXEL_CPU_VERSION}.dist-info"
            metadata = BytesParser().parsebytes(archive.read(f"{dist}/METADATA"))
            wheel = BytesParser().parsebytes(archive.read(f"{dist}/WHEEL"))
            _, _, build, py, abi, plat = spec.filename[:-4].split("-")
            if (metadata["Name"] != OVOXEL_CPU_DISTRIBUTION
                    or metadata["Version"] != OVOXEL_CPU_VERSION
                    or wheel["Root-Is-Purelib"] != "false"
                    or wheel.get_all("Tag") != [f"{py}-{abi}-{plat}"]
                    or wheel["Build"] != build):
                _error("wheel metadata does not match its binary identity")
            rows = list(csv.reader(io.StringIO(archive.read(records[0]).decode("utf-8"))))
            if any(len(row) != 3 for row in rows) or len(rows) != len(names) or {row[0] for row in rows} != set(names):
                _error("wheel RECORD does not cover its exact file set")
            for name, digest, size in rows:
                if name == records[0]:
                    if digest or size:
                        _error("wheel RECORD self-hash is invalid")
                    continue
                expected = "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(spec.files[name])).rstrip(b"=").decode("ascii")
                if digest != expected or size != str(archive.getinfo(name).file_size):
                    _error("wheel RECORD contains a mismatched hash or size")
    except (BadZipFile, OSError, KeyError) as exc:
        raise BinaryWheelError("BINARY_WHEEL_INVALID", "cannot inspect operator wheel") from exc


def ensure_wheel(cache_root: Path, spec: WheelSpec, *, log=print) -> Path:
    path = ensure_asset(cache_root, spec.asset, log=log)
    verify_wheel(path, spec)
    return path


_INSTALLED_PROBE = r'''
import hashlib, importlib.metadata, json, os, pathlib, stat, sys
spec = json.loads(os.environ['MODLY_LATO2_BINARY_SPEC'])
prefix = pathlib.Path(sys.prefix).resolve()
dist = importlib.metadata.distribution('modly-lato2-ovoxel-cpu')
for relative, expected in spec['files'].items():
    path = pathlib.Path(dist.locate_file(relative))
    path.relative_to(prefix)
    cursor = path
    while cursor != prefix:
        info = cursor.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise RuntimeError('binary package path is an alias')
        cursor = cursor.parent
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError('binary package file is not a private regular file')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != expected:
        raise RuntimeError('installed binary package content changed: ' + relative)
import torch
if torch.__version__ != spec['key'].split('/')[-1]:
    raise RuntimeError('installed Torch build does not match operator binary')
print(json.dumps({'wheelSha256': spec['sha256'], 'filesVerified': len(spec['files'])}))
'''


def verify_installed(python: Path, spec: WheelSpec, *, env: Mapping[str, str]) -> dict:
    child_env = dict(env)
    child_env["MODLY_LATO2_BINARY_SPEC"] = json.dumps(spec.to_dict())
    try:
        result = subprocess.run([str(python), "-I", "-c", _INSTALLED_PROBE], env=child_env,
                                stdin=subprocess.DEVNULL, check=True, capture_output=True, text=True, timeout=120)
        report = json.loads(result.stdout)
        if report != {"wheelSha256": spec.sha256, "filesVerified": len(spec.files)}:
            raise ValueError("binary probe returned unexpected identity")
        return report
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise BinaryWheelError("BINARY_INSTALLED_INVALID", "installed operator differs from its pinned wheel or Torch build; run Repair") from exc
