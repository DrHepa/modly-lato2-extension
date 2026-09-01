"""Pinned, isolated dependency provisioning for LATO.2.

The ``exact-upstream`` profile installs every dependency used by the pinned
LATO.2 inference implementation, including all four CUDA extensions imported
by ``o_voxel``.  The ``portable`` profile is a separately fingerprinted route
for Modly's compatibility overlays; it must never be mistaken for, or reused
as, an exact-upstream environment.

No function in this module invokes ``git``.  Native sources and their
submodules are fetched as immutable, SHA-256-verified archives so Install from
GitHub also works on Windows machines without Git.
"""

from __future__ import annotations

import base64
import csv
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import sysconfig
import tempfile
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid
import zipfile

from .ovoxel_cpu import (
    EIGEN_TREE_SHA256,
    LICENSE_SOURCE_SPECS,
    OVOXEL_CPU_BUILD_IDENTITY,
    OVOXEL_CPU_DISTRIBUTION,
    OVOXEL_CPU_VERSION,
    TEMPLATE_TREE_SHA256,
)

from .integrity import (
    TreeIntegrityError,
    inventory_tree,
    read_owned_regular_bytes,
    sha256_regular_file,
)


LogFunction = Callable[[str], None]
CommandRunner = Callable[[Sequence[str], Mapping[str, str] | None], None]

DEPENDENCY_SCHEMA = "modly.lato2.dependencies.v2"
SOURCE_SCHEMA = "modly.lato2.native-sources.v1"
STATE_MAX_BYTES = 128 * 1024
DOWNLOAD_CHUNK = 1024 * 1024
COMMAND_TIMEOUT_SECONDS = 4 * 60 * 60
SOURCE_FILE_LIMIT = 25_000
SOURCE_UNCOMPRESSED_LIMIT = 2 * 1024 * 1024 * 1024
PYPI_INDEX = "https://pypi.org/simple"
SUPPORTED_PYTHON_VERSIONS = frozenset({(3, 11), (3, 12)})
_SUPPORTED_RELEASE_ABIS = {
    ((3, 11), "linux-x86_64", "x86_64"): "cpython-311-x86_64-linux-gnu",
    ((3, 12), "linux-x86_64", "x86_64"): "cpython-312-x86_64-linux-gnu",
    ((3, 11), "linux-aarch64", "aarch64"): "cpython-311-aarch64-linux-gnu",
    ((3, 12), "linux-aarch64", "aarch64"): "cpython-312-aarch64-linux-gnu",
    ((3, 11), "win-amd64", "amd64"): "cp311-win_amd64",
    ((3, 12), "win-amd64", "amd64"): "cp312-win_amd64",
}

_SENSITIVE_ENV_NAME = re.compile(
    r"TOKEN|SECRET|PASSWORD|PASSWD|AUTH|API_KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIAL|COOKIE",
    re.IGNORECASE,
)
_CLOUD_ENV_PREFIXES = ("AWS_", "AZURE_", "GOOGLE_", "GCP_")
_SAFE_ENV_NAMES = frozenset(
    {
        "APPDATA",
        "ARCHFLAGS",
        "BUILD_TARGET",
        "CC",
        "CFLAGS",
        "CL",
        "COMSPEC",
        "COMMONPROGRAMFILES",
        "COMMONPROGRAMFILES(X86)",
        "COMMONPROGRAMW6432",
        "COMPILER_PATH",
        "CPATH",
        "CPPFLAGS",
        "CPLUS_INCLUDE_PATH",
        "CUDACXX",
        "CUDA_CACHE_PATH",
        "CUDA_HOME",
        "CUDA_MODULE_LOADING",
        "CUDA_PATH",
        "CXX",
        "CXXFLAGS",
        "DISTUTILS_USE_SDK",
        "DEVENVDIR",
        "EGL_PLATFORM",
        "EXTENSIONSDKDIR",
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "INCLUDE",
        "LANG",
        "LANGUAGE",
        "LDFLAGS",
        "LINK",
        "LIB",
        "LIBPATH",
        "LIBRARY_PATH",
        "LOCALAPPDATA",
        "MAX_JOBS",
        "MODLY_LATO2_CACHE_DIR",
        "MSSDK",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATH",
        "PATHEXT",
        "PIP_CACHE_DIR",
        "PKG_CONFIG_PATH",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_ARCHITEW6432",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "PYTHONNOUSERSITE",
        "SOURCE_DATE_EPOCH",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_CUDA_ARCH_LIST",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "TZ",
        "USERPROFILE",
        "VISUALSTUDIOVERSION",
        "WINDIR",
        "WINDOWSLIBPATH",
        "XDG_RUNTIME_DIR",
        "_CL_",
        "_LINK_",
        "__VSCMD_PREINIT_PATH",
    }
)
_SAFE_ENV_PREFIXES = (
    "CUDA_",
    "FRAMEWORK",
    "LC_",
    "NETFX",
    "NVCC_",
    "NVVMIR_",
    "OMP_",
    "UCRT",
    "UNIVERSALCRT",
    "VCINSTALLDIR",
    "VCTOOLS",
    "VSCMD_",
    "VSINSTALLDIR",
    "WINDOWSSDK",
)
_NETWORK_ENV_NAMES = frozenset(
    {
        "ALL_PROXY",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
    }
)

# Complete tree digests derived once from the immutable SOURCE_ARCHIVES below,
# after applying this module's deterministic patches and source layout. These
# extension-controlled values, rather than a marker stored beside the cache,
# are the authority for reuse. Tree hashing includes every regular file and
# directory path, including source-lock.json.
NATIVE_SOURCE_TREE_SHA256 = {
    "linux": "6a115baebee751968548139d1955f44e88e504c735cc16a1f18bdeff1fa56696",
    "win32": "f8740e67bdde95f36d254976bf702f361f1315924853aeb5f2ba39d40385a88c",
}
PORTABLE_CPU_SOURCE_TREE_SHA256 = (
    "c1ac8616f74c13d86acb0d3945ee648e9cc0da23fe2527108e99bb6006fa5a1d"
)

BOOTSTRAP_REQUIREMENTS = (
    "pip==25.1.1",
    "setuptools==80.9.0",
    "wheel==0.45.1",
    "packaging==25.0",
)

# Versions shared with the pinned upstream Space where possible.  Direct
# dependencies omitted by that Space but required by upstream setup.sh are
# pinned here as well.  CPython 3.11 retains its original closure; CPython
# 3.12 selects a distinct ABI-keyed lane below.  Both are supplied to every
# pip install without allowing their caches or native build products to mix.
BASE_REQUIREMENTS = (
    "numpy==2.2.6",
    "trimesh==4.10.1",
    "tqdm==4.67.1",
    "pillow==12.0.0",
    "ninja==1.13.0",
    "psutil==7.1.3",
    "opencv-python-headless==4.12.0.88",
    "huggingface-hub==0.36.0",
    "plyfile==1.1",
    "zstandard==0.25.0",
    "easydict==1.13",
    "einops==0.8.1",
    # FlexGEMM imports FileLock directly from its autotuner.  Pin it here
    # because every local native source is deliberately installed --no-deps;
    # Torch providing it transitively is not a stable dependency contract.
    "filelock==3.20.0",
)
X64_RENDER_REQUIREMENTS = ("open3d==0.19.0",)

# Resolver/metadata-verified closures as of this release.  These are split by
# the dimensions that actually change wheel metadata; PEP 508 markers cannot
# express the selected CUDA/Torch lane or GPU SM.  Keep the selector below
# fail-closed when adding a new lane or platform.
COMMON_TRANSITIVE_REQUIREMENTS = (
    "certifi==2026.7.22",
    "charset-normalizer==3.5.1",
    "fsspec==2026.7.0",
    "hf-xet==1.6.0",
    "idna==3.19",
    "jinja2==3.1.6",
    "markupsafe==3.0.3",
    "mpmath==1.3.0",
    "networkx==3.6.1",
    "pyyaml==6.0.3",
    "requests==2.34.2",
    "typing-extensions==4.16.0",
    "urllib3==2.7.0",
)
X64_RENDER_COMMON_TRANSITIVE_REQUIREMENTS = (
    "annotated-types==0.8.0",
    "asttokens==3.0.2",
    "attrs==26.1.0",
    "blinker==1.9.0",
    "click==8.5.0",
    "comm==0.2.3",
    "configargparse==1.7.5",
    "dash==4.4.1",
    "executing==2.2.1",
    "fastjsonschema==2.22.2",
    "flask==3.1.3",
    "importlib-metadata==9.0.1",
    "ipython-pygments-lexers==1.1.1",
    "ipython==9.17.0",
    "ipywidgets==8.1.9",
    "itsdangerous==2.2.0",
    "janus==2.0.0",
    "jedi==0.20.0",
    "jsonschema-specifications==2025.9.1",
    "jsonschema==4.26.0",
    "jupyter-core==5.9.1",
    "jupyterlab-widgets==3.0.17",
    "matplotlib-inline==0.2.2",
    "narwhals==2.25.0",
    "nbformat==5.11.1",
    "nest-asyncio==1.6.0",
    "parso==0.8.7",
    "platformdirs==4.11.5",
    "plotly==7.0.0",
    "prompt-toolkit==3.0.53",
    "pure-eval==0.2.3",
    "pydantic-core==2.46.5",
    "pydantic==2.13.5",
    "pygments==2.21.0",
    "referencing==0.37.0",
    "retrying==1.4.2",
    "rpds-py==2026.6.3",
    "stack-data==0.6.3",
    "traitlets==5.16.1",
    "typing-inspection==0.4.4",
    "wcwidth==0.8.3",
    "werkzeug==3.1.8",
    "widgetsnbextension==4.0.16",
    "zipp==4.1.0",
)
LINUX_X64_RENDER_TRANSITIVE_REQUIREMENTS = (
    "addict==2.4.0",
    "contourpy==1.3.3",
    "cycler==0.12.1",
    "fonttools==4.63.0",
    "joblib==1.5.3",
    "kiwisolver==1.5.1",
    "matplotlib==3.11.1",
    "pandas==3.0.5",
    "pexpect==4.9.0",
    "ptyprocess==0.7.0",
    "pyparsing==3.3.2",
    "pyquaternion==0.9.9",
    "python-dateutil==2.9.0.post0",
    "scikit-learn==1.9.0",
    "scipy==1.17.1",
    "six==1.17.0",
    "threadpoolctl==3.6.0",
)
# The cp311 win_amd64 Open3D wheel intentionally omits the Linux scientific
# stack.  Current jupyter-core 5.9.1 also has no pywin32 runtime requirement;
# IPython's only Windows marker in this closure is colorama.
WINDOWS_X64_RENDER_TRANSITIVE_REQUIREMENTS = ("colorama==0.4.6",)

TORCH_LANE_COMMON_TRANSITIVE_REQUIREMENTS = {
    "cu124": ("sympy==1.13.1",),
    "cu126": ("sympy==1.13.1",),
    "cu128": ("sympy==1.14.0",),
}
TORCH_CU124_LINUX_X64_TRANSITIVE_REQUIREMENTS = (
    "nvidia-cublas-cu12==12.4.5.8",
    "nvidia-cuda-cupti-cu12==12.4.127",
    "nvidia-cuda-nvrtc-cu12==12.4.127",
    "nvidia-cuda-runtime-cu12==12.4.127",
    "nvidia-cudnn-cu12==9.1.0.70",
    "nvidia-cufft-cu12==11.2.1.3",
    "nvidia-curand-cu12==10.3.5.147",
    "nvidia-cusolver-cu12==11.6.1.9",
    "nvidia-cusparse-cu12==12.3.1.170",
    "nvidia-cusparselt-cu12==0.6.2",
    "nvidia-nccl-cu12==2.21.5",
    "nvidia-nvjitlink-cu12==12.4.127",
    "nvidia-nvtx-cu12==12.4.127",
    "triton==3.2.0",
)
TORCH_CU128_LINUX_TRANSITIVE_REQUIREMENTS = (
    "nvidia-cublas-cu12==12.8.4.1",
    "nvidia-cuda-cupti-cu12==12.8.90",
    "nvidia-cuda-nvrtc-cu12==12.8.93",
    "nvidia-cuda-runtime-cu12==12.8.90",
    "nvidia-cudnn-cu12==9.10.2.21",
    "nvidia-cufft-cu12==11.3.3.83",
    "nvidia-cufile-cu12==1.13.1.3",
    "nvidia-curand-cu12==10.3.9.90",
    "nvidia-cusolver-cu12==11.7.3.90",
    "nvidia-cusparse-cu12==12.5.8.93",
    "nvidia-cusparselt-cu12==0.7.1",
    "nvidia-nccl-cu12==2.27.5",
    "nvidia-nvjitlink-cu12==12.8.93",
    "nvidia-nvshmem-cu12==3.3.20",
    "nvidia-nvtx-cu12==12.8.90",
    "triton==3.5.1",
)
EXACT_TRANSITIVE_REQUIREMENTS = (
    "ccimport==0.4.4",
    "fire==0.7.1",
    "lark==1.3.1",
    "pccm==0.4.16",
    "portalocker==4.3.0",
    "pybind11==3.1.0",
    "termcolor==3.3.0",
)

PYTORCH_INDEXES = {
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
}
TORCH_LANES = {
    "cu124": ("torch==2.6.0+cu124", "torchvision==0.21.0+cu124"),
    "cu126": ("torch==2.6.0+cu126", "torchvision==0.21.0+cu126"),
    "cu128": ("torch==2.9.1+cu128", "torchvision==0.24.1+cu128"),
}


def _torch_requirements(lane: str, arch: str) -> tuple[str, str]:
    torch_requirement, vision_requirement = TORCH_LANES[lane]
    # Official Linux ARM64 torchvision wheels in the CUDA indexes deliberately
    # omit the PEP 440 local CUDA suffix, unlike torch and x64 torchvision.
    if arch == "arm64":
        vision_requirement = vision_requirement.split("+", 1)[0]
    return torch_requirement, vision_requirement

EXACT_SPARSE_REQUIREMENTS = (
    "cumm-cu124==0.7.11",
    "spconv-cu124==2.3.8",
)
TORCH_SCATTER_REQUIREMENT = "torch-scatter==2.1.2+pt26cu124"
TORCH_SCATTER_LINKS = "https://data.pyg.org/whl/torch-2.6.0+cu124.html"
XFORMERS_REQUIREMENT = "xformers==0.0.29.post2"
FLASH_ATTN_REQUIREMENT = "flash-attn==2.7.4.post1"
LINUX_TRITON_REQUIREMENT = "triton==3.2.0"
WINDOWS_TRITON_REQUIREMENT = "triton-windows==3.2.0.post21"
NATIVE_SOURCE_DISTRIBUTION_REQUIREMENTS = (
    "nvdiffrast==0.4.0",
    "cumesh==0.0.1",
    "flex-gemm==1.0.0",
    "o-voxel==0.0.1",
)

NVDIFFRAST_REVISION = "253ac4fcea7de5f396371124af597e6cc957bfae"
CUMESH_REVISION = "12289e1062f0603f2f0d0771b02e1395d247f26f"
CUBVH_REVISION = "ce92267a24ef6ad7d2c8ccbc2ae2c021a6597e70"
CUBVH_EIGEN_REVISION = "e63d9f6ccb7f6f29f31241b87c542f3f0ab3112b"
FLEXGEMM_REVISION = "6dd94a859c26ee8246888502eada3dd8ad85532e"
TRELLIS2_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
OVOXEL_EIGEN_REVISION = "21e4582d1739107337a03460c81412981130373e"
FLEXGEMM_CACHE_PATCH = "modly-flexgemm-owned-cache-v1"
OVOXEL_WINDOWS_PATCH = "trellis2-pr100-57c494a-compatible-v1"
OVOXEL_WINDOWS_PATCH_REFERENCE = "https://github.com/microsoft/TRELLIS.2/pull/100"


class DependencyError(RuntimeError):
    """Stable-code setup failure suitable for a Modly installation message."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.public_message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class _CusparseLtNormalizationContract:
    """Exact installed-tree identity for the audited ARM64 wheel correction."""

    schema: str
    distribution: str
    version: str
    dist_info: str
    package_root: str
    library_relative: str
    metadata_relative: str
    wheel_relative: str
    record_relative: str
    package_directories: tuple[str, ...]
    record_order: tuple[str, ...]
    fixed_files: tuple[tuple[str, str, int], ...]
    original_wheel: bytes
    normalized_wheel: bytes
    original_record_sha256: str
    normalized_record_sha256: str


_CUSPARSELT_DIST_INFO = "nvidia_cusparselt_cu12-0.7.1.dist-info"
_CUSPARSELT_PACKAGE_ROOT = "nvidia/cusparselt"
_CUSPARSELT_ORIGINAL_WHEEL = (
    b"Wheel-Version: 1.0\n"
    b"Generator: setuptools (75.8.0)\n"
    b"Root-Is-Purelib: true\n"
    b"Tag: py3-none-manylinux2014_sbsa\n\n"
)
_CUSPARSELT_NORMALIZED_WHEEL = _CUSPARSELT_ORIGINAL_WHEEL.replace(
    b"manylinux2014_sbsa", b"manylinux2014_aarch64"
)
_CUSPARSELT_NORMALIZATION_CONTRACT = _CusparseLtNormalizationContract(
    schema="modly.lato2.cusparselt-installed-metadata.v1",
    distribution="nvidia-cusparselt-cu12",
    version="0.7.1",
    dist_info=_CUSPARSELT_DIST_INFO,
    package_root=_CUSPARSELT_PACKAGE_ROOT,
    library_relative=f"{_CUSPARSELT_PACKAGE_ROOT}/lib/libcusparseLt.so.0",
    metadata_relative=f"{_CUSPARSELT_DIST_INFO}/METADATA",
    wheel_relative=f"{_CUSPARSELT_DIST_INFO}/WHEEL",
    record_relative=f"{_CUSPARSELT_DIST_INFO}/RECORD",
    package_directories=("include", "lib"),
    record_order=(
        f"{_CUSPARSELT_PACKAGE_ROOT}/LICENSE.txt",
        f"{_CUSPARSELT_PACKAGE_ROOT}/include/cusparseLt.h",
        f"{_CUSPARSELT_PACKAGE_ROOT}/lib/libcusparseLt.so.0",
        f"{_CUSPARSELT_DIST_INFO}/INSTALLER",
        f"{_CUSPARSELT_DIST_INFO}/METADATA",
        f"{_CUSPARSELT_DIST_INFO}/RECORD",
        f"{_CUSPARSELT_DIST_INFO}/WHEEL",
        f"{_CUSPARSELT_DIST_INFO}/top_level.txt",
    ),
    fixed_files=(
        (
            f"{_CUSPARSELT_PACKAGE_ROOT}/LICENSE.txt",
            "e8d158885a681b95ec7a6fc06dd8d4a52989f374cb1380c8a4c8fb27fd3d5d5e",
            17948,
        ),
        (
            f"{_CUSPARSELT_PACKAGE_ROOT}/include/cusparseLt.h",
            "74580d3104ed58e1708d2ee746f65c2a9e1557f91b1d158d2d93c1591e118c38",
            17876,
        ),
        (
            f"{_CUSPARSELT_PACKAGE_ROOT}/lib/libcusparseLt.so.0",
            "2c677e678d1955a6dedd66274dfe4cc0f930fea6421f1b8a5ec08cbb1ea18b17",
            440496193,
        ),
        (
            f"{_CUSPARSELT_DIST_INFO}/INSTALLER",
            "ceebae7b8927a3227e5303cf5e0f1f7b34bb542ad7250ac03fbcde36ec2f1508",
            4,
        ),
        (
            f"{_CUSPARSELT_DIST_INFO}/METADATA",
            "b264cea6951b70b52a3fd2ffaa88f9109eaf379633aa8d6e0832ecc92ddf6fba",
            6974,
        ),
        (
            f"{_CUSPARSELT_DIST_INFO}/top_level.txt",
            "a1f202f9d6cad2cdb68cde79b207d5a5b847593eb765d24b59e49be8aff5f812",
            18,
        ),
    ),
    original_wheel=_CUSPARSELT_ORIGINAL_WHEEL,
    normalized_wheel=_CUSPARSELT_NORMALIZED_WHEEL,
    original_record_sha256=(
        "e02265be65d5f1aab74b97ffd2d53ca2ccf9d41ba0ca9cddf8d9b4bd34262425"
    ),
    normalized_record_sha256=(
        "54b7063009a7cb86f1a0b2dd0bc0af777f8946440084365671e1745a1d61a976"
    ),
)


@dataclass(frozen=True)
class SourceArchive:
    name: str
    url: str
    size: int
    sha256: str
    root: str


SOURCE_ARCHIVES = (
    SourceArchive(
        "nvdiffrast",
        f"https://codeload.github.com/NVlabs/nvdiffrast/zip/{NVDIFFRAST_REVISION}",
        10_766_125,
        "145530b41dcb47092985f786a32ede3cc735f83fc9ffad21cbca708ac314b0fa",
        f"nvdiffrast-{NVDIFFRAST_REVISION}",
    ),
    SourceArchive(
        "cumesh",
        f"https://codeload.github.com/JeffreyXiang/CuMesh/zip/{CUMESH_REVISION}",
        154_059,
        "384faf56a343c8849e8b76b004574d3242102e4fdd093bed718b18a5ac2ccae3",
        f"CuMesh-{CUMESH_REVISION}",
    ),
    SourceArchive(
        "cubvh",
        f"https://codeload.github.com/JeffreyXiang/cubvh/zip/{CUBVH_REVISION}",
        112_584,
        "56bf3b8c7b911f11e052932316f98ed4ea1e7bf61bf426c19fff5f66b129846f",
        f"cubvh-{CUBVH_REVISION}",
    ),
    SourceArchive(
        "cubvh-eigen",
        (
            "https://gitlab.com/libeigen/eigen/-/archive/"
            f"{CUBVH_EIGEN_REVISION}/eigen-{CUBVH_EIGEN_REVISION}.zip"
        ),
        4_035_163,
        "8045353965d01cb43920e7d75ded2b2d98626727fac2009c1dcc331af2b4f061",
        f"eigen-{CUBVH_EIGEN_REVISION}",
    ),
    SourceArchive(
        "flexgemm",
        f"https://codeload.github.com/JeffreyXiang/FlexGEMM/zip/{FLEXGEMM_REVISION}",
        536_810,
        "b56b8f9504bee24a339b248ccc8b68be0b6a8ceee776ad12f6a78254bf368ec1",
        f"FlexGEMM-{FLEXGEMM_REVISION}",
    ),
    SourceArchive(
        "trellis2",
        f"https://codeload.github.com/microsoft/TRELLIS.2/zip/{TRELLIS2_REVISION}",
        18_222_787,
        "d7abd456f75585c5b4ce112a4abdb9bd3cc3e72b6074470a65e0bf3a89b3e06f",
        f"TRELLIS.2-{TRELLIS2_REVISION}",
    ),
    SourceArchive(
        "ovo-eigen",
        (
            "https://gitlab.com/libeigen/eigen/-/archive/"
            f"{OVOXEL_EIGEN_REVISION}/eigen-{OVOXEL_EIGEN_REVISION}.zip"
        ),
        4_236_331,
        "096270eac1ee338e5e40ad082725a8187c65001b9df08898e6f1a9c89582f035",
        f"eigen-{OVOXEL_EIGEN_REVISION}",
    ),
)


# CPython 3.12 is frozen independently from the original cp311 constants.
# Keep these as explicit tuple constructions rather than aliases: coincident
# pins may diverge without mutating or silently widening the upstream lane.
CP312_BOOTSTRAP_REQUIREMENTS = tuple(
    [
        "pip==25.1.1",
        "setuptools==80.9.0",
        "wheel==0.45.1",
        "packaging==25.0",
    ]
)
CP312_BASE_REQUIREMENTS = tuple(
    [
        "numpy==2.2.6",
        "trimesh==4.10.1",
        "tqdm==4.67.1",
        "pillow==12.0.0",
        "ninja==1.13.0",
        "psutil==7.1.3",
        "opencv-python-headless==4.12.0.88",
        "huggingface-hub==0.36.0",
        "plyfile==1.1",
        "zstandard==0.25.0",
        "easydict==1.13",
        "einops==0.8.1",
        "filelock==3.20.0",
    ]
)
CP312_COMMON_TRANSITIVE_REQUIREMENTS = tuple(
    [
        "certifi==2026.7.22",
        "charset-normalizer==3.5.1",
        "fsspec==2026.7.0",
        "hf-xet==1.6.0",
        "idna==3.19",
        "jinja2==3.1.6",
        "markupsafe==3.0.3",
        "mpmath==1.3.0",
        "networkx==3.6.1",
        "pyyaml==6.0.3",
        "requests==2.34.2",
        "typing-extensions==4.16.0",
        "urllib3==2.7.0",
    ]
)
CP312_X64_RENDER_REQUIREMENTS = tuple(["open3d==0.19.0"])
CP312_X64_RENDER_COMMON_TRANSITIVE_REQUIREMENTS = tuple(
    [
        "annotated-types==0.8.0",
        "asttokens==3.0.2",
        "attrs==26.1.0",
        "blinker==1.9.0",
        "click==8.5.0",
        "comm==0.2.3",
        "configargparse==1.7.5",
        "dash==4.4.1",
        "executing==2.2.1",
        "fastjsonschema==2.22.2",
        "flask==3.1.3",
        "importlib-metadata==9.0.1",
        "ipython-pygments-lexers==1.1.1",
        "ipython==9.17.0",
        "ipywidgets==8.1.9",
        "itsdangerous==2.2.0",
        "janus==2.0.0",
        "jedi==0.20.0",
        "jsonschema-specifications==2025.9.1",
        "jsonschema==4.26.0",
        "jupyter-core==5.9.1",
        "jupyterlab-widgets==3.0.17",
        "matplotlib-inline==0.2.2",
        "narwhals==2.25.0",
        "nbformat==5.11.1",
        "nest-asyncio==1.6.0",
        "parso==0.8.7",
        "platformdirs==4.11.5",
        "plotly==7.0.0",
        "prompt-toolkit==3.0.53",
        "pure-eval==0.2.3",
        "pydantic-core==2.46.5",
        "pydantic==2.13.5",
        "pygments==2.21.0",
        "referencing==0.37.0",
        "retrying==1.4.2",
        "rpds-py==2026.6.3",
        "stack-data==0.6.3",
        "traitlets==5.16.1",
        "typing-inspection==0.4.4",
        "wcwidth==0.8.3",
        "werkzeug==3.1.8",
        "widgetsnbextension==4.0.16",
        "zipp==4.1.0",
    ]
)
CP312_LINUX_X64_RENDER_TRANSITIVE_REQUIREMENTS = tuple(
    [
        "addict==2.4.0",
        "contourpy==1.3.3",
        "cycler==0.12.1",
        "fonttools==4.63.0",
        "joblib==1.5.3",
        "kiwisolver==1.5.1",
        "matplotlib==3.11.1",
        "pandas==3.0.5",
        "pexpect==4.9.0",
        "ptyprocess==0.7.0",
        "pyparsing==3.3.2",
        "pyquaternion==0.9.9",
        "python-dateutil==2.9.0.post0",
        "scikit-learn==1.9.0",
        "scipy==1.17.1",
        "six==1.17.0",
        "threadpoolctl==3.6.0",
    ]
)
CP312_WINDOWS_X64_RENDER_TRANSITIVE_REQUIREMENTS = tuple(["colorama==0.4.6"])
CP312_TORCH_LANES = tuple(
    [
        ("cu124", ("torch==2.6.0+cu124", "torchvision==0.21.0+cu124")),
        ("cu126", ("torch==2.6.0+cu126", "torchvision==0.21.0+cu126")),
        ("cu128", ("torch==2.9.1+cu128", "torchvision==0.24.1+cu128")),
    ]
)
CP312_TORCH_INDEXES = tuple(
    [
        ("cu124", "https://download.pytorch.org/whl/cu124"),
        ("cu126", "https://download.pytorch.org/whl/cu126"),
        ("cu128", "https://download.pytorch.org/whl/cu128"),
    ]
)
CP312_TORCH_COMMON_TRANSITIVE_REQUIREMENTS = tuple(
    [
        ("cu124", tuple(["sympy==1.13.1"])),
        ("cu126", tuple(["sympy==1.13.1"])),
        ("cu128", tuple(["sympy==1.14.0"])),
    ]
)
CP312_TORCH_CU124_LINUX_X64_TRANSITIVE_REQUIREMENTS = tuple(
    [
        "nvidia-cublas-cu12==12.4.5.8",
        "nvidia-cuda-cupti-cu12==12.4.127",
        "nvidia-cuda-nvrtc-cu12==12.4.127",
        "nvidia-cuda-runtime-cu12==12.4.127",
        "nvidia-cudnn-cu12==9.1.0.70",
        "nvidia-cufft-cu12==11.2.1.3",
        "nvidia-curand-cu12==10.3.5.147",
        "nvidia-cusolver-cu12==11.6.1.9",
        "nvidia-cusparse-cu12==12.3.1.170",
        "nvidia-cusparselt-cu12==0.6.2",
        "nvidia-nccl-cu12==2.21.5",
        "nvidia-nvjitlink-cu12==12.4.127",
        "nvidia-nvtx-cu12==12.4.127",
        "triton==3.2.0",
    ]
)
CP312_TORCH_CU128_LINUX_TRANSITIVE_REQUIREMENTS = tuple(
    [
        "nvidia-cublas-cu12==12.8.4.1",
        "nvidia-cuda-cupti-cu12==12.8.90",
        "nvidia-cuda-nvrtc-cu12==12.8.93",
        "nvidia-cuda-runtime-cu12==12.8.90",
        "nvidia-cudnn-cu12==9.10.2.21",
        "nvidia-cufft-cu12==11.3.3.83",
        "nvidia-cufile-cu12==1.13.1.3",
        "nvidia-curand-cu12==10.3.9.90",
        "nvidia-cusolver-cu12==11.7.3.90",
        "nvidia-cusparse-cu12==12.5.8.93",
        "nvidia-cusparselt-cu12==0.7.1",
        "nvidia-nccl-cu12==2.27.5",
        "nvidia-nvjitlink-cu12==12.8.93",
        "nvidia-nvshmem-cu12==3.3.20",
        "nvidia-nvtx-cu12==12.8.90",
        "triton==3.5.1",
    ]
)
CP312_EXACT_SPARSE_REQUIREMENTS = tuple(
    ["cumm-cu124==0.7.11", "spconv-cu124==2.3.8"]
)
CP312_TORCH_SCATTER_LINKS = "https://data.pyg.org/whl/torch-2.6.0+cu124.html"
CP312_EXACT_NATIVE_REQUIREMENTS = tuple(
    [
        ("always", "torch-scatter==2.1.2+pt26cu124"),
        ("always", "xformers==0.0.29.post2"),
        ("linux", "triton==3.2.0"),
        ("win32", "triton-windows==3.2.0.post21"),
        ("flash", "flash-attn==2.7.4.post1"),
        ("always", "nvdiffrast==0.4.0"),
        ("always", "cumesh==0.0.1"),
        ("always", "flex-gemm==1.0.0"),
        ("always", "o-voxel==0.0.1"),
        ("always", "ccimport==0.4.4"),
        ("always", "fire==0.7.1"),
        ("always", "lark==1.3.1"),
        ("always", "pccm==0.4.16"),
        ("always", "portalocker==4.3.0"),
        ("always", "pybind11==3.1.0"),
        ("always", "termcolor==3.3.0"),
    ]
)


@dataclass(frozen=True)
class PythonABI:
    """Interpreter identity that partitions every native dependency lock."""

    implementation: str
    version: tuple[int, int]
    cache_tag: str
    abiflags: str
    soabi: str
    platform: str
    machine: str
    pointer_bits: int

    @property
    def lane(self) -> str:
        return f"cp{self.version[0]}{self.version[1]}"

    def payload(self) -> dict[str, object]:
        return {
            "lane": self.lane,
            "implementation": self.implementation,
            "version": list(self.version),
            "cacheTag": self.cache_tag,
            "abiflags": self.abiflags,
            "soabi": self.soabi,
            "platform": self.platform,
            "machine": self.machine,
            "pointerBits": self.pointer_bits,
        }


def _current_python_fingerprint() -> dict[str, object]:
    return {
        "implementation": sys.implementation.name,
        "version": list(sys.version_info[:2]),
        "cache_tag": sys.implementation.cache_tag,
        "abiflags": getattr(sys, "abiflags", ""),
        "soabi": sysconfig.get_config_var("SOABI"),
        "platform": sysconfig.get_platform().lower(),
        "machine": platform.machine().lower(),
        "pointer_bits": struct.calcsize("P") * 8,
    }


def python_abi_from_fingerprint(
    fingerprint: Mapping[str, object],
) -> PythonABI:
    """Validate and normalize one supported 64-bit CPython ABI fingerprint."""

    raw_version = fingerprint.get("version")
    if (
        not isinstance(raw_version, (list, tuple))
        or len(raw_version) != 2
        or any(isinstance(part, bool) or not isinstance(part, int) for part in raw_version)
    ):
        raise DependencyError(
            "PYTHON_ABI_UNSUPPORTED",
            "this release supports only 64-bit CPython 3.11 and 3.12",
        )
    version = (raw_version[0], raw_version[1])
    implementation = str(fingerprint.get("implementation") or "").casefold()
    cache_tag = str(fingerprint.get("cache_tag") or "")
    abiflags = str(fingerprint.get("abiflags") or "")
    soabi = str(fingerprint.get("soabi") or "")
    platform_tag = str(fingerprint.get("platform") or "").strip().casefold()
    machine = str(fingerprint.get("machine") or "").strip().casefold()
    pointer_bits = fingerprint.get("pointer_bits")
    expected_tag = f"cpython-{version[0]}{version[1]}"
    expected_soabi = _SUPPORTED_RELEASE_ABIS.get((version, platform_tag, machine))
    if (
        implementation != "cpython"
        or version not in SUPPORTED_PYTHON_VERSIONS
        or pointer_bits != 64
        or cache_tag != expected_tag
        or abiflags != ""
        or soabi != expected_soabi
    ):
        raise DependencyError(
            "PYTHON_ABI_UNSUPPORTED",
            "this release supports only 64-bit CPython 3.11 and 3.12",
        )
    return PythonABI(
        implementation=implementation,
        version=version,
        cache_tag=cache_tag,
        abiflags=abiflags,
        soabi=soabi,
        platform=platform_tag,
        machine=machine,
        pointer_bits=pointer_bits,
    )


@dataclass(frozen=True)
class PythonRequirementLane:
    """One explicit version-locked closure selected by a Python ABI."""

    name: str
    version: tuple[int, int]
    bootstrap: tuple[str, ...]
    base: tuple[str, ...]
    common_transitive: tuple[str, ...]
    x64_render: tuple[str, ...]
    x64_render_common: tuple[str, ...]
    linux_x64_render: tuple[str, ...]
    windows_x64_render: tuple[str, ...]
    torch_lanes: tuple[tuple[str, tuple[str, str]], ...]
    torch_indexes: tuple[tuple[str, str], ...]
    torch_common: tuple[tuple[str, tuple[str, ...]], ...]
    torch_platform: tuple[
        tuple[tuple[str, str, str], tuple[str, ...]], ...
    ]
    exact_sparse: tuple[str, ...]
    exact_native: tuple[tuple[str, str], ...]
    torch_scatter_links: str

    def torch_requirements_for(self, torch_lane: str, arch: str) -> tuple[str, str]:
        try:
            torch_requirement, vision_requirement = dict(self.torch_lanes)[torch_lane]
        except KeyError as exc:
            raise DependencyError(
                "DEPENDENCY_PLAN_UNSUPPORTED",
                "the selected Python/Torch lane has no audited dependency closure",
            ) from exc
        if arch == "arm64":
            vision_requirement = vision_requirement.split("+", 1)[0]
        return torch_requirement, vision_requirement

    def torch_index_for(self, torch_lane: str) -> str:
        try:
            return dict(self.torch_indexes)[torch_lane]
        except KeyError as exc:
            raise DependencyError(
                "DEPENDENCY_PLAN_UNSUPPORTED",
                "the selected Python/Torch lane has no audited package index",
            ) from exc

    def torch_common_for(self, torch_lane: str) -> tuple[str, ...]:
        try:
            return dict(self.torch_common)[torch_lane]
        except KeyError as exc:
            raise DependencyError(
                "DEPENDENCY_PLAN_UNSUPPORTED",
                "the selected Python/Torch lane has no audited dependency closure",
            ) from exc

    def torch_platform_for(
        self, torch_lane: str, system: str, arch: str
    ) -> tuple[str, ...]:
        try:
            return dict(self.torch_platform)[(torch_lane, system, arch)]
        except KeyError as exc:
            raise DependencyError(
                "DEPENDENCY_PLAN_UNSUPPORTED",
                "the selected Python/OS/architecture/Torch lane has no audited dependency closure",
            ) from exc

    def exact_native_for(self, system: str, install_flash_attn: bool) -> tuple[str, ...]:
        selected = []
        for scope, requirement in self.exact_native:
            if scope == "always" or scope == system or (
                scope == "flash" and install_flash_attn
            ):
                selected.append(requirement)
        return tuple(selected)

    def exact_requirement_for(
        self, name: str, system: str, install_flash_attn: bool
    ) -> str:
        matches = tuple(
            requirement
            for requirement in self.exact_native_for(system, install_flash_attn)
            if _requirement_name(requirement) == name
        )
        if len(matches) != 1:
            raise DependencyError(
                "DEPENDENCY_PLAN_INVALID",
                f"the selected Python lane must lock exactly one {name} requirement",
            )
        return matches[0]


# CPython 3.11 remains the upstream Modly lane.  CPython 3.12 is an explicit
# second lane rather than an expansion of cp311; keeping separate records and
# lock identities lets either closure diverge safely when package metadata does.
PYTHON_REQUIREMENT_LANES = {
    "cp311": PythonRequirementLane(
        name="cp311",
        version=(3, 11),
        bootstrap=BOOTSTRAP_REQUIREMENTS,
        base=BASE_REQUIREMENTS,
        common_transitive=COMMON_TRANSITIVE_REQUIREMENTS,
        x64_render=X64_RENDER_REQUIREMENTS,
        x64_render_common=X64_RENDER_COMMON_TRANSITIVE_REQUIREMENTS,
        linux_x64_render=LINUX_X64_RENDER_TRANSITIVE_REQUIREMENTS,
        windows_x64_render=WINDOWS_X64_RENDER_TRANSITIVE_REQUIREMENTS,
        torch_lanes=tuple(TORCH_LANES.items()),
        torch_indexes=tuple(PYTORCH_INDEXES.items()),
        torch_common=tuple(
            (name, requirements)
            for name, requirements in TORCH_LANE_COMMON_TRANSITIVE_REQUIREMENTS.items()
        ),
        torch_platform=tuple(
            [
                (
                    ("cu124", "linux", "x64"),
                    TORCH_CU124_LINUX_X64_TRANSITIVE_REQUIREMENTS,
                ),
                (("cu124", "win32", "x64"), ()),
                (("cu126", "linux", "arm64"), ()),
                (
                    ("cu128", "linux", "x64"),
                    TORCH_CU128_LINUX_TRANSITIVE_REQUIREMENTS,
                ),
                (
                    ("cu128", "linux", "arm64"),
                    TORCH_CU128_LINUX_TRANSITIVE_REQUIREMENTS,
                ),
                (("cu128", "win32", "x64"), ()),
            ]
        ),
        exact_sparse=EXACT_SPARSE_REQUIREMENTS,
        exact_native=tuple(
            [
                ("always", TORCH_SCATTER_REQUIREMENT),
                ("always", XFORMERS_REQUIREMENT),
                ("linux", LINUX_TRITON_REQUIREMENT),
                ("win32", WINDOWS_TRITON_REQUIREMENT),
                ("flash", FLASH_ATTN_REQUIREMENT),
                *[("always", item) for item in NATIVE_SOURCE_DISTRIBUTION_REQUIREMENTS],
                *[("always", item) for item in EXACT_TRANSITIVE_REQUIREMENTS],
            ]
        ),
        torch_scatter_links=TORCH_SCATTER_LINKS,
    ),
    "cp312": PythonRequirementLane(
        name="cp312",
        version=(3, 12),
        bootstrap=CP312_BOOTSTRAP_REQUIREMENTS,
        base=CP312_BASE_REQUIREMENTS,
        common_transitive=CP312_COMMON_TRANSITIVE_REQUIREMENTS,
        x64_render=CP312_X64_RENDER_REQUIREMENTS,
        x64_render_common=CP312_X64_RENDER_COMMON_TRANSITIVE_REQUIREMENTS,
        linux_x64_render=CP312_LINUX_X64_RENDER_TRANSITIVE_REQUIREMENTS,
        windows_x64_render=CP312_WINDOWS_X64_RENDER_TRANSITIVE_REQUIREMENTS,
        torch_lanes=CP312_TORCH_LANES,
        torch_indexes=CP312_TORCH_INDEXES,
        torch_common=CP312_TORCH_COMMON_TRANSITIVE_REQUIREMENTS,
        torch_platform=tuple(
            [
                (
                    ("cu124", "linux", "x64"),
                    CP312_TORCH_CU124_LINUX_X64_TRANSITIVE_REQUIREMENTS,
                ),
                (("cu124", "win32", "x64"), tuple([])),
                (("cu126", "linux", "arm64"), tuple([])),
                (
                    ("cu128", "linux", "x64"),
                    CP312_TORCH_CU128_LINUX_TRANSITIVE_REQUIREMENTS,
                ),
                (
                    ("cu128", "linux", "arm64"),
                    CP312_TORCH_CU128_LINUX_TRANSITIVE_REQUIREMENTS,
                ),
                (("cu128", "win32", "x64"), tuple([])),
            ]
        ),
        exact_sparse=CP312_EXACT_SPARSE_REQUIREMENTS,
        exact_native=CP312_EXACT_NATIVE_REQUIREMENTS,
        torch_scatter_links=CP312_TORCH_SCATTER_LINKS,
    ),
}


def _python_requirement_lane_for_abi(python_abi: PythonABI) -> PythonRequirementLane:
    try:
        lane = PYTHON_REQUIREMENT_LANES[python_abi.lane]
    except KeyError as exc:
        raise DependencyError(
            "DEPENDENCY_PLAN_UNSUPPORTED",
            "the selected Python ABI has no audited dependency closure",
        ) from exc
    if lane.version != python_abi.version:
        raise DependencyError(
            "DEPENDENCY_PLAN_INVALID",
            "the selected Python ABI does not match its dependency lane",
        )
    return lane


def _python_requirement_lane(plan: "DependencyPlan") -> PythonRequirementLane:
    return _python_requirement_lane_for_abi(plan.python_abi)


@dataclass(frozen=True)
class DependencyPlan:
    profile: str
    system: str
    arch: str
    gpu_sm: int
    torch_lane: str
    torch_requirements: tuple[str, str]
    torch_index: str
    attention_backend: str
    install_flash_attn: bool
    install_native_stack: bool
    support_level: str
    note: str
    python_abi: PythonABI

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["torch_requirements"] = list(self.torch_requirements)
        payload.pop("python_abi")
        payload["python"] = self.python_abi.payload()
        return payload


@dataclass(frozen=True)
class NativeSources:
    root: Path
    nvdiffrast: Path
    cumesh: Path
    flexgemm: Path
    ovoxel: Path


@dataclass(frozen=True)
class PortableCpuSources:
    """Raw pinned inputs consumed by ``materialize_ovoxel_cpu_build``."""

    root: Path
    ovoxel: Path
    eigen: Path


def _environment_value(environment: Mapping[str, str], wanted: str) -> str | None:
    wanted_folded = wanted.casefold()
    return next(
        (
            str(value)
            for key, value in environment.items()
            if str(key).casefold() == wanted_folded and str(value)
        ),
        None,
    )


def _validated_pip_cache(value: str | None) -> str | None:
    if not value or "://" in value:
        return None
    raw = Path(value)
    if not raw.is_absolute():
        return None
    try:
        info = raw.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400
            or not stat.S_ISDIR(info.st_mode)
        ):
            return None
        return str(raw.resolve(strict=True))
    except OSError:
        return None


def _proxy_is_credential_free(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        return bool(
            parsed.scheme
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
    except (TypeError, ValueError):
        return False


def sanitize_subprocess_environment(
    source: Mapping[str, str] | None = None,
    *,
    allow_network: bool = False,
    for_pip: bool = False,
) -> dict[str, str]:
    """Return the minimal non-secret environment allowed into child processes."""

    original = os.environ if source is None else source
    sanitized: dict[str, str] = {}
    for raw_key, raw_value in original.items():
        key = str(raw_key)
        value = str(raw_value)
        upper = key.upper()
        if (
            not key
            or not value
            or _SENSITIVE_ENV_NAME.search(upper)
            or upper.startswith(_CLOUD_ENV_PREFIXES)
            or upper in {"PYTHONHOME", "PYTHONPATH"}
        ):
            continue
        if upper.startswith("PIP_"):
            if upper != "PIP_CACHE_DIR":
                continue
            validated_cache = _validated_pip_cache(value)
            if validated_cache is not None:
                sanitized["PIP_CACHE_DIR"] = validated_cache
            continue
        if allow_network and upper in _NETWORK_ENV_NAMES:
            if upper in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}:
                if not _proxy_is_credential_free(value):
                    continue
            sanitized[key] = value
            continue
        if upper in _SAFE_ENV_NAMES or upper.startswith(_SAFE_ENV_PREFIXES):
            sanitized[key] = value
    if for_pip:
        # pip's --isolated still loads global/site configuration.  Its own
        # documented os.devnull sentinel disables every config-file tier.
        sanitized["PIP_CONFIG_FILE"] = os.devnull
    return sanitized


def _validated_venv_python_path(python: Path) -> Path:
    """Return a canonical venv container path without resolving its Python link."""

    raw = Path(python).expanduser()
    if not raw.is_absolute():
        raise DependencyError(
            "PYTHON_PATH_INVALID",
            "the extension virtualenv Python path must be absolute",
        )
    scripts_name = raw.parent.name
    executable_name = raw.name
    valid_layout = (
        scripts_name == "bin" and executable_name == "python"
    ) or (
        scripts_name.casefold() == "scripts"
        and executable_name.casefold() == "python.exe"
    )
    if not valid_layout:
        raise DependencyError(
            "PYTHON_PATH_INVALID",
            "the Python executable is outside a supported virtualenv layout",
        )

    venv_root = raw.parent.parent
    try:
        root_info = venv_root.lstat()
        scripts_info = raw.parent.lstat()
        if (
            stat.S_ISLNK(root_info.st_mode)
            or getattr(root_info, "st_file_attributes", 0) & 0x400
            or not stat.S_ISDIR(root_info.st_mode)
            or stat.S_ISLNK(scripts_info.st_mode)
            or getattr(scripts_info, "st_file_attributes", 0) & 0x400
            or not stat.S_ISDIR(scripts_info.st_mode)
        ):
            raise OSError("virtualenv container is aliased or not a directory")
        canonical_root = venv_root.resolve(strict=True)
        canonical_scripts = raw.parent.resolve(strict=True)
        if canonical_scripts.parent != canonical_root:
            raise OSError("virtualenv scripts directory escaped its container")

        config = canonical_root / "pyvenv.cfg"
        config_info = config.lstat()
        if (
            stat.S_ISLNK(config_info.st_mode)
            or getattr(config_info, "st_file_attributes", 0) & 0x400
            or not stat.S_ISREG(config_info.st_mode)
        ):
            raise OSError("virtualenv configuration is not a regular file")

        logical_python = canonical_scripts / executable_name
        python_info = logical_python.lstat()
        python_is_link = stat.S_ISLNK(python_info.st_mode)
        if (
            getattr(python_info, "st_file_attributes", 0) & 0x400
            or (not python_is_link and not stat.S_ISREG(python_info.st_mode))
            or not logical_python.is_file()
        ):
            raise OSError("virtualenv Python is unavailable")
    except OSError as exc:
        raise DependencyError(
            "PYTHON_PATH_INVALID",
            "the extension virtualenv Python path is unsafe or incomplete",
        ) from exc
    # Linux venvs intentionally use bin/python -> the base interpreter.  Keep
    # that final logical component so CPython discovers the adjacent pyvenv.cfg.
    return logical_python


def _safe_installed_relative(value: str) -> PurePosixPath:
    relative = PurePosixPath(str(value))
    if (
        not str(value)
        or relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or "\\" in str(value)
        or "\x00" in str(value)
    ):
        raise DependencyError(
            "CUSPARSELT_METADATA_INVALID",
            "cuSPARSELt installed metadata contains an unsafe path",
        )
    return relative


def _record_sha256_value(digest_hex: str) -> str:
    try:
        digest = bytes.fromhex(digest_hex)
    except ValueError as exc:
        raise DependencyError(
            "CUSPARSELT_CONTRACT_INVALID",
            "the audited cuSPARSELt file digest is malformed",
        ) from exc
    if len(digest) != hashlib.sha256().digest_size:
        raise DependencyError(
            "CUSPARSELT_CONTRACT_INVALID",
            "the audited cuSPARSELt file digest is malformed",
        )
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _expected_cusparselt_record(
    contract: _CusparseLtNormalizationContract, *, normalized: bool
) -> bytes:
    fixed = {relative: (digest, size) for relative, digest, size in contract.fixed_files}
    if len(fixed) != len(contract.fixed_files):
        raise DependencyError(
            "CUSPARSELT_CONTRACT_INVALID",
            "the audited cuSPARSELt file set contains duplicates",
        )
    wheel = contract.normalized_wheel if normalized else contract.original_wheel
    rows: list[str] = []
    for relative in contract.record_order:
        _safe_installed_relative(relative)
        if relative == contract.record_relative:
            rows.append(f"{relative},,")
        else:
            if relative == contract.wheel_relative:
                digest = hashlib.sha256(wheel).hexdigest()
                size = len(wheel)
            else:
                try:
                    digest, size = fixed[relative]
                except KeyError as exc:
                    raise DependencyError(
                        "CUSPARSELT_CONTRACT_INVALID",
                        "the audited cuSPARSELt RECORD file set is incomplete",
                    ) from exc
            rows.append(
                f"{relative},sha256={_record_sha256_value(digest)},{size}"
            )
    return ("\r\n".join(rows) + "\r\n").encode("ascii")


def _validate_cusparselt_contract(
    contract: _CusparseLtNormalizationContract,
) -> tuple[bytes, bytes]:
    paths = tuple(contract.record_order)
    fixed_paths = tuple(relative for relative, _digest, _size in contract.fixed_files)
    expected_paths = set(fixed_paths) | {
        contract.wheel_relative,
        contract.record_relative,
    }
    for relative in (
        contract.dist_info,
        contract.package_root,
        contract.library_relative,
        contract.metadata_relative,
        contract.wheel_relative,
        contract.record_relative,
        *paths,
        *fixed_paths,
    ):
        _safe_installed_relative(relative)
    if (
        len(paths) != len(set(paths))
        or set(paths) != expected_paths
        or contract.record_relative not in paths
        or contract.wheel_relative not in paths
        or contract.metadata_relative not in fixed_paths
        or contract.library_relative not in fixed_paths
        or contract.original_wheel.count(b"manylinux2014_sbsa") != 1
        or contract.normalized_wheel
        != contract.original_wheel.replace(
            b"manylinux2014_sbsa", b"manylinux2014_aarch64"
        )
    ):
        raise DependencyError(
            "CUSPARSELT_CONTRACT_INVALID",
            "the audited cuSPARSELt normalization contract is inconsistent",
        )
    original_record = _expected_cusparselt_record(contract, normalized=False)
    normalized_record = _expected_cusparselt_record(contract, normalized=True)
    if (
        hashlib.sha256(original_record).hexdigest()
        != contract.original_record_sha256
        or hashlib.sha256(normalized_record).hexdigest()
        != contract.normalized_record_sha256
    ):
        raise DependencyError(
            "CUSPARSELT_CONTRACT_INVALID",
            "the audited cuSPARSELt RECORD identities are inconsistent",
        )
    return original_record, normalized_record


def _require_linux_fd_security() -> None:
    required_dir_fd = (os.open, os.stat, os.unlink)
    if (
        not sys.platform.startswith("linux")
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or any(function not in os.supports_dir_fd for function in required_dir_fd)
        or os.stat not in os.supports_follow_symlinks
        or not Path("/proc/self/fd").is_dir()
    ):
        raise DependencyError(
            "CUSPARSELT_FD_SECURITY_UNAVAILABLE",
            "the audited cuSPARSELt correction requires Linux openat and /proc file-descriptor semantics",
        )


def _stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_size,
        getattr(info, "st_mtime_ns", 0),
        getattr(info, "st_ctime_ns", 0),
        getattr(info, "st_nlink", 1),
    )


def _same_inode(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev == second.st_dev
        and first.st_ino == second.st_ino
        and stat.S_IFMT(first.st_mode) == stat.S_IFMT(second.st_mode)
    )


def _validate_directory_info(info: os.stat_result) -> None:
    if not stat.S_ISDIR(info.st_mode):
        raise OSError("entry is not a directory")


def _validate_regular_info(info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_nlink", 1) != 1:
        raise OSError("entry is not an owned single-link regular file")


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _file_flags() -> int:
    return os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_directory_path(path: Path) -> int:
    descriptor: int | None = None
    try:
        before = path.lstat()
        _validate_directory_info(before)
        descriptor = os.open(path, _directory_flags())
        opened = os.fstat(descriptor)
        after = path.lstat()
        _validate_directory_info(opened)
        if not _same_inode(before, opened) or not _same_inode(opened, after):
            raise OSError("directory changed while opened")
        return descriptor
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _safe_leaf_name(name: str) -> str:
    if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise OSError("unsafe directory entry name")
    return name


def _open_directory_at(parent_descriptor: int, name: str) -> int:
    descriptor: int | None = None
    name = _safe_leaf_name(name)
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        _validate_directory_info(before)
        descriptor = os.open(
            name, _directory_flags(), dir_fd=parent_descriptor
        )
        opened = os.fstat(descriptor)
        after = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        _validate_directory_info(opened)
        if not _same_inode(before, opened) or not _same_inode(opened, after):
            raise OSError("directory changed while opened")
        return descriptor
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _open_regular_at(parent_descriptor: int, name: str) -> int:
    descriptor: int | None = None
    name = _safe_leaf_name(name)
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        _validate_regular_info(before)
        descriptor = os.open(name, _file_flags(), dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        after = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        _validate_regular_info(opened)
        if (
            _stat_identity(before) != _stat_identity(opened)
            or _stat_identity(opened) != _stat_identity(after)
        ):
            raise OSError("file changed while opened")
        return descriptor
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _verify_directory_binding(
    parent_descriptor: int, name: str, descriptor: int
) -> None:
    linked = os.stat(
        _safe_leaf_name(name),
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    opened = os.fstat(descriptor)
    _validate_directory_info(linked)
    _validate_directory_info(opened)
    if not _same_inode(linked, opened):
        raise OSError("directory binding changed")


@dataclass(frozen=True)
class _OpenFileSnapshot:
    relative: str
    name: str
    parent_descriptor: int
    descriptor: int
    identity: tuple[int, ...]
    digest: str
    size: int
    prefix: bytes
    data: bytes | None


def _snapshot_open_file(
    descriptor: int,
    parent_descriptor: int,
    name: str,
    relative: str,
    *,
    capture_limit: int | None,
) -> _OpenFileSnapshot:
    before = os.fstat(descriptor)
    _validate_regular_info(before)
    if capture_limit is not None and before.st_size > capture_limit:
        raise OSError("captured metadata file exceeds its safety limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    prefix = bytearray()
    captured = bytearray() if capture_limit is not None else None
    total = 0
    while True:
        block = os.read(descriptor, DOWNLOAD_CHUNK)
        if not block:
            break
        total += len(block)
        digest.update(block)
        if len(prefix) < 20:
            prefix.extend(block[: 20 - len(prefix)])
        if captured is not None:
            captured.extend(block)
            if len(captured) > capture_limit:
                raise OSError("captured metadata file exceeds its safety limit")
    after = os.fstat(descriptor)
    _validate_regular_info(after)
    if _stat_identity(before) != _stat_identity(after) or total != before.st_size:
        raise OSError("file changed while hashed")
    return _OpenFileSnapshot(
        relative=relative,
        name=name,
        parent_descriptor=parent_descriptor,
        descriptor=descriptor,
        identity=_stat_identity(before),
        digest=digest.hexdigest(),
        size=before.st_size,
        prefix=bytes(prefix),
        data=bytes(captured) if captured is not None else None,
    )


def _verify_file_snapshot(snapshot: _OpenFileSnapshot) -> None:
    linked = os.stat(
        snapshot.name,
        dir_fd=snapshot.parent_descriptor,
        follow_symlinks=False,
    )
    opened = os.fstat(snapshot.descriptor)
    _validate_regular_info(linked)
    _validate_regular_info(opened)
    if (
        _stat_identity(linked) != snapshot.identity
        or _stat_identity(opened) != snapshot.identity
    ):
        raise OSError("file binding changed after hashing")


@dataclass
class _FdInventory:
    files: dict[str, _OpenFileSnapshot]
    directories: tuple[str, ...]
    directory_bindings: tuple[tuple[int, str, int], ...]
    descriptors: tuple[int, ...]

    def revalidate(self) -> None:
        for parent_descriptor, name, descriptor in self.directory_bindings:
            _verify_directory_binding(parent_descriptor, name, descriptor)
        for snapshot in self.files.values():
            _verify_file_snapshot(snapshot)

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _inventory_open_tree(
    root_descriptor: int,
    root_relative: str,
    *,
    capture_limits: Mapping[str, int],
) -> _FdInventory:
    files: dict[str, _OpenFileSnapshot] = {}
    directories: set[str] = set()
    bindings: list[tuple[int, str, int]] = []
    descriptors: list[int] = []

    def walk(descriptor: int, relative_root: str) -> None:
        names = os.listdir(descriptor)
        if len(names) != len(set(names)):
            raise OSError("directory contains duplicate names")
        for raw_name in sorted(names):
            name = _safe_leaf_name(str(raw_name))
            relative = f"{relative_root}/{name}"
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child = _open_directory_at(descriptor, name)
                descriptors.append(child)
                bindings.append((descriptor, name, child))
                directories.add(relative)
                walk(child, relative)
            elif stat.S_ISREG(info.st_mode):
                child = _open_regular_at(descriptor, name)
                descriptors.append(child)
                files[relative] = _snapshot_open_file(
                    child,
                    descriptor,
                    name,
                    relative,
                    capture_limit=capture_limits.get(relative),
                )
            else:
                raise OSError("tree contains an alias or special entry")

    try:
        walk(root_descriptor, root_relative)
        inventory = _FdInventory(
            files=files,
            directories=tuple(sorted(directories)),
            directory_bindings=tuple(bindings),
            descriptors=tuple(descriptors),
        )
        inventory.revalidate()
        return inventory
    except OSError:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


@dataclass
class _CusparseLtOpenTree:
    venv_root: Path
    root_descriptor: int
    site_descriptor: int
    package_descriptor: int
    dist_descriptor: int
    directory_bindings: tuple[tuple[int, str, int], ...]
    descriptors: tuple[int, ...]

    def revalidate(self) -> None:
        linked_root = self.venv_root.lstat()
        opened_root = os.fstat(self.root_descriptor)
        _validate_directory_info(linked_root)
        _validate_directory_info(opened_root)
        if not _same_inode(linked_root, opened_root):
            raise OSError("virtualenv root binding changed")
        for parent_descriptor, name, descriptor in self.directory_bindings:
            _verify_directory_binding(parent_descriptor, name, descriptor)

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _open_cusparselt_tree(
    python: Path,
    plan: DependencyPlan,
    contract: _CusparseLtNormalizationContract,
) -> _CusparseLtOpenTree:
    _require_linux_fd_security()
    logical_python = _validated_venv_python_path(Path(python))
    venv_root = logical_python.parent.parent
    descriptors: list[int] = []
    bindings: list[tuple[int, str, int]] = []
    try:
        root_descriptor = _open_directory_path(venv_root)
        descriptors.append(root_descriptor)
        current = root_descriptor
        for name in (
            "lib",
            f"python{plan.python_abi.version[0]}.{plan.python_abi.version[1]}",
            "site-packages",
        ):
            child = _open_directory_at(current, name)
            descriptors.append(child)
            bindings.append((current, name, child))
            current = child
        site_descriptor = current

        expected_dist_folded = contract.dist_info.casefold().replace("_", "-")
        distribution_prefix = f"{contract.distribution.casefold()}-"
        candidates = [
            name
            for name in os.listdir(site_descriptor)
            if str(name).casefold().replace("_", "-").startswith(distribution_prefix)
            and str(name).casefold().endswith(".dist-info")
        ]
        if (
            len(candidates) != 1
            or str(candidates[0]).casefold().replace("_", "-")
            != expected_dist_folded
            or str(candidates[0]) != contract.dist_info
        ):
            raise DependencyError(
                "CUSPARSELT_METADATA_INVALID",
                "the installed cuSPARSELt distribution identity is not unique and exact",
            )

        current = site_descriptor
        for name in _safe_installed_relative(contract.package_root).parts:
            child = _open_directory_at(current, name)
            descriptors.append(child)
            bindings.append((current, name, child))
            current = child
        package_descriptor = current
        dist_parts = _safe_installed_relative(contract.dist_info).parts
        if len(dist_parts) != 1:
            raise DependencyError(
                "CUSPARSELT_CONTRACT_INVALID",
                "the audited cuSPARSELt dist-info directory is not a direct child",
            )
        dist_descriptor = _open_directory_at(site_descriptor, dist_parts[0])
        descriptors.append(dist_descriptor)
        bindings.append((site_descriptor, dist_parts[0], dist_descriptor))
        tree = _CusparseLtOpenTree(
            venv_root=venv_root,
            root_descriptor=root_descriptor,
            site_descriptor=site_descriptor,
            package_descriptor=package_descriptor,
            dist_descriptor=dist_descriptor,
            directory_bindings=tuple(bindings),
            descriptors=tuple(descriptors),
        )
        tree.revalidate()
        return tree
    except DependencyError:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    except OSError as exc:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise DependencyError(
            "CUSPARSELT_TREE_UNSAFE",
            "the installed cuSPARSELt tree could not be opened without following aliases",
        ) from exc


def _parse_cusparselt_record(
    encoded: bytes,
    contract: _CusparseLtNormalizationContract,
) -> dict[str, tuple[str, str]]:
    try:
        text = encoded.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeError, csv.Error) as exc:
        raise DependencyError(
            "CUSPARSELT_METADATA_INVALID",
            "the installed cuSPARSELt RECORD is malformed",
        ) from exc
    parsed: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            raise DependencyError(
                "CUSPARSELT_METADATA_INVALID",
                "the installed cuSPARSELt RECORD is malformed",
            )
        relative, digest, size = row
        _safe_installed_relative(relative)
        if relative in parsed:
            raise DependencyError(
                "CUSPARSELT_METADATA_INVALID",
                "the installed cuSPARSELt RECORD contains duplicate paths",
            )
        parsed[relative] = (digest, size)
    if set(parsed) != set(contract.record_order):
        raise DependencyError(
            "CUSPARSELT_METADATA_INVALID",
            "the installed cuSPARSELt RECORD file set is not the audited set",
        )
    return parsed


def _metadata_header_values(encoded: bytes, name: str) -> list[str]:
    try:
        text = encoded.decode("utf-8")
    except UnicodeError as exc:
        raise DependencyError(
            "CUSPARSELT_METADATA_INVALID",
            "the installed cuSPARSELt METADATA is not UTF-8",
        ) from exc
    prefix = f"{name}: "
    values: list[str] = []
    for line in text.splitlines():
        if not line:
            break
        if line.startswith(prefix):
            values.append(line[len(prefix) :])
    return values


def _validate_aarch64_elf(
    library: _OpenFileSnapshot, loader: Callable[[str], object]
) -> None:
    try:
        before = os.fstat(library.descriptor)
        _validate_regular_info(before)
        if _stat_identity(before) != library.identity or library.size < 20:
            raise OSError("library changed after hashing")
        header = library.prefix
        if header[:4] != b"\x7fELF" or header[4] != 2 or header[5] not in {1, 2}:
            raise OSError("library is not ELF64")
        byte_order = "little" if header[5] == 1 else "big"
        if int.from_bytes(header[18:20], byte_order) != 183:
            raise OSError("library is not AArch64")
        descriptor_path = f"/proc/self/fd/{library.descriptor}"
        proc_info = os.stat(descriptor_path)
        if not _same_inode(before, proc_info):
            raise OSError("proc descriptor does not reference the hashed library")
        loaded = loader(descriptor_path)
        after = os.fstat(library.descriptor)
        proc_after = os.stat(descriptor_path)
        _validate_regular_info(after)
        if (
            _stat_identity(after) != library.identity
            or not _same_inode(after, proc_after)
        ):
            raise OSError("library changed while loaded")
        del loaded
    except (OSError, ValueError) as exc:
        raise DependencyError(
            "CUSPARSELT_LIBRARY_INVALID",
            "the installed cuSPARSELt library is not a loadable AArch64 ELF64 object",
        ) from exc


def _inspect_cusparselt_metadata(
    tree: _CusparseLtOpenTree,
    plan: DependencyPlan,
    *,
    loader: Callable[[str], object],
) -> tuple[str, dict[str, object], bytes, bytes]:
    identity = _cusparselt_normalization_identity(plan)
    if identity is None:
        raise DependencyError(
            "CUSPARSELT_NORMALIZATION_UNSUPPORTED",
            "the cuSPARSELt correction is not applicable to this dependency plan",
        )
    contract = _CUSPARSELT_NORMALIZATION_CONTRACT
    original_record, normalized_record = _validate_cusparselt_contract(contract)
    fixed_by_path = {
        relative: (digest, size)
        for relative, digest, size in contract.fixed_files
    }
    capture_limits = {
        contract.wheel_relative: max(
            len(contract.original_wheel), len(contract.normalized_wheel)
        ),
        contract.record_relative: STATE_MAX_BYTES,
        contract.metadata_relative: fixed_by_path[contract.metadata_relative][1],
    }
    package_inventory: _FdInventory | None = None
    dist_inventory: _FdInventory | None = None
    try:
        package_inventory = _inventory_open_tree(
            tree.package_descriptor,
            contract.package_root,
            capture_limits=capture_limits,
        )
        dist_inventory = _inventory_open_tree(
            tree.dist_descriptor,
            contract.dist_info,
            capture_limits=capture_limits,
        )
        files = {**package_inventory.files, **dist_inventory.files}
        expected_package_directories = tuple(
            sorted(
                f"{contract.package_root}/{relative}"
                for relative in contract.package_directories
            )
        )
        if (
            set(files) != set(contract.record_order)
            or package_inventory.directories != expected_package_directories
            or dist_inventory.directories
        ):
            raise DependencyError(
                "CUSPARSELT_FILE_SET_INVALID",
                "the installed cuSPARSELt file set is not the audited wheel file set",
            )

        wheel = files[contract.wheel_relative].data
        record = files[contract.record_relative].data
        metadata = files[contract.metadata_relative].data
        if wheel is None or record is None or metadata is None:
            raise DependencyError(
                "CUSPARSELT_TREE_UNSAFE",
                "the installed cuSPARSELt metadata files could not be captured from their audited descriptors",
            )
        parsed_record = _parse_cusparselt_record(record, contract)
        if wheel == contract.original_wheel and record == original_record:
            state = "original"
        elif wheel == contract.normalized_wheel and record == normalized_record:
            state = "normalized"
        else:
            raise DependencyError(
                "CUSPARSELT_STATE_INVALID",
                "the installed cuSPARSELt metadata is hybrid or not an exact audited state",
            )

        for relative in contract.record_order:
            digest_field, size_field = parsed_record[relative]
            if relative == contract.record_relative:
                if digest_field or size_field:
                    raise DependencyError(
                        "CUSPARSELT_METADATA_INVALID",
                        "the installed cuSPARSELt RECORD row must be unhashed",
                    )
                continue
            snapshot = files[relative]
            expected_digest_field = (
                f"sha256={_record_sha256_value(snapshot.digest)}"
            )
            if digest_field != expected_digest_field or size_field != str(snapshot.size):
                raise DependencyError(
                    "CUSPARSELT_METADATA_INVALID",
                    "the installed cuSPARSELt RECORD hash or size does not match its file",
                )

        for relative, expected_digest, expected_size in contract.fixed_files:
            snapshot = files[relative]
            if snapshot.digest != expected_digest or snapshot.size != expected_size:
                raise DependencyError(
                    "CUSPARSELT_FILE_HASH_MISMATCH",
                    "an installed cuSPARSELt wheel file does not match the audited identity",
                )
        if (
            _metadata_header_values(metadata, "Name") != [contract.distribution]
            or _metadata_header_values(metadata, "Version") != [contract.version]
        ):
            raise DependencyError(
                "CUSPARSELT_METADATA_INVALID",
                "the installed cuSPARSELt METADATA name or version is not exact",
            )
        _validate_aarch64_elf(files[contract.library_relative], loader)
        package_inventory.revalidate()
        dist_inventory.revalidate()
        tree.revalidate()
        report = {
            **identity,
            "state": state,
            "applied": False,
            "wheelSha256": files[contract.wheel_relative].digest,
            "recordSha256": files[contract.record_relative].digest,
        }
        return state, report, original_record, normalized_record
    except DependencyError:
        raise
    except (KeyError, OSError) as exc:
        raise DependencyError(
            "CUSPARSELT_TREE_UNSAFE",
            "the installed cuSPARSELt tree changed while it was inspected",
        ) from exc
    finally:
        if package_inventory is not None:
            package_inventory.close()
        if dist_inventory is not None:
            dist_inventory.close()


@dataclass
class _StagedLeaf:
    name: str
    descriptor: int
    identity: tuple[int, ...]
    published: bool = False


def _write_fsynced_temporary_at(
    directory_descriptor: int, label: str, encoded: bytes
) -> _StagedLeaf:
    name = f".{_safe_leaf_name(label)}.{uuid.uuid4().hex}.tmp"
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, 0o644, dir_fd=directory_descriptor)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("temporary metadata write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        _validate_regular_info(opened)
        _validate_regular_info(linked)
        if (
            opened.st_size != len(encoded)
            or _stat_identity(opened) != _stat_identity(linked)
        ):
            raise OSError("temporary metadata identity changed")
        return _StagedLeaf(name, descriptor, _stat_identity(opened))
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(name, dir_fd=directory_descriptor)
        except OSError:
            pass
        raise DependencyError(
            "CUSPARSELT_WRITE_FAILED",
            "the normalized cuSPARSELt metadata could not be staged safely",
        ) from exc


def _dist_info_leaf(contract_relative: str, contract: _CusparseLtNormalizationContract) -> str:
    relative = _safe_installed_relative(contract_relative)
    if relative.parts[:-1] != (contract.dist_info,):
        raise DependencyError(
            "CUSPARSELT_CONTRACT_INVALID",
            "the audited cuSPARSELt metadata leaf is outside dist-info",
        )
    return _safe_leaf_name(relative.name)


def _read_leaf_bytes_at(
    directory_descriptor: int, name: str, *, max_bytes: int
) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = _open_regular_at(directory_descriptor, name)
        snapshot = _snapshot_open_file(
            descriptor,
            directory_descriptor,
            name,
            name,
            capture_limit=max_bytes,
        )
        _verify_file_snapshot(snapshot)
        if snapshot.data is None:
            raise OSError("metadata leaf was not captured")
        return snapshot.data
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verify_published_leaf(
    directory_descriptor: int,
    target_name: str,
    staged: _StagedLeaf,
    expected: bytes,
) -> None:
    linked = os.stat(
        target_name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
    opened = os.fstat(staged.descriptor)
    _validate_regular_info(linked)
    _validate_regular_info(opened)
    if (
        not _same_inode(linked, opened)
        or (opened.st_dev, opened.st_ino, stat.S_IFMT(opened.st_mode), opened.st_size)
        != staged.identity[:4]
    ):
        raise OSError("published metadata leaf does not match its staged inode")
    snapshot = _snapshot_open_file(
        staged.descriptor,
        directory_descriptor,
        target_name,
        target_name,
        capture_limit=len(expected),
    )
    _verify_file_snapshot(snapshot)
    if snapshot.data != expected:
        raise OSError("published metadata leaf bytes changed")


def _cleanup_staged_leaf(directory_descriptor: int, staged: _StagedLeaf | None) -> None:
    if staged is None:
        return
    try:
        os.close(staged.descriptor)
    except OSError:
        pass
    if not staged.published:
        try:
            os.unlink(staged.name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _publish_cusparselt_metadata(
    tree: _CusparseLtOpenTree,
    contract: _CusparseLtNormalizationContract,
    original_record: bytes,
    normalized_record: bytes,
) -> None:
    wheel_name = _dist_info_leaf(contract.wheel_relative, contract)
    record_name = _dist_info_leaf(contract.record_relative, contract)
    wheel_staged: _StagedLeaf | None = None
    record_staged: _StagedLeaf | None = None
    try:
        tree.revalidate()
        wheel_staged = _write_fsynced_temporary_at(
            tree.dist_descriptor, "WHEEL", contract.normalized_wheel
        )
        record_staged = _write_fsynced_temporary_at(
            tree.dist_descriptor, "RECORD", normalized_record
        )
        tree.revalidate()
        if (
            _read_leaf_bytes_at(
                tree.dist_descriptor,
                wheel_name,
                max_bytes=len(contract.original_wheel),
            )
            != contract.original_wheel
            or _read_leaf_bytes_at(
                tree.dist_descriptor,
                record_name,
                max_bytes=len(original_record),
            )
            != original_record
        ):
            raise DependencyError(
                "CUSPARSELT_STATE_CHANGED",
                "cuSPARSELt metadata changed while normalization was staged",
            )

        os.replace(
            wheel_staged.name,
            wheel_name,
            src_dir_fd=tree.dist_descriptor,
            dst_dir_fd=tree.dist_descriptor,
        )
        wheel_staged.published = True
        _verify_published_leaf(
            tree.dist_descriptor,
            wheel_name,
            wheel_staged,
            contract.normalized_wheel,
        )
        os.fsync(tree.dist_descriptor)
        tree.revalidate()

        os.replace(
            record_staged.name,
            record_name,
            src_dir_fd=tree.dist_descriptor,
            dst_dir_fd=tree.dist_descriptor,
        )
        record_staged.published = True
        _verify_published_leaf(
            tree.dist_descriptor,
            record_name,
            record_staged,
            normalized_record,
        )
        os.fsync(tree.dist_descriptor)
        tree.revalidate()
    except DependencyError:
        raise
    except (OSError, TypeError) as exc:
        raise DependencyError(
            "CUSPARSELT_WRITE_FAILED",
            "the normalized cuSPARSELt metadata could not be published through its pinned directory descriptor",
        ) from exc
    finally:
        _cleanup_staged_leaf(tree.dist_descriptor, wheel_staged)
        _cleanup_staged_leaf(tree.dist_descriptor, record_staged)


def normalize_cusparselt_metadata(
    python: Path,
    plan: DependencyPlan,
    *,
    _cdll_loader: Callable[[str], object] = ctypes.CDLL,
) -> dict[str, object] | None:
    """Normalize the one audited installed ARM64 wheel; never touch caches."""

    if _cusparselt_normalization_identity(plan) is None:
        return None
    contract = _CUSPARSELT_NORMALIZATION_CONTRACT
    tree = _open_cusparselt_tree(python, plan, contract)
    try:
        state, report, original_record, normalized_record = (
            _inspect_cusparselt_metadata(tree, plan, loader=_cdll_loader)
        )
        if state == "normalized":
            return report
        if tree.venv_root.name != "venv.__modly_staging":
            raise DependencyError(
                "CUSPARSELT_NOT_STAGING",
                "cuSPARSELt metadata may be changed only inside the setup staging virtualenv",
            )
        _publish_cusparselt_metadata(
            tree, contract, original_record, normalized_record
        )
        final_state, final_report, *_rest = _inspect_cusparselt_metadata(
            tree, plan, loader=_cdll_loader
        )
        if final_state != "normalized":
            raise DependencyError(
                "CUSPARSELT_POSTCONDITION_FAILED",
                "the normalized cuSPARSELt metadata did not reach its exact final state",
            )
        final_report["applied"] = True
        return final_report
    finally:
        tree.close()


def _verify_cusparselt_metadata(
    python: Path,
    plan: DependencyPlan,
    *,
    _cdll_loader: Callable[[str], object] = ctypes.CDLL,
) -> dict[str, object] | None:
    if _cusparselt_normalization_identity(plan) is None:
        return None
    contract = _CUSPARSELT_NORMALIZATION_CONTRACT
    tree = _open_cusparselt_tree(python, plan, contract)
    try:
        state, report, *_rest = _inspect_cusparselt_metadata(
            tree, plan, loader=_cdll_loader
        )
        if state != "normalized":
            raise DependencyError(
                "CUSPARSELT_NOT_NORMALIZED",
                "the installed cuSPARSELt metadata is not in its required final state",
            )
        return report
    finally:
        tree.close()


def isolated_pip_command(
    python: Path,
    arguments: Sequence[str],
    environment: Mapping[str, str],
) -> list[str]:
    """Construct pip without user config/index overrides, retaining Modly cache."""

    command = [
        str(_validated_venv_python_path(Path(python))),
        "-m",
        "pip",
        "--isolated",
        "--disable-pip-version-check",
        "--no-input",
    ]
    cache = _validated_pip_cache(
        _environment_value(environment, "PIP_CACHE_DIR")
    )
    if cache is not None:
        command.extend(("--cache-dir", cache))
    command.extend(str(argument) for argument in arguments)
    return command


def _normalize_system(value: object) -> str:
    raw = str(value or sys.platform).strip().casefold()
    if raw.startswith("win"):
        return "win32"
    if raw.startswith("linux"):
        return "linux"
    return raw


def _normalize_arch(value: object) -> str:
    raw = str(value or platform.machine()).strip().casefold().replace("-", "_")
    if raw in {"amd64", "x86_64", "x64"}:
        return "x64"
    if raw in {"aarch64", "arm64"}:
        return "arm64"
    return raw


def _normalize_profile(value: object) -> str:
    raw = str(value or "auto").strip().casefold().replace("_", "-")
    aliases = {"exact": "exact-upstream", "upstream": "exact-upstream"}
    raw = aliases.get(raw, raw)
    if raw not in {"auto", "exact-upstream", "portable"}:
        raise DependencyError(
            "DEPENDENCY_PROFILE_INVALID",
            "dependency profile must be auto, exact-upstream, or portable",
        )
    return raw


def select_dependency_plan(
    context: Mapping[str, object],
    requested_profile: str | None = None,
    *,
    interpreter_fingerprint: Mapping[str, object] | None = None,
) -> DependencyPlan:
    """Resolve a deterministic dependency lane from Modly setup metadata."""

    python_abi = python_abi_from_fingerprint(
        _current_python_fingerprint()
        if interpreter_fingerprint is None
        else interpreter_fingerprint
    )
    python_lane = _python_requirement_lane_for_abi(python_abi)
    system = _normalize_system(context.get("platform"))
    arch = _normalize_arch(context.get("arch"))
    accelerator = str(context.get("accelerator") or "").strip().casefold()
    gpu_sm = int(context.get("gpu_sm") or 0)
    try:
        cuda_driver_hint = int(float(str(context.get("cuda_version") or 0)) * (10 if "." in str(context.get("cuda_version") or "") else 1))
    except (TypeError, ValueError):
        raise DependencyError("CUDA_VERSION_INVALID", "cuda_version metadata is invalid")
    profile = _normalize_profile(
        requested_profile
        or context.get("dependency_profile")
        or os.environ.get("MODLY_LATO2_DEPENDENCY_PROFILE")
        or "auto"
    )

    if system not in {"linux", "win32"}:
        raise DependencyError(
            "PLATFORM_UNSUPPORTED", "LATO.2 supports Windows and Linux setup only"
        )
    if arch not in {"x64", "arm64"} or (system == "win32" and arch != "x64"):
        raise DependencyError(
            "ARCH_UNSUPPORTED", "this operating-system and architecture pair is unsupported"
        )
    if accelerator not in {"cuda", "nvidia"} or gpu_sm <= 0:
        raise DependencyError(
            "CUDA_REQUIRED", "LATO.2 inference requires a detected NVIDIA CUDA GPU"
        )

    if profile == "auto":
        # Windows defaults to the self-contained compatibility route.  The
        # complete upstream CUDA build remains an explicit, toolchain-dependent
        # opt-in there; Linux x64 keeps the exact upstream default.
        profile = (
            "portable"
            if system == "win32" or arch == "arm64" or gpu_sm < 80 or gpu_sm >= 100
            else "exact-upstream"
        )

    if profile == "exact-upstream":
        if arch != "x64":
            raise DependencyError(
                "EXACT_ARM64_UNAVAILABLE",
                "the exact upstream stack has no coherent published ARM64 wheel set; use the separately fingerprinted portable profile",
            )
        if gpu_sm >= 100:
            raise DependencyError(
                "EXACT_GPU_ARCH_UNSUPPORTED",
                "Torch 2.6/cu124 does not contain kernels for this GPU architecture; use the portable cu128 profile",
            )
        if gpu_sm < 80:
            raise DependencyError(
                "EXACT_BF16_UNSUPPORTED",
                "the pinned upstream scripts require BF16, which this GPU cannot execute; use the portable FP16 profile",
            )
        if cuda_driver_hint and cuda_driver_hint < 124:
            raise DependencyError(
                "DRIVER_CUDA_TOO_OLD",
                f"the exact cu124 runtime needs an NVIDIA driver compatible with CUDA 12.4 (Modly reports {cuda_driver_hint})",
            )
        use_flash = system == "linux" and gpu_sm >= 80
        return DependencyPlan(
            profile=profile,
            system=system,
            arch=arch,
            gpu_sm=gpu_sm,
            torch_lane="cu124",
            torch_requirements=python_lane.torch_requirements_for("cu124", arch),
            torch_index=python_lane.torch_index_for("cu124"),
            attention_backend="flash_attn" if use_flash else "xformers",
            install_flash_attn=use_flash,
            install_native_stack=True,
            support_level="toolchain-dependent",
            note=(
                "Complete pinned upstream dependency graph; native extensions are built locally."
            ),
            python_abi=python_abi,
        )

    # Portable is deliberately separate: Modly compatibility overlays replace
    # spconv, torch-scatter, Open3D rendering and the native o_voxel import tree.
    # Their validation belongs to the extension runtime, not this exact stack.
    if gpu_sm >= 100:
        # Modly 0.4.2 reports 128 for every driver major >=570 and does not
        # expose the raw driver version.  Selecting cu130 from that capped hint
        # would be unsafe: CUDA 13 wheels require a newer driver than many
        # valid 12.8 hosts.  The separate official cu128 CPython 3.11 and 3.12
        # wheels contain the required Blackwell kernels on Linux ARM64/x64 and
        # Windows x64.
        lane = "cu128"
        note = "Experimental portable cu128 compatibility lane for SM 10.0 or newer."
    elif arch == "arm64":
        lane = "cu126"
        note = "Experimental ARM64 portable lane; not equivalent to the exact upstream native stack."
    else:
        lane = "cu124"
        note = "Portable compatibility lane using the Torch 2.6/cu124 model ABI."
    minimum_driver_hint = {"cu124": 124, "cu126": 126, "cu128": 128}[lane]
    if cuda_driver_hint and cuda_driver_hint < minimum_driver_hint:
        raise DependencyError(
            "DRIVER_CUDA_TOO_OLD",
            f"the selected {lane} PyTorch runtime needs a newer NVIDIA driver (Modly reports CUDA {cuda_driver_hint})",
        )
    return DependencyPlan(
        profile=profile,
        system=system,
        arch=arch,
        gpu_sm=gpu_sm,
        torch_lane=lane,
        torch_requirements=python_lane.torch_requirements_for(lane, arch),
        torch_index=python_lane.torch_index_for(lane),
        attention_backend="sdpa",
        install_flash_attn=False,
        install_native_stack=False,
        support_level="experimental" if arch == "arm64" or gpu_sm >= 100 else "compatibility",
        note=note,
        python_abi=python_abi,
    )


def _requirement_name(requirement: str) -> str:
    name, separator, version = str(requirement).partition("==")
    if not separator or not name.strip() or not version.strip():
        raise DependencyError(
            "DEPENDENCY_LOCK_INVALID",
            f"locked dependency is not an exact requirement: {requirement!r}",
        )
    return re.sub(r"[-_.]+", "-", name.strip()).casefold()


def _deduplicated_requirements(requirements: Sequence[str]) -> tuple[str, ...]:
    selected: list[str] = []
    by_name: dict[str, str] = {}
    for raw in requirements:
        requirement = str(raw)
        name = _requirement_name(requirement)
        previous = by_name.get(name)
        if previous is not None:
            if previous != requirement:
                raise DependencyError(
                    "DEPENDENCY_LOCK_CONFLICT",
                    f"locked requirements conflict for {name}: {previous!r} and {requirement!r}",
                )
            continue
        by_name[name] = requirement
        selected.append(requirement)
    return tuple(selected)


def _torch_platform_transitive_requirements(
    plan: DependencyPlan,
) -> tuple[str, ...]:
    python_lane = _python_requirement_lane(plan)
    torch_platform = python_lane.torch_platform_for(
        plan.torch_lane, plan.system, plan.arch
    )
    expected_torch = python_lane.torch_requirements_for(plan.torch_lane, plan.arch)
    if (
        tuple(plan.torch_requirements) != expected_torch
        or plan.torch_index != python_lane.torch_index_for(plan.torch_lane)
    ):
        raise DependencyError(
            "DEPENDENCY_PLAN_INVALID",
            "the selected Torch requirements do not match the audited lane",
        )
    if plan.install_native_stack and not (
        plan.torch_lane == "cu124"
        and plan.arch == "x64"
        and plan.system in {"linux", "win32"}
    ):
        raise DependencyError(
            "DEPENDENCY_PLAN_INVALID",
            "exact-upstream is only locked for explicit CPython 3.11 or 3.12 x64 cu124 lanes",
        )
    if plan.profile == "exact-upstream":
        expected_attention = "flash_attn" if plan.system == "linux" else "xformers"
        expected_flash = plan.system == "linux"
        coherent = (
            plan.install_native_stack
            and plan.install_flash_attn is expected_flash
            and plan.attention_backend == expected_attention
            and 80 <= plan.gpu_sm < 100
        )
    elif plan.profile == "portable":
        coherent = (
            not plan.install_native_stack
            and not plan.install_flash_attn
            and plan.attention_backend == "sdpa"
            and ((plan.torch_lane == "cu128") == (plan.gpu_sm >= 100))
        )
    else:
        coherent = False
    if not coherent:
        raise DependencyError(
            "DEPENDENCY_PLAN_INVALID",
            "the dependency profile flags do not match the audited closure",
        )
    return torch_platform


def constraint_requirements(plan: DependencyPlan) -> tuple[str, ...]:
    """Return the complete exact dependency closure for one explicit Python lane."""

    python_abi_from_fingerprint(
        {
            "implementation": plan.python_abi.implementation,
            "version": plan.python_abi.version,
            "cache_tag": plan.python_abi.cache_tag,
            "abiflags": plan.python_abi.abiflags,
            "soabi": plan.python_abi.soabi,
            "platform": plan.python_abi.platform,
            "machine": plan.python_abi.machine,
            "pointer_bits": plan.python_abi.pointer_bits,
        }
    )
    python_lane = _python_requirement_lane(plan)
    torch_platform = _torch_platform_transitive_requirements(plan)
    requirements: list[str] = [
        *python_lane.bootstrap,
        *python_lane.base,
        *python_lane.common_transitive,
        f"{OVOXEL_CPU_DISTRIBUTION}=={OVOXEL_CPU_VERSION}",
    ]
    if plan.arch == "x64":
        requirements.extend(python_lane.x64_render)
        requirements.extend(python_lane.x64_render_common)
        requirements.extend(
            python_lane.linux_x64_render
            if plan.system == "linux"
            else python_lane.windows_x64_render
        )
    requirements.extend(plan.torch_requirements)
    requirements.extend(python_lane.torch_common_for(plan.torch_lane))
    requirements.extend(torch_platform)
    if plan.install_native_stack:
        requirements.extend(python_lane.exact_sparse)
        requirements.extend(
            python_lane.exact_native_for(plan.system, plan.install_flash_attn)
        )
    return _deduplicated_requirements(requirements)


def _cusparselt_normalization_identity(
    plan: DependencyPlan,
) -> dict[str, object] | None:
    """Return the audited correction identity for its one applicable plan."""

    if not (
        plan.system == "linux"
        and plan.arch == "arm64"
        and plan.torch_lane == "cu128"
    ):
        return None
    if plan.python_abi.lane not in {"cp311", "cp312"}:
        raise DependencyError(
            "CUSPARSELT_NORMALIZATION_UNSUPPORTED",
            "the cuSPARSELt metadata correction has no audited Python ABI lane",
        )
    torch_platform = _torch_platform_transitive_requirements(plan)
    if (
        tuple(plan.torch_requirements)
        != ("torch==2.9.1+cu128", "torchvision==0.24.1")
        or plan.torch_index != PYTORCH_INDEXES["cu128"]
        or tuple(
            requirement
            for requirement in torch_platform
            if _requirement_name(requirement) == "nvidia-cusparselt-cu12"
        )
        != ("nvidia-cusparselt-cu12==0.7.1",)
    ):
        raise DependencyError(
            "CUSPARSELT_NORMALIZATION_UNSUPPORTED",
            "the selected Torch closure does not match the audited cuSPARSELt correction",
        )
    contract = _CUSPARSELT_NORMALIZATION_CONTRACT
    return {
        "schema": contract.schema,
        "distribution": contract.distribution,
        "version": contract.version,
        "pythonLanes": ["cp311", "cp312"],
        "platform": "linux",
        "architecture": "arm64",
        "torch": "torch==2.9.1+cu128",
        "library": {
            "path": contract.library_relative,
            "elfClass": 64,
            "machine": 183,
            "dlopenRequired": True,
        },
        "wheel": {
            "path": contract.wheel_relative,
            "beforeTag": "py3-none-manylinux2014_sbsa",
            "afterTag": "py3-none-manylinux2014_aarch64",
            "beforeSha256": hashlib.sha256(contract.original_wheel).hexdigest(),
            "afterSha256": hashlib.sha256(contract.normalized_wheel).hexdigest(),
        },
        "record": {
            "path": contract.record_relative,
            "beforeSha256": contract.original_record_sha256,
            "afterSha256": contract.normalized_record_sha256,
        },
    }


def dependency_lock_payload(plan: DependencyPlan) -> dict[str, object]:
    python_lane = _python_requirement_lane(plan)
    payload: dict[str, object] = {
        "schema": DEPENDENCY_SCHEMA,
        "plan": plan.payload(),
        "bootstrap": list(python_lane.bootstrap),
        "base": list(python_lane.base),
        "constraints": list(constraint_requirements(plan)),
        "platform": list(python_lane.x64_render if plan.arch == "x64" else ()),
        # Both profiles expose the compatibility backend.  Exact additionally
        # exposes the full native upstream backend, but still installs this
        # small CPU operator so users can switch backend without Repair.
        "portableCpu": {
            "distribution": OVOXEL_CPU_DISTRIBUTION,
            "version": OVOXEL_CPU_VERSION,
            "buildIdentity": OVOXEL_CPU_BUILD_IDENTITY,
            "templateTreeSha256": TEMPLATE_TREE_SHA256,
            "trellis2Revision": TRELLIS2_REVISION,
            "eigenRevision": OVOXEL_EIGEN_REVISION,
            "eigenTreeSha256": EIGEN_TREE_SHA256,
            "sourceTreeSha256": PORTABLE_CPU_SOURCE_TREE_SHA256,
            "licenseSha256": {
                name: digest
                for name, (_relative, digest) in sorted(
                    LICENSE_SOURCE_SPECS.items()
                )
            },
            "sourceArchives": [
                asdict(item)
                for item in SOURCE_ARCHIVES
                if item.name in {"trellis2", "ovo-eigen"}
            ],
        },
    }
    if plan.install_native_stack:
        payload["exact"] = {
            "extra": list(python_lane.x64_render),
            "sparse": list(python_lane.exact_sparse),
            "torchScatter": python_lane.exact_requirement_for(
                "torch-scatter", plan.system, plan.install_flash_attn
            ),
            "torchScatterLinks": python_lane.torch_scatter_links,
            "xformers": python_lane.exact_requirement_for(
                "xformers", plan.system, plan.install_flash_attn
            ),
            "flashAttn": (
                python_lane.exact_requirement_for(
                    "flash-attn", plan.system, plan.install_flash_attn
                )
                if plan.install_flash_attn
                else None
            ),
            "triton": python_lane.exact_requirement_for(
                "triton-windows" if plan.system == "win32" else "triton",
                plan.system,
                plan.install_flash_attn,
            ),
            "sources": [asdict(item) for item in SOURCE_ARCHIVES],
            "sourceTreeSha256": NATIVE_SOURCE_TREE_SHA256[plan.system],
            "patches": [
                FLEXGEMM_CACHE_PATCH,
                OVOXEL_WINDOWS_PATCH if plan.system == "win32" else None,
            ],
        }
    else:
        payload["portable"] = {
            "overlays": "pure-torch-sparse-sdpa-software-renderer-v2",
        }
    normalization = _cusparselt_normalization_identity(plan)
    if normalization is not None:
        payload["installedMetadataNormalization"] = normalization
    return payload


def dependency_lock_digest(plan: DependencyPlan) -> str:
    encoded = json.dumps(
        dependency_lock_payload(plan), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dependency_state_payload(
    plan: DependencyPlan, interpreter_fingerprint: Mapping[str, object]
) -> dict[str, object]:
    """State setup must compare before reusing an existing venv."""

    payload = {
        "schema": DEPENDENCY_SCHEMA,
        "lockDigest": dependency_lock_digest(plan),
        "plan": plan.payload(),
        "interpreter": dict(interpreter_fingerprint),
    }
    normalization = _cusparselt_normalization_identity(plan)
    if normalization is not None:
        payload["installedMetadataNormalization"] = normalization
    return payload


def state_matches(path: Path, expected: Mapping[str, object]) -> bool:
    try:
        encoded = read_owned_regular_bytes(path, max_bytes=STATE_MAX_BYTES)
        actual = json.loads(encoded.decode("utf-8"))
    except (TreeIntegrityError, UnicodeError, json.JSONDecodeError):
        return False
    return actual == dict(expected)


def write_state(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(dict(payload), sort_keys=True, indent=2) + "\n").encode("utf-8")
    if len(encoded) > STATE_MAX_BYTES:
        raise DependencyError("STATE_TOO_LARGE", "dependency state exceeds its safety limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise DependencyError("STATE_WRITE_FAILED", "dependency state could not be written") from exc


def _archive_is_valid(path: Path, spec: SourceArchive) -> bool:
    try:
        info = path.lstat()
        return (
            stat.S_ISREG(info.st_mode)
            and not stat.S_ISLNK(info.st_mode)
            and not getattr(info, "st_file_attributes", 0) & 0x400
            and info.st_size == spec.size
            and sha256_regular_file(path) == spec.sha256
        )
    except (OSError, TreeIntegrityError):
        return False


def _validated_cache_root(cache_root: Path) -> Path:
    raw = Path(cache_root)
    if not raw.is_absolute():
        raise DependencyError("CACHE_PATH_RELATIVE", "dependency cache path must be absolute")
    try:
        raw.mkdir(parents=True, exist_ok=True)
        info = raw.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400
            or not stat.S_ISDIR(info.st_mode)
        ):
            raise DependencyError(
                "CACHE_PATH_ALIAS", "dependency cache must be a regular directory"
            )
        return raw.resolve(strict=True)
    except DependencyError:
        raise
    except OSError as exc:
        raise DependencyError("CACHE_PATH_INVALID", "dependency cache is unavailable") from exc


def _remove_owned_node(current: Path) -> None:
    """Recursively remove one already-contained node without following aliases."""

    try:
        info = current.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise DependencyError("CACHE_REPAIR_FAILED", "cache entry cannot be inspected") from exc
    attributes = getattr(info, "st_file_attributes", 0)
    is_entry_alias = stat.S_ISLNK(info.st_mode) or bool(attributes & 0x400)
    if is_entry_alias:
        try:
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                current.rmdir()
            else:
                current.unlink()
        except IsADirectoryError:
            current.rmdir()
        except OSError as exc:
            raise DependencyError(
                "CACHE_REPAIR_FAILED", "cache alias cannot be removed"
            ) from exc
        return
    if stat.S_ISDIR(info.st_mode):
        try:
            entries = list(os.scandir(current))
        except OSError as exc:
            raise DependencyError(
                "CACHE_REPAIR_FAILED", "cache directory cannot be inspected"
            ) from exc
        for entry in entries:
            _remove_owned_node(Path(entry.path))
        try:
            current.rmdir()
        except OSError as exc:
            raise DependencyError(
                "CACHE_REPAIR_FAILED", "cache directory cannot be removed"
            ) from exc
        return
    try:
        current.unlink()
    except OSError as exc:
        raise DependencyError("CACHE_REPAIR_FAILED", "cache file cannot be removed") from exc


def _remove_owned_entry(path: Path, owned_parent: Path) -> None:
    """Remove one immediate child without following a symlink/reparse alias."""

    target = Path(path).absolute()
    parent = Path(owned_parent).absolute()
    if target.parent != parent:
        raise DependencyError("CACHE_REPAIR_ESCAPE", "cache repair target is unsafe")
    try:
        parent_info = parent.lstat()
        canonical_parent = parent.resolve(strict=True)
        canonical_target_parent = target.parent.resolve(strict=True)
    except OSError as exc:
        raise DependencyError(
            "CACHE_REPAIR_ESCAPE", "cache repair parent cannot be verified"
        ) from exc
    if (
        stat.S_ISLNK(parent_info.st_mode)
        or getattr(parent_info, "st_file_attributes", 0) & 0x400
        or not stat.S_ISDIR(parent_info.st_mode)
        or canonical_target_parent != canonical_parent
    ):
        raise DependencyError("CACHE_REPAIR_ESCAPE", "cache repair parent is unsafe")
    _remove_owned_node(target)


def _owned_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise DependencyError(
            "CACHE_ENTRY_INVALID", "cache entry cannot be inspected"
        ) from exc
    return True


_BUILD_TRANSIENT_DIRECTORIES = frozenset({"build", "dist", "__pycache__"})
_BUILD_TRANSIENT_SUFFIXES = frozenset(
    {
        ".a",
        ".cubin",
        ".dll",
        ".dylib",
        ".exp",
        ".fatbin",
        ".ilk",
        ".lib",
        ".o",
        ".obj",
        ".pdb",
        ".ptx",
        ".pyc",
        ".pyd",
        ".so",
    }
)
_BUILD_WORKSPACE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_BUILD_LOCK_SCHEMA = "modly.lato2.build-lock.v1"
_BUILD_LOCK_MARKER = ".dependency-lock.json"
_BUILD_LOCK_PREFIX_LENGTH = 20


def _ensure_owned_directory(parent: Path, name: str) -> Path:
    """Create one ordinary child directory and reject aliases at the boundary."""

    child = parent / name
    if child.parent != parent:
        raise DependencyError("BUILD_WORKSPACE_ESCAPE", "build workspace path is unsafe")
    try:
        child.mkdir(parents=False, exist_ok=True)
        info = child.lstat()
    except OSError as exc:
        raise DependencyError(
            "BUILD_WORKSPACE_INVALID", "build workspace directory is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or getattr(info, "st_file_attributes", 0) & 0x400
        or not stat.S_ISDIR(info.st_mode)
    ):
        raise DependencyError(
            "BUILD_WORKSPACE_ALIAS", "build workspace must be a regular directory"
        )
    return child


def _build_lock_payload(plan: DependencyPlan) -> dict[str, object]:
    return {
        "schema": _BUILD_LOCK_SCHEMA,
        "lockDigest": dependency_lock_digest(plan),
        "plan": plan.payload(),
    }


def _prepare_build_lock_directory(
    cache: Path, category: str, plan: DependencyPlan
) -> Path:
    """Resolve a short MAX_PATH-friendly directory authenticated by full lock."""

    category_root = _ensure_owned_directory(cache, category)
    expected = _build_lock_payload(plan)
    lock_name = str(expected["lockDigest"])[:_BUILD_LOCK_PREFIX_LENGTH]
    lock_root = _ensure_owned_directory(category_root, lock_name)
    marker = lock_root / _BUILD_LOCK_MARKER
    if state_matches(marker, expected):
        return lock_root

    # A missing/corrupt marker or digest-prefix collision must never mix build
    # output.  This cache is disposable, and recursive removal unlinks aliases
    # without traversing their targets.
    _remove_owned_entry(lock_root, category_root)
    lock_root = _ensure_owned_directory(category_root, lock_name)
    write_state(lock_root / _BUILD_LOCK_MARKER, expected)
    if not state_matches(lock_root / _BUILD_LOCK_MARKER, expected):
        raise DependencyError(
            "BUILD_LOCK_INVALID", "build workspace lock marker could not be verified"
        )
    return lock_root


def _is_build_transient(relative: Path, *, directory: bool) -> bool:
    name = relative.name.casefold()
    if directory:
        return name in _BUILD_TRANSIENT_DIRECTORIES or name.endswith(".egg-info")
    return (
        name.endswith(".egg-info")
        or name in {".ninja_deps", ".ninja_log", "build.ninja"}
        or Path(name).suffix in _BUILD_TRANSIENT_SUFFIXES
        or ".so." in name
    )


def _copy_clean_build_tree(source: Path, destination: Path) -> None:
    """Copy source files without aliases, special files, or prior build output."""

    try:
        root_info = source.lstat()
    except OSError as exc:
        raise DependencyError(
            "BUILD_SOURCE_MISSING", "native build source directory is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(root_info.st_mode)
        or getattr(root_info, "st_file_attributes", 0) & 0x400
        or not stat.S_ISDIR(root_info.st_mode)
    ):
        raise DependencyError(
            "BUILD_SOURCE_ALIAS", "native build source must be a regular directory"
        )

    destination.mkdir(parents=False, exist_ok=False)
    pending = [(source, destination, Path())]
    while pending:
        current_source, current_destination, relative_root = pending.pop()
        try:
            entries = sorted(os.scandir(current_source), key=lambda entry: entry.name)
        except OSError as exc:
            raise DependencyError(
                "BUILD_SOURCE_INVALID", "native build source cannot be enumerated"
            ) from exc
        for entry in entries:
            source_entry = Path(entry.path)
            relative = relative_root / entry.name
            try:
                info = source_entry.lstat()
            except OSError as exc:
                raise DependencyError(
                    "BUILD_SOURCE_INVALID", "native build source entry cannot be inspected"
                ) from exc
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise DependencyError(
                    "BUILD_SOURCE_ALIAS",
                    f"native build source contains an alias: {relative.as_posix()}",
                )
            destination_entry = current_destination / entry.name
            if stat.S_ISDIR(info.st_mode):
                if _is_build_transient(relative, directory=True):
                    continue
                destination_entry.mkdir()
                pending.append((source_entry, destination_entry, relative))
                continue
            if not stat.S_ISREG(info.st_mode):
                raise DependencyError(
                    "BUILD_SOURCE_SPECIAL",
                    f"native build source contains a special entry: {relative.as_posix()}",
                )
            if _is_build_transient(relative, directory=False):
                continue
            # follow_symlinks=False prevents a path swapped to an alias
            # between lstat and copy from being dereferenced by shutil.
            shutil.copy2(source_entry, destination_entry, follow_symlinks=False)
            copied_info = destination_entry.lstat()
            if (
                stat.S_ISLNK(copied_info.st_mode)
                or getattr(copied_info, "st_file_attributes", 0) & 0x400
                or not stat.S_ISREG(copied_info.st_mode)
            ):
                raise DependencyError(
                    "BUILD_SOURCE_CHANGED",
                    f"native build source changed while copied: {relative.as_posix()}",
                )


def prepare_build_workspace(
    source_root: Path,
    cache_root: Path,
    plan: DependencyPlan,
    purpose: str,
) -> Path:
    """Reset and populate a clean, dependency-lock-keyed local build tree.

    Local PEP 517/setuptools builds may leave ABI-specific objects both in
    ``build`` and beside Python packages.  Building only from this staging
    tree prevents Repair, profile changes, and Torch/SM lane transitions from
    consuming those objects.  Source aliases are rejected rather than
    followed; ignored transient directories are never traversed.
    """

    normalized_purpose = str(purpose).strip().casefold()
    if not _BUILD_WORKSPACE_NAME.fullmatch(normalized_purpose):
        raise DependencyError(
            "BUILD_WORKSPACE_NAME", "build workspace purpose is invalid"
        )
    cache = _validated_cache_root(Path(cache_root))
    lock_root = _prepare_build_lock_directory(cache, "build-workspaces", plan)
    final = lock_root / normalized_purpose
    staging = lock_root / f".{normalized_purpose}.{uuid.uuid4().hex}.staging"
    try:
        _copy_clean_build_tree(Path(source_root).absolute(), staging)
        _remove_owned_entry(final, lock_root)
        os.replace(staging, final)
    except Exception:
        try:
            _remove_owned_entry(staging, lock_root)
        except DependencyError:
            pass
        raise
    return final


def _download_archive(spec: SourceArchive, destination: Path, log: LogFunction) -> None:
    if _archive_is_valid(destination, spec):
        log(f"Reusing verified source archive: {spec.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    request = Request(spec.url, headers={"User-Agent": "modly-lato2-extension/1.0"})
    log(f"Downloading pinned source archive: {spec.name}")
    try:
        digest = hashlib.sha256()
        total = 0
        with urlopen(request, timeout=120) as response, temporary.open("xb") as handle:
            while block := response.read(DOWNLOAD_CHUNK):
                total += len(block)
                if total > spec.size:
                    raise DependencyError(
                        "SOURCE_SIZE_MISMATCH", f"{spec.name} exceeded its pinned size"
                    )
                digest.update(block)
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        if total != spec.size or digest.hexdigest() != spec.sha256:
            raise DependencyError(
                "SOURCE_INTEGRITY_FAILED",
                f"{spec.name} does not match the pinned archive inventory",
            )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _safe_extract_zip(archive: Path, destination: Path, expected_root: str) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    count = 0
    total = 0
    try:
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                count += 1
                total += member.file_size
                if count > SOURCE_FILE_LIMIT or total > SOURCE_UNCOMPRESSED_LIMIT:
                    raise DependencyError(
                        "SOURCE_ARCHIVE_LIMIT", "source archive exceeds extraction limits"
                    )
                name = member.filename.replace("\\", "/")
                pure = PurePosixPath(name)
                if (
                    not name
                    or pure.is_absolute()
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or pure.parts[0] != expected_root
                ):
                    raise DependencyError(
                        "SOURCE_ARCHIVE_PATH", "source archive contains an unsafe path"
                    )
                unix_mode = (member.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(unix_mode):
                    raise DependencyError(
                        "SOURCE_ARCHIVE_ALIAS", "source archive contains a symbolic link"
                    )
            zipped.extractall(destination)
        root = destination / expected_root
        if not root.is_dir():
            raise DependencyError("SOURCE_ARCHIVE_ROOT", "source archive root is missing")
        return root
    except Exception:
        try:
            _remove_owned_entry(destination, destination.parent)
        except DependencyError:
            pass
        raise


def _replace_exact(path: Path, old: str, new: str, *, count: int = 1) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DependencyError("SOURCE_PATCH_READ", f"cannot read source patch target {path.name}") from exc
    if text.count(old) != count:
        raise DependencyError(
            "SOURCE_PATCH_CONTEXT",
            f"pinned patch context changed in {path.name}; refusing an unsafe patch",
        )
    patched = text.replace(old, new, count)
    path.write_text(patched, encoding="utf-8", newline="\n")


def _patch_flexgemm(root: Path) -> None:
    setup_path = root / "setup.py"
    old_setup = '''os.makedirs(os.path.expanduser("~/.flex_gemm"), exist_ok=True)
src_cache_path = os.path.join(ROOT, "autotune_cache.json")
dst_cache_path = os.path.expanduser("~/.flex_gemm/autotune_cache.json")'''
    new_setup = '''modly_cache_root = os.environ.get("MODLY_LATO2_CACHE_DIR")
if not modly_cache_root:
    raise RuntimeError("MODLY_LATO2_CACHE_DIR is required for isolated FlexGEMM setup")
dst_cache_dir = os.path.join(os.path.abspath(modly_cache_root), "flex_gemm")
os.makedirs(dst_cache_dir, exist_ok=True)
src_cache_path = os.path.join(ROOT, "autotune_cache.json")
dst_cache_path = os.path.join(dst_cache_dir, "autotune_cache.json")'''
    _replace_exact(setup_path, old_setup, new_setup)

    init_path = root / "flex_gemm" / "__init__.py"
    old_init = '''AUTOTUNE_CACHE_PATH = os.environ.get(
    'FLEX_GEMM_AUTOTUNE_CACHE_PATH',
    os.path.expanduser('~/.flex_gemm/autotune_cache.json')
)'''
    new_init = '''_MODLY_CACHE_ROOT = os.environ.get('MODLY_LATO2_CACHE_DIR')
AUTOTUNE_CACHE_PATH = os.environ.get('FLEX_GEMM_AUTOTUNE_CACHE_PATH')
if AUTOTUNE_CACHE_PATH is None:
    if not _MODLY_CACHE_ROOT:
        raise RuntimeError('MODLY_LATO2_CACHE_DIR is required for isolated FlexGEMM runtime')
    AUTOTUNE_CACHE_PATH = os.path.join(
        os.path.abspath(_MODLY_CACHE_ROOT), 'flex_gemm', 'autotune_cache.json'
    )'''
    _replace_exact(init_path, old_init, new_init)


def _patch_ovo_windows(root: Path) -> None:
    setup_path = root / "setup.py"
    _replace_exact(setup_path, "import os\n", "import os\nimport platform\n")
    anchor = '''else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

setup('''
    replacement = '''else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

if platform.system() == "Windows":
    extra_compile_args = {
        "cxx": ["/O2", "/std:c++17", "/EHsc", "/permissive-", "/Zc:__cplusplus"],
        "nvcc": ["-O3", "-std=c++17", "-Xcompiler=/std:c++17", "-Xcompiler=/EHsc", "-Xcompiler=/permissive-", "-Xcompiler=/Zc:__cplusplus"] + cc_flag,
    }
else:
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "-std=c++17"] + cc_flag,
    }

setup('''
    _replace_exact(setup_path, anchor, replacement)
    old_args = '''            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3","-std=c++17"] + cc_flag,
            }'''
    _replace_exact(setup_path, old_args, "            extra_compile_args=extra_compile_args")

    replacements = {
        root / "src" / "convert" / "flexible_dual_grid.cpp": (
            ("1e-6d", "1e-6", 1),
            ("0.0d", "0.0", 1),
        ),
        root / "src" / "io" / "filter_neighbor.cpp": (
            ("torch::zeros({N, C}", "torch::zeros({static_cast<int64_t>(N), static_cast<int64_t>(C)}", 2),
        ),
        root / "src" / "io" / "filter_parent.cpp": (
            ("torch::zeros({N_leaf, C}", "torch::zeros({static_cast<int64_t>(N_leaf), static_cast<int64_t>(C)}", 2),
        ),
        root / "src" / "io" / "svo.cpp": (
            ("{svo.size()}", "{static_cast<int64_t>(svo.size())}", 1),
            ("{codes.size()}", "{static_cast<int64_t>(codes.size())}", 1),
        ),
    }
    for path, edits in replacements.items():
        for old, new, count in edits:
            _replace_exact(path, old, new, count=count)


def _move_tree_contents(source: Path, destination: Path) -> None:
    if _owned_entry_exists(destination):
        _remove_owned_entry(destination, destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _source_marker_payload(system: str) -> dict[str, object]:
    return {
        "schema": SOURCE_SCHEMA,
        "system": system,
        "archives": [asdict(item) for item in SOURCE_ARCHIVES],
        "patches": {
            "flexgemm": FLEXGEMM_CACHE_PATCH,
            "ovoxelWindows": OVOXEL_WINDOWS_PATCH if system == "win32" else None,
            "ovoxelReference": OVOXEL_WINDOWS_PATCH_REFERENCE if system == "win32" else None,
        },
    }


def _source_layout(root: Path) -> NativeSources:
    return NativeSources(
        root=root,
        nvdiffrast=root / "nvdiffrast",
        cumesh=root / "CuMesh",
        flexgemm=root / "FlexGEMM",
        ovoxel=root / "o-voxel",
    )


def _regular_owned_directory(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and not getattr(info, "st_file_attributes", 0) & 0x400
    )


def _prepare_source_cache_directory(parent: Path, name: str) -> Path:
    """Create or safely replace one exact extension-owned cache directory."""

    path = parent / name
    if path.parent != parent:
        raise DependencyError("SOURCE_CACHE_ESCAPE", "source cache path is unsafe")
    try:
        present = path.lstat()
    except FileNotFoundError:
        present = None
    except OSError as exc:
        raise DependencyError(
            "SOURCE_CACHE_INVALID", "source cache directory cannot be inspected"
        ) from exc
    if present is not None and not _regular_owned_directory(path):
        _remove_owned_entry(path, parent)
    try:
        path.mkdir(parents=False, exist_ok=True)
    except OSError as exc:
        raise DependencyError(
            "SOURCE_CACHE_INVALID", "source cache directory cannot be created"
        ) from exc
    if not _regular_owned_directory(path):
        raise DependencyError(
            "SOURCE_CACHE_ALIAS", "source cache directory is an unsafe alias"
        )
    return path


def _constraint_file_content(plan: DependencyPlan) -> bytes:
    lock_digest = dependency_lock_digest(plan)
    lines = [
        "# Generated by modly-lato2-extension; do not edit.",
        f"# dependency-lock-sha256: {lock_digest}",
        *constraint_requirements(plan),
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _constraint_file_matches(path: Path, expected: bytes) -> bool:
    try:
        info = path.lstat()
        return (
            stat.S_ISREG(info.st_mode)
            and not stat.S_ISLNK(info.st_mode)
            and not getattr(info, "st_file_attributes", 0) & 0x400
            and info.st_size == len(expected)
            and path.read_bytes() == expected
        )
    except OSError:
        return False


def validate_dependency_constraints_file(
    path: Path, plan: DependencyPlan
) -> Path:
    """Revalidate the exact bytes before a subprocess may consume the lock."""

    candidate = Path(path)
    if not _constraint_file_matches(candidate, _constraint_file_content(plan)):
        raise DependencyError(
            "DEPENDENCY_CONSTRAINTS_INVALID",
            "the locked pip constraints file is missing, unsafe, or modified",
        )
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise DependencyError(
            "DEPENDENCY_CONSTRAINTS_INVALID",
            "the locked pip constraints file cannot be resolved",
        ) from exc


def materialize_dependency_constraints(
    cache_root: Path, plan: DependencyPlan
) -> Path:
    """Atomically publish the authenticated complete lock for pip commands."""

    cache = _validated_cache_root(Path(cache_root))
    owned_root = _prepare_source_cache_directory(cache, "dependency-constraints")
    expected = _constraint_file_content(plan)
    lock_digest = dependency_lock_digest(plan)
    final = owned_root / f"{lock_digest[:_BUILD_LOCK_PREFIX_LENGTH]}.txt"
    if _constraint_file_matches(final, expected):
        return validate_dependency_constraints_file(final, plan)
    if _owned_entry_exists(final):
        _remove_owned_entry(final, owned_root)
    temporary = owned_root / f".{lock_digest[:_BUILD_LOCK_PREFIX_LENGTH]}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final)
    except OSError as exc:
        try:
            _remove_owned_entry(temporary, owned_root)
        except DependencyError:
            pass
        raise DependencyError(
            "DEPENDENCY_CONSTRAINTS_WRITE_FAILED",
            "the locked pip constraints file could not be published",
        ) from exc
    return validate_dependency_constraints_file(final, plan)


def _source_archives_are_trusted(root: Path, specs: Sequence[SourceArchive]) -> bool:
    archives = root.parent / "archives"
    if not _regular_owned_directory(archives):
        return False
    return all(
        _archive_is_valid(
            archives / f"{spec.name}-{spec.sha256[:12]}.zip",
            spec,
        )
        for spec in specs
    )


def _source_tree_has_digest(root: Path, expected: str) -> bool:
    try:
        return inventory_tree(root).digest == expected
    except (OSError, TreeIntegrityError):
        return False


def _native_sources_valid(root: Path, system: str) -> bool:
    expected_digest = NATIVE_SOURCE_TREE_SHA256.get(system)
    if expected_digest is None or not _source_archives_are_trusted(root, SOURCE_ARCHIVES):
        return False
    return state_matches(
        root / "source-lock.json", _source_marker_payload(system)
    ) and _source_tree_has_digest(
        root, expected_digest
    )


def prepare_native_sources(
    cache_root: Path, system: str, *, log: LogFunction = print
) -> NativeSources:
    """Provision the exact native source graph without relying on Git."""

    system = _normalize_system(system)
    if system not in {"linux", "win32"}:
        raise DependencyError("SOURCE_PLATFORM", "native sources support Windows and Linux only")
    lock_material = json.dumps(_source_marker_payload(system), sort_keys=True).encode("utf-8")
    source_id = hashlib.sha256(lock_material).hexdigest()[:16]
    cache = _validated_cache_root(Path(cache_root))
    owned_root = _prepare_source_cache_directory(cache, "native-sources")
    archives_dir = _prepare_source_cache_directory(owned_root, "archives")
    final_root = owned_root / source_id
    if _native_sources_valid(final_root, system):
        log("Reusing verified pinned native source tree")
        return _source_layout(final_root)

    archive_paths: dict[str, Path] = {}
    for spec in SOURCE_ARCHIVES:
        archive_path = archives_dir / f"{spec.name}-{spec.sha256[:12]}.zip"
        _download_archive(spec, archive_path, log)
        archive_paths[spec.name] = archive_path

    staging = owned_root / f".{source_id}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=False, exist_ok=False)
    extraction_root = staging / "extract"
    extraction_root.mkdir()
    try:
        extracted: dict[str, Path] = {}
        for spec in SOURCE_ARCHIVES:
            target = extraction_root / spec.name
            extracted[spec.name] = _safe_extract_zip(
                archive_paths[spec.name], target, spec.root
            )

        os.replace(extracted["nvdiffrast"], staging / "nvdiffrast")
        os.replace(extracted["cumesh"], staging / "CuMesh")
        os.replace(extracted["flexgemm"], staging / "FlexGEMM")
        os.replace(extracted["trellis2"] / "o-voxel", staging / "o-voxel")
        _move_tree_contents(
            extracted["cubvh"], staging / "CuMesh" / "third_party" / "cubvh"
        )
        _move_tree_contents(
            extracted["cubvh-eigen"],
            staging / "CuMesh" / "third_party" / "cubvh" / "third_party" / "eigen",
        )
        _move_tree_contents(
            extracted["ovo-eigen"], staging / "o-voxel" / "third_party" / "eigen"
        )
        _remove_owned_entry(extraction_root, staging)

        _patch_flexgemm(staging / "FlexGEMM")
        if system == "win32":
            _patch_ovo_windows(staging / "o-voxel")
        write_state(staging / "source-lock.json", _source_marker_payload(system))
        if _owned_entry_exists(final_root):
            quarantine = owned_root / f".{source_id}.{uuid.uuid4().hex}.invalid"
            os.replace(final_root, quarantine)
            try:
                os.replace(staging, final_root)
            except BaseException:
                if not _owned_entry_exists(final_root) and _owned_entry_exists(quarantine):
                    os.replace(quarantine, final_root)
                raise
            _remove_owned_entry(quarantine, owned_root)
        else:
            os.replace(staging, final_root)
    except Exception:
        try:
            _remove_owned_entry(staging, owned_root)
        except DependencyError:
            pass
        raise
    if not _native_sources_valid(final_root, system):
        raise DependencyError("SOURCE_VERIFY_FAILED", "prepared native source tree is incomplete")
    return _source_layout(final_root)


def _portable_cpu_marker_payload() -> dict[str, object]:
    return {
        "schema": SOURCE_SCHEMA,
        "kind": "portable-ovoxel-cpu-inputs",
        "archives": [
            asdict(item)
            for item in SOURCE_ARCHIVES
            if item.name in {"trellis2", "ovo-eigen"}
        ],
    }


def _portable_cpu_sources_valid(root: Path) -> bool:
    specs = tuple(
        item for item in SOURCE_ARCHIVES if item.name in {"trellis2", "ovo-eigen"}
    )
    if not _source_archives_are_trusted(root, specs):
        return False
    return state_matches(
        root / "source-lock.json", _portable_cpu_marker_payload()
    ) and _source_tree_has_digest(
        root, PORTABLE_CPU_SOURCE_TREE_SHA256
    )


def prepare_portable_cpu_sources(
    cache_root: Path, *, log: LogFunction = print
) -> PortableCpuSources:
    """Fetch only the two verified archives needed by the portable CPU build."""

    specs = {
        item.name: item
        for item in SOURCE_ARCHIVES
        if item.name in {"trellis2", "ovo-eigen"}
    }
    lock_material = json.dumps(_portable_cpu_marker_payload(), sort_keys=True).encode("utf-8")
    source_id = hashlib.sha256(lock_material).hexdigest()[:16]
    cache = _validated_cache_root(Path(cache_root))
    owned_root = _prepare_source_cache_directory(cache, "portable-cpu-sources")
    archives_dir = _prepare_source_cache_directory(owned_root, "archives")
    final_root = owned_root / source_id
    if _portable_cpu_sources_valid(final_root):
        log("Reusing verified portable o-voxel CPU source inputs")
        return PortableCpuSources(final_root, final_root / "o-voxel", final_root / "eigen")

    archive_paths: dict[str, Path] = {}
    for name, spec in specs.items():
        archive_path = archives_dir / f"{name}-{spec.sha256[:12]}.zip"
        _download_archive(spec, archive_path, log)
        archive_paths[name] = archive_path

    staging = owned_root / f".{source_id}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        trellis = _safe_extract_zip(
            archive_paths["trellis2"], staging / "trellis-extract", specs["trellis2"].root
        )
        eigen = _safe_extract_zip(
            archive_paths["ovo-eigen"], staging / "eigen-extract", specs["ovo-eigen"].root
        )
        os.replace(trellis / "o-voxel", staging / "o-voxel")
        os.replace(eigen, staging / "eigen")
        _remove_owned_entry(staging / "trellis-extract", staging)
        _remove_owned_entry(staging / "eigen-extract", staging)
        write_state(staging / "source-lock.json", _portable_cpu_marker_payload())
        if _owned_entry_exists(final_root):
            quarantine = owned_root / f".{source_id}.{uuid.uuid4().hex}.invalid"
            os.replace(final_root, quarantine)
            try:
                os.replace(staging, final_root)
            except BaseException:
                if not _owned_entry_exists(final_root) and _owned_entry_exists(quarantine):
                    os.replace(quarantine, final_root)
                raise
            _remove_owned_entry(quarantine, owned_root)
        else:
            os.replace(staging, final_root)
    except Exception:
        try:
            _remove_owned_entry(staging, owned_root)
        except DependencyError:
            pass
        raise
    if not _portable_cpu_sources_valid(final_root):
        raise DependencyError(
            "PORTABLE_SOURCE_VERIFY_FAILED", "portable CPU source inputs are incomplete"
        )
    return PortableCpuSources(final_root, final_root / "o-voxel", final_root / "eigen")


def _default_runner(command: Sequence[str], env: Mapping[str, str] | None) -> None:
    subprocess.run(
        list(command),
        check=True,
        env=dict(env) if env is not None else None,
        stdin=subprocess.DEVNULL,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )


def _pip(
    python: Path,
    arguments: Sequence[str],
    *,
    env: Mapping[str, str],
    runner: CommandRunner,
    log: LogFunction,
    stage: str,
    network: bool = False,
    constraint_file: Path | None = None,
    constraint_plan: DependencyPlan | None = None,
) -> None:
    log(stage)
    pip_arguments = [str(argument) for argument in arguments]
    if pip_arguments and pip_arguments[0] == "install":
        if constraint_file is None:
            raise DependencyError(
                "DEPENDENCY_CONSTRAINTS_MISSING",
                "every pip install must use the complete locked constraints file",
            )
        if constraint_plan is None:
            raise DependencyError(
                "DEPENDENCY_CONSTRAINTS_INVALID",
                "the locked pip constraints plan is unavailable",
            )
        constraint = validate_dependency_constraints_file(
            constraint_file, constraint_plan
        )
        pip_arguments[1:1] = ("--constraint", str(constraint))
    clean_env = sanitize_subprocess_environment(
        env, allow_network=network, for_pip=True
    )
    runner(isolated_pip_command(python, pip_arguments, clean_env), clean_env)


def _parse_nvcc_version(output: str) -> tuple[int, int] | None:
    match = re.search(r"release\s+(\d+)\.(\d+)", output, re.IGNORECASE)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _find_cuda_home(base_env: Mapping[str, str]) -> Path | None:
    base_env = sanitize_subprocess_environment(base_env)
    candidates: list[Path] = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        if base_env.get(key):
            candidates.append(Path(base_env[key]))
    nvcc_on_path = shutil.which("nvcc", path=base_env.get("PATH"))
    if nvcc_on_path:
        candidates.append(Path(nvcc_on_path).resolve().parent.parent)
    if os.name == "nt":
        program_files = base_env.get("ProgramFiles") or base_env.get("PROGRAMFILES")
        if program_files:
            candidates.append(
                Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v12.4"
            )
    else:
        candidates.extend((Path("/usr/local/cuda-12.4"), Path("/usr/local/cuda")))
    executable = "nvcc.exe" if os.name == "nt" else "nvcc"
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        nvcc = resolved / "bin" / executable
        if not nvcc.is_file():
            continue
        try:
            completed = subprocess.run(
                [str(nvcc), "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
                env=dict(base_env),
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if _parse_nvcc_version(completed.stdout + completed.stderr) == (12, 4):
            return resolved
    return None


def _msvc_environment(base_env: Mapping[str, str]) -> dict[str, str]:
    env = sanitize_subprocess_environment(base_env)
    if shutil.which("cl.exe", path=env.get("PATH")):
        include = env.get("INCLUDE", "")
        if env.get("WindowsSdkDir") or "Windows Kits" in include:
            return env
        raise DependencyError(
            "WINDOWS_SDK_MISSING",
            "MSVC is active but a Windows SDK environment is missing; install the Desktop C++ workload",
        )
    roots = [
        env.get("ProgramFiles(x86)"),
        env.get("ProgramFiles"),
        env.get("PROGRAMFILES"),
    ]
    vswhere = next(
        (
            Path(root) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
            for root in roots
            if root
            and (Path(root) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe").is_file()
        ),
        None,
    )
    if vswhere is None:
        raise DependencyError(
            "MSVC_MISSING",
            "Visual Studio 2022 Build Tools with Desktop C++, MSVC v143 and a Windows SDK are required",
        )
    result = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-version",
            "[17.0,18.0)",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    installation = Path(result.stdout.strip())
    vcvars = installation / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.is_file():
        raise DependencyError("MSVC_VCVARS_MISSING", "Visual Studio C++ environment script is missing")
    command = f'call "{vcvars}" >nul && set'
    result = subprocess.run(
        ["cmd.exe", "/d", "/s", "/c", command],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key:
                env[key] = value
    if not shutil.which("cl.exe", path=env.get("PATH")):
        raise DependencyError("MSVC_ACTIVATION_FAILED", "MSVC cl.exe was not activated")
    include = env.get("INCLUDE", "")
    if not env.get("WindowsSdkDir") and "Windows Kits" not in include:
        raise DependencyError(
            "WINDOWS_SDK_MISSING",
            "Visual Studio 2022 is present but the Windows SDK/Desktop C++ workload is incomplete",
        )
    env["CXX"] = str(shutil.which("cl.exe", path=env.get("PATH")) or "cl.exe")
    return sanitize_subprocess_environment(env)


def _linux_cxx_environment(
    base_env: Mapping[str, str], *, require_openmp: bool
) -> dict[str, str]:
    """Select and compile-probe the same C++ driver PyTorch will invoke."""

    env = sanitize_subprocess_environment(base_env)
    explicit = env.get("CXX")
    if explicit:
        candidates = [explicit]
    else:
        candidates = ["c++", "g++", "clang++"]
    compiler: str | None = None
    for candidate in candidates:
        found = shutil.which(candidate, path=env.get("PATH"))
        if found:
            compiler = found
            break
        path = Path(candidate)
        if path.is_absolute() and path.is_file():
            compiler = str(path)
            break
    if compiler is None:
        if explicit:
            raise DependencyError(
                "CXX_INVALID", f"the configured CXX compiler is unavailable: {explicit}"
            )
        raise DependencyError(
            "CXX_MISSING", "a working C++17 compiler is required for native extensions"
        )
    with tempfile.TemporaryDirectory(prefix="modly-lato2-cxx-") as temporary:
        root = Path(temporary)
        source = root / "probe.cpp"
        output = root / "probe"
        source.write_text(
            "#include <vector>\n"
            + ("#include <omp.h>\n" if require_openmp else "")
            + "int main(){std::vector<int> v{1,2,3};"
            + ("return omp_get_max_threads() < 1;" if require_openmp else "return v.size()!=3;")
            + "}\n",
            encoding="utf-8",
        )
        command = [compiler, "-std=c++17"]
        if require_openmp:
            command.append("-fopenmp")
        command.extend((str(source), "-o", str(output)))
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=120,
                env=env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            requirement = "C++17 with OpenMP" if require_openmp else "C++17"
            raise DependencyError(
                "CXX_TOOLCHAIN_INCOMPLETE",
                f"the selected compiler cannot build a {requirement} probe: {compiler}",
            ) from exc
    env["CXX"] = compiler
    return env


def native_build_environment(
    plan: DependencyPlan, cache_root: Path, *, base_env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Validate the exact-profile toolchain and return its isolated build env."""

    if not plan.install_native_stack:
        raise DependencyError("NATIVE_PROFILE_REQUIRED", "portable profile has no native build")
    inherited = sanitize_subprocess_environment(
        os.environ if base_env is None else base_env,
        allow_network=True,
    )
    env = dict(inherited)
    cuda_home = _find_cuda_home(env)
    if cuda_home is None:
        raise DependencyError(
            "NVCC_MISSING",
            "CUDA Toolkit 12.4 with a working nvcc is required; another default toolkit or an NVIDIA driver alone is insufficient",
        )
    nvcc = cuda_home / "bin" / ("nvcc.exe" if plan.system == "win32" else "nvcc")
    result = subprocess.run(
        [str(nvcc), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    version = _parse_nvcc_version(result.stdout + result.stderr)
    if version != (12, 4):
        shown = "unknown" if version is None else f"{version[0]}.{version[1]}"
        raise DependencyError(
            "NVCC_VERSION_MISMATCH",
            f"exact-upstream requires CUDA Toolkit 12.4; detected nvcc {shown}",
        )
    if plan.system == "win32":
        env = _msvc_environment(env)
    else:
        env = _linux_cxx_environment(env, require_openmp=True)
    for key, value in inherited.items():
        if key.upper() in _NETWORK_ENV_NAMES:
            env[key] = value
    cache = _validated_cache_root(Path(cache_root))
    flex_cache = cache / "flex_gemm"
    flex_cache.mkdir(parents=True, exist_ok=True)
    torch_extensions = _prepare_build_lock_directory(
        cache, "torch-extensions", plan
    )
    env["CUDA_HOME"] = str(cuda_home)
    env["CUDA_PATH"] = str(cuda_home)
    env["CUDACXX"] = str(nvcc)
    env["PATH"] = str(cuda_home / "bin") + os.pathsep + env.get("PATH", "")
    env["BUILD_TARGET"] = "cuda"
    env["TORCH_CUDA_ARCH_LIST"] = f"{plan.gpu_sm // 10}.{plan.gpu_sm % 10}"
    env["MODLY_LATO2_CACHE_DIR"] = str(cache)
    env["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = str(flex_cache / "autotune_cache.json")
    env["TORCH_EXTENSIONS_DIR"] = str(torch_extensions)
    env.setdefault("MAX_JOBS", str(max(1, min(os.cpu_count() or 1, 8))))
    return env


def cpu_build_environment(
    plan: DependencyPlan, cache_root: Path, *, base_env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return a checked C++ build environment for the portable CPU operator."""

    env = sanitize_subprocess_environment(
        os.environ if base_env is None else base_env
    )
    if plan.system == "win32":
        env = _msvc_environment(env)
    else:
        env = _linux_cxx_environment(env, require_openmp=False)
    cache = _validated_cache_root(Path(cache_root))
    torch_extensions = _prepare_build_lock_directory(
        cache, "torch-extensions", plan
    )
    env["TORCH_EXTENSIONS_DIR"] = str(torch_extensions)
    env["PYTHONNOUSERSITE"] = "1"
    # This Eigen-heavy translation unit can exceed modest host RAM when
    # compiled concurrently with other extensions, especially on ARM boards.
    env["MAX_JOBS"] = "1"
    return env


# Backwards-compatible descriptive alias used by early setup integration.
portable_build_environment = cpu_build_environment


def _assert_target_python(
    python: Path, expected: PythonABI | None = None
) -> PythonABI:
    probe = (
        "import json, struct, sys, sysconfig; "
        "print(json.dumps({"
        "'implementation': sys.implementation.name, "
        "'version': list(sys.version_info[:2]), "
        "'cache_tag': sys.implementation.cache_tag, "
        "'abiflags': getattr(sys, 'abiflags', ''), "
        "'soabi': sysconfig.get_config_var('SOABI'), "
        "'platform': sysconfig.get_platform().lower(), "
        "'machine': __import__('platform').machine().lower(), "
        "'pointer_bits': struct.calcsize('P') * 8"
        "}, sort_keys=True))"
    )
    try:
        result = subprocess.run(
            [str(python), "-I", "-S", "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=sanitize_subprocess_environment(),
        )
        fingerprint = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise DependencyError(
            "PYTHON_PROBE_FAILED", "the extension virtualenv Python ABI could not be inspected"
        ) from exc
    if not isinstance(fingerprint, dict):
        raise DependencyError(
            "PYTHON_PROBE_INVALID", "the extension virtualenv returned invalid ABI metadata"
        )
    actual = python_abi_from_fingerprint(fingerprint)
    if expected is not None and actual != expected:
        raise DependencyError(
            "PYTHON_ABI_MISMATCH",
            "the extension virtualenv Python ABI does not match the selected dependency lane",
        )
    return actual


def _validate_linux_glibc(plan: DependencyPlan) -> None:
    if plan.system != "linux":
        return
    libc, version = platform.libc_ver()
    try:
        parsed = tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        parsed = ()
    minimum = (2, 31) if plan.arch == "x64" else (2, 28)
    if libc.casefold() != "glibc" or parsed < minimum:
        raise DependencyError(
            "GLIBC_TOO_OLD",
            f"the selected Linux wheels require glibc {minimum[0]}.{minimum[1]} or newer",
        )


def install_dependencies(
    python: Path,
    plan: DependencyPlan,
    cache_root: Path,
    *,
    log: LogFunction = print,
    runner: CommandRunner = _default_runner,
) -> Path:
    """Install a complete pinned plan.  Every pip command is checked."""

    python = _validated_venv_python_path(Path(python))
    _assert_target_python(python, plan.python_abi)
    _validate_linux_glibc(plan)
    cache = _validated_cache_root(Path(cache_root))
    constraint_file = materialize_dependency_constraints(cache, plan)
    python_lane = _python_requirement_lane(plan)
    env = sanitize_subprocess_environment(allow_network=True)
    if "PIP_CACHE_DIR" not in env:
        env["PIP_CACHE_DIR"] = str(_prepare_source_cache_directory(cache, "pip"))
    env["PYTHONNOUSERSITE"] = "1"

    _pip(
        python,
        [
            "install",
            "--upgrade",
            "--only-binary=:all:",
            "--index-url",
            PYPI_INDEX,
            *python_lane.bootstrap,
        ],
        env=env,
        runner=runner,
        log=log,
        stage="Pinning Python build tools",
        network=True,
        constraint_file=constraint_file,
        constraint_plan=plan,
    )
    _pip(
        python,
        [
            "install",
            "--only-binary=:all:",
            "--index-url",
            PYPI_INDEX,
            *python_lane.base,
            *(python_lane.x64_render if plan.arch == "x64" else ()),
        ],
        env=env,
        runner=runner,
        log=log,
        stage="Installing pinned LATO.2 Python dependencies",
        network=True,
        constraint_file=constraint_file,
        constraint_plan=plan,
    )
    _pip(
        python,
        [
            "install",
            "--only-binary=:all:",
            "--index-url",
            plan.torch_index,
            *plan.torch_requirements,
        ],
        env=env,
        runner=runner,
        log=log,
        stage=f"Installing pinned PyTorch {plan.torch_lane} runtime",
        network=True,
        constraint_file=constraint_file,
        constraint_plan=plan,
    )
    if _cusparselt_normalization_identity(plan) is not None:
        log("Normalizing audited cuSPARSELt ARM64 installed metadata")
        normalize_cusparselt_metadata(python, plan)

    if plan.install_native_stack:
        build_env = native_build_environment(plan, cache_root, base_env=env)
        sources = prepare_native_sources(Path(cache_root), plan.system, log=log)
        torch_scatter_requirement = python_lane.exact_requirement_for(
            "torch-scatter", plan.system, plan.install_flash_attn
        )
        triton_requirement = python_lane.exact_requirement_for(
            "triton-windows" if plan.system == "win32" else "triton",
            plan.system,
            plan.install_flash_attn,
        )
        xformers_requirement = python_lane.exact_requirement_for(
            "xformers", plan.system, plan.install_flash_attn
        )
        flash_attn_requirement = (
            python_lane.exact_requirement_for(
                "flash-attn", plan.system, plan.install_flash_attn
            )
            if plan.install_flash_attn
            else None
        )
        _pip(
            python,
            [
                "install",
                "--only-binary=:all:",
                "--index-url",
                PYPI_INDEX,
                *python_lane.exact_sparse,
            ],
            env=build_env,
            runner=runner,
            log=log,
            stage="Installing pinned spconv and cumm wheels",
            network=True,
            constraint_file=constraint_file,
            constraint_plan=plan,
        )
        _pip(
            python,
            [
                "install",
                "--no-deps",
                "--only-binary=:all:",
                "--no-index",
                torch_scatter_requirement,
                "--find-links",
                python_lane.torch_scatter_links,
            ],
            env=build_env,
            runner=runner,
            log=log,
            stage="Installing pinned torch-scatter wheel",
            network=True,
            constraint_file=constraint_file,
            constraint_plan=plan,
        )
        _pip(
            python,
            [
                "install",
                "--no-deps",
                "--only-binary=:all:",
                "--index-url",
                PYPI_INDEX,
                triton_requirement,
            ],
            env=build_env,
            runner=runner,
            log=log,
            stage="Installing the pinned Triton runtime",
            network=True,
            constraint_file=constraint_file,
            constraint_plan=plan,
        )
        _pip(
            python,
            [
                "install",
                "--no-deps",
                "--only-binary=:all:",
                "--index-url",
                plan.torch_index,
                xformers_requirement,
            ],
            env=build_env,
            runner=runner,
            log=log,
            stage="Installing the pinned upstream xformers backend",
            network=True,
            constraint_file=constraint_file,
            constraint_plan=plan,
        )
        if plan.install_flash_attn:
            _pip(
                python,
                [
                    "install",
                    "--no-build-isolation",
                    "--no-deps",
                    "--index-url",
                    PYPI_INDEX,
                    str(flash_attn_requirement),
                ],
                env=build_env,
                runner=runner,
                log=log,
                stage="Building the pinned upstream FlashAttention backend",
                network=True,
                constraint_file=constraint_file,
                constraint_plan=plan,
            )
        for purpose, name, source in (
            ("nvdiffrast", "nvdiffrast v0.4.0", sources.nvdiffrast),
            ("cumesh", "CuMesh", sources.cumesh),
            ("flexgemm", "FlexGEMM", sources.flexgemm),
            ("o-voxel", "o-voxel", sources.ovoxel),
        ):
            build_source = prepare_build_workspace(
                source,
                Path(cache_root),
                plan,
                purpose,
            )
            _pip(
                python,
                [
                    "install",
                    "--no-build-isolation",
                    "--no-deps",
                    "--no-index",
                    str(build_source),
                ],
                env=build_env,
                runner=runner,
                log=log,
                stage=f"Building pinned {name}",
                constraint_file=constraint_file,
                constraint_plan=plan,
            )
        env = build_env

    _pip(
        python,
        ["check"],
        env=env,
        runner=runner,
        log=log,
        stage="Checking the installed dependency graph",
    )
    verify_dependencies(python, plan, Path(cache_root), env=env)
    return constraint_file


def expected_distribution_versions(plan: DependencyPlan) -> dict[str, str]:
    def split(requirement: str) -> tuple[str, str]:
        name, version = requirement.split("==", 1)
        return name.casefold().replace("_", "-"), version

    # The portable CPU addon is installed immediately after the base graph by
    # setup.py and is checked (including licenses and ABI) by its dedicated
    # smoke.  Keep it in pip's constraints, but do not demand it in the earlier
    # base-environment smoke run.
    return {
        name: version
        for name, version in map(split, constraint_requirements(plan))
        if name != OVOXEL_CPU_DISTRIBUTION
    }


def verify_portable_cpu_extension(
    python: Path,
    plan: DependencyPlan,
    cache_root: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Verify the separately installed portable CPU extension and pip graph.

    Setup must call this after installing the build returned by
    ``materialize_ovoxel_cpu_build`` and only then publish dependency state.
    """

    python = _validated_venv_python_path(Path(python))
    smoke_env = cpu_build_environment(plan, cache_root, base_env=env)
    portable_identity = {
        "distribution": OVOXEL_CPU_DISTRIBUTION,
        "version": OVOXEL_CPU_VERSION,
        "buildIdentity": OVOXEL_CPU_BUILD_IDENTITY,
        "templateTreeSha256": TEMPLATE_TREE_SHA256,
        "licenseSha256": {
            name: digest
            for name, (_relative, digest) in sorted(LICENSE_SOURCE_SPECS.items())
        },
    }
    script = r'''
import hashlib
import importlib.metadata
import json
import os
from pathlib import PurePosixPath
import stat
import torch
import lato2_ovoxel_cpu
from lato2_ovoxel_cpu import _C

identity = json.loads(os.environ["MODLY_LATO2_OVOXEL_CPU_IDENTITY"])
distribution = importlib.metadata.distribution(identity["distribution"])
metadata_name = distribution.metadata.get("Name", "")
normalized_name = metadata_name.casefold().replace("_", "-")
if normalized_name != identity["distribution"]:
    raise RuntimeError(f"portable CPU extension name mismatch: {metadata_name}")
version = distribution.version
if version != identity["version"]:
    raise RuntimeError(f"portable CPU extension version mismatch: {version}")

expected_licenses = identity["licenseSha256"]
declared_licenses = {
    value.replace("\\", "/")
    for value in (distribution.metadata.get_all("License-File") or [])
}
expected_declarations = {f"LICENSES/{name}" for name in expected_licenses}
if declared_licenses != expected_declarations:
    raise RuntimeError("portable CPU extension license metadata is incomplete")

distribution_files = list(distribution.files or [])
for name, expected_sha256 in expected_licenses.items():
    suffix = f".dist-info/licenses/LICENSES/{name}"
    candidates = []
    for package_file in distribution_files:
        normalized = str(package_file).replace("\\", "/")
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or ".." in pure.parts:
            raise RuntimeError("portable CPU extension metadata contains an unsafe path")
        if normalized.endswith(suffix):
            candidates.append(package_file)
    if len(candidates) != 1:
        raise RuntimeError(f"portable CPU extension license file is missing: {name}")
    license_path = distribution.locate_file(candidates[0])
    info = license_path.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or getattr(info, "st_file_attributes", 0) & 0x400
        or not stat.S_ISREG(info.st_mode)
    ):
        raise RuntimeError(f"portable CPU extension license file is unsafe: {name}")
    digest = hashlib.sha256()
    with license_path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError(f"portable CPU extension license hash mismatch: {name}")

symbol = getattr(_C, "mesh_to_flexible_dual_grid_cpu", None)
if symbol is None:
    raise RuntimeError("mesh_to_flexible_dual_grid_cpu symbol is missing")

# Exercise the ABI and implementation, not only the import table.  This closed
# tetrahedron is small enough for setup yet intersects several cells.
vertices = torch.tensor([
    [0.50, 0.50, 0.50],
    [1.50, 0.50, 0.50],
    [0.50, 1.50, 0.50],
    [0.50, 0.50, 1.50],
], dtype=torch.float32).contiguous()
faces = torch.tensor([
    [0, 2, 1],
    [0, 1, 3],
    [0, 3, 2],
    [1, 2, 3],
], dtype=torch.int32).contiguous()
voxel_size = torch.tensor([0.25, 0.25, 0.25], dtype=torch.float32).contiguous()
grid_range = torch.tensor([[0, 0, 0], [8, 8, 8]], dtype=torch.int32).contiguous()
outputs = symbol(vertices, faces, voxel_size, grid_range, 1.0, 1.0, 0.1, False)
if not isinstance(outputs, tuple) or len(outputs) != 3:
    raise RuntimeError("portable CPU operator returned an invalid tuple")
coords, dual_vertices, intersected = outputs
for name, tensor in zip(("coords", "dual_vertices", "intersected"), outputs):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        raise RuntimeError(f"{name} is not a CPU tensor")
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise RuntimeError(f"{name} has an invalid shape: {tuple(tensor.shape)}")
if coords.shape[0] == 0 or not (coords.shape[0] == dual_vertices.shape[0] == intersected.shape[0]):
    raise RuntimeError("portable CPU operator returned empty or inconsistent outputs")
if coords.dtype != torch.int32 or dual_vertices.dtype != torch.float32 or intersected.dtype != torch.bool:
    raise RuntimeError("portable CPU operator returned invalid dtypes")
if not torch.isfinite(dual_vertices).all().item():
    raise RuntimeError("portable CPU operator returned non-finite vertices")
print(json.dumps({
    "ok": True,
    "version": version,
    "buildIdentity": identity["buildIdentity"],
    "templateTreeSha256": identity["templateTreeSha256"],
    "licensesVerified": sorted(expected_licenses),
    "symbol": symbol.__name__,
    "voxelCount": int(coords.shape[0]),
}, sort_keys=True))
'''
    smoke_env["MODLY_LATO2_OVOXEL_CPU_IDENTITY"] = json.dumps(
        portable_identity, sort_keys=True
    )
    try:
        pip_check_env = sanitize_subprocess_environment(
            smoke_env, for_pip=True
        )
        subprocess.run(
            isolated_pip_command(python, ["check"], smoke_env),
            check=True,
            capture_output=True,
            text=True,
            timeout=10 * 60,
            env=pip_check_env,
        )
        result = subprocess.run(
            [str(python), "-I", "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=10 * 60,
            env=smoke_env,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError, IndexError) as exc:
        raise DependencyError(
            "PORTABLE_CPU_SMOKE_FAILED",
            "the portable o-voxel CPU extension failed its import/symbol smoke check",
        ) from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise DependencyError("PORTABLE_CPU_SMOKE_INVALID", "portable CPU smoke result is invalid")
    if (
        payload.get("version") != OVOXEL_CPU_VERSION
        or payload.get("buildIdentity") != OVOXEL_CPU_BUILD_IDENTITY
        or payload.get("templateTreeSha256") != TEMPLATE_TREE_SHA256
        or payload.get("licensesVerified") != sorted(LICENSE_SOURCE_SPECS)
    ):
        raise DependencyError(
            "PORTABLE_CPU_SMOKE_IDENTITY",
            "the portable o-voxel CPU extension identity or license set is invalid",
        )
    return payload


def verify_dependencies(
    python: Path,
    plan: DependencyPlan,
    cache_root: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run import, version, native-symbol and CUDA smoke checks in the venv."""

    smoke_env = sanitize_subprocess_environment(
        os.environ if env is None else env
    )
    cache = _validated_cache_root(Path(cache_root))
    smoke_env["MODLY_LATO2_CACHE_DIR"] = str(cache)
    smoke_env["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = str(
        cache / "flex_gemm" / "autotune_cache.json"
    )
    smoke_env["PYTHONNOUSERSITE"] = "1"
    smoke_env["CUDA_MODULE_LOADING"] = "LAZY"
    if plan.system == "linux":
        xdg_runtime = cache / "xdg-runtime"
        xdg_runtime.mkdir(parents=True, exist_ok=True)
        try:
            xdg_runtime.chmod(0o700)
        except OSError:
            pass
        smoke_env["XDG_RUNTIME_DIR"] = str(xdg_runtime)
        smoke_env["EGL_PLATFORM"] = "surfaceless"
    python = _validated_venv_python_path(Path(python))
    _verify_cusparselt_metadata(python, plan)
    expected = expected_distribution_versions(plan)
    script = r'''
import importlib
import importlib.metadata
import json
import os

expected = json.loads(os.environ["MODLY_LATO2_EXPECTED_DISTS"])
for name, wanted in expected.items():
    actual = importlib.metadata.version(name)
    if actual != wanted:
        raise RuntimeError(f"{name} version {actual!r} != {wanted!r}")

import numpy
import trimesh
import tqdm
import PIL
import ninja
import psutil
import cv2
import huggingface_hub
import plyfile
import zstandard
import easydict
import einops
import filelock
import torch
import torchvision

if os.environ.get("MODLY_LATO2_OPEN3D_SMOKE") == "1":
    import open3d
    if os.environ.get("MODLY_LATO2_OPEN3D_RENDER_SMOKE") == "1":
        renderer = open3d.visualization.rendering.OffscreenRenderer(16, 16)
        rendered = numpy.asarray(renderer.render_to_image())
        if rendered.ndim != 3 or rendered.shape[:2] != (16, 16):
            raise RuntimeError("Open3D OffscreenRenderer returned an invalid image")
        del renderer

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the CUDA GPU")
device = torch.device("cuda")
x = torch.ones((16, 16), device=device)
y = x @ x
if float(y[0, 0].item()) != 16.0:
    raise RuntimeError("CUDA matrix smoke check returned an invalid value")
torch.cuda.synchronize()

native = os.environ.get("MODLY_LATO2_NATIVE_SMOKE") == "1"
attention = os.environ.get("MODLY_LATO2_ATTN_BACKEND", "sdpa")
if native:
    import spconv.pytorch as spconv
    import torch_scatter
    import xformers.ops as xops
    import nvdiffrast.torch as dr
    import cumesh
    import flex_gemm
    import flex_gemm.kernels.cuda as flex_cuda
    import o_voxel
    import o_voxel._C as ovoxel_cuda
    from o_voxel.convert import mesh_to_flexible_dual_grid
    required = {
        "spconv.SparseConvTensor": getattr(spconv, "SparseConvTensor", None),
        "torch_scatter.scatter_mean": getattr(torch_scatter, "scatter_mean", None),
        "xformers.memory_efficient_attention": getattr(xops, "memory_efficient_attention", None),
        "nvdiffrast.rasterize": getattr(dr, "rasterize", None),
        "cumesh.CuMesh": getattr(cumesh, "CuMesh", None),
        "flex_gemm.cuda": flex_cuda,
        "o_voxel._C": ovoxel_cuda,
        "o_voxel.mesh_to_flexible_dual_grid": mesh_to_flexible_dual_grid,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise RuntimeError("missing native symbols: " + ", ".join(missing))

    # Exercise the actual CUDA kernels and ABIs used by LATO.2/DINOv2. Imports
    # alone can succeed for a wheel built against an incompatible torch/CUDA
    # ABI and defer the failure until the first user generation.
    indices = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
        dtype=torch.int32,
        device=device,
    )
    sparse_input = spconv.SparseConvTensor(
        torch.randn((4, 2), device=device), indices, [2, 2, 2], 1
    )
    sparse_layer = spconv.SubMConv3d(2, 3, 3, bias=False).to(device)
    sparse_output = sparse_layer(sparse_input).features
    if sparse_output.shape != (4, 3) or not torch.isfinite(sparse_output).all().item():
        raise RuntimeError("spconv CUDA kernel smoke returned invalid output")

    scatter_source = torch.tensor([[1.0], [3.0], [2.0], [6.0]], device=device)
    scatter_index = torch.tensor([0, 0, 1, 1], device=device)
    scatter_output = torch_scatter.scatter_mean(scatter_source, scatter_index, dim=0)
    if not torch.allclose(
        scatter_output, torch.tensor([[2.0], [4.0]], device=device)
    ):
        raise RuntimeError("torch-scatter CUDA kernel smoke returned invalid output")

    ovo_vertices = torch.tensor(
        [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5], [0.5, 0.5, 1.5]],
        dtype=torch.float32,
    ).contiguous()
    ovo_faces = torch.tensor(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.int32
    ).contiguous()
    ovo_outputs = ovoxel_cuda.mesh_to_flexible_dual_grid_cpu(
        ovo_vertices,
        ovo_faces,
        torch.tensor([0.25, 0.25, 0.25], dtype=torch.float32),
        torch.tensor([[0, 0, 0], [8, 8, 8]], dtype=torch.int32),
        1.0,
        1.0,
        0.1,
        False,
    )
    if (
        not isinstance(ovo_outputs, tuple)
        or len(ovo_outputs) != 3
        or ovo_outputs[0].numel() == 0
    ):
        raise RuntimeError("o-voxel CPU operator smoke returned invalid output")

    attention_dtype = torch.bfloat16
    q = torch.randn((1, 8, 2, 32), device=device, dtype=attention_dtype)
    xformers_output = xops.memory_efficient_attention(q, q, q)
    if xformers_output.shape != q.shape or not torch.isfinite(xformers_output).all().item():
        raise RuntimeError("xformers CUDA attention smoke returned invalid output")
    if attention == "flash_attn":
        import flash_attn
        if getattr(flash_attn, "flash_attn_varlen_func", None) is None:
            raise RuntimeError("flash_attn_varlen_func is missing")
        flash_output = flash_attn.flash_attn_func(q, q, q, dropout_p=0.0)
        if flash_output.shape != q.shape or not torch.isfinite(flash_output).all().item():
            raise RuntimeError("FlashAttention CUDA kernel smoke returned invalid output")
    torch.cuda.synchronize()

print(json.dumps({
    "ok": True,
    "torch": torch.__version__,
    "torchCuda": torch.version.cuda,
    "capability": list(torch.cuda.get_device_capability()),
    "device": torch.cuda.get_device_name(0),
    "native": native,
    "attention": attention,
}, sort_keys=True))
'''
    smoke_env["MODLY_LATO2_EXPECTED_DISTS"] = json.dumps(expected, sort_keys=True)
    smoke_env["MODLY_LATO2_NATIVE_SMOKE"] = "1" if plan.install_native_stack else "0"
    smoke_env["MODLY_LATO2_OPEN3D_SMOKE"] = "1" if plan.arch == "x64" else "0"
    smoke_env["MODLY_LATO2_OPEN3D_RENDER_SMOKE"] = (
        "1" if plan.install_native_stack else "0"
    )
    smoke_env["MODLY_LATO2_ATTN_BACKEND"] = plan.attention_backend
    try:
        completed = subprocess.run(
            [str(python), "-I", "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=20 * 60,
            env=smoke_env,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError, IndexError) as exc:
        raise DependencyError(
            "DEPENDENCY_SMOKE_FAILED",
            "the installed LATO.2 dependency stack failed its import/CUDA/native-symbol smoke check",
        ) from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise DependencyError("DEPENDENCY_SMOKE_INVALID", "dependency smoke result is invalid")
    capability = payload.get("capability")
    if capability != [plan.gpu_sm // 10, plan.gpu_sm % 10]:
        raise DependencyError(
            "GPU_CHANGED",
            "the CUDA device capability differs from the capability selected during setup",
        )
    return payload


__all__ = [
    "DependencyError",
    "DependencyPlan",
    "NativeSources",
    "PortableCpuSources",
    "PythonABI",
    "PythonRequirementLane",
    "PYTHON_REQUIREMENT_LANES",
    "SUPPORTED_PYTHON_VERSIONS",
    "constraint_requirements",
    "dependency_lock_digest",
    "dependency_lock_payload",
    "dependency_state_payload",
    "expected_distribution_versions",
    "install_dependencies",
    "isolated_pip_command",
    "cpu_build_environment",
    "native_build_environment",
    "materialize_dependency_constraints",
    "normalize_cusparselt_metadata",
    "prepare_build_workspace",
    "prepare_native_sources",
    "portable_build_environment",
    "prepare_portable_cpu_sources",
    "python_abi_from_fingerprint",
    "sanitize_subprocess_environment",
    "select_dependency_plan",
    "state_matches",
    "verify_dependencies",
    "verify_portable_cpu_extension",
    "validate_dependency_constraints_file",
    "write_state",
]
