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

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
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

DEPENDENCY_SCHEMA = "modly.lato2.dependencies.v1"
SOURCE_SCHEMA = "modly.lato2.native-sources.v1"
STATE_MAX_BYTES = 128 * 1024
DOWNLOAD_CHUNK = 1024 * 1024
COMMAND_TIMEOUT_SECONDS = 4 * 60 * 60
SOURCE_FILE_LIMIT = 25_000
SOURCE_UNCOMPRESSED_LIMIT = 2 * 1024 * 1024 * 1024
PYPI_INDEX = "https://pypi.org/simple"

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
# pinned here as well.  The complete CPython 3.11 transitive closure is locked
# below per OS/architecture/Torch lane and supplied to every pip install.
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

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["torch_requirements"] = list(self.torch_requirements)
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


def isolated_pip_command(
    python: Path,
    arguments: Sequence[str],
    environment: Mapping[str, str],
) -> list[str]:
    """Construct pip without user config/index overrides, retaining Modly cache."""

    command = [
        str(Path(python).resolve()),
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
    context: Mapping[str, object], requested_profile: str | None = None
) -> DependencyPlan:
    """Resolve a deterministic dependency lane from Modly setup metadata."""

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
            torch_requirements=_torch_requirements("cu124", arch),
            torch_index=PYTORCH_INDEXES["cu124"],
            attention_backend="flash_attn" if use_flash else "xformers",
            install_flash_attn=use_flash,
            install_native_stack=True,
            support_level="toolchain-dependent",
            note=(
                "Complete pinned upstream dependency graph; native extensions are built locally."
            ),
        )

    # Portable is deliberately separate: Modly compatibility overlays replace
    # spconv, torch-scatter, Open3D rendering and the native o_voxel import tree.
    # Their validation belongs to the extension runtime, not this exact stack.
    if gpu_sm >= 100:
        # Modly 0.4.2 reports 128 for every driver major >=570 and does not
        # expose the raw driver version.  Selecting cu130 from that capped hint
        # would be unsafe: CUDA 13 wheels require a newer driver than many
        # valid 12.8 hosts.  The official cu128 CPython 3.11 wheels contain the
        # required Blackwell kernels on Linux ARM64/x64 and Windows x64.
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
        torch_requirements=_torch_requirements(lane, arch),
        torch_index=PYTORCH_INDEXES[lane],
        attention_backend="sdpa",
        install_flash_attn=False,
        install_native_stack=False,
        support_level="experimental" if arch == "arm64" or gpu_sm >= 100 else "compatibility",
        note=note,
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
    key = (plan.torch_lane, plan.system, plan.arch)
    supported: dict[tuple[str, str, str], tuple[str, ...]] = {
        ("cu124", "linux", "x64"): TORCH_CU124_LINUX_X64_TRANSITIVE_REQUIREMENTS,
        ("cu124", "win32", "x64"): (),
        ("cu126", "linux", "arm64"): (),
        ("cu128", "linux", "x64"): TORCH_CU128_LINUX_TRANSITIVE_REQUIREMENTS,
        ("cu128", "linux", "arm64"): TORCH_CU128_LINUX_TRANSITIVE_REQUIREMENTS,
        ("cu128", "win32", "x64"): (),
    }
    if key not in supported:
        raise DependencyError(
            "DEPENDENCY_PLAN_UNSUPPORTED",
            "the selected OS/architecture/Torch lane has no audited dependency closure",
        )
    expected_torch = _torch_requirements(plan.torch_lane, plan.arch)
    if (
        tuple(plan.torch_requirements) != expected_torch
        or plan.torch_index != PYTORCH_INDEXES.get(plan.torch_lane)
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
            "exact-upstream is only locked for the CPython 3.11 x64 cu124 lanes",
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
    return supported[key]


def constraint_requirements(plan: DependencyPlan) -> tuple[str, ...]:
    """Return the complete exact CPython 3.11 dependency closure for a plan."""

    torch_platform = _torch_platform_transitive_requirements(plan)
    requirements: list[str] = [
        *BOOTSTRAP_REQUIREMENTS,
        *BASE_REQUIREMENTS,
        *COMMON_TRANSITIVE_REQUIREMENTS,
        f"{OVOXEL_CPU_DISTRIBUTION}=={OVOXEL_CPU_VERSION}",
    ]
    if plan.arch == "x64":
        requirements.extend(X64_RENDER_REQUIREMENTS)
        requirements.extend(X64_RENDER_COMMON_TRANSITIVE_REQUIREMENTS)
        requirements.extend(
            LINUX_X64_RENDER_TRANSITIVE_REQUIREMENTS
            if plan.system == "linux"
            else WINDOWS_X64_RENDER_TRANSITIVE_REQUIREMENTS
        )
    requirements.extend(plan.torch_requirements)
    try:
        requirements.extend(TORCH_LANE_COMMON_TRANSITIVE_REQUIREMENTS[plan.torch_lane])
    except KeyError as exc:
        raise DependencyError(
            "DEPENDENCY_PLAN_UNSUPPORTED",
            "the selected Torch lane has no audited dependency closure",
        ) from exc
    requirements.extend(torch_platform)
    if plan.install_native_stack:
        requirements.extend(EXACT_SPARSE_REQUIREMENTS)
        requirements.extend(
            (
                TORCH_SCATTER_REQUIREMENT,
                XFORMERS_REQUIREMENT,
                WINDOWS_TRITON_REQUIREMENT
                if plan.system == "win32"
                else LINUX_TRITON_REQUIREMENT,
                *(
                    (FLASH_ATTN_REQUIREMENT,)
                    if plan.install_flash_attn
                    else ()
                ),
                *NATIVE_SOURCE_DISTRIBUTION_REQUIREMENTS,
                *EXACT_TRANSITIVE_REQUIREMENTS,
            )
        )
    return _deduplicated_requirements(requirements)


def dependency_lock_payload(plan: DependencyPlan) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": DEPENDENCY_SCHEMA,
        "plan": plan.payload(),
        "bootstrap": list(BOOTSTRAP_REQUIREMENTS),
        "base": list(BASE_REQUIREMENTS),
        "constraints": list(constraint_requirements(plan)),
        "platform": list(X64_RENDER_REQUIREMENTS if plan.arch == "x64" else ()),
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
            "extra": list(X64_RENDER_REQUIREMENTS),
            "sparse": list(EXACT_SPARSE_REQUIREMENTS),
            "torchScatter": TORCH_SCATTER_REQUIREMENT,
            "xformers": XFORMERS_REQUIREMENT,
            "flashAttn": FLASH_ATTN_REQUIREMENT if plan.install_flash_attn else None,
            "triton": (
                WINDOWS_TRITON_REQUIREMENT
                if plan.system == "win32"
                else LINUX_TRITON_REQUIREMENT
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

    return {
        "schema": DEPENDENCY_SCHEMA,
        "lockDigest": dependency_lock_digest(plan),
        "plan": plan.payload(),
        "interpreter": dict(interpreter_fingerprint),
    }


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


def _assert_target_python(python: Path) -> None:
    result = subprocess.run(
        [str(python), "-I", "-S", "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=sanitize_subprocess_environment(),
    )
    if result.stdout.strip() != "3.11":
        raise DependencyError(
            "PYTHON_ABI_UNSUPPORTED", "this release targets Modly's CPython 3.11 runtime"
        )


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

    python = Path(python).resolve()
    if not python.is_file():
        raise DependencyError("PYTHON_MISSING", "extension virtualenv Python is missing")
    _assert_target_python(python)
    _validate_linux_glibc(plan)
    cache = _validated_cache_root(Path(cache_root))
    constraint_file = materialize_dependency_constraints(cache, plan)
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
            *BOOTSTRAP_REQUIREMENTS,
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
            *BASE_REQUIREMENTS,
            *(X64_RENDER_REQUIREMENTS if plan.arch == "x64" else ()),
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

    if plan.install_native_stack:
        build_env = native_build_environment(plan, cache_root, base_env=env)
        sources = prepare_native_sources(Path(cache_root), plan.system, log=log)
        _pip(
            python,
            [
                "install",
                "--only-binary=:all:",
                "--index-url",
                PYPI_INDEX,
                *EXACT_SPARSE_REQUIREMENTS,
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
                TORCH_SCATTER_REQUIREMENT,
                "--find-links",
                TORCH_SCATTER_LINKS,
            ],
            env=build_env,
            runner=runner,
            log=log,
            stage="Installing pinned torch-scatter wheel",
            network=True,
            constraint_file=constraint_file,
            constraint_plan=plan,
        )
        triton_requirement = (
            WINDOWS_TRITON_REQUIREMENT if plan.system == "win32" else LINUX_TRITON_REQUIREMENT
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
                XFORMERS_REQUIREMENT,
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
                    FLASH_ATTN_REQUIREMENT,
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
            [str(Path(python).resolve()), "-I", "-c", script],
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
            [str(Path(python).resolve()), "-I", "-c", script],
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
    "prepare_build_workspace",
    "prepare_native_sources",
    "portable_build_environment",
    "prepare_portable_cpu_sources",
    "sanitize_subprocess_environment",
    "select_dependency_plan",
    "state_matches",
    "verify_dependencies",
    "verify_portable_cpu_extension",
    "validate_dependency_constraints_file",
    "write_state",
]
