"""Standard-library-only Python ABI probing shared by Install and Repair.

CPython 3.11/3.12 on Windows may omit SOABI. In that case only the exact
release EXT_SUFFIX for a supported platform/version can establish the ABI.
An absent SOABI is not, by itself, evidence of an unsupported interpreter.
"""
from __future__ import annotations

import _imp
import json
import platform
import struct
import sys
import sysconfig
from typing import Mapping, NoReturn

SUPPORTED_PYTHON_VERSIONS = frozenset({(3, 11), (3, 12)})
SUPPORTED_RELEASE_ABIS = {
    ((3, 11), "linux-x86_64", "x86_64"): "cpython-311-x86_64-linux-gnu",
    ((3, 12), "linux-x86_64", "x86_64"): "cpython-312-x86_64-linux-gnu",
    ((3, 11), "linux-aarch64", "aarch64"): "cpython-311-aarch64-linux-gnu",
    ((3, 12), "linux-aarch64", "aarch64"): "cpython-312-aarch64-linux-gnu",
    ((3, 11), "win-amd64", "amd64"): "cp311-win_amd64",
    ((3, 12), "win-amd64", "amd64"): "cp312-win_amd64",
}


def current_python_fingerprint() -> dict[str, object]:
    """Collect raw evidence; executable paths are deliberately not ABI fields."""
    return {
        "implementation": sys.implementation.name,
        "version": list(sys.version_info[:2]),
        "version_full": list(sys.version_info[:3]),
        "cache_tag": sys.implementation.cache_tag,
        "abiflags": getattr(sys, "abiflags", ""),
        "soabi": sysconfig.get_config_var("SOABI"),
        "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "extension_suffixes": _imp.extension_suffixes(),
        "is_debug": bool(sysconfig.get_config_var("Py_DEBUG"))
        or hasattr(sys, "gettotalrefcount"),
        "platform": sysconfig.get_platform().lower(),
        "machine": platform.machine().lower(),
        "pointer_bits": struct.calcsize("P") * 8,
    }


# Self-contained: -I -S must not import the extension or any site packages.
# The real-subprocess regression test compares this against the collector above.
INTERPRETER_PROBE = r'''
import _imp
import json
import platform
import struct
import sys
import sysconfig

print(json.dumps({
    "implementation": sys.implementation.name,
    "version": list(sys.version_info[:2]),
    "version_full": list(sys.version_info[:3]),
    "cache_tag": sys.implementation.cache_tag,
    "abiflags": getattr(sys, "abiflags", ""),
    "soabi": sysconfig.get_config_var("SOABI"),
    "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
    "extension_suffixes": _imp.extension_suffixes(),
    "is_debug": bool(sysconfig.get_config_var("Py_DEBUG"))
        or hasattr(sys, "gettotalrefcount"),
    "platform": sysconfig.get_platform().lower(),
    "machine": platform.machine().lower(),
    "pointer_bits": struct.calcsize("P") * 8,
}, sort_keys=True))
'''

_DIAGNOSTIC_FIELDS = (
    "implementation", "version", "version_full", "cache_tag", "abiflags",
    "soabi", "ext_suffix", "extension_suffixes", "is_debug", "platform",
    "machine", "pointer_bits",
)


def python_abi_diagnostic(fingerprint: Mapping[str, object]) -> str:
    """Format only interpreter evidence, never arbitrary payload/env keys."""
    return json.dumps(
        {name: fingerprint.get(name) for name in _DIAGNOSTIC_FIELDS},
        sort_keys=True, ensure_ascii=True, default=repr,
    )


class PythonABIError(ValueError):
    """Specific validation failure; the caller supplies its stable error code."""


def normalize_python_abi(fingerprint: Mapping[str, object]) -> dict[str, object]:
    """Return the existing PythonABI fields, retaining canonical lock identities.

    Older fingerprints with an explicit valid SOABI remain supported. All new
    probes also report EXT_SUFFIX, the loader suffixes and debug-build evidence.
    Missing SOABI requires an exact Windows release EXT_SUFFIX, not a guessed
    string or a wildcard acceptance of .pyd files. Linux remains fail-closed.
    """
    if not isinstance(fingerprint, Mapping):
        raise PythonABIError("Python ABI metadata must be an object")

    def reject(reason: str) -> NoReturn:
        raise PythonABIError(
            "this release supports only release-build 64-bit CPython 3.11 and 3.12; "
            f"{reason}; detected={python_abi_diagnostic(fingerprint)}"
        )

    raw_version = fingerprint.get("version")
    if (
        not isinstance(raw_version, (list, tuple))
        or len(raw_version) != 2
        or any(isinstance(part, bool) or not isinstance(part, int) for part in raw_version)
    ):
        reject("version must contain exactly two integers (major, minor)")
    version = (raw_version[0], raw_version[1])
    if version not in SUPPORTED_PYTHON_VERSIONS:
        reject("unsupported Python version")
    implementation = str(fingerprint.get("implementation") or "").casefold()
    if implementation != "cpython":
        reject("implementation is not CPython")
    pointer_bits = fingerprint.get("pointer_bits")
    if type(pointer_bits) is not int or pointer_bits != 64:
        reject("interpreter pointer width is not 64 bits")
    cache_tag = str(fingerprint.get("cache_tag") or "")
    if cache_tag != f"cpython-{version[0]}{version[1]}":
        reject("cache_tag does not match the Python version")
    abiflags = str(fingerprint.get("abiflags") or "")
    if abiflags:
        reject("non-release ABI flags are unsupported")
    if fingerprint.get("is_debug", False) is not False:
        reject("debug-build evidence is true or malformed")

    platform_tag = str(fingerprint.get("platform") or "").strip().casefold()
    machine = str(fingerprint.get("machine") or "").strip().casefold()
    expected_soabi = SUPPORTED_RELEASE_ABIS.get((version, platform_tag, machine))
    if expected_soabi is None:
        reject("unsupported or inconsistent platform/machine combination")

    windows = platform_tag == "win-amd64"
    expected_suffix = f".{expected_soabi}.{'pyd' if windows else 'so'}"
    soabi = fingerprint.get("soabi")
    ext_suffix = fingerprint.get("ext_suffix")
    if soabi not in (None, "") and soabi != expected_soabi:
        reject(f"SOABI does not match the supported release ABI {expected_soabi!r}")
    if ext_suffix not in (None, "") and ext_suffix != expected_suffix:
        reject(f"EXT_SUFFIX does not match the release suffix {expected_suffix!r}")
    if soabi in (None, ""):
        if not windows:
            reject("SOABI is required for the supported Linux ABI")
        if ext_suffix != expected_suffix:
            reject(f"missing Windows SOABI requires exact EXT_SUFFIX {expected_suffix!r}")

    # The first loader suffix is authoritative (sysconfig uses it on Windows).
    # Merely finding a release suffix later in a debug loader's list is unsafe.
    if "extension_suffixes" in fingerprint:
        suffixes = fingerprint["extension_suffixes"]
        if (
            not isinstance(suffixes, (list, tuple))
            or not suffixes
            or any(not isinstance(suffix, str) for suffix in suffixes)
            or suffixes[0] != expected_suffix
        ):
            reject(f"extension loader does not prefer the release suffix {expected_suffix!r}")

    return {
        "implementation": implementation,
        "version": version,
        "cache_tag": cache_tag,
        "abiflags": abiflags,
        "soabi": expected_soabi,
        "platform": platform_tag,
        "machine": machine,
        "pointer_bits": pointer_bits,
    }
