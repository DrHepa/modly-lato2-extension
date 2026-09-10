"""Source-derived test fixtures, not captures from a Windows/CUDA machine."""


def windows_fingerprint(minor=11):
    suffix = f".cp3{minor}-win_amd64.pyd"
    return {
        "implementation": "cpython", "version": [3, minor],
        "version_full": [3, minor, 9 if minor == 11 else 10],
        "cache_tag": f"cpython-3{minor}", "abiflags": "", "soabi": None,
        "ext_suffix": suffix, "extension_suffixes": [suffix, ".pyd"],
        "is_debug": False, "platform": "win-amd64", "machine": "amd64",
        "pointer_bits": 64,
    }


def linux_fingerprint(minor=11, arch="x86_64"):
    soabi = f"cpython-3{minor}-{arch}-linux-gnu"
    return {
        "implementation": "cpython", "version": [3, minor],
        "version_full": [3, minor, 9 if minor == 11 else 10],
        "cache_tag": f"cpython-3{minor}", "abiflags": "", "soabi": soabi,
        "ext_suffix": f".{soabi}.so",
        "extension_suffixes": [f".{soabi}.so", ".abi3.so", ".so"],
        "is_debug": False, "platform": f"linux-{arch}", "machine": arch,
        "pointer_bits": 64,
    }
