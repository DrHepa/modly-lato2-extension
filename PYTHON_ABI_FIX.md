# Windows CPython ABI detection fix

## Root cause

The installer previously required `sysconfig.get_config_var("SOABI")` to equal
`cp311-win_amd64` or `cp312-win_amd64` on Windows. CPython 3.11/3.12 Windows
builds may not populate `SOABI`. Their `EXT_SUFFIX` instead comes from the
extension loader. Consequently a valid 64-bit CPython 3.11.9 could fail with
`PYTHON_ABI_UNSUPPORTED` before any dependencies were installed.

Primary references:
- CPython 3.11.9 `Lib/sysconfig.py`, `_init_non_posix`:
  https://github.com/python/cpython/blob/v3.11.9/Lib/sysconfig.py
- CPython 3.11.9 tagged release/debug loader suffixes:
  https://github.com/python/cpython/blob/v3.11.9/Python/dynload_win.c
- Modly's standalone runtime pin (`3.11.9`, `20240726`):
  https://github.com/lightningpixel/modly/blob/main/scripts/download-python-embed.js

## Behavior

`lato2_modly/python_abi.py` centralizes probing and normalization. `setup.py`
and dependency verification now use the same isolated probe. Raw metadata
records the patch version, `SOABI`, `EXT_SUFFIX`, loader suffixes and debug
build evidence. Executable paths are logged separately, not added to ABI
identity, because base Python and its venv have different executable paths.

When Windows `SOABI` is absent, only the exact supported release `EXT_SUFFIX`
can establish the canonical ABI. A wrong explicit `SOABI` is never overridden.
Generic `.pyd`, debug `_d` suffixes, wrong versions/architectures, unsupported
platforms, non-CPython implementations, non-64-bit builds and debug evidence
remain rejected. Linux x64 and ARM64 still require their supported `SOABI`.

The existing `PythonABI` dataclass and payload are retained. An explicit and
an EXT_SUFFIX-derived Windows ABI have identical canonical lock identities;
`cp311` and `cp312` remain distinct. This change does not alter package pins,
CUDA profiles, asset download logic, weights, or inference code. The expanded
raw fingerprint can invalidate an older environment state on the next Repair;
normal setup verification decides whether that environment must be rebuilt.

## Tests

From the complete patched repository:

```sh
python -m unittest discover -s tests -p "test_python_abi*.py" -v
python tools/check_python_abi.py --check-venv
python -m unittest discover -s tests -p "test_*.py" -v
```

To probe a user's actual Modly interpreter rather than a different Python
found on PATH, supply its executable explicitly:

```sh
python tools/check_python_abi.py --python "PATH_TO_MODLY_PYTHON" --check-venv
```

The diagnostic prints a JSON report and returns a nonzero exit code on failure.
The temporary venv uses `--without-pip`; no package or model downloads occur.
The fixture module explicitly labels Windows/Linux data as source-derived
fixtures, not captures from physical machines. Real subprocess and venv tests
run on the interpreter executing the suite.

The GitHub Actions matrix covers CPython 3.11/3.12 on Windows x64, Linux x64,
and native Linux ARM64. A separate Windows job downloads the exact standalone
CPython 3.11.9 distribution pinned by Modly upstream. Linux x64 also runs the
existing offline suite. The jobs have no GPU and do not establish CUDA,
end-to-end setup, asset download, or inference compatibility. Those remain
separate release acceptance checks; in particular native ARM64 ABI testing
is not a substitute for local Linux ARM64 CUDA testing.
