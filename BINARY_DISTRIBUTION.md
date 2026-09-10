# Portable binary distribution

## User installation

`auto` and `portable` install the CPU voxelizer from precompiled wheels. Python
3.11/3.12, Windows x64, Linux x64 and Linux ARM64 have separate records, further
partitioned by the full Torch build (`2.6.0+cu124`, `2.6.0+cu126`, `2.9.1+cu128`).
The compiled operator is CPU-only, but its LibTorch linkage still requires the
matching Torch distribution. A wheel filename's Python/platform tags alone are
not sufficient evidence of that compatibility.

`lato2_modly/binary-wheels.json` is the committed trust root. Setup never fetches
an unpinned remote index or guesses URLs for missing artifacts. Each record pins
the wheel hash/size, installed file hashes, source commit, template and operator
source/license identity. Downloads use the existing confined, resumable asset
manager, under the selected models revision's `runtime-cache/binary-wheels`.
Repair reuses a valid cached wheel without downloading it again.

Before pip installation, the whole file and every wheel member are checked,
including the filename identity, METADATA, WHEEL tags, RECORD, and license hashes.
Installation uses the verified local wheel, `--no-index --no-deps
--only-binary=:all:` and the existing locked dependency constraints. There is no
source fallback. Missing artifacts fail in preflight before model downloads.

Verification checks the installed bytes and exact Torch build before running
the real tetrahedron voxelization smoke. Runtime verification and promoted-venv
checks no longer call `cpu_build_environment`. The selected wheel SHA is bound
to saved environment state, so replacing a release artifact or changing an
inventory entry cannot silently reuse an old environment. Staging/rollback is
unchanged. Model asset IDs and hashes are unchanged.

## Backend choice

`auto` now selects portable on all supported platforms. Linux x64 SM 8.x/9.x
previously selected exact-upstream automatically: this is an explicit default
change, not a claim that the compatibility backend is numerically identical.
Users retaining the exact backend must select `exact-upstream`. That developer
profile still builds additional upstream CUDA dependencies and requires its
existing compiler/SDK/Toolkit stack. It is never used as a missing-wheel fallback.

## Maintainer recipe

The wheel workflow builds the authenticated TRELLIS.2/Eigen sources in CI. It
normalizes two GCC-specific redundant double-literal suffixes (`1e-6d`, `0.0d`)
to ordinary C++ double literals, preserving their type/value for MSVC. Original
source and license pins remain checked before this compiler-portability edit;
the recipe's exact source commit is recorded in the binary inventory.

Wheels use a build tag incorporating Torch/CUDA, avoiding filename collisions
between otherwise identical Python/platform wheels. Linux builds run in a
manylinux glibc 2.28 container. These wheels retain `linux_*` tags: no unperformed
`auditwheel` certification is implied. The inventory declares the build baseline;
consumer import tests remain necessary on every claimed deployment image.

Only tested artifacts should be copied into a release, and their generated
records must be reviewed/committed to the extension. Publishing a native-library
pack does not merge or publish the extension code. No real hashes should be
replaced by placeholders and no untested lane should be advertised as verified.

## Validation scope

Unit tests forbid compiler discovery/spawn in portable preflight, operator
installation and runtime verification; cover SHA corruption, aliases, unsafe
paths, duplicate/missing inventory records, exact Python/Torch selection, binary
pip flags and wheel-state binding. Existing transaction tests retain source-lane
coverage alongside the binary tests.

Build jobs install each wheel into a separate consumer venv, remove compiler
settings/PATH, and execute the real CPU operator twice with shape/dtype/finite
and deterministic-geometry checks. This is not a clean-Windows VM certification:
CI Windows images still contain system runtimes and development software.
A clean Windows Install/Repair and full LATO.2 GPU inference remain separate
acceptance checks. Likewise, native ARM64 CPU operator tests do not certify a
complete Linux ARM64 CUDA inference session.
