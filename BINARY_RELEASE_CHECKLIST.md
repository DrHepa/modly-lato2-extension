# Binary release and acceptance checklist

## Maintainer workflow

1. Run `Build portable operator wheels` manually with a **new** `ovoxel-cpu-*`
   release identifier. Build jobs have read-only repository permissions and do
   not publish or modify the extension automatically.
2. Require all twelve build jobs to finish successfully. Each produces a wheel,
   its `.whl.json` record and a `smoke.json` report. Verify the build run's exact
   source SHA and repository, then review the generated records against the
   intended Python/platform/exact-Torch matrix. Never promote fork artifacts or
   interpolate unreviewed records into release commands.
3. Verify every wheel with `binary_wheels.parse_spec` and
   `binary_wheels.verify_wheel`. Commit the reviewed records into
   `lato2_modly/binary-wheels.json`; this local inventory, not a remotely mutable
   index, authorizes installation. Archive/member/license hashes and source
   identities are mandatory. Keep model asset pins unchanged for binary-only
   releases.
4. Publish the twelve wheels, the matching inventory and SHA256SUMS in a draft
   GitHub release whose tag matches the records. Do not overwrite artifacts of
   an existing release. Redownload all draft assets, verify again against the
   committed inventory, then publish the binary pack.
5. Run `Compiler-free binary consumer` against the committed integration. The
   thirteen jobs cover every published binary plus the exact Windows standalone
   Python 3.11.9 pinned by Modly upstream. This uses production preflight,
   download, wheel installation, installed-file verification and CPU operator
   smoke, with compiler discovery forbidden. Repeat cache verification blocks
   network access. Optional model/Torch fixtures in the general offline test
   suite remain explicitly skipped rather than treated as passed.

## First pack provenance

Tag: `ovoxel-cpu-post2-20260910.1`.

- Linux x64/ARM64 builds: run `34464232916`, source
  `d7592043c3db5e50fef030afb2334e69925f4239` (eight successful Linux jobs; its early
  Windows jobs failed and were not promoted).
- Windows x64 builds: run `34465721218`, source
  `08af54551be69225096cde2bd8e4e25c3f0202d8` (four successful jobs).
- Authenticated assembly, complete available offline tests and redownload
  verification: run `34467029207`.

The compiler-portability edit removes only redundant `d` suffixes from two
C++ double literals. Pinned original sources are checked before applying it.
The fixed source revisions, build recipe and bundled license files remain
available in this repository's history.

## User-facing acceptance still separate from CI

A Windows runner with its compiler removed from PATH is **not** a clean Windows
installation. Before claiming complete installer or inference certification,
record a real Modly Install and Repair on Windows without Build Tools/SDK and a
successful LATO.2 job with an appropriate NVIDIA GPU/driver. Also retain a native
Linux ARM64 CUDA lane for full model acceptance. CPU binary tests do not claim
GPU compatibility, model quality, VRAM capacity or complete offline inference.

Normal setup no longer requires development tools. The OS and compatible
NVIDIA driver remain platform prerequisites; existing system/PyTorch runtime
libraries are not equivalent to installing a compiler or CUDA development SDK.
Do not silently install system toolchains or bypass failed binary verification.
