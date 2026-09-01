# LATO.2 portable runtime overlay

The extension keeps the verified upstream LATO.2 source at commit
`fbb1f5a5755e6db8700cf6922fd506830b7cdccd` unchanged. Setup may generate a
second, additive runtime tree with:

```python
from lato2_modly.portable import materialize_portable_runtime

report = materialize_portable_runtime(exact_source_dir, portable_source_dir)
assert report.portable_precision_env is True
```

The callable checks SHA-256 for every patched upstream file, copies through a
staging directory, writes `.modly-portable.json`, and atomically publishes the
copy. Calling it again validates and reuses that copy. A setup state file can
copy `report.to_dict()`; in particular it should retain
`"portable_precision_env": true`.

## What is upstream and what is portable

| Area | Portable change | Provenance and limitation |
|---|---|---|
| Sparse convolution | Adds `SPARSE_BACKEND=torch` and keeps `spconv`/`torchsparse` selectable | `lite.py` and `conv_torch.py` derive from the official LATO.2 Space commit `25ec65da46236e5ef46b88d9a510a0fe33b2bc63`. The Space reports a `6e-8` maximum absolute difference against spconv 2.3.8 for the submanifold operations LATO.2 instantiates. |
| Sparse attention | Adds `SPARSE_ATTN_BACKEND=sdpa`; FlashAttention and xFormers stay intact | PyTorch SDPA is applied independently to the same packed sequence boundaries. It is semantically block-diagonal but still needs checkpoint-level GPU parity testing. |
| Precision | Replaces hard-coded BF16 autocast in all four inference scripts | `LATO2_PRECISION=auto` uses `torch.cuda.is_bf16_supported()`: BF16 when true, otherwise FP16. Explicit values are `bfloat16` and `float16`. There is no new CLI flag, so all upstream CLI parameters remain unchanged. |
| Conditioning renderer | Open3D remains first choice | On Open3D import/context/render failure, `auto` falls back to a deterministic Pillow software renderer. It preserves camera controls and RGB shape, but is not pixel-equivalent to Filament and may change DINO conditioning. `LATO2_RENDERER=open3d` makes failure fatal; `software` explicitly selects the fallback for diagnostics. |
| Mesh voxelization | Builds only the pinned CPU function LATO.2 calls | `lato2_modly.ovoxel_cpu` compiles the exact `mesh_to_flexible_dual_grid_cpu` C++ source from TRELLIS.2 commit `75fbf018…` with its Eigen submodule commit `21e4582d…`. The wrapper avoids importing unrelated IO/rasterize/serialize/postprocess code. It does not approximate voxelization and does not need CUDA/nvcc. The complete o-voxel package remains installed on the exact upstream lane. |
| `torch_scatter.scatter_mean` | Uses the installed extension when importable, otherwise the official Space's PyTorch fallback | The fallback is additive and is not a reason to remove `torch-scatter` from the complete upstream dependency installation. |

The overlay applies to all upstream entry points without removing options or
renaming outputs:

- `scripts/e2e_inference.py`
- `scripts/vflow_inference.py`
- `scripts/vvae_inference.py`
- `scripts/tflow_inference.py`

Runtime should select the exact source on the native upstream lane and the
portable copy on compatibility lanes such as Windows/Turing or Linux ARM64. For
the portable copy, the safe defaults are `SPARSE_BACKEND=torch`,
`SPARSE_ATTN_BACKEND=sdpa`, and `LATO2_PRECISION=auto`. The generated source
retains the upstream backend choices, but those choices are executable only in
an environment where their exact native dependencies were installed; the
runtime rejects an unavailable profile instead of falling back silently.

### Minimal voxelizer build contract

After PyTorch is installed, setup prepares and installs the portable native
operator as follows:

```python
from lato2_modly.ovoxel_cpu import materialize_ovoxel_cpu_build

native = materialize_ovoxel_cpu_build(
    ovoxel_source_root=verified_trellis_source,
    eigen_source_root=verified_eigen_source,
    build_root=revision_dir / "native" / "ovoxel_cpu-build",
)
subprocess.run(
    [venv_python, *native.pip_install_args],
    check=True,
    env={**os.environ, "MAX_JOBS": "1"},
)
```

The source archives must already have passed setup's immutable URL, byte-size,
and SHA-256 checks. The helper separately validates the two copied o-voxel
source files, records a complete Eigen tree digest, creates an atomic reusable
build tree, and returns the checked pip arguments. Setup then verifies that
`lato2_ovoxel_cpu._C.mesh_to_flexible_dual_grid_cpu` imports. The build needs a
C++17 compiler: MSVC Build Tools on Windows or GCC/Clang on Linux/ARM64.

The materializer also copies an SHA-256-anchored license payload into the build
tree. Setuptools publishes `modly-lato2-ovoxel-cpu` 0.0.1.post2 with the Modly
wrapper MIT text, TRELLIS.2 MIT text, and the complete Eigen license/notice set
in the wheel's `.dist-info/licenses/` directory. A single exported build
identity covers the complete template, both pinned TRELLIS.2 source files, the
complete pinned Eigen tree, and every bundled license hash; it is recorded in
the materialization marker/report and dependency lock. Tests build a metadata
wheel and inspect its version, license members, `License-File` fields, and
build identity.

## Dependency closure

Each supported OS/architecture/Torch plan has two explicit exact-version
constraint lanes. `cp311` preserves upstream Modly's original CPython 3.11
closure; `cp312` is selected only for a validated 64-bit CPython 3.12
fingerprint. Python version, cache tag, SOABI, ABI flags, and pointer width are
included in the dependency-lock payload and digest. The resulting constraints
and native build caches are therefore distinct even where both lanes currently
pin the same package versions. The selected constraints are materialized in a
validated extension-owned cache and injected into every pip install, including
the local O-Voxel CPU add-on. Remote stages use explicit indexes and binary
wheels only; the exact profile's pinned FlashAttention source build is the sole
external sdist exception, while authenticated local native sources are
installed with `--no-index`. Setup verifies every constrained distribution
version, runs `pip check`, and then runs the applicable ABI/import/CUDA/render
smokes before publishing state.

The lock controls package names and versions but is not an artifact-hash
wheelhouse: not every third-party wheel or the external FlashAttention source
artifact has a wrapper-controlled SHA-256. The model and source archives are
separately size/SHA-256 pinned, and each authenticated native-source or
portable-runtime tree has an extension-controlled complete-tree digest.
The eight original CPython 3.11 requirement-set hashes remain frozen, and the
eight CPython 3.12 plans have separate lock identities. This source-level
validation is not a substitute for the pending real-GPU platform tests.

Authenticated asset and source-tree validation rejects symlinks, hardlinks,
and Windows reparse aliases before reuse or repair.

## Runtime diagnostics

Input normalization is geometry-only and calls trimesh with
`skip_materials=True`; OBJ/PLY material files and glTF/GLB images are not
opened. External geometry buffers referenced by glTF/GLB are copied into a
controlled staging tree only after confinement, alias, file-identity, and
aggregate-size checks. The main input plus staged buffer copies are limited to
2 GiB. Embedded GLB buffers and valid base64 data-buffer URIs are supported.

Successful `upstream.log` files are bounded, sanitized, and end with the
effective `software`, `open3d`, or `no-render` renderer. Inference and
post-inference output failures can atomically publish only a sanitized log and
`run-failure.json` under a `failed-<node-id>-…` directory; partial geometry and
upstream sidecars are never copied into that diagnostic. Protocol log/progress
callbacks are still emitted, but Modly 0.4.2 does not surface every such
callback in every UI flow, so the persistent files are authoritative.

## Verification boundary

The repository tests cover pinned-source refusal, atomic/idempotent
materialization, full-tree integrity, the minimal native build plan, syntax
compilation, the narrow o-voxel import surface, SDPA sequence semantics when
PyTorch is available, and deterministic renderer output.
The pure sparse implementation has the official Space's local spconv comparison
evidence. A complete checkpoint generation still requires hardware tests on each
target lane (Windows/Turing and Linux ARM64 CUDA); until those pass, the portable
lane must be described as implemented but awaiting full-model platform validation.
