# LATO.2 for Modly

A Python process extension that exposes the complete pinned LATO.2 inference
pipeline as four mesh workflow nodes in upstream Modly.

> **License notice:** the exact native profile installs NVIDIA
> [nvdiffrast](https://github.com/NVlabs/nvdiffrast), whose NVIDIA Source Code
> License (1-Way Commercial), Section 3.3, limits the Work and derivatives to
> non-commercial research or evaluation. Read
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before installation or use.

> **DINOv2 archive notice:** the selected DINOv2 register-token path is
> Apache-2.0, but the complete pinned source archive also contains imported
> Cell-DINO and X-Ray-DINO modules with separate terms, including
> noncommercial-research material. This extension does not download or select
> their checkpoints. See the DINOv2 section of
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the exact scope, license
> copies, and required X-Ray DINO notice.

## Release status

This is a pre-hardware release candidate. The implementation targets upstream
Modly 0.4.2 audited at commit
`8d08249fbd4f2678d923c213a8cb2902d57611c9`, but a full **Install from GitHub → setup/Repair → UI
workflow → real checkpoint generation** has not yet been completed on the
target GPUs. In particular, Windows/Turing and Linux ARM64 full-model parity
remain unverified. The compatibility rows below describe implemented setup
routes, not completed hardware certification.

The highest completed validation tier is the repository's automated
contract/unit suite and strict extension validation on Linux ARM64 with
CPython 3.12. This proves setup-contract selection and lock separation, not a
real installation. Native compilation, real-GPU dependency setup, and
checkpoint generation are the next tier and remain pending on every platform
row below.

## Usage

### Nodes

Every invocation accepts one mesh and publishes one Modly-compatible GLB while
retaining the original upstream artifacts beside it.

| Node | ID | Upstream path | Returned GLB |
| --- | --- | --- | --- |
| LATO.2 End-to-End | `lato2-e2e` | Conditioning render → V-Flow vertices → T-Flow topology | Triangle mesh when topology is decoded; point GLB if no faces are produced |
| LATO.2 V-Flow | `lato2-vflow` | Vertex generation, with optional V-VAE reconstruction sidecars | Point GLB generated from `input_pred.ply` |
| LATO.2 V-VAE | `lato2-vvae` | Encode and reconstruct the input vertices | Point GLB generated from `input_recon.ply` |
| LATO.2 T-Flow | `lato2-tflow` | Generate connectivity for the input vertices | Triangle mesh when topology is decoded; point GLB if no faces are produced |

Accepted inputs are `.obj`, `.glb`, `.gltf`, `.ply`, `.stl`, and
`.off`. The input must resolve to finite triangle geometry with at least
three vertices and one face. Normalization is deliberately geometry-only:
OBJ/PLY material references and glTF/GLB images are not opened, and materials
and textures are not carried into the normalized input. External glTF/GLB
buffers required by geometry must be regular, non-aliased files beneath the
mesh file's directory; the main file plus staged buffer copies may total at
most 2 GiB. Embedded GLB buffers and correctly formed base64 data-buffer URIs
remain supported. Unsafe, missing, escaping, URL, symlink/reparse, or hardlink
references fail input validation before trimesh runs.

Modly's process contract supplies one mesh per run, so upstream `batch_size`
and `num_samples` are fixed to `1`.

## Installation

### Prerequisites

- A 64-bit CPython runtime selected by Modly: upstream Modly 0.4.2's CPython
  3.11 lane remains supported unchanged, and CPython 3.12 uses a separate
  dependency/ABI lane. Setup rejects every other Python implementation,
  bitness, or major/minor version.
- A supported NVIDIA CUDA GPU. CPU, ROCm, macOS, and Windows ARM64 are not
  supported by this release.
- An NVIDIA driver compatible with the selected PyTorch lane: cu124 for the
  exact/x64 portable profiles, cu126 for Linux ARM64 below SM 10.0, or cu128
  for SM 10.0+. Modly 0.4.2 does not expose enough driver detail to select a
  CUDA 13 runtime safely, so this release does not select cu130 automatically.
- Internet access during setup. Inference itself is forced offline.
- About **4.82 GB (4.49 GiB) of immutable model/source downloads**. Before
  downloading, setup requires the remaining asset bytes plus 2 GiB of
  extraction headroom on the Models volume. When an environment must be
  rebuilt, it also reserves 8 GiB there for portable dependency/build caches
  or 12 GiB for exact-upstream caches, and separately reserves 12 GiB on the
  Extensions volume for the transactional portable environment or 16 GiB for
  exact-upstream. Requirements are summed when Models and Extensions share a
  filesystem. These are conservative preflight free-space requirements, not
  estimates of final installed size.
- For the exact upstream profile: CUDA Toolkit **12.4** with `nvcc` and a
  C++17 compiler. A newer default toolkit may remain installed side by side,
  but `CUDA_HOME` or `CUDA_PATH` must identify the 12.4 installation; setup
  accepts only a candidate whose `nvcc --version` reports 12.4. Windows
  requires Visual Studio 2022 Build Tools with Desktop C++, MSVC v143, a
  Windows SDK, and working `/openmp` support. Linux requires a working C++17
  GCC/OpenMP toolchain (`-fopenmp` and `libgomp1`) or a separately validated
  Clang/libomp equivalent, plus glibc 2.31 or newer on x86_64.
- Exact Open3D rendering on Linux also needs `libegl1`, `libgl1`,
  `libglu1-mesa`, `libgomp1`, `libx11-6`, `libxext6`, `libxrender1`,
  `libxrandr2`, `libxinerama1`, `libxcursor1`, `libxi6`, `libsm6`, and
  `libice6`. Setup creates and reads a 16×16 Open3D `OffscreenRenderer` under
  `EGL_PLATFORM=surfaceless` and a private `XDG_RUNTIME_DIR`; a missing
  EGL/Filament runtime therefore fails exact setup before state is published.
- The portable profile still compiles the exact CPU
  `mesh_to_flexible_dual_grid` operation and therefore needs a C++17
  compiler, but it does not require `nvcc`.

Upstream reports roughly 8 GB of VRAM for its reference run. That is an
upstream estimate, not a guarantee for every input, parameter set, GPU
architecture, or backend. Leave headroom for CUDA and rendering allocations.

### Install from GitHub

1. Open Modly's **Extensions** page.
2. Choose **Install from GitHub**.
3. Enter `https://github.com/DrHepa/modly-lato2-extension`.
4. Keep Modly open while setup installs the isolated `venv`, resolves the
   configured `models_dir`, downloads and verifies the pinned assets, and
   builds the selected dependency profile.
5. Open **Workflows**, connect a mesh-producing node to one of the four LATO.2
   nodes, keep the defaults for the first run, and run the workflow.

There is no separate Models-page weight action for this process extension.
Setup owns its immutable asset snapshot because upstream Modly 0.4 does not
expose the model-weight UI lifecycle to process extensions.

### Repair and updates

**Repair** is safe to rerun. It validates the asset inventory, source trees,
dependency lock, Python ABI, CUDA plan, and setup state. Every running process
holds a shared lease on setup state, so Repair cannot mutate its venv or model
snapshot; Repair waits up to 30 seconds and otherwise exits safely for a later
retry. Concurrent inference readers remain allowed. A complete snapshot is
reused without network access; interrupted downloads use owned `.part` files
and resume only after validating the HTTP range. Corrupt or incomplete owned
artifacts are repaired, and readiness state is written last.

Modly 0.4.2 cannot terminate an already spawned Python process before it
renames an extension directory for **Update** or **Uninstall**. Do not update
or uninstall this extension while one of its workflows is active. This host
limitation is separate from Repair, which is coordinated by the shared lease
above.

The immutable snapshot lives under:

```text
<models_dir>/modly-lato2-extension/lato2/revisions/lato-a91090e-dino-7764ea0f/
```

It contains seven LATO.2 checkpoints, the exact LATO.2 source, the exact
DINOv2 source/checkpoint cache, and revision-owned runtime data. Updating or
reinstalling the extension does not redownload a snapshot that still passes
size, SHA-256, source-tree, and readiness-marker validation.

## Requirements and compatibility

### Exact and portable profiles

The installer keeps the exact pinned upstream source unchanged. Where the
published upstream wheel/toolchain set is not viable, it can additionally
materialize a separately fingerprinted portable copy; the node-level
`backend` selector never silently falls back to an unavailable choice.

| Area | Exact upstream | Portable compatibility |
| --- | --- | --- |
| Sparse convolution | spconv 2.3.8 | Pure-PyTorch sparse operations derived from the official LATO.2 Space |
| Sparse attention | FlashAttention 2 on supported Linux GPUs, otherwise xFormers | PyTorch SDPA with independent packed-sequence boundaries |
| Precision | Upstream BF16 behavior | `auto` selects BF16 when supported, otherwise FP16 |
| Conditioning render | Open3D/Filament | Open3D first when available; deterministic Pillow renderer on import/context/render failure |
| Voxelization | Full pinned O-Voxel native stack | Exact TRELLIS.2 `mesh_to_flexible_dual_grid_cpu` compiled as a narrow C++ extension |

The portable profile preserves all four inference entry points, parameters,
checkpoint formats, and upstream sidecars. It does **not** claim numerical or
pixel equivalence: SDPA still needs checkpoint-level parity testing, and the
software renderer can change DINOv2 conditioning. See
[PORTABLE_BACKEND.md](PORTABLE_BACKEND.md) for provenance and validation
boundaries.

The setup profile defaults from the detected platform/GPU. Advanced users may
set `MODLY_LATO2_DEPENDENCY_PROFILE=exact-upstream` or
`MODLY_LATO2_DEPENDENCY_PROFILE=portable` before starting Modly and running
Install/Repair. An unsupported explicit profile fails with an actionable
message; it is not downgraded silently.

> **Before selecting the exact profile:** it downloads and builds nvdiffrast
> and NVIDIA-derived cubvh material subject to the non-commercial
> research/evaluation limitations described in
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The portable profile does
> not install that exact native stack.

### Platform matrix

| Platform | Automatic route | Required local toolchain | Validation status |
| --- | --- | --- | --- |
| Windows x64 + NVIDIA CUDA | Portable compatibility by default; exact upstream is an explicit opt-in on BF16-capable GPUs | Exact: CUDA Toolkit 12.4 + VS 2022 C++/MSVC v143/Windows SDK. Portable: VS 2022 C++ | Implemented; full setup and generation not yet hardware-validated, and exact Windows is not certified |
| Linux x86_64 + NVIDIA CUDA, SM 8.x/9.x | Exact upstream; FlashAttention on supported GPUs, xFormers otherwise | CUDA Toolkit 12.4 + C++17 compiler; glibc 2.31+ | Implemented; full setup and generation not yet hardware-validated |
| Linux x86_64 + NVIDIA CUDA, pre-Ampere | Portable compatibility for FP16-capable inference | C++17 compiler | Implemented; full setup and generation not yet hardware-validated |
| Linux ARM64 + NVIDIA CUDA | Portable cu126 for SM <10.0; portable cu128 for SM 10.0+ | C++17 compiler; glibc 2.28+ for both lanes | Experimental; full setup and generation not yet hardware-validated |
| Linux/Windows x64, SM 10.0+ | Portable cu128 compatibility | C++17 compiler | Experimental; full setup and generation not yet hardware-validated |
| CPU-only, ROCm, macOS, Windows ARM64 | Unsupported | — | Setup rejects the platform |

Package availability alone is not treated as platform validation. Do not
remove the exact native dependencies to make an unsupported profile appear to
install: the exact profile intentionally provisions Open3D, spconv,
torch-scatter, xFormers/FlashAttention, nvdiffrast, CuMesh, FlexGEMM, and the
full O-Voxel package and verifies their native symbols.

### Dependency locks and direct pins

- Build/bootstrap: pip 25.1.1, setuptools 80.9.0, wheel 0.45.1, and
  packaging 25.0.
- Shared: NumPy 2.2.6, trimesh 4.10.1, tqdm 4.67.1, Pillow 12.0.0,
  Ninja 1.13.0, psutil 7.1.3, OpenCV headless 4.12.0.88,
  huggingface-hub 0.36.0, plyfile 1.1, zstandard 0.25.0, easydict 1.13,
  einops 0.8.1, and filelock 3.20.0. Open3D 0.19.0 is installed on x64.
- cu124/cu126: PyTorch 2.6.0 and torchvision 0.21.0. The Linux ARM64
  torchvision wheel uses the upstream version without a local CUDA suffix.
- cu128: PyTorch 2.9.1+cu128 and torchvision 0.24.1+cu128 on x64/Windows.
  The Linux ARM64 torchvision wheel is version 0.24.1 without a local CUDA
  suffix.
- Both profiles build `modly-lato2-ovoxel-cpu` 0.0.1.post2 locally from the
  pinned TRELLIS.2/Eigen inputs. Its build identity covers the complete
  template and license payload, so a changed input invalidates environment
  reuse without invalidating or redownloading the separate model snapshot.
- Exact native: cumm-cu124 0.7.11, spconv-cu124 2.3.8,
  torch-scatter 2.1.2 for PyTorch 2.6/cu124, xFormers 0.0.29.post2,
  Triton 3.2.0 (or triton-windows 3.2.0.post21), and FlashAttention
  2.7.4.post1 on the selected Linux route, plus the source revisions listed
  under Reproducibility pins.

Setup carries two explicit Python dependency lanes. `cp311` preserves the
original exact-version constraint closures for upstream Modly; `cp312` is a
new, independently selected lane for Modly runtimes embedding CPython 3.12.
Each lane covers the eight audited Windows x64, Linux x64, Linux ARM64,
cu124/cu126/cu128, and Windows/Linux exact plans. The Python version, cache
tag, SOABI, ABI flags, and pointer width are part of the dependency-lock
payload and digest, so constraints, native build caches, and environments can
never be reused across `cp311` and `cp312`. The selected closure is
materialized atomically in the extension-owned cache and supplied to every
pip install stage, including the locally built CPU O-Voxel add-on. An
unaudited Python/OS/architecture/Torch combination fails closed.

Remote package stages use explicit fixed indexes and binary wheels only,
except for the pinned FlashAttention 2.7.4.post1 external source build. Native
projects materialized from the authenticated source archives are installed
locally with `--no-index`. User pip index/config overrides are neutralized;
only a validated cache and credential-free proxy/CA settings can reach remote
stages. Setup then checks the installed metadata version of every constrained
distribution, requires `pip check`, and runs import, ABI, renderer, and CUDA
smoke tests before publishing state. The exact profile additionally verifies
the native symbols from every upstream extension and performs a real Open3D
offscreen render.

This is a complete **version closure**, not an artifact-hash wheelhouse:
third-party PyPI/PyTorch wheels and the FlashAttention source artifact do not
all have wrapper-controlled SHA-256 pins. Model/source downloads and the
native source trees are independently pinned by byte size, SHA-256, immutable
revision, and complete-tree digest. The original eight CPython 3.11
requirement-set hashes remain frozen, while CPython 3.12 resolves through its
own lock identity. This source-level closure validation does not replace the
pending real-GPU and Modly UI workflow tests described under Release status.

Native and portable source archives are also size/SHA-256 pinned. Their
materialized cache trees are checked against extension-controlled complete-tree
digests, including path/type validation that rejects symlinks, hardlinks, and
Windows reparse aliases. Repair reconstructs a changed tree from the
authenticated ZIP rather than trusting a marker stored beside mutable sources.

## Parameters

The four common parameters are identical on every node:

| Parameter | Default | Allowed values | Effect |
| --- | --- | --- | --- |
| `backend` | `auto` | `auto`, `upstream`, `portable` | Selects the installed runtime tree. `auto` uses setup's recorded default; an unavailable explicit choice fails. |
| `precision` | `auto` | `auto`, `bfloat16`, `float16` | Exact upstream accepts Auto/BFloat16. Portable Auto selects the best supported CUDA dtype; Float16 requires portable support. |
| `seed` | `42` | integer 0…4,294,967,295 | Unsigned 32-bit seed accepted by the upstream NumPy `RandomState` path and forwarded to PyTorch and NumPy. |
| `num_workers` | `4` | integer 0…64 | Upstream DataLoader worker count. Set zero to disable multiprocessing on constrained Windows/headless hosts. |

### End-to-End

| Parameter | Default | UI range/options | Effect |
| --- | --- | --- | --- |
| `inference_threshold` | `0.5` | 0…1, step 0.01 | V-VAE occupancy threshold. |
| `vflow_steps` | `24` | integer 1…2,147,483,647 | Vertex-flow Euler steps. |
| `cfg_strength` | `3.0` | finite float, step 0.1 | Classifier-free guidance strength. |
| `rescale_t` | `1.0` | finite float, step 0.05 | Vertex-flow timestep rescaling. |
| `vert_num` | `2000` | integer 1…2,147,483,647 | Fixed vertex-count conditioning target. |
| `use_gt_vert_count` | `false` | `true` / `false` | Uses the quantized input vertex count instead of `vert_num`. |
| `scaler` | `1.0` | finite float, step 0.05; shown when input count is enabled | Multiplies the input-derived count. |
| `min_verts` | `200.0` | finite float, step 1; shown when input count is enabled | Lower clamp for the scaled count. |
| `max_verts` | `5000.0` | finite float, step 1; shown when input count is enabled | Upper clamp for the scaled count; must be at least `min_verts`. |
| `tflow_steps` | `50` | integer 1…2,147,483,647 | Topology-flow Euler steps. |
| `edge_threshold` | `0.0` | finite float, step 0.05 | T-VAE edge-logit threshold. |
| `chunk_size` | `20000` | integer 1…2,147,483,647, step 1000 | T-VAE topology decode chunk size. Smaller values can reduce peak decode memory. |
| `fill_quad_rings` | `true` | `true` / `false` | Splits chordless four-vertex rings into triangles. |
| `render_azimuth` | `45.0` | −360…360°, step 1 | DINOv2 conditioning-view azimuth. |
| `render_elevation` | `30.0` | −90…90°, step 1 | DINOv2 conditioning-view elevation. |
| `img_res` | `518` | integer 1…4096, step 14 | Square conditioning-render resolution. |

### V-Flow

| Parameter | Default | UI range/options | Effect |
| --- | --- | --- | --- |
| `pc_sample_number` | `819200` | integer 1…2,147,483,647 | Surface/VDF samples used by the upstream encoder. |
| `sample_type` | `dora` | `dora`, `uniform` | Upstream point-sampling strategy. |
| `inference_threshold` | `0.5` | 0…1, step 0.01 | V-VAE occupancy threshold. |
| `reconstruct` | `false` | `true` / `false` | Also writes V-VAE reconstruction PLY sidecars. |
| `sample_posterior` | `true` | `true` / `false`; shown when reconstruction is enabled | Samples the reconstruction posterior instead of taking its mode. |
| `vflow_steps` | `24` | integer 1…2,147,483,647 | Vertex-flow Euler steps. |
| `cfg_strength` | `3.0` | finite float, step 0.1 | Classifier-free guidance strength. |
| `rescale_t` | `1.0` | finite float, step 0.05 | Vertex-flow timestep rescaling. |
| `vert_num` | `2000` | integer 1…2,147,483,647 | Fixed vertex-count conditioning target. |
| `use_gt_vert_count` | `false` | `true` / `false` | Uses the quantized input vertex count instead of `vert_num`. |
| `scaler` | `1.0` | finite float, step 0.05; shown when input count is enabled | Multiplies the input-derived count. |
| `min_verts` | `200.0` | finite float, step 1; shown when input count is enabled | Lower clamp for the scaled count. |
| `max_verts` | `5000.0` | finite float, step 1; shown when input count is enabled | Upper clamp; must be at least `min_verts`. |
| `render_azimuth` | `45.0` | −360…360°, step 1 | DINOv2 conditioning-view azimuth. |
| `render_elevation` | `30.0` | −90…90°, step 1 | DINOv2 conditioning-view elevation. |
| `img_res` | `518` | integer 1…4096, step 14 | Square conditioning-render resolution. |

### V-VAE

| Parameter | Default | UI range/options | Effect |
| --- | --- | --- | --- |
| `pc_sample_number` | `819200` | integer 1…2,147,483,647 | Surface/VDF samples used by the encoder. |
| `sample_type` | `dora` | `dora`, `uniform` | Upstream point-sampling strategy. |
| `inference_threshold` | `0.5` | 0…1, step 0.01 | V-VAE occupancy threshold. |
| `sample_posterior` | `true` | `true` / `false` | Samples the posterior instead of taking its mode. |

### T-Flow

| Parameter | Default | UI range/options | Effect |
| --- | --- | --- | --- |
| `tflow_steps` | `50` | integer 1…2,147,483,647 | Topology-flow Euler steps. |
| `use_cond` | `true` | `true` / `false` | Conditions topology generation on the active voxel field; false uses the learned null token. |
| `edge_threshold` | `0.0` | finite float, step 0.05 | T-VAE edge-logit threshold. |
| `chunk_size` | `20000` | integer 1…2,147,483,647, step 1000 | T-VAE topology decode chunk size. |
| `fill_quad_rings` | `true` | `true` / `false` | Splits chordless four-vertex rings into triangles. |
| `save_voxel_field` | `true` | `true` / `false`; shown when conditioning is enabled | Preserves the active-voxel point-cloud sidecar. |

## Outputs and sidecars

Each successful run publishes:

```text
<workspaceDir>/Workflows/LATO2/<node-id>-<UTC timestamp>-<token>/result.glb
```

`result.glb` is the path returned to Modly. Mesh-producing results contain
triangle faces. V-Flow, V-VAE, and no-topology fallbacks are encoded as GLB
point geometry so the process still returns a stable `mesh` artifact.

The same run directory keeps:

- `run.json`, recording the node ID, selected backend, precision, attention
  backend, result kind (`mesh` or `points`), topology-decoded status for E2E
  and T-Flow, effective conditioning renderer (`open3d`, `software`, or
  `no-render`), and effective parameters. A legitimate no-face fallback is
  also emitted as a process log; an unsafe, unreadable, or empty topology file
  fails closed instead of being hidden by that fallback.
- `upstream.log`, a bounded record containing at most the final 2 MiB of the
  upstream process output, plus no more than 100 bytes of framing/footer. The
  persisted log replaces extension-controlled local paths with labels and
  redacts common credential forms. Its final footer records the effective
  renderer. A portable software-renderer fallback therefore remains visible
  here and in `run.json`, even if its early upstream warning fell outside the
  retained tail.
- End-to-End upstream sidecars: predicted PLY/OBJ, integer-coordinate PLY/OBJ,
  and DINOv2 render.
- V-Flow upstream sidecars: GT, generated, and optional reconstructed PLY
  pairs plus the render.
- V-VAE upstream sidecars: GT and reconstructed PLY pairs.
- T-Flow upstream sidecars: predicted OBJ or PLY fallback, known-vertex PLY,
  and optional voxel field PLY.

The input copy is temporary and removed after the run. A durable run directory
is published atomically only after upstream inference and GLB validation
succeed.

For `INFERENCE_FAILED`, `OUTPUT_MISSING`, `OUTPUT_CONVERSION`, or
`OUTPUT_INVALID`, the extension makes a best-effort atomic publication when a
verified upstream log exists:

```text
<workspaceDir>/Workflows/LATO2/failed-<node-id>-<UTC YYYYMMDDTHHMMSS>-<token>/
  upstream.log
  run-failure.json
```

The failure directory contains only the bounded, sanitized log and
`run-failure.json`. That JSON records the real error code/stage, node, backend,
and precision without paths, parameters, or secrets; no input, generated
result, or upstream sidecar is copied into the directory. If safe diagnostic
publication is impossible, the original processing error is still returned
and no partial diagnostic directory is kept. The extension also emits
protocol log/progress messages, but Modly 0.4.2's current IPC path
does not surface every process-extension callback in every UI flow; UI
visibility is therefore not guaranteed. `run.json` and the successful
`upstream.log` are the durable source of renderer truth, while `run-failure.json`
and its adjacent log are the durable failure record when available.

## Limitations

- LATO.2 normalizes the mesh into a centered `[-0.5, 0.5]` frame. The
  original world transform and physical scale are not restored in generated
  artifacts.
- This is geometry/topology generation, not attribute-preserving editing.
  Materials, textures, UVs, authored normals, rigs, skin weights, blend
  shapes, and animation are not preserved.
- Generated connectivity is triangular. `fill_quad_rings` fills a detected
  four-vertex ring by splitting it into triangles; it does not produce a quad
  retopology.
- Upstream warns that generated meshes can contain holes or incorrect
  connectivity. Watertightness, manifoldness, face orientation, and target
  vertex count are not guaranteed.
- The portable renderer is not pixel-equivalent to Open3D/Filament and may
  change image-conditioned V-Flow/E2E results. Portable SDPA has not yet been
  proven checkpoint-equivalent to the exact attention kernels.
- Current upstream Modly does not expose cooperative cancellation to Python
  process extensions. No partial result is returned while upstream inference
  is running.
- Very large point-sample counts, vertex targets, topology chunks, or render
  resolutions can exhaust RAM/VRAM or make a run impractically slow even when
  the UI accepts the value.

## Troubleshooting

- **Setup reports `NVCC_MISSING` or `NVCC_VERSION_MISMATCH`:** the exact
  profile needs the CUDA 12.4 Toolkit and its `nvcc`, not only an NVIDIA
  driver. Install/activate that toolkit, then run Repair.
- **Setup reports an MSVC/C++ error:** install the required compiler workload
  described above, restart Modly so it inherits the toolchain environment, and
  run Repair.
- **`BACKEND_UNAVAILABLE` or `PRECISION_UNAVAILABLE`:** choose Auto, or run
  Repair with the intended dependency profile. The exact backend does not
  accept Float16.
- **`SETUP_INVALID` or missing model storage:** keep Modly open, confirm its
  configured Models path is writable, and run Repair. Do not move one revision
  directory by hand.
- **`REQUEST_INPUT`:** use a finite triangle mesh. For `.gltf`/`.glb`, keep
  every external geometry buffer beneath the mesh directory, remove URL or
  absolute references, and keep the complete staged geometry bundle below
  2 GiB. Materials and images are intentionally ignored.
- **CUDA out of memory:** close other GPU workloads, return to default
  parameters, reduce `vert_num`, `pc_sample_number`, `chunk_size`, or
  `img_res` as relevant, and keep `num_workers=0`.
- **Inference or output-stage failure:** inspect the newest matching
  `failed-<node-id>-…` directory under `Workflows/LATO2`. `run-failure.json`
  identifies the real stage; `upstream.log` contains the sanitized retained
  tail. A diagnostic is best-effort and is omitted if its source log or target
  directory cannot be verified safely.
- **No faces were decoded:** inspect `upstream.log` and the PLY sidecars. The
  returned GLB is intentionally a point result when LATO.2 produces no
  topology.
- **Renderer failure:** Auto on the portable profile can use the software
  renderer. `LATO2_RENDERER=open3d` makes an Open3D failure explicit;
  `LATO2_RENDERER=software` selects the diagnostic fallback.

## Uninstalling model assets

Removing the extension in Modly does not delete the revision-owned model
snapshot. After uninstalling, manually remove only:

```text
<models_dir>/modly-lato2-extension/
```

Do not delete the whole Modly Models directory. Removing this extension-owned
folder discards the checkpoints, pinned source archives/trees, and runtime
caches; a future installation will download them again.

## Reproducibility pins

| Component | Immutable revision |
| --- | --- |
| LATO.2 source | `fbb1f5a5755e6db8700cf6922fd506830b7cdccd` |
| LATO.2 checkpoints, `0x4c48/LATO.2` | `a91090e8077b9318ab87ac08fd9eb905903d4515` |
| DINOv2 source | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| Official LATO.2 Space portable reference | `25ec65da46236e5ef46b88d9a510a0fe33b2bc63` |
| nvdiffrast | `253ac4fcea7de5f396371124af597e6cc957bfae` |
| CuMesh | `12289e1062f0603f2f0d0771b02e1395d247f26f` |
| cubvh | `ce92267a24ef6ad7d2c8ccbc2ae2c021a6597e70` |
| FlexGEMM | `6dd94a859c26ee8246888502eada3dd8ad85532e` |
| TRELLIS.2 / O-Voxel | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` |

Every downloaded model/source asset has a fixed byte size and SHA-256 in the
installer inventory. Runtime sets the Hugging Face and Transformers offline
flags, points Torch Hub at the local DINOv2 cache, removes Hub tokens from the
inference child, and does not perform implicit model downloads.

## Credits

- **Modly extension integration:** DrHepa —
  [extension repository](https://github.com/DrHepa/modly-lato2-extension).
- **LATO.2 model, source, and checkpoints:** Hang Long, Tianhao Zhao, Junkai
  Lin, Youjia Zhang, Huipeng Guo, Rendong Liang, Jiale Xu, Jozef Hladký,
  Matthias Nießner, Yuanming Hu, Wei Yang, and their contributors —
  [upstream repository](https://github.com/LoHhhha/LATO.2) and
  [paper](https://arxiv.org/abs/2607.10623).
- **DINOv2:** Meta AI Research and contributors —
  [upstream repository](https://github.com/facebookresearch/dinov2).
- **Modly host:** Lightning Pixel —
  [upstream repository](https://github.com/lightningpixel/modly).

These credits identify separate works; repository ownership does not transfer
authorship of the model, host, or integration.

## License

The extension wrapper is Copyright (c) 2026 DrHepa and licensed under the MIT
License in [LICENSE](LICENSE). LATO.2 source and checkpoint release are MIT.
The selected DINOv2 core source and register-token checkpoint are Apache-2.0;
the complete pinned DINOv2 archive also carries separately licensed Cell-DINO,
X-Ray-DINO, and embedded CLIP material. Native dependencies retain their own
terms, including the non-commercial research/evaluation restriction in
nvdiffrast's license and the separately noticed NVIDIA-derived cubvh material.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the complete copies
under [LICENSES/](LICENSES/).
