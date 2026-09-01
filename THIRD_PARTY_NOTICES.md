# Third-Party Notices

This document separates the **LATO.2 for Modly** integration written by
DrHepa from the upstream model, checkpoints, source archives, and native
libraries that its installer obtains. The extension's MIT license does not
replace or relax any third-party license.

## Extension wrapper

Copyright (c) 2026 DrHepa. The integration code is licensed under the MIT
License in [`LICENSE`](LICENSE), except where a file or this notice identifies
third-party material.

## Important nvdiffrast use restriction

The upstream-compatible native runtime installs NVIDIA's `nvdiffrast` at
revision `253ac4fcea7de5f396371124af597e6cc957bfae`. It is **not MIT-licensed**.
It is covered by the **NVIDIA Source Code License (1-Way Commercial)**.
Section 3.3 limits use of the Work and derivative works to non-commercial
research or evaluation, with the separate exception stated there for NVIDIA
and its affiliates. Read the complete terms before using or redistributing
that runtime:

- [`LICENSES/nvdiffrast-NVIDIA-Source-Code-License.txt`](LICENSES/nvdiffrast-NVIDIA-Source-Code-License.txt)
- [Upstream nvdiffrast license](https://github.com/NVlabs/nvdiffrast/blob/253ac4fcea7de5f396371124af597e6cc957bfae/LICENSE.txt)

The restriction belongs to nvdiffrast; it does not change the MIT license of
the Modly wrapper. A use that combines the components must nevertheless
comply with every applicable license.

On Linux x86_64 GPUs with compute capability 8.x or 9.x, the automatic setup
route selects this exact native profile. Select the separately fingerprinted
`portable` dependency profile before Install/Repair if these NVIDIA-licensed
components are not appropriate for the intended use. Selecting portable does
not change the terms of any exact-profile copy already downloaded or retained.

The pinned CuMesh dependency also brings a cubvh source bundle that contains
separately noticed NVIDIA-derived material. Its bundled NVIDIA Source Code
License likewise contains a Section 3.3 non-commercial research/evaluation
limitation. See
[`LICENSES/cubvh-NVIDIA-Source-Code-License.txt`](LICENSES/cubvh-NVIDIA-Source-Code-License.txt)
in addition to cubvh's MIT notice. Two compiled cubvh headers have their own
permissive terms: NVIDIA `gpu_memory.h` uses a BSD 3-Clause notice, while
`pcg32.h` carries the Apache-2.0 PCG/Wenzel Jakob attribution. Exact notices
are reproduced in
[`LICENSES/cubvh-NVIDIA-BSD-3-Clause.txt`](LICENSES/cubvh-NVIDIA-BSD-3-Clause.txt)
and
[`LICENSES/cubvh-PCG-Apache-2.0-NOTICE.txt`](LICENSES/cubvh-PCG-Apache-2.0-NOTICE.txt);
the standard Apache-2.0 terms are in
[`LICENSES/DINOv2-Apache-2.0.txt`](LICENSES/DINOv2-Apache-2.0.txt).

The pinned nvdiffrast archive also retains its upstream
`samples/data/NOTICE.txt`. That notice covers sample-only environment-map and
Earth mesh/texture material under separate Wave Engine MIT and TurboSquid 3D
Model terms. Those samples are not imported by the Modly inference path, but
their notice remains beside them in the verified source cache. See the
[notice at the pinned nvdiffrast revision](https://github.com/NVlabs/nvdiffrast/blob/253ac4fcea7de5f396371124af597e6cc957bfae/samples/data/NOTICE.txt)
before reusing or redistributing the sample data.

## DINOv2 archive contains separately licensed modules

The extension selects only the Apache-2.0 `dinov2_vitl14_reg` backbone and its
register-token checkpoint. It nevertheless installs the complete pinned
DINOv2 source archive. That archive's `hubconf.py` imports Cell-DINO and
X-Ray-DINO modules, and the tree carries licenses in addition to DINOv2's main
Apache-2.0 license. No Cell-DINO or X-Ray-DINO checkpoint is downloaded or
selected by this extension, but their code and license files remain present in
the installed source tree.

- `LICENSE_CELL_DINO_CODE` contains the Creative Commons Attribution 4.0
  International text. The pinned upstream README describes the Cell-DINO code
  as “CC BY NC”; because that label and the bundled license text are not the
  same, users should review the exact upstream release before relying on that
  module. Exact copy:
  [`LICENSES/DINOv2-Cell-DINO-Code-CC-BY-4.0.txt`](LICENSES/DINOv2-Cell-DINO-Code-CC-BY-4.0.txt).
- `LICENSE_CELL_DINO_MODELS` is the FAIR Noncommercial Research License and
  incorporated Acceptable Use Policy. Exact copy:
  [`LICENSES/DINOv2-Cell-DINO-Models-FAIR-Noncommercial.txt`](LICENSES/DINOv2-Cell-DINO-Models-FAIR-Noncommercial.txt).
- `LICENSE_XRAY_DINO_MODEL` is the X-Ray DINO Research License. It limits the
  covered Materials and their outputs/results to noncommercial research uses,
  includes clinical, trade-control, redistribution, and other restrictions,
  and requires this notice in distributed copies: “Materials are licensed
  under the X-Ray DINO Research License, Copyright © Meta Platforms, Inc. All
  Rights Reserved.” Exact license and notice copies:
  [`LICENSES/DINOv2-X-Ray-DINO-Research-License.txt`](LICENSES/DINOv2-X-Ray-DINO-Research-License.txt)
  and
  [`LICENSES/DINOv2-X-Ray-DINO-NOTICE.txt`](LICENSES/DINOv2-X-Ray-DINO-NOTICE.txt).
- The embedded DINOv2 `thirdparty/CLIP` code retains its MIT license. Exact
  copy: [`LICENSES/DINOv2-CLIP-MIT.txt`](LICENSES/DINOv2-CLIP-MIT.txt).

Cell-DINO is credited upstream to Théo Moutakanni, Camille Couprie, Seungeun
Yi, Elouan Gardes, Piotr Bojanowski, Hugo Touvron, Michael Doron, Zitong S.
Chen, Nikita Moshkov, Mathilde Caron, Armand Joulin, Wolfgang M. Pernice, Juan
C. Caicedo, and their contributors. The complete attribution and source links
remain in the pinned DINOv2 README.

## Model and inference sources

| Component | Exact source | License | Purpose |
| --- | --- | --- | --- |
| LATO.2 source | [`LoHhhha/LATO.2`](https://github.com/LoHhhha/LATO.2/tree/fbb1f5a5755e6db8700cf6922fd506830b7cdccd) at `fbb1f5a5755e6db8700cf6922fd506830b7cdccd` | MIT, with embedded MIT attributions retained in source headers | The four upstream inference paths and model definitions. Copies/notices: [`LICENSES/LATO.2-MIT.txt`](LICENSES/LATO.2-MIT.txt) and [`LICENSES/LATO.2-Embedded-MIT-Attributions.txt`](LICENSES/LATO.2-Embedded-MIT-Attributions.txt). |
| LATO.2 checkpoints | [`0x4c48/LATO.2`](https://huggingface.co/0x4c48/LATO.2/tree/a91090e8077b9318ab87ac08fd9eb905903d4515) at `a91090e8077b9318ab87ac08fd9eb905903d4515` | MIT, as declared by the pinned model card | Seven checkpoint files: V-Flow, V-VAE, offset head, T-Flow, T-VAE, voxel encoder, and VDF encoder. The LATO.2 MIT text is reproduced in [`LICENSES/LATO.2-MIT.txt`](LICENSES/LATO.2-MIT.txt). |
| DINOv2 source archive | [`facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8) at `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | Apache License 2.0 for the DINOv2 core, with separately licensed Cell-DINO, X-Ray-DINO, and embedded CLIP material | Provides the exact local `dinov2_vitl14_reg` conditioning implementation. See the DINOv2 warning above and every DINOv2 license copy under [`LICENSES/`](LICENSES/). |
| DINOv2 ViT-L/14 register-token checkpoint | `dinov2_vitl14_reg4_pretrain.pth`, SHA-256 `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51` | Apache License 2.0 project distribution | Exact upstream conditioning checkpoint, installed locally so inference does not fetch it at runtime. |
| Official LATO.2 Space source | [`0x4c48/lato2-mesh-generation`](https://huggingface.co/spaces/0x4c48/lato2-mesh-generation/tree/25ec65da46236e5ef46b88d9a510a0fe33b2bc63) at `25ec65da46236e5ef46b88d9a510a0fe33b2bc63` | MIT | Reference for the portable execution path. License copy: [`LICENSES/LATO.2-Space-MIT.txt`](LICENSES/LATO.2-Space-MIT.txt). |

## Native and numerical dependencies

The installer resolves the runtime into the extension-owned virtual
environment. The following projects are especially important because they
provide compiled code or source used during a local build.

| Component | Exact source | License | Notes |
| --- | --- | --- | --- |
| TRELLIS.2 / O-Voxel | [`microsoft/TRELLIS.2`](https://github.com/microsoft/TRELLIS.2/tree/75fbf0183001ed9876c8dbb35de6b68552ee08bd) at `75fbf0183001ed9876c8dbb35de6b68552ee08bd` | MIT | Provides the O-Voxel mesh-to-flexible-dual-grid operation used by LATO.2. License copy: [`LICENSES/TRELLIS.2-MIT.txt`](LICENSES/TRELLIS.2-MIT.txt). |
| CuMesh | [`JeffreyXiang/CuMesh`](https://github.com/JeffreyXiang/CuMesh/tree/12289e1062f0603f2f0d0771b02e1395d247f26f) at `12289e1062f0603f2f0d0771b02e1395d247f26f` | MIT | Native mesh operations. License copy: [`LICENSES/CuMesh-MIT.txt`](LICENSES/CuMesh-MIT.txt). |
| FlexGEMM | [`JeffreyXiang/FlexGEMM`](https://github.com/JeffreyXiang/FlexGEMM/tree/6dd94a859c26ee8246888502eada3dd8ad85532e) at `6dd94a859c26ee8246888502eada3dd8ad85532e` | MIT | Native matrix kernels. License copy: [`LICENSES/FlexGEMM-MIT.txt`](LICENSES/FlexGEMM-MIT.txt). |
| nvdiffrast | [`NVlabs/nvdiffrast`](https://github.com/NVlabs/nvdiffrast/tree/253ac4fcea7de5f396371124af597e6cc957bfae) at `253ac4fcea7de5f396371124af597e6cc957bfae` | NVIDIA Source Code License (1-Way Commercial) | Restricted component; see the warning above and the complete bundled license. |
| cubvh | [`JeffreyXiang/cubvh`](https://github.com/JeffreyXiang/cubvh/tree/ce92267a24ef6ad7d2c8ccbc2ae2c021a6597e70) at `ce92267a24ef6ad7d2c8ccbc2ae2c021a6597e70` | MIT, NVIDIA Source Code License, NVIDIA BSD 3-Clause, and Apache-2.0 PCG notice | CuMesh submodule. Copies/notices: [`LICENSES/cubvh-MIT.txt`](LICENSES/cubvh-MIT.txt), [`LICENSES/cubvh-NVIDIA-Source-Code-License.txt`](LICENSES/cubvh-NVIDIA-Source-Code-License.txt), [`LICENSES/cubvh-NVIDIA-BSD-3-Clause.txt`](LICENSES/cubvh-NVIDIA-BSD-3-Clause.txt), and [`LICENSES/cubvh-PCG-Apache-2.0-NOTICE.txt`](LICENSES/cubvh-PCG-Apache-2.0-NOTICE.txt). |
| xatlas | Source embedded by the pinned CuMesh revision | MIT, with embedded MIT attributions and OpenNL BSD 3-Clause material | Copies/notices: [`LICENSES/xatlas-MIT.txt`](LICENSES/xatlas-MIT.txt), [`LICENSES/xatlas-Embedded-MIT-Attributions.txt`](LICENSES/xatlas-Embedded-MIT-Attributions.txt), and [`LICENSES/xatlas-OpenNL-BSD-3-Clause.txt`](LICENSES/xatlas-OpenNL-BSD-3-Clause.txt). |
| Eigen | Revisions pinned by the native source bundles | MPL-2.0 with separately licensed files | See the exact copy set listed below. |

The bundled Eigen copy set is
[`LICENSES/Eigen-MPL-2.0.txt`](LICENSES/Eigen-MPL-2.0.txt),
[`LICENSES/Eigen-Apache-2.0-notice.txt`](LICENSES/Eigen-Apache-2.0-notice.txt),
[`LICENSES/Eigen-BSD-notice.txt`](LICENSES/Eigen-BSD-notice.txt),
[`LICENSES/Eigen-MINPACK-notice.txt`](LICENSES/Eigen-MINPACK-notice.txt), and
[`LICENSES/Eigen-COPYING-README.txt`](LICENSES/Eigen-COPYING-README.txt).

The portable profile's narrow `modly-lato2-ovoxel-cpu` 0.0.1.post2 build copies the exact
`mesh_to_flexible_dual_grid_cpu` implementation and API header from the pinned
TRELLIS.2 revision, together with Eigen revision
`21e4582d1739107337a03460c81412981130373e`. The Modly binding/build template
is MIT integration code; the copied TRELLIS.2 and Eigen material remains under
the licenses listed above. The materializer verifies the license payload
against wrapper-controlled SHA-256 values, and the locally built wheel carries
the wrapper MIT text, TRELLIS.2 MIT text, and the Eigen notices applicable to
the compiled header subset under `.dist-info/licenses/`. The complete Eigen
source tree remains in the local build/cache with its original per-file and
benchmark notices; copying those unrelated sources requires reviewing their
own terms.

PyTorch, torchvision, xFormers or FlashAttention, spconv, torch-scatter,
Open3D, trimesh, NumPy, Pillow, OpenCV, and the other Python packages installed
into the isolated environment retain their own license notices and package
metadata. Their installation does not make them part of the extension's MIT
grant.

## Host application

[Modly](https://github.com/lightningpixel/modly) is created by Lightning
Pixel and is not distributed as part of this extension. The extension targets
Modly's public process-extension contract.

## Integration authorship

The Modly manifest, installer, process-protocol bridge, storage integration,
portable compatibility code, validation tooling, and documentation in this
repository are the work of DrHepa unless a file says otherwise. LATO.2,
DINOv2, the checkpoints, and the dependency projects remain the work of their
respective authors and contributors.
