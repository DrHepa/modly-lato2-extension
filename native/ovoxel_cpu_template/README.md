# Minimal o-voxel CPU build

This template is completed by `lato2_modly.ovoxel_cpu`: it copies the exact
`flexible_dual_grid.cpp` and `api.h` from TRELLIS.2 commit
`75fbf0183001ed9876c8dbb35de6b68552ee08bd`, plus Eigen submodule commit
`21e4582d1739107337a03460c81412981130373e`.

Install only after PyTorch is present in the extension venv:

```text
python -m pip install --no-build-isolation --no-deps <materialized-build-dir>
```

It is a `CppExtension`, not a `CUDAExtension`: no CUDA toolkit or nvcc is
required. A C++17 compiler is required (MSVC Build Tools on Windows, or a
supported GCC/Clang toolchain on Linux including ARM64).

The materializer also adds an integrity-checked `LICENSES/` directory. The
result is `modly-lato2-ovoxel-cpu` 0.0.1.post2; its wheel carries the Modly
wrapper MIT text, the TRELLIS.2 MIT text, and the complete Eigen license/notice
set under `.dist-info/licenses/`. The extension records one build identity
covering this complete template, pinned native inputs, and license payload.
