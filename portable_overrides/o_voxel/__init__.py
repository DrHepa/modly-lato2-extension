"""Narrow o-voxel compatibility package for the one API LATO.2 invokes.

The extension is backed by a CPU-only C++ build of the exact pinned upstream
``mesh_to_flexible_dual_grid_cpu`` source. It intentionally does not import the
full o-voxel package's unrelated CUDA/IO/rasterization/serialization modules.
"""

__all__ = ["convert"]
