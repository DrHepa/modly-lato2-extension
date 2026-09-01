"""Pinned o-voxel flexible-dual-grid wrapper with a minimal native module.

The argument normalization and call are the upstream TRELLIS.2/o-voxel logic
at commit 75fbf0183001ed9876c8dbb35de6b68552ee08bd. Only the function LATO.2
uses is exposed; the native implementation is compiled unchanged from that
commit by :mod:`lato2_modly.ovoxel_cpu`.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from lato2_ovoxel_cpu import _C

__all__ = ["mesh_to_flexible_dual_grid"]


@torch.no_grad()
def mesh_to_flexible_dual_grid(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    aabb: Union[list, tuple, np.ndarray, torch.Tensor] = None,
    face_weight: float = 1.0,
    boundary_weight: float = 1.0,
    regularization_weight: float = 0.1,
    timing: bool = False,
):
    vertices = vertices.float()
    faces = faces.int()
    if voxel_size is None and grid_size is None:
        raise AssertionError("Either voxel_size and grid_size must be provided")

    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        if not isinstance(voxel_size, torch.Tensor):
            raise TypeError(f"unsupported voxel_size type: {type(voxel_size)}")
        if voxel_size.dim() != 1 or voxel_size.size(0) != 3:
            raise ValueError("voxel_size must contain three elements")

    if grid_size is not None:
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32)
        if not isinstance(grid_size, torch.Tensor):
            raise TypeError(f"unsupported grid_size type: {type(grid_size)}")
        if grid_size.dim() != 1 or grid_size.size(0) != 3:
            raise ValueError("grid_size must contain three elements")

    if aabb is not None:
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        if not isinstance(aabb, torch.Tensor):
            raise TypeError(f"unsupported aabb type: {type(aabb)}")
        if aabb.shape != (2, 3):
            raise ValueError("aabb must have shape [2, 3]")

    if aabb is None:
        min_xyz = vertices.min(dim=0).values
        max_xyz = vertices.max(dim=0).values
        if voxel_size is not None:
            padding = (
                torch.ceil((max_xyz - min_xyz) / voxel_size) * voxel_size
                - (max_xyz - min_xyz)
            )
            min_xyz -= padding * 0.5
            max_xyz += padding * 0.5
        if grid_size is not None:
            padding = (max_xyz - min_xyz) / (grid_size - 1)
            min_xyz -= padding * 0.5
            max_xyz += padding * 0.5
        aabb = torch.stack([min_xyz, max_xyz], dim=0).float()

    if voxel_size is None:
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    if grid_size is None:
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()

    # The pinned C++ function is CPU-only. LATO.2 passes CPU tensors here; make
    # that invariant explicit so a misuse cannot dereference a CUDA data_ptr.
    if any(tensor.is_cuda for tensor in (vertices, faces, voxel_size, grid_size, aabb)):
        raise ValueError("mesh_to_flexible_dual_grid expects CPU tensors")
    vertices = (vertices - aabb[0].reshape(1, 3)).contiguous()
    faces = faces.contiguous()
    voxel_size = voxel_size.float().contiguous()
    grid_range = torch.stack([torch.zeros_like(grid_size), grid_size], dim=0).int().contiguous()
    return _C.mesh_to_flexible_dual_grid_cpu(
        vertices,
        faces,
        voxel_size,
        grid_range,
        float(face_weight),
        float(boundary_weight),
        float(regularization_weight),
        bool(timing),
    )
