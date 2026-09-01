# Pure-PyTorch drop-in replacement for `spconv.pytorch.SparseConvTensor`.
#
# Vendored from the official LATO.2 Hugging Face Space at commit
# 25ec65da46236e5ef46b88d9a510a0fe33b2bc63.  The Space authors validated the
# submanifold convolution against spconv 2.3.8 (maximum absolute difference
# 6e-8).  This file intentionally keeps their implementation unchanged.

from typing import *

import torch

# Radix used to pack (batch, x, y, z) coordinates into a single int64 key.
# LATO.2 works at resolutions up to 1024, so 4096 leaves plenty of head room
# for the +/-1 kernel offsets while keeping the key unique.
_RADIX = 4096


class LiteSparseConvTensor:
    """Minimal stand-in for ``spconv.pytorch.SparseConvTensor``.

    Only the attributes that LATO.2 (via ``modules/sparse/basic.py``) touches
    are implemented, with the same positional constructor signature so that
    ``SparseTensor.replace()`` keeps working unchanged.
    """

    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: Any = None,
        batch_size: Optional[int] = None,
        grid: Any = None,
        voxel_num: Any = None,
        indice_dict: Optional[dict] = None,
        benchmark: bool = False,
    ):
        self._features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.grid = grid
        self.voxel_num = voxel_num
        self.indice_dict = indice_dict if indice_dict is not None else {}
        self.benchmark = benchmark
        self.benchmark_record = {}
        self.thrust_allocator = None
        self._timer = None
        self.force_algo = None
        self.int8_scale = None

    # spconv exposes `.features` as a plain attribute; SparseTensor also pokes
    # `_features` directly to keep >2D feature tensors around.
    @property
    def features(self) -> torch.Tensor:
        return self._features

    @features.setter
    def features(self, value: torch.Tensor):
        self._features = value

    def replace_feature(self, feats: torch.Tensor) -> "LiteSparseConvTensor":
        out = LiteSparseConvTensor(
            feats,
            self.indices,
            self.spatial_shape,
            self.batch_size,
            self.grid,
            self.voxel_num,
            self.indice_dict,
        )
        return out

    def dense(self, channels_first: bool = True) -> torch.Tensor:
        feats = self._features.reshape(self._features.shape[0], -1)
        spatial = [int(s) for s in self.spatial_shape]
        shape = [int(self.batch_size)] + spatial + [feats.shape[1]]
        out = torch.zeros(shape, dtype=feats.dtype, device=feats.device)
        idx = self.indices.long()
        out[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = feats
        if channels_first:
            out = out.permute(0, 4, 1, 2, 3).contiguous()
        return out


# ---------------------------------------------------------------------------
# Neighbour-map cache
# ---------------------------------------------------------------------------
# Building the submanifold neighbour map is the expensive part; it depends only
# on the coordinates, which are shared (same tensor object) by every layer that
# operates at a given resolution.  Keying a weak dict on the coords tensor gives
# us the same reuse that spconv's `indice_key` mechanism provides.
try:
    from torch.utils.weak import WeakTensorKeyDictionary as _WeakDict

    _NBMAP_CACHE = _WeakDict()
except Exception:  # pragma: no cover - very old torch
    _NBMAP_CACHE = {}


def _cache_get(coords: torch.Tensor, key):
    try:
        entry = _NBMAP_CACHE.get(coords)
    except Exception:
        return None
    if entry is None:
        return None
    return entry.get(key)


def _cache_put(coords: torch.Tensor, key, value):
    try:
        entry = _NBMAP_CACHE.get(coords)
        if entry is None:
            entry = {}
            _NBMAP_CACHE[coords] = entry
        entry[key] = value
    except Exception:
        # Caching is an optimization only; tensor semantics do not depend on it.
        return


def build_submanifold_nbmap(
    coords: torch.Tensor, kernel_size: int, dilation: int = 1
) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Return, per kernel tap, the (in_idx, out_idx) index pairs.

    ``out[out_idx] += feats[in_idx] @ W[tap].T`` reproduces spconv's
    ``SubMConv3d`` exactly.
    """
    cache_key = (int(kernel_size), int(dilation))
    cached = _cache_get(coords, cache_key)
    if cached is not None:
        return cached

    n = coords.shape[0]
    c = coords.long()
    device = coords.device
    key = ((c[:, 0] * _RADIX + c[:, 1]) * _RADIX + c[:, 2]) * _RADIX + c[:, 3]
    order = torch.argsort(key)
    skey = key[order]

    off = (kernel_size - 1) // 2
    maps: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
    for kx in range(kernel_size):
        for ky in range(kernel_size):
            for kz in range(kernel_size):
                d = torch.tensor(
                    [
                        (kx - off) * dilation,
                        (ky - off) * dilation,
                        (kz - off) * dilation,
                    ],
                    device=device,
                    dtype=torch.long,
                )
                nb = c[:, 1:] + d
                valid = ((nb >= 0) & (nb < _RADIX)).all(dim=1)
                nkey = (
                    (c[:, 0] * _RADIX + nb[:, 0]) * _RADIX + nb[:, 1]
                ) * _RADIX + nb[:, 2]
                pos = torch.searchsorted(skey, nkey.clamp(min=0)).clamp(max=n - 1)
                found = valid & (skey[pos] == nkey)
                out_idx = torch.nonzero(found, as_tuple=False).squeeze(1)
                if out_idx.numel() == 0:
                    maps.append(None)
                else:
                    in_idx = order[pos[out_idx]]
                    maps.append((in_idx.contiguous(), out_idx.contiguous()))

    _cache_put(coords, cache_key, maps)
    return maps
