# Pure-PyTorch submanifold sparse convolution backend.
#
# Vendored from the official LATO.2 Hugging Face Space at commit
# 25ec65da46236e5ef46b88d9a510a0fe33b2bc63.  Parameters deliberately retain
# spconv's state-dict keys and layout so the released checkpoints load without
# remapping.

import math

import torch
import torch.nn as nn

from .. import SparseTensor
from ..lite import build_submanifold_nbmap


class _SubMConv3dParams(nn.Module):
    """Holds the weights in spconv's layout (keeps state-dict keys identical)."""

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, kernel_size, kernel_size, kernel_size, in_channels)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.in_channels * (self.kernel_size**3)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)


class SparseConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding=None,
        bias=True,
        indice_key=None,
    ):
        super().__init__()
        stride_t = (
            tuple(stride)
            if isinstance(stride, (list, tuple))
            else (stride, stride, stride)
        )
        if any(s != 1 for s in stride_t) or padding is not None:
            raise RuntimeError(
                "The pure-PyTorch sparse backend only supports submanifold "
                "convolutions (stride=1, padding=None)."
            )
        self.conv = _SubMConv3dParams(
            in_channels, out_channels, kernel_size, bias=bias
        )
        self.dilation = dilation
        self.stride = stride_t
        self.padding = padding
        self.indice_key = indice_key

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x: SparseTensor) -> SparseTensor:
        dtype_ = x.feats.dtype
        coords = x.coords
        k = self.conv.kernel_size

        # spconv's wrapper runs the convolution in fp32 even in autocast.
        with torch.autocast(device_type=coords.device.type, enabled=False):
            feats = x.feats.reshape(x.feats.shape[0], -1).float()
            weight = self.conv.weight.float()
            if k == 1:
                # Preserve spconv's checkpoint-compatible 1x1 buffer view.
                out = feats @ weight.reshape(
                    self.conv.in_channels, self.conv.out_channels
                )
            else:
                # [C_out, kx, ky, kz, C_in] -> [k^3, C_in, C_out]
                w = weight.permute(1, 2, 3, 4, 0).reshape(
                    k * k * k, self.conv.in_channels, self.conv.out_channels
                )
                out = torch.zeros(
                    feats.shape[0],
                    self.conv.out_channels,
                    dtype=feats.dtype,
                    device=feats.device,
                )
                nbmap = build_submanifold_nbmap(coords, k, self.dilation)
                for ki, entry in enumerate(nbmap):
                    if entry is None:
                        continue
                    in_idx, out_idx = entry
                    out.index_add_(0, out_idx, feats[in_idx] @ w[ki])
            if self.conv.bias is not None:
                out = out + self.conv.bias.float()

        return x.replace(out.type(dtype_))


class SparseInverseConv3d(nn.Module):
    """Declared for upstream utility introspection; unused by LATO.2."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError(
            "SparseInverseConv3d is not supported by the pure-PyTorch sparse backend."
        )
