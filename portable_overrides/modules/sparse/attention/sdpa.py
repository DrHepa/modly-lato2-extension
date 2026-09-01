"""PyTorch SDPA adapters for LATO.2's packed sparse sequences.

This is an additive attention backend.  The upstream ``flash_attn`` and
``xformers`` implementations remain available and selectable; this module is
used only when ``SPARSE_ATTN_BACKEND=sdpa``.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def dense_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Run SDPA on tensors shaped ``[batch, sequence, heads, channels]``."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("dense_sdpa expects four-dimensional q, k, and v")
    if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v batch sizes differ")
    if q.shape[2] != k.shape[2] or k.shape[2] != v.shape[2]:
        raise ValueError("q, k, and v head counts differ")
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    )
    return out.transpose(1, 2).contiguous()


def packed_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lengths: Sequence[int],
    kv_lengths: Sequence[int] | None = None,
) -> torch.Tensor:
    """Run SDPA independently on packed variable-length sequences.

    Inputs and output use LATO.2's packed ``[tokens, heads, channels]`` layout.
    Processing each sample independently exactly enforces the block-diagonal
    attention mask used by the upstream FlashAttention/xFormers paths, without
    padding tokens into a potentially much larger dense allocation.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("packed_sdpa expects three-dimensional q, k, and v")
    kv_lengths = q_lengths if kv_lengths is None else kv_lengths
    q_lengths = tuple(int(length) for length in q_lengths)
    kv_lengths = tuple(int(length) for length in kv_lengths)
    if len(q_lengths) != len(kv_lengths):
        raise ValueError("query and key/value sequence counts differ")
    if any(length <= 0 for length in q_lengths + kv_lengths):
        raise ValueError("SDPA sequence lengths must be positive")
    if sum(q_lengths) != q.shape[0]:
        raise ValueError("query lengths do not cover the packed query tensor")
    if sum(kv_lengths) != k.shape[0] or k.shape[0] != v.shape[0]:
        raise ValueError("key/value lengths do not cover the packed tensors")

    outputs = []
    q_start = 0
    kv_start = 0
    for q_length, kv_length in zip(q_lengths, kv_lengths):
        q_part = q[q_start : q_start + q_length].unsqueeze(0)
        k_part = k[kv_start : kv_start + kv_length].unsqueeze(0)
        v_part = v[kv_start : kv_start + kv_length].unsqueeze(0)
        outputs.append(dense_sdpa(q_part, k_part, v_part).squeeze(0))
        q_start += q_length
        kv_start += kv_length
    return torch.cat(outputs, dim=0)
