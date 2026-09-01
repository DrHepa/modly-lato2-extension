"""CUDA inference precision selection for Modly's portable LATO.2 copy."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager

import torch


PRECISION_ENV = "LATO2_PRECISION"
_ALIASES = {
    "auto": "auto",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
}


def precision_name(value: str | None = None) -> str:
    """Resolve ``auto`` to the best CUDA dtype supported by this GPU."""
    raw = (value if value is not None else os.environ.get(PRECISION_ENV, "auto"))
    normalized = _ALIASES.get(str(raw).strip().lower())
    if normalized is None:
        allowed = ", ".join(sorted(_ALIASES))
        raise ValueError(f"invalid {PRECISION_ENV}={raw!r}; expected one of: {allowed}")
    if normalized == "auto":
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return normalized


def autocast_context(device: torch.device) -> AbstractContextManager:
    """Return the upstream CUDA autocast context with portable dtype choice."""
    if device.type != "cuda":
        raise RuntimeError("LATO.2 upstream inference requires a CUDA device")
    name = precision_name()
    dtype = torch.bfloat16 if name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)
