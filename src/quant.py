"""Simulated (fake) symmetric absmax quantization, round-to-nearest."""
from typing import Optional

import torch


def _fake_quant(x: torch.Tensor, scale: torch.Tensor, qmax: int) -> torch.Tensor:
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


def quantize_weight(W: torch.Tensor, bits: int = 4, group_size: Optional[int] = None) -> torch.Tensor:
    """Quantize W [out, in]. group_size=None: one scale per output row;
    else one scale per contiguous group of `group_size` input columns."""
    qmax = 2 ** (bits - 1) - 1
    out_f, in_f = W.shape
    w = W.float()
    if group_size is None:
        scale = (w.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-8)
        return _fake_quant(w, scale, qmax).to(W.dtype)
    if in_f % group_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by group_size={group_size}")
    w = w.reshape(out_f, in_f // group_size, group_size)
    scale = (w.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-8)
    return _fake_quant(w, scale, qmax).reshape(out_f, in_f).to(W.dtype)


def quantize_activation(x: torch.Tensor, bits: Optional[int]) -> torch.Tensor:
    """Per-token symmetric quantization over the last dim. bits=None: identity."""
    if bits is None:
        return x
    qmax = 2 ** (bits - 1) - 1
    xf = x.float()
    scale = (xf.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-8)
    return _fake_quant(xf, scale, qmax).to(x.dtype)
