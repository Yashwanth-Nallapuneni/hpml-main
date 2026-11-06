"""RotatedQuantLinear: nn.Linear with optional Hadamard rotation + fake quantization."""
import math
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hadamard import BlockHadamardRotation, _is_pow2
from .quant import quantize_activation, quantize_weight

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_warned = False


@dataclass
class QuantConfig:
    w_bits: Optional[int] = 4
    group_size: Optional[int] = 128
    a_bits: Optional[int] = None
    out_bits: Optional[int] = None
    rotate_in: bool = False
    rotate_out: bool = False
    block_in: Union[int, str, None] = 128
    block_out: Union[int, str, None] = 128
    seed: int = 0
    collect_stats: bool = True


def resolve_block(dim: int, block: Union[int, str, None]) -> int:
    """'full'/None -> dim if power of 2, else largest power-of-2 <= 128 dividing dim."""
    global _warned
    if block is None or block == "full":
        if _is_pow2(dim):
            return dim
        b = 128
        while dim % b != 0:
            b //= 2
        if not _warned:
            print(f"WARNING: dim {dim} is not a power of 2; 'full' falls back to block {b} for such dims")
            _warned = True
        return b
    return int(block)


class RotatedQuantLinear(nn.Module):
    """Computes y = x W^T + b via W' = R_out^T W R_in (optionally quantized).

    Derivation: x' = x R_in; y' = x' W'^T = x R_in R_in^T W^T R_out = x W^T R_out
    (R orthogonal), so y = y' R_out^T = x W^T exactly when quantization is off.
    """

    def __init__(self, linear: nn.Linear, cfg: QuantConfig, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        W = linear.weight.data
        dev, dt = W.device, W.dtype
        out_f, in_f = W.shape
        self.r_in = (BlockHadamardRotation(in_f, resolve_block(in_f, cfg.block_in), seed, dev)
                     if cfg.rotate_in else None)
        self.r_out = (BlockHadamardRotation(out_f, resolve_block(out_f, cfg.block_out), seed + 1, dev)
                      if cfg.rotate_out else None)
        Wp = W.float()
        if self.r_in is not None:
            Wp = self.r_in.apply(Wp)  # W @ R_in
        if self.r_out is not None:
            Wp = self.r_out.apply(Wp.T).T  # R_out^T A = (A^T R_out)^T
        if cfg.w_bits is not None:
            Wp = quantize_weight(Wp, cfg.w_bits, cfg.group_size)
        self.register_buffer("weight_q", Wp.to(dt).contiguous())
        self.register_buffer("weight_ref", W.clone())
        self.register_buffer("bias", None if linear.bias is None else linear.bias.data.clone())
        self.reset_stats()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = self.r_in.apply(x) if self.r_in is not None else x
        xr = quantize_activation(xr, self.cfg.a_bits)
        yr = F.linear(xr, self.weight_q)
        yr = quantize_activation(yr, self.cfg.out_bits)  # in the rotated output basis
        y = self.r_out.apply_T(yr) if self.r_out is not None else yr
        if self.bias is not None:
            y = y + self.bias
        if self.cfg.collect_stats:
            self._accumulate(x, y)
        return y.to(x.dtype)

    @torch.no_grad()
    def _accumulate(self, x: torch.Tensor, y: torch.Tensor) -> None:
        # NOTE: measured in-context (x already carries upstream quantization error)
        # and in the ORIGINAL channel basis (y is already unrotated).
        ref = F.linear(x, self.weight_ref, self.bias).double().reshape(-1, self.weight_ref.shape[0])
        err = y.double().reshape_as(ref) - ref
        if self._sse_c is None:
            z = torch.zeros(ref.shape[1], dtype=torch.float64, device=ref.device)
            self._sse_c, self._ssr_c = z.clone(), z.clone()
        self._sse_c += err.pow(2).sum(0)
        self._ssr_c += ref.pow(2).sum(0)
        self._n += ref.shape[0]

    def reset_stats(self) -> None:
        self._sse_c: Optional[torch.Tensor] = None
        self._ssr_c: Optional[torch.Tensor] = None
        self._n = 0

    def stats(self) -> dict:
        if self._n == 0:
            return {}
        out_f = self._sse_c.numel()
        sse, ssr = self._sse_c.sum().item(), self._ssr_c.sum().item()
        return {
            "rms_err": math.sqrt(sse / (self._n * out_f)),
            "rms_ref": math.sqrt(ssr / (self._n * out_f)),
            "rel_err": math.sqrt(sse / max(ssr, 1e-30)),
            "snr_db": 10 * math.log10(max(ssr, 1e-30) / max(sse, 1e-30)),
            "ch_rms_err": (self._sse_c / self._n).sqrt().tolist(),
            "ch_rms_ref": (self._ssr_c / self._n).sqrt().tolist(),
            "tokens": self._n,
        }


def wrap_model(model: nn.Module, cfg: QuantConfig, projections: Sequence[str] = PROJECTIONS,
               layers: Optional[Sequence[int]] = None) -> Dict[str, RotatedQuantLinear]:
    """Replace decoder-layer projections with RotatedQuantLinear (never embeddings/lm_head)."""
    targets = []
    for name, mod in model.named_modules():
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if m and isinstance(mod, nn.Linear) and name.split(".")[-1] in projections:
            idx = int(m.group(1))
            if layers is None or idx in layers:
                targets.append((name, mod, idx))
    wrapped: Dict[str, RotatedQuantLinear] = {}
    for name, mod, idx in targets:
        w = RotatedQuantLinear(mod, cfg, seed=cfg.seed * 1_000_003 + 2 * idx)
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, w)
        wrapped[name] = w
    print(f"Wrapped {len(wrapped)} linear layers")
    return wrapped
