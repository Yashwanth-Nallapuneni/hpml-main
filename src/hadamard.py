"""Randomized block Hadamard rotation, applied without forming dense matrices."""
import math

import torch
import torch.nn as nn


def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def sylvester_hadamard(n: int) -> torch.Tensor:
    """Normalized (1/sqrt(n)) symmetric Sylvester Hadamard matrix, float32."""
    if not _is_pow2(n):
        raise ValueError(f"Sylvester Hadamard needs a power-of-2 size, got {n}")
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


_H_CACHE = {}


def _shared_hadamard(n: int, device: str) -> torch.Tensor:
    """Read-only Sylvester H, built once per (n, device) and shared by every rotation module."""
    key = (n, device)
    if key not in _H_CACHE:
        _H_CACHE[key] = sylvester_hadamard(n).to(device)
    return _H_CACHE[key]


class BlockHadamardRotation(nn.Module):
    """Orthogonal R = D @ B, D = diag(random +-1), B = blockdiag of Hadamards.

    Row-vector convention: apply(t) = t @ R, apply_T(t) = t @ R^T.
    """

    def __init__(self, dim: int, block=None, seed: int = 0, device="cpu"):
        super().__init__()
        if block is None or block == "full":
            block = dim
            if not _is_pow2(dim):
                raise ValueError(
                    f"dim={dim} is not a power of 2, so a full Hadamard is impossible. "
                    f"Use a block size that divides it, e.g. block=128 (dim/128={dim / 128:g})."
                )
        block = int(block)
        if not _is_pow2(block) or dim % block != 0:
            raise ValueError(f"block={block} must be a power of 2 dividing dim={dim}")
        self.dim, self.block = dim, block
        g = torch.Generator().manual_seed(seed)
        signs = torch.randint(0, 2, (dim,), generator=g).float() * 2 - 1
        self.register_buffer("signs", signs.to(device))
        # One shared H per (size, device): a full 8192x8192 H is 256 MB, so per-module copies run out of memory.
        self.register_buffer("H", _shared_hadamard(block, str(device)), persistent=False)

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        """t @ R along the last dim: scale by signs, then blockwise matmul by H."""
        x = t.float() * self.signs
        x = (x.reshape(*t.shape[:-1], self.dim // self.block, self.block) @ self.H)
        return x.reshape(t.shape).to(t.dtype)

    def apply_T(self, t: torch.Tensor) -> torch.Tensor:
        """t @ R^T = (blockwise t @ H) then scale by signs (H is symmetric)."""
        x = t.float().reshape(*t.shape[:-1], self.dim // self.block, self.block) @ self.H
        return (x.reshape(t.shape) * self.signs).to(t.dtype)


def self_test() -> None:
    """Check R R^T = I, R matches the dense definition, apply/apply_T invert."""
    dim, block = 256, 64
    rot = BlockHadamardRotation(dim, block, seed=3)
    R = rot.apply(torch.eye(dim))
    dense = torch.diag(rot.signs) @ torch.block_diag(*[sylvester_hadamard(block)] * (dim // block))
    assert torch.allclose(R, dense, atol=1e-6), "apply != dense D@B"
    assert torch.allclose(R @ R.T, torch.eye(dim), atol=1e-5), "R R^T != I"
    t = torch.randn(5, 7, dim)
    assert torch.allclose(rot.apply_T(rot.apply(t)), t, atol=1e-5), "apply_T(apply(t)) != t"
    assert torch.allclose(rot.apply(rot.apply_T(t)), t, atol=1e-5), "apply(apply_T(t)) != t"
    print("hadamard self_test: OK")
