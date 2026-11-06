"""Reproduce CAT (arXiv 2603.04359) Table 1, Llama-3.2-1B-Instruct, RTN W4A4KV4: FP 13.16, RTN 412.51, QuaRot 29.89,
CAT (block, no training) 18.25 +- 0.21. Their code is not released; the set-up is taken from the paper (Sec. 6) and, where the
paper is silent, from the same group's FPTQuant repo:
  weights     : per output channel, symmetric, levels -8..7, round half to even, range by L2.4 error over 80 shrink ratios (GPTQ grid)
  activations : inputs of q/k/v/o/gate/up/down, 4 bit, dynamic per token, asymmetric (range includes 0)
  KV cache    : 4 bit, dynamic per token and head, asymmetric; K after RoPE. QuaRot/CAT: Sylvester Hadamard on the head dim of Q, K, V
  not quantized: embeddings, lm_head, norms, attention scores
  QuaRot      : RMSNorm gamma fused into the next linear; Sylvester Hadamard (no random signs) on every linear input
                (= residual R1 + online H before down_proj + head-wise Pv and cross-head Po for o_proj)
  CAT (block) : x' = (x M) H on q/k/v (shared M), gate/up (shared M) and down_proj; M from Eq. 7 per k x k block on DCLM-edu
                calibration (128 x 2048, seeded); o_proj gets QuaRot's H only (paper Fig. 7 caption)
Choices the paper leaves open (FPTQuant convention used): CAT does not fuse gamma (M absorbs any diagonal up to an
orthogonal factor ahead of H); weight scale grid starts at max(|min|/8, max/7) rather than GPTQ's 2 absmax/15.
Every linear is emulated as y = Qa(x A) Q(W A^-T)^T, exact in fp when quantization is off (asserted at start-up).
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from src.data import load_wikitext2
from src.hadamard import sylvester_hadamard
from run_mackin import cat_blocks

SHARE = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "gate_proj": "gu", "up_proj": "gu", "down_proj": "down"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--methods", default="fp,rtn,quarot,cat", help="comma list of fp, rtn, quarot, cat")
    ap.add_argument("--cat-k", default="128", help="comma list of CAT block sizes (must divide 2048 and 8192); 1 = diagonal")
    ap.add_argument("--k1-formula", choices=["eq8", "text"], default="eq8",
                    help="k = 1 only: eq8 = Eq. 7/8, m = (|w_i|^2 / E x_i^2)^(1/4); text = Sec. 4 text, m = sqrt(E x_i^2 / |w_i|^2)")
    ap.add_argument("--seeds", default="0", help="comma list; CAT calibration sample seed")
    ap.add_argument("--calib", choices=["dclm", "wikitext"], default="dclm")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--a-bits", type=int, default=4, help="0 = no activation quantization")
    ap.add_argument("--kv-bits", type=int, default=4, help="0 = no KV quantization")
    ap.add_argument("--w-range", choices=["l2.4", "minmax"], default="l2.4")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--eval-windows", type=int, default=None, help="default: whole WikiText-2 test split")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--sx-skip", type=int, default=0, help="leading positions of each window left out of Sx (attention-sink diagnostic)")
    ap.add_argument("--ridge", type=float, default=1e-6, help="CAT damping, relative to the mean diagonal")
    ap.add_argument("--out", required=True, help="results JSONL, appended")
    return ap.parse_args()


# ---------------------------------------------------------------- quantizers
def quant_w(W: torch.Tensor, bits: int, how: str) -> torch.Tensor:
    """Per-row symmetric, levels -2^(b-1)..2^(b-1)-1, scale chosen by L_2.4 error over shrink ratios 1.00..0.21."""
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    W = W.float()
    s0 = torch.maximum(W.amin(1, keepdim=True).abs() / -lo, W.amax(1, keepdim=True) / hi).clamp_min(1e-10)
    q = lambda s: torch.clamp(torch.round(W / s), lo, hi) * s
    if how == "minmax":
        return q(s0)
    best, best_err = q(s0), torch.full_like(s0, float("inf"))
    for i in range(80):  # GPTQ mse search: grid 100, maxshrink 0.8, norm 2.4
        Wq = q(s0 * (1 - i / 100))
        err = (Wq - W).abs().pow(2.4).sum(1, keepdim=True)
        better = err < best_err
        best, best_err = torch.where(better, Wq, best), torch.where(better, err, best_err)
    return best


def quant_a(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Dynamic asymmetric per row of the last dim (per token; per token and head for K/V); range always contains 0."""
    if not bits:
        return x
    qmax = 2 ** bits - 1
    xf = x.float()
    mn, mx = xf.amin(-1, keepdim=True).clamp_max(0), xf.amax(-1, keepdim=True).clamp_min(0)
    s = ((mx - mn) / qmax).clamp_min(1e-10)
    z = torch.round(-mn / s)
    return ((torch.clamp(torch.round(xf / s) + z, 0, qmax) - z) * s).to(x.dtype)


# ---------------------------------------------------------------- emulated linear and attention
class QLinear(nn.Module):
    """y = Qa(x A) Wq^T + b with Wq = Q(W A^-T); A = None means identity."""

    def __init__(self, lin: nn.Linear, A, args):
        super().__init__()
        W = lin.weight.data.double()
        if A is not None:
            A = A.double()
            W = W @ torch.linalg.inv(A).T
            x = torch.randn(4, A.shape[0], dtype=torch.float64, device=A.device)
            gap = ((x @ A @ W.T - x @ lin.weight.data.double().T).norm() / (x @ lin.weight.data.double().T).norm()).item()
            assert gap < 1e-4, f"transform not function-preserving: {gap:.1e}"
        self.register_buffer("A", None if A is None else A.float())
        self.register_buffer("Wq", quant_w(W, args.w_bits, args.w_range) if args.w_bits else W.float())
        self.register_buffer("bias", None if lin.bias is None else lin.bias.data.float())
        self.a_bits = args.a_bits

    def forward(self, x):
        h = x.float()
        if self.A is not None:
            h = h @ self.A
        return F.linear(quant_a(h, self.a_bits), self.Wq, self.bias).to(x.dtype)


KV = {"bits": 0, "H": None}


def kvq_attention(module, query, key, value, attention_mask, **kw):
    """sdpa with K (after RoPE) and V quantized per token and head; optional head-dim Hadamard on Q, K, V (undone on the output)."""
    H = KV["H"]
    if KV["bits"] or H is not None:  # rotation also runs with bits = 0, so the fp self-check covers it
        dt = query.dtype
        if H is not None:
            query, key, value = (t.float() @ H for t in (query, key, value))
        key, value = quant_a(key, KV["bits"]), quant_a(value, KV["bits"])
        if H is not None:
            value = value @ H  # attention is linear in V: (P (V H)) H^T = P V, H symmetric orthogonal
        query, key, value = query.to(dt), key.to(dt), value.to(dt)
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)


AttentionInterface.register("kvq_sdpa", kvq_attention)
try:  # transformers >= 4.53 picks the mask builder by attention name
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionMaskInterface.register("kvq_sdpa", sdpa_mask)
except ImportError:
    pass


# ---------------------------------------------------------------- calibration and CAT
_DCLM = {}


def calib_windows(tok, args, seed: int) -> torch.Tensor:
    if args.calib == "wikitext":
        return load_wikitext2(tok, split="train", seq_len=args.seq_len, max_windows=args.n_calib)
    g = torch.Generator().manual_seed(seed)
    # first 4 * n documents of DCLM-edu with >= seq_len tokens, then n of them at random, random crop each.
    # Streamed once per process: a second streaming pass took the 15 GB host past OOM (2026-11-03).
    ck = (args.seq_len, args.n_calib)
    if ck not in _DCLM:
        from datasets import load_dataset
        docs = []
        for ex in load_dataset("HuggingFaceTB/dclm-edu", split="train", streaming=True):
            ids = tok(ex["text"], return_tensors="pt").input_ids[0]
            if ids.numel() >= args.seq_len:
                docs.append(ids)
                if len(docs) == 4 * args.n_calib:
                    break
        _DCLM[ck] = docs
    long_docs = _DCLM[ck]
    pick = torch.randperm(len(long_docs), generator=g)[: args.n_calib].tolist()
    out = []
    for i in pick:
        st = torch.randint(0, long_docs[i].numel() - args.seq_len + 1, (1,), generator=g).item()
        out.append(long_docs[i][st: st + args.seq_len])
    return torch.stack(out)


@torch.no_grad()
def input_covariances(model, targets, ids, K: int, skip: int = 0) -> dict:
    """Diagonal K x K blocks of E[x x^T] over all calibration tokens, input of one module per share group (fp model)."""
    S, n = {}, 0
    want = {}
    for l, name, m in targets:
        want.setdefault((l, SHARE.get(name.split(".")[-1])), (name, m))
    def hook(key):
        def h(mod, inp, out):
            x = inp[0].detach()[:, skip:].reshape(-1, inp[0].shape[-1] // K, K).double()
            S[key] = S.get(key, 0) + torch.einsum("tbi,tbj->bij", x, x)
        return h
    hs = [m.register_forward_hook(hook(k)) for k, (_, m) in want.items() if k[1] is not None]
    for i in range(ids.shape[0]):
        model(input_ids=ids[i:i + 1].cuda(), use_cache=False)
        n += ids.shape[1] - skip
    for h in hs:
        h.remove()
    return {k: v / n for k, v in S.items()}


def cat_matrix(Sx, Sw, k, formula: str, ridge: float = 1e-6) -> torch.Tensor:
    """M blocks [d/k, k, k] from K x K diagonal blocks of Sx, Sw [d/K, K, K] (k divides K)."""
    K = Sx.shape[-1]
    blk = lambda S: torch.stack([B[i * k:(i + 1) * k, i * k:(i + 1) * k] for B in S for i in range(K // k)])
    if k == 1 and formula == "text":
        return (blk(Sx) / blk(Sw).clamp_min(1e-30)).sqrt()
    return cat_blocks(blk(Sw), blk(Sx), ridge)[0]


# ---------------------------------------------------------------- model surgery and evaluation
def find_targets(model):
    out = []
    for i, layer in enumerate(model.model.layers):
        for name, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                out.append((i, f"model.layers.{i}.{name}", m))
    return out


def norm_gamma(model, layer: int, short: str):
    L = model.model.layers[layer]
    if short in ("q_proj", "k_proj", "v_proj"):
        return L.input_layernorm.weight.data
    if short in ("gate_proj", "up_proj"):
        return L.post_attention_layernorm.weight.data
    return None


def build(model, targets, method: str, args, Mx=None) -> dict:
    """Replace every decoder linear with a QLinear for `method`; return {name: original nn.Linear} to undo."""
    Hs = {}
    H = lambda d: Hs.setdefault(d, sylvester_hadamard(d).cuda().double())
    orig = {}
    for l, name, m in targets:
        short = name.split(".")[-1]
        A = None
        if method == "quarot" or (method == "cat" and short == "o_proj"):
            g = norm_gamma(model, l, short)
            A = H(m.in_features) if g is None else torch.diag(1 / g.double()) @ H(m.in_features)
        elif method == "cat":
            A = torch.block_diag(*Mx[(l, SHARE[short])]) @ H(m.in_features)
        parent, _, child = name.rpartition(".")
        orig[name] = m
        setattr(model.get_submodule(parent), child, QLinear(m, A, args))
    KV["bits"] = args.kv_bits
    KV["H"] = H(model.config.head_dim).float() if method in ("quarot", "cat") else None
    return orig


def restore(model, orig: dict) -> None:
    for name, m in orig.items():
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, m)
    KV["bits"], KV["H"] = 0, None


def rel_gap(a, b) -> float:
    return ((a - b).norm() / b.norm()).item()


@torch.no_grad()
def ppl(model, test, batch: int) -> float:
    tot = 0.0
    for i in range(0, test.shape[0], batch):
        ids = test[i:i + batch].cuda()
        logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum").item()
    return math.exp(tot / (test.shape[0] * (test.shape[1] - 1)))


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="kvq_sdpa").eval()
    targets = find_targets(model)
    test = load_wikitext2(tok, split="test", seq_len=args.seq_len, max_windows=args.eval_windows)
    print(f"{args.model}: {len(targets)} linears, {test.shape[0]} test windows", flush=True)

    def record(method, ppl_val, t0, **extra):
        row = {"method": method, "ppl": ppl_val, "seconds": round(time.time() - t0, 1), "w_bits": args.w_bits,
               "a_bits": args.a_bits, "kv_bits": args.kv_bits, "w_range": args.w_range, "calib": args.calib,
               "n_calib": args.n_calib, "test_windows": test.shape[0], "model": args.model, **extra}
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    # self-check: with all quantization off, every transform must reproduce the fp model exactly
    fp_bits = argparse.Namespace(**{**vars(args), "w_bits": 0, "a_bits": 0, "kv_bits": 0})
    probe = test[:1].cuda()
    with torch.no_grad():
        ref = model(input_ids=probe, use_cache=False).logits.float()
        for method in ("quarot",):
            orig = build(model, targets, method, fp_bits)
            gap = rel_gap(model(input_ids=probe, use_cache=False).logits.float(), ref)
            restore(model, orig)
            print(f"self-check {method} with quantization off: relative logit error {gap:.4f}", flush=True)
            assert gap < 0.02, "transformed model differs from fp (bf16 tolerance)"

    for method in args.methods.split(","):
        if method == "cat":
            continue
        t0 = time.time()
        orig = {} if method == "fp" else build(model, targets, method, args)
        record(method, ppl(model, test, args.batch), t0)
        restore(model, orig)

    if "cat" in args.methods.split(","):
        ks = [int(k) for k in args.cat_k.split(",")]
        K = max(ks)  # accumulate K x K diagonal blocks once; every k (powers of 2) is a sub-block
        assert all(K % k == 0 for k in ks)
        Sw = {}
        for l, name, m in targets:
            key = (l, SHARE.get(name.split(".")[-1]))
            if key[1] is not None:
                W = m.weight.data.double().view(m.out_features, -1, K)
                Sw[key] = Sw.get(key, 0) + torch.einsum("obi,obj->bij", W, W)
        for seed in [int(s) for s in args.seeds.split(",")]:
            t_cal = time.time()
            Sx = input_covariances(model, targets, calib_windows(tok, args, seed), K, args.sx_skip)
            t_cal = round(time.time() - t_cal, 1)
            for k in args.cat_k.split(","):
                t0 = time.time()
                Mx = {}
                for key in Sx:
                    Mx[key] = cat_matrix(Sx[key], Sw[key], int(k), args.k1_formula, args.ridge)
                with torch.no_grad():  # self-check: CAT with quantization off must reproduce the fp model
                    orig = build(model, targets, "cat", fp_bits, Mx)
                    gap = rel_gap(model(input_ids=probe, use_cache=False).logits.float(), ref)
                    restore(model, orig)
                print(f"self-check cat k={k} with quantization off: relative logit error {gap:.4f}", flush=True)
                assert gap < 0.02, "CAT-transformed model differs from fp (bf16 tolerance)"
                t0 = time.time()
                orig = build(model, targets, "cat", args, Mx)
                record("cat", ppl(model, test, args.batch), t0, cat_k=k, seed=seed, k1_formula=args.k1_formula, sx_skip=args.sx_skip, ridge=args.ridge,
                       calib_seconds=t_cal)
                restore(model, orig)
                del Mx
    print("REPRO DONE", flush=True)


if __name__ == "__main__":
    main()
