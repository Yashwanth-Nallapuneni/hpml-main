"""
Pipeline for each q/k/v/o/gate/up/down (weights only, W4A16, RTN, group = Hadamard block = --group):
  alignment : one scale per input channel, s_j = mean|x_j|^a (geomean 1), a in {0,.25,.5,.75,1} chosen per module
              on calibration text to minimise the module's output error under the full pipeline. This is the
              k = 1 (diagonal, SmoothQuant/AWQ-like) case of CAT's alignment; x W^T = (x/s)(W s)^T, so it is exact in fp.
  Hadamard  : block rotation on the input and output sides, seeds as src/wrapper.py (seed 0).
  W_eff = R_out Q(R_out^T (W s) R_in) R_in^T / s, applied as an fp32 weight (same math as rotating x and y).

Modes:
  est : for every output channel, on WikiText-2 TRAIN text (calibration), for the real error the channel gets:
        pre4 = aligned INT4, no Hadamard (Mackin: "after the alignment, before the Hadamard"), rot4 = aligned INT4 +
        Hadamard, pre8 = aligned INT8 (second quantization type). Fields per channel (per token):
        g1 = sum g.delta (true labels, first order), qf = mean_k (sum gs.delta)^2 (sampled labels, = delta^T GGN delta),
        fish = mean_k sum gs^2, en = sum delta^2, ey = sum y^2 (channel output energy).
  grad: g1 only, from the weight gradient: g1_c = sum_j E[c,j] G[c,j] / n_tok (E = W_eff - W), one backward per window, so
        hundreds of windows are cheap. Also g1 on the first 8 windows (= est's g1) and on each half (split-half check).
  gt  : Mackin's ground truth on VALIDATION text: the fp model with ONE channel's real quantized output (pre4) blended in;
        one-sided dloss per token over --gt-windows windows, also per half, to check it replicates.
  eval: perplexity on TEST text: fp, int4, int4+H, aligned, aligned+H, then aligned+H with the top-k channels of every
        module restored to fp (Mackin's simple version) or isolated (dropped from the rotation mix and kept fp),
        chosen by each selector or at random. --zs-tasks swaps the perplexity for lm-eval zero-shot accuracy (zs.csv).
"""
import argparse
import csv
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import sdpa_attention

from src.data import PILE, load_text, load_wikitext2
from src.hadamard import BlockHadamardRotation
from src.quant import quantize_weight
from src.wrapper import PROJECTIONS
from run_sensitivity import find_targets
from run_qsens import make_hook, nll_rows, write_csv

KINDS = {"pre4": (4, False), "rot4": (4, True), "pre8": (8, False)}
SELECTORS = {  # score per channel from the est files; higher = restore first
    "pred_pre": lambda e: e["pre4"]["g1"] + e["pre4"]["qf"] / 2,
    "curv_pre": lambda e: e["pre4"]["qf"] / 2,
    "pred_rot": lambda e: e["rot4"]["g1"] + e["rot4"]["qf"] / 2,
    "curv_rot": lambda e: e["rot4"]["qf"] / 2,
    "fisher": lambda e: e["pre4"]["fish"],
    "sens_lowE": lambda e: e["pre4"]["fish"] / e["pre4"]["ey"].clamp_min(1e-30),
    "err_rot": lambda e: e["rot4"]["en"],
    "energy": lambda e: e["pre4"]["ey"],
    "fishen_rot": lambda e: e["rot4"]["fish"] * e["rot4"]["en"],  # SqueezeLLM-like Fisher-weighted error, own channel
    # factorized curvature under rotation (THEORY M2): Fisher times the 128-block mean error energy
    # shortcut: block hard-coded to 128 (= --group, the rotation block, in every run so far); pass the group if that ever changes
    "fishblk_rot": lambda e: e["rot4"]["fish"] * e["rot4"]["en"].view(-1, 128).mean(1).repeat_interleave(128),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["est", "grad", "gt", "eval"], required=True)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--calib", type=int, default=8, help="est: train windows")
    ap.add_argument("--est-data", choices=["wikitext", "c4", *PILE], default="wikitext", help="est: train text (c4 = C4 train shard 0; out-of-domain check)")
    ap.add_argument("--fisher-samples", type=int, default=8)
    ap.add_argument("--gt-windows", type=int, default=16)
    ap.add_argument("--per-module", type=int, default=4, help="gt: random channels per module")
    ap.add_argument("--gt-kinds", default="pre4", help="gt: comma list of error kinds from KINDS (pre4 = no Hadamard, rot4 = with Hadamard)")
    ap.add_argument("--gt-layers", default="", help="gt: comma list of layer indices to test; empty = all layers")
    ap.add_argument("--batch", type=int, default=4, help="gt: tests per forward; eval: windows per forward")
    ap.add_argument("--max-minutes", type=float, default=1e9)
    ap.add_argument("--eval-windows", type=int, default=64, help="0 = whole split")
    ap.add_argument("--eval-data", choices=["wikitext", "c4", *PILE], default="wikitext")
    ap.add_argument("--k", default="2", help="eval: channels per module, comma list; 'f0.03125' = that fraction of each module's rows")
    ap.add_argument("--hows", default="restore,isolate", help="eval: comma list of restore (fp), isolate (fp), isolate8 (INT8, no rotation)")
    ap.add_argument("--selectors", default="all", help="eval: comma list of selector names, or all")
    ap.add_argument("--budget-group", default="", help="eval: comma list; also run al_H with these quantization groups (Hadamard block stays --group)")
    ap.add_argument("--rot-seed", type=int, default=0, help="Hadamard sign seed offset; 0 = the seeds used so far")
    ap.add_argument("--random-seeds", type=int, default=3)
    ap.add_argument("--align", choices=["grid", "cat"], default="grid",
                    help="grid: diagonal scale searched on a grid (k=1); cat: block CAT, arXiv 2603.04359 Eq. 7")
    ap.add_argument("--cat-k", type=int, default=None, help="CAT block size (default --group)")
    ap.add_argument("--zs-tasks", default="", help="eval: comma list of lm-eval tasks; zero-shot accuracy to zs.csv instead of perplexity (needs the lm-eval venv)")
    ap.add_argument("--zs-batch", type=int, default=16)
    ap.add_argument("--zs-configs", default="", help="eval, zs: comma list of config:selector:k:how to evaluate; empty = all")
    # bf16 eval adds bf16 rounding to the dequantized weights and bf16 logits: never mix with fp32 results in one --out-dir
    ap.add_argument("--weights-dtype", choices=["fp32", "bf16"], default="fp32", help="dtype the model is loaded in; estimator math stays fp32/fp64")
    ap.add_argument("--grad-hooks", action="store_true", help="grad: g1 (rot4 only) from output-gradient hooks, no weight gradients (8B)")
    ap.add_argument("--orig-on-cpu", action="store_true", help="eval: keep the original weights in pinned CPU memory (8B)")
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def rmul(W, B) -> torch.Tensor:
    """W [out, in] times blockdiag(B), B [n_blocks, k, k]."""
    nb, k, _ = B.shape
    return torch.einsum("onk,nkj->onj", W.reshape(W.shape[0], nb, k), B).reshape(W.shape)


ROT = {"seed": 0}


def eff_weight(W, s, layer: int, bits: int, group: int, rotate: bool, drop=None, block=None) -> torch.Tensor:
    """fp32 weight the aligned (+rotated) quantized pipeline applies to x. drop: rows taken out of the mix.
    s: per-input-channel scale vector (x/s, W s), or a CAT pair (M, M^-1) of block matrices (x M, W M^-1)."""
    cat = isinstance(s, tuple)
    Ws = rmul(W.float(), s[1]) if cat else W.float() * s
    if drop is not None and len(drop):
        Ws = Ws.clone()
        Ws[drop] = 0
    undo = (lambda A: rmul(A, s[0])) if cat else (lambda A: A / s)
    if not rotate:
        return undo(quantize_weight(Ws, bits, group))
    off = 1_000_003 * ROT["seed"]
    r_in = BlockHadamardRotation(W.shape[1], block or group, 2 * layer + off, W.device)
    r_out = BlockHadamardRotation(W.shape[0], block or group, 2 * layer + 1 + off, W.device)
    Wp = r_out.apply(r_in.apply(Ws).T).T  # R_out^T (W s) R_in
    return undo(r_out.apply_T(r_in.apply_T(quantize_weight(Wp, bits, group)).T).T)  # R_out Q R_in^T, then undo alignment


def spd_pow(A, p):
    w, V = torch.linalg.eigh((A + A.transpose(-1, -2)) / 2)
    w = w.clamp_min(w.amax(-1, keepdim=True) * 1e-12)
    return (V * w.unsqueeze(-2) ** p) @ V.transpose(-1, -2)


def cat_blocks(Sw, Sx, ridge=1e-6):
    """Per block, M = (Sw # Sx^-1)^(1/2) with A # B = A^(1/2) (A^(-1/2) B A^(-1/2))^(1/2) A^(1/2) (Eq. 7)."""
    k = Sx.shape[-1]
    eye = torch.eye(k, dtype=Sx.dtype, device=Sx.device)
    damp = lambda S: S + ridge * S.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * eye
    Sx, Sw = damp(Sx), damp(Sw)
    Ah, Aih = spd_pow(Sw, 0.5), spd_pow(Sw, -0.5)
    M = spd_pow(Ah @ spd_pow(Aih @ spd_pow(Sx, -1.0) @ Aih, 0.5) @ Ah, 0.5)
    Mi = torch.linalg.inv(M)
    # Eq. 8 self-check: M Sx M = M^-1 Sw M^-1
    l, r = M @ Sx @ M, Mi @ Sw @ Mi
    err = ((l - r).norm(dim=(-2, -1)) / r.norm(dim=(-2, -1))).max().item()
    assert err < 1e-3, f"CAT Eq. 8 check failed: relative error {err:.2e}"
    return M, Mi


@torch.no_grad()
def get_cat(args, tok, model, targets) -> dict:
    """Block CAT alignment per shared input (q/k/v and gate/up share one, as in the paper), from 8 train windows."""
    path = os.path.join(args.out_dir, "cat.pt")
    if os.path.exists(path):
        return {n: (M.cuda(), Mi.cuda()) for n, (M, Mi) in torch.load(path).items()}
    k = args.cat_k or args.group
    ids = load_wikitext2(tok, split="train", seq_len=args.seq_len, max_windows=8).cuda()
    S = {}
    def acc(n):
        def h(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double().view(-1, inp[0].shape[-1] // k, k)
            S[n] = S.get(n, 0) + torch.einsum("tbi,tbj->bij", x, x)
        return h
    hs = [m.register_forward_hook(acc(n)) for _, n, m in targets]
    for i in range(ids.shape[0]):
        model(input_ids=ids[i:i + 1], use_cache=False)
    for h in hs:
        h.remove()
    share = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "gate_proj": "gu", "up_proj": "gu"}
    groups = {}
    for layer, n, m in targets:
        groups.setdefault((layer, share.get(n.split(".")[-1], n.split(".")[-1])), []).append((n, m))
    out = {}
    for members in groups.values():
        Sx = S[members[0][0]] / ids.numel()
        Sw = sum(torch.einsum("obi,obj->bij", *(2 * [m.weight.double().view(m.out_features, -1, k)])) for _, m in members)
        M, Mi = cat_blocks(Sw, Sx)
        for n, m in members:
            W = m.weight.float()
            gap = ((rmul(rmul(W, Mi.float()), M.float()) - W).norm() / W.norm()).item()
            assert gap < 1e-3, f"{n}: CAT round trip not exact ({gap:.1e})"
            out[n] = (M.float(), Mi.float())
    torch.save({n: (M.cpu(), Mi.cpu()) for n, (M, Mi) in out.items()}, path)
    print(f"CAT k={k}: {len(groups)} groups", flush=True)
    return out


PCA_SHARE = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "gate_proj": "gu", "up_proj": "gu"}
PCA_N = {"qkv": 3, "gu": 2}


def pca_fam(layer: int, name: str):
    f = PCA_SHARE.get(name.split(".")[-1], name.split(".")[-1])
    return layer, f, PCA_N.get(f, 1)


def pca_rank(out: int, inp: int, k: int, nshare: int, g: int) -> int:
    """Columns r whose fp16 cost matches k restored rows. Rows add k*in*(16-4-16/g) bits (the 4-bit slots are replaced);
    PCA keeps W_lo dense at 4 bit, so it pays W_hi = 16*r*out plus U = 16*r*in/nshare (U shared by the modules on one input):
    r = k*in*(12-16/g) / (16*(out+in/nshare)), rounded; extra bits then match within half a column."""
    return min(inp, round(k * inp * (12 - 16 / g) / (16 * (out + inp / nshare))))


def pca_eff(W, U, s, layer: int, group: int, rotate: bool, block) -> torch.Tensor:
    """W_hi = W U U^T stays fp16 (kept in the original x space); the rest W - W_hi goes through the aligned (+rotated) INT4 pipeline."""
    Wf = W.float()
    Whi = (Wf @ U) @ U.T
    return eff_weight(Wf - Whi, s, layer, 4, group, rotate, block=block) + Whi


@torch.no_grad()
def get_pca(args, tok, model, targets, rmax) -> dict:
    """Top-rmax eigenvectors (descending, fp32, CPU) of E[x x^T] per shared input, from the same 8 train windows as get_cat.
    One layer at a time (forward stopped after the layer), fp64 covariance of at most 4 inputs alive (8192^2 fp64 = 512 MB each)."""
    class Stop(Exception):
        pass
    def stop(mod, inp, out):
        raise Stop
    ids = load_wikitext2(tok, split="train", seq_len=args.seq_len, max_windows=8).cuda()
    U = {}
    for layer in sorted({l for l, _, _ in targets}):
        mods = [(n, m) for l, n, m in targets if l == layer]
        C, hs = {}, []
        for n, m in mods:
            f = pca_fam(layer, n)[:2]
            if f not in C:
                C[f] = 0
                def acc(mod, inp, out, f=f):
                    x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
                    C[f] = C[f] + x.T @ x
                hs.append(m.register_forward_hook(acc))
        hs.append(mods[-1][1].register_forward_hook(stop))  # last module of the layer: nothing later is needed
        for i in range(ids.shape[0]):
            try:
                model(input_ids=ids[i:i + 1], use_cache=False)
            except Stop:
                pass
        for h in hs:
            h.remove()
        for f, c in C.items():
            r = rmax[f]
            U[f] = torch.linalg.eigh(c / ids.numel())[1][:, -r:].flip(1).float().cpu() if r else torch.zeros(c.shape[0], 0)
        del C
    return U


@torch.no_grad()
def get_scales(args, tok, model, targets) -> dict:
    path = os.path.join(args.out_dir, "scales.pt")
    if os.path.exists(path):
        return {n: s.cuda() for n, s in torch.load(path).items()}
    ids = load_wikitext2(tok, split="train", seq_len=args.seq_len, max_windows=1).cuda()
    X = {}
    hs = [m.register_forward_hook(lambda mod, inp, out, n=n: X.__setitem__(n, inp[0].detach()[0].float()))
          for _, n, m in targets]
    model(input_ids=ids, use_cache=False)
    for h in hs:
        h.remove()
    scales, alphas = {}, {}
    for layer, n, m in targets:
        x, W = X.pop(n), m.weight.float()
        ref, xm = x @ W.T, x.abs().mean(0).clamp_min(1e-5)
        best = None
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            s = xm ** a
            s = s / s.log().mean().exp()
            err = (x @ eff_weight(W, s, layer, 4, args.group, True).T - ref).pow(2).mean().item()
            if best is None or err < best[0]:
                best = (err, a, s)
        scales[n], alphas[n] = best[2], best[1]
    torch.save({n: s.cpu() for n, s in scales.items()}, path)
    json.dump(alphas, open(os.path.join(args.out_dir, "alphas.json"), "w"), indent=1)
    print(f"alignment: alpha counts {dict(sorted({a: list(alphas.values()).count(a) for a in alphas.values()}.items()))}", flush=True)
    return scales


def run_est(args, tok, model, targets, scales) -> None:
    ids = load_text(tok, args.est_data, True, args.seq_len, args.calib).cuda()
    K = args.fisher_samples
    layer_of = {n: l for l, n, _ in targets}
    st = {"pass": None}
    z = lambda n: torch.zeros(n, dtype=torch.float64, device="cuda")
    acc = {(k, n): {f: z(m.out_features) for f in ("g1", "qf", "fish", "en", "ey")} for k in KINDS for _, n, m in targets}

    def accumulate(name, mod, x, y, g):
        g, x = g.float().double(), x.float()  # x kept in the module dtype between hooks (no fp32 copy alive for all modules)
        W = mod.weight
        for k, (bits, rot) in KINDS.items():
            d = (x @ (eff_weight(W, scales[name], layer_of[name], bits, args.group, rot) - W.float()).T).double()
            gd = g * d
            a = acc[k, name]
            if st["pass"] == "real":
                a["g1"] += gd.sum(0)
                a["en"] += d.pow(2).sum(0)
                a["ey"] += y.double().pow(2).sum(0)
            else:
                a["qf"] += gd.sum(0).pow(2) / K
                a["fish"] += g.pow(2).sum(0) / K

    def fwd_hook(name):
        def h(mod, inp, out):
            if out.requires_grad:
                x, y = inp[0].detach()[0], out.detach()[0]
                out.register_hook(lambda g, x=x, y=y, mod=mod: accumulate(name, mod, x, y, g[0]))
        return h

    hs = [m.register_forward_hook(fwd_hook(n)) for _, n, m in targets]
    gen = torch.Generator(device="cuda").manual_seed(0)
    t0, clean = time.time(), 0.0
    for w in range(ids.shape[0]):
        win = ids[w:w + 1]
        emb = model.get_input_embeddings()(win).detach().requires_grad_()
        logits = model(inputs_embeds=emb, use_cache=False).logits[0, :-1].float()
        st["pass"] = "real"
        L = F.cross_entropy(logits, win[0, 1:], reduction="sum")
        clean += L.item()
        torch.autograd.grad(L, emb, retain_graph=True)
        probs = torch.softmax(logits.detach(), dim=-1)
        for k in range(K):
            st["pass"] = k
            y = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            torch.autograd.grad(F.cross_entropy(logits, y, reduction="sum"), emb, retain_graph=k < K - 1)
        del logits, probs, emb, L
        print(f"est window {w + 1}/{ids.shape[0]}, {time.time() - t0:.0f} s", flush=True)
    for h in hs:
        h.remove()
    n_tok = ids.shape[0] * (ids.shape[1] - 1)
    for k in KINDS:
        rows = []
        for layer, n, m in targets:
            a = {f: (v / n_tok).tolist() for f, v in acc[k, n].items()}
            rows += [{"layer": layer, "module": n.split(".")[-1], "channel": c, **{f: a[f][c] for f in a}} for c in range(m.out_features)]
        write_csv(os.path.join(args.out_dir, f"est_{k}.csv"), rows)
    json.dump({"clean_loss_calib": clean / n_tok, "n_tok": n_tok, "wall_s": time.time() - t0, "args": vars(args)},
              open(os.path.join(args.out_dir, "est_meta.json"), "w"), indent=1)
    print(f"EST DONE {time.time() - t0:.0f} s", flush=True)


@torch.no_grad()
def grad_scores(targets, scales, args) -> dict:
    """Per kind and module: sum_j E[c,j] G[c,j] with G the weight gradient accumulated so far (E built once per call)."""
    out = {}
    for k, (bits, rot) in KINDS.items():
        for layer, n, m in targets:
            W = m.weight.detach()
            E = eff_weight(W, scales[n], layer, bits, args.group, rot) - W.float()
            out[k, n] = (E.double() * m.weight.grad.double()).sum(1)
            del E
    return out


def run_grad(args, tok, model, targets, scales) -> None:
    ids = load_text(tok, args.est_data, True, args.seq_len, args.calib).cuda()
    N, tpw = ids.shape[0], ids.shape[1] - 1  # the loader returns fewer windows than asked when the text is short
    assert N >= 16, f"need >= 16 windows, got {N}"
    nA = math.ceil(N / 2)
    kinds = {"rot4": KINDS["rot4"]} if args.grad_hooks else KINDS  # hooks: rot4 only (the one eval reads); E is rebuilt per hook call
    if args.grad_hooks:  # g1_c = sum_t g_tc d_tc with d = x E^T, as run_est's real pass; E per call: all-module E does not fit next to 8B weights
        layer_of, hs = {n: l for l, n, _ in targets}, []
        acc = {n: torch.zeros(m.out_features, dtype=torch.float64, device="cuda") for _, n, m in targets}
        def fwd_hook(n):
            def h(mod, inp, out):
                if out.requires_grad:
                    x = inp[0].detach()[0]
                    def bw(g):
                        W = mod.weight
                        d = x.float() @ (eff_weight(W, scales[n], layer_of[n], 4, args.group, True) - W.float()).T
                        acc[n] += (g[0].double() * d.double()).sum(0)
                    out.register_hook(bw)
            return h
        hs = [m.register_forward_hook(fwd_hook(n)) for _, n, m in targets]
        snap = lambda: {("rot4", n): v.clone() for n, v in acc.items()}
    else:
        for _, _, m in targets:
            m.weight.requires_grad_(True)
        snap = lambda: grad_scores(targets, scales, args)
    snaps, t0 = {}, time.time()
    for w in range(N):
        win = ids[w:w + 1]
        if args.grad_hooks:
            emb = model.get_input_embeddings()(win).detach().requires_grad_()  # grad must reach the projection outputs
            logits = model(inputs_embeds=emb, use_cache=False).logits[0, :-1].float()
            torch.autograd.grad(F.cross_entropy(logits, win[0, 1:], reduction="sum"), emb)
            del emb
            assert w > 0 or any(v.any() for v in acc.values()), "grad hooks never fired"
        else:
            logits = model(input_ids=win, use_cache=False).logits[0, :-1].float()
            F.cross_entropy(logits, win[0, 1:], reduction="sum").backward()
        del logits
        if w + 1 == 8:
            snaps["g1_8"] = snap()
        if w + 1 == nA:
            snaps["g1a"] = snap()
            if args.grad_hooks:
                for v in acc.values():
                    v.zero_()
            else:
                for _, _, m in targets:
                    m.weight.grad = None
        print(f"grad window {w + 1}/{N}, {time.time() - t0:.0f} s", flush=True)
    snaps["g1b"] = snap()
    for h in hs if args.grad_hooks else []:
        h.remove()
    assert all(m.weight.grad is None for _, _, m in targets) or not args.grad_hooks, "grad-hooks must not materialize weight gradients"
    den = {"g1_8": 8 * tpw, "g1a": nA * tpw, "g1b": (N - nA) * tpw}
    check = {}
    for k in kinds:
        rows = []
        for layer, n, m in targets:
            v = {f: (snaps[f][k, n] / den[f]).tolist() for f in den}
            rows += [{"layer": layer, "module": n.split(".")[-1], "channel": c, "g1": (snaps["g1a"][k, n][c].item() + snaps["g1b"][k, n][c].item()) / (N * tpw),
                      **{f: v[f][c] for f in v}} for c in range(m.out_features)]
        # old est got g1 from output-gradient hooks (same math), so on the same 8 windows it must agree
        pe, pm = (os.path.join(args.out_dir, f"est_{k}.csv"), os.path.join(args.out_dir, "est_meta.json"))
        ma = json.load(open(pm))["args"] if os.path.exists(pm) else {}
        if os.path.exists(pe) and (ma.get("est_data", "wikitext"), ma.get("seq_len"), ma.get("calib")) == (args.est_data, args.seq_len, 8):
            old = {(r["layer"], r["module"], r["channel"]): float(r["g1"]) for r in csv.DictReader(open(pe))}
            a = torch.tensor([old[str(r["layer"]), r["module"], str(r["channel"])] for r in rows], dtype=torch.float64)
            b = torch.tensor([r["g1_8"] for r in rows], dtype=torch.float64)
            check[k] = {"pearson": torch.corrcoef(torch.stack([a, b]))[0, 1].item(), "max_abs_rel": ((a - b).abs().max() / a.abs().max()).item()}
            print(f"self-check {k}: {check[k]}", flush=True)
            assert check[k]["pearson"] > 0.99, f"grad g1_8 disagrees with est g1 for {k}: {check[k]}"
        else:
            check[k] = "skipped"
        write_csv(os.path.join(args.out_dir, f"est_grad_{k}.csv"), rows)
    json.dump({"n_windows": N, "nA": nA, "nB": N - nA, "est_data": args.est_data, "seq_len": args.seq_len, "wall_s": time.time() - t0,
               "self_check": check, "args": vars(args)}, open(os.path.join(args.out_dir, "est_grad_meta.json"), "w"), indent=1)
    print(f"GRAD DONE {time.time() - t0:.0f} s", flush=True)


@torch.no_grad()
def run_gt(args, tok, model, targets, scales) -> None:
    ids = load_wikitext2(tok, split="validation", seq_len=args.seq_len, max_windows=args.gt_windows).cuda()
    N, T, Bt = ids.shape[0], ids.shape[1], args.batch
    dec, head = model.get_decoder(), model.get_output_embeddings()
    all_layers = dec.layers
    clean = torch.cat([nll_rows(head, dec(input_ids=ids[i:i + 4], use_cache=False).last_hidden_state, ids[i:i + 4]) for i in range(0, N, 4)])

    def from_layer(l, h):
        dec.layers = all_layers[l:]
        try:
            return dec(inputs_embeds=h, use_cache=False).last_hidden_state
        finally:
            dec.layers = all_layers

    def layer_inputs(l):
        out = []
        hk = all_layers[l].register_forward_pre_hook(lambda mod, a, kw: out.append((a[0] if a else kw["hidden_states"]).detach().clone()), with_kwargs=True)
        dec.layers = all_layers[: l + 1]
        try:
            for i in range(0, N, 4):
                dec(input_ids=ids[i:i + 4], use_cache=False)
        finally:
            dec.layers = all_layers
            hk.remove()
        return torch.cat(out)

    kinds = args.gt_kinds.split(",")
    assert all(k in KINDS for k in kinds), f"--gt-kinds must be from {list(KINDS)}"
    out_path = os.path.join(args.out_dir, "gt.csv")
    old = list(csv.DictReader(open(out_path))) if os.path.exists(out_path) else []
    assert not old or "kind" in old[0], f"{out_path} is from the old format (no kind column): use a new --out-dir"
    done = {(int(r["layer"]), r["module"]) for r in old}
    layers = sorted({l for l, _, _ in targets})
    if args.gt_layers:
        want = {int(x) for x in args.gt_layers.split(",")}
        assert want <= set(layers), f"--gt-layers {sorted(want - set(layers))} not in the model"
        layers = [l for l in layers if l in want]
    order = [layers[i] for i in torch.randperm(len(layers), generator=torch.Generator().manual_seed(0)).tolist()]
    rep = ids.repeat(Bt, 1)
    meta = {"noop_max_abs": 0.0, "start_gap": {}, "clean_loss": (clean.sum() / (N * (T - 1))).item(), "args": vars(args)}
    t0 = time.time()
    for l in order:
        mods = [(mi, n, m) for mi, (layer, n, m) in enumerate(targets) if layer == l and (l, n.split(".")[-1]) not in done]
        if not mods:
            continue
        if (time.time() - t0) / 60 > args.max_minutes:
            print(f"time limit reached before layer {l}", flush=True)
            break
        hrep = layer_inputs(l).repeat(Bt, 1, 1)
        base = nll_rows(head, from_layer(l, hrep), rep)
        gap = (base.view(Bt, N)[0] - clean).abs().max().item() / (T - 1)
        meta["start_gap"][str(l)] = gap
        assert gap < 1e-5, f"layer {l}: partial forward differs from full by {gap}"
        for mi, n, m in mods:
            t1 = time.time()
            # E[k][c] = quantized row c minus fp row c, so the hook makes channel c's output its quantized output (rot4: after the Hadamard round trip)
            E = {k: eff_weight(m.weight, scales[n], l, KINDS[k][0], args.group, KINDS[k][1]) - m.weight.float() for k in kinds}
            chans = sorted(torch.randperm(m.out_features, generator=torch.Generator().manual_seed(mi))[:args.per_module].tolist())
            tests, res = [(k, c) for k in kinds for c in chans], {}
            for b in range(0, len(tests), Bt):
                chunk = tests[b:b + Bt] + [("noop", 0)] * max(0, Bt - len(tests[b:b + Bt]))
                hd = m.register_forward_hook(make_hook(chunk, N, E))
                try:
                    d = ((nll_rows(head, from_layer(l, hrep), rep) - base) / (T - 1)).view(Bt, N)
                finally:
                    hd.remove()
                for t, dv in zip(chunk, d):
                    if t[0] == "noop":
                        meta["noop_max_abs"] = max(meta["noop_max_abs"], dv.abs().max().item())
                    else:
                        res[t] = dv
            h = N // 2
            write_csv(out_path, [{"layer": l, "module": n.split(".")[-1], "kind": k, "channel": c, "d_all": res[k, c].mean().item(),
                                  "d_h1": res[k, c][:h].mean().item(), "d_h2": res[k, c][h:].mean().item()} for k, c in tests], append=True)
            json.dump(meta, open(os.path.join(args.out_dir, "gt_meta.json"), "w"), indent=1)
            print(f"{time.strftime('%H:%M:%S')} L{l} {n.split('.')[-1]}: {len(tests)} tests, {time.time() - t1:.1f} s | "
                  f"noop {meta['noop_max_abs']:.1e} | gap {gap:.1e}", flush=True)
    print("GT DONE", flush=True)


@torch.no_grad()
def run_eval(args, tok, model, targets, scales) -> None:
    n_win = args.eval_windows or None
    test = load_text(tok, args.eval_data, False, args.seq_len, n_win).cuda()
    print(f"eval on {args.eval_data}: {test.shape[0]} windows of {test.shape[1]}", flush=True)
    dec, head = model.get_decoder(), model.get_output_embeddings()
    orig = {n: m.weight.data.to("cpu", copy=True).pin_memory() if args.orig_on_cpu else m.weight.data.clone() for _, n, m in targets}
    assert all(o.is_pinned() for o in orig.values()) or not args.orig_on_cpu
    est = {k: {} for k in ("pre4", "rot4")}
    for k in est:
        for r in csv.DictReader(open(os.path.join(args.out_dir, f"est_{k}.csv"))):
            est[k].setdefault(f"{r['layer']}.{r['module']}", []).append([float(r[f]) for f in ("g1", "qf", "fish", "en", "ey")])
    key = {n: f"{l}.{n.split('.')[-1]}" for l, n, _ in targets}
    big, gpath = {}, os.path.join(args.out_dir, "est_grad_rot4.csv")  # g1 from many windows (--mode grad); optional
    if os.path.exists(gpath):  # extra columns s_<tag> are finished scores built offline (e.g. multi-domain); each becomes a selector
        for r in csv.DictReader(open(gpath)):
            big.setdefault(f"{r['layer']}.{r['module']}", []).append([float(r["g1"])] + [float(r[c]) for c in r if c.startswith("s_")])
        extra = [c for c in r if c.startswith("s_")]
    score = {}
    for _, n, _ in targets:
        e = {k: dict(zip(("g1", "qf", "fish", "en", "ey"), torch.tensor(est[k][key[n]], dtype=torch.float64).T)) for k in est}
        score[n] = {s: f(e) for s, f in SELECTORS.items()}
        if big:
            b = torch.tensor(big[key[n]], dtype=torch.float64).T
            score[n]["g1_rot_big"] = b[0]
            score[n].update(zip(extra, b[1:]))
            score[n]["pred_rot_big"] = score[n]["g1_rot_big"] + score[n]["curv_rot"]
        o = orig[n].cuda().float()
        Ws = rmul(o, scales[n][1]) if isinstance(scales[n], tuple) else o * scales[n]
        score[n]["wmax"] = Ws.abs().amax(1).double().cpu()  # largest aligned weight per row (weight-outlier rows, SpQR-like)

    restore = {}  # module name -> (channel indices, their weight rows): those outputs are recomputed with these rows
    def hook(n):
        def h(mod, inp, out):
            if n in restore:
                idx, rows = restore[n]
                b = None if mod.bias is None else mod.bias[idx]
                out[..., idx] = F.linear(inp[0], rows, b).to(out.dtype)
            return out
        return h
    hs = [m.register_forward_hook(hook(n)) for _, n, m in targets]

    bits = {"total": 0.0, "n": 0, "r": 0}  # r: PCA columns kept fp16 (how=pca)
    def configure(align, rotate, sel=None, how="restore", group=None):
        """how: restore = chosen outputs recomputed in fp; isolate = chosen rows also left out of the rotation mix, kept fp;
        isolate8 = left out of the mix and quantized to INT8 (aligned, not rotated). Tracks bits per weight (scales fp16)."""
        restore.clear()
        g = group or args.group
        bits["total"], bits["n"], bits["r"] = 0.0, 0, 0
        for l, n, m in targets:
            W = orig[n].cuda()  # no-op unless --orig-on-cpu; copy_ below casts the fp32 eff_weight to the module dtype
            bits["n"] += W.numel()
            if align is None:
                m.weight.data.copy_(W)
                bits["total"] += 16 * W.numel()
                continue
            s = scales[n] if align else torch.ones(W.shape[1], device=W.device)  # (M, M^-1) pair under --align cat
            if how == "pca":  # sel = columns r per module; W_lo dense at 4 bit, W_hi = W U U^T and U fp16 (see pca_rank)
                r, (_, fam, ns) = sel[n], pca_fam(l, n)
                m.weight.data.copy_(pca_eff(W, Upca[(l, fam)][:, :r].cuda(), s, l, g, rotate, args.group))
                # shortcut: U charged per module at its own r; with GQA k/v use a smaller r than the shared U holds (slight undercount, favours the baseline)
                # n_restored counts PCA columns, not rows: compare methods on bpw
                bits["total"] += (4 + 16 / g) * W.numel() + 16 * r * (W.shape[0] + W.shape[1] / ns)
                bits["r"] += r
                continue
            idx = sel[n] if sel is not None else None
            m.weight.data.copy_(eff_weight(W, s, l, 4, g, rotate, drop=idx if how != "restore" else None, block=args.group))
            bits["total"] += (4 + 16 / g) * W.numel()
            if idx is not None and len(idx):
                if how == "isolate8":
                    rows, row_bits = eff_weight(W[idx], s, l, 8, g, False).to(W.dtype), 8 + 16 / g
                else:
                    rows, row_bits = W[idx], 16
                restore[n] = (idx, rows)
                bits["total"] += (row_bits - 4 - 16 / g) * len(idx) * W.shape[1]

    def ppl():
        tot = sum(nll_rows(head, dec(input_ids=test[i:i + args.batch], use_cache=False).last_hidden_state, test[i:i + args.batch]).sum().item()
                  for i in range(0, test.shape[0], args.batch))
        nll = tot / (test.shape[0] * (test.shape[1] - 1))
        return nll, math.exp(nll)

    out_path = os.path.join(args.out_dir, "eval.csv")
    old = list(csv.DictReader(open(out_path))) if os.path.exists(out_path) else []
    for r in old:  # one setting per directory: resuming must not mix data sets, window counts or rotation seeds
        assert (r.get("data", "wikitext"), int(r.get("windows", test.shape[0])), int(r.get("rot_seed", 0))) == \
            (args.eval_data, test.shape[0], args.rot_seed), f"{out_path} holds another setting; use a fresh --out-dir"
    done = {(r["config"], r["selector"], r["k"], r["how"]) for r in old}
    zs = [t for t in args.zs_tasks.split(",") if t]
    zs_path, zs_sel = os.path.join(args.out_dir, "zs.csv"), {tuple(c.split(":")) for c in args.zs_configs.split(",") if c}
    if zs:
        zs_rows = list(csv.DictReader(open(zs_path))) if os.path.exists(zs_path) else []
        done = {key4 for key4 in {(r["config"], r["selector"], r["k"], r["how"]) for r in zs_rows}
                if {r["task"] for r in zs_rows if (r["config"], r["selector"], r["k"], r["how"]) == key4} >= set(zs)}
    def run(config, selector, k, how, **kw):
        if (config, selector, str(k), how) in done or (zs and zs_sel and (config, selector, str(k), how) not in zs_sel):
            return
        t1 = time.time()
        configure(how=how if how != "-" else "restore", **kw)
        if zs:
            import lm_eval  # only in the separate lm-eval venv
            from lm_eval.models.huggingface import HFLM
            # instance passed as is: HFLM only reads its device and does not cast or move it; the restore hooks stay on the modules
            res = lm_eval.simple_evaluate(model=HFLM(pretrained=model, tokenizer=tok, batch_size=args.zs_batch, max_length=args.seq_len),
                                          tasks=zs, num_fewshot=0, log_samples=False, bootstrap_iters=1000)["results"]
            rows = [{"config": config, "selector": selector, "k": k, "how": how, "task": t, "metric": m.split(",")[0], "value": v,
                     "stderr": r.get(m.replace(",", "_stderr,"), ""), "n_restored": bits["r"] + sum(len(x[0]) for x in restore.values()),
                     "bpw": bits["total"] / bits["n"], "rot_seed": args.rot_seed, "seconds": time.time() - t1}
                    for t, r in res.items() for m, v in r.items() if m.split(",")[0] in ("acc", "acc_norm", "perplexity")]
            write_csv(zs_path, rows, append=True)
            print(f"{time.strftime('%H:%M:%S')} {config:8s} {selector:10s} k={k} {how:8s} " +
                  " ".join(f"{x['task']}/{x['metric']} {x['value']:.4f}" for x in rows) + f" ({time.time() - t1:.0f} s)", flush=True)
            return
        nll, p = ppl()
        n_restored = bits["r"] + sum(len(v[0]) for v in restore.values())
        write_csv(out_path, [{"config": config, "selector": selector, "k": k, "how": how, "nll": nll, "ppl": p,
                              "n_restored": n_restored, "bpw": bits["total"] / bits["n"], "data": args.eval_data,
                              "windows": test.shape[0], "rot_seed": args.rot_seed}], append=True)
        print(f"{time.strftime('%H:%M:%S')} {config:8s} {selector:10s} k={k} {how:8s} ppl {p:.4f} ({time.time() - t1:.0f} s)", flush=True)

    run("fp", "-", 0, "-", align=None, rotate=False)
    run("int4", "-", 0, "-", align=False, rotate=False)
    run("int4_H", "-", 0, "-", align=False, rotate=True)
    run("al", "-", 0, "-", align=True, rotate=False)
    run("al_H", "-", 0, "-", align=True, rotate=True)
    for bg in [int(x) for x in args.budget_group.split(",") if x]:
        run(f"al_H_g{bg}", "-", 0, "-", align=True, rotate=True, group=bg)
    names = args.selectors.split(",")
    if args.selectors == "all":
        names = list(SELECTORS) + ["wmax", "pred_rot_global"] + (["g1_rot_big", "pred_rot_big_global"] if big else [])
    for s in names:
        assert s.removesuffix("_global") in score[targets[0][1]], f"selector {s} unavailable (est_grad_rot4.csv missing?)"
    glob = {s: s.removesuffix("_global") for s in names if s.endswith("_global")}
    kofs = {ks: (lambda m, ks=ks: max(1, round(float(ks[1:]) * m.out_features))) if ks.startswith("f") else (lambda m, ks=ks: int(ks)) for ks in args.k.split(",")}
    rcols, Upca = {}, {}
    if "pca" in args.hows.split(","):  # ResQ/ROSAQ-style baseline: eigenvectors once for the largest r any k needs
        rcols = {ks: {n: pca_rank(m.out_features, m.in_features, kofs[ks](m), pca_fam(l, n)[2], args.group) for l, n, m in targets} for ks in kofs}
        rmax = {}
        for l, n, _ in targets:
            f = pca_fam(l, n)[:2]
            rmax[f] = max(rmax.get(f, 0), *(rc[n] for rc in rcols.values()))
        configure(None, False)  # covariances from the fp model, not the al_H weights left by the runs above
        Upca = get_pca(args, tok, model, targets, rmax)
        n0, W0 = targets[0][1], orig[targets[0][1]].cuda()  # self-check: r = in is exact, r = 0 is plain al_H
        l0 = targets[0][0]
        full = pca_eff(W0, torch.linalg.qr(torch.randn(W0.shape[1], W0.shape[1], generator=torch.Generator().manual_seed(0)).to(W0.device))[0], scales[n0], l0, args.group, True, args.group)
        assert (full - W0.float()).norm() / W0.norm() < 1e-4, "PCA r = in is not exact"
        assert torch.equal(pca_eff(W0, Upca[(l0, pca_fam(l0, n0)[1])][:, :0].cuda(), scales[n0], l0, args.group, True, args.group),
                           eff_weight(W0, scales[n0], l0, 4, args.group, True, block=args.group)), "PCA r = 0 differs from al_H"
    for kspec in args.k.split(","):
        kof = (lambda m: max(1, round(float(kspec[1:]) * m.out_features))) if kspec.startswith("f") else (lambda m: int(kspec))
        sels = {s: {n: score[n][s].topk(kof(m)).indices.cuda() for _, n, m in targets} for s in names if s not in glob}
        for gname, base in glob.items():  # same total count, chosen across all modules by predicted dloss (comparable units)
            allsc = torch.cat([score[n][base] for _, n, _ in targets])
            top = set(allsc.topk(sum(kof(m) for _, _, m in targets)).indices.tolist())
            sels[gname], o = {}, 0
            for _, n, m in targets:
                sels[gname][n] = torch.tensor(sorted(i - o for i in top if o <= i < o + m.out_features), dtype=torch.long).cuda()
                o += m.out_features
        for seed in range(args.random_seeds):
            sels[f"random{seed}"] = {n: torch.randperm(m.out_features, generator=torch.Generator().manual_seed(seed * 100_003 + mi))[:kof(m)].cuda()
                                     for mi, (_, n, m) in enumerate(targets)}
        for how in args.hows.split(","):
            if how == "pca":  # one row per k, no selector
                run("al_H", "pca", kspec, "pca", align=True, rotate=True, sel=rcols[kspec])
                continue
            for s, sel in sels.items():
                run("al_H", s, kspec, how, align=True, rotate=True, sel=sel)
    for h in hs:
        h.remove()
    print("EVAL DONE", flush=True)


def main() -> None:
    args = parse_args()
    ROT["seed"] = args.rot_seed
    assert args.mode == "grad" or not args.grad_hooks, "--grad-hooks only applies to --mode grad"
    os.makedirs(args.out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if args.weights_dtype == "bf16" else torch.float32, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    targets = find_targets(model, PROJECTIONS)
    print(f"{args.model}: {len(targets)} modules, {sum(m.out_features for _, _, m in targets)} channels, mode {args.mode}", flush=True)
    sdpa_attention.use_gqa_in_sdpa = lambda *a, **k: False  # see run_qsens.py: keeps fp32 SDPA on the memory-efficient kernel
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        scales = (get_cat if args.align == "cat" else get_scales)(args, tok, model, targets)
        {"est": run_est, "grad": run_grad, "gt": run_gt, "eval": run_eval}[args.mode](args, tok, model, targets, scales)


if __name__ == "__main__":
    main()
