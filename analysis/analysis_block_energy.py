"""Can the per-output-block error energy under rotation (est_rot4.csv "en") be predicted without simulating Q?

Derivation (read from exp_2_cheap_est.eff_weight, cat alignment, row-vector convention, symmetric INT4 RTN, scale = amax/7 per 128 input cols):
  P = R_out^T (W M^-1) R_in,  Delta = Q(P) - P,  W_eff - W = R_out Delta R_in^T M   (M = blockdiag, rmul(A, M) = A M)
  d_t = x (W_eff - W)^T = u_t Delta^T R_out^T with u_t = x M^T R_in, so d_{t,c} = sum_{c'} R_out[c,c'] (Delta_c' . u_t), c' in block(c)
  noise model: E d_{t,c}^2 = (1/B) sum_{c' in block(c)} sum_g sigma2[c',g] U[g], U[g] = E_t ||u_{t,g}||^2,
  sigma2 = scale^2/12 (noise) or mean(Delta^2) over the group (emp). Symmetric quantizer, so no zero-point correction.
Forward hooks only, no gradients. Uses cat.pt alignment (as exp_2_cheap_est.py --align cat).
"""
import argparse
import contextlib
import csv
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_2_cheap_est import ROT, eff_weight, rmul
from run_sensitivity import find_targets
from src.data import load_wikitext2
from src.hadamard import BlockHadamardRotation
from src.quant import quantize_weight
from src.wrapper import PROJECTIONS

BITS = 4


def rots(W, layer, group):
    off = 1_000_003 * ROT["seed"]  # same seeds as eff_weight
    return (BlockHadamardRotation(W.shape[1], group, 2 * layer + off, W.device),
            BlockHadamardRotation(W.shape[0], group, 2 * layer + 1 + off, W.device))


def parts(W, MM, layer, group):
    """P = R_out^T (W M^-1) R_in and Delta = Q(P) - P."""
    r_in, r_out = rots(W, layer, group)
    P = r_out.apply(r_in.apply(rmul(W.float(), MM[1])).T).T
    return P, quantize_weight(P, BITS, group) - P


def u_of(x, MM, layer, group):
    """u = x M^T R_in, the input in the rotated aligned space (rows = tokens)."""
    return rots(x.new_zeros(group, x.shape[1]), layer, group)[0].apply(rmul(x, MM[0].transpose(-1, -2)))


def sigmas(P, Delta, group):
    """[out, n_groups] noise-model sigma2 = scale^2/12 and empirical mean(Delta^2)."""
    O, I = P.shape
    sc = (P.reshape(O, I // group, group).abs().amax(-1) / (2 ** (BITS - 1) - 1)).clamp_min(1e-8)
    return sc.pow(2) / 12, Delta.reshape(O, I // group, group).pow(2).mean(-1)


def block_mean(v, group):
    return v.reshape(-1, group).mean(1, keepdim=True).expand(-1, group).reshape(-1)


def rank(v):
    return v.argsort().argsort().double()


def pear(a, b):
    return torch.corrcoef(torch.stack([a.double(), b.double()]))[0, 1].item()


def spear(a, b):
    return pear(rank(a), rank(b))


def self_check() -> None:
    """Toy layer: simulated W_eff - W must equal u Delta^T R_out^T (identity M and a random block M)."""
    torch.manual_seed(0)
    O, I, g, layer = 256, 256, 128, 3
    W, x = torch.randn(O, I), torch.randn(50, I)
    eye = torch.eye(g).expand(I // g, g, g).clone()
    rnd = torch.eye(g) + 0.05 * torch.randn(I // g, g, g)
    for M in (eye, rnd):
        MM = (M, torch.linalg.inv(M))
        P, Delta = parts(W, MM, layer, g)
        d_sim = x @ (eff_weight(W, MM, layer, BITS, g, True) - W).T
        d_formula = rots(W, layer, g)[1].apply_T(u_of(x, MM, layer, g) @ Delta.T)
        assert torch.allclose(d_sim, d_formula, atol=1e-3 * d_sim.abs().max()), (d_sim - d_formula).abs().max()
        vn, ve = sigmas(P, Delta, g)
        U = u_of(x, MM, layer, g).pow(2).reshape(50, -1, g).sum((0, 2)) / 50
        meas = d_sim.pow(2).mean(0)
        r = (block_mean(vn @ U, g) / meas).median().item()
        assert 0.5 < r < 2, f"noise-model ratio {r}"
    print(f"self-check OK (noise/measured median ratio {r:.2f})", flush=True)


def stats(m, p, block):
    cm, cp = m.log10().clamp_min(-30), p.log10().clamp_min(-30)
    bm, bp = m.reshape(-1, block).mean(1), p.reshape(-1, block).mean(1)
    return {"spearman": spear(m, p), "pearson_raw": pear(m, p), "pearson_log": pear(cm, cp), "median_ratio": (p / m.clamp_min(1e-30)).median().item(),
            "block_spearman": spear(bm, bp), "block_pearson_log": pear(bm.log10(), bp.log10())}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--out-dir", required=True, help="dir holding cat.pt")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", default="", help="comma list of layers; empty = all")
    args = ap.parse_args()
    self_check()
    dev, g = args.device, args.group
    os.makedirs(args.save_dir, exist_ok=True)
    cat = {n: (M.to(dev).float(), Mi.to(dev).float()) for n, (M, Mi) in torch.load(os.path.join(args.out_dir, "cat.pt"), map_location="cpu").items()}
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, device_map=dev).eval()
    keep = {int(l) for l in args.layers.split(",") if l}
    targets = [t for t in find_targets(model, PROJECTIONS) if not keep or t[0] in keep]
    ids = load_wikitext2(tok, split="train", seq_len=args.seq_len, max_windows=args.windows).to(dev)
    acc, sig = {}, {}  # acc[n] = [sum d^2 [out], sum ||u_g||^2 [n_g], n_tok]

    def hook(layer, n):
        def h(mod, inp, out):
            x, W = inp[0].reshape(-1, inp[0].shape[-1]).float(), mod.weight.float()
            d = x @ (eff_weight(W, cat[n], layer, BITS, g, True) - W).T
            u = u_of(x, cat[n], layer, g)
            if n not in sig:
                sig[n] = sigmas(*parts(W, cat[n], layer, g), g)
                acc[n] = [0, 0, 0]
            a = acc[n]
            a[0] = a[0] + d.pow(2).sum(0)
            a[1] = a[1] + u.pow(2).reshape(len(x), -1, g).sum((0, 2))
            a[2] += len(x)
        return h

    hs = [m.register_forward_hook(hook(l, n)) for l, n, m in targets]
    sdpa = contextlib.nullcontext()
    if dev == "cuda":  # as exp_2_cheap_est.main
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers.integrations import sdpa_attention
        sdpa_attention.use_gqa_in_sdpa = lambda *a, **k: False
        sdpa = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    with sdpa:
        for i in range(ids.shape[0]):
            model.get_decoder()(input_ids=ids[i:i + 1], use_cache=False)  # no lm_head: hooks only need the projections
            print(f"window {i + 1}/{ids.shape[0]}", flush=True)
    for h in hs:
        h.remove()
    rows, mods = [], {}
    for layer, n, m in targets:
        d2, U, nt = acc[n]
        meas, vn, ve = (d2 / nt).cpu(), block_mean(sig[n][0] @ (U / nt), g).cpu(), block_mean(sig[n][1] @ (U / nt), g).cpu()
        mods[n] = (meas.double(), vn.double(), ve.double())
        rows += [{"layer": layer, "module": n.split(".")[-1], "channel": c, "en_meas": meas[c].item(), "en_pred_noise": vn[c].item(), "en_pred_emp": ve[c].item()}
                 for c in range(len(meas))]
    with open(os.path.join(args.save_dir, "block_energy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cat_ = lambda i: torch.cat([v[i] for v in mods.values()])
    cv = torch.cat([v[0].reshape(-1, g).std(1) / v[0].reshape(-1, g).mean(1) for v in mods.values()])
    summ = {"model": args.model, "windows": args.windows, "seq_len": args.seq_len, "group": g, "n_modules": len(mods),
            "within_block_cv_median": cv.median().item()}
    for i, k in ((1, "noise"), (2, "emp")):
        summ[k] = stats(cat_(0), cat_(i), g)
        summ[k]["median_module_spearman"] = torch.tensor([spear(v[0], v[i]) for v in mods.values()]).median().item()
    json.dump(summ, open(os.path.join(args.save_dir, "block_energy_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
