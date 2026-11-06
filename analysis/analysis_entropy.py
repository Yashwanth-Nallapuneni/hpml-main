"""Entropy / calibration check: does restoring first-order-selected channels just sharpen the output distribution?

Hypothesis: restore(s_rob_global) lowers held-out ppl below fp (Gemma-3-1b-pt wikitext 14.9 vs 20.2) with no zero-shot gain because it
pulls p toward the hard label (the model was distilled on soft targets), not because it adds ability.
Per config on held-out windows: ppl, mean entropy H(p), mean max prob, ECE (15 bins, top-1 prob vs correctness), NLL on tokens where the
fp model's top-1 is correct / wrong. Configs: fp, al_H, al_H + restore top f1/32 by s_rob_global, by curv_rot, and random.
run_mackin's run_eval keeps configure/hook/score as closures, so restore is reproduced here exactly (forward hook, F.linear(input, original rows)).
Needs --out-dir with cat.pt, est_rot4.csv and est_grad_rot4.csv (with an s_rob column). Logits are made 512 tokens at a time (262k vocab).
"""
import argparse
import csv
import math
import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import sdpa_attention

from run_mackin import eff_weight, get_cat
from run_qsens import write_csv
from run_sensitivity import find_targets
from src.data import load_text
from src.wrapper import PROJECTIONS

FRAC, CHUNK, BINS = 1 / 32, 512, 15


def ece(conf: torch.Tensor, corr: torch.Tensor, bins: int = BINS) -> float:
    """Expected calibration error, equal-width bins on the top-1 probability."""
    b = (conf.double() * bins).long().clamp(max=bins - 1)
    n = torch.bincount(b, minlength=bins).double()
    c, a = torch.bincount(b, conf.double(), bins), torch.bincount(b, corr.double(), bins)
    return ((c - a).abs().sum() / n.sum()).item()  # sum_b n_b/N |conf_b - acc_b| with the per-bin means cancelling n_b


def token_stats(head, h: torch.Tensor, ids: torch.Tensor) -> dict:
    """Per-token nll, entropy, max prob, top-1 correct for one window; log-softmax over CHUNK tokens at a time."""
    out = {k: [] for k in ("nll", "ent", "maxp", "corr")}
    for i in range(0, h.shape[0] - 1, CHUNK):
        lp = F.log_softmax(head(h[i:min(i + CHUNK, h.shape[0] - 1)]).float(), dim=-1)
        y = ids[1 + i:1 + i + lp.shape[0]]
        out["nll"].append(-lp.gather(1, y[:, None]).squeeze(1).double())
        out["ent"].append(-(lp.exp() * lp).sum(1).double())
        mx, am = lp.max(1)
        out["maxp"].append(mx.exp().double())
        out["corr"].append(am == y)
        del lp
    return {k: torch.cat(v).cpu() for k, v in out.items()}


def read_est(path: str, fields) -> dict:
    d = {}
    for r in csv.DictReader(open(path)):
        d.setdefault(f"{r['layer']}.{r['module']}", []).append([float(r[f]) for f in fields])
    return {k: torch.tensor(v, dtype=torch.float64).T for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-data", default="wikitext")
    ap.add_argument("--eval-windows", type=int, default=40)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--save", required=True)
    ap.add_argument("--check-csv", default="")
    args = ap.parse_args()
    args.cat_k = None
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    targets = find_targets(model, PROJECTIONS)
    sdpa_attention.use_gqa_in_sdpa = lambda *a, **k: False  # as run_mackin main: fp32 SDPA on the memory-efficient kernel
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        scales = get_cat(args, tok, model, targets)
        test = load_text(tok, args.eval_data, False, args.seq_len, args.eval_windows).cuda()
        dec, head = model.get_decoder(), model.get_output_embeddings()
        orig = {n: m.weight.data.clone() for _, n, m in targets}
        key = {n: f"{l}.{n.split('.')[-1]}" for l, n, _ in targets}
        rot = read_est(os.path.join(args.out_dir, "est_rot4.csv"), ("qf",))
        rob = read_est(os.path.join(args.out_dir, "est_grad_rot4.csv"), ("s_rob",))
        kof = lambda m: max(1, round(FRAC * m.out_features))
        sels = {"curv_rot": {n: (rot[key[n]][0] / 2).topk(kof(m)).indices.cuda() for _, n, m in targets}}
        allsc = torch.cat([rob[key[n]][0] for _, n, _ in targets])  # global ranking, as run_eval's *_global selectors
        top, o, g = set(allsc.topk(sum(kof(m) for _, _, m in targets)).indices.tolist()), 0, {}
        for _, n, m in targets:
            g[n] = torch.tensor(sorted(i - o for i in top if o <= i < o + m.out_features), dtype=torch.long).cuda()
            o += m.out_features
        sels["s_rob_global"] = g
        sels["random0"] = {n: torch.randperm(m.out_features, generator=torch.Generator().manual_seed(mi))[:kof(m)].cuda()
                           for mi, (_, n, m) in enumerate(targets)}  # seed 0 as run_eval's random0
        restore = {}
        def hook(n):
            def h(mod, inp, out):
                if n in restore:
                    idx, rows = restore[n]
                    out[..., idx] = F.linear(inp[0], rows, None if mod.bias is None else mod.bias[idx]).to(out.dtype)
                return out
            return h
        for _, n, m in targets:
            m.register_forward_hook(hook(n))
        def configure(quant, sel=None):
            restore.clear()
            for l, n, m in targets:
                W = orig[n]
                m.weight.data.copy_(eff_weight(W, scales[n], l, 4, args.group, True, block=args.group) if quant else W)
                if sel is not None:
                    restore[n] = (sel[n], W[sel[n]])
        configs = [("fp", "-", False, None), ("al_H", "-", True, None), ("al_H", "s_rob_global", True, sels["s_rob_global"]),
                   ("al_H", "curv_rot", True, sels["curv_rot"]), ("al_H", "random0", True, sels["random0"])]
        rows, fp_ok, per_ppl = [], None, {}
        for config, selector, quant, sel in configs:
            configure(quant, sel)
            st = [token_stats(head, dec(input_ids=test[i:i + 1], use_cache=False).last_hidden_state[0], test[i]) for i in range(test.shape[0])]
            st = {k: torch.stack([s[k] for s in st]) for k in st[0]}  # [windows, T-1]
            if fp_ok is None:
                fp_ok = st["corr"]
            ok, nll = fp_ok, st["nll"]
            per_ppl[config, selector] = math.exp(nll.mean().item())  # all windows: run with the same --eval-windows as the --check-csv eval.csv
            rows.append({"config": config, "selector": selector, "ppl": math.exp(nll.mean().item()), "entropy": st["ent"].mean().item(),
                         "maxprob": st["maxp"].mean().item(), "ece": ece(st["maxp"].flatten(), st["corr"].flatten()),
                         "nll_fp_correct": nll[ok].mean().item(), "nll_fp_wrong": nll[~ok].mean().item(), "n_tok": nll.numel()})
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        write_csv(args.save, rows)
        if args.check_csv:
            ref = list(csv.DictReader(open(args.check_csv)))
            for sel_name in ("-", "s_rob_global"):
                r = next(r for r in ref if r["config"] == "al_H" and r["selector"] == sel_name and r["how"] in ("-", "restore") and r["k"] in ("0", "f0.03125"))
                got = per_ppl["al_H", sel_name]
                assert abs(got - float(r["ppl"])) / float(r["ppl"]) < 1e-3, f"self-check {sel_name}: {got} vs {r['ppl']}"
                print(f"self-check ok {sel_name}: {got:.4f} vs {float(r['ppl']):.4f}", flush=True)


if __name__ == "__main__":
    main()
