"""Cheap per-channel sensitivity estimators for EVERY output channel of every q/k/v/o/gate/up/down projection.

One forward + a few backward passes per window on an fp32 model, with autograd taken w.r.t. each module's output:
  fisher_emp  : sum over tokens of g^2, g = d(loss with the true next token)/d(channel output)
  fisher_true : same with labels sampled from the model (--fisher-samples draws, averaged)
  gxa, gxa_abs: sum over tokens of g*a (signed, and its absolute value), a = the channel's output
  ch_rms      : sqrt(mean a^2) over all windows and tokens
Compare against run_sensitivity.py ground truth with compare_estimators.py.

Usage (on the VM):
  python run_estimators.py --model HuggingFaceTB/SmolLM2-135M --out /workspace/results_sens/estimators_smollm2.csv
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_wikitext2
from src.wrapper import PROJECTIONS
from run_sensitivity import find_targets


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--split", default="validation", help="not test: groupings chosen from these scores are judged on test")
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--fisher-samples", type=int, default=4, help="sampled-label backward passes per window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--projections", default=",".join(PROJECTIONS))
    ap.add_argument("--out", required=True, help="CSV path")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = load_wikitext2(tok, split=args.split, seq_len=args.seq_len, max_windows=args.windows).cuda()
    W, T = ids.shape
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, device_map="cuda").eval()
    targets = find_targets(model, args.projections.split(","))
    print(f"{len(targets)} modules, {sum(m.out_features for _, _, m in targets)} output channels, "
          f"{W} windows x {T} tokens, fp32", flush=True)

    outs = {}  # module name -> its output of the current forward, kept in the graph
    def store(name):
        def hook(mod, inp, out):
            outs[name] = out
        return hook
    hs = [m.register_forward_hook(store(n)) for _, n, m in targets]

    acc = {n: {k: torch.zeros(m.out_features, dtype=torch.float64, device="cuda") for k in ("fisher_emp", "fisher_true", "gxa", "sq")} for _, n, m in targets}
    names = [n for _, n, _ in targets]
    gen = torch.Generator(device="cuda").manual_seed(args.seed)

    t0 = time.time()
    clean = 0.0
    for w in range(W):
        window = ids[w:w + 1]
        outs.clear()
        # shortcut: plain lm_head, softcapping models need their own head, same as run_sensitivity.py
        logits = model(input_ids=window, use_cache=False).logits[0, :-1].float()
        L = F.cross_entropy(logits, window[0, 1:], reduction="sum")
        clean += L.item()
        xs = [outs[n] for n in names]
        g = torch.autograd.grad(L, xs, retain_graph=True)
        for n, x, gi in zip(names, xs, g):
            a = x.detach()[0].double()  # (T, C)
            gi = gi[0].double()
            acc[n]["fisher_emp"] += gi.pow(2).sum(0)
            acc[n]["gxa"] += (gi * a).sum(0)
            acc[n]["sq"] += a.pow(2).sum(0)
        del g
        probs = torch.softmax(logits.detach(), dim=-1)
        for _ in range(args.fisher_samples):
            y = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            Ls = F.cross_entropy(logits, y, reduction="sum")
            gs = torch.autograd.grad(Ls, xs, retain_graph=True)
            for n, gi in zip(names, gs):
                acc[n]["fisher_true"] += gi[0].double().pow(2).sum(0) / args.fisher_samples
            del gs
        del xs, logits, L, probs
        outs.clear()
        print(f"window {w + 1}/{W} done, {time.time() - t0:.1f} s", flush=True)
    for h in hs:
        h.remove()
    print(f"clean loss {clean / (W * (T - 1)):.5f} | wall time {time.time() - t0:.1f} s", flush=True)

    n_rows = W * T
    rows = []
    for layer, name, mod in targets:
        a = acc[name]
        cols = {k: a[k].cpu() for k in ("fisher_emp", "fisher_true", "gxa")}
        cols["gxa_abs"] = cols["gxa"].abs()  # |first-order dloss of zeroing the channel|
        rms = (a["sq"] / n_rows).sqrt().cpu()
        for c in range(mod.out_features):
            rows.append({"layer": layer, "module": name.split(".")[-1], "channel": c, "ch_rms": rms[c].item(),
                         **{k: v[c].item() for k, v in cols.items()}})

    est = ("ch_rms", "fisher_emp", "fisher_true", "gxa", "gxa_abs")
    assert all(torch.isfinite(torch.tensor([r[k] for r in rows], dtype=torch.float64)).all() for k in est), "non-finite estimator"
    assert sum(r["fisher_emp"] for r in rows) > 0, "fisher_emp sums to 0"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    os.replace(tmp, args.out)
    print(f"wrote {len(rows)} channels to {args.out}", flush=True)


if __name__ == "__main__":
    main()
