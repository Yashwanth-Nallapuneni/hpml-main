"""Analyse run_mackin.py outputs. Usage: python analyze_mackin.py DIR [DIR ...]

1. Mackin's ground truth (one-sided, real pre-Hadamard INT4 error, fp model): does it replicate (half 1 vs half 2)?
2. Cheap estimators from calibration (train) text vs that ground truth.
3. Proof of concept: perplexity when restoring / isolating the top-k channels per module vs k random channels.
4. Two quantization types: do 4-bit and 8-bit pick the same channels?
"""
import os
import sys

import numpy as np
import pandas as pd


def sp(a, b) -> float:
    return pd.Series(np.asarray(a, float)).rank().corr(pd.Series(np.asarray(b, float)).rank())


def score(p, y, m) -> tuple:
    df = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float), "m": np.asarray(m)})
    w = [sp(g.p, g.y) for _, g in df.groupby("m") if len(g) >= 4]
    k = max(1, len(df) // 10)
    return sp(df.p, df.y), float(np.nanmedian(w)) if w else float("nan"), len(set(df.p.nlargest(k).index) & set(df.y.nlargest(k).index)) / k


def report(d: str) -> None:
    print(f"\n######## {d}")
    on = ["layer", "module", "channel"]
    est = {k: pd.read_csv(f"{d}/est_{k}.csv") for k in ("pre4", "rot4", "pre8") if os.path.exists(f"{d}/est_{k}.csv")}
    if os.path.exists(f"{d}/gt.csv") and "pre4" in est:
        gt = pd.read_csv(f"{d}/gt.csv").merge(est["pre4"], on=on)
        gt["key"] = gt.layer.astype(str) + "." + gt.module
        r = sp(gt.d_h1, gt.d_h2)
        print(f"\n== Mackin ground truth: {len(gt)} channels, {gt.key.nunique()} modules; share dloss>0 {(gt.d_all > 0).mean():.2f}")
        print(f"   replication half 1 vs half 2: pooled {r:.3f}, within-module {score(gt.d_h1, gt.d_h2, gt.key)[1]:.3f}; "
              f"ceiling for the full set (Spearman-Brown) {2 * r / (1 + r):.3f}")
        print(f"   {'estimator (train text)':32s} {'pooled':>7s} {'within':>7s} {'top10%':>7s}")
        for name, p in {"g1 + qf/2 (2nd-order Taylor)": gt.g1 + gt.qf / 2, "g1 (first order)": gt.g1, "qf/2 (curvature)": gt.qf / 2,
                        "fish x en": gt.fish * gt.en, "fish (no error)": gt.fish, "en (error energy)": gt.en, "ey (channel energy)": gt.ey}.items():
            s = score(p, gt.d_all, gt.key)
            print(f"   {name:32s} {s[0]:7.3f} {s[1]:7.3f} {s[2]:7.3f}")
    if os.path.exists(f"{d}/eval.csv"):
        ev = pd.read_csv(f"{d}/eval.csv", dtype={"k": str})
        b0 = ev[ev.k == "0"].set_index("config")
        base = b0.ppl
        bpw = (lambda c: f" [{b0.bpw[c]:.3f} bpw]") if "bpw" in ev else (lambda c: "")
        data = f"{ev.data.iloc[0]}, {ev.windows.iloc[0]} windows, rot seed {ev.rot_seed.iloc[0]}" if "data" in ev else "test"
        print(f"\n== Perplexity ({data}): " + ", ".join(f"{c} {p:.3f}{bpw(c)}" for c, p in base.items()))
        alH, fp = base["al_H"], base["fp"]
        for (k, how), g in ev[ev.k != "0"].groupby(["k", "how"]):
            rnd = g[g.selector.str.startswith("random")].ppl
            print(f"-- k={k} per module, {how} ({int(g.n_restored.iloc[0])} channels"
                  + (f", {g[g.selector.str.startswith('random')].bpw.iloc[0]:.3f} bpw for random" if "bpw" in g else "") + f"): random {rnd.mean():.3f} ± {rnd.std():.3f} "
                  f"(n={len(rnd)}, min {rnd.min():.3f}); gain from random {alH - rnd.mean():+.3f}")
            for _, r in g[~g.selector.str.startswith("random")].sort_values("ppl").iterrows():
                z = (rnd.mean() - r.ppl) / rnd.std() if rnd.std() > 0 else float("nan")
                print(f"   {r.selector:15s} ppl {r.ppl:.3f}" + (f" [{r.bpw:.3f} bpw]" if "bpw" in r else "") + f"  gain {alH - r.ppl:+.3f} = " + (f"{(alH - r.ppl) / (alH - rnd.mean()):5.1f}x random, " if alH - rnd.mean() > 1e-3 else "  n/a x random, ") +
                      f"{z:5.1f} sd better than random, closes {(alH - r.ppl) / (alH - fp):.1%} of the gap to fp")
    if {"pre4", "pre8"} <= set(est):
        a, b = est["pre4"], est["pre8"]
        key = a.layer.astype(str) + "." + a.module
        print("\n== 4-bit vs 8-bit (same channels?)")
        for name, f in {"g1 + qf/2": lambda e: e.g1 + e.qf / 2, "qf/2": lambda e: e.qf / 2}.items():
            pa, pb = f(a), f(b)
            df = pd.DataFrame({"a": pa, "b": pb, "m": key})
            within = df.groupby("m").apply(lambda g: sp(g.a, g.b)).median()
            top = df.groupby("m").apply(lambda g: len(set(g.a.nlargest(2).index) & set(g.b.nlargest(2).index)) / 2).mean()
            print(f"   {name:10s}: within-module Spearman median {within:.3f}; top-2 per module overlap {top:.2f}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        report(d)
