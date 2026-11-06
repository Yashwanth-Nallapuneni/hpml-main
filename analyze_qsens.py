"""Compare run_qsens.py estimators with its deterministic ground truth. Usage: python analyze_qsens.py DIR [DIR ...]

Ground truth per test: d_A, d_B on two independent text windows. With a kind and its mirror (int4, int4n = -error):
  sym  = (d(+e) + d(-e)) / 2  -> curvature part, first-order term cancelled exactly (the stable "sensitivity")
  anti = (d(+e) - d(-e)) / 2  -> first-order part (text-specific)
Estimators come from the calibration set C (held-out text) unless marked in-sample. Spearman throughout.
"""
import glob
import sys

import numpy as np
import pandas as pd


def sp(a, b) -> float:
    return pd.Series(np.asarray(a, float)).rank().corr(pd.Series(np.asarray(b, float)).rank())


def score(pred, y, mod) -> tuple:
    df = pd.DataFrame({"p": np.asarray(pred, float), "y": np.asarray(y, float), "m": np.asarray(mod)})
    within = [sp(g.p, g.y) for _, g in df.groupby("m") if len(g) >= 4]
    k = max(1, len(df) // 10)
    top = len(set(df.p.nlargest(k).index) & set(df.y.nlargest(k).index)) / k
    return sp(df.p, df.y), float(np.nanmedian(within)) if within else float("nan"), top


def predictors(e: pd.DataFrame) -> dict:
    return {
        "en (output error energy)": e.en,
        "enw (weight error energy)": e.enw,
        "fish x en": e.fish * e.en,
        "fd/2 (token-diag Fisher)": e.fd / 2,
        "fde/2 (token-diag emp. Fisher)": e.fde / 2,
        "wf/2 (diag weight Fisher)": e.wf / 2,
        "qf/2 (GGN quadratic form)": e.qf / 2,
        "g1 (first order)": e.g1,
        "g1 + qf/2 (2nd-order Taylor)": e.g1 + e.qf / 2,
    }


def row(name, r):
    print(f"  {name:40s} {r[0]:7.3f} {r[1]:7.3f} {r[2]:7.3f}")


def report(d: str) -> None:
    gt = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{d}/gt*.csv"))]).drop_duplicates(["layer", "module", "kind", "channel"])
    gt["key"] = gt.layer.astype(str) + "." + gt.module
    on = ["layer", "module", "channel"]
    print(f"\n######## {d}  ({gt.key.nunique()} modules with ground truth)")
    for kind in [k for k in ("int4", "rot") if k in set(gt.kind)]:
        est = {s: pd.read_csv(f"{d}/est_{s}_{kind}.csv") for s in "ABC"}
        g = gt[gt.kind == kind].merge(gt[gt.kind == kind + "n"], on=on + ["key"], suffixes=("", "_n"), how="left")
        for s in "AB":
            g[f"sym_{s}"] = (g[f"d_{s}"] + g[f"d_{s}_n"]) / 2
            g[f"anti_{s}"] = (g[f"d_{s}"] - g[f"d_{s}_n"]) / 2
        g["d_AB"], g["sym_AB"] = (g.d_A + g.d_B) / 2, (g.sym_A + g.sym_B) / 2
        ch, md = g[g.channel >= 0], g[g.channel < 0]
        j = {s: ch.merge(e, on=on, how="left") for s, e in est.items()}
        assert j["C"].en.notna().all(), "estimator rows missing for some ground-truth channels"
        has_sym = ch.sym_A.notna()
        print(f"\n== {kind} channels: {len(ch)} one-sided, {has_sym.sum()} with mirror")
        print(f"  replication A vs B (pooled / within-module): one-sided {sp(ch.d_A, ch.d_B):.3f} / {score(ch.d_A, ch.d_B, ch.key)[1]:.3f}"
              + (f" | sym {sp(ch.sym_A[has_sym], ch.sym_B[has_sym]):.3f} / {score(ch.sym_A[has_sym], ch.sym_B[has_sym], ch.key[has_sym])[1]:.3f}"
                 f" | share sym>0 {(ch.sym_AB[has_sym] > 0).mean():.2f}"
                 f" | median |anti|/|sym| {(ch.anti_A.abs() / ch.sym_A.abs())[has_sym].median():.1f}" if has_sym.any() else ""))
        print(f"  {'predictor (calibration text C unless noted)':40s} {'pooled':>7s} {'within':>7s} {'top10%':>7s}")
        targets = [("one-sided d_AB", "d_AB", slice(None))] + ([("sym_AB (curvature)", "sym_AB", has_sym.values)] if has_sym.any() else [])
        for tname, t, m in targets:
            print(f" target {tname}")
            jc = j["C"][m]
            for name, p in predictors(jc).items():
                row(name, score(p, jc[t], jc.key))
        if has_sym.any():
            r = sp(ch.sym_A[has_sym], ch.sym_B[has_sym])
            jc = j["C"][has_sym.values]
            print(f"  ceiling for sym_AB (Spearman-Brown of A vs B): {2 * r / (1 + r):.3f}")
            row("fish (no error used) vs sym_AB / en", score(jc.fish, jc.sym_AB / jc.en, jc.key))
            row("qf/en vs sym_AB / en", score(jc.qf / jc.en, jc.sym_AB / jc.en, jc.key))
        for s in "AB":
            js = j[s]
            row(f"g1 in-sample vs one-sided d_{s}", score(js.g1, js[f"d_{s}"], js.key))
            if has_sym.any():
                jm = js[has_sym.values]
                row(f"g1 in-sample vs anti_{s}", score(jm.g1, jm[f"anti_{s}"], jm.key))
                row(f"qf/2 in-sample vs sym_{s}", score(jm.qf / 2, jm[f"sym_{s}"], jm.key))
        if len(md):
            em = md.merge(pd.read_csv(f"{d}/estmod_C_{kind}.csv"), on=["layer", "module"])
            print(f"-- {kind} whole modules ({len(em)}): mean one-sided d_AB {em.d_AB.mean():.2e}; "
                  f"replication A vs B {sp(em.d_A, em.d_B):.3f}" + (f", sym {sp(em.sym_A, em.sym_B):.3f}" if em.sym_A.notna().all() else ""))
            for name, p in {"en": em.en, "qf/2 full": em.qf / 2, "qf_diag/2 (channels independent)": em.qf_diag / 2}.items():
                print(f"   {name:34s} vs one-sided d_AB {sp(p, em.d_AB):.3f}" + (f" | vs sym_AB {sp(p, em.sym_AB):.3f}" if em.sym_AB.notna().all() else ""))
            print(f"   cross-channel share of qf: median (qf - qf_diag)/qf = {((em.qf - em.qf_diag) / em.qf).median():.2f}")
            if em.sym_AB.notna().all():
                print(f"   calibration: median (qf/2)/sym_AB = {(em.qf / 2 / em.sym_AB).median():.2f}")
    mods = gt[gt.channel < 0].pivot_table(index="key", columns="kind", values=["d_A", "d_B"])
    if {"int4", "rot"} <= set(gt.kind):
        e = {k: pd.read_csv(f"{d}/estmod_C_{k}.csv").assign(key=lambda x: x.layer.astype(str) + "." + x.module).set_index("key")
             for k in ("int4", "rot")}
        harm = ((mods["d_A"].rot + mods["d_B"].rot) - (mods["d_A"].int4 + mods["d_B"].int4)).dropna() / 2
        if {"int4n", "rotn"} <= set(gt.kind):
            symm = {k: (mods["d_A"][k] + mods["d_B"][k] + mods["d_A"][k + "n"] + mods["d_B"][k + "n"]) / 4 for k in ("int4", "rot")}
            hs = (symm["rot"] - symm["int4"]).dropna()
            print(f"\n-- rotation harm, curvature part (sym, {len(hs)} modules): rotation hurts in {(hs > 0).mean():.2f}")
            for k in ("int4", "rot"):
                ps = (e[k].qf / 2).reindex(symm[k].dropna().index)
                print(f"   {k} module sym vs qf/2: Spearman {sp(ps, symm[k].dropna()):.3f}, median ratio qf/2 / sym {(ps / symm[k].dropna()).median():.2f}")
            for name, f in {"qf/2": lambda x: x.qf / 2, "en": lambda x: x.en}.items():
                ph = (f(e["rot"]) - f(e["int4"])).reindex(hs.index)
                print(f"   harm predicted by {name:5s}: Spearman {sp(ph, hs):.3f}, sign agreement {(np.sign(ph) == np.sign(hs)).mean():.2f}")
        print(f"\n-- rotation harm per module (rot - int4, one-sided, {len(harm)} modules): rotation hurts in {(harm > 0).mean():.2f}; "
              f"total output error falls with rotation in {(e['rot'].en.reindex(harm.index) < e['int4'].en.reindex(harm.index)).mean():.2f}")
        for name, f in {"qf/2": lambda x: x.qf / 2, "en": lambda x: x.en, "g1 + qf/2": lambda x: x.g1 + x.qf / 2}.items():
            ph = (f(e["rot"]) - f(e["int4"])).reindex(harm.index)
            print(f"   predicted by {name:10s}: Spearman {sp(ph, harm):.3f}, sign agreement {(np.sign(ph) == np.sign(harm)).mean():.2f}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        report(d)
