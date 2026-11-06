"""Compare cheap estimators (run_estimators.py) with the run_sensitivity.py ground truth, by Spearman and top-10% overlap.
Usage: python compare_estimators.py GROUND_TRUTH_DIR ESTIMATORS_CSV"""
import glob
import os
import re
import sys

import pandas as pd

KEYS = ["layer", "module", "channel"]
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load(d):
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(d, "layer*.csv")))], ignore_index=True)
    noise = [c for c in df.columns if re.fullmatch(r"d_noise\d+", c)]  # pair averages, not the raw +/- columns
    df["d_noise"] = df[noise].mean(axis=1)
    df["s_noise"] = df["d_noise"] / df["sigma"] ** 2
    return df, noise


def sp(a, b):
    return a.rank().corr(b.rank())  # Spearman without scipy (not installed on the VM)


def overlap(truth, est, frac=0.1):
    k = max(1, round(frac * len(truth)))
    return len(set(truth.nlargest(k).index) & set(est.nlargest(k).index)) / k


def med_module(df, est, tgt):
    return pd.Series([sp(g[est], g[tgt]) for _, g in df.groupby(["layer", "module"])]).median()


def main() -> None:
    gt, noise = load(sys.argv[1])
    est = pd.read_csv(sys.argv[2])
    gt = gt.drop(columns=["ch_rms"])  # the estimator file's ch_rms is the one under test
    df = gt.merge(est, on=KEYS, how="inner")
    print(f"joined {len(df)} channels (ground truth {len(gt)}, estimators {len(est)}), {len(df.groupby(['layer', 'module']))} modules")
    ceil = sum(sp(gt[a], gt[b]) for i, a in enumerate(noise) for b in noise[i + 1:]) / (len(noise) * (len(noise) - 1) / 2)
    print(f"ceiling: noise draw-to-draw Spearman, mean pairwise, pooled {ceil:.3f}")

    ests = ["fisher_emp", "fisher_true", "ch_rms"]
    print("\nvs s_noise (Spearman; overlap = share of true top 10% also in the estimator's top 10%)")
    print(f"{'estimator':<13}{'pooled':>8}{'med/module':>12}{'top10%':>8}")
    for e in ests:
        print(f"{e:<13}{sp(df[e], df.s_noise):>8.3f}{med_module(df, e, 's_noise'):>12.3f}{overlap(df.s_noise, df[e]):>8.3f}")
    print("\nper module type, Spearman vs s_noise (pooled over that type's channels)")
    print(f"{'module':<11}{'n':>6}" + "".join(f"{e:>13}" for e in ests))
    for m in MODULES:
        s = df[df.module == m]
        if len(s):
            print(f"{m:<11}{len(s):>6}" + "".join(f"{sp(s[e], s.s_noise):>13.3f}" for e in ests))

    df["neg_gxa"] = -df["gxa"]
    print("\nvs d_zero")
    print(f"{'estimator':<13}{'pooled':>8}{'top10%':>8}")
    for e in ["gxa_abs", "neg_gxa"]:
        print(f"{e:<13}{sp(df[e], df.d_zero):>8.3f}{overlap(df.d_zero, df[e]):>8.3f}")


if __name__ == "__main__":
    main()
