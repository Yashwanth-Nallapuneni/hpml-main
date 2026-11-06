"""Summarize run_sensitivity.py output. Optional second dir: compare the two runs on shared channels.
Usage: python summarize_sensitivity.py DIR [DIR2]"""
import glob
import json
import os
import re
import sys

import pandas as pd


def load(d):
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(d, "layer*.csv")))], ignore_index=True)
    noise = [c for c in df.columns if re.fullmatch(r"d_noise\d+", c)]  # pair averages, not the raw +/- columns
    df["d_noise"] = df[noise].mean(axis=1)
    df["s_noise"] = df["d_noise"] / df["sigma"] ** 2
    return df, noise, json.load(open(os.path.join(d, "meta.json")))


def sp(a, b):
    return a.rank().corr(b.rank())  # Spearman without scipy (not installed on the VM)


def main() -> None:
    df, noise, meta = load(sys.argv[1])
    print(f"{sys.argv[1]}: {len(df)} channels, {df.layer.nunique()} layers, clean loss {meta['clean_loss']:.5f}")
    print(f"max |dloss| on untouched rows: {max(meta['noop_max_abs'].values()):.2e}  (should be ~0)")
    print("Spearman between noise draws:", ", ".join(
        f"{a[-1]}-{b[-1]} {sp(df[a], df[b]):.3f}" for i, a in enumerate(noise) for b in noise[i + 1:]))
    print(f"Spearman zero vs mean {sp(df.d_zero, df.d_mean):.3f} | zero vs noise {sp(df.d_zero, df.d_noise):.3f} | "
          f"mean vs noise {sp(df.d_mean, df.d_noise):.3f}")
    print(f"share of channels with mean dloss from noise > 0: {(df.d_noise > 0).mean():.3f}")
    print("\nmedian dloss by module:")
    print(df.groupby("module")[["d_zero", "d_mean", "d_noise", "s_noise"]].median().to_string(float_format="%.3e"))
    cols = ["layer", "module", "channel", "ch_rms", "d_zero", "d_mean", "d_noise", "s_noise"]
    for col in ["d_zero", "d_mean", "s_noise"]:
        print(f"\ntop 10 by {col}:")
        print(df.nlargest(10, col)[cols].to_string(index=False, float_format="%.3e"))
    if len(sys.argv) > 2:
        df2 = load(sys.argv[2])[0]
        m = df.merge(df2, on=["layer", "module", "channel"], suffixes=("", "_2"))
        print(f"\nvs {sys.argv[2]} on {len(m)} shared channels, Spearman: "
              + ", ".join(f"{c} {sp(m[c], m[c + '_2']):.3f}" for c in ["d_zero", "d_mean", "d_noise", "s_noise"]))


if __name__ == "__main__":
    main()
