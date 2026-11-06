"""Plot per-channel / per-layer quantization error from two run JSONs (a vs b)."""
import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def reduced(la: dict, lb: dict):
    a, b = np.array(la["ch_rms_err"]), np.array(lb["ch_rms_err"])
    return a, b, int((b < a).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="JSON, e.g. int4 run")
    ap.add_argument("--b", required=True, help="JSON, e.g. int4_rot run")
    ap.add_argument("--layer", required=True, help="module name or 'all'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    A, B = (json.load(open(p))["layers"] for p in (args.a, args.b))
    names = [n for n in A if n in B and A[n] and B[n]]
    if not names:
        raise SystemExit("No common layers with stats in the two JSONs")
    label_a, label_b = (os.path.basename(p).split("_seed")[0] for p in (args.a, args.b))

    if args.layer != "all":
        a, b, k = reduced(A[args.layer], B[args.layer])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(np.clip(a, 1e-12, None), np.clip(b, 1e-12, None), s=3, alpha=0.5)
        lim = [min(a.min(), b.min()) + 1e-12, max(a.max(), b.max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(xscale="log", yscale="log", xlabel=f"{label_a} per-channel RMS error",
               ylabel=f"{label_b} per-channel RMS error",
               title=f"{args.layer}: {k} of {len(a)} channels reduced error")
        out = args.out or os.path.splitext(args.b)[0] + f"_{args.layer}.png"
    else:
        by_layer: dict = {}
        for n in names:
            idx = int(re.search(r"layers\.(\d+)\.", n).group(1))
            by_layer.setdefault(idx, []).append(n)
            a, b, k = reduced(A[n], B[n])
            print(f"{n}: {k}/{len(a)} channels reduced")
        idxs = sorted(by_layer)
        ra = [np.mean([A[n]["rel_err"] for n in by_layer[i]]) for i in idxs]
        rb = [np.mean([B[n]["rel_err"] for n in by_layer[i]]) for i in idxs]
        x = np.arange(len(idxs))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - 0.2, ra, 0.4, label=label_a)
        ax.bar(x + 0.2, rb, 0.4, label=label_b)
        ax.set_xticks(x, idxs)
        ax.set(xlabel="layer index", ylabel="mean rel_err over projections", title="Per-layer relative error")
        ax.legend()
        out = args.out or os.path.splitext(args.b)[0] + "_all.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
