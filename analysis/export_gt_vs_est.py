"""One CSV: measured loss increase (ground truth) next to the cheap score (g1 + qf/2) for every tested channel and kind.

Usage: python analysis/export_gt_vs_est.py results/uddhav/gt_rot [OUT.csv]
Reads <dir>/<model>/gt.csv and est_<kind>.csv for each model folder; writes <dir>/gt_vs_est.csv by default.
"""
import os
import sys

import pandas as pd

ON = ["layer", "module", "channel"]


def export(root: str) -> pd.DataFrame:
    out = []
    for model in sorted(os.listdir(root)):
        d = os.path.join(root, model)
        if not os.path.exists(os.path.join(d, "gt.csv")):
            continue
        gt = pd.read_csv(os.path.join(d, "gt.csv"))
        for kind, g in gt.groupby("kind"):
            est = pd.read_csv(os.path.join(d, f"est_{kind}.csv"))[ON + ["g1", "qf"]]
            m = g.merge(est, on=ON, how="left", validate="one_to_one")
            assert m.g1.notna().all(), f"{model} {kind}: {m.g1.isna().sum()} tested channels have no estimator row"
            m = m.rename(columns={"d_all": "gt_dloss", "d_h1": "gt_h1", "d_h2": "gt_h2", "g1": "est_g1", "qf": "est_qf"})
            m["est"] = m.est_g1 + m.est_qf / 2  # the 2nd-order Taylor score used in analysis.txt and the handover table
            m["rank_gt"] = m.gt_dloss.rank(ascending=False).astype(int)  # 1 = most sensitive
            m["rank_est"] = m.est.rank(ascending=False).astype(int)
            m.insert(0, "model", model)
            out.append(m)
    cols = ["model", "layer", "module", "channel", "kind", "gt_dloss", "gt_h1", "gt_h2", "est", "est_g1", "est_qf", "rank_gt", "rank_est"]
    return pd.concat(out)[cols].sort_values(["model", "kind", "rank_gt"])


if __name__ == "__main__":
    root = sys.argv[1]
    df = export(root)
    path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "gt_vs_est.csv")
    df.to_csv(path, index=False)
    print(f"{len(df)} rows -> {path}")
    for (m, k), g in df.groupby(["model", "kind"]):
        print(f"  {m} {k}: {len(g)} channels, Spearman est vs gt {g.rank_est.corr(g.rank_gt):.3f}")
