"""Upload the result files in results/ to a W&B project, so there is one dashboard for all experiments.

One W&B run per result folder (any folder with CSV files): every CSV under 2 MB becomes a table, *_meta.json files become the
run config, and headline numbers become summary metrics (perplexity per config from eval.csv; Spearman and noise ceiling per
error kind from analysis.txt). The rotated ground truth also gets a score-vs-measured scatter per model and kind.
Nothing is recomputed: it only reads the files in results/.

Usage, from the repo root (needs `pip install wandb` and `wandb login` once):
    python analysis/log_wandb.py                 # all of results/
    python analysis/log_wandb.py results/uddhav  # one subtree
Options (env): WANDB_PROJECT (default hpml-main), WANDB_ENTITY (default: your W&B user).
"""
import glob
import json
import os
import re
import sys

import pandas as pd
import wandb

PROJECT = os.environ.get("WANDB_PROJECT", "hpml-main")
MAX_BYTES = 2_000_000
ROW = re.compile(r"^\s*g1 \+ qf/2 \(2nd-order Taylor\)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)")
HEAD = re.compile(r"== Mackin ground truth \[(\w+)\]")
CEIL = re.compile(r"replication half 1 vs half 2: pooled (-?[\d.]+).*Spearman-Brown\) (-?[\d.]+)")


def parse_analysis(path: str) -> dict:
    """Summary metrics from analyze_mackin.py output: score vs ground truth and its noise ceiling, per error kind."""
    out, kind = {}, None
    for line in open(path):
        if m := HEAD.search(line):
            kind = m.group(1)
        elif kind and (m := CEIL.search(line)):
            out[f"gt/{kind}/replication"], out[f"gt/{kind}/ceiling"] = float(m.group(1)), float(m.group(2))
        elif kind and (m := ROW.search(line)):
            out[f"gt/{kind}/spearman_pooled"], out[f"gt/{kind}/spearman_within"] = float(m.group(1)), float(m.group(2))
            out[f"gt/{kind}/top10_overlap"] = float(m.group(3))
    return out


def log_folder(d: str, root: str) -> None:
    rel = os.path.relpath(d, root)
    parts = rel.split(os.sep)
    config = {"folder": rel}
    for meta in sorted(glob.glob(f"{d}/*meta*.json")):
        try:
            config[os.path.basename(meta)[:-5]] = json.load(open(meta))
        except json.JSONDecodeError:
            pass
    run = wandb.init(project=PROJECT, name=rel, group="/".join(parts[:2]), job_type=parts[1] if len(parts) > 1 else parts[0],
                     config=config, reinit="finish_previous")
    summary = {}
    for f in sorted(glob.glob(f"{d}/*.csv")):
        if os.path.getsize(f) > MAX_BYTES:
            continue
        df = pd.read_csv(f)
        run.log({f"table/{os.path.basename(f)[:-4]}": wandb.Table(dataframe=df)})
        if os.path.basename(f) == "eval.csv" and {"k", "config", "ppl"} <= set(df):
            for _, r in df[df.k.astype(str) == "0"].iterrows():
                summary[f"ppl/{r.config}"] = float(r.ppl)
        if os.path.basename(f) == "gt_vs_est.csv":
            for (model, kind), g in df.groupby(["model", "kind"]):
                t = wandb.Table(dataframe=g[["rank_est", "rank_gt"]])
                rho = g.rank_est.corr(g.rank_gt)
                run.log({f"scatter/{model}_{kind}": wandb.plot.scatter(t, "rank_est", "rank_gt",
                                                                     title=f"{model} {kind}: score rank vs measured rank (rho {rho:.3f})")})
    if os.path.exists(f"{d}/analysis.txt"):
        summary.update(parse_analysis(f"{d}/analysis.txt"))
        run.save(f"{d}/analysis.txt", base_path=d, policy="now")
    run.summary.update(summary)
    run.finish()


if __name__ == "__main__":
    root = "results"
    start = sys.argv[1] if len(sys.argv) > 1 else root
    folders = sorted({os.path.dirname(f) for f in glob.glob(f"{start}/**/*.csv", recursive=True)})
    assert folders, f"no CSV files under {start}"
    for i, d in enumerate(folders, 1):
        print(f"[{i}/{len(folders)}] {d}")
        log_folder(d, root)
