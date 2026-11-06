"""Collect every summary.csv (and summary_tasks.csv) under a results folder into one table (seeds → mean ± sample sd).

Usage: python make_table.py /workspace/results_v3 [--md out.md]
Rows: model × setting (precision / bit-widths) × config. Reads CSVs only, never loads a model.
"""
import argparse
import glob
import os

import pandas as pd

ORDER = {"fp": 0, "int4": 1, "int4_rot": 2}


def setting_name(row) -> str:
    a = "16" if pd.isna(row["a_bits"]) else str(int(row["a_bits"]))
    o = "" if pd.isna(row["out_bits"]) else f"+O{int(row['out_bits'])}"
    return f"W4A{a}{o}" if row["config"] != "fp" else "fp"


def fmt(series, digits) -> str:
    if series.isna().all():
        return "n/a"
    m = series.mean()
    if len(series) == 1:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {series.std(ddof=1):.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--md", default=None, help="also write the table as Markdown")
    args = ap.parse_args()

    frames = []
    for path in glob.glob(os.path.join(args.root, "**", "summary.csv"), recursive=True):
        df = pd.read_csv(path)
        rel = os.path.relpath(os.path.dirname(os.path.dirname(path)), args.root)
        df["precision"] = "fp32" if rel.split(os.sep)[0] == "fp32" else "bf16"
        frames.append(df)
    if not frames:
        raise SystemExit(f"no summary.csv under {args.root}")
    df = pd.concat(frames, ignore_index=True)
    df["setting"] = df.apply(setting_name, axis=1)
    if "next_token_acc" not in df:
        df["next_token_acc"] = float("nan")

    rows = []
    for (model, prec, setting, config), g in df.groupby(["model", "precision", "setting", "config"]):
        rows.append({
            "model": model, "precision": prec, "setting": setting, "config": config,
            "seeds": len(g),
            "ppl": fmt(g["ppl"], 3),
            "next_token_acc_%": fmt(g["next_token_acc"] * 100, 2),
            "mean_rel_err": fmt(g["mean_rel_err"], 4),
            "mean_snr_db": fmt(g["mean_snr_db"], 2),
            "_k": (model, prec != "bf16", setting != "fp", setting, ORDER.get(config, 9)),
        })
    out = pd.DataFrame(rows)
    # task accuracy (run_tasks.py) joins on model × setting × config; bf16 only
    tframes = [pd.read_csv(p) for p in glob.glob(os.path.join(args.root, "**", "summary_tasks.csv"), recursive=True)]
    if tframes:
        t = pd.concat(tframes, ignore_index=True)
        t["setting"] = t.apply(setting_name, axis=1)
        acc_cols = [c for c in t.columns if c.endswith("_acc")]
        trows = []
        for (model, setting, config), g in t.groupby(["model", "setting", "config"]):
            trows.append({"model": model, "precision": "bf16", "setting": setting, "config": config,
                          **{c.replace("_acc", "_%"): fmt(g[c] * 100, 2) for c in acc_cols}})
        out = out.merge(pd.DataFrame(trows), on=["model", "precision", "setting", "config"], how="left")
    out = out.sort_values("_k").drop(columns="_k")
    print(out.to_string(index=False))
    if args.md:
        with open(args.md, "w") as f:
            try:
                f.write(out.to_markdown(index=False))
            except ImportError:  # to_markdown needs the optional 'tabulate' package
                f.write(out.to_string(index=False))
        print("wrote", args.md)


if __name__ == "__main__":
    main()
