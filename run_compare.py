"""Compare fp / int4 / int4_rot on perplexity and per-layer numerical error."""
import argparse
import csv
import gc
import json
import os
import time
from datetime import datetime

import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_wikitext2
from src.evaluate import evaluate_lm
from src.wrapper import QuantConfig, wrap_model

CSV_FIELDS = ["model", "config", "seed", "group_size", "block", "a_bits", "out_bits",
              "ppl", "next_token_acc", "mean_rel_err", "mean_snr_db", "time_s"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", default="fp,int4,int4_rot")
    ap.add_argument("--group-size", type=int, default=128, help="0 = per-channel")
    ap.add_argument("--block", default="128", help="int or 'full'")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--a-bits", type=int, default=None)
    ap.add_argument("--out-bits", type=int, default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--layers", default=None, help="comma list of layer indices")
    ap.add_argument("--out-dir", default="/workspace/results" if os.path.isdir("/workspace/results") else "./results")
    ap.add_argument("--no-collect-stats", dest="collect_stats", action="store_false")
    return ap.parse_args()


def run_one(args, config: str, seed: int, windows, out_root: str) -> dict:
    block = "full" if args.block == "full" else int(args.block)
    layers = [int(i) for i in args.layers.split(",")] if args.layers else None
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cuda")
    wrapped = {}
    if config != "fp":
        rot = config == "int4_rot"
        cfg = QuantConfig(w_bits=4, group_size=args.group_size or None, a_bits=args.a_bits,
                          out_bits=args.out_bits, rotate_in=rot, rotate_out=rot, block_in=block,
                          block_out=block, seed=seed, collect_stats=args.collect_stats)
        wrapped = wrap_model(model, cfg, layers=layers)
    print(f"== {config} seed={seed}", flush=True)
    metrics = evaluate_lm(model, windows)
    ppl, acc = metrics["ppl"], metrics["next_token_acc"]
    layer_stats = {n: w.stats() for n, w in wrapped.items()} if args.collect_stats else {}
    elapsed, peak = time.time() - t0, torch.cuda.max_memory_allocated() / 2**30
    del model, wrapped
    gc.collect()
    torch.cuda.empty_cache()

    rels = [s["rel_err"] for s in layer_stats.values() if s]
    snrs = [s["snr_db"] for s in layer_stats.values() if s]
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    short = args.model.split("/")[-1]
    out_dir = os.path.join(out_root, short)
    os.makedirs(out_dir, exist_ok=True)
    record = {"args": vars(args), "model": args.model, "config": config, "seed": seed,
              "perplexity": ppl, "next_token_acc": acc, "wall_time_s": elapsed, "peak_gpu_mem_gib": peak,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "layers": layer_stats}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(out_dir, f"{config}_seed{seed}_{stamp}.json"), "w") as f:
        json.dump(record, f)
    row = {"model": short, "config": config, "seed": seed, "group_size": args.group_size,
           "block": args.block, "a_bits": args.a_bits, "out_bits": args.out_bits, "ppl": ppl, "next_token_acc": acc,
           "mean_rel_err": mean(rels), "mean_snr_db": mean(snrs), "time_s": elapsed}
    csv_path = os.path.join(out_dir, "summary.csv")
    new = not os.path.exists(csv_path)
    if not new:
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
        if header != CSV_FIELDS:
            raise SystemExit(f"{csv_path} has an older column layout; use a fresh --out-dir")
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def main() -> None:
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    windows = load_wikitext2(tok, seq_len=args.seq_len, max_windows=args.max_windows)
    print(f"{windows.shape[0]} windows of {args.seq_len} tokens")
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for config in args.configs.split(","):
        for seed in (seeds if config == "int4_rot" else [0]):
            rows.append(run_one(args, config, seed, windows, args.out_dir))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
