"""Zero-shot task accuracy (lm-eval-harness) for fp / int4 / int4_rot.

Same model loading and wrapping as run_compare.py; only the evaluation differs.
Tasks (standard in QuaRot/SpinQuant tables): arc_easy, piqa, hellaswag, winogrande.
Reported metric per task: acc_norm if the task provides it, else acc.

Usage (on the VM):
  python run_tasks.py --model google/gemma-3-1b-pt --configs fp,int4,int4_rot --seeds 0,1,2 \
      --out-dir /workspace/results_v3/tasks
"""
import argparse
import csv
import gc
import json
import os
import time
from datetime import datetime

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.wrapper import QuantConfig, wrap_model

TASKS = ["arc_easy", "piqa", "hellaswag", "winogrande"]
CSV_FIELDS = ["model", "config", "seed", "group_size", "block", "a_bits", "out_bits",
              *[f"{t}_acc" for t in TASKS], "avg_acc", "time_s"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", default="fp,int4,int4_rot")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--group-size", type=int, default=128, help="0 = per-channel")
    ap.add_argument("--block", default="128", help="int or 'full'")
    ap.add_argument("--a-bits", type=int, default=None)
    ap.add_argument("--out-bits", type=int, default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="examples per task (smoke tests only)")
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def pick_metric(res: dict) -> float:
    """lm-eval keys look like 'acc_norm,none'. Prefer acc_norm, fall back to acc."""
    for key in ("acc_norm,none", "acc,none"):
        if key in res:
            return float(res[key])
    raise KeyError(f"no acc/acc_norm in {sorted(res)}")


def run_one(args, config: str, seed: int, tasks, out_root: str) -> dict:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    block = "full" if args.block == "full" else int(args.block)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cuda")
    if config != "fp":
        cfg = QuantConfig(w_bits=4, group_size=args.group_size or None, a_bits=args.a_bits,
                          out_bits=args.out_bits, rotate_in=config == "int4_rot",
                          rotate_out=config == "int4_rot", block_in=block, block_out=block,
                          seed=seed, collect_stats=False)  # no error stats: saves time and memory
        wrap_model(model, cfg)
    print(f"== tasks {config} seed={seed}", flush=True)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
    out = simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, limit=args.limit,
                          log_samples=False, random_seed=0, numpy_random_seed=0, torch_random_seed=0)
    accs = {t: pick_metric(out["results"][t]) for t in tasks}
    elapsed = time.time() - t0
    del lm, model
    gc.collect()
    torch.cuda.empty_cache()

    short = args.model.split("/")[-1]
    out_dir = os.path.join(out_root, short)
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(out_dir, f"tasks_{config}_seed{seed}_{stamp}.json"), "w") as f:
        json.dump({"args": vars(args), "model": args.model, "config": config, "seed": seed,
                   "task_acc": accs, "lm_eval_results": out["results"],
                   "lm_eval_versions": out.get("versions"), "wall_time_s": elapsed,
                   "torch": torch.__version__, "transformers": transformers.__version__}, f, default=str)
    row = {"model": short, "config": config, "seed": seed, "group_size": args.group_size,
           "block": args.block, "a_bits": args.a_bits, "out_bits": args.out_bits,
           **{f"{t}_acc": accs.get(t) for t in TASKS},
           "avg_acc": sum(accs.values()) / len(accs), "time_s": elapsed}
    csv_path = os.path.join(out_dir, "summary_tasks.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print({k: round(v, 4) for k, v in accs.items()}, f"avg {row['avg_acc']:.4f}", flush=True)
    return row


def main() -> None:
    args = parse_args()
    tasks = args.tasks.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    for config in args.configs.split(","):
        for seed in (seeds if config == "int4_rot" else [0]):
            run_one(args, config, seed, tasks, args.out_dir)


if __name__ == "__main__":
    main()
