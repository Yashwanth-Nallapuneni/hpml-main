#!/usr/bin/env bash
# Half-size error, whole modules: is the curvature part purely quadratic (falls to 1/4)? Qwen3 and Llama in parallel.
set -u
source /workspace/venv/bin/activate
cd "$(dirname "$0")"
export HF_HUB_CACHE=/workspace/hf_cache/hub HF_DATASETS_CACHE=/workspace/hf_cache/datasets
R=/workspace/results_qsens
for x in "Qwen/Qwen3-0.6B qwen3" "meta-llama/Llama-3.2-1B llama"; do
  set -- $x
  python run_qsens.py --model $1 --group 128 --out-dir $R/$2 --mode gt --kinds int4h --chan-kinds int4h \
    --per-module 0 --batch 2 --gt-file gt_half.csv > $R/logs/$2_half.log 2>&1 &
done
wait
echo "$(date -u) HALF DONE"
