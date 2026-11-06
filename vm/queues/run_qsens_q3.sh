#!/usr/bin/env bash
# After the Qwen3 ground truth (pid $1) exits: Qwen3 and Llama estimators. Retries once if the GPU was full.
set -u
source /workspace/venv/bin/activate
cd "$(dirname "$0")"
export HF_HUB_CACHE=/workspace/hf_cache/hub HF_DATASETS_CACHE=/workspace/hf_cache/datasets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/results_qsens
while kill -0 $1 2>/dev/null; do sleep 20; done
for m in "Qwen/Qwen3-0.6B qwen3" "meta-llama/Llama-3.2-1B llama"; do
  set -- $m
  python run_qsens.py --model $1 --group 128 --out-dir $R/$2 --mode est > $R/logs/$2_est.log 2>&1 && continue
  while pgrep -f "mode gt" > /dev/null; do sleep 30; done
  python run_qsens.py --model $1 --group 128 --out-dir $R/$2 --mode est > $R/logs/$2_est.log 2>&1
done
echo "$(date -u) Q3 DONE"
