#!/usr/bin/env bash
# Second queue of run_qsens_job.sh, restarted after the attention-memory fix. Waits for SmolLM2's ground truth,
# then shares the GPU with the Llama ground truth; a step that runs out of memory retries once the GPU is free.
set -u
source /workspace/venv/bin/activate
cd "$(dirname "$0")"
export HF_HUB_CACHE=/workspace/hf_cache/hub HF_DATASETS_CACHE=/workspace/hf_cache/datasets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/results_qsens
Q="--model Qwen/Qwen3-0.6B --group 128 --out-dir $R/qwen3"
L="--model meta-llama/Llama-3.2-1B --group 128 --out-dir $R/llama"
run() {  # run LOG ARGS...: on failure, wait until no other run_qsens job is running, then try once more
  local log=$1; shift
  python run_qsens.py "$@" > $log 2>&1 && return
  while pgrep -f "run_qsens.py .*--mode gt" > /dev/null; do sleep 30; done
  echo "$(date -u) retrying $log" >> $R/logs/q2.log
  python run_qsens.py "$@" > $log 2>&1
}
until grep -q "GT DONE" $R/logs/smollm2_gt.log 2>/dev/null; do sleep 30; done
run $R/logs/qwen3_est.log $Q --mode est
run $R/logs/llama_est.log $L --mode est
run $R/logs/qwen3_gt.log $Q --mode gt --per-module 8 --chan-kinds int4 --batch 10 --max-minutes 70
echo "$(date -u) Q2 DONE"
