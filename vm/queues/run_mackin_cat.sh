#!/usr/bin/env bash
# Block-CAT (arXiv 2603.04359) version of the Mackin runs. Two queues; a step that fails (e.g. GPU memory while
# sharing) retries once when no other exp_2_cheap_est job is running.
set -u
source /workspace/venv/bin/activate
cd "$(dirname "$0")"
export HF_HUB_CACHE=/workspace/hf_cache/hub HF_DATASETS_CACHE=/workspace/hf_cache/datasets PYTORCH_ALLOC_CONF=expandable_segments:True
R=/workspace/results_mackin_cat
mkdir -p $R/logs
run() {  # run LOG ARGS...
  local log=$1; shift
  python exp_2_cheap_est.py --align cat "$@" > $log 2>&1 && return
  sleep 5
  while [ $(pgrep -fc "[p]ython exp_2_cheap_est.py") -gt 0 ]; do sleep 30; done
  echo "$(date -u) retry $log" >> $R/logs/retries.log
  python exp_2_cheap_est.py --align cat "$@" > $log 2>&1
}
S="--model HuggingFaceTB/SmolLM2-135M --group 64 --cat-k 192 --out-dir $R/smollm2"
Q="--model Qwen/Qwen3-0.6B --group 128 --cat-k 128 --out-dir $R/qwen3"
L="--model meta-llama/Llama-3.2-1B --group 128 --cat-k 128 --out-dir $R/llama"
G="--model google/gemma-3-1b-pt --group 128 --cat-k 128 --out-dir $R/gemma"
EV="--mode eval --eval-windows 40 --random-seeds 10"
(
  run $R/logs/smollm2_est.log $S --mode est --calib 16
  run $R/logs/smollm2_eval.log $S $EV --k 2,8
  run $R/logs/qwen3_est.log $Q --mode est --calib 8
  run $R/logs/qwen3_eval.log $Q $EV --k 2
  run $R/logs/smollm2_gt.log $S --mode gt --gt-windows 16 --per-module 4 --batch 4 --max-minutes 40
) &
(
  until grep -q "GT DONE" /workspace/results_mackin/logs/smollm2_gt.log 2>/dev/null; do sleep 20; done
  run $R/logs/llama_est.log $L --mode est --calib 8
  run $R/logs/llama_eval.log $L $EV --k 2
  run $R/logs/gemma_est.log $G --mode est --calib 8 --seq-len 1024
  run $R/logs/gemma_eval.log $G $EV --k 2
) &
wait
echo "$(date -u) CAT DONE"
