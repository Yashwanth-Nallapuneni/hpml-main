#!/usr/bin/env bash
# Deterministic sensitivity study on 3 models, two queues sharing the L4. Run on the VM:
#   nohup bash run_qsens_job.sh > /workspace/results_qsens/logs/job.log 2>&1 < /dev/null &
set -u
source /workspace/venv/bin/activate
cd "$(dirname "$0")"
export HF_HUB_CACHE=/workspace/hf_cache/hub HF_DATASETS_CACHE=/workspace/hf_cache/datasets
R=/workspace/results_qsens
mkdir -p $R/logs
S="--model HuggingFaceTB/SmolLM2-135M --group 64 --out-dir $R/smollm2"
Q="--model Qwen/Qwen3-0.6B --group 128 --out-dir $R/qwen3"
L="--model meta-llama/Llama-3.2-1B --group 128 --out-dir $R/llama"
(
  python run_qsens.py $S --mode est > $R/logs/smollm2_est.log 2>&1
  python run_qsens.py $S --mode gt --per-module 8 --batch 18 > $R/logs/smollm2_gt.log 2>&1
  python run_qsens.py $L --mode gt --per-module 6 --chan-kinds int4 --batch 8 --max-minutes 60 > $R/logs/llama_gt.log 2>&1
) &
(
  python run_qsens.py $Q --mode est > $R/logs/qwen3_est.log 2>&1
  python run_qsens.py $L --mode est > $R/logs/llama_est.log 2>&1
  python run_qsens.py $Q --mode gt --per-module 8 --chan-kinds int4 --batch 10 --max-minutes 70 > $R/logs/qwen3_gt.log 2>&1
) &
wait
echo "$(date -u) ALL DONE"
