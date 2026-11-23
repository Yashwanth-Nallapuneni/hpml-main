#!/usr/bin/env bash
# Mackin proof of concept on 3 models, two queues sharing the L4. Run on the VM:
#   nohup bash run_mackin_job.sh > /workspace/results_mackin/logs/job.log 2>&1 < /dev/null &
set -u
source /workspace/venv/bin/activate
cd "$(dirname "$0")"
export HF_HUB_CACHE=/workspace/hf_cache/hub HF_DATASETS_CACHE=/workspace/hf_cache/datasets PYTORCH_ALLOC_CONF=expandable_segments:True
R=/workspace/results_mackin
mkdir -p $R/logs
S="--model HuggingFaceTB/SmolLM2-135M --group 64 --out-dir $R/smollm2"
L="--model meta-llama/Llama-3.2-1B --group 128 --out-dir $R/llama"
Q="--model Qwen/Qwen3-0.6B --group 128 --out-dir $R/qwen3"
(
  python exp_2_cheap_est.py $S --mode est --calib 16 > $R/logs/smollm2_est.log 2>&1
  python exp_2_cheap_est.py $S --mode eval --eval-windows 200 --k 1,2,4,8,16 --random-seeds 10 > $R/logs/smollm2_eval.log 2>&1
  python exp_2_cheap_est.py $L --mode est --calib 8 > $R/logs/llama_est.log 2>&1
  python exp_2_cheap_est.py $L --mode eval --eval-windows 40 --k 2,8 --random-seeds 10 > $R/logs/llama_eval.log 2>&1
) &
(
  until [ -f $R/smollm2/scales.pt ]; do sleep 10; done
  python exp_2_cheap_est.py $S --mode gt --gt-windows 16 --per-module 4 --batch 4 --max-minutes 45 > $R/logs/smollm2_gt.log 2>&1
  python exp_2_cheap_est.py $Q --mode est --calib 8 > $R/logs/qwen3_est.log 2>&1
  python exp_2_cheap_est.py $Q --mode eval --eval-windows 40 --k 2,8 --random-seeds 10 > $R/logs/qwen3_eval.log 2>&1
  until [ -f $R/llama/est_rot4.csv ]; do sleep 30; done
  python exp_2_cheap_est.py $L --mode gt --gt-windows 16 --per-module 2 --batch 1 --max-minutes 40 > $R/logs/llama_gt.log 2>&1
) &
wait
echo "$(date -u) ALL DONE"
