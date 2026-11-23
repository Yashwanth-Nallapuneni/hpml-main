#!/usr/bin/env bash
# Third queue: SmolLM2 eval at 40 windows (the 200-window run was too slow), then Gemma-3-1B after Llama's estimator.
set -u
source /workspace/venv/bin/activate
cd "$(dirname "$0")"
export HF_HUB_CACHE=/workspace/hf_cache/hub HF_DATASETS_CACHE=/workspace/hf_cache/datasets PYTORCH_ALLOC_CONF=expandable_segments:True
R=/workspace/results_mackin
S="--model HuggingFaceTB/SmolLM2-135M --group 64 --out-dir $R/smollm2"
G="--model google/gemma-3-1b-pt --group 128 --out-dir $R/gemma"
mv $R/smollm2/eval.csv $R/smollm2/eval_200windows_partial.csv 2>/dev/null
python exp_2_cheap_est.py $S --mode eval --eval-windows 40 --k 2,8 --random-seeds 10 > $R/logs/smollm2_eval.log 2>&1
until [ -f $R/llama/est_rot4.csv ]; do sleep 30; done
python exp_2_cheap_est.py $G --mode est --calib 8 --seq-len 1024 > $R/logs/gemma_est.log 2>&1
python exp_2_cheap_est.py $G --mode eval --eval-windows 40 --k 2,8 --random-seeds 10 > $R/logs/gemma_eval.log 2>&1
python exp_2_cheap_est.py $G --mode gt --gt-windows 16 --per-module 2 --batch 1 --max-minutes 40 > $R/logs/gemma_gt.log 2>&1
echo "$(date -u) Q3 DONE"
