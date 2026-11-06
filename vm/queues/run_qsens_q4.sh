#!/usr/bin/env bash
# Mirrored (-e) ground truth for runs started before mirroring existed: SmolLM2 after its + run, then Qwen3 after its + run.
set -u
source /workspace/venv/bin/activate
cd "$(dirname "$0")"
export HF_HUB_CACHE=/workspace/hf_cache/hub HF_DATASETS_CACHE=/workspace/hf_cache/datasets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/results_qsens
until grep -q "GT DONE" $R/logs/smollm2_gt.log 2>/dev/null; do sleep 20; done
python run_qsens.py --model HuggingFaceTB/SmolLM2-135M --group 64 --out-dir $R/smollm2 --mode gt --no-sym --kinds int4n,rotn \
  --chan-kinds int4n,rotn --per-module 8 --batch 18 --gt-file gt_neg.csv > $R/logs/smollm2_gtneg.log 2>&1
until grep -q "GT DONE" $R/logs/qwen3_gt.log 2>/dev/null; do sleep 20; done
python run_qsens.py --model Qwen/Qwen3-0.6B --group 128 --out-dir $R/qwen3 --mode gt --no-sym --kinds int4n,rotn \
  --chan-kinds int4n --per-module 8 --batch 10 --max-minutes 45 --gt-file gt_neg.csv > $R/logs/qwen3_gtneg.log 2>&1
echo "$(date -u) Q4 DONE"
