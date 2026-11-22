#!/usr/bin/env bash
# Rotated ground truth (Mackin ask 1) for Qwen3, Llama, Gemma, one after another.
# After EACH model: copies the small result files into results/uddhav/gt_rot/<model>/ in this repo, commits and pushes,
# so a crash or a stopped VM never loses a finished model. Rerunning skips finished work (est if present, gt per module).
# Usage, from the repo root on the VM:  nohup bash vm/queues/run_gt_rot.sh > gt_rot.out 2>&1 &
# Options (env): MODELS="qwen3 llama gemma"  KINDS=pre4,rot4  PER_MODULE=16  STOP_WHEN_DONE=1 (shuts the VM down at the end)
set -u
cd "$(dirname "$0")/../.."
REPO=$PWD
source /workspace/venv/bin/activate
export PYTHONPATH=$REPO:$REPO/pipeline PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/workspace/hf_cache/hub HF_DATASETS_CACHE=/workspace/hf_cache/datasets
R=/workspace/results_gt_rot; mkdir -p $R/logs
OUT=$REPO/results/uddhav/gt_rot

declare -A NAME=([qwen3]=Qwen/Qwen3-0.6B [llama]=meta-llama/Llama-3.2-1B [gemma]=google/gemma-3-1b-pt)
declare -A SEQ=([qwen3]=2048 [llama]=2048 [gemma]=1024)
declare -A LAYERS=([qwen3]=3,13,23 [llama]=2,8,14 [gemma]=3,13,23)  # Llama-3.2-1B has 16 layers

for m in ${MODELS:-qwen3 llama gemma}; do
  D=$R/$m
  C="--model ${NAME[$m]} --align cat --group 128 --cat-k 128 --seq-len ${SEQ[$m]} --out-dir $D"
  echo "$(date) $m: est"
  [ -f $D/est_rot4.csv ] || python pipeline/run_mackin.py $C --mode est --calib 8 > $R/logs/${m}_est.log 2>&1 \
    || { echo "$m est FAILED, see $R/logs/${m}_est.log"; continue; }
  echo "$(date) $m: gt"
  python pipeline/run_mackin.py $C --mode gt --gt-kinds ${KINDS:-pre4,rot4} --gt-layers ${LAYERS[$m]} \
    --gt-windows 16 --batch 2 --per-module ${PER_MODULE:-16} > $R/logs/${m}_gt.log 2>&1 \
    || echo "$m gt FAILED, saving what exists; see $R/logs/${m}_gt.log"
  python analysis/analyze_mackin.py $D > $D/analysis.txt 2>&1

  # save: small files only; est_*.csv keeps only the tested layers (full files are too big for git), same name so analyze_mackin.py runs on the copy
  mkdir -p $OUT/$m
  cp $D/gt.csv $D/gt_meta.json $D/est_meta.json $D/analysis.txt $R/logs/${m}_est.log $R/logs/${m}_gt.log $OUT/$m/ 2>/dev/null
  for k in pre4 rot4; do
    awk -F, -v L=",${LAYERS[$m]}," 'NR == 1 || index(L, "," $1 ",")' $D/est_$k.csv > $OUT/$m/est_$k.csv  # only the tested layers
  done
  git add -f $OUT/$m && git commit -q -m "Rotated ground truth results: $m" && { git push -q || echo "PUSH FAILED for $m: push by hand"; }
  echo "$(date) $m: saved to $OUT/$m"
done
echo "$(date) ALL DONE"
[ "${STOP_WHEN_DONE:-0}" = 1 ] && sudo shutdown -h now
