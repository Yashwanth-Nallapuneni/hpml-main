#!/usr/bin/env bash
# Recompute every experiment so far on ONE VM, into a fresh results folder.
# Run on the VM from /workspace/repo:  bash run_all.sh            (about 3 h on one L4: bf16 pass, fp32 control, activation/output sweep, task accuracy)
# Output: $OUT/{<model>/,fp32/,w4a8/,w4a4/,w4a8o8/,tasks/,logs/,plots/,results_table.txt}   (old /workspace/results is left untouched)
set -uo pipefail
source /workspace/venv/bin/activate
cd "$(dirname "$0")"
OUT=${OUT:-/workspace/results_v2}   # override: OUT=/workspace/results_v3 bash run_all.sh
mkdir -p "$OUT/logs" "$OUT/plots"
MODELS="google/gemma-3-1b-pt meta-llama/Llama-3.2-1B"

{
  echo "host: $(hostname)  started: $(date -u)"
  nvidia-smi --query-gpu=name,driver_version,uuid --format=csv,noheader
  python -c "import torch,transformers,datasets;print('torch',torch.__version__,'transformers',transformers.__version__,'datasets',datasets.__version__)"
  md5sum src/*.py run_*.py plot_channels.py
} > "$OUT/logs/env.txt" 2>&1

for m in $MODELS; do
  short=${m##*/}
  echo "== exactness $short"
  python run_exactness.py --model "$m" > "$OUT/logs/${short}_exactness.log" 2>&1
  grep -E "max abs|PASS|FAIL" "$OUT/logs/${short}_exactness.log"

  echo "== compare $short (fp, int4, int4_rot seeds 0,1,2)"
  python run_compare.py --model "$m" --configs fp,int4,int4_rot --seeds 0,1,2 --out-dir "$OUT" \
    > "$OUT/logs/${short}_compare.log" 2>&1
  tail -7 "$OUT/logs/${short}_compare.log"

  echo "== plots $short"
  A=$(ls "$OUT/$short"/int4_seed0_*.json | tail -1)
  B=$(ls "$OUT/$short"/int4_rot_seed1_*.json | tail -1)
  python plot_channels.py --a "$A" --b "$B" --layer all --out "$OUT/plots/${short}_per_layer.png" \
    > "$OUT/plots/${short}_channels_reduced.txt" 2>&1
done

# fp32 control: in bf16, int4_rot gets 2 extra bf16 roundings (rotated input, unrotated output)
# that int4 does not. fp32 removes that confound. Written to a separate folder.
for m in $MODELS; do
  short=${m##*/}
  echo "== compare $short in fp32 (bf16-rounding control)"
  python run_compare.py --model "$m" --configs fp,int4,int4_rot --seeds 0,1,2 --dtype fp32 --out-dir "$OUT/fp32" \
    > "$OUT/logs/${short}_compare_fp32.log" 2>&1
  tail -7 "$OUT/logs/${short}_compare_fp32.log"
done

# Activation / output quantization sweep (bf16). fp is unaffected by --a-bits/--out-bits, so only
# int4 and int4_rot are run; compare against the fp row from the main pass above.
#   w4a8   : weights INT4 (g128), input activations INT8 per-token absmax (after rotation)
#   w4a4   : weights INT4 (g128), input activations INT4 per-token absmax (after rotation)
#   w4a8o8 : as w4a8, plus outputs INT8 per-token absmax before un-rotation (analog-like: DAC in, ADC out)
for setting in "w4a8:--a-bits 8" "w4a4:--a-bits 4" "w4a8o8:--a-bits 8 --out-bits 8"; do
  tag=${setting%%:*}; flags=${setting#*:}
  for m in $MODELS; do
    short=${m##*/}
    echo "== compare $short $tag ($flags)"
    python run_compare.py --model "$m" --configs int4,int4_rot --seeds 0,1,2 $flags --out-dir "$OUT/$tag" \
      > "$OUT/logs/${short}_compare_${tag}.log" 2>&1
    tail -5 "$OUT/logs/${short}_compare_${tag}.log"
  done
done

# Task accuracy (lm-eval-harness, zero-shot): arc_easy, piqa, hellaswag, winogrande. W4A16, bf16.
for m in $MODELS; do
  short=${m##*/}
  echo "== tasks $short (fp, int4, int4_rot seeds 0,1,2)"
  python run_tasks.py --model "$m" --configs fp,int4,int4_rot --seeds 0,1,2 --out-dir "$OUT/tasks" \
    > "$OUT/logs/${short}_tasks.log" 2>&1
  grep -E "^\{|avg|Traceback|Error" "$OUT/logs/${short}_tasks.log" | tail -6
done

L="$OUT/Llama-3.2-1B"
for s in 0 1 2; do
  python plot_channels.py --a "$(ls $L/int4_seed0_*.json | tail -1)" --b "$(ls $L/int4_rot_seed${s}_*.json | tail -1)" \
    --layer model.layers.1.mlp.down_proj --out "$OUT/plots/Llama-3.2-1B_L1_down_proj_seed$s.png" >/dev/null 2>&1
done

echo "finished: $(date -u)" | tee -a "$OUT/logs/env.txt"
python make_table.py "$OUT" > "$OUT/results_table.txt" 2>&1 || true
touch "$OUT/logs/ALL.done"
