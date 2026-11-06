#!/bin/bash
# Session 4 queue: g1 from the weight gradient over 512 windows (--mode grad), then eval with the big-sample selectors.
source /workspace/venv/bin/activate
cd /workspace/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/workspace/results_s4
mkdir -p $S/logs

declare -A NAME=([qwen3]=Qwen/Qwen3-0.6B [llama]=meta-llama/Llama-3.2-1B [gemma]=google/gemma-3-1b-pt)
declare -A SEQ=([qwen3]=2048 [llama]=2048 [gemma]=1024)
SELS=pred_rot,pred_rot_global,pred_rot_big,pred_rot_big_global,g1_rot_big,curv_rot
for m in qwen3 llama gemma; do
  B=/workspace/results_mackin_cat/$m
  C="--model ${NAME[$m]} --align cat --group 128 --cat-k 128 --seq-len ${SEQ[$m]}"
  # grad on WikiText-2 train (uses the actual window count if the split has fewer than 512)
  D=$S/$m/grad_wt; mkdir -p $D; cp $B/cat.pt $B/est_*.csv $B/est_meta.json $D/
  python run_mackin.py $C --mode grad --calib 512 --est-data wikitext --out-dir $D > $S/logs/${m}_grad_wt.log 2>&1
  if [ $m != gemma ]; then
    D=$S/$m/grad_c4; mkdir -p $D; cp $B/cat.pt $D/
    python run_mackin.py $C --mode grad --calib 512 --est-data c4 --out-dir $D > $S/logs/${m}_grad_c4.log 2>&1
  fi
  for ev in wt c4; do
    E=$S/$m/eval_$ev; mkdir -p $E; cp $B/cat.pt $B/est_*.csv $B/est_meta.json $S/$m/grad_wt/est_grad_*.csv $S/$m/grad_wt/est_grad_meta.json $E/
    if [ $ev = wt ]; then X="--eval-windows 40"; else X="--eval-data c4 --eval-windows 128"; fi
    python run_mackin.py $C --mode eval --out-dir $E $X --k 2,f0.03125 --hows restore --random-seeds 3 --batch 2 \
      --selectors $SELS > $S/logs/${m}_eval_$ev.log 2>&1
  done
done
echo GRAD QUEUE DONE >> $S/logs/done.log
