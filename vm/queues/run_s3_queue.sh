#!/bin/bash
# Session 3 queue (2026-11-03): runs after run_repro_queue.sh. One setting per output directory.
source /workspace/venv/bin/activate
cd /workspace/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/workspace/results_s3
mkdir -p $S/logs
until grep -q "QUEUE DONE" /workspace/results_cat_repro/logs/main.log 2>/dev/null; do sleep 60; done

# smoke: new exp_2_cheap_est features on SmolLM2 (2 windows) so a bug shows up in minutes, not hours
D=$S/smoke; mkdir -p $D; cp /workspace/results_mackin_cat/smollm2/{cat.pt,est_pre4.csv,est_rot4.csv,est_pre8.csv,est_meta.json} $D/
python exp_2_cheap_est.py --model HuggingFaceTB/SmolLM2-135M --align cat --group 64 --cat-k 192 --mode eval --out-dir $D --eval-windows 2 \
  --k 2,f0.03125 --hows restore,isolate,isolate8 --budget-group 32 --random-seeds 2 > $S/logs/smoke.log 2>&1
python analyze_gt.py $D >> $S/logs/smoke.log 2>&1

# item 5: W4A4 (CAT paper setting): plain vs sensitivity-weighted vs shuffled CAT on 4 calibration seeds; exemption on seed 0
python run_w4a4_sens.py --sw none,fisher,shuffled --seeds 0,1,2,3 --exempt-k 2,8 --out $S/w4a4_sens.jsonl > $S/logs/w4a4_sens.log 2>&1

declare -A NAME=([qwen3]=Qwen/Qwen3-0.6B [llama]=meta-llama/Llama-3.2-1B [gemma]=google/gemma-3-1b-pt)
declare -A SEQ=([qwen3]=2048 [llama]=2048 [gemma]=1024)
SELS=pred_rot,pred_pre,curv_rot,energy,err_rot
for m in qwen3 llama gemma; do
  SRC=/workspace/results_mackin_cat/$m
  C="--model ${NAME[$m]} --align cat --group 128 --cat-k 128 --seq-len ${SEQ[$m]} --random-seeds 5 --batch 2"
  prep() { mkdir -p $1; cp $SRC/cat.pt $1/; [ "$2" = est ] && cp $SRC/est_*.csv $SRC/est_meta.json $1/; }
  # item 4: equal bit budget (INT8 or fp for 1/32 of rows vs smaller groups everywhere), plus global allocation
  D=$S/$m/budget; prep $D est
  python exp_2_cheap_est.py $C --mode eval --out-dir $D --eval-windows 40 --k f0.03125 --hows isolate8,restore --budget-group 64,32 \
    --selectors pred_rot,pred_rot_global,err_rot,energy,fisher,wmax > $S/logs/${m}_budget.log 2>&1
  # item 3: whole WikiText-2 test split, k = 2
  D=$S/$m/wt_full; prep $D est
  python exp_2_cheap_est.py $C --mode eval --out-dir $D --eval-windows 0 --k 2 --hows restore --selectors $SELS > $S/logs/${m}_wt_full.log 2>&1
  # item 3: two more Hadamard seeds (estimator rerun for each, since rot4 errors change)
  for r in 1 2; do
    D=$S/$m/rot$r; prep $D
    python exp_2_cheap_est.py $C --mode est --rot-seed $r --out-dir $D > $S/logs/${m}_rot${r}_est.log 2>&1
    python exp_2_cheap_est.py $C --mode eval --rot-seed $r --out-dir $D --eval-windows 40 --k 2 --hows restore --selectors $SELS > $S/logs/${m}_rot${r}_eval.log 2>&1
  done
  # item 3: k sweep incl. literature-style baselines (wmax ~ SpQR outlier rows, err_rot ~ GPTQ/SpQR output error, fisher ~ HAWQ/SqueezeLLM, energy ~ AWQ/LLM.int8 magnitude)
  D=$S/$m/ksweep; prep $D est
  python exp_2_cheap_est.py $C --mode eval --out-dir $D --eval-windows 40 --k 1,4,16 --hows restore \
    --selectors pred_rot,curv_rot,pred_pre,fisher,err_rot,energy,wmax,pred_rot_global > $S/logs/${m}_ksweep.log 2>&1
  # item 3: second data set (C4 validation, GPTQ recipe, 128 windows)
  D=$S/$m/c4; prep $D est
  python exp_2_cheap_est.py $C --mode eval --out-dir $D --eval-data c4 --eval-windows 128 --k 2 --hows restore --selectors $SELS > $S/logs/${m}_c4.log 2>&1
done
# item 3: Mackin's ground truth on one target model
D=$S/qwen3/gt; mkdir -p $D; cp /workspace/results_mackin_cat/qwen3/{cat.pt,est_pre4.csv,est_rot4.csv,est_pre8.csv,est_meta.json} $D/
python exp_2_cheap_est.py --model Qwen/Qwen3-0.6B --align cat --group 128 --cat-k 128 --mode gt --gt-windows 16 --per-module 4 --batch 2 \
  --max-minutes 60 --out-dir $D > $S/logs/qwen3_gt.log 2>&1
echo S3 QUEUE DONE >> $S/logs/done.log
