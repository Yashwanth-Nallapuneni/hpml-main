#!/bin/bash
# Session 5 queue: multi-domain estimator. Calibrate on 6 domains, evaluate on 4 held-out domains never used for calibration.
# Runs from /workspace/repo_xd (a copy, so the running session-4 queue keeps its code); starts after ZS DONE.
source /workspace/venv/bin/activate
cd /workspace/repo_xd
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/workspace/results_xd
mkdir -p $S/logs
until grep -q "ZS DONE" /workspace/results_s4/logs/done.log 2>/dev/null; do sleep 60; done

declare -A NAME=([qwen3]=Qwen/Qwen3-0.6B [llama]=meta-llama/Llama-3.2-1B [gemma]=google/gemma-3-1b-pt)
declare -A SEQ=([qwen3]=2048 [llama]=2048 [gemma]=1024)
declare -A NG=([qwen3]=128 [llama]=128 [gemma]=256)  # grad windows per domain: 262k tokens each
GRAD=c4,pilecc,github,stackex,pmc,freelaw   # calibration domains for g1
QF=c4,github,pmc                            # calibration domains for the curvature term
HELD="wikitext arxiv uspto owt2"            # never used for calibration
SELS=curv_rot,pred_rot_big,s_rob,s_rob_global,s_c4
for m in qwen3 llama gemma; do
  B=/workspace/results_mackin_cat/$m
  C="--model ${NAME[$m]} --align cat --group 128 --cat-k 128 --seq-len ${SEQ[$m]}"
  for d in ${QF//,/ }; do
    D=$S/$m/est_$d; mkdir -p $D; cp $B/cat.pt $D/
    if [ $d = c4 ] && [ -f /workspace/results_s3/$m/est_c4/est_rot4.csv ]; then cp /workspace/results_s3/$m/est_c4/est_*.csv /workspace/results_s3/$m/est_c4/est_meta.json $D/; continue; fi
    python exp_2_cheap_est.py $C --mode est --calib 8 --est-data $d --out-dir $D > $S/logs/${m}_est_$d.log 2>&1 || echo "FAIL $m est $d" >> $S/logs/done.log
  done
  for d in ${GRAD//,/ }; do
    D=$S/$m/grad_$d; mkdir -p $D; cp $B/cat.pt $D/
    python exp_2_cheap_est.py $C --mode grad --calib ${NG[$m]} --est-data $d --out-dir $D > $S/logs/${m}_grad_$d.log 2>&1 || echo "FAIL $m grad $d" >> $S/logs/done.log
  done
  python build_xd.py $S/$m $S/$m/built --grad $GRAD --qf $QF > $S/logs/${m}_build.log 2>&1 || { echo "FAIL $m build" >> $S/logs/done.log; continue; }
  for ev in $HELD; do
    E=$S/$m/eval_$ev; mkdir -p $E; cp $B/cat.pt $S/$m/built/* $E/
    python exp_2_cheap_est.py $C --mode eval --out-dir $E --eval-data $ev --eval-windows 40 --k 2,f0.03125 --hows restore --random-seeds 2 --batch 2 \
      --selectors $SELS > $S/logs/${m}_eval_$ev.log 2>&1 || echo "FAIL $m eval $ev" >> $S/logs/done.log
  done
done
grep -q FAIL $S/logs/done.log && echo "XD DONE WITH FAILURES" >> $S/logs/done.log || echo XD DONE >> $S/logs/done.log
