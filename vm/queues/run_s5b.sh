#!/bin/bash
# Session 5b queue (after XD DONE): factorized-curvature selectors + PCA (ResQ/ROSAQ-style) baseline at equal bits on 2 held-out domains,
# then validate the 8B path (bf16 + --grad-hooks) on Qwen3 against the fp32 est g1. Code in /workspace/repo_b (run_mackin.py md5 e7ee2a74).
source /workspace/venv/bin/activate
cd /workspace/repo_b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/workspace/results_xd
until grep -q "XD DONE" $S/logs/done.log 2>/dev/null; do sleep 60; done

declare -A NAME=([qwen3]=Qwen/Qwen3-0.6B [llama]=meta-llama/Llama-3.2-1B [gemma]=google/gemma-3-1b-pt)
declare -A SEQ=([qwen3]=2048 [llama]=2048 [gemma]=1024)
for m in qwen3 llama gemma; do
  B=/workspace/results_mackin_cat/$m
  C="--model ${NAME[$m]} --align cat --group 128 --cat-k 128 --seq-len ${SEQ[$m]}"
  [ -f $S/$m/built/est_grad_rot4.csv ] || { echo "FAIL $m no built" >> $S/logs/done_b.log; continue; }
  for ev in wikitext arxiv; do
    E=$S/$m/evalb_$ev; mkdir -p $E; cp $B/cat.pt $S/$m/built/* $E/
    python run_mackin.py $C --mode eval --out-dir $E --eval-data $ev --eval-windows 40 --k 2,f0.03125 --hows restore,pca --random-seeds 1 --batch 2 \
      --selectors fishblk_rot,fishen_rot,curv_rot,s_rob > $S/logs/${m}_evalb_$ev.log 2>&1 || echo "FAIL $m evalb $ev" >> $S/logs/done_b.log
  done
done
# 8B-path validation: bf16 weights + hook g1 on 8 WikiText windows; run_grad's self-check asserts Pearson > 0.99 vs the fp32 est g1
D=$S/bf16check; mkdir -p $D; cp /workspace/results_mackin_cat/qwen3/{cat.pt,est_rot4.csv,est_meta.json} $D/
python run_mackin.py --model Qwen/Qwen3-0.6B --align cat --group 128 --cat-k 128 --seq-len 2048 --mode grad --calib 16 --est-data wikitext \
  --weights-dtype bf16 --grad-hooks --out-dir $D > $S/logs/bf16check.log 2>&1 || echo "FAIL bf16check" >> $S/logs/done_b.log
echo S5B DONE >> $S/logs/done_b.log
