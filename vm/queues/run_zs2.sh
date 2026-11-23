#!/bin/bash
# Zero-shot for the domain-robust (s_rob) and curvature-only (curv_rot) selectors at f1/32, after S5B DONE.
# fp / al_H / pred_rot / random rows already exist in /workspace/results_s4/{m}/zs (same tasks, same lm-eval venv).
cd /workspace/repo_b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/workspace/results_xd
until grep -q "S5B DONE" $S/logs/done_b.log 2>/dev/null; do sleep 60; done

declare -A NAME=([qwen3]=Qwen/Qwen3-0.6B [llama]=meta-llama/Llama-3.2-1B [gemma]=google/gemma-3-1b-pt)
declare -A SEQ=([qwen3]=2048 [llama]=2048 [gemma]=1024)
for m in qwen3 llama gemma; do
  [ -f $S/$m/built/est_grad_rot4.csv ] || { echo "FAIL $m no built" >> $S/logs/done_zs2.log; continue; }
  B=/workspace/results_mackin_cat/$m
  C="--model ${NAME[$m]} --align cat --group 128 --cat-k 128 --seq-len ${SEQ[$m]}"
  Z=$S/$m/zs2; mkdir -p $Z; cp $B/cat.pt $S/$m/built/* $Z/
  /workspace/venv_lme2/bin/python exp_2_cheap_est.py $C --mode eval --out-dir $Z --eval-windows 2 --k f0.03125 --hows restore --random-seeds 0 \
    --selectors s_rob,curv_rot --zs-tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,lambada_openai \
    --zs-configs "al_H:s_rob:f0.03125:restore,al_H:curv_rot:f0.03125:restore" > $S/logs/${m}_zs2.log 2>&1 || echo "FAIL $m zs2" >> $S/logs/done_zs2.log
done
echo ZS2 DONE >> $S/logs/done_zs2.log
