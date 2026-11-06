#!/bin/bash
# Zero-shot downstream accuracy (lm-eval, separate venv) for fp, al_H and al_H with 3% restored channels; runs after the grad queue.
cd /workspace/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/workspace/results_s4
until grep -q "GRAD QUEUE DONE" $S/logs/done.log 2>/dev/null; do sleep 60; done

declare -A NAME=([qwen3]=Qwen/Qwen3-0.6B [llama]=meta-llama/Llama-3.2-1B [gemma]=google/gemma-3-1b-pt)
declare -A SEQ=([qwen3]=2048 [llama]=2048 [gemma]=1024)
for m in qwen3 llama gemma; do
  B=/workspace/results_mackin_cat/$m
  C="--model ${NAME[$m]} --align cat --group 128 --cat-k 128 --seq-len ${SEQ[$m]}"
  Z=$S/$m/zs; mkdir -p $Z; cp $B/cat.pt $B/est_*.csv $B/est_meta.json $S/$m/grad_wt/est_grad_*.csv $S/$m/grad_wt/est_grad_meta.json $Z/
  # --eval-windows 2: run_eval loads the test set, unused in zs mode
  /workspace/venv_lme2/bin/python run_mackin.py $C --mode eval --out-dir $Z --eval-windows 2 --k f0.03125 --hows restore --random-seeds 1 \
    --selectors pred_rot,pred_rot_big_global --zs-tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,lambada_openai \
    --zs-configs "fp:-:0:-,al_H:-:0:-,al_H:pred_rot:f0.03125:restore,al_H:pred_rot_big_global:f0.03125:restore,al_H:random0:f0.03125:restore" > $S/logs/${m}_zs.log 2>&1
done
echo ZS DONE >> $S/logs/done.log
