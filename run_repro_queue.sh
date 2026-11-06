#!/bin/bash
source /workspace/venv/bin/activate
cd /workspace/repo
O=/workspace/results_cat_repro
mkdir -p $O/logs
python run_cat_repro.py --methods fp,rtn,quarot,cat --cat-k 128 --seeds 0,1,2,3 --out $O/main.jsonl > $O/logs/main.log 2>&1
python run_cat_repro.py --methods cat --cat-k 1,32,64,256,512 --seeds 0 --out $O/ksweep.jsonl > $O/logs/ksweep.log 2>&1
python run_cat_repro.py --methods cat --cat-k 1 --k1-formula text --seeds 0 --out $O/ksweep.jsonl > $O/logs/k1text.log 2>&1
echo QUEUE DONE >> $O/logs/main.log
