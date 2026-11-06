#!/usr/bin/env bash
# Unattended SmolLM2-135M sensitivity sample on the VM: sanity layer -> gate -> all layers -> summary -> VM shutdown.
# Run on the VM:  nohup bash run_sens_job.sh > /workspace/results_sens/logs/smollm2_fp32_s20.log 2>&1 < /dev/null &
set -u
source /workspace/venv/bin/activate
cd "$(dirname "$0")"
OUT=/workspace/results_sens/smollm2_fp32_s20
ARGS="--dtype fp32 --sample-per-module 20 --out-dir $OUT"
sudo shutdown -h +420   # hard cap: the VM powers off after 7 h whatever happens

echo "$(date -u) start: sanity layer 0"
python run_sensitivity.py $ARGS --layers 0
python - "$OUT" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(f"{sys.argv[1]}/layer00.csv")
r = [d[f"d_noise{a}"].rank().corr(d[f"d_noise{b}"].rank()) for a, b in [(0, 1), (0, 2), (1, 2)]]
print(f"GATE draw agreement (Spearman) {[round(x, 3) for x in r]} mean {sum(r) / 3:.3f} on {len(d)} channels")
sys.exit(0 if sum(r) / 3 >= 0.8 else 1)
EOF
if [ $? -eq 0 ]; then
  echo "$(date -u) gate passed: running all layers"
  python run_sensitivity.py $ARGS
else
  echo "$(date -u) GATE FAILED: noise draws disagree, not running the full sample"
fi
python summarize_sensitivity.py "$OUT" > "$OUT/summary.txt" 2>&1
echo "$(date -u) ALL DONE"
sudo shutdown -h now
