# HPML Team 20 · Per-Channel Sensitivity for Hadamard Rotations

COMS 6998 (HPML), Fall 2026. Mentor: Dr. Charles Mackin (IBM Research).

Shared code, small results and handovers for the team. Starting a work session (you or your coding agent)? Read `AGENTS.md` first.

## Layout
```
AGENTS.md            rules for every teammate and agent (read first)
STATE.md             one-page project status and who is on what
requirements.txt

src/                 shared library: data loading, quantizer, Hadamard, model wrapper, evaluation
pipeline/            main estimator pipeline (run_mackin.py) and the scripts it imports
analysis/            offline analysis, tables and plots
early_experiments/   session 1-3 experiments (kept for reproducibility)
vm/                  VM helper (vm.sh), VM setup, and queues/ = exact job queues that were run

results/<name>/      each person's small result files (CSV/JSON/TXT/PNG under 1 MB), one folder per session/experiment
handovers/<name>/    each person's handovers: LATEST.md + dated copies
decisions/           one short file per big decision
meetings/            mentor meeting notes and action items
report/              course report drafts and figures
```

## Running code
From the repo root:
```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/pipeline
python pipeline/run_mackin.py --help
```
Model runs need a GPU (we use the team GCP VM, see `vm/vm.sh`).

Large files (model weights, `.pt`, archives, logs, per-channel score files over 1 MB) are not in git. Keep them on the VM or in shared storage and link them from your handover.
