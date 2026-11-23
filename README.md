# HPML Team 20 · Per-Channel Sensitivity for Hadamard Rotations

COMS 6998 (HPML), Fall 2026. Mentor: Dr. Charles Mackin (IBM Research).

Code and small result files for the project.

## Layout
```
requirements.txt

src/                 shared library: data loading, quantizer, Hadamard, model wrapper, evaluation
pipeline/            main estimator pipeline (run_mackin.py) and the scripts it imports
analysis/            offline analysis, tables and plots
early_experiments/   session 1-3 experiments (kept for reproducibility)
vm/                  VM helper (vm.sh), VM setup, and queues/ = exact job queues that were run

results/<name>/      small result files (CSV/JSON/TXT/PNG under 1 MB), one folder per session/experiment
```

## Running code
From the repo root:
```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/pipeline
python pipeline/run_mackin.py --help
```
Model runs need a GPU (we use a GCP VM with one L4, see `vm/vm.sh`).

Known limits: `analysis/analysis_lodo.py` only runs on the VM (hard-coded to `/workspace/results_xd`). The `vm/queues/` scripts are records of what was run and use the old flat layout (`/workspace/repo*`), so fix their paths before reusing them.

Large files (model weights, `.pt`, archives, logs, per-channel score files over 1 MB) are not in git.
