# HPML Team 20 · Per-Channel Sensitivity for Hadamard Rotations

COMS 6998 (HPML), Fall 2026. Mentor: Dr. Charles Mackin (IBM Research).

Shared code, small results and handovers for the team.

- **New to the project? Read `OVERVIEW.md`** (what the project is, how the estimator works, what we found).
- **Prefer video?** Watch the 34-min explainer: [Which Channels Matter?](https://github.com/Yashwanth-Nallapuneni/team20-hpml/releases/tag/video-2026-11-09) (GitHub Release, download the .mp4).
- Starting a work session (you or your coding agent)? Read `AGENTS.md` first.

## Layout
```
OVERVIEW.md          plain-language explanation of the whole project (start here)
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

Known limits: `analysis/analysis_lodo.py` only runs on the VM (hard-coded to `/workspace/results_xd`). The `vm/queues/` scripts are records of what was run and use the old flat layout (`/workspace/repo*`), so fix their paths before reusing them.

Large files (model weights, `.pt`, archives, logs, per-channel score files over 1 MB) are not in git. Keep them on the VM or in shared storage and link them from your handover.
