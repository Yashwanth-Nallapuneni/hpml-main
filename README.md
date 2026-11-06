# HPML Team 20 · Per-Channel Sensitivity for Hadamard Rotations

COMS 6998 (HPML), Fall 2026. Mentor: Dr. Charles Mackin (IBM Research).

This repo holds the team's shared code, small results and handovers. If you (or your coding agent) are starting a work session, read `AGENTS.md` first.

## Layout
| folder | what goes here |
|---|---|
| `src/` | shared library code (data loading, quantizer, Hadamard, model wrapper, evaluation) |
| repo root `*.py`, `*.sh` | entry points, VM queues and analysis scripts. They import each other and `src/`, so run them from the repo root (for example `python run_mackin.py --help`) |
| `paper/` | report / paper drafts, figures script, bib |
| `results/<date>_<name>_<experiment>/` | small result files (CSV/JSON summaries, under 1 MB) plus a README with the command, commit and cost |
| `handovers/<name>/` | each person's handovers: `LATEST.md` plus dated copies |
| `decisions/` | one short file per big decision |
| `meetings/` | mentor meeting notes and action items |
| `STATE.md` | one-page current status of the project |

Large files (model weights, `.pt`, archives, full logs) do not go in git. Put them in shared cloud storage and link them from the result's README.
