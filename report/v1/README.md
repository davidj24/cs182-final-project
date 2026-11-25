# Report V1: reproducing the figures and tables

On Windows, some archived evidence files have long paths. If checkout fails with "Filename too long", clone with
`git clone -c core.longpaths=true -b v1-report https://github.com/davidj24/cs182-final-project.git`.

## Figures (about a minute, CPU only, no model or dataset needed)

```bash
python -m pip install matplotlib      # or use the project environment: uv run ...
python scripts/make_report_figures.py
```

This reads the committed result tables in `results/pilot_v1/csv/` and writes `report/v1/figures/`.

Tables and quoted numbers (standard library only):

```bash
python scripts/make_report_tables.py
```

| In the report | Source file under `results/pilot_v1/csv/` |
|---|---|
| E1 figure, E1 tables | `results_v3_best_ckpt/e1_error_vs_context.csv`, `e1_metrics.csv`, `e1_paired.csv` |
| E2 figure and tables | `results_v3_best_ckpt/e2_same_vs_distinct.csv`, `e2_paired.csv`; final checkpoint row from `results_v4_last_ckpt_sensitivity/e2_paired.csv` |
| E3 figure and numbers | `results_v3_best_ckpt/e3_redundancy.csv`, `e3_prediction_change.csv` |
| E4 figure and table | `results_v3_best_ckpt/e4_heavy_ood.csv`, `supp_broad_prior_bayes/bayes_prior_comparison.csv` |
| Per-position validation figure | `per_position_val_loss.csv` |
| One-push audit numbers | `c1_audit_summary.json`; counterfactual checks: `results/meeting_packet_v1/GUIDE_READ_FIRST.md` section 8.2 |
| Training-curve figure | `results/pilot_v1/figures/F05_training_curves.png`, saved by the pilot run. Its raw `metrics.jsonl` is **not** committed yet. |
| Simulator and contact-scene numbers | `results/meeting_packet_v1/` (`GUIDE_READ_FIRST.md`, `simulator_evidence/`) |

## Full pipeline (data, training, evaluation)

Requires a GPU for training (the pilot took 33.6 minutes on one RTX 4090). Run from the repo root with
`PYTHONPATH=src` and `CS282A_DATA` set to a directory outside the repo.

```bash
export CS282A_DATA=/path/to/data PYTHONPATH=$PWD/src:$PWD/scripts
python scripts/make_pilot_dataset.py --workers 16 --seed 0
python scripts/train.py --dataset $CS282A_DATA/datasets/pilot_v1 --steps 150000 --seed 0 --noise 0 \
    --eval-every 1000 --run-id pilot_v1_n0_s0
python scripts/pilot_load_run.py --run-dir $CS282A_DATA/runs/pilot_v1_n0_s0 --ckpt best.pt
CS282A_RUN_DIR=$CS282A_DATA/runs/pilot_v1_n0_s0 CS282A_CKPT=best.pt \
    python scripts/eval_run.py --dataset-dir $CS282A_DATA/datasets/pilot_v1 --model-loader pilot_load_run:load
```

A retrained model is not expected to match the pilot's checkpoint bit for bit (GPU arithmetic is not
deterministic across hardware). The pilot's trained checkpoints are not yet released.

## Compiling the report

`report/v1/main.tex` uses the NeurIPS 2015 style file `nips15submit_e.sty` from
https://neurips.cc/Conferences/2015/PaperInformation/StyleFiles placed next to it.
