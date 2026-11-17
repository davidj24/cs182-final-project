# Exact commands

```bash
. /home/exx/soomi/soomi_projects/cs282a_project/workflow/env.sh          # from inside the checkout; sets CS282A_DATA, PYTHONPATH=$PWD/src

# 1. Simulator validation + gate packet (source 3c74ca1; CPU, <=16 workers)
python scripts/sim_validate.py --workers 16
python scripts/sim_gate_packet.py --csv $CS282A_DATA/validation/sim_validate/v1/validation.csv

# 2. Dataset (source f15567c; ~8 s)
python scripts/make_pilot_dataset.py --workers 16 --seed 0
python scripts/make_dataset_card.py --dataset-dir $CS282A_DATA/datasets/pilot_v1

# 3. Training (clean release checkout at e6c51af; launched through the budget guard)
CUDA_VISIBLE_DEVICES=1 workflow/run_job.sh pilot_train_v1_n0_s0 0.9 3300 -- \
  python scripts/train.py --dataset $CS282A_DATA/datasets/pilot_v1 --steps 150000 --seed 0 --noise 0 \
  --eval-every 1000 --max-seconds 3000 --run-id pilot_v1_n0_s0
# smoke check before it: same command with --steps 2000 --run-id pilot_v1_smoke2k (0.0086 GPU-h)

# 4. Verify the checkpoints, then evaluate (clean release checkout at 2f3d53d; CPU)
python scripts/pilot_load_run.py --run-dir $CS282A_DATA/runs/pilot_v1_n0_s0 --ckpt best.pt     # prints VERIFY OK
export PYTHONPATH=$PWD/src:$PWD/scripts CS282A_RUN_DIR=$CS282A_DATA/runs/pilot_v1_n0_s0
CS282A_CKPT=best.pt python scripts/eval_run.py --dataset-dir $CS282A_DATA/datasets/pilot_v1 --model-loader pilot_load_run:load       # -> results/v3
CS282A_CKPT=last.pt python scripts/eval_run.py --dataset-dir $CS282A_DATA/datasets/pilot_v1 --model-loader pilot_load_run:load \
   --reuse-cache $CS282A_DATA/results/v3                                                                                         # -> results/v4 (sensitivity)

# 5. Figures and packet material (control-authored scripts; repro/ in this packet)
python scripts/pilot_make_figures.py --results-dir $CS282A_DATA/results/v3 --out-dir <new dir>
python repro/make_packet_figures.py <packet dir>
python repro/guide_facts.py <new dir>
python repro/supp_broad_bayes.py <new dir>
PYTHONPATH=<isolated checkout of 5e0ae3f>/src python repro/contact_failure_reproduction.py <new dir>
```
