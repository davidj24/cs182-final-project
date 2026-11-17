# T05: Model and training (adapt Garg et al.)

> PILOT OVERRIDE (docs/PILOT_SPEC.md wins over this brief on scope and counts): one model: noise 0, seed 0; pre-training checks (shape, causal, normalization, leakage, fixed-set overfit) are mandatory; step budget chosen from a benchmark for ~30-45 min; warmup ~5% (cap 2,000); lr 1e-4, wd 0.01, batch 256. No seed/noise sweep, no 20k-step minimum.
> Bayes is optional; never make your deliverable depend on it. Read docs/PILOT_SPEC.md first.
> A plan (docs/plans/Txx.md, no code) is owned by this task if control asks for one.

Worktree: train | Writer: Claude Sonnet (writer-sonnet; Opus if it struggles) | Reviewer: fresh Claude Opus
Reads: PROTOCOL §5, §7 | Gate: H5
Can start after H0 on synthetic rows from physics.py; swap in real data after H4.
Owns: src/cs182_icl/model/, src/cs182_icl/train/, third_party/, tests/model/, tests/train/,
scripts/train.py, scripts/bench_train.py, configs/train/ (templates only; frozen run
configs are created by the human from DECISIONS.md), docs/handoffs/T05.md,
docs/questions/T05.md, docs/runbook/T05.md

Deliverables
- vendor Garg et al. GPT-2 model + loop into third_party/ with license note
- tokenization per PROTOCOL §5 (5-dim tokens; read prediction at push token)
- loss: MSE over all 11 positions; clean targets
- AdamW, warmup, cosine; checkpoint on best val; config template per run
  (noise in {0, sigma_d} x seed in {0,1,2})
- tests: overfit fixed 64-row diagnostic set (threshold in GATES.md H5); causal masking;
  shape/normalization
- scripts/train.py + scripts/bench_train.py (2k steps; prints tokens/s, memory, projected time)
- logs to $CS282A_DATA/runs/<run_id>/ (metrics.jsonl, config, ckpt); wandb per team convention
- docs/runbook/T05.md entries
