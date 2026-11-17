# T04: Datasets and test-set builders

> PILOT OVERRIDE (docs/PILOT_SPEC.md wins over this brief on scope and counts): pilot sizes: train 1,000 cubes x 32 pushes (min 500), val 100 x 32, E1 100 items, E2 50, E3 50 (no-noise copies only), E4 50 heavy-OOD (m log-uniform [1.5,2.0]); no noisy variants, no other OOD sets. Disjoint cube ids, train-only normalization, generation seed + hashes stored.
> Bayes is optional; never make your deliverable depend on it. Read docs/PILOT_SPEC.md first.
> A plan (docs/plans/Txx.md, no code) is owned by this task if control asks for one.

Worktree: data | Writer: Claude Sonnet (writer-sonnet) | Reviewer: fresh Claude Opus
Reads: PROTOCOL §5-6, §9, DECISIONS | Gate: H4
Needs: H2 decisions and merged sim, synced into this worktree.
Owns: src/cs182_icl/data/, tests/data/, scripts/make_*, docs/handoffs/T04.md,
docs/questions/T04.md, docs/runbook/T04.md

Deliverables
- train pool: 10k cubes x 32 pushes; val 1k with frozen contexts + noise
- row sampler for training: 11 distinct pushes, random order, fresh noise, clean targets
- builders: E1 (10-push contexts, prefixes), E2 (query first; 3 context types x
  {includes F_q, excludes F_q}; c in {1,2,4,8}; shared duration multiset; force-separation
  sweep), E3 (base >= 6 N apart; kinds copy/fresh/new-duration; k in {0,1,2,4,8}),
  E4 (4 OOD sets with the cells PROTOCOL §9 lists)
- measurement_id handling exactly per PROTOCOL §5; fresh measurements reuse clean outcome;
  copies reuse noisy value
- normalization constants from train only, saved to meta.json
- assertions (fail loudly) for every E2/E3 construction rule
- scripts/make_dataset_card.py -> $CS282A_DATA/gates/H4/attempt-NN/
- docs/runbook/T04.md entries for generation (human launches)
