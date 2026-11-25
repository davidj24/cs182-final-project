# Loss-weighting ablation (written before either run started, 2026-11-24)

**Question.** Is the transformer's error plateau for c >= 2 caused by the no-context position dominating the
training loss?

**Runs.** Same code, data (`pilot_v1`), model, seed 0, optimizer, batch size and schedule; 20,000 steps each
(GTX 1650, float32).
- A (baseline): loss over all 11 positions, as in the pilot.
- B: `--loss-first-pos 1`, loss over positions 1-10 only. This is the only change.

**Primary comparison (fixed now).** The final checkpoint (`last.pt`) of each run, so that no
checkpoint-selection rule differs between them. Metrics: E1 standardized MSE and median relative error at
each c >= 1; per-position validation MSE for positions 1-10. E2 and E3 are secondary.

**Reading.** If B is clearly lower than A at c >= 2, loss dominance contributes to the error at this budget.
If not, it is not the main cause at this budget. Either way: one seed, a 7.5x shorter budget than the
pilot, so the result does not transfer to the 150k-step model without further runs. B's c = 0 prediction is
untrained and is not interpreted.
