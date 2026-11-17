# Independent fact-check of the first draft of the packet (read-only reviewer; 1 launch, the 24th)

Checked: `GUIDE_READ_FIRST.md`, `README_PACKET.md`, `PRESENTATION_meeting.pptx` (slides and notes), `results/SUMMARY.md`, against the code at the evaluation commit, `datasets/pilot_v1`, `runs/pilot_v1_n0_s0`, `results/v3` and `v4`, the reviews, the reproduction outputs and the ledger.
Result for the first draft: **CRITICAL 1, MAJOR 6, MINOR 17.** The correction pass below was done by the controller. **The corrected versions were not fact-checked again** (no worker launches remained).

## What the checker verified (strongest evidence)
Guide Section 0 and Section 8 tables against `results/v3` (byte-identical to `csv/results_v3_best_ckpt`); the worked physics example, same-force rule, ambiguity example, token z-scores and both decoded dataset examples; the model/training table against code and `config.json` (parameter count hand-counted; checkpoint sha256 values; wall time, GPU time, throughput); the dataset table against `meta.json` (including recomputing all seven npz sha256 values and `d_obs = d_clean`); every reproduction CSV/JSON value for the contact failure; the simulator description against `rollout.py` and `cube.xml`; SHAs, GPU-hours (0.5608 / 0.5705), 23/24 launches at the time, repair counts, the 19/23 Bayes cases.

## Findings and dispositions
| ID | Finding | Disposition |
|---|---|---|
| C1 | E2 c=4 headline ("transformer cannot show the effect") contradicted by the saved `last.pt` run (+3.0e-5, [4.7e-6, 6.2e-5] vs best.pt −9.3e-6) and not disclosed | **Fixed**: guide §0, §6.5, §8.3 and limitations, slide 7 (retitled), a new backup slide and figure `B02`, SUMMARY and README now state the checkpoint dependence and call E2 at c=4 inconclusive |
| M1 | Shipped E2 figure (F09, slide crop, repo copy) was stale: Bayes distinct-force curve not drawn (figures made with the pre-fix script in the evaluation checkout) | **Fixed**: all results figures regenerated with the current script; slide versions regenerated; repo copies recommitted |
| M2 | Worked-example replay displacement caption wrong (zero point taken after the first step) | **Fixed** (`x0 = 0`); caption now 0.79437 vs 0.79437 m |
| M3 | "Floor ~2–3e-5" stated as fixed, but familiar-force cells reach ~1e-5 | **Fixed** in guide §8.3, §9.2 and the slide wording |
| M4 | "Matches Bayes" at c=2 over-claims (interval [1.5e-5, 1.0e-4]) | **Fixed** to "consistent with" |
| M5 | σ_d computed from the wrong median (0.99 m is the H2 validation sample; the training median is 0.957 m) | **Fixed**: σ_d ≈ 0.048 m (guide, SUMMARY) |
| M6 | SUMMARY said "1e-9 or better on 19 adversarial cases" (one case is 1e-7) | **Fixed**: "1e-7 or better (most ≤ 1e-9)" |
| m1 | Plateau "0.3–0.4%" understates c=2, 4 (0.426%, 0.437%) | **Fixed**: 0.33–0.44% everywhere |
| m2 | Analytic ≈ 7e-14 is standardized (1.7e-13 m²) | **Fixed** |
| m3 | 74 vs 71 steps/s | **Fixed** (both stated) |
| m4 | 0.3700 and 6.292 | **Fixed** |
| m5 | Near-duplicate-cube count attributed to T04 (it is from the audit, B6) | **Fixed** |
| m6 | "8 s" attributed to `meta.json` | **Fixed** (source: T04 report / `provenance/commands.md`) |
| m7 | Creep values: reviewer reproduced two of three | **Fixed** |
| m8 | "five scene redesigns" not traceable | **Fixed** ("several") |
| m9 | "pinned to under 1%" vs 1.2% at large ΔF | **Fixed** |
| m10 | "did not compute a paired difference" (it exists: −0.028 m², [−0.127, +0.059]) | **Fixed** |
| m11 | Hypothesis did not cover the no-force hop; "within 0.2%" is 0.2005% | **Fixed**: the writer's speed-dependent mechanism and its limits are stated |
| m12 | Slide 4 grouped 50/20-setting checks under "2000 settings" | **Fixed** |
| m13 | Slide 6 called the project's Bayes "exact Bayes" | **Fixed** |
| m14 | SUMMARY "ignores" context-conditioned performance | **Fixed** ("barely reflects") |
| m15 | Appendix A: transformer "beats" training-prior Bayes on heavy cubes (true only in MSE) | **Fixed** |
| m16 | Prediction stated as fact; GPU-hour estimate assumption; slide-18 notes pointed to a nonexistent table; this file was cited but missing | **Fixed** (this file now exists) |
| m17 | Appendix B omitted the repair-cap overrun and the unwired wandb; slide layout defects (clipped x-axis labels on slides 7–8) | **Fixed**: rows added; slide figures regenerated without baked-in captions |

## Items the checker could not verify, and what I found
* "The revoked full-campaign approval planned 6 runs": consistent with the earlier `PREAPPROVAL.md` (six training runs) — checked by me.
* "Bayes supports σ_d > 0": the likelihood code takes `sigma_d`, but the public predictor fixes `SIGMA_D = 0.0`; the guide now says so.
* "7%–1000%" displacement error for the `solreffriction` attempt: appears only in the writer's questions file; its handoff table lists 6.6%–383% at that commit. Both are now stated.
