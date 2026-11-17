# Pre-approval for the autonomous meeting pilot (replaces the full-campaign T00-T07 approval)

Status: APPROVED 2026-11-16 by user (message instructing: replace the full-campaign approval with
the pilot caps, reconcile R1-R10, then proceed autonomously). Scope: docs/PILOT_SPEC.md only.

The control session decides everything below by rule, without asking. Anything these rules do not
determine is "unresolved scientific ambiguity" and stops the run (docs/AUTONOMY.md §Stop).
The full-campaign deadline (5 days), 36 GPU-hours, 60 launches, six training runs and the
20,000-step minimum are REVOKED. They do not apply to the pilot.

## A. Budgets (hard caps, enforced by workflow/budget_check.sh where measurable)
| Item | Cap | Enforcement |
|---|---|---|
| Goal | one validated reproducible training run + E1 packet + SUMMARY.md; completion-based, no clock cutoff | n/a |
| Deadline | none (DEADLINE in budget.env is empty; if ever set it is read in America/Los_Angeles) | script |
| GPU time (one training run at a time, one GPU) | <= 2.0 device-hours total, including reserved/active jobs | reserve before launch, ledger GPUH |
| Worker/reviewer launches | <= 24 total; <= 2 repair cycles per task | ledger LAUNCH, REPAIR |
| CPU simulation workers | <= 16 (fewer if load average is high) | script prints load; jobs pass --workers |
| New pilot outputs | <= 10 GB, measured as bytes under $CS282A_DATA minus the baseline recorded at approval | script |
| Free disk on / | >= 30 GB (disk is 94% full) | script |
| Agent/dollar spend | NOT measurable, NOT enforced; launches are the proxy | stated honestly |
If finishing requires exceeding any cap: write progress to the ledger, tell the user, and ASK
before exceeding it. Never exceed silently.

## B. Scientific decision rules (replace the human H2/H5 decisions)
R1 Forward model / Bayes validity. On the validation run, over training-domain moving pushes:
rel = |sim - analytic| / analytic. Accept the analytic model WITH a discrepancy term for Bayes if
median rel <= 1% and p95 rel <= 3% and dt-halving (50 settings) changes sim d by median <= 0.5%,
p95 <= 2%. If this fails, Bayes is "blocked" (reason recorded) and the pilot continues; no
table-building campaign tonight. A dt-halving failure is first investigated as a possible physics
error (stop if it is) or a discretization effect (Bayes blocked, recorded). Changing dt is a
protocol change and needs the user.
R2 r_model, tau_abs. Only if R1 passes: delta = sim - f; r_model = RMS(delta/f) over training-domain
pushes with f >= 0.1 m; tau_abs = RMS(delta) over those with f < 0.1 m, floored at 1e-4 m; both
rounded up to 2 significant figures. If R1 fails these stay unset.
R3 Ranges. FROZEN as in PILOT_SPEC §C. No automatic adjustment. Rotation-flagged and timed-out
rollouts are kept, counted and reported. Only broken physics (G1 mechanics failing: friction
readout, force timing, settling, inertia) stops data generation. A material flag rate
(> 1% rotation or any timeout) is reported prominently in SUMMARY; control does not raise the mass bound.
R4 sigma_d. Not needed (training noise 0). Not computed tonight.
R5 Target space. FROZEN: raw standardized displacement, clean MuJoCo target. No log target tonight
(may be reported as a diagnostic from validation data only).
R6 Step budget. One no-noise seed-0 model. After a benchmark of a few hundred steps, choose steps so
real training takes about 30-45 minutes (about 0.5-0.75 GPU-hours); warmup ~5% of steps (cap 2,000);
cosine over the actual budget. No minimum step count. The training job is launched through
workflow/run_job.sh with a reserved GPU-hour budget and a wall-clock timeout.
R7 Seeds/conditions. Exactly one run: noise 0, seed 0. No other seeds or noise conditions. Bayes grid
only if R1 passes: 400x400 with a finer check on a small hard-case subset.
R8 First-model gate (the only training gate). Real run only after the mandatory pre-training checks
pass: shape, causal, training-only normalization, leakage, fixed-set overfit. After training: no NaN;
final train and val loss clearly below initial; metric hand-fixture passes; Bayes-as-model harness
check passes if Bayes exists; E1 evaluation completes. Closeness to Bayes and the shape of the
error-vs-c curve are results, never gates. Failure after 2 fix cycles, or a fix requiring a
protocol change: STOP and ask.
R9 Merges. Control merges a SHA into soomi-integration, only in the dedicated integration checkout
cs282a_wt/integration, only if a review of that exact SHA has ZERO unresolved CRITICAL or MAJOR
findings and `pytest -m "not slow"` passes there. Every integrated SHA goes in the ledger. Never
main, never soomi, never teammates' branches, never push.
R10 Results labels. Descriptive, single-seed. SUMMARY reports numbers and paired differences over
shared items (paired SE or item bootstrap where cheap), with wording like "in this pilot run".
No "consistent/inconsistent with the hypothesis" verdicts, no "generalizes", no cross-seed
significance claims. Surprises are re-checked (second item subset or independent recomputation)
before being called findings; otherwise labelled unverified. Interpretation is a DRAFT.

## C. Human-only checks that become automated (a real downgrade; accept knowingly)
- H1 viewer inspection -> numeric: max rotation < 1 deg, max height excursion < 1 mm, settled
  height 0.025 +- 1e-4 m; control also saves PNG frames for later inspection.
- H3 calculator hand-check -> a fresh reviewer recomputes 5 same-force predictions in plain Python
  independent of the project code (only if the Bayes/physics-fit reference is delivered).
- H4/H5/H6/H8/H9 -> assertions and thresholds in GATES.md; the user reviews afterwards.

## D. Not allowed without the user
Editing PROTOCOL.md; changing dt, cube geometry, ranges, hypotheses, E1-E4 definitions; pushing;
touching main/soomi or teammates' branches; deleting or overwriting anything under $CS282A_DATA or
hw9_hardware_check/; installing system packages; stopping unrelated processes; plain `uv sync` on
the shared env; exceeding any cap in A.
