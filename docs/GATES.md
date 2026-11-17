# Human gates

At each gate the agent prepares a packet in `$CS282A_DATA/gates/Hn/attempt-NN/`
(README.md + figures + test output). A rerun gets a new attempt directory; never overwrite
earlier evidence. You read the packet, not the whole repo.
Pass -> merge the reviewed SHA into soomi-integration and record decisions in DECISIONS.md.
Fail -> write findings for the agent (its docs/questions/Txx.md or a message) and send it back.
Reviews: `$CS282A_DATA/reviews/Txx/<sha>/review.md` must exist for the SHA you merge.

## Stop-the-line (check whenever you glance at a diff)
- PROTOCOL.md, DECISIONS.md, or files outside the task's "Owns" edited
- tests deleted, skipped, xfailed, or tolerances loosened
- "works" claimed with no command output
- data or large files committed
- NaN losses, test-set statistics used for normalization, test cubes in train
- exact analytic branches applied to MuJoCo observations

---

## H0: Interfaces and skeleton (15-20 min)  [after T00]
- [ ] INTERFACES.md: SI units, d in meters (float64), duration as integer steps k
- [ ] measurement_id semantics match PROTOCOL §5
- [ ] file formats and paths live under $CS282A_DATA, versioned
- [ ] ownership table matches docs/tasks briefs (source, tests, scripts per task)
- [ ] pyproject has only dependency additions (mujoco, pytest); you run `uv lock` and `uv sync`
- [ ] closed-form tests pass and include: m=0.5, mu=0.3, F=26, T=0.2 -> 17.34 m;
      m=1.5, mu=0.6, F=13, T=0.1 -> 2.05 cm; clipped to 0 below threshold
Pass -> merge exp/spec; `setup_worktrees.sh build`; `tmux_session.sh` (adds windows).

## H1: Simulation scene (20 min)  [after T01 + review]
- [ ] viewer on 3 pushes (light+slippery, heavy+grippy, middle): flat slide, no bounce or tipping
- [ ] settled center height ~0.025 m; size uses half-lengths
- [ ] contact friction readout equals the cube's mu (floor friction 0)
- [ ] mj_setConst after mass/inertia change; inertia uses full side 0.05
- [ ] test proves force is applied for exactly k steps
- [ ] creep with zero force over 1 s < 0.1 mm
- [ ] rollout timing per push printed
Pass -> merge exp/sim (first merge; T02 scripts come in the second merge at H2).

## H2: Validation gate and pilot (30-45 min, YOUR decisions)  [after T02 runs]
Packet: sim-vs-analytic log-log scatter; relative and absolute discrepancy histograms and
quantiles; discrepancy vs F/(mu m g) and vs d; timestep-halving table; T^2 check; contact
friction check; flag counts (rotation, timeout); displacement and stopping-time histograms;
median training displacement; seconds per rollout and projected dataset time.
Decide and write into DECISIONS.md:
- [ ] analytic forward model with discrepancy term, or table fallback (PROTOCOL §4)
- [ ] r_model and tau_abs from the measured discrepancy (acceptance does NOT compare with sigma_d)
- [ ] keep or narrow ranges
- [ ] sigma_d value
- [ ] raw vs log target space
Pass -> commit decisions on soomi-integration, merge exp/sim (second merge), then in the
ref, train, data and eval worktrees run `git merge soomi-integration`.

## H3: Reference estimators (30 min)  [after T03 final, i.e. after H2 sync]
Analytic-data checks (exact branches, analytic observations only):
- [ ] noise-free exact solve and single-force integral reproduce the analytic oracle (rel. error < 1e-6)
- [ ] single-force Jacobian matches a brute-force fine grid with tiny sigma
- [ ] same-force rule test
MuJoCo-mode Bayes (discrepancy-aware likelihood, PROTOCOL §8):
- [ ] copies (same measurement_id) leave the posterior bitwise unchanged
- [ ] at sigma_d > 0, fresh measurements of the same action change the posterior; at sigma_d = 0
      deterministic repeats change nothing beyond the group-mean rule
- [ ] likelihood groups by action and uses variance sigma_d^2/n + tau_model^2 (unit test)
- [ ] posterior defined at sigma_d = 0 (variance = tau_model^2 > 0), including OOD
- [ ] 400 vs 800 grid differences on the hard-case subset reported and small
- [ ] broad prior uses the clipped forward model; log-sum-exp everywhere
- [ ] physics fit returns "not determined" where PROTOCOL §8 says so
- [ ] exact analytic branches are unreachable from the MuJoCo-mode API
- [ ] YOU hand-check one same-force prediction with a calculator
- [ ] review report has no unresolved CRITICAL/MAJOR items
Pass -> merge exp/ref; record grid size.

## H4: Datasets (30 min)  [after T04 runs]
Packet: dataset card.
- [ ] counts per split; cube ids disjoint across splits
- [ ] parameter and action histograms match DECISIONS
- [ ] k integer; flagged pushes counted and listed
- [ ] normalization constants computed from train only
- [ ] E2 assertions: new-force cells all |F - F_q| >= 3 N; same duration multiset across
      same-force and distinct-force cells; query identical across cells of one item
- [ ] E3: measurement_id counts per kind as designed
- [ ] noise frozen (hash of each test file recorded)
- [ ] 3 test items printed in full and look right
Pass -> merge exp/data.

## H5: Training smoke test (15 min)  [after T05]
- [ ] overfit test on a fixed diagnostic set (64 rows, fixed targets, all 11 positions):
      final MSE < 1e-3 x initial MSE within the test's step cap
- [ ] causal test: changing token j+1.. leaves outputs at positions <= j unchanged (max abs diff < 1e-6)
- [ ] 2k-step run on real data: loss falls, no NaN; tokens/s and GPU memory printed
- [ ] projected time for the step ceiling
Decide step budget -> DECISIONS.md. Pass -> merge exp/train.

## H6: Evaluation harness correctness (30 min)  [after T06; before any model exists]
- [ ] tiny hand-calculated metric fixture (a few items, numbers computed by hand) passes
- [ ] feeding Bayes predictions as the "model" reproduces the Bayes metrics exactly
- [ ] NMSE denominator is training-prior Bayes c=0 on the same test set
- [ ] paired differences use shared items; prefix contexts paired across c
- [ ] test items scored against the clean simulated target, never the noisy value
Pass -> merge exp/eval (harness now exists on soomi-integration).

## H7: First model end to end (30 min)  [no-noise model, seed 0, E1 only]
Run from a clean `release` checkout of the accepted commit; record the SHA in DECISIONS.md.
- [ ] training curves: no NaN, validation tracks training
- [ ] 10 per-item predictions printed next to Bayes and oracle
- [ ] pilot findings (c=0 vs Bayes, error vs c) recorded as RESULTS, not tuned toward
- [ ] you decide whether to spend compute on the other 5 runs
No agent changes the experiment to make a curve look right.

## H8: Main runs (5 min per check, async)
- [ ] Haiku status summary per run: last loss, val loss, steps/s, ETA, anomalies
- [ ] all curves together; no divergence; seeds really differ

## H9: Results (45-60 min)  [after T07]
- [ ] every figure compared against PROTOCOL §10 predictions
- [ ] each surprise re-checked on a second seed or item subset before believing it
- [ ] OOD results shown against both Bayes references and the analytic fit
- [ ] numbers in the draft text match the result files
Sign off before report writing.
