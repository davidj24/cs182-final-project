# Interfaces (T00, pilot initial_e2e_2026_11_16)

Pilot scope comes from docs/PILOT_SPEC.md (wins over task briefs); long-term plan is PROTOCOL.
Pilot values: configs/pilot/initial_e2e_2026_11_16.json.

## Units and constants
- SI everywhere: m in kg, F in N, displacement d in meters (float64), time in seconds.
- Push duration is an integer step count k; T = k * DT, DT = 0.001 s (`cs182_icl.types.DT`).
- g = 9.81 m/s^2 (`cs182_icl.physics.G`), used by every analytic/Bayes computation.
- Physics and Bayes code use float64. Every random draw takes an explicit seed.

## Types (`src/cs182_icl/types.py`, frozen dataclasses)
| Type | Fields |
|---|---|
| Cube | m: float, mu: float, cube_id: int |
| Action | F: float, k: int; property T = k*DT |
| Observation | action: Action, measurement_id: int, d_clean: float, d_obs: float |
| PushResult | d: float, t_stop: float, max_rotation_deg: float, timed_out: bool |
| TestItem | cube: Cube, query: Action, d_query_clean: float, context: list[Observation], condition: str |
| Prediction | d_pred: float or None, status: "ok" / "not_determined" / "blocked", reason: str |

TestItem.condition examples: "E1", "E2/A" ... "E2/D", "E3/k=2", "E4/heavy". The scoring target
is always d_query_clean (clean MuJoCo outcome). Prediction is the return type of references.

## Functions
| Function | Module (owner) | Status after T00 |
|---|---|---|
| analytic_displacement(m, mu, F, T, g=G) | physics.py (T00) | implemented |
| simulate_push(cube, action, *, dt=DT) -> PushResult | sim/rollout.py (T01) | NotImplementedError |
| physics_fit_predict(context, query) -> Prediction | reference/physics_fit.py (T03) | NotImplementedError |
| bayes_predict(context, query, *, prior="train") -> Prediction | reference/bayes.py (T03) | returns "blocked" |
| model_predict(model, items, norm) -> np.ndarray (meters, shape (len(items),)) | model/predict.py (T05) | NotImplementedError |

physics.py (scalars or numpy arrays; 0-d results are Python floats):
A = 1/(2 m^2 mu g), B = 1/(2m); `line_coeffs(m, mu) -> (A, B)`; `motion_threshold(m, mu) = mu m g`;
`analytic_displacement = F T^2 max(AF - B, 0)`, exactly 0 when F <= mu m g;
`stop_time = F T/(m mu g)` for moving pushes, 0.0 if the cube never moves;
`same_force_displacement(d1, T1, Tq) = d1 (Tq/T1)^2`; `line_value(d, F, T) = d/(F T^2)`;
`params_from_line(A, B) -> (m = 1/(2B), mu = 2B^2/(A g))`; `duration(k) = k*DT`.

Optional Bayes: bayes_predict never raises in the pilot. Until T03 implements it after
PREAPPROVAL R1 passes, it returns Prediction(None, "blocked", reason). Callers record
"Bayes reference: blocked, reason: ..." and continue. Data, training and E1 never depend on it.
physics_fit_predict returns "not_determined" (d_pred None) when the context does not identify
the query (fewer than two distinct forces and no observation at query.F); never a pseudoinverse.
Analytic-exact Bayes branches (PROTOCOL §8) are analytic-data only, never on MuJoCo data.

## measurement_id (PROTOCOL §5)
int64, unique within a dataset version (across all splits); -1 marks padding.
- A copied observation keeps its ID and its observed value.
- A fresh measurement of the same action gets a new ID and reuses the action's clean simulated
  outcome (deterministic simulator); in noisy conditions it gets new noise (pilot noise is 0).
- A different action gets a new ID and its own simulation.
Bayes deduplicates by measurement_id before grouping by action.

## Dataset files: $CS282A_DATA/datasets/<version>/ (version = v1, v2, ...; never overwrite)
`meta.json`: version, pilot config path + sha256, generation_seed, source git SHA, dt, g,
ranges, counts per file, sha256 of every .npz, flag counts (rotation > 1 deg, timeouts), and
`normalization` computed from train.npz ONLY: mean/std of F, T, d_obs, asinh(d_obs/0.01).

`train.npz`, `val.npz` (N cubes, P = 32 pushes per cube):
| field | dtype | shape |
|---|---|---|
| cube_id | int64 | (N,) |
| m, mu | float64 | (N,) |
| F | float64 | (N, P) |
| k | int64 | (N, P) |
| measurement_id | int64 | (N, P) |
| d_clean, d_obs | float64 | (N, P) |
| t_stop, max_rotation_deg | float64 | (N, P) |
| timed_out | bool | (N, P) |
val.npz also stores `row_idx` int64 (R, 11): fixed push indices (into axis P) per validation row,
and `row_cube` int64 (R,): row index into axis N. Training rows sample 11 distinct pushes on the fly.

`e1.npz`, `e2.npz`, `e3.npz`, `e4.npz` (I items, Cmax = max context length, padded):
| field | dtype | shape |
|---|---|---|
| item_id | int64 | (I,) pairing key: same cube+query across conditions |
| condition | str (`<U32`) | (I,) |
| cube_id | int64 | (I,) |
| m, mu | float64 | (I,) |
| query_F, d_query_clean, query_t_stop, query_max_rotation_deg | float64 | (I,) |
| query_k | int64 | (I,) |
| query_timed_out | bool | (I,) |
| ctx_len | int64 | (I,) |
| ctx_F, ctx_d_clean, ctx_d_obs | float64 | (I, Cmax), NaN padded |
| ctx_k, ctx_measurement_id | int64 | (I, Cmax), -1 padded |
E1/E4 store the 10-push context once per item; context for c is the first c pushes (paired).
E2/E3 store one row per (cell or copy count, c), linked by item_id. Flagged rollouts are kept.

## Ownership (sources / tests / scripts / task files docs/{handoffs,questions,runbook}/Txx.md)
| Task | Sources | Tests | Scripts / other |
|---|---|---|---|
| T01 | src/cs182_icl/sim/ | tests/sim/ | scripts/sim_view.py, assets/ |
| T02 | (none) | tests/sim/test_validate_*.py | scripts/sim_validate.py, sim_pilot.py, sim_gate_packet.py |
| T03 | src/cs182_icl/reference/ | tests/reference/ | (none) |
| T04 | src/cs182_icl/data/ | tests/data/ | scripts/make_* |
| T05 | src/cs182_icl/model/, src/cs182_icl/train/, third_party/ | tests/model/, tests/train/ | scripts/train.py, scripts/bench_train.py, configs/train/ |
| T06 | src/cs182_icl/eval/ | tests/eval/ | scripts/eval_*, scripts/plot_* |
| T07 | (none) | (none) | scripts/run_all_eval.sh, results/ |

## Pilot scope per task (PILOT_SPEC)
- T01: simulate_push + §G1 mechanics checks; frames for light-slippery, middle, heavy-grippy; workers <= 16.
- T02: throughput benchmark, 500-2,000 stratified settings, ~50 at dt=0.0005, ~20 T^2 checks; R1 numbers, G3 two judgments.
- T03: oracle, analytic, physics fit required; Bayes only if R1 passes, else "blocked".
- T04: train 1,000x32, val 100x32, E1 100, E2 50, E3 50 (copies), E4 50 heavy; noise 0; train-only normalization.
- T05: one model, noise 0, seed 0; pre-training checks; budget from benchmark (~30-45 min), warmup ~5%.
- T06: MSE, standardized MSE, median relative error, tertiles; NMSE/gap only if Bayes valid; E1 first.
- T07: E1 + results/SUMMARY.md + meeting packet first; then E2, E3, E4 if resources remain.
