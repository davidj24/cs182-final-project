# Experiment protocol: in-context learning of push dynamics

Values marked "set at H2" are frozen in docs/DECISIONS.md after the simulator pilot.
Section numbers are referenced by task briefs as "PROTOCOL §N".

## 1. Question and hypotheses

**Question.** How do the amount and kind of interaction history determine how
accurately a transformer predicts a new push outcome, compared with reference
estimators that know the physics?

**Central hypothesis.** Accuracy depends on how much independent information the
context contains, not on its length.

| | Prediction |
|---|---|
| H1 | Without noise, one moving push suffices to predict a query at the same force; a query at a new force needs two distinct forces. |
| H2 | Pushes at one force with different durations never separate mass from friction; under additive noise they still improve precision. |
| H3 | Under noise, fresh measurements add information; copied tokens add none. |
| H4 | Out-of-distribution behavior is open: the transformer may follow a Bayes estimator limited by its training prior, extrapolate like a physics fit, or neither. |

## 2. Physical model

Ideal assumptions: cube starts at rest; horizontal force $F$ at the center of mass
for duration $T$; pure translation; Coulomb friction $\mu$; normal force $mg$.
With $A = 1/(2m^2\mu g)$ and $B = 1/(2m)$:

$$
d = FT^2\max(AF - B,\ 0), \qquad t_{\text{stop}} = \frac{FT}{m\mu g}\ \text{(moving pushes)}.
$$

- Line form (moving pushes): $d/(FT^2) = AF - B$. Two distinct forces fix the line:
  $m = 1/(2B)$, $\mu = 2B^2/(Ag)$.
- Same-force rule: if $F_q = F_1$, then $d_q = d_1 (T_q/T_1)^2$.
- These statements are exact for the ideal model only. Agreement with MuJoCo is
  measured (§4), not assumed.

| Quantity | Definition |
|---|---|
| Analytical prediction | The closed form above |
| Simulator oracle | MuJoCo clean outcome for the true parameters, protocol in §4 |
| Analytical discrepancy | Simulator oracle minus analytical prediction |

## 3. Parameters and actions

Training distribution (forces sampled independently of the cube):

| Quantity | Distribution |
|---|---|
| Mass $m$ | log-uniform on [0.5, 1.5] kg |
| Friction $\mu$ | uniform on [0.3, 0.6] |
| Force $F$ | uniform on [13, 26] N |
| Duration | $T = k\,\Delta t$, $k$ uniform integer in {100, ..., 200}, $\Delta t = 0.001$ s |

OOD test sets (one parameter shifted at a time):

| Set | Mass $m$ | Friction $\mu$ |
|---|---|---|
| Heavy | log-uniform [1.5, 2.0] kg | training range |
| Light | log-uniform [0.35, 0.5] kg | training range |
| Grippy | training range | uniform [0.6, 0.7] |
| Slippery | training range | uniform [0.2, 0.3] |

All test pushes move: the largest threshold in any core set is
$2.0 \times 0.6 \times 9.81 = 11.77$ N < 13 N. The light-and-slippery corner
(about 55 m) belongs to the combined-shift extension.

Analytical extremes: training $d$ from 2.0 cm to 17.3 m, $t_{\text{stop}} \le 3.5$ s;
heavy minimum 3.4 mm; grippy minimum 1.1 cm; light maximum 36.0 m (5.1 s);
slippery maximum 26.5 m (5.3 s). Report heavy near-threshold pushes separately.

Pilot (10k random pushes): histograms of $d$ and $t_{\text{stop}}$; no timeouts;
MuJoCo vs closed form (§4); some pushes near threshold. If the displacement range is
unmanageable, narrow mass before force. Final ranges: set at H2.

## 4. Simulation protocol

Scene: cube `size="0.025 0.025 0.025"` (half-lengths, 5 cm cube); plane
`friction="0 0 0"` so the cube's friction governs under MuJoCo's element-wise max;
`condim="3"`, `cone="elliptic"`, `impratio="10"`, `timestep="0.001"`, `implicitfast`.
Add `noslip_iterations` only if the pilot shows creep.

Per cube: set `body_mass` to $m$; inertia $ms^2/6$ per axis with full side
$s = 0.05$ m; cube sliding friction $\mu$; call `mj_setConst`, then `mj_resetData`;
settle under gravity for 0.5 s, confirm rest, save the state.

Per push: restore the settled state; `xfrc_applied` = $(F, 0, 0)$ for exactly $k$
steps, then zero; step until linear speed < $10^{-4}$ m/s and angular speed
< $10^{-3}$ rad/s for 50 consecutive steps (cap 15 s); record $x$ displacement from the
settled start, maximum orientation change over the trajectory (all axes), stopping
time, timeout flag. Flag > 1 degree rotation or timeouts; report counts, never drop
silently.

Validation gate (2,000 settings across training and OOD domains): median and
95th-percentile relative and absolute discrepancy; discrepancy vs $F/(\mu m g)$ and
vs $d$; rerun at $\Delta t = 0.0005$ s with $2k$ force steps and a 100-step dwell;
$T^2$ scaling at fixed cube and force; contact-friction readout.

Acceptance does not compare discrepancy with $\sigma_d$ ($\sigma_d$ can be 0). The
human decides whether the analytic model serves as the Bayes forward model with an
explicit discrepancy term (§8), and fixes $r_{\text{model}}$ and $\tau_{\text{abs}}$
from the measured discrepancy (set at H2). Otherwise a simulated table is used:
40 points in $\log m$, 40 in $\mu$, forces every 0.5 N, $k$ every 10 steps,
multilinear interpolation, error measured on 500 off-grid points; its interpolation
error then plays the role of the discrepancy.

## 5. Noise, targets, encoding

Noise: additive, $d^{\text{obs}} = d + \varepsilon$, $\varepsilon \sim \mathcal N(0, \sigma_d^2)$.
Conditions: $\sigma_d = 0$ and $\sigma_d = 0.05 \times$ median training displacement
(set at H2). Training draws fresh noise per sampled row; validation and test noise is
fixed and stored.

Measurement IDs: a copied observation keeps its ID and noisy value; a fresh
measurement of the same action gets a new ID, reuses the action's clean simulated
outcome (the simulator is deterministic), and gets new noise; a different action gets
a new ID and its own simulation.

Target: the clean simulated query displacement, in training and evaluation.
Target space: raw displacement, standardized; switch to a log target only if the pilot
shows long slides dominate the loss, decided before main runs (set at H2). If
switched, the Bayes reference becomes the posterior mean of the log target and all
metrics move to log space.

Tokens (5-dim): push $[\tilde F, \tilde T, 0, 0, 0]$; outcome
$[0, 0, \tilde d^{\text{obs}}, \widetilde{\operatorname{asinh}}(d^{\text{obs}}/0.01), 1]$.
The prediction for push $j$ is read at push token $j$, before its outcome token.
All standardization constants come from the training split only.

## 6. Datasets

| Split | Contents |
|---|---|
| Train | 10,000 cubes x 32 pushes; each row samples 11 distinct pushes in random order |
| Validation | 1,000 cubes, fixed contexts and noise |
| Test | 2,000 cubes per condition, built per §9, shared across seeds |

Cubes are disjoint across splits. Each test item stores cube parameters, query action
and clean outcome, and the context list (action, measurement ID, clean outcome, noisy
value).

## 7. Model and training

Garg et al. GPT-2 backbone: 12 layers, embedding 256, 8 heads (about 9.5M parameters);
22-token sequences; MSE averaged over all 11 push positions.
AdamW, learning rate $10^{-4}$, weight decay 0.01, batch 256, 2k warmup, cosine decay.
100k steps is a ceiling; the schedule is set from a benchmark (set at H5). Keep the
best validation checkpoint.
Runs: 2 noise conditions x 3 seeds = 6 runs, reused for all experiments.

## 8. Reference estimators

**No context.** Training-prior Bayes prediction at $c = 0$ (prior average outcome),
plus the transformer's own $c = 0$ output.

**Analytical physics fit.**
- Noise-free: least squares on $(F^2T^2,\ -FT^2)$ with at least two distinct forces;
  same-force rule for same-force queries.
- Noisy, same-force queries: $\hat h = \sum_j d_j T_j^2 / \sum_j T_j^4$ over context
  pushes at $F_q$; predict $\hat h\, T_q^2$.
- Noisy, new-force queries: report mean error only for $c \ge 4$; median at $c = 2, 3$
  (expected error is unbounded at $c = 2$ and borderline at $c = 3$ with close forces).
- Otherwise: "not determined". Least squares is the maximum-likelihood estimator
  under an unrestricted linear Gaussian model, not the minimum-MSE predictor.

**Bayes estimator (MuJoCo observations).** Observation model:
$$
d^{\text{obs}} = f(\theta, a) + \delta(a) + \varepsilon, \quad
\varepsilon \sim \mathcal N(0, \sigma_d^2), \quad
\delta(a) \sim \mathcal N(0, \tau_{\text{model}}(a)^2), \quad
\tau_{\text{model}}(a) = r_{\text{model}}\, f(\theta, a) + \tau_{\text{abs}}.
$$
$f$ is the clipped analytic model (or the simulated table if chosen at H2). $\delta$
is shared by all measurements of the same action and treated as independent across
actions (an approximation; state it in the report).

Likelihood: (1) deduplicate by `measurement_id`; (2) group the remaining observations
by identical action; (3) for a group with $n$ independent measurements and mean
$\bar y$, use $\mathcal N(\bar y;\ f(\theta, a),\ \sigma_d^2/n + \tau_{\text{model}}(a)^2)$,
including the $\theta$-dependent log-variance term. At $\sigma_d = 0$ the variance is
$\tau_{\text{model}}^2 > 0$, so the posterior is defined everywhere, including OOD.

Grid: 400 x 400 uniform in $(\log m, \mu)$; the log-uniform prior gives constant cell
weights (no extra $1/m$); log-sum-exp; prediction = posterior mean of the predicted
query outcome in the target space. Check 800 x 800 on a 200-item hard-case subset.
Cache predictions per unique action; update incrementally along context prefixes.

Training prior: log-uniform $m$, uniform $\mu$, independent (training ranges).
Broad prior (OOD only): log-uniform $m$ on [0.35, 2.0] kg, uniform $\mu$ on
[0.2, 0.7], with the clipped forward model.

**Exact analytic branches** (closed-form solve with at least two distinct forces;
single-force 1-D integral with weights $p(m, \mu(m)) \cdot 2m^2 g\,\mu(m)^2 / F^2$ and
an extra factor $m$ on a log-mass grid, where
$\mu(m) = F^2 / [2m^2 g (h + F/(2m))]$) are used ONLY on analytically generated
observations, as correctness tests and as an analytic-data sanity variant. They are
never applied to MuJoCo observations.

**Simulator oracle.** MuJoCo with true parameters; also report the analytic
prediction with true parameters to show the discrepancy floor.

## 9. Experiments

Score every prediction against the clean simulated query outcome.

**E1 Amount of context.** One 10-push context per item from the training
distribution, query from the same distribution; context for $c$ = first $c$ pushes
(paired across $c$); $c = 0, \dots, 10$; all references.

**E2 Diversity.** Per item: sample the cube, then the query ($F_q$ uniform on
[17, 22] N, $k_q$ from training); build every condition around that fixed query.

| Context type | Includes $F_q$ (same-force query) | Excludes $F_q$ (new-force query) |
|---|---|---|
| Same force, varied durations | all at $F_q$ | all at one force at least 3 N from $F_q$ |
| Distinct forces | $c$ distinct forces, one is $F_q$ | $c$ distinct forces from $[13, F_q-3] \cup [F_q+3, 26]$, spread evenly |
| Identical action | one action at $F_q$, repeated | one action at least 3 N from $F_q$, repeated |

$c \in \{1, 2, 4, 8\}$ (at $c = 1$ only the same-force cells). Same-force and
distinct-force cells share one duration multiset per item, excluding $k_q$. Every push
is a fresh measurement. Force-separation sweep (noisy models): $c = 2$,
$\Delta F \in \{1, 3, 6, 11\}$ N, random center, training-distribution queries.
Matching durations does not equalize all precision effects
($\operatorname{Var}(\varepsilon/FT^2) = \sigma_d^2/(F^2T^4)$); interpret noisy results
against Bayes on the same contexts.

**E3 Redundancy.** E2 query construction. Base: two pushes at least 6 N apart (new-force
queries: one in $[13, F_q-3]$, one in $[F_q+3, 26]$; same-force queries: $F_q$ plus a
force at least 6 N away). Append $k \in \{0, 1, 2, 4, 8\}$, alternating base pushes:
(i) copy (same ID, same noisy value); (ii) fresh measurement (same action, new ID, new
noise); (iii) same force, new duration, new ID.
Reference at $\sigma_d = 0$: fresh measurements equal copies (deterministic simulator),
so neither changes the Bayes prediction. At $\sigma_d > 0$, fresh measurements reduce
that action's variance toward $\tau_{\text{model}}^2$; copies never change it.

**E4 Out of distribution.** On all four OOD sets: full E1; E2 distinct-force/new-force
at $c = 2$ and same-force/same-force at $c = 1$; E3 kind (ii). References: training-prior
Bayes, broad-prior Bayes, analytic fit, simulator oracle.

Caveat: training contexts are independent pushes, where exact repeats and shared
forces essentially never occur, so E2 and E3 also test generalization to structured
histories.

## 10. Reference predictions

Expected risk under the stated models; finite-sample curves need not be monotone.
For noise-free conditions, "near the oracle" means near the oracle within the error
implied by the discrepancy model.

| Condition | Prediction |
|---|---|
| E1, no noise | Bayes near the oracle from $c = 2$ |
| E1, noise | Bayes error falls with $c$, stays above the oracle |
| E2, same-force query | Near the oracle from $c = 1$ |
| E2, same force + new-force query | Ambiguous at any $c$; error set by the prior |
| E2, distinct forces + new-force query | Near the oracle at $c = 2$ (no noise); under noise worsens as $\Delta F$ shrinks |
| E2, identical action | Like same force, without the precision from varying duration |
| E3 (i) copies | No change from $k = 0$ |
| E3 (ii), (iii) with noise | Improvement only along already-measured directions, floored by $\tau_{\text{model}}$ |
| E4 | Analytic fit extrapolates; training-prior Bayes biased toward training range; transformer open |

## 11. Metrics and statistics

Primary: MSE vs the clean simulated target; NMSE = MSE / D with D the training-prior
Bayes $c = 0$ error on the same test set (shared by all methods, including OOD);
absolute gap $\text{MSE}_{\text{model}} - \text{MSE}_{\text{Bayes}}$.
Secondary: median relative error; MSE by displacement tertile; heavy near-threshold
subset.
Statistics: mean +- SE over 3 seeds; paired differences on shared items; training and
validation curves for every run. If the target moves to log space, all metrics and
references move with it.

## 12. Extensions (only after the core)

Sometimes-stationary pushes (piecewise model, inequality likelihood); structured-context
training mixture; task-diversity sweep (1k, 10k, 100k cubes; optional comparison with
Bayes restricted to training cubes if it differs materially); linear probes for
$\log m$ and $\mu$; more noise levels; combined OOD shifts; query-force extrapolation.

## 13. Milestones

1. Simulator: scene, pilot, validation gate, rollout cost; freeze ranges, $\sigma_d$,
   target space, forward model, $r_{\text{model}}$, $\tau_{\text{abs}}$.
2. References: analytic-data exact checks; MuJoCo-mode Bayes; copies leave Bayes
   unchanged; grid convergence.
3. Data: training, validation, E1–E4 test sets with stored noise and measurement IDs.
4. One end-to-end model (no noise, seed 0): training converges, error falls with
   context, evaluation correct. Closeness to Bayes is a result, not a gate.
5. Main runs: benchmark, set budget, train 6 models, run E1–E4 from one recorded commit.
