# Pilot specification: initial_e2e_2026_11_16 (version 1)

Source: the user's meeting-pilot specification of 2026-11-16, saved verbatim in sections A-R below
except for the "Revisions and recorded assumptions" block, which is the only editorial addition.
PROTOCOL.md stays the long-term specification and is not modified. All pilot reductions live here
and in configs/pilot/initial_e2e_2026_11_16.json (created by T00).

## Precedence (every worker reads this file together with its task brief)
1. Where this spec and a task brief (docs/tasks/Txx) or PROTOCOL disagree on scope, counts,
   budgets, or deliverables, THIS SPEC WINS for the pilot. Full-campaign deliverables not
   needed by the pilot (noisy conditions, seeds 1-2, E2 sweeps, table fallback, four OOD sets,
   10k-cube datasets, 100k steps) are out of scope tonight.
2. Correctness requirements are not reducible: never loosen a test or a scientific threshold.
3. Bayes is optional. Incomplete Bayes never blocks MuJoCo data generation, training, or E1.
4. If something needs a decision this spec does not make, write it in docs/questions/Txx.md and stop.

## Revisions and recorded assumptions (editorial, 2026-11-16 ~20:00 PST)
- COMPLETION-BASED GOAL (user, replaces section P timing as a stopping rule): finish one validated,
  reproducible pilot training run, evaluate E1, produce its plots, CSVs and the meeting summary.
  Do not stop because of a clock time. Section P times are scheduling targets only. If finishing
  would exceed a section D limit, report progress and ASK before exceeding it.
- Order: finish and save the E1 packet and results/SUMMARY.md first; then run small E2, E3, E4 if
  resources remain; update the summary afterwards. An invalid simulator still blocks training.
- Assumption A1 (user declined the question; recommended default adopted): Bayes validity test is
  PREAPPROVAL R1 (median rel <= 1%, p95 <= 3% between MuJoCo and analytic over training-domain
  moving pushes, plus dt-halving change median <= 0.5%, p95 <= 2%). Failure => "Bayes blocked",
  pilot continues. The table fallback is not built tonight.
- Assumption A2: ranges in section C are frozen. Rotation-flagged or timed-out rollouts are kept,
  counted and reported, never dropped. Only broken physics (friction, force timing, settling,
  mass/inertia) stops data generation. Target space is raw standardized; a log target is not
  adopted tonight (it may be reported as a diagnostic only).
- Assumption A4 (user decision, 2026-11-16, supersedes the earlier translation-only-contact choice): the pilot simulator is an
  EXPLICITLY IDEALIZED 1D TRANSLATIONAL model implemented in MuJoCo: an x slide joint with the cube's mass, MuJoCo dry joint
  frictionloss = mu*m*g (assumes N = m*g), external push F for exactly k steps, floor/contact visual only. If frictionloss is
  unsuitable, an explicit Coulomb friction force inside the MuJoCo stepping loop is used. Reason (finding for the meeting): realistic
  MuJoCo contact under the attempted free-cube and constrained setups does not satisfy N = m*g (the cone solver raises N above m*g
  whenever F > mu*N; free cube hops/tumbles; armature-1000 variant agrees only through contact chatter with N spikes ~100 m*g and
  was rejected). The G1 contact-friction items are replaced by checks of the implemented ideal law (frictionloss readout = mu*m*g,
  push acceleration (F - mu*m*g)/m, coast deceleration -mu*g, displacement vs closed form, T^2 scaling, force duration, dt
  sensitivity, below-threshold stiction). max_rotation_deg is vacuous (0). The simulator is an implementation of the declared ideal
  dynamics, NOT an independent validation of MuJoCo contact physics; SUMMARY must say so. Realistic contact is a later relaxation.
  FALLBACK (user-authorized): if the third T01 repair cycle fails, stop simulator repair and generate pilot data directly from the
  analytic closed form; no fourth cycle.
- Assumption A3: dollar/agent spend cannot be measured and is NOT enforced; launches and GPU-hours are.

---

# Tonight's goal

Build a **small, scientifically defensible end-to-end pilot** for tomorrow's project meeting.

Do not attempt the full research campaign tonight.

The goal is to leave tonight with:

1. A validated MuJoCo push simulator.
2. One reproducible no-noise training dataset.
3. One trained transformer, seed 0.
4. A complete preliminary E1 result.
5. Small but interpretable E2, E3, and E4 results if the pipeline is healthy.
6. A meeting-ready evidence packet explaining what we learned and what experiment should come next.

Use the full protocol as the long-term scientific specification. Put all pilot-specific reductions in a separate pilot configuration; do not silently modify the full protocol.

---

# A. Scientific question for tonight

The overall project asks:

> Does the transformer benefit from the **information content** of a push history, or merely from receiving a longer context?

Tonight should establish the first evidence for the hierarchy:

- **E1:** Does more context help at all?
- **E2:** Does informative force diversity help more than uninformative context?
- **E3:** Does simply duplicating evidence change the transformer's prediction?
- **E4:** Does the learned behavior survive a simple physical distribution shift?

This is preliminary single-seed evidence, not a final statistical conclusion.

---

# B. Hard priority order

If time becomes tight, preserve this order:

1. Environment verified.
2. Minimal interfaces needed by the pilot.
3. Simulator physically validated.
4. Pilot dataset generated reproducibly.
5. One transformer trained successfully.
6. E1 evaluated and plotted.
7. Meeting summary completed.
8. E2.
9. E3.
10. E4.

Reduce item counts before reducing numerical correctness.

Do not sacrifice E1 or the final summary in order to finish optional E3/E4.

Do not spend tonight implementing unused features from the full T00-T07 campaign.

Preserve compatible interfaces so the pilot can be scaled later.

---

# C. Pilot configuration

Use:

```text
pilot name: initial_e2e_2026_11_16
training noise: 0
training seed: 0
measurement noise: 0
target: clean MuJoCo displacement
primary target space: raw standardized displacement
```

Unless simulator validation demonstrates a concrete reason to change them, start with:

```text
m ~ log-uniform [0.5, 1.5] kg
mu ~ uniform [0.3, 0.6]
F ~ uniform [13, 26] N
k ~ uniform integer {100,...,200}
dt = 0.001 s
T = k * dt
```

Every action must be sampled independently of hidden cube parameters.

All training/validation/test cubes must be disjoint.

All normalization statistics must come from training data only.

---

# D. Resource and safety limits

For this pilot:

```text
GPU runs: one training run at a time
total GPU budget: <= 2 device-hours
CPU simulation workers: <= 16 and reduce if machine is busy
new pilot outputs: <= 10 GB
worker/reviewer launches: <= 24
repair cycles per task: <= 2
```

Never:

- push to the remote;
- modify teammates' branches;
- merge into `main` or a teammate branch;
- overwrite old experiment data;
- stop unrelated processes;
- run a plain `uv sync` that could replace the working CUDA-compatible PyTorch;
- loosen tests or scientific thresholds to finish faster.

Local integration into the designated pilot/integration branch is allowed only for a task commit whose exact SHA has passed its required tests and review with no unresolved critical issue.

Record every integrated SHA.

---

# E. Phase 0 - environment verification

Before changing code, record:

```text
git SHA
git status
Python version
uv version
MuJoCo version
PyTorch version
torch.version.cuda
torch.cuda.is_available()
GPU count and GPU names
nvidia-smi output
available disk space
```

Run a tiny GPU forward/backward test.

Verify both RTX 4090s are visible, but use only one GPU for the pilot training run unless there is a specific reason otherwise.

Do not modify the working CUDA/PyTorch setup if it already passes the GPU test.

Save this under the pilot results directory.

---

# F. Phase 1 - minimal T00

Implement only the interfaces required for tonight's pipeline.

Required core types:

```text
Cube(m, mu, cube_id)
Action(F, k)
Observation(action, measurement_id, d_clean, d_obs)
PushResult(d, t_stop, max_rotation_deg, timed_out)
TestItem(cube, query, d_query_clean, context, condition)
```

Required functions/interfaces:

```text
analytic_displacement(...)
simulate_push(...)
bayes_predict(...)
physics_fit_predict(...)
model_predict(...)
```

Required analytical tests include:

```text
closed-form displacement
clipping below the motion threshold
same-force duration rule
line-form identities
```

Do not build unrelated infrastructure tonight.

---

# G. Phase 2 - simulator validation

## G1. Physical simulator checks

Before producing training data, verify:

- Cube side length is 0.05 m.
- MuJoCo box half-size is 0.025 m.
- Mass is updated correctly.
- Inertia is \(ms^2/6\) on every axis with \(s=0.05\) m.
- `mj_setConst` is called after changing mass/inertia.
- Floor friction does not override cube friction.
- Contact friction readout equals the intended cube coefficient.
- Cube settles before every push.
- Force is applied for exactly `k` simulator steps.
- Force is then zeroed.
- Zero-force creep is negligible.
- Linear and angular stopping conditions are checked.
- Maximum rotation over the complete trajectory is recorded.
- Timeouts are recorded.
- No flagged rollout is silently discarded.

Save frames or a short replay for three representative cases:

```text
light / slippery
middle
heavy / grippy
```

## G2. Analytic comparison

First benchmark simulator throughput.

Then run a stratified validation sample.

Target:

```text
500-2,000 simulator settings
```

Use 500 as the minimum meeting-pilot target and increase toward 2,000 only if measured runtime permits.

Also run:

```text
~50 settings at dt = 0.0005 s
~20 controlled T^2-scaling checks
```

Produce:

```text
simulator displacement vs analytical displacement scatter
absolute discrepancy histogram
relative discrepancy histogram
error vs F/(mu*m*g)
rotation/timeout counts
simulation seconds per rollout
```

## G3. Two separate decisions

Make two distinct judgments.

### Physical simulator validity

If force timing, friction, settling, rotation, or other simulator mechanics are wrong:

**stop dependent data generation and training.**

Fix the simulator first.

### Analytical-model validity

If the simulator is physically valid but differs systematically from the closed form:

do **not** call the analytical model an exact simulator oracle.

Training may still continue on valid MuJoCo data, but the result packet must distinguish:

```text
MuJoCo clean target / simulator oracle
analytic true-parameter prediction
analytic discrepancy
```

Use an analytical Bayes reference on MuJoCo data only if the pilot has established and recorded a defensible simulator-discrepancy likelihood.

Do not fabricate such a likelihood simply to obtain a Bayes curve.

---

# H. Phase 3 - pilot data

## H1. Target sizes

Start with:

```text
Training:
1,000 cubes x 32 simulated pushes

Validation:
100 cubes x 32 simulated pushes
fixed validation examples

E1:
100 test cubes

E2:
50 paired test items

E3:
50 paired test items

E4:
50 heavy-OOD test cubes
```

The test sets must be shared across all compared methods.

If simulation throughput makes this infeasible, reduce in this order:

```text
E4: 50 -> 30
E3: 50 -> 30
E2: 50 -> 30
E1: 100 -> 50
training: 1000 -> 500 cubes
```

Do not go below 500 training cubes or 50 E1 items without recording that the pilot is severely underpowered.

Preserve enough pushes per training cube to sample an 11-push sequence.

## H2. Reproducibility

Store:

```text
cube IDs
actions
clean displacement
measurement IDs
split assignment
generation seed
dataset SHA/hash
normalization statistics
simulator/source SHA
```

No test statistic may influence training normalization.

---

# I. Phase 4 - reference methods

Tonight's required references are:

```text
1. Simulator oracle
2. Analytical true-parameter prediction
3. Analytical physics fit where determined
4. Training-prior Bayes, if its MuJoCo likelihood is scientifically validated
```

## I1. Simulator oracle

The target itself is the clean deterministic MuJoCo query outcome at the true cube parameters.

This is the actual prediction ceiling for the simulator task.

## I2. Analytical prediction

Also evaluate the analytical formula at the true \(m,\mu\).

The difference between it and MuJoCo explicitly shows simulator discrepancy.

## I3. Physics fit

For moving pushes:

\[
d = A F^2T^2-BFT^2.
\]

For two or more sufficiently distinct force levels, fit \(A,B\) and predict the query.

For a query at a force already observed in context, the same-force duration relation may be used.

Return "not determined" rather than inventing a pseudoinverse prediction when the information is insufficient.

## I4. Bayes

If the analytic forward model has been accepted for MuJoCo inference with a recorded discrepancy model, implement the training-prior Bayes reference.

Use the existing protocol's prior and numerical-convergence checks.

For the meeting pilot:

- keep numerical accuracy;
- reduce the number of test items before coarsening the posterior enough to change conclusions;
- validate the pilot grid against a finer grid on a small hard-case subset;
- cache reference results.

If Bayes cannot be made scientifically valid tonight, **do not block the entire pilot**.

Report:

```text
Bayes reference: incomplete / blocked
reason: ...
```

and retain simulator oracle + physics + analytical prediction.

The meeting should not show an invalid Bayes curve.

---

# J. Phase 5 - transformer training

Train exactly one model:

```text
noise condition = 0
seed = 0
```

Use the protocol architecture:

```text
GPT-2-style causal transformer
12 layers
embedding dimension 256
8 attention heads
~9.5M parameters
22-token maximum sequence
```

Each row contains alternating push/outcome tokens.

Read each displacement prediction from its push token, before that push's outcome is visible.

Use the protocol encoding.

## J1. Required pre-training checks

Before the real run:

### Shape test
Verify expected batch, sequence and output shapes.

### Causal test
Changing a future outcome must not change an earlier prediction.

### Normalization test
Verify training-only normalization.

### Leakage test
The current query outcome cannot appear anywhere visible to the model before its prediction.

### Fixed-small-dataset overfit
Use a fixed diagnostic set.

Confirm the model can drive its training loss very low.

This tests optimization/plumbing; it is not evidence of ICL.

## J2. Training budget

Benchmark several hundred steps first.

Choose a step budget that gives approximately:

```text
30-45 minutes of real training
```

while keeping enough time for evaluation.

Do not require the full protocol's step count.

Use:

```text
AdamW
lr = 1e-4
weight decay = 0.01
batch = 256
```

For the shortened run, set warmup to approximately 5% of the pilot step budget, with a sensible cap, rather than blindly using a 2,000-step warmup when the entire run is short.

Use cosine decay over the actual pilot budget.

Evaluate validation frequently enough to save the best checkpoint.

Save:

```text
best checkpoint
last checkpoint
train loss curve
validation loss curve
step count
tokens/s
wall-clock time
GPU time
configuration
source SHA
dataset version
```

Do not alter training to force the expected E1 curve.

---

# K. Phase 6 - E1, required

This is tonight's primary scientific result.

## K1. Construction

Use 100 test cubes if possible.

For every item:

1. Sample one cube.
2. Sample ten independently distributed context pushes.
3. Sample one independent query push.
4. Generate all clean MuJoCo outcomes.
5. Use the same ordered ten-push context for every context length.

Evaluate:

```text
c = 0, 1, 2, 4, 8, 10
```

The context at `c` is the first `c` observations of the ten-push list.

Therefore comparisons across context lengths are paired.

## K2. Compare

Plot:

```text
Transformer
training-prior Bayes, if valid
physics fit where defined
analytic true-parameter prediction
simulator oracle
```

## K3. Metrics

At minimum report:

```text
MSE against clean MuJoCo query displacement
standardized MSE
median relative error
error by displacement tertile
```

If the Bayes reference is valid, also report the protocol's shared-denominator NMSE and gap to Bayes.

Do not normalize each method with a different denominator.

## K4. Required E1 figure

One meeting-ready figure:

```text
x-axis: context length
y-axis: prediction error
methods: transformer + valid references
```

Also save the underlying CSV.

Print approximately ten item-level examples:

```text
query F,T
true displacement
transformer prediction
Bayes prediction if available
physics-fit prediction
```

## K5. Question answered

E1 asks:

> Does the transformer actually use previous interactions?

A decreasing curve is evidence that it does.

A flat or strange curve is also a result, once implementation correctness has been checked.

---

# L. Phase 7 - E2, targeted and important

E2 is the highest-priority result after E1 because it directly tests the central hypothesis.

Use approximately 50 items, paired across conditions.

For each item:

1. Sample the cube.
2. Sample one fixed query first.
3. Build every context condition around the same cube and same query.

Use:

```text
F_q in roughly [17,22] N
query duration from the training duration distribution
```

## L1. Four core cells

### Cell A - same-force history, familiar-force query

Context force:

```text
F = F_q
```

Vary durations.

Evaluate at:

```text
c = 1, 2, 4
```

Ideal no-noise prediction: one observation is enough for the same-force duration relation.

### Cell B - same-force history, new-force query

Choose one context force at least 3 N from \(F_q\).

All context pushes use that force with varying durations.

Evaluate:

```text
c = 2, 4
```

This context can become longer without identifying both hidden physical parameters.

### Cell C - distinct-force history, familiar-force query

Use distinct forces and ensure one is \(F_q\).

Evaluate:

```text
c = 2, 4
```

### Cell D - distinct-force history, new-force query

Use distinct forces, none within 3 N of \(F_q\).

Evaluate:

```text
c = 2, 4
```

## L2. Pairing

For the corresponding same-force and distinct-force comparisons:

- use the same cube;
- same query;
- same number of context observations;
- same duration multiset where possible.

Randomize context order.

## L3. Main E2 figure

The useful comparison is:

```text
same-force context
vs
distinct-force context
```

for:

```text
familiar-force query
new-force query
```

This directly asks whether **informative diversity**, rather than token count, explains performance.

---

# M. Phase 8 - E3, minimal no-noise version

Do not pretend tonight's zero-noise pilot tests the full E3 hypothesis.

With zero measurement noise:

```text
fresh measurement of identical action
and
copied observation
```

have the same numeric action/outcome tokens.

Therefore tonight E3 tests only:

> Is the transformer robust to redundant duplicated context?

## M1. Construction

Use approximately 50 items.

Start from an informative base context:

```text
two distinct forces
force separation >= 6 N
```

Use a new-force query.

Append:

```text
k = 0, 1, 2, 4
```

copies of existing observations.

A copy must retain the identical:

```text
action
outcome
measurement ID
```

## M2. Reference behavior

After measurement-ID deduplication:

```text
Bayes prediction should be unchanged
physics information should be unchanged
```

The transformer may or may not change because the sequence itself is longer.

That is the result.

## M3. Figure

Plot transformer error or prediction change versus number of appended copies.

Label clearly:

> Preliminary no-noise redundancy test. This does not test the benefit of independent noisy repeated measurements.

The full noisy E3 experiment comes later.

---

# N. Phase 9 - E4, minimal OOD

Use one OOD direction only tonight:

```text
Heavy cubes:
m ~ log-uniform [1.5, 2.0] kg
mu remains in the training range
```

Use approximately 50 cubes.

Do not retrain the transformer.

## N1. Initial E4 evaluation

Use an E1-style context for each heavy cube and evaluate:

```text
c = 0, 2, 10
```

If inexpensive, also evaluate the E2 distinct-force/new-force cell at `c=2`.

## N2. References

Use:

```text
simulator oracle
analytic true-parameter prediction
physics fit
training-prior Bayes if valid
broad-prior Bayes if valid
```

The important comparison is whether the transformer's gain from context persists when mass lies outside its training range.

## N3. Interpretation

Do not conclude "generalizes" from 50 cubes and one seed.

Label it:

> preliminary heavy-OOD behavior.

---

# O. Evaluation correctness checks

Before interpreting any plot:

1. Feed known predictions into the metric code and verify hand-calculated MSE.
2. Run the Bayes-as-model harness sanity check if Bayes is implemented.
3. Verify every method receives the same query items.
4. Verify every score uses the clean MuJoCo target.
5. Verify context prefixes are paired.
6. Verify no test normalization statistic entered training.
7. Verify model checkpoint SHA/config/dataset match the reported run.

Unexpected outcomes should trigger verification, not automatic "correction."

---

# P. Tonight's timing targets

Use these as scheduling targets, not scientific rules. (Revised: not a stopping rule; see the
completion-based goal above.)

```text
~19:50-20:10
environment + minimal T00

~20:10-20:50
in parallel:
- simulator implementation
- analytic/reference infrastructure
- transformer/training infrastructure

~20:50-21:15
simulator checks + validation + freeze pilot settings

~21:15-21:40
pilot dataset generation
finalize references
finalize real-data training input

~21:40-22:30
training benchmark + one real training run

~22:30-23:10
E1 first
then E2
then small E3
then small E4

~23:10-23:59
final plots
CSV verification
results/SUMMARY.md
meeting packet
```

If behind schedule:

1. Reduce test counts.
2. Reduce E4.
3. Reduce E3.
4. Reduce E2 secondary cells.
5. Never remove E1 or the summary.

Do not shorten simulator correctness checks enough to make the data scientifically questionable.

---

# Q. Meeting packet

Produce one directory containing:

## 1. Reproducibility

```text
source SHA
dataset versions
model config
checkpoint
environment versions
exact commands
actual runtime
GPU time
CPU worker count
```

## 2. Simulator evidence

```text
representative frames/replay
sim-vs-analytic scatter
discrepancy summary
rotation/timeouts
```

## 3. Training evidence

```text
training loss
validation loss
best checkpoint step
```

## 4. E1

Required:

```text
error vs context curve
metric CSV
item-level examples
```

## 5. E2

If completed:

```text
same-force vs distinct-force comparison
familiar vs new query comparison
```

## 6. E3

If completed:

```text
duplicate-context robustness plot
```

with the no-noise limitation stated explicitly.

## 7. E4

If completed:

```text
in-distribution vs heavy-OOD comparison
```

labeled preliminary.

## 8. `results/SUMMARY.md`

Write this for tomorrow's discussion, not as a final paper.

Use these sections:

### What we ran
Exact data size, checkpoint, steps, conditions, and references.

### What is trustworthy
Which simulator/reference/training checks passed.

### Preliminary findings
Only claims supported directly by saved results.

### What surprised us
Unexpected behavior that survived a correctness check.

### Limitations
At least:

```text
one training seed
small training set
short training run
no observation-noise model
small E2/E3/E4 test sets
structured-context distribution shift
```

### What should we do next?
Rank the next 3 experiments by scientific value.

For each recommendation state:

```text
question
why today's result motivates it
experiment needed
expected discriminating result
```

---

# R. What counts as success tonight

Success does **not** mean:

```text
transformer reaches Bayes
E1 has the expected shape
E2 proves the hypothesis
all E1-E4 boxes are checked
```

Success means:

```text
the simulator is trustworthy enough to generate data;
one complete training run is reproducible;
the evaluation pipeline works end to end;
E1 produces interpretable evidence;
and we have enough preliminary E2/E3/E4 evidence to decide what is worth scaling tomorrow.
```

The priority is:

> **a small result that changes tomorrow's research discussion, not a rushed miniature final paper.**
