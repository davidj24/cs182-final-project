# ledger for pilot initial_e2e_2026_11_16 (append-only; written by ledger.sh)
NOTE pilot approved 2026-11-16; setup/pilot integrated as 5d9da70 (spec SHA 1fb6618)
# 2026-11-16 20:03:45 PST  NOTE pilot approved 2026-11-16; setup/pilot integrated as 5d9da70 (spec SHA 1fb6618)
INTEGRATED setup 5d9da70
# 2026-11-16 20:03:45 PST  INTEGRATED setup 5d9da70
JOBSTART test_job 1794888236 0.01
# 2026-11-16 20:03:56 PST  JOBSTART test_job 1794888236 0.01
JOBEND test_job 0.0011
# 2026-11-16 20:04:00 PST  JOBEND test_job 0.0011
NOTE installed mujoco 4.0.1 + pytest 9.2.0 into shared env via uv pip (torch 2.14.1+cu126 unchanged, GPU test passes); uv.lock not updated (T00 adds deps to pyproject)
# 2026-11-16 20:04:46 PST  NOTE installed mujoco 4.0.1 + pytest 9.2.0 into shared env via uv pip (torch 2.14.1+cu126 unchanged, GPU test passes); uv.lock not updated (T00 adds deps to pyproject)
LAUNCH writer-opus T00
# 2026-11-16 20:05:00 PST  LAUNCH writer-opus T00
NOTE T00 launched: plan committed first (docs/plans/T00.md) then implementation; fresh reviewer-opus reviews plan+code together at the SHA (saves launches: cap 24)
# 2026-11-16 20:05:00 PST  NOTE T00 launched: plan committed first (docs/plans/T00.md) then implementation; fresh reviewer-opus reviews plan+code together at the SHA (saves launches: cap 24)
LAUNCH reviewer-opus T00
# 2026-11-16 20:08:50 PST  LAUNCH reviewer-opus T00
INTEGRATED T00 23a088c
# 2026-11-16 20:10:31 PST  INTEGRATED T00 23a088c
NOTE soomi-integration now 561b21c after T00 merge
# 2026-11-16 20:10:32 PST  NOTE soomi-integration now 561b21c after T00 merge
LAUNCH writer-sonnet T01
# 2026-11-16 20:11:17 PST  LAUNCH writer-sonnet T01
LAUNCH writer-sonnet T05
# 2026-11-16 20:11:17 PST  LAUNCH writer-sonnet T05
LAUNCH writer-sonnet T06
# 2026-11-16 20:11:17 PST  LAUNCH writer-sonnet T06
LAUNCH writer-opus T03
# 2026-11-16 20:11:17 PST  LAUNCH writer-opus T03
NOTE parallel wave 1 launched after T00 integration 561b21c: T01 sim, T05 train, T06 eval, T03 ref (oracle/analytic/physics-fit only; Bayes deferred until R1)
# 2026-11-16 20:11:17 PST  NOTE parallel wave 1 launched after T00 integration 561b21c: T01 sim, T05 train, T06 eval, T03 ref (oracle/analytic/physics-fit only; Bayes deferred until R1)
LAUNCH reviewer-opus T03
# 2026-11-16 20:14:33 PST  LAUNCH reviewer-opus T03
DECISION physics-fit default fit space = regress d on (F^2T^2, -FT^2) per PILOT_SPEC I3/PROTOCOL 8; line-form d/(FT^2) kept as fit_space='line' sensitivity only; T06 must report not_determined coverage per c
# 2026-11-16 20:14:33 PST  DECISION physics-fit default fit space = regress d on (F^2T^2, -FT^2) per PILOT_SPEC I3/PROTOCOL 8; line-form d/(FT^2) kept as fit_space='line' sensitivity only; T06 must report not_determined coverage per c
LAUNCH reviewer-opus T01
# 2026-11-16 20:16:36 PST  LAUNCH reviewer-opus T01
NOTE T01 writer reports solimp=0.99 0.999 0.001 added to cube.xml (default solimp sag 0.108mm > 1e-4 tol) and COM-push pitching up to 8.8deg (flagged, kept); T02 must confirm solimp effect on d
# 2026-11-16 20:16:36 PST  NOTE T01 writer reports solimp=0.99 0.999 0.001 added to cube.xml (default solimp sag 0.108mm > 1e-4 tol) and COM-push pitching up to 8.8deg (flagged, kept); T02 must confirm solimp effect on d
INTEGRATED T03 49c15e1
# 2026-11-16 20:16:52 PST  INTEGRATED T03 49c15e1
NOTE soomi-integration 795b326; T03 minors M1,M4,M5 deferred (no repair cycle used); T06 must report not_determined coverage + unphysical-fit flags per c
# 2026-11-16 20:16:52 PST  NOTE soomi-integration 795b326; T03 minors M1,M4,M5 deferred (no repair cycle used); T06 must report not_determined coverage + unphysical-fit flags per c
LAUNCH reviewer-opus T06
# 2026-11-16 20:17:54 PST  LAUNCH reviewer-opus T06
NOTE T06 assumptions to propagate: meta.json normalization.d_obs.std (T04), --model-loader module:function returning (model,norm) (T05)
# 2026-11-16 20:17:54 PST  NOTE T06 assumptions to propagate: meta.json normalization.d_obs.std (T04), --model-loader module:function returning (model,norm) (T05)
REPAIR T01
# 2026-11-16 20:20:38 PST  REPAIR T01
LAUNCH writer-sonnet T01
# 2026-11-16 20:20:38 PST  LAUNCH writer-sonnet T01
NOTE T01 CRITICAL C1 (hopping/tumbling) -> PILOT_SPEC G3: dependent data generation and training BLOCKED; repair cycle 1/2 = contact-settings search keeping dt/geometry/friction; fallback (slide-joint translation-only) is a PROTOCOL 4 change => ask user
# 2026-11-16 20:20:38 PST  NOTE T01 CRITICAL C1 (hopping/tumbling) -> PILOT_SPEC G3: dependent data generation and training BLOCKED; repair cycle 1/2 = contact-settings search keeping dt/geometry/friction; fallback (slide-joint translation-only) is a PROTOCOL 4 change => ask user
REPAIR T06
# 2026-11-16 20:21:04 PST  REPAIR T06
LAUNCH writer-sonnet T06
# 2026-11-16 20:21:04 PST  LAUNCH writer-sonnet T06
NOTE T06 repair cycle 1/2: 4 MAJOR (see review.md)
# 2026-11-16 20:21:04 PST  NOTE T06 repair cycle 1/2: 4 MAJOR (see review.md)
LAUNCH reviewer-opus T06
# 2026-11-16 20:24:26 PST  LAUNCH reviewer-opus T06
LAUNCH reviewer-opus T05
# 2026-11-16 20:27:09 PST  LAUNCH reviewer-opus T05
INTEGRATED T06 babf15b
# 2026-11-16 20:27:32 PST  INTEGRATED T06 babf15b
NOTE soomi-integration adf7910; T06 minors deferred: cell-B single-force check (m1), model input must mask cube m/mu (m4), pin npz-hash key after T04 (m7)
# 2026-11-16 20:27:32 PST  NOTE soomi-integration adf7910; T06 minors deferred: cell-B single-force check (m1), model input must mask cube m/mu (m4), pin npz-hash key after T04 (m7)
STOP T01 repair 1/2 failed: no contact/solver setting removes hop/tumble (30 candidates; hop persists with zero applied force: soft-contact friction/normal coupling). Remaining options change PROTOCOL 4 scene (A translation-only, C pin z only, B accept rotation flags ~91% of pushes). T02/T04 blocked. Awaiting user decision. Writer branch exp/sim 5e0ae3f NOT integrated (5 strict xfail flatness tests).
# 2026-11-16 20:27:44 PST  STOP T01 repair 1/2 failed: no contact/solver setting removes hop/tumble (30 candidates; hop persists with zero applied force: soft-contact friction/normal coupling). Remaining options change PROTOCOL 4 scene (A translation-only, C pin z only, B accept rotation flags ~91% of pushes). T02/T04 blocked. Awaiting user decision. Writer branch exp/sim 5e0ae3f NOT integrated (5 strict xfail flatness tests).
INTEGRATED T05 077405b
# 2026-11-16 20:31:26 PST  INTEGRATED T05 077405b
NOTE soomi-integration e31cae4; T05 minors deferred; before real run: set steps from bench on launch GPU, 2k-step run on real data, add load_run helper + real-data meta guards
# 2026-11-16 20:31:26 PST  NOTE soomi-integration e31cae4; T05 minors deferred; before real run: set steps from bench on launch GPU, 2k-step run on real data, add load_run helper + real-data meta guards
DECISION user chose option A (translation-only scene) for simulator; PROTOCOL 4 deviation recorded in PILOT_SPEC A4 and DECISIONS; T01 repair cycle 2/2 (last)
# 2026-11-16 20:45:05 PST  DECISION user chose option A (translation-only scene) for simulator; PROTOCOL 4 deviation recorded in PILOT_SPEC A4 and DECISIONS; T01 repair cycle 2/2 (last)
REPAIR T01
# 2026-11-16 20:45:05 PST  REPAIR T01
LAUNCH writer-sonnet T01
# 2026-11-16 20:45:05 PST  LAUNCH writer-sonnet T01
STOP T01 repair 2/2 exhausted (ea06492, evidence only, no scene change): translation-only options (x,y slides: N=0 so no friction; x,y,z slides: contact dropouts, N/mg 0..110) fail; root cause = MuJoCo cone solver raises N above mg whenever F>mu*N (first-step N/mg 1.593 for m=1,mu=.45,F=20, matches (mg+mu F)/(1+mu^2)). Only armature-1000 on z keeps cube flat and d~analytic (train median 3.7e-3, p95 2.1e-2; heavy OOD median 9.9e-3; dt-halving 2.9e-3) but via 9-20% contact duty with N spikes ~100 mg. Needs user decision; a third T01 cycle also exceeds the 2-repair cap. Launches 14/24.
# 2026-11-16 20:49:20 PST  STOP T01 repair 2/2 exhausted (ea06492, evidence only, no scene change): translation-only options (x,y slides: N=0 so no friction; x,y,z slides: contact dropouts, N/mg 0..110) fail; root cause = MuJoCo cone solver raises N above mg whenever F>mu*N (first-step N/mg 1.593 for m=1,mu=.45,F=20, matches (mg+mu F)/(1+mu^2)). Only armature-1000 on z keeps cube flat and d~analytic (train median 3.7e-3, p95 2.1e-2; heavy OOD median 9.9e-3; dt-halving 2.9e-3) but via 9-20% contact duty with N spikes ~100 mg. Needs user decision; a third T01 cycle also exceeds the 2-repair cap. Launches 14/24.
DECISION USER: Option 2 (idealized 1D translational sim, frictionloss=mu*m*g preferred, else explicit Coulomb) + third T01 repair cycle authorized (REPAIR_CAP raised 2->3 by user authorization); no 4th cycle: on failure switch pilot to direct analytic data generation. Armature-1000 rejected. Contact failure is a meeting finding.
# 2026-11-16 20:54:51 PST  DECISION USER: Option 2 (idealized 1D translational sim, frictionloss=mu*m*g preferred, else explicit Coulomb) + third T01 repair cycle authorized (REPAIR_CAP raised 2->3 by user authorization); no 4th cycle: on failure switch pilot to direct analytic data generation. Armature-1000 rejected. Contact failure is a meeting finding.
REPAIR T01
# 2026-11-16 20:54:51 PST  REPAIR T01
LAUNCH writer-sonnet T01
# 2026-11-16 20:54:51 PST  LAUNCH writer-sonnet T01
LAUNCH reviewer-opus T01
# 2026-11-16 20:58:12 PST  LAUNCH reviewer-opus T01
NOTE T01 cycle 3 result f28ff1e: explicit Coulomb force in qfrc_applied (frictionloss creeps below threshold); sim-vs-analytic ~1e-6 (implementation check only); 123 passed 1 skipped
# 2026-11-16 20:58:12 PST  NOTE T01 cycle 3 result f28ff1e: explicit Coulomb force in qfrc_applied (frictionloss creeps below threshold); sim-vs-analytic ~1e-6 (implementation check only); 123 passed 1 skipped
INTEGRATED T01 f28ff1e
# 2026-11-16 21:01:11 PST  INTEGRATED T01 f28ff1e
NOTE soomi-integration 342cabe; T01 minors deferred (stale cube.xml comment, t_stop semantics doc, sim_validate.py superseded by T02)
# 2026-11-16 21:01:11 PST  NOTE soomi-integration 342cabe; T01 minors deferred (stale cube.xml comment, t_stop semantics doc, sim_validate.py superseded by T02)
LAUNCH writer-sonnet T02
# 2026-11-16 21:01:41 PST  LAUNCH writer-sonnet T02
LAUNCH writer-sonnet T04
# 2026-11-16 21:01:41 PST  LAUNCH writer-sonnet T04
NOTE wave 2 launched on soomi-integration 342cabe: T02 (sim worktree) and T04 (data worktree)
# 2026-11-16 21:01:41 PST  NOTE wave 2 launched on soomi-integration 342cabe: T02 (sim worktree) and T04 (data worktree)
LAUNCH reviewer-opus T02
# 2026-11-16 21:05:11 PST  LAUNCH reviewer-opus T02
INTEGRATED T02 3c74ca1
# 2026-11-16 21:07:45 PST  INTEGRATED T02 3c74ca1
DECISION R1 numerically passed (by construction, A4) -> Bayes allowed for the declared ideal dynamics; r_model=9.5e-7, tau_abs=1e-4 (floor); Bayes phase B optional, runs parallel to training, never blocks E1
# 2026-11-16 21:07:45 PST  DECISION R1 numerically passed (by construction, A4) -> Bayes allowed for the declared ideal dynamics; r_model=9.5e-7, tau_abs=1e-4 (floor); Bayes phase B optional, runs parallel to training, never blocks E1
NOTE soomi-integration ebf0876; T02 minors deferred
# 2026-11-16 21:07:45 PST  NOTE soomi-integration ebf0876; T02 minors deferred
LAUNCH writer-opus T03B
# 2026-11-16 21:08:17 PST  LAUNCH writer-opus T03B
NOTE T03 Phase B (Bayes) launched on exp/ref @ebf0876: optional, must never block E1; numerical hazard: tau_abs=1e-4 m gives posteriors much narrower than a 400x400 grid cell
# 2026-11-16 21:08:17 PST  NOTE T03 Phase B (Bayes) launched on exp/ref @ebf0876: optional, must never block E1; numerical hazard: tau_abs=1e-4 m gives posteriors much narrower than a 400x400 grid cell
LAUNCH reviewer-opus T04
# 2026-11-16 21:08:31 PST  LAUNCH reviewer-opus T04
INTEGRATED T04 f5e651b
# 2026-11-16 21:13:20 PST  INTEGRATED T04 f5e651b
NOTE soomi-integration e6c51af; dataset pilot_v1 (hash f722c512bae19c38...) generated from clean source f15567c; T04 minors deferred; e4_cellD not read by T06 yet (optional cell)
# 2026-11-16 21:13:20 PST  NOTE soomi-integration e6c51af; dataset pilot_v1 (hash f722c512bae19c38...) generated from clean source f15567c; T04 minors deferred; e4_cellD not read by T06 yet (optional cell)
JOBSTART pilot_smoke2k 1794892413 0.05
# 2026-11-16 21:13:33 PST  JOBSTART pilot_smoke2k 1794892413 0.05
JOBEND pilot_smoke2k 0.0086
# 2026-11-16 21:14:04 PST  JOBEND pilot_smoke2k 0.0086
DECISION training budget: 150000 steps (~35 min at 71 steps/s measured on launch GPU 1, real data), warmup min(2000,5%), cosine over 150000, lr 1e-4 wd 0.01 batch 256 clip 1.0, seed 0 noise 0, run id pilot_v1_n0_s0, source e6c51af, dataset pilot_v1; reserve 0.9 GPUh, timeout 3300 s; max-seconds guard 3000
# 2026-11-16 21:14:21 PST  DECISION training budget: 150000 steps (~35 min at 71 steps/s measured on launch GPU 1, real data), warmup min(2000,5%), cosine over 150000, lr 1e-4 wd 0.01 batch 256 clip 1.0, seed 0 noise 0, run id pilot_v1_n0_s0, source e6c51af, dataset pilot_v1; reserve 0.9 GPUh, timeout 3300 s; max-seconds guard 3000
JOBSTART pilot_train_v1_n0_s0 1794892472 0.9
# 2026-11-16 21:14:32 PST  JOBSTART pilot_train_v1_n0_s0 1794892472 0.9
NOTE control-authored scripts/pilot_load_run.py committed on soomi-integration 1a0289e (unreviewed glue; self-verified: reproduces smoke2k logged val loss 0.06212269 vs 0.06212231); training run pilot_v1_n0_s0 in progress from release e6c51af
# 2026-11-16 21:16:06 PST  NOTE control-authored scripts/pilot_load_run.py committed on soomi-integration 1a0289e (unreviewed glue; self-verified: reproduces smoke2k logged val loss 0.06212269 vs 0.06212231); training run pilot_v1_n0_s0 in progress from release e6c51af
LAUNCH reviewer-opus T03B
# 2026-11-16 21:37:08 PST  LAUNCH reviewer-opus T03B
NOTE T03B result e2b049c: Bayes ok for declared ideal dynamics; adaptive nested quadrature base N=400; vs brute force 2.0e-7 rel; 1 failing T00 test (stub asserts blocked) needs a reviewed update; E4 broad prior needs wiring
# 2026-11-16 21:37:08 PST  NOTE T03B result e2b049c: Bayes ok for declared ideal dynamics; adaptive nested quadrature base N=400; vs brute force 2.0e-7 rel; 1 failing T00 test (stub asserts blocked) needs a reviewed update; E4 broad prior needs wiring
JOBEND pilot_train_v1_n0_s0 0.5608
# 2026-11-16 21:48:11 PST  JOBEND pilot_train_v1_n0_s0 0.5608
NOTE E1-E4 model evaluation run 1 saved: results/v2 (best.pt step 118000, ckpt sha 6101448e..., Bayes unavailable at that time since T03B not integrated); control-authored scripts/pilot_make_figures.py committed; training pilot_v1_n0_s0: 150000 steps, 2016 s wall, 0.5608 GPUh, best val 0.061374 @118000, no NaN, no early stop
# 2026-11-16 21:50:06 PST  NOTE E1-E4 model evaluation run 1 saved: results/v2 (best.pt step 118000, ckpt sha 6101448e..., Bayes unavailable at that time since T03B not integrated); control-authored scripts/pilot_make_figures.py committed; training pilot_v1_n0_s0: 150000 steps, 2016 s wall, 0.5608 GPUh, best val 0.061374 @118000, no NaN, no early stop
LAUNCH reviewer-opus AUDIT_GLUE_C1
# 2026-11-16 23:13:44 PST  LAUNCH reviewer-opus AUDIT_GLUE_C1
NOTE USER: finish pilot, no retrain; independent review of control glue (pilot_load_run.py, pilot_make_figures.py) + audit of strong E1 c=1 result (leakage/indexing/data generation) + compare with Bayes after review; label idealized 1D Coulomb dynamics; keep contact failure evidence as meeting finding; summary with exact SHAs
# 2026-11-16 23:13:44 PST  NOTE USER: finish pilot, no retrain; independent review of control glue (pilot_load_run.py, pilot_make_figures.py) + audit of strong E1 c=1 result (leakage/indexing/data generation) + compare with Bayes after review; label idealized 1D Coulomb dynamics; keep contact failure evidence as meeting finding; summary with exact SHAs
REPAIR GLUE
# 2026-11-16 23:19:55 PST  REPAIR GLUE
NOTE AUDIT_GLUE_C1: Part A 2 MAJOR (figure ylabels) fixed by control and verified (regenerated + tamper tests: +0.5 weight and last-as-best now fail; +0.01 weight still passes at 1e-6 tol => sha256 is the integrity record); fixes NOT re-reviewed (last launch kept for Bayes). Part B: no leakage/indexing defect; exact-Bayes c=1 mse 8.21e-5 vs transformer 1.31e-4 (ratio 1.6). integration HEAD 99a7260
# 2026-11-16 23:21:00 PST  NOTE AUDIT_GLUE_C1: Part A 2 MAJOR (figure ylabels) fixed by control and verified (regenerated + tamper tests: +0.5 weight and last-as-best now fail; +0.01 weight still passes at 1e-6 tol => sha256 is the integrity record); fixes NOT re-reviewed (last launch kept for Bayes). Part B: no leakage/indexing defect; exact-Bayes c=1 mse 8.21e-5 vs transformer 1.31e-4 (ratio 1.6). integration HEAD 99a7260
INTEGRATED T03B e2b049c
# 2026-11-16 23:58:48 PST  INTEGRATED T03B e2b049c
NOTE soomi-integration 2f3d53ddb406442f949216dbced92622983b8a50: T03B Bayes integrated (0 CRITICAL/MAJOR; ideal dynamics only); T00 stub test updated per reviewer; 168 passed 0 skipped
# 2026-11-16 23:58:48 PST  NOTE soomi-integration 2f3d53ddb406442f949216dbced92622983b8a50: T03B Bayes integrated (0 CRITICAL/MAJOR; ideal dynamics only); T00 stub test updated per reviewer; 168 passed 0 skipped
JOBSTART eval_final_best 1794902335 0
# 2026-11-16 23:58:55 PST  JOBSTART eval_final_best 1794902335 0
JOBEND eval_final_best 0.0000
# 2026-11-16 23:59:22 PST  JOBEND eval_final_best 0.0000
JOBSTART eval_final_last 1794902369 0
# 2026-11-16 23:59:29 PST  JOBSTART eval_final_last 1794902369 0
JOBEND eval_final_last 0.0000
# 2026-11-16 23:59:34 PST  JOBEND eval_final_last 0.0000
NOTE created isolated tools venv /home/exx/soomi/soomi_projects/cs282a_project/cs282a_data/pilot/tools_venv (python-pptx, markdown, pillow) for the meeting deck; shared env untouched
# 2026-11-17 00:01:21 PST  NOTE created isolated tools venv /home/exx/soomi/soomi_projects/cs282a_project/cs282a_data/pilot/tools_venv (python-pptx, markdown, pillow) for the meeting deck; shared env untouched
