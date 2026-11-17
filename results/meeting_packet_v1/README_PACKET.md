# Meeting packet `meeting_packet_v1` — pilot `initial_e2e_2026_11_16`

**Exact directory:** `/home/exx/soomi/soomi_projects/cs282a_project/cs282a_data/results/meeting_packet_v1/`

> Everything is about **idealized 1D Coulomb dynamics implemented in MuJoCo**, one training seed, no observation noise. The original free-cube contact scene failed validation; that failure is preserved as a finding.

## Read / open first
1. **Read first:** `GUIDE_READ_FIRST.md` (or `GUIDE_READ_FIRST.html`, same text, images embedded). Section 0 is the one-page version; Sections 1–2 teach the physics; 3–4 the simulator failure and change; 5–7 data, model, references; 8 results; 9–10 limitations, next steps and questions; Appendix A is a Q&A.
2. **Open for the meeting:** `PRESENTATION_meeting.pptx` (10 main slides + 8 backup, speaker notes on every slide; `PRESENTATION_meeting.pdf` is a rendering without notes).
3. Short repo-side summary: `results/SUMMARY.md` on branch `soomi-integration` (copy in `results/pilot_v1/`).

## Contents
| Path | What |
|---|---|
| `GUIDE_READ_FIRST.md/.html` | the learning guide |
| `PRESENTATION_meeting.pptx/.pdf` | the deck |
| `figures/` | all guide/deck figures (F01–F12, B01, B02), each results figure names its checkpoint in its caption; `figures/_superseded_before_factcheck/` = the first-draft versions (kept; F09 was stale); `figures_src_v3_r3/`, `figures_src_v4_last_r3/` = raw generator outputs; `figures_slides*/` = caption-free versions used on the slides; the unsuffixed / `_r2` directories are superseded |
| `csv/` | CSVs behind every figure: `results_v3_best_ckpt/` (primary), `results_v4_last_ckpt_sensitivity/`, per-position validation loss, c=1 audit, supplementary broad-prior Bayes, `guide_facts.json` |
| `simulator_evidence/` | `contact_reproduction_5e0ae3f_v2/` (NEW reproduction of the original failure), `historical_contact_evidence_saved_during_run/` (saved at the time: replays, candidate tables, questions file), `ideal_simulator_H2_gate_packet/` (validation packet for the accepted simulator), `final_scene_replays/` |
| `reviews/` | every saved reviewer report (T00–T06, T01 cycles, Bayes, glue/c=1 audit) |
| `provenance/` | exact SHAs, environment records, commands, ledger snapshot, budget status, configs, checkpoint provenance |
| `repro/` | the scripts that generated the packet material (also committed under `scripts/packet/`) |
| `guide_assets/` | facts JSON, slide crops |

## Figures: essential vs backup
**Essential (one per message):** F05 training curves • F07 E1 error vs context • F08 one-observation audit • F09 E2 same vs distinct force • F10 E3 duplicates • F12 E4 by prior • simulator failure `simulator_evidence/contact_reproduction_5e0ae3f_v2/fig1_contact_failure_traces.png` • F03 worked example.
**Backup:** F01 pipeline, F02 token diagram, F04 parameter ambiguity, F06 per-position validation loss, F11 E4 detail, B01 and B02 `last.pt` sensitivity (E1, E2), `fig2_random_1000.png`, `fig3_no_force_sliding.png`, the H2 gate-packet PNGs.

## Exact identifiers
Training commit `e6c51afba43894d75c544243be6bcddc6acb27b1` • evaluation commit `2f3d53ddb406442f949216dbced92622983b8a50` • dataset `pilot_v1` hash `f722c512bae19c386748f9396fb85a5c968e06fa7ac77f728a36832833559918` (generated from `f15567c`) • checkpoint `best.pt` step 118,000 sha256 `6101448ec7b9430d9bc1145644307cf9ce565ad4ba2920f21821d14457edf230` • results/figures/packet scripts committed on `soomi-integration` at `866d8da6084c0cde016fd44b91531e77c7c768cf`. Details: `provenance/SHAS_AND_ENVIRONMENT.md`, `provenance/commands.md`.

## Outstanding checks and incomplete results (read before presenting)
* **Fact-check:** an independent read-only reviewer checked the first draft against the saved files and found 1 critical, 6 major, 17 minor discrepancies; all were corrected (`provenance/FACT_CHECK.md`). **The corrected guide, deck, summary and README were not fact-checked a second time** (all 24 worker launches are used).
* **E2 at c=4 is inconclusive:** its sign flips between the selected checkpoint and the `last.pt` sensitivity run (guide §8.3).
* **Not independently reviewed:** the controller-authored scripts (loader, figures, packet scripts, deck builder) — only the loader and `pilot_make_figures.py` received a read-only audit, and its two figure-label bugs were fixed and verified by the controller, not re-reviewed. The final change to `pilot_make_figures.py` (draw Bayes in both E2 styles) and the packet scripts were not re-audited.
* **Bayes review:** the final report was a partial one on the adversarial synthetic cases (19 of 23 complete); the independent-implementation comparison on 80 real item cases was complete.
* **Training length:** 150,000 steps exceeds PROTOCOL §7's 100k "ceiling"; the pilot spec had no maximum, and I did not flag it at launch.
* **Hypotheses (untested):** the cause of the contact failure mechanism, and the cause of the transformer's plateau at c ≥ 2.
* **Not done:** E4 cell-D (extra file generated, not evaluated); noise; seeds; contact repair; Bayes at σ_d > 0.
* **Stale text:** the header comment in `assets/cube.xml` still mentions MuJoCo `frictionloss` (the code uses the explicit Coulomb force).
