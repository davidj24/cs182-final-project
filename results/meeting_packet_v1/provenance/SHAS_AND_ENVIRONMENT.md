# Exact identifiers (all read from saved files)

| What | Identifier |
|---|---|
| **Training source commit** (clean tree, `dirty=false`) | `e6c51afba43894d75c544243be6bcddc6acb27b1` |
| Training run id / config hash | `pilot_v1_n0_s0` / `a872534830616b822746a3995a2bd943bdcd7d1a78a8bf82b6d1d98ab3553fc6` |
| Steps / seed / noise | 150000 / 0 / 0.0 |
| Selected checkpoint | `best.pt`, step 118000, sha256 `6101448ec7b9430d9bc1145644307cf9ce565ad4ba2920f21821d14457edf230` |
| Sensitivity checkpoint | `last.pt`, step 150000, sha256 `f645625d252286b274445269df999f247fe8335c37fcde78c7d17a434f86c85a` |
| **Evaluation source commit** (clean release checkout; results `results/v3` best, `results/v4` last) | `2f3d53ddb406442f949216dbced92622983b8a50` |
| Figure/packet scripts | integration branch `soomi-integration`, commit `866d8da6084c0cde016fd44b91531e77c7c768cf` (figures, SUMMARY, scripts/packet) |
| Dataset | `pilot_v1`, dataset hash `f722c512bae19c386748f9396fb85a5c968e06fa7ac77f728a36832833559918`, generated from source commit `f15567cae923c0b9bcd6dabc7d0119e2017af70a` (clean: True), seed 0, 16 workers |
| Per-file sha256 (`meta.json`) | train.npz: 084457246996…, val.npz: 6f8f211f5a9e…, e1.npz: a1663884060d…, e2.npz: 7795c445d808…, e3.npz: f1059be066ef…, e4.npz: 06e3308c6bd9…, e4_cellD.npz: f86af27bdaf9… |
| Reviewed task SHAs integrated | T00 `23a088c`, T03A `49c15e1`, T06 `babf15b`, T05 `077405b`, T01 `f28ff1e` (after 3 repair cycles), T02 `3c74ca1`, T04 `f5e651b`, T03B (Bayes) `e2b049c`; setup `1fb6618` |
| Historical contact evidence | scene at commits `9876867` and `5e0ae3f`; reproduction run from an isolated checkout of `5e0ae3f` |

## Environment (`env_*.txt` in this directory; second file is after installing mujoco/pytest)
```
python: Python 3.12.16
uv: uv 0.13.8 (x86_64-unknown-linux-gnu)
mujoco: 4.0.1
pytest: 9.2.0
torch: 2.14.1+cu126 cuda: 12.6 available: True gpus: 2
 gpu 0 NVIDIA GeForce RTX 4090
 gpu 1 NVIDIA GeForce RTX 4090
```
Shared environment was never re-synced; torch 2.14.1+cu126 is unchanged (the lockfile pins cu130; see `workflow/STEPS.md` appendix).

## Runtime and resources (ledger / logs)
* Training: 2,016 s wall, 2,012 s GPU time, 1 × RTX 4090 (GPU 1), ≈ 420k tokens/s; ledger GPU-hours 0.5608 (run) + smoke checks = **0.5705 of 2.0**.
* Dataset generation: ~8 s wall, 16 CPU workers. Final evaluation (all of E1–E4 with Bayes): 27 s, CPU; `last.pt` sensitivity: 5 s with the cache.
* Launch counts (ledger): 23 of 24 worker/reviewer launches; repair cycles: T01 3 (the third explicitly authorized by the user), T06 1, control-glue 1. New outputs 0.38 GB of the 10 GB cap.
