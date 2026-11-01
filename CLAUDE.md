# CLAUDE.md

Context for Claude (and teammates) working in this repo. Fill in the TODOs once the group settles on a research question.

## Project overview

- **Course:** CS 182 / 282A, Designing, Visualizing and Understanding Deep Neural Networks (UC Berkeley, Fall 2026)
- **Topic:** In-context learning (ICL)
- **Team members:** TODO (4 people: name, GitHub handle, hardware)
- **Staff contact:** Matteo Guarrera (ICL track)
- This is a **group project**. Four teammates work on their own machines with different hardware, so don't assume a particular GPU, OS or file path. Code must run on any teammate's setup (CUDA, MPS or CPU) through `uv sync` / `uv run`.

## Research question / hypothesis

TODO

## What we train

TODO

Course rule: **every project must train something**, whether a model from scratch, a soft prompt/prefix, a LoRA adapter, or a small module of a larger system. We have to show training and validation curves. API-only projects are rejected.

## Hardware & compute budget

TODO: fill in from the HW9 P4 hardware-check results (one row per teammate), then estimate the total GPU-hours we can actually spend.

| Teammate | GPU | VRAM | Throughput (HW9 P4) | Availability notes |
|---|---|---|---|---|
| David | RTX 5090 | 31.35 GiB usable (32 GB) | FP32 inference 47 / 34 tok/s (Qwen3.5 0.8B / 4B); LoRA ~0.63 s/step up to 4B | Linux dual-boot (only available when booted into Linux); 62 GiB host RAM |
| TODO | | | | |
| TODO | | | | |
| TODO | | | | |

### David's HW9 P4 results (2026-11-01)

Fixed HW9 workload: FP32 everywhere, batch 1, 128 input tokens, no AMP; training = 2 AdamW steps. Largest size that succeeded per mode (Qwen3.5 ladder 0.8B → 2B → 4B → 9B):

| Mode | Largest OK | First failure | Numbers |
|---|---|---|---|
| FP32 inference, GPU | 4B | 9B (CUDA OOM) | 47.4 / 45.9 / 33.9 tok/s at 0.8B / 2B / 4B; 15.8 GiB peak at 4B |
| Full fine-tune, GPU | 0.8B | 2B (CUDA OOM) | 0.56 s/step, 13.2 GiB peak at 0.8B |
| LoRA r=8, GPU | 4B | 9B (CUDA OOM) | 0.64 / 0.64 / 0.63 s/step; 18.5 GiB peak at 4B |
| LoRA + activation offload | 4B | 9B (CUDA OOM) | 5.95 s/step at 4B |
| CPU-offload inference | 9B (top of ladder) | none | 0.36 tok/s at 9B |
| FP32 / disk inference, CPU | 9B (top of ladder) | none | 1.02 / 0.13 tok/s at 9B |
| INT8 inference, CPU | 4B | 9B (180 s timeout) | 4.5 tok/s at 4B |

Ling-3.0-tiny (7.9B MoE) and Mellum2.1-12B both stopped at the ~53 GiB host-RAM cap while loading FP32 weights, in every mode. That's a host-RAM limit of the FP32 loading path, not VRAM.

**What this means for us:** these are FP32 limits. bf16/mixed precision roughly halves weight/gradient memory, so the real ceilings are somewhat higher than the table.
- Small transformers trained from scratch (Garg et al. scale, ≲100M params) are trivial on this GPU. We can afford many seeds and sweeps.
- LoRA / soft-prompt / prefix training on a pretrained model up to ~4B (e.g. Qwen3.5 0.8B–4B) fits comfortably, even in FP32. A ~1B model is easy; 9B needs bf16 or 4-bit (QLoRA, not tested here).
- Full fine-tuning in FP32 with AdamW is limited to ~1B (16 bytes/param). For ~2B full fine-tunes, use bf16 + 8-bit optimizer or gradient checkpointing.
- CPU-side paths (offload, INT8, disk) are far too slow for training loops; treat them as eval-only.

## Key deadlines

All 10:59 PM PT.

| Date | Milestone |
|---|---|
| Wed Nov 4 | HW9 P4 hardware check (upload `hardware-results.csv` to Gradescope; must be done before staff meeting #1) |
| Week of Nov 2 | 1:1 staff meeting #1 (whole team; bring draft proposal) |
| Wed Nov 11 / Thu Nov 12 | Proposal (Ed says "Wed Nov 12", which is ambiguous; aim for Nov 11 until staff clarify) |
| Week of Nov 16 | 1:1 staff meeting #2 |
| Mon Nov 23 | Report V1 (for peer review) |
| Mon Nov 30 | Peer reviews, round 1 (individual; skipping one = zero on the project) |
| Mon Dec 7 | Report V2 |
| Wed Dec 9 | Peer reviews, round 2 |
| Sat Dec 12 | Final submission: report + GitHub repo with code + checkpoints that reproduce every plot/table |
| TBD | Oral exam / poster session |

## Experiment conventions

- Fixed seeds, with **multiple seeds per config** so we can draw meaningful error bars (required).
- Always log **training and validation** loss curves (wandb, shared team project).
- Save the config alongside every run's results.
- Every plot and table in the report must be reproducible from a script in `scripts/` plus a saved checkpoint (or a committed CSV in `results/`).
- Don't commit data, checkpoints or wandb/run directories (see `.gitignore`). Small CSVs and final plots go in `results/`.
- Use `uv add` for dependencies and commit `uv.lock`. Don't add flash-linear-attention, transformers or peft until the direction calls for them.

## AI-use policy

- We must cite the AI tools we used and describe how we used them in a report appendix. Keep a running log in [docs/ai_usage.md](docs/ai_usage.md) and add an entry at the end of every session that produced code, text or ideas.
- **Never invent citations, numbers or results.** Hallucinations in the report are severely penalized. Every claim must trace to a real paper or a real run in this repo.
- The oral exam asks about the code, so every teammate needs to understand AI-written code before it's merged. Keep it simple and readable.

## Session workflow

- Each Claude session covers **one part** of the project: setup, hardware check, lit review/proposal, each experiment, report writing.
- At the start of a session, state which part you're responsible for. When that part is done, remind the user to start a new session for the next part. If asked to do work outside your part, suggest a separate session.
- Keep the Notion project page (below) up to date as the long-term record: decisions, results, links to runs.
- Teammates may run their own sessions on their own machines. Coordinate through git (branches + PRs) and the Notion page, not local files.

## Links

- Notion project page: https://app.notion.com/p/CS182-Final-Project-40b830ccb89a804196aac3defc8a7f91
- Ed #287 (deadlines, requirements, logistics): https://edstem.org/us/courses/104046/discussion/8451366
- Ed #300 (ICL projects: meeting signup, HW9-P4, proposal draft): https://edstem.org/us/courses/104046/discussion/8459234
