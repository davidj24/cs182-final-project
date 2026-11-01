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
| David | RTX 5090 | 32 GB | TODO | Linux dual-boot (only available when booted into Linux) |
| TODO | | | | |
| TODO | | | | |
| TODO | | | | |

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
