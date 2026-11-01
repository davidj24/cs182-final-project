# AI usage log

Running record of generative-AI use for the report's required appendix (cite tools and describe how they were used; it doesn't count toward page limits). Add an entry whenever an AI tool contributes code, text, figures, literature search or ideas. Note what was verified by hand.

| Date | Teammate | Tool (model) | Project part | What it was used for | How we verified / what we changed |
|---|---|---|---|---|---|
| 2026-11-01 | David | Claude Code (Claude Opus 5.5) | Repo setup | Created repo scaffolding: uv project and PyTorch/CUDA index config, `.gitignore`, directory layout, README, CLAUDE.md, `scripts/check_env.py` | Ran the torch/CUDA check on an RTX 5090; reviewed generated files |
| 2026-11-01 | David | Claude Code (Claude Opus 5.5) | HW9 P4 hardware check | Set up a separate environment for the course's unmodified `hardware_check.py`, ran the suite, and summarized the measured limits in CLAUDE.md's compute-budget section | All numbers come from the generated `hardware-results.csv` (exit code 2 = recorded limits); the script was not edited |
