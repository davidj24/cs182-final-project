# AI usage log

Running record of generative-AI use for the report's required appendix (cite tools and describe how they were used; it doesn't count toward page limits). Add an entry whenever an AI tool contributes code, text, figures, literature search or ideas. Note what was verified by hand.

| Date | Teammate | Tool (model) | Project part | What it was used for | How we verified / what we changed |
|---|---|---|---|---|---|
| 2026-11-01 | David | Claude Code (Claude Opus 5.5) | Repo setup | Created repo scaffolding: uv project and PyTorch/CUDA index config, `.gitignore`, directory layout, README, CLAUDE.md, `scripts/check_env.py` | Ran the torch/CUDA check on an RTX 5090; reviewed generated files |
| 2026-11-16 | Soomi (session to be confirmed by Soomi) | Claude Code (Claude Opus 5.5, Claude Sonnet 5.5; per commit trailers) | Pilot | Multi-agent implementation of simulator, data, model, training, references, evaluation, figures, pilot summary and guide (branch `soomi-integration`) | Automated tests and separate agent reviews saved under `results/meeting_packet_v1/reviews/`; human verification still to be recorded |
| 2026-11-24 | David | Claude Code (Claude Sonnet 5.5) | Report V1 | Drafted the V1 text from the proposal and saved pilot results; literature search and checks against arXiv abstracts and index records; wrote `scripts/make_report_figures.py` | Numbers transcribed from committed CSVs and pilot documents; team review pending |
