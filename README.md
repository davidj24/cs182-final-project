# CS 182 Final Project: In-Context Learning

Final project for CS 182 / 282A (Designing, Visualizing and Understanding Deep Neural Networks), UC Berkeley, Fall 2026. Topic: **in-context learning (ICL)**.

**Team:** Soomi Lee, Catherine Zhuang, Kenneth Llontop, David Jo

**Research question:** how do the number and type of pushes in a transformer's context determine how accurately it predicts a new push on an unseen block, compared with an estimator that uses the same context optimally?

## Reviewers: start here

[report/v1/README.md](report/v1/README.md) explains how to regenerate the report's figures and tables (CPU only, about a minute):

```bash
python -m pip install matplotlib
python scripts/make_report_figures.py
python scripts/make_report_tables.py
```

The sections below describe the full development environment.

## Setup

Each teammate runs this on their own machine. Linux, Windows and Apple Silicon Macs are supported.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   (Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`)
2. Clone the repo and install everything. uv downloads Python 3.12 itself if needed:
   ```bash
   git clone <repo-url> cs182-final-project
   cd cs182-final-project
   uv sync
   ```
3. Check that PyTorch sees your accelerator:
   ```bash
   uv run python scripts/check_env.py
   ```
4. Log in to Weights & Biases (one-time; free academic accounts are available):
   ```bash
   uv run wandb login
   ```
   Log runs to the shared team project (TODO: entity/project name) so everyone's runs can be compared in one place.

### GPU / CUDA notes

- **NVIDIA (Linux/Windows):** PyTorch is installed from the CUDA 13.0 wheel index (configured in `pyproject.toml` under `[tool.uv.sources]`). This needs an **NVIDIA driver ≥ 580** (check with `nvidia-smi`). You don't need a separate CUDA toolkit; the wheels bundle the runtime. Works on Blackwell (RTX 50xx) as well as older GPUs.
- **Apple Silicon:** you get the regular PyPI build, which uses the `mps` device. Expect it to be much slower than a CUDA GPU, and some ops are unsupported.
- **No GPU:** everything installs and runs on CPU. That's fine for small debugging runs.
- **Other platforms** (Intel Mac, Linux ARM): add them to `[tool.uv] environments` in `pyproject.toml` and run `uv lock`.
- The torch version is pinned to one minor release so that all of us run the same version. If you bump it, bump it for everyone in a PR.

## Running things

Always go through `uv run` so the project environment is used:

```bash
uv run python scripts/<script>.py              # one-off scripts
uv run python experiments/<exp>/train.py --config configs/<exp>.yaml   # experiments (once they exist)
uv run jupyter lab                             # notebooks; pick the project's .venv kernel
```

To add a dependency, run `uv add <pkg>` and commit both `pyproject.toml` and `uv.lock`. After pulling, run `uv sync` again.

## Repository layout

```
src/cs182_icl/        shared package: models, data generation, training/eval utilities
experiments/          one subfolder per experiment (entry points that use src/cs182_icl)
configs/              experiment configs; a copy is saved alongside every run's results
scripts/              standalone scripts (env check, plotting, checkpoint upload, ...)
notebooks/            exploration only. Report figures come from scripts, not notebooks
results/              small committed outputs: CSVs of metrics, final plots
hw9_hardware_check/   HW9 Problem 4 (hardware check) code and hardware-results.csv
report/               LaTeX proposal and report
docs/ai_usage.md      running log of AI-tool use (required for the report appendix)
```

These are **not** committed (see `.gitignore`): `data/`, `checkpoints/`, `wandb/`, `runs/`, `.venv/`, and model weights (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`).

## Reproducibility rules

Every plot and table in the report has to be reproducible from a script plus a saved checkpoint. Concretely:

- Fixed seeds, and multiple seeds per config so we can report meaningful error bars.
- Log training **and** validation loss for every run.
- Save the exact config next to each run's results.
- Each report figure has a script in `scripts/` that regenerates it from checkpoints or `results/` CSVs.

## Sharing checkpoints

The final submission (Dec 12) must include trained checkpoints that reproduce every plot and table. Nothing is set up yet. The options are:

- **Hugging Face Hub** (a model repo under one teammate's account or an org): handles large files well, gives each checkpoint a stable URL, and we'd add a `scripts/download_checkpoints.py`. Probably the better fit for many or large checkpoints.
- **Git LFS** in this repo: keeps checkpoints next to the code, but GitHub's free LFS storage and bandwidth quotas are small, which matters for 4 people cloning repeatedly.

Until we decide, keep checkpoints in `checkpoints/` locally (git-ignored), and record which run/config produced each one.

## Collaboration

- Work on branches and open PRs into `main`. Don't push large files.
- Log AI-tool use in [docs/ai_usage.md](docs/ai_usage.md) as you go.
- Deadlines, requirements and conventions are in [CLAUDE.md](CLAUDE.md).
