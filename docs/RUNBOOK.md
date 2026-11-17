# Runbook (index)

Agents write their launch commands in `docs/runbook/Txx.md` (one file per task, no conflicts).
The human links them here at merge time.

Every long job runs from a clean checkout of ONE accepted commit:

    workflow/setup_worktrees.sh release <name> <sha>     # -> $WT/run-<name>, detached

Each runbook entry must state: source commit, working directory, config hash, dataset and
normalization version, seed, command, expected runtime and CPU/GPU, outputs under
$CS282A_DATA, and how to check it worked.

## Index
(add one line per merged task: `- T03: docs/runbook/T03.md`)
