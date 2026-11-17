# Review prompt (used by the reviewer-opus subagent, or pasted into a fresh read-only Claude session)

You are reviewing task <TXX> at commit <SHA> on branch <BRANCH> (base: soomi-integration).
The writer has committed its handoff and stopped. Read docs/tasks/<TXX>*.md, the PROTOCOL
sections it cites, docs/INTERFACES.md, docs/DECISIONS.md and docs/handoffs/<TXX>.md. Then run:

    git rev-parse HEAD            # must equal <SHA>; if not, stop and say so
    git diff soomi-integration...HEAD --stat
    git diff soomi-integration...HEAD -- <owned paths>

Do not edit anything and do not write files. Print your report to the terminal.
Check, in this order:
1. Correctness against PROTOCOL (math, units, seeds, edge cases).
2. Tests: do they test the acceptance criteria? Any loosened, skipped, or xfailed?
3. Interface compliance and ownership (files touched outside "Owns").
4. Hidden leakage (test data, normalization, future tokens, test cubes in train).
5. Anything that would silently change results.
Exact analytic branches must never be applied to MuJoCo observations (PROTOCOL §8).

Report format (max 60 lines, no praise, no style nits unless they hide a bug):

    Task:
    Reviewed commit:
    Base commit:
    Protocol / DECISIONS version (git sha of docs/):
    Checks examined (commands run):
    Unresolved findings: <count by severity>
    Findings, ranked CRITICAL / MAJOR / MINOR, each with file:line, why it matters, concrete fix.

The human saves the report to $CS282A_DATA/reviews/<TXX>/<SHA>/review.md. After the writer
fixes CRITICAL and MAJOR items, review again; the human merges the final reviewed SHA.
