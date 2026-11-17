# Review T03 (pilot Phase A) @ 49c15e1 -- saved by control
Unresolved CRITICAL 0, MAJOR 0, MINOR 5. 37 fast tests pass. Reviewer hand-recomputed 5 physics-fit predictions in plain python, all match.
M1 unphysical fit (A<=0 or B<=0) flagged only via reason substring (extrapolation error up to +62% on light/slippery with 5-8% discrepancy and 3N span; B<0 will be common) -> T06 must report flagged fraction/error separately.
M2 FORCE_SEP_MIN=3.0 N is a writer choice (~41% of c=2 E1 items not_determined) -> recorded in DECISIONS.
M3 fit space (d vs line) -> recorded: default 'd' per PROTOCOL 8.
M4 missing tests: move_eps boundary, unphysical flag, perturbed data. M5 dedupe ignores d_clean.
Bayes: not implemented (blocked stub). Nothing depends on Bayes.
