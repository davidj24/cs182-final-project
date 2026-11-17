# Review T00 @ 23a088c (reviewer-opus, fresh context) -- saved by control
Unresolved: CRITICAL 0, MAJOR 0, MINOR 5. 14 fast tests passed. Physics recomputed independently:
d(0.5,0.3,26,0.2)=17.335807 m; d(1.5,0.6,13,0.1)=0.0204716 m. Config checked item by item vs PILOT_SPEC C/D/H1/K/L/M/N/J: all match.
bayes_predict returns a 'blocked' Prediction, nothing depends on Bayes. Ownership OK.
MINOR-1 Action.T = k*DT only at dt=0.001; T01/T02 must compute T=k*dt explicitly for the dt=0.0005 check.
MINOR-2 warmup_cap_steps=2000 is a writer choice (assumption; record at H5).
MINOR-3 .gitignore exceptions un-ignore src/cs182_icl/data and tests/data; T04 must write data only under $CS282A_DATA.
MINOR-4 metric list / method list / duration-multiset rule only in PILOT_SPEC prose, not in config.
MINOR-5 history: daf9ec4 malformed subject, a5d0e33 fails tests alone; not rewritable, does not block --no-ff.
Recommendation: eligible for --no-ff merge.
