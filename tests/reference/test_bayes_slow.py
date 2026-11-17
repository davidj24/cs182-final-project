"""@slow: zoomed-grid Bayes vs independent brute force on the hard-case subset (24 items).

For each case: posterior mean of the query at base N = 400 and N = 800 (same zoom), and an
independent dense-grid brute force (tests/reference/_bayes_brute.py):
  * force span >= 1 N: dense (log m, mu) grid around the TRUE parameters when they lie in the
    prior, else around an independently located mode; box grown until free sides are
    negligible (2001 x 2001 final grid);
  * one force or force span < 1 N (ridge): dense (log m, v = f(theta, a0)) grid over the full
    mass range (200,000 x 200).
Required: relative agreement < TOL for N400 vs N800 and N400 vs brute force; no outer cell at
the minimum width. Run with  pytest -m slow tests/reference/test_bayes_slow.py -s  to print
the table.
"""

import math

import pytest

import cs182_icl.reference.bayes as B
from cs182_icl.physics import analytic_displacement
from cs182_icl.reference.priors import get_prior
from tests.reference import _bayes_brute as bb
from tests.reference._bayes_cases import HARD_CASES

TOL = 1e-5  # relative; stricter than the 1e-3 floor asked for in the brief


def _brute(hc, groups, q):
    pr = get_prior(hc.prior)
    bg = [(g.F, g.k * 1e-3, g.ybar, g.n) for g in groups]
    Fs = [F for F, _ in hc.acts]
    if max(Fs) - min(Fs) < 1.0:
        return "ridge", bb.brute_ridge(bg, (q.F, q.T), pr)
    inside = pr.m_lo <= hc.m <= pr.m_hi and pr.mu_lo <= hc.mu <= pr.mu_hi
    center = (math.log(hc.m), hc.mu) if inside else None
    return ("box@truth" if inside else "box@mode"), bb.brute_box(bg, (q.F, q.T), pr, center=center)[0]


@pytest.mark.slow
def test_zoom_matches_800_and_brute_force_on_hard_cases():
    assert len(HARD_CASES) >= 20
    worst48 = worst4b = 0.0
    lines = []
    for hc in HARD_CASES:
        groups, q = B.make_groups(hc.context()), hc.qaction()
        a, sd, ia = B.posterior_mean(groups, q, prior=hc.prior, n=400)
        b, _, ib = B.posterior_mean(groups, q, prior=hc.prior, n=800)
        how, bf = _brute(hc, groups, q)
        r48, r4b = abs(a - b) / abs(b), abs(a - bf) / abs(bf)
        truth = analytic_displacement(hc.m, hc.mu, q.F, q.T)
        lines.append(
            f"{hc.name:30s} {hc.prior:5s} c={len(hc.acts):2d} {how:9s} N400={a:.10g} brute={bf:.10g} "
            f"|400-800|={r48:.1e} |400-brute|={r4b:.1e} sd/mean={sd / a:.1e} (mean-truth)/truth={(a - truth) / truth:+.1e}"
        )
        assert ia["resolved"] and ib["resolved"]
        assert not ia["hit_min_width"] and not ib["hit_min_width"]
        worst48, worst4b = max(worst48, r48), max(worst4b, r4b)
    print("\n" + "\n".join(lines))
    print(f"max |400-800| = {worst48:.2e}, max |400-brute| = {worst4b:.2e} (relative)")
    assert worst48 < TOL and worst4b < TOL
