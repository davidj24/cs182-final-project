"""Independent brute-force posterior means for validating reference/bayes.py.

Written separately from bayes.py on purpose: no weighted-least-squares windows, no adaptive
splitting, its own displacement formula and Gaussian log-likelihood. Two plain dense midpoint
grids:
  * brute_box: dense n x n grid in (u = log m, mu) on a box around an independently located
    mode (iterated dense argmax search), grown until every box side that is not a prior edge
    lies >= EDGE_DROP nats below the maximum. Box sides on prior edges coincide with them.
    For contexts with >= 2 distinct forces (compact posterior).
  * brute_ridge: dense grid in (u, v) with v = unclipped f(theta, a0) for the first action a0,
    covering the full prior u range and v within +-K tau of the observation. For one-force
    (or nearly equal-force) contexts the posterior is a thin ridge v ~ const. The prior density in (u, v) is
    |d mu / d v| = 2 m^2 g mu^2 / (F0^2 T0^2) times the box indicator.
"""

from __future__ import annotations

import math

import numpy as np

G = 9.81
DT = 0.001
R_MODEL, TAU_ABS = 9.5e-7, 1e-4
EDGE_DROP = 35.0


def disp(m, mu, F, T):
    """Clipped closed form (written independently of physics.py)."""
    thr = mu * m * G
    d = F * T * T * (F - thr) / (2.0 * m * m * mu * G)
    return np.where(F > thr, d, 0.0)


def loglik(m, mu, groups):
    """groups: list of (F, T, ybar, n); sigma_d = 0."""
    out = 0.0
    for F, T, y, _n in groups:
        f = disp(m, mu, F, T)
        var = (R_MODEL * f + TAU_ABS) ** 2
        out = out - 0.5 * (y - f) ** 2 / var - 0.5 * np.log(2 * math.pi * var)
    return out


def _grid_mean(groups, q, ua, ub, ma, mb, n, chunk=500):
    hu, hm = (ub - ua) / n, (mb - ma) / n
    u = ua + hu * (np.arange(n) + 0.5)
    mu = ma + hm * (np.arange(n) + 0.5)
    ll = np.empty((n, n))
    for i in range(0, n, chunk):
        ll[i : i + chunk] = loglik(np.exp(u[i : i + chunk, None]), mu[None, :], groups)
    mx = ll.max()
    w = np.exp(ll - mx)
    fq = np.empty((n, n))
    for i in range(0, n, chunk):
        fq[i : i + chunk] = disp(np.exp(u[i : i + chunk, None]), mu[None, :], q[0], q[1])
    mean = float((w * fq).sum() / w.sum())
    edges = {"u_lo": ll[0].max(), "u_hi": ll[-1].max(), "mu_lo": ll[:, 0].max(), "mu_hi": ll[:, -1].max()}
    return mean, mx, edges, ll


def find_mode(groups, prior, n=600, iters=40):
    """Iterated dense argmax search over the prior box; returns (u, mu)."""
    ua, ub, ma, mb = math.log(prior.m_lo), math.log(prior.m_hi), prior.mu_lo, prior.mu_hi
    U0, M0 = ub - ua, mb - ma
    box = [ua, ub, ma, mb]
    for _ in range(iters):
        _, _, _, ll = _grid_mean(groups, (1.0, 0.1), *box, n)
        i, j = np.unravel_index(np.argmax(ll), ll.shape)
        hu, hm = (box[1] - box[0]) / n, (box[3] - box[2]) / n
        cu, cm = box[0] + hu * (i + 0.5), box[2] + hm * (j + 0.5)
        on_edge = i in (0, n - 1) or j in (0, n - 1)
        su = (box[1] - box[0]) / 2 if on_edge else 4 * hu
        sm = (box[3] - box[2]) / 2 if on_edge else 4 * hm
        new = [max(ua, cu - su), min(ub, cu + su), max(ma, cm - sm), min(mb, cm + sm)]
        if new == box or (su < 1e-12 * U0 and sm < 1e-12 * M0):
            break
        box = new
    return cu, cm


def brute_box(groups, q, prior, center=None, n=2001, n_search=301):
    """Dense (u, mu) grid around the mode, grown (on a coarse n_search grid) until all free
    sides are negligible, then evaluated at n x n and the side condition re-checked."""
    ua, ub, ma, mb = math.log(prior.m_lo), math.log(prior.m_hi), prior.mu_lo, prior.mu_hi
    cu, cm = center if center is not None else find_mode(groups, prior)
    du, dm = 1e-7 * (ub - ua), 1e-7 * (mb - ma)
    for _ in range(80):
        box = (max(ua, cu - du), min(ub, cu + du), max(ma, cm - dm), min(mb, cm + dm))
        free = {"u_lo": box[0] > ua, "u_hi": box[1] < ub, "mu_lo": box[2] > ma, "mu_hi": box[3] < mb}
        mean, mx, e, _ = _grid_mean(groups, q, *box, n_search)
        bad = [k for k in e if free[k] and e[k] > mx - EDGE_DROP]
        if not bad:
            mean, mx, e, _ = _grid_mean(groups, q, *box, n)
            bad = [k for k in e if free[k] and e[k] > mx - EDGE_DROP]
            if not bad:
                return mean, box
        if any(k.startswith("u") for k in bad):
            du *= 2.0
        if any(k.startswith("mu") for k in bad):
            dm *= 2.0
    raise RuntimeError("brute_box did not enclose the posterior")


def brute_ridge(groups, q, prior, nu=200_000, nv=200, K=15.0, chunk=5000):
    """Dense (u, v) grid; v = unclipped f at the first action. The first group alone confines
    v to +-K tau, so this is valid for any context, but the u axis (nu cells over the full
    range) only resolves posteriors that are long in u: one force or nearly equal forces.
    Accumulates sum w and sum w f_q chunk by chunk with a running max (log-sum-exp)."""
    F0, T0, y0, _ = groups[0]
    ua, ub = math.log(prior.m_lo), math.log(prior.m_hi)
    tau = R_MODEL * abs(y0) + TAU_ABS
    va, vb = y0 - K * tau, y0 + K * tau
    hu, hv = (ub - ua) / nu, (vb - va) / nv
    v = va + hv * (np.arange(nv) + 0.5)
    mx, s0, s1, edge = -np.inf, 0.0, 0.0, -np.inf
    for i in range(0, nu, chunk):
        u = ua + hu * (np.arange(i, min(nu, i + chunk)) + 0.5)
        m = np.exp(u)[:, None]
        mu = F0 * F0 * T0 * T0 / (2 * m * m * G * (v[None, :] + F0 * T0 * T0 / (2 * m)))
        inside = (mu >= prior.mu_lo) & (mu <= prior.mu_hi)
        if not inside.any():
            continue
        mu_c = np.clip(mu, prior.mu_lo, prior.mu_hi)
        jac = 2 * m * m * G * mu_c * mu_c / (F0 * F0 * T0 * T0)
        ll = np.where(inside, loglik(m, mu_c, groups) + np.log(jac), -np.inf)
        fq = disp(m, mu_c, q[0], q[1])
        edge = max(edge, ll[:, 0].max(), ll[:, -1].max())
        cmx = ll.max()
        if cmx > mx:
            scale = math.exp(mx - cmx) if np.isfinite(mx) else 0.0
            s0, s1, mx = s0 * scale, s1 * scale, cmx
        w = np.exp(ll - mx)
        s0 += float(w.sum())
        s1 += float((w * fq).sum())
    if not np.isfinite(mx):
        raise ValueError("no prior mass in the ridge window")
    if edge > mx - EDGE_DROP:
        raise ValueError("ridge v-window too narrow")
    return s1 / s0
