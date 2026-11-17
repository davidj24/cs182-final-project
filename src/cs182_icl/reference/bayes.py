"""MuJoCo-mode Bayes reference (PROTOCOL §8, PILOT_SPEC §I4). Float64 throughout.

Valid for the DECLARED IDEAL DYNAMICS only: the pilot simulator implements the ideal 1-D law
(PILOT_SPEC A4), so R1 passes by construction (DECISIONS).

Model. theta = (u = log m, mu), prior uniform on a box in (u, mu) (priors.py). For an action
a = (F, k), f(theta, a) = analytic_displacement (clipped). Likelihood:
  1. deduplicate by measurement_id (first copy kept; copies change nothing, bitwise);
  2. group by identical action (F, k), groups in canonical (F, k) order, group mean ybar of
     the n members sorted by measurement_id;
  3. per group  log N(ybar; f, sigma_d^2/n + tau^2),  tau = R_MODEL * f + TAU_ABS,
     including the theta-dependent -0.5 log(2 pi var) term.
Prediction = posterior mean of f(theta, query) in metres. The target space (raw standardized,
DECISIONS) is an affine map of d, so the standardized posterior mean is the same affine map
of this value.

Numerical scheme (the posterior after >= 2 distinct forces has relative width ~1e-5..1e-4, far
below a 400-grid cell; one force gives a thin ridge across the box). Nested adaptive
quadrature with base size N (default 400):
  * inner, mu at fixed u: when moving, f = c1 w - c0 is affine in w = 1/mu, so the conditional
    likelihood is Gaussian in w up to the tiny tau(f) dependence. Weighted least squares gives
    its centre and width; the window keeps the part of [1/mu_hi, 1/mu_lo] within exp(-K^2/2)
    of the interval maximum. N midpoint nodes uniform in mu on that window (constant weights,
    exact prior; window ends are the prior edges whenever clipped). The exact likelihood is
    evaluated there; interior window ends must sit >= INNER_EDGE_DROP nats below the row max,
    else the window is widened and recomputed. No data: the whole interval.
  * outer, u: composite Gauss-Legendre (GL_NODES nodes per panel; base N/GL_NODES equal
    panels, i.e. N nodes), log Z(u) = log of the inner integral at each node. Multiresolution
    zoom: a panel holding a significant node (within NEGLIGIBLE nats of the max, or next to
    one) where log Z jumps by more than JUMP nats between neighbouring nodes, or where
    |d2 log Z/du2| * spacing^2 > CURV, is split into SPLIT panels; repeat until none is
    flagged. This zooms onto blobs (>= 2 forces) and onto sharp ridge exits (1 force);
    panel edges at the prior bounds keep the truncation exact.
  * everything in log-sum-exp; posterior mean = sum w Z M / sum w Z over outer nodes (w the
    Gauss-Legendre weights) with M the inner conditional mean of f(theta, query).
The window construction uses the analytic form of f (a simulated-table forward model is out of
pilot scope). Accuracy is verified against independent brute-force grids in tests/reference.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from cs182_icl.physics import G, analytic_displacement
from cs182_icl.reference.priors import Prior, get_prior
from cs182_icl.types import DT, Action, Observation, Prediction

# Frozen values (DECISIONS.md, H2): do not change here.
R_MODEL = 9.5e-7  # relative discrepancy
TAU_ABS = 1e-4  # absolute discrepancy, m
SIGMA_D = 0.0  # pilot noise is 0 (sigma_d not used)

# Numerical controls (verified in tests/reference/test_bayes*.py).
N_GRID = 400
K_WINDOW = 12.0  # inner window half-width in units of the conditional sd (72 nats)
INNER_EDGE_DROP = 30.0  # nats: required drop at interior inner-window ends
NEGLIGIBLE = 50.0  # nats below the max: a cell this low carries < e^-50 relative mass
GL_NODES = 4  # Gauss-Legendre nodes per outer panel
JUMP = 0.5  # nats: max |log Z| change between neighbouring significant outer nodes
CURV = 0.1  # nats: max |d2 log Z/du2| * spacing^2 at significant outer nodes
SPLIT = 4
MAX_LEVELS = 60
MIN_REL_WIDTH = 1e-13  # smallest outer cell, relative to the u range
ROW_BLOCK = 64  # u nodes per vectorized block (speed only)

_x, _w = np.polynomial.legendre.leggauss(GL_NODES)
_GL_X = 0.5 * (_x + 1.0)  # nodes on [0, 1]
_GL_W = 0.5 * _w  # weights on [0, 1] (sum 1)

__all__ = [
    "R_MODEL",
    "TAU_ABS",
    "SIGMA_D",
    "N_GRID",
    "Group",
    "make_groups",
    "posterior_mean",
    "bayes_predict",
    "bayes_predict_train",
    "bayes_predict_broad",
    "predict_prefixes",
]


@dataclass(frozen=True)
class Group:
    F: float
    k: int
    n: int
    ybar: float


class _Groups:
    """Incremental dedupe + grouping; groups() returns the canonical tuple."""

    def __init__(self):
        self._seen: dict[int, Observation] = {}
        self._by_action: dict[tuple[float, int], list[tuple[int, float]]] = {}

    def add(self, obs: Observation) -> None:
        mid = int(obs.measurement_id)
        if mid == -1:
            return  # padding
        prev = self._seen.get(mid)
        if prev is not None:
            if prev.action != obs.action or prev.d_obs != obs.d_obs:
                raise ValueError(
                    f"measurement_id {mid} repeated with a different action or d_obs; "
                    "a copy must keep identical action, outcome and ID"
                )
            return
        self._seen[mid] = obs
        key = (float(obs.action.F), int(obs.action.k))
        self._by_action.setdefault(key, []).append((mid, float(obs.d_obs)))

    def groups(self) -> tuple[Group, ...]:
        out = []
        for F, k in sorted(self._by_action):
            vals = [v for _, v in sorted(self._by_action[(F, k)])]
            out.append(Group(F, k, len(vals), math.fsum(vals) / len(vals)))
        return tuple(out)


def make_groups(context: Sequence[Observation]) -> tuple[Group, ...]:
    """Dedupe by measurement_id, group by identical action, canonical order."""
    g = _Groups()
    for o in context:
        g.add(o)
    return g.groups()


# ----------------------------------------------------------------------------- quadrature


def _forward(A, B, F, T):
    """Clipped analytic displacement in line form F T^2 max(A F - B, 0); the same function as
    physics.analytic_displacement (A = 1/(2 m^2 mu g), B = 1/(2 m)), cheaper when A and B are
    shared by many groups (tested equal to rounding)."""
    return F * T * T * np.maximum(A * F - B, 0.0)


def _loglik(m, mu, groups: tuple[Group, ...], sigma_d: float):
    """Sum over groups of log N(ybar; f, sigma_d^2/n + tau^2); arrays broadcast (m, mu).
    In-place arithmetic on small row blocks (see _inner) keeps the temporaries in cache."""
    shape = np.broadcast(m, mu).shape
    A = np.broadcast_to(1.0 / (2.0 * m * m * mu * G), shape)
    B = np.broadcast_to(1.0 / (2.0 * m), shape)
    ll = np.full(shape, -0.5 * math.log(2.0 * math.pi) * len(groups))
    f, t = np.empty(shape), np.empty(shape)
    for g in groups:
        T = g.k * DT
        np.multiply(A, g.F, out=f)  # f = F T^2 max(A F - B, 0)  (= _forward)
        f -= B
        np.maximum(f, 0.0, out=f)
        f *= g.F * T * T
        np.multiply(f, R_MODEL, out=t)  # tau = r_model f + tau_abs
        t += TAU_ABS
        if sigma_d != 0.0:
            np.multiply(t, t, out=t)  # var = sigma_d^2/n + tau^2
            t += sigma_d * sigma_d / g.n
            np.subtract(g.ybar, f, out=f)
            f *= f
            f /= t
            f *= 0.5
            np.log(t, out=t)
            t *= 0.5
        else:  # var = tau^2: -0.5 log var = -log tau
            np.subtract(g.ybar, f, out=f)
            f /= t
            f *= f
            f *= 0.5
            np.log(t, out=t)
        ll -= f
        ll -= t
    return ll


def _logsumexp(a) -> float:
    a = np.asarray(a, np.float64)
    mx = a.max()
    return float(np.log(np.sum(np.exp(a - mx))) + mx)


def _inner(u, groups, query: Action, prior: Prior, n: int, sigma_d: float):
    """For each u: log of int L(u, mu) dmu over the prior mu range, and the conditional first
    and second moments of f(theta, query). Returns (logZ, M1, M2), each shape (len(u),)."""
    u = np.asarray(u, np.float64)
    m = np.exp(u)
    w_lo, w_hi = 1.0 / prior.mu_hi, 1.0 / prior.mu_lo
    if groups:
        F = np.array([g.F for g in groups])
        T = np.array([g.k * DT for g in groups])
        y = np.array([g.ybar for g in groups])
        nn = np.array([g.n for g in groups], np.float64)
        tau_y = R_MODEL * np.abs(y) + TAU_ABS
        omega = 1.0 / (sigma_d * sigma_d / nn + tau_y * tau_y)
        c1 = (F * F * T * T)[None, :] / (2.0 * m[:, None] ** 2 * G)
        c0 = (F * T * T)[None, :] / (2.0 * m[:, None])
        P = np.sum(omega * c1 * c1, axis=1)
        w_hat = np.sum(omega * c1 * (y + c0), axis=1) / P
        s = 1.0 / np.sqrt(P)
        w_star = np.clip(w_hat, w_lo, w_hi)
        R = np.sqrt((w_star - w_hat) ** 2 + (K_WINDOW * s) ** 2)
        mu_a = 1.0 / np.minimum(w_hi, w_hat + R)
        mu_b = 1.0 / np.maximum(w_lo, w_hat - R)
    else:
        mu_a = np.full(u.shape, prior.mu_lo)
        mu_b = np.full(u.shape, prior.mu_hi)

    frac = (np.arange(n) + 0.5) / n
    rows = np.arange(u.size)
    logZ, M1, M2 = np.empty(u.size), np.empty(u.size), np.empty(u.size)
    for _ in range(20):
        bad = []
        for i in range(0, rows.size, ROW_BLOCK):
            r = rows[i : i + ROW_BLOCK]
            mm = m[r, None]
            mu = mu_a[r, None] + (mu_b[r] - mu_a[r])[:, None] * frac[None, :]
            ll = _loglik(mm, mu, groups, sigma_d)
            mx = ll.max(axis=1)
            bad_lo = (mu_a[r] > prior.mu_lo) & (ll[:, 0] > mx - INNER_EDGE_DROP)
            bad_hi = (mu_b[r] < prior.mu_hi) & (ll[:, -1] > mx - INNER_EDGE_DROP)
            ll -= mx[:, None]
            wgt = np.exp(ll, out=ll)
            sw = wgt.sum(axis=1)
            fq = analytic_displacement(mm, mu, query.F, query.T)
            logZ[r] = np.log(sw) + mx + np.log((mu_b[r] - mu_a[r]) / n)
            wgt *= fq
            M1[r] = wgt.sum(axis=1) / sw
            wgt *= fq
            M2[r] = wgt.sum(axis=1) / sw
            bad.append(r[bad_lo | bad_hi])
        rows = np.concatenate(bad)
        if rows.size == 0:
            return logZ, M1, M2
        # widen the failing windows by a factor 4 about their centre, clipped to the prior
        c = 0.5 * (mu_a[rows] + mu_b[rows])
        half = 2.0 * (mu_b[rows] - mu_a[rows])
        mu_a[rows] = np.maximum(prior.mu_lo, c - half)
        mu_b[rows] = np.minimum(prior.mu_hi, c + half)
    raise RuntimeError("inner mu window did not converge")


def _posterior(groups, query: Action, prior: Prior, n: int, sigma_d: float):
    """Outer adaptive composite Gauss-Legendre quadrature in u. Returns (mean, sd, info)."""
    lo, hi = prior.u_lo, prior.u_hi
    min_w = MIN_REL_WIDTH * (hi - lo)
    n_pan = max(1, n // GL_NODES)  # base: n outer nodes in n / GL_NODES panels
    h = np.full(n_pan, (hi - lo) / n_pan)
    left = lo + h * np.arange(n_pan)

    def evaluate(left, h):
        nodes = left[:, None] + h[:, None] * _GL_X[None, :]
        out = _inner(nodes.ravel(), groups, query, prior, n, sigma_d)
        return [a.reshape(nodes.shape) for a in out]

    L, M1, M2 = evaluate(left, h)
    hit_min, resolved, levels = False, False, 0
    for levels in range(MAX_LEVELS + 1):
        Lf = L.ravel()  # panels are sorted by left, nodes ascending: Lf is ordered in u
        sig = Lf >= Lf.max() - NEGLIGIBLE  # >=: stays true when |L| is so large that 50 nats < 1 ulp
        near = sig.copy()
        near[1:] |= sig[:-1]
        near[:-1] |= sig[1:]
        d1 = np.abs(np.diff(Lf))
        jump = np.zeros(Lf.size)
        jump[1:] = d1
        jump[:-1] = np.maximum(jump[:-1], d1)
        # curvature: |d2 log Z / du2| * (local node spacing)^2 from divided differences
        # (exact for quadratics on the non-uniform Gauss-Legendre node set)
        x = (left[:, None] + h[:, None] * _GL_X[None, :]).ravel()
        dx = np.diff(x)
        slope = np.diff(Lf) / dx
        curv = np.zeros(Lf.size)
        if Lf.size > 2:
            d2 = 2.0 * np.diff(slope) / (dx[1:] + dx[:-1])
            curv[1:-1] = np.abs(d2) * np.maximum(dx[1:], dx[:-1]) ** 2
        bad = near & ((jump > JUMP) | (curv > CURV))
        flag = bad.reshape(L.shape).any(axis=1)
        small = h / SPLIT < min_w
        hit_min = hit_min or bool((flag & small).any())
        flag &= ~small
        if not flag.any():
            resolved = True
            break
        idx = np.nonzero(flag)[0]
        sub_h = np.repeat(h[idx] / SPLIT, SPLIT)
        sub_left = np.repeat(left[idx], SPLIT) + sub_h * np.tile(np.arange(SPLIT), idx.size)
        sL, sM1, sM2 = evaluate(sub_left, sub_h)
        keep = ~flag
        left = np.concatenate([left[keep], sub_left])
        h = np.concatenate([h[keep], sub_h])
        L = np.concatenate([L[keep], sL])
        M1 = np.concatenate([M1[keep], sM1])
        M2 = np.concatenate([M2[keep], sM2])
        order = np.argsort(left, kind="stable")
        left, h, L, M1, M2 = left[order], h[order], L[order], M1[order], M2[order]
    if not resolved:
        return None, None, {"resolved": False, "levels": levels, "panels": int(h.size)}
    logw = (np.log(h)[:, None] + np.log(_GL_W)[None, :] + L).ravel()
    lse = _logsumexp(logw)
    p = np.exp(logw - lse)
    p /= p.sum()
    mean = float(np.sum(p * M1.ravel()))
    var = float(np.sum(p * M2.ravel())) - mean * mean
    info = {
        "resolved": True,
        "levels": levels,
        "panels": int(h.size),
        "sig_nodes": int(np.sum(logw >= logw.max() - NEGLIGIBLE)),
        "hit_min_width": hit_min,
        "log_evidence": lse - math.log(hi - lo) - math.log(prior.mu_hi - prior.mu_lo),
    }
    return mean, math.sqrt(max(var, 0.0)), info


@lru_cache(maxsize=200_000)
def _cached(groups, F, k, prior_name, n, sigma_d):
    return _posterior(groups, Action(F, k), get_prior(prior_name), n, sigma_d)


def posterior_mean(groups: tuple[Group, ...], query: Action, *, prior: str = "train",
                   n: int = N_GRID, sigma_d: float = SIGMA_D):
    """(posterior mean of f(theta, query) in m, posterior sd in m, diagnostics). Cached."""
    return _cached(tuple(groups), float(query.F), int(query.k), prior, int(n), float(sigma_d))


def _to_prediction(groups, query, prior, n) -> Prediction:
    try:
        mean, sd, info = posterior_mean(groups, query, prior=prior, n=n)
    except Exception as e:  # the reference never raises (INTERFACES)
        return Prediction(None, "blocked", f"bayes error: {e!r}")
    if not info["resolved"] or mean is None or not np.isfinite(mean):
        return Prediction(None, "blocked", f"bayes quadrature unresolved: {info}")
    note = " (outer cell at minimum width)" if info["hit_min_width"] else ""
    return Prediction(mean, "ok", f"bayes {prior} prior, N={n}, posterior sd {sd:.3e} m{note}")


def bayes_predict(
    context: Sequence[Observation], query: Action, *, prior: str = "train", n: int = N_GRID
) -> Prediction:
    """Posterior-mean query displacement in MuJoCo mode (PROTOCOL §8), metres.

    prior: "train" (training ranges) or "broad" (OOD only). Returns Prediction(d, "ok"), or
    Prediction(None, "blocked", reason) on corrupt input or unresolved quadrature; never raises.
    """
    try:
        groups = make_groups(context)
    except Exception as e:
        return Prediction(None, "blocked", f"bayes error: {e!r}")
    return _to_prediction(groups, query, prior, n)


def bayes_predict_train(context, query):
    """fn(context, query) for the T06 harness (CS282A_BAYES_PREDICTOR=...:bayes_predict_train)."""
    return bayes_predict(context, query, prior="train")


def bayes_predict_broad(context, query):
    """Broad-prior variant for E4 (CS282A_BAYES_PREDICTOR=...:bayes_predict_broad)."""
    return bayes_predict(context, query, prior="broad")


def predict_prefixes(context: Sequence[Observation], query: Action, cs: Sequence[int], *,
                     prior: str = "train", n: int = N_GRID) -> list[Prediction]:
    """Predictions for the prefixes context[:c], c in cs. Group statistics are updated
    incrementally in one pass along the context; results equal bayes_predict per prefix."""
    out: dict[int, Prediction] = {}
    acc = _Groups()
    done = 0
    try:
        for c in sorted(set(cs)):
            if not 0 <= c <= len(context):
                raise ValueError(f"prefix length {c} outside 0..{len(context)}")
            for o in context[done:c]:
                acc.add(o)
            done = c
            out[c] = _to_prediction(acc.groups(), query, prior, n)
    except Exception as e:
        return [Prediction(None, "blocked", f"bayes error: {e!r}") for _ in cs]
    return [out[c] for c in cs]
