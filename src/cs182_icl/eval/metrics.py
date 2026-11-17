"""Metrics against the CLEAN MuJoCo target only (PILOT_SPEC K3, PROTOCOL §11).

Never imputed: predictions with status != "ok" are excluded from error means and counted.
Standardized MSE = MSE / Var_train(d) (denominator = training-set variance of d, from
meta.json normalization). NMSE and gap-to-Bayes exist only when a valid Bayes set is supplied;
its shared denominator D is the Bayes c=0 MSE on the same item set. The two normalisations
have different names and are never mixed.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cs182_icl.eval.cases import Case, assert_clean_targets
from cs182_icl.eval.predictors import Predictor, predict_all
from cs182_icl.types import Prediction


@dataclass
class Scores:
    method: str
    kind: str
    keys: list[tuple]  # Case.key per case
    d_true: np.ndarray  # clean target
    d_pred: np.ndarray  # NaN where status != ok
    status: list[str]
    c: np.ndarray
    item_id: np.ndarray

    @property
    def ok(self) -> np.ndarray:
        return np.array([s == "ok" for s in self.status], dtype=bool)

    @property
    def sq_err(self) -> np.ndarray:
        return (self.d_pred - self.d_true) ** 2  # NaN where not ok


def score_cases(method: str, kind: str, cases: Sequence[Case], preds: Sequence[Prediction]) -> Scores:
    assert_clean_targets(cases)
    d_pred = np.array(
        [p.d_pred if p.status == "ok" else np.nan for p in preds], dtype=np.float64
    )
    return Scores(
        method=method,
        kind=kind,
        keys=[cs.key for cs in cases],
        d_true=np.array([cs.target for cs in cases], dtype=np.float64),
        d_pred=d_pred,
        status=[p.status for p in preds],
        c=np.array([cs.c for cs in cases]),
        item_id=np.array([cs.item_id for cs in cases]),
    )


def evaluate_all(
    cases: Sequence[Case], predictors: Sequence[Predictor], cache=None
) -> dict[str, Scores]:
    """Run every predictor over the SAME cases (cached for references) and check alignment."""
    out = {}
    for p in predictors:
        if p.name in out:
            raise ValueError(f"duplicate predictor name {p.name}")
        preds = cache.get_or_compute(p, cases) if cache is not None else predict_all(p, cases)
        out[p.name] = score_cases(p.name, p.kind, cases, preds)
    assert_same_items(out)
    return out


def assert_same_items(scores: dict[str, Scores]) -> None:
    """Every method was scored on identical (item_id, condition, c) keys and identical targets."""
    ref = None
    for name, s in scores.items():
        if ref is None:
            ref = (name, s)
            continue
        if s.keys != ref[1].keys:
            raise AssertionError(f"{name} and {ref[0]} were scored on different items")
        if not np.array_equal(s.d_true, ref[1].d_true):
            raise AssertionError(f"{name} and {ref[0]} have different targets")


# ---------------------------------------------------------------- elementary metrics
def train_d_variance(meta: dict) -> float:
    """Training-set Var(d) from meta.json normalization (computed from train.npz only).

    Reads normalization["d_obs"]["std"] (pilot noise is 0, so d_obs == d_clean); also accepts
    flat "d_obs_std" / "d_std". Raises if absent so a wrong denominator is never guessed.
    """
    norm = meta["normalization"]
    for getter in (
        lambda n: n["d_obs"]["std"],
        lambda n: n["d_obs_std"],
        lambda n: n["d_std"],
    ):
        try:
            return float(getter(norm)) ** 2
        except (KeyError, TypeError):
            continue
    raise KeyError("meta.json normalization has no d_obs std")


def tertile_edges(d_unique: np.ndarray) -> tuple[float, float]:
    """Edges at the 1/3 and 2/3 quantiles (linear interpolation) of the item targets."""
    q1, q2 = np.quantile(np.asarray(d_unique, dtype=np.float64), [1 / 3, 2 / 3])
    return float(q1), float(q2)


def tertile_index(d: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    """0 if d <= q1, 1 if q1 < d <= q2, 2 if d > q2."""
    d = np.asarray(d)
    return (d > edges[0]).astype(int) + (d > edges[1]).astype(int)


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def _median(x: np.ndarray) -> float:
    return float(np.median(x)) if len(x) else float("nan")


def unique_item_targets(s: Scores) -> np.ndarray:
    """One target per item_id (queries are shared across c / cells)."""
    _, idx = np.unique(s.item_id, return_index=True)
    return s.d_true[idx]


# ---------------------------------------------------------------- table
COLUMNS = [
    "group",
    "c",
    "method",
    "kind",
    "n",
    "n_ok",
    "n_not_determined",
    "n_blocked",
    "mse_m2",
    "smse_trainvar",
    "median_rel_err",
    "n_zero_target_excluded_from_rel",
    "mse_tertile1_m2",
    "mse_tertile2_m2",
    "mse_tertile3_m2",
    "n_tertile1_ok",
    "n_tertile2_ok",
    "n_tertile3_ok",
    "nmse_bayes_c0",
    "gap_to_bayes_m2",
]


def bayes_denominator(bayes_c0: Scores | None) -> float | None:
    """D = Bayes c=0 MSE on the item set; None unless Bayes is valid (all ok, nonzero)."""
    if bayes_c0 is None:
        return None
    if not bayes_c0.ok.all():
        return None
    D = _mean(bayes_c0.sq_err)
    return D if D > 0 else None


def metric_table(
    scores: dict[str, Scores],
    var_d: float,
    *,
    group_fn: Callable[[tuple], str] = lambda key: key[1],
    bayes: Scores | None = None,
    bayes_D: float | None = None,
    edges: tuple[float, float] | None = None,
) -> list[dict]:
    """Rows per (group, c, method). `bayes` must be a Scores over the same keys as `scores`
    for gap_to_Bayes; `bayes_D` the shared denominator. Both None -> Bayes columns blank."""
    assert_same_items(scores)
    first = next(iter(scores.values()))
    if bayes is not None and bayes.keys != first.keys:
        raise AssertionError("Bayes scores are not on the same items as the other methods")
    if edges is None:
        edges = tertile_edges(unique_item_targets(first))
    tert = tertile_index(first.d_true, edges)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, key in enumerate(first.keys):
        groups[(group_fn(key), key[2])].append(i)
    rows = []
    for (g, c), idx in groups.items():
        idx = np.array(idx)
        for name, s in scores.items():
            ok = s.ok[idx]
            err = s.sq_err[idx][ok]
            dt = s.d_true[idx][ok]
            dp = s.d_pred[idx][ok]
            nz = dt != 0.0
            rel = np.abs(dp[nz] - dt[nz]) / np.abs(dt[nz])
            t_ok = tert[idx][ok]
            mse = _mean(err)
            row = {
                "group": g,
                "c": c,
                "method": name,
                "kind": s.kind,
                "n": len(idx),
                "n_ok": int(ok.sum()),
                "n_not_determined": sum(s.status[i] == "not_determined" for i in idx),
                "n_blocked": sum(s.status[i] == "blocked" for i in idx),
                "mse_m2": mse,
                "smse_trainvar": mse / var_d,
                "median_rel_err": _median(rel),
                "n_zero_target_excluded_from_rel": int((~nz).sum()),
                "nmse_bayes_c0": "",
                "gap_to_bayes_m2": "",
            }
            for t in range(3):
                row[f"mse_tertile{t + 1}_m2"] = _mean(err[t_ok == t])
                row[f"n_tertile{t + 1}_ok"] = int((t_ok == t).sum())
            if bayes is not None and bayes_D is not None:
                row["nmse_bayes_c0"] = mse / bayes_D
                bok = bayes.ok[idx]
                both = ok & bok
                if both.any():
                    row["gap_to_bayes_m2"] = _mean(
                        s.sq_err[idx][both] - bayes.sq_err[idx][both]
                    )
            rows.append(row)
    return rows


def header_lines(var_d: float, bayes_D: float | None, edges: tuple[float, float], extra=()) -> list[str]:
    lines = [
        f"standardized MSE (smse_trainvar) denominator: training-set Var(d) = {var_d:.6e} m^2 "
        "(meta.json normalization d_obs std squared); same for every method",
        "MSE is against the clean MuJoCo query displacement only; mse_m2 in m^2",
        "not_determined / blocked predictions are counted in n_not_determined / n_blocked and "
        "excluded from error means, never imputed",
        f"displacement tertile edges (m): <= {edges[0]:.6e}, <= {edges[1]:.6e}, above",
    ]
    if bayes_D is None:
        lines.append("bayes: unavailable (nmse_bayes_c0 and gap_to_bayes_m2 omitted)")
    else:
        lines.append(
            f"bayes: available; nmse_bayes_c0 = MSE / D with D = {bayes_D:.6e} m^2 "
            "(training-prior Bayes at c=0, same item set); gap_to_bayes_m2 = MSE_method - MSE_bayes "
            "on shared valid items"
        )
    return lines + list(extra)


def write_csv(path: str | Path, rows: list[dict], columns: Sequence[str], header: Sequence[str] = ()):
    """CSV with '# ' comment header lines (read with csv after skipping lines starting '#')."""
    path = Path(path)
    with open(path, "x", newline="") as f:  # 'x': never overwrite
        for line in header:
            f.write(f"# {line}\n")
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path: str | Path) -> tuple[list[str], list[dict]]:
    with open(path) as f:
        lines = f.read().splitlines()
    comments = [ln[2:] for ln in lines if ln.startswith("# ")]
    rows = list(csv.DictReader(ln for ln in lines if not ln.startswith("#")))
    return comments, rows
