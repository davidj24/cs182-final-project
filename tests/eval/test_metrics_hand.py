"""Hand-calculated metric fixture (SYNTHETIC numbers, no physics involved).

Six items, targets d (m):      0.01  0.02  0.03  0.04  0.05  0.06
Model predictions (m):         0.012 0.018 0.033 0.036 0.055 0.066
errors pred-true:              +.002 -.002 +.003 -.004 +.005 +.006
squared errors (1e-6 m^2):      4     4     9     16    25    36    sum 94
  MSE = 94e-6 / 6 = 15.6667e-6 m^2
Train Var(d) = 4e-4 m^2 (std 0.02)  ->  standardized MSE = 15.6667e-6 / 4e-4 = 0.0391667
Relative errors |e|/d: 0.2 0.1 0.1 0.1 0.1 0.1 -> sorted median = (0.1+0.1)/2 = 0.1
Tertile edges: quantiles of [.01..06] at 1/3 and 2/3 (linear): position 5/3 -> 0.02+ (2/3)*0.01 =
  0.026667; position 10/3 -> 0.04 + (1/3)*0.01 = 0.043333. Tertiles: {.01,.02}, {.03,.04}, {.05,.06}
  MSE per tertile: (4+4)/2 = 4e-6; (9+16)/2 = 12.5e-6; (25+36)/2 = 30.5e-6
Second method with item index 1 not_determined: MSE over 5 = (94-4)e-6/5 = 18e-6, n_ok 5, n_nd 1
  (never imputed: the item is excluded and counted)
Stub Bayes (synthetic) predicts true + 0.001 everywhere: sq err 1e-6 each, MSE 1e-6 = D
  model NMSE = 15.6667e-6 / 1e-6 = 15.6667 ; gap = 15.6667e-6 - 1e-6 = 14.6667e-6
Paired diffs model - bayes (1e-6): 3 3 8 15 24 35, mean 88/6 = 14.6667 (= the gap)
  deviations from the mean: -11.667 -11.667 -6.667 0.333 9.333 20.333; squares sum 817.33
  var (ddof=1) = 817.33/5 = 163.47, sd = 12.785, SE = 12.785/sqrt(6) = 5.2195
"""

import numpy as np
import pytest

from cs182_icl.eval.cases import Case
from cs182_icl.eval.metrics import (
    COLUMNS, bayes_denominator, evaluate_all, header_lines, metric_table, tertile_edges,
    tertile_index, unique_item_targets, write_csv, read_csv,
)
from cs182_icl.eval.paired import paired_sq_err_diff, summarize_diff
from cs182_icl.types import Action, Cube, Prediction, TestItem

D_TRUE = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
MODEL = [0.012, 0.018, 0.033, 0.036, 0.055, 0.066]
VAR_D = 4e-4


class Table:
    """Predictor returning fixed values by item_id (synthetic)."""

    kind = "reference"

    def __init__(self, name, values, nd=(), kind="reference"):
        self.name, self.values, self.nd, self.kind = name, values, set(nd), kind

    def cache_key(self):
        return None

    def predict(self, case):
        if case.item_id in self.nd:
            return Prediction(None, "not_determined", "fixture")
        return Prediction(self.values[case.item_id], "ok")


def cases(c=0):
    return [
        Case(i, 0, TestItem(Cube(1.0, 0.5, i), Action(15.0, 200), d, [], "fixture"))
        for i, d in enumerate(D_TRUE)
    ]


def run(extra=()):
    ps = [Table("model", MODEL, kind="model"), *extra]
    return evaluate_all(cases(), ps)


def row(rows, method):
    return next(r for r in rows if r["method"] == method)


def test_known_predictions_give_hand_mse():
    rows = metric_table(run(), VAR_D)
    r = row(rows, "model")
    assert r["n_ok"] == 6
    assert r["mse_m2"] == pytest.approx(15.6667e-6, rel=1e-4)
    assert r["smse_trainvar"] == pytest.approx(0.0391667, rel=1e-4)
    assert r["median_rel_err"] == pytest.approx(0.1)
    assert [r[f"mse_tertile{t}_m2"] for t in (1, 2, 3)] == pytest.approx([4e-6, 12.5e-6, 30.5e-6], rel=1e-9)
    assert r["nmse_bayes_c0"] == "" and r["gap_to_bayes_m2"] == ""


def test_tertile_logic():
    d = np.array(D_TRUE)
    edges = tertile_edges(d)
    assert edges == pytest.approx((0.0266667, 0.0433333), rel=1e-5)
    assert list(tertile_index(d, edges)) == [0, 0, 1, 1, 2, 2]
    # values exactly on an edge go to the lower tertile (<=)
    assert list(tertile_index(np.array([edges[0], edges[1]]), edges)) == [0, 1]


def test_not_determined_counted_not_imputed():
    nd = Table("fit", MODEL, nd={1})
    r = row(metric_table(run([nd]), VAR_D), "fit")
    assert (r["n"], r["n_ok"], r["n_not_determined"], r["n_blocked"]) == (6, 5, 1, 0)
    assert r["mse_m2"] == pytest.approx(18e-6, rel=1e-9)


def test_zero_targets_excluded_from_relative_error_only():
    cs = cases()
    cs[0] = Case(0, 0, TestItem(Cube(1, 0.5, 0), Action(15.0, 200), 0.0, [], "fixture"))
    sc = evaluate_all(cs, [Table("model", [0.001] + MODEL[1:])])
    r = metric_table(sc, VAR_D)[0]
    assert r["n_zero_target_excluded_from_rel"] == 1
    assert r["median_rel_err"] == pytest.approx(0.1)
    assert r["n_ok"] == 6  # still in the squared-error metrics


def test_bayes_metrics_when_supplied():
    bayes = Table("bayes_train", [d + 0.001 for d in D_TRUE])
    sc = run([bayes])
    D = bayes_denominator(sc["bayes_train"])
    assert D == pytest.approx(1e-6, rel=1e-9)
    rows = metric_table(sc, VAR_D, bayes=sc["bayes_train"], bayes_D=D)
    r = row(rows, "model")
    assert r["nmse_bayes_c0"] == pytest.approx(15.6667, rel=1e-4)
    assert r["gap_to_bayes_m2"] == pytest.approx(14.6667e-6, rel=1e-4)
    assert row(rows, "bayes_train")["nmse_bayes_c0"] == pytest.approx(1.0)
    diff = paired_sq_err_diff(sc["model"], sc["bayes_train"])
    assert diff * 1e6 == pytest.approx([3, 3, 8, 15, 24, 35])
    s = summarize_diff(diff, seed=0)
    assert s["mean_diff"] == pytest.approx(14.6667e-6, rel=1e-4)
    assert s["se"] == pytest.approx(5.2195e-6, rel=1e-4)
    assert s["ci_lo"] < s["mean_diff"] < s["ci_hi"]


def test_bayes_with_invalid_c0_gives_no_denominator():
    bayes = Table("bayes_train", [d + 0.001 for d in D_TRUE], nd={0})
    assert bayes_denominator(run([bayes])["bayes_train"]) is None
    blocked = [Prediction(None, "blocked", "R1 failed")] * 6
    from cs182_icl.eval.metrics import score_cases
    assert bayes_denominator(score_cases("b", "reference", cases(), blocked)) is None


def test_csv_header_states_denominator_and_bayes_unavailable(tmp_path):
    sc = run()
    edges = tertile_edges(unique_item_targets(next(iter(sc.values()))))
    rows = metric_table(sc, VAR_D, edges=edges)
    p = tmp_path / "m.csv"
    write_csv(p, rows, COLUMNS, header_lines(VAR_D, None, edges))
    comments, parsed = read_csv(p)
    text = "\n".join(comments)
    assert "denominator: training-set Var(d) = 4.000000e-04" in text
    assert "bayes: unavailable" in text
    assert parsed[0]["nmse_bayes_c0"] == ""
    with pytest.raises(FileExistsError):
        write_csv(p, rows, COLUMNS)  # never overwrite
