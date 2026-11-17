"""Shared pytest header for the eval tests. Nothing is hidden from collection: the real-Bayes
test is always collected and shows up as a visible SKIP (with the reason from bayes_predict)
when Bayes is blocked."""

from cs182_icl.eval.predictors import bayes_status

BAYES_PREDICTOR, BAYES_REASON = bayes_status()


def pytest_report_header(config):
    if BAYES_PREDICTOR is None:
        return f"T06: bayes unavailable ({BAYES_REASON}); real-Bayes harness test will SKIP"
    return f"T06: bayes predictor supplied ({BAYES_PREDICTOR.name})"
