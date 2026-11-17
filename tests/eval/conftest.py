"""Collection rule for the Bayes-as-model harness test (explicit, documented mechanism).

tests/eval/test_bayes_harness.py is collected ONLY when a valid Bayes predictor is supplied
(env CS282A_BAYES_PREDICTOR="module:attr", or cs182_icl.reference.bayes.bayes_predict returns a
status other than "blocked"). Otherwise the file is not collected and the pytest header says
"bayes unavailable". There is no skip/xfail marker involved.
"""

from cs182_icl.eval.predictors import bayes_predictor_or_none

_BAYES = bayes_predictor_or_none()
BAYES_PREDICTOR = _BAYES  # imported by test_bayes_harness.py


def pytest_report_header(config):
    if _BAYES is None:
        return "T06: bayes unavailable -> tests/eval/test_bayes_harness.py not collected"
    return f"T06: bayes predictor supplied ({_BAYES.name}) -> Bayes-as-model harness test collected"


def pytest_ignore_collect(collection_path, config):
    if collection_path.name == "test_bayes_harness.py" and _BAYES is None:
        return True
    return None
