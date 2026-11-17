"""Reference-prediction cache under $CS282A_DATA/results/<version>/ref_cache.

Files are keyed by predictor name + predictor cache key + fingerprint of the cases, written once
with an atomic no-overwrite link, and reused afterwards (across seeds, across reruns).
Results versions are new directories (results/v1, v2, ...); nothing is overwritten.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from cs182_icl.eval.cases import Case, fingerprint
from cs182_icl.eval.predictors import Predictor, predict_all
from cs182_icl.types import Prediction


def next_results_version(data_root: str | Path) -> Path:
    """Create and return $CS282A_DATA/results/vN with N one above the highest existing."""
    base = Path(data_root) / "results"
    base.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name[1:]) for p in base.glob("v*") if p.name[1:].isdigit()]
    path = base / f"v{max(existing, default=0) + 1}"
    path.mkdir()
    return path


class RefCache:
    def __init__(self, results_dir: str | Path, read_dirs: Sequence[str | Path] = ()):
        self.read_dirs = [Path(d) / "ref_cache" for d in read_dirs]  # earlier versions, read-only
        self.dir = Path(results_dir) / "ref_cache"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, predictor: Predictor, fp: str) -> Path:
        return self.dir / f"{predictor.name}__{predictor.cache_key()}__{fp[:16]}.json"

    def get_or_compute(self, predictor: Predictor, cases: Sequence[Case]) -> list[Prediction]:
        if predictor.cache_key() is None:
            return predict_all(predictor, cases)
        fp = fingerprint(cases)
        path = self._path(predictor, fp)
        for d in self.read_dirs:
            old = d / path.name
            if old.exists():
                path = old
                break
        if path.exists():
            data = json.loads(path.read_text())
            if data["fingerprint"] != fp or len(data["preds"]) != len(cases):
                raise RuntimeError(f"corrupt/mismatched cache file {path}")
            self.hits += 1
            return [Prediction(d, s, r) for d, s, r in data["preds"]]
        self.misses += 1
        preds = predict_all(predictor, cases)
        payload = {
            "predictor": predictor.name,
            "cache_key": predictor.cache_key(),
            "fingerprint": fp,
            "preds": [[p.d_pred, p.status, p.reason] for p in preds],
        }
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        try:
            os.link(tmp, path)  # fails if another run wrote it first: never overwrite
        except FileExistsError:
            pass
        finally:
            os.unlink(tmp)
        return preds
