"""Evaluation cases: one (item, context length c) unit that every method must predict.

Pairing rules (PILOT_SPEC K1, O):
- E1/E4 store one ordered 10-push context per item; the case for c uses the FIRST c pushes.
- E2/E3 store one row per (cell or copy count, c); rows are linked by item_id.
The scoring target is always TestItem.d_query_clean (clean MuJoCo outcome).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from cs182_icl.types import Action, Cube, Observation, TestItem


@dataclass(frozen=True)
class Case:
    item_id: int  # pairing key: same cube + query across conditions / context lengths
    c: int  # number of context observations the predictor may see
    item: TestItem  # item.context has exactly c observations

    def __post_init__(self):
        if len(self.item.context) != self.c:
            raise ValueError(f"case c={self.c} but context has {len(self.item.context)} obs")

    @property
    def key(self) -> tuple:
        return (self.item_id, self.item.condition, self.c)

    @property
    def target(self) -> float:
        """Clean MuJoCo query displacement (m): the only scoring target."""
        return self.item.d_query_clean


def prefix_cases(rows: Sequence[tuple[int, TestItem]], cs: Iterable[int]) -> list[Case]:
    """E1/E4: for each (item_id, item) with a full ordered context, one case per c (first c)."""
    cs = list(cs)
    out = []
    for item_id, item in rows:
        if len(item.context) < max(cs):
            raise ValueError(f"item {item_id}: context {len(item.context)} < max c {max(cs)}")
        for c in cs:
            out.append(Case(item_id, c, replace(item, context=list(item.context[:c]))))
    return out


def rows_as_cases(rows: Sequence[tuple[int, TestItem]]) -> list[Case]:
    """E2/E3: each stored row is already one case with c = its context length."""
    return [Case(i, len(it.context), it) for i, it in rows]


def load_npz_rows(path: str | Path) -> list[tuple[int, TestItem]]:
    """Read an e*.npz file (docs/INTERFACES.md) into (item_id, TestItem) rows.

    The query target is d_query_clean. d_obs of the context is kept only as context input.
    """
    z = np.load(path, allow_pickle=False)
    rows = []
    for i in range(len(z["item_id"])):
        n = int(z["ctx_len"][i])
        ctx = [
            Observation(
                Action(float(z["ctx_F"][i, j]), int(z["ctx_k"][i, j])),
                int(z["ctx_measurement_id"][i, j]),
                float(z["ctx_d_clean"][i, j]),
                float(z["ctx_d_obs"][i, j]),
            )
            for j in range(n)
        ]
        item = TestItem(
            cube=Cube(float(z["m"][i]), float(z["mu"][i]), int(z["cube_id"][i])),
            query=Action(float(z["query_F"][i]), int(z["query_k"][i])),
            d_query_clean=float(z["d_query_clean"][i]),
            context=ctx,
            condition=str(z["condition"][i]),
        )
        rows.append((int(z["item_id"][i]), item))
    return rows


def fingerprint(cases: Sequence[Case]) -> str:
    """sha256 of everything a predictor can see (not the target), used as cache key."""
    h = hashlib.sha256()
    for cs in cases:
        it = cs.item
        h.update(
            repr(
                (
                    cs.item_id,
                    cs.c,
                    it.condition,
                    it.cube.m,
                    it.cube.mu,
                    it.cube.cube_id,
                    it.query.F,
                    it.query.k,
                    [(o.action.F, o.action.k, o.measurement_id, o.d_obs) for o in it.context],
                )
            ).encode()
        )
    return h.hexdigest()


def assert_prefix_paired(
    cases: Sequence[Case], group_key: Callable[[Case], tuple] | None = None
) -> None:
    """Within each group, cases sorted by c must share cube/query/target and the context of a
    shorter case must be an exact prefix of a longer one (same ordered list).

    Default group: (item_id, condition). For E3 use group_key=lambda cs: (cs.item_id,) since
    the condition encodes the number of appended copies.
    """
    group_key = group_key or (lambda cs: (cs.item_id, cs.item.condition))
    groups = defaultdict(list)
    for cs in cases:
        groups[group_key(cs)].append(cs)
    for key, grp in groups.items():
        grp = sorted(grp, key=lambda cs: cs.c)
        for a, b in zip(grp, grp[1:]):
            if (a.item.cube, a.item.query, a.item.d_query_clean) != (
                b.item.cube,
                b.item.query,
                b.item.d_query_clean,
            ):
                raise AssertionError(f"group {key}: cube/query/target differ across c")
            if list(b.item.context[: a.c]) != list(a.item.context):
                raise AssertionError(
                    f"group {key}: context at c={a.c} is not a prefix of context at c={b.c}"
                )


def assert_clean_targets(cases: Sequence[Case]) -> None:
    """Targets must be finite and be the item's clean value (never a context d_obs)."""
    for cs in cases:
        if not np.isfinite(cs.target):
            raise AssertionError(f"case {cs.key}: non-finite target")
