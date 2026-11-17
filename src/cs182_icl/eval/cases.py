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


def assert_finite_targets(cases: Sequence[Case]) -> None:
    """Every case's scoring target (item.d_query_clean) must be finite. This checks finiteness
    only; that the target is the clean outcome (not a context d_obs) is by construction in
    Case.target and tested in test_scoring_uses_clean_target_only."""
    for cs in cases:
        if not np.isfinite(cs.target):
            raise AssertionError(f"case {cs.key}: non-finite target")


F_NEAR = 3.0  # N: PILOT_SPEC L1 "at least 3 N from F_q" / "none within 3 N of F_q"


def assert_e2_cells(cases: Sequence[Case]) -> None:
    """E2 structural checks (PILOT_SPEC L1, L2). Raises AssertionError on any violation.

    - force constraints per cell: A all context forces == F_q; B all >= 3 N from F_q;
      C contains F_q and has distinct forces; D distinct forces, none within 3 N of F_q.
    - pairing: for every item_id, all cells share cube, query and d_query_clean, and the A-vs-C
      and B-vs-D pairs have the same c on both sides (every c shared by a pair is present in both).
    """
    by_item: dict[int, dict[str, list[Case]]] = defaultdict(lambda: defaultdict(list))
    for cs in cases:
        cell = cs.item.condition.split("/")[-1]
        if cell not in ("A", "B", "C", "D"):
            raise AssertionError(f"case {cs.key}: unknown E2 cell {cell!r}")
        Fq = cs.item.query.F
        forces = [o.action.F for o in cs.item.context]
        if cell == "A" and any(f != Fq for f in forces):
            raise AssertionError(f"item {cs.item_id} A c={cs.c}: context force != F_q")
        if cell == "B" and any(abs(f - Fq) < F_NEAR for f in forces):
            raise AssertionError(f"item {cs.item_id} B c={cs.c}: a context force within 3 N of F_q")
        if cell in ("C", "D") and len(set(forces)) != len(forces):
            raise AssertionError(f"item {cs.item_id} {cell} c={cs.c}: forces not distinct")
        if cell == "C" and Fq not in forces:
            raise AssertionError(f"item {cs.item_id} C c={cs.c}: context lacks F_q")
        if cell == "D" and any(abs(f - Fq) < F_NEAR for f in forces):
            raise AssertionError(f"item {cs.item_id} D c={cs.c}: a context force within 3 N of F_q")
        by_item[cs.item_id][cell].append(cs)
    for iid, cells in by_item.items():
        flat = [cs for lst in cells.values() for cs in lst]
        ref = flat[0].item
        for cs in flat:
            if (cs.item.cube, cs.item.query, cs.item.d_query_clean) != (
                ref.cube, ref.query, ref.d_query_clean,
            ):
                raise AssertionError(f"item {iid}: cube/query/d_query_clean differ across cells")
        for a, b in (("A", "C"), ("B", "D")):
            ca = {cs.c for cs in cells.get(a, [])}
            cb = {cs.c for cs in cells.get(b, [])}
            if not (ca & cb):
                raise AssertionError(f"item {iid}: {a} vs {b} share no context length c")
            if not cb <= ca:  # every c of the distinct-force cell has a same-force partner
                raise AssertionError(f"item {iid}: {b} has c {sorted(cb - ca)} with no {a} pair")


def assert_e3_copies(cases: Sequence[Case]) -> None:
    """E3 copy semantics: a repeated measurement_id must be an exact copy (same action, d_obs,
    d_clean), so reference deduplication is a no-information-change operation."""
    for cs in cases:
        first: dict[int, Observation] = {}
        for o in cs.item.context:
            if o.measurement_id in first and first[o.measurement_id] != o:
                raise AssertionError(f"case {cs.key}: measurement_id {o.measurement_id} is not an exact copy")
            first.setdefault(o.measurement_id, o)
