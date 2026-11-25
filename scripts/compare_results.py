"""Compare two evaluation result folders number by number.

    python scripts/compare_results.py <committed folder> <rerun folder> [--method transformer]

For every CSV in the first folder, rows are matched by position and their text columns checked for
equality. For each file it prints the largest relative difference between numeric cells, separately for
the transformer and for the reference methods. Reference errors near 1e-10 and below are differences of
nearly equal numbers, so their relative differences are expected to be noisier.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("committed", type=Path)
    ap.add_argument("rerun", type=Path)
    a = ap.parse_args()
    for path in sorted(a.committed.glob("*.csv")):
        old, new = read(path), read(a.rerun / path.name)
        if len(old) != len(new):
            print(f"{path.name}: ROW COUNT DIFFERS {len(old)} vs {len(new)}")
            continue
        worst = {"transformer": 0.0, "references": 0.0}
        text_mismatch = 0
        for x, y in zip(old, new):
            who = "transformer" if "transformer" in (x.get("method"), x.get("method_a")) else "references"
            for key in x:
                try:
                    u, v = float(x[key]), float(y[key])
                except (ValueError, TypeError):
                    text_mismatch += x[key] != y.get(key)
                    continue
                if math.isnan(u) and math.isnan(v):
                    continue
                scale = max(abs(u), abs(v))
                rel = 0.0 if u == v else (math.inf if math.isnan(u - v) else abs(u - v) / scale)
                worst[who] = max(worst[who], rel)
        print(f"{path.name:<28} rows={len(old):<3} text mismatches={text_mismatch}  "
              f"max relative difference: transformer {worst['transformer']:.2g}, references {worst['references']:.2g}")


if __name__ == "__main__":
    main()
