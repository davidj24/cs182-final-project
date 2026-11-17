"""Generate the pilot dataset (train 1000x32, val 100x32, E1-E4) into a NEW versioned directory.

    python scripts/make_pilot_dataset.py --workers 16 --seed 0

Writes $CS282A_DATA/datasets/pilot_vN (first N that does not exist; never overwrites). Refuses to run
on a dirty git tree unless --allow-dirty. Expected runtime: about 1 minute with 16 workers.
"""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cs182_icl.data.build import Sizes, build_dataset, next_version_dir  # noqa: E402
from cs182_icl.data.store import git_provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default=os.path.join(os.environ.get("CS282A_DATA", ""), "datasets"))
    ap.add_argument("--version", help="default: next free pilot_vN")
    ap.add_argument("--config", default=str(REPO / "configs/pilot/initial_e2e_2026_11_16.json"))
    ap.add_argument("--allow-dirty", action="store_true")
    for f in ("train", "val", "e1", "e2", "e3", "e4"):
        ap.add_argument(f"--{f}", type=int, default=getattr(Sizes(), f))
    a = ap.parse_args()
    if not os.environ.get("CS282A_DATA"):
        sys.exit("CS282A_DATA is not set (source workflow/env.sh)")
    if a.workers > 16:
        sys.exit("--workers must be <= 16")
    root = Path(a.out_root)
    root.mkdir(parents=True, exist_ok=True)
    if a.version:
        version, final = a.version, root / a.version
        if final.exists() or Path(str(final) + ".tmp").exists():
            sys.exit(f"{final} exists; datasets are write-once (omit --version for the next free one)")
    else:
        version, final = next_version_dir(root)
    prov = git_provenance(REPO, require_clean=not a.allow_dirty)
    sizes = Sizes(**{f: getattr(a, f) for f in ("train", "val", "e1", "e2", "e3", "e4")})
    tmp = Path(str(final) + ".tmp")
    meta = build_dataset(tmp, sizes=sizes, seed=a.seed, workers=a.workers, version=version, provenance=prov,
                         config_path=Path(a.config))
    tmp.rename(final)
    print(f"dataset written to {final}")
    print(f"version={version} seed={a.seed} source_git_sha={meta['source_git_sha']} dataset_hash={meta['dataset_hash']}")
    print("counts:", meta["counts"])
    print("flags:", meta["flags"])


if __name__ == "__main__":
    main()
