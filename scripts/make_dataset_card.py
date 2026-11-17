"""Dataset card (gate H4) for a dataset directory -> $CS282A_DATA/gates/H4/attempt-NN (next free NN).

    python scripts/make_dataset_card.py --dataset-dir $CS282A_DATA/datasets/pilot_v1

Re-reads the npz files, re-verifies hashes and normalization, re-runs every construction check
(data/checks.py), and writes card.md, hist.png, card.json. Never overwrites a previous attempt.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cs182_icl.data import checks  # noqa: E402
from cs182_icl.data import sampling as S  # noqa: E402
from cs182_icl.data.store import compute_normalization, content_sha256, file_sha256  # noqa: E402


def hist_text(x, lo, hi, bins=10, width=40):
    h, e = np.histogram(x, bins=bins, range=(lo, hi))
    top = max(h.max(), 1)
    return "\n".join(f"  [{e[i]:7.3f},{e[i+1]:7.3f}) {h[i]:6d} {'#' * int(width * h[i] / top)}" for i in range(bins))


def fmt_item(a, i):
    n = int(a["ctx_len"][i])
    s = (f"  row {i}: item_id={a['item_id'][i]} condition={a['condition'][i]} cube_id={a['cube_id'][i]} "
         f"m={a['m'][i]:.5f} mu={a['mu'][i]:.5f}\n    query F={a['query_F'][i]:.5f} k={a['query_k'][i]} "
         f"d_query_clean={a['d_query_clean'][i]:.6f} t_stop={a['query_t_stop'][i]:.3f} "
         f"timed_out={bool(a['query_timed_out'][i])}\n    context ({n}):\n")
    for j in range(n):
        s += (f"      F={a['ctx_F'][i,j]:.5f} k={a['ctx_k'][i,j]} mid={a['ctx_measurement_id'][i,j]} "
              f"d_clean={a['ctx_d_clean'][i,j]:.6f} d_obs={a['ctx_d_obs'][i,j]:.6f}\n")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-root", default=os.path.join(os.environ.get("CS282A_DATA", ""), "gates/H4"))
    a = ap.parse_args()
    ds = Path(a.dataset_dir)
    meta = json.loads((ds / "meta.json").read_text())
    D = {}
    for n in meta["npz_sha256"]:
        with np.load(ds / n) as z:
            D[n[:-4]] = {k: z[k] for k in z.files}
    root = Path(a.out_root)
    root.mkdir(parents=True, exist_ok=True)
    k = 1
    while (root / f"attempt-{k:02d}").exists():
        k += 1
    out = root / f"attempt-{k:02d}"
    out.mkdir()

    L = [f"# Dataset card: {meta['version']} (gate H4, {out.name})", "",
         f"- dataset dir: {ds}", f"- generation seed: {meta['generation_seed']}; sim workers {meta['sim_workers']}",
         f"- source git SHA: {meta['source_git_sha']} (tree clean at generation: {meta['source_tree_clean']})",
         f"- dataset hash (sha256 of per-file content hashes): {meta['dataset_hash']}",
         f"- pilot config sha256: {meta['pilot_config_sha256']}", f"- noise sigma_d: {meta['noise_sigma_d']}", ""]
    L += ["## Counts and hashes", "", "| file | items/cubes | file sha256 | content sha256 |", "|---|---|---|---|"]
    ok = True
    for n, h in meta["npz_sha256"].items():
        got, cgot = file_sha256(ds / n), content_sha256(D[n[:-4]])
        ok &= got == h and cgot == meta["content_sha256"][n]
        L.append(f"| {n} | {meta['counts'][n[:-4]]} | {got} | {cgot} |")
    assert ok, "hash mismatch between files and meta.json"
    dh = hashlib.sha256("".join(f"{k_}:{v}\n" for k_, v in sorted(meta["content_sha256"].items())).encode()).hexdigest()
    assert dh == meta["dataset_hash"], "dataset hash mismatch"
    L += ["", "All file and content hashes re-verified against meta.json; dataset hash recomputed.", ""]

    res = checks.run_all(D)  # raises on any violation
    L += ["## Assertion results (all passed; each raises on violation)", ""]
    for name, detail in res:
        L.append(f"- {name}: {detail}")
    L += ["", "## Disjointness proof (cube ids)", "", "Cube id ranges (half-open) and set sizes; pairwise intersections are empty (check_cube_ids):", ""]
    for n, r in meta["cube_id_ranges"].items():
        L.append(f"- {n}: ids [{r[0]}, {r[1]}), {len(set(D[n]['cube_id'].tolist()))} distinct")
    L.append("- e4_cellD uses exactly the e4 cubes (its own file; same cube parameters).")
    L += ["", "## Flags (kept and counted, never dropped)", "", "```", json.dumps(meta["flags"], indent=1), "```", "",
          f"Unique simulator rollouts: {meta['n_unique_rollouts']}.", ""]
    norm = compute_normalization(D["train"]["F"], D["train"]["k"], D["train"]["d_obs"])
    assert norm == meta["normalization"], "normalization in meta.json is not the train-only recomputation"
    L += ["## Normalization (train.npz only; recomputed and equal to meta.json)", "", "```", json.dumps(norm, indent=1), "```", ""]

    L += ["## Histograms vs frozen ranges", ""]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    panels = [("m (kg)", lambda a_: a_["m"], S.M_RANGE), ("mu", lambda a_: a_["mu"], S.MU_RANGE),
              ("F (N)", lambda a_: a_["F"].ravel(), S.F_RANGE), ("k (steps)", lambda a_: a_["k"].ravel(), S.K_RANGE)]
    for r, sp in enumerate(("train", "val")):
        for c, (lab, fn, rng_) in enumerate(panels):
            x = fn(D[sp])
            ax[r, c].hist(x, bins=20, range=rng_)
            ax[r, c].set_title(f"{sp}: {lab} in {list(rng_)}")
            L += [f"{sp} {lab}: min {x.min():.4f} max {x.max():.4f} range {list(rng_)}", "```", hist_text(x, *rng_), "```"]
    fig.tight_layout()
    fig.savefig(out / "hist.png", dpi=80)
    t = D["train"]
    cm = np.corrcoef(np.repeat(t["m"], t["F"].shape[1]), t["F"].ravel())[0, 1]
    cmu = np.corrcoef(np.repeat(t["mu"], t["k"].shape[1]), t["k"].ravel())[0, 1]
    L += ["", f"Action/cube independence (train): corr(m, F) = {cm:+.4f}, corr(mu, k) = {cmu:+.4f} (expect ~0).",
          f"Log-uniform mass: fraction of train cubes with m < sqrt(0.5*1.5)={np.sqrt(0.75):.4f}: "
          f"{(t['m'] < np.sqrt(0.75)).mean():.3f} (expect 0.5).", "", "![hist](hist.png)", ""]

    L += ["## Summary of test sets", ""]
    for n in ("e1", "e2", "e3", "e4", "e4_cellD"):
        e = D[n]
        conds = {c: int((e["condition"] == c).sum()) for c in sorted(set(e["condition"].tolist()))}
        L.append(f"- {n}: rows {e['item_id'].shape[0]}, items {len(set(e['item_id'].tolist()))}, "
                 f"m range [{e['m'].min():.3f}, {e['m'].max():.3f}], rows per condition {conds}")
    L += ["", "## Three test items printed in full", "", "### E1 item 0", "```", fmt_item(D["e1"], 0), "```",
          "### E2 item 0 (all nine rows: cells A c=1,2,4; B,C,D c=2,4)", "```"]
    e2 = D["e2"]
    L += [fmt_item(e2, i) for i in np.flatnonzero(e2["item_id"] == e2["item_id"][0])] + ["```", "### E3 item 0 (copies k=0,1,2,4)", "```"]
    e3 = D["e3"]
    L += [fmt_item(e3, i) for i in np.flatnonzero(e3["item_id"] == e3["item_id"][0])] + ["```"]
    (out / "card.md").write_text("\n".join(L) + "\n")
    (out / "card.json").write_text(json.dumps({"checks": [[n, d] for n, d in res], "meta": meta}, indent=1, default=str))
    print(f"card: {out / 'card.md'}\nhist: {out / 'hist.png'}\njson: {out / 'card.json'}")


if __name__ == "__main__":
    main()
