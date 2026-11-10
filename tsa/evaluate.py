"""Score trained weights: recording accuracy on the held-out DISC-B fold and on DISC-C (overall, per
traffic family, per sensing budget M), and macro F1 on DISC-C; mean +- sd over the runs found.

    python -m tsa.evaluate --data data --weights weights            # released or trained weights
    python -m tsa.evaluate --data data --weights runs_vote --pooling vote
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import Bags, load_splits, recording_bags
from .model import Model


@torch.no_grad()
def predict(model, store: Bags, chunk: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(store), chunk):
        b = store.batch(range(i, min(i + chunk, len(store))))
        out.append(model(b["x"], b["bag"], b["pos"], b["n_bags"]).argmax(1).cpu().numpy())
    return np.concatenate(out)


def macro_f1(y, p, n_classes):
    f1 = []
    for c in range(n_classes):
        tp, fp, fn = ((p == c) & (y == c)).sum(), ((p == c) & (y != c)).sum(), ((p != c) & (y == c)).sum()
        if tp + fp + fn:
            f1.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True, help="folder of f<fold>_s<seed>.pt")
    ap.add_argument("--pooling", default="attention", choices=["attention", "mean", "vote"])
    ap.add_argument("--out", type=Path, default=None, help="per-recording predictions (json)")
    a = ap.parse_args()
    discb, discc = load_splits()
    nc = len(discb["classes"])
    skip = set(discb["not_in_discb_column"]["stems"])
    c_stems = sorted(discc["recordings"])
    c_info = discc["recordings"]
    c_store = Bags(recording_bags(c_stems, c_info, a.data / "maps" / "discc"))
    y_c = np.asarray([c_info[s]["label"] for s in c_stems])
    cells = {"DISC-C": np.ones(len(c_stems), bool)}
    for fam in ("psu_cs", "library", "ug"):
        cells[fam] = np.asarray([c_info[s]["traffic"] == fam for s in c_stems])
    for m in (4, 8, 16, 32):
        cells[f"M={m}"] = np.asarray([c_info[s]["M"] == m for s in c_stems])

    rows, preds = [], {}
    for run in discb["runs"]:
        name = f"f{run['fold']}_s{run['seed']}"
        path = a.weights / f"{name}.pt"
        if not path.exists():
            continue
        model = Model(nc, a.pooling)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        test = [s for s in run["test"] if s not in skip]
        b_store = Bags(recording_bags(test, discb["recordings"], a.data / "maps" / "discb"))
        p_b, p_c = predict(model, b_store), predict(model, c_store)
        y_b = np.asarray([discb["recordings"][s]["label"] for s in test])
        row = {"DISC-B": float((p_b == y_b).mean())}
        row.update({k: float((p_c[m] == y_c[m]).mean()) for k, m in cells.items()})
        row["DISC-C F1"] = macro_f1(y_c, p_c, nc)
        rows.append(row)
        preds[name] = {"discb": dict(zip(test, p_b.tolist())), "discc": dict(zip(c_stems, p_c.tolist()))}
    if not rows:
        raise SystemExit(f"no weights f<fold>_s<seed>.pt in {a.weights}")

    print(f"{len(rows)} runs, recording accuracy (mean +- sd over runs)")
    for k in rows[0]:
        v = np.asarray([r[k] for r in rows])
        sd = v.std(ddof=1) if v.size > 1 else 0.0
        print(f"  {k:10s} {v.mean():.3f} +- {sd:.3f}")
    pc = np.concatenate([[p for p in preds[n]["discc"].values()] for n in preds])
    yc = np.tile(y_c, len(preds))
    print("DISC-C F1 per class, pooled over runs")
    for c, cname in enumerate(discb["classes"]):
        tp, fp, fn = ((pc == c) & (yc == c)).sum(), ((pc == c) & (yc != c)).sum(), ((pc != c) & (yc == c)).sum()
        print(f"  {cname:17s} {2 * tp / max(2 * tp + fp + fn, 1):.3f}")
    if a.out:
        a.out.write_text(json.dumps(preds, indent=1))


if __name__ == "__main__":
    main()
