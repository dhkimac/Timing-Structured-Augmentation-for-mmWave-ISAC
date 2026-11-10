"""Raw DISC .mat files -> micro-Doppler maps and the timing-structured training copies.

    python -m tsa.prepare --discb DISC_B/raw_data --discc DISC_C/raw_data --out data
    python -m tsa.prepare ... --copies mix replay removal shuffled    # also the Table II copies

Writes data/maps/discb/<stem>.npy (162), data/maps/discc/<stem>.npy (39) and
data/copies/<kind>/<stem>__k<k>.npy. Existing files are kept, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import timing
from .data import load_splits
from .frontend import micro_doppler
from .io import read_discb, read_discc, read_traffic

# copy set -> {kind: number of copies}
NEEDS = {"mix": {"replay": 8, "perturb": 8}, "replay": {"replay": 16},
         "removal": {"removal": 16}, "shuffled": {"shuffled": 16}}

_traffic = None


def _init(discc_dir):
    global _traffic
    _traffic = read_traffic(discc_dir)


def _save(path: Path, m: np.ndarray):
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, m)
    tmp.replace(path)


def discb_job(args):
    stem, src, out, kinds = args
    todo = [(None, 0)] if not (out / "maps" / "discb" / f"{stem}.npy").exists() else []
    todo += [(kind, k) for kind, n in kinds.items() for k in range(n)
             if not (out / "copies" / kind / f"{stem}__k{k}.npy").exists()]
    if not todo:
        return stem, 0
    cir, times = read_discb(src / f"{stem}.mat")
    for kind, k in todo:
        if kind is None:
            _save(out / "maps" / "discb" / f"{stem}.npy", micro_doppler(cir, times))
            continue
        n_slots = None
        if kind in ("replay", "perturb"):
            c, t = timing.traffic_copy(stem, k, cir, times, _traffic, perturb=kind == "perturb")
        elif kind == "removal":
            c, t = timing.removal_copy(stem, k, cir, times)
        else:
            c, t, n_slots = timing.shuffled_copy(stem, k, cir, times, _traffic)
        _save(out / "copies" / kind / f"{stem}__k{k}.npy", micro_doppler(c, t, n_slots))
    return stem, len(todo)


def discc_job(args):
    stem, src, out = args
    dst = out / "maps" / "discc" / f"{stem}.npy"
    if not dst.exists():
        _save(dst, micro_doppler(*read_discc(src / f"{stem}.mat")))
    return stem, 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--discb", type=Path, required=True, help="DISC-B raw_data folder (.mat)")
    ap.add_argument("--discc", type=Path, required=True, help="DISC-C raw_data folder (.mat)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--copies", nargs="+", default=["mix"], choices=list(NEEDS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    kinds = {}
    for c in a.copies:
        for kind, n in NEEDS[c].items():
            kinds[kind] = max(kinds.get(kind, 0), n)
    for d in ["maps/discb", "maps/discc", *[f"copies/{k}" for k in kinds]]:
        (a.out / d).mkdir(parents=True, exist_ok=True)
    discb, discc = load_splits()
    jobs = [(s, a.discb, a.out, kinds) for s in sorted(discb["recordings"])]
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.discc,)) as pool:
        for i, (stem, n) in enumerate(pool.imap_unordered(discb_job, jobs), 1):
            print(f"[{time.time() - t0:7.0f}s] DISC-B {i}/{len(jobs)} {stem}: {n} maps", flush=True)
        cj = [(s, a.discc, a.out) for s in sorted(discc["recordings"])]
        for i, (stem, _) in enumerate(pool.imap_unordered(discc_job, cj), 1):
            print(f"[{time.time() - t0:7.0f}s] DISC-C {i}/{len(cj)} {stem}", flush=True)


if __name__ == "__main__":
    main()
