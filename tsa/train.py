"""Train the proposed model (or an ablation) for one or all of the 20 runs.

    python -m tsa.train --data data --out runs                 # all 20 runs, proposed method
    python -m tsa.train --data data --out runs --runs f1_s42   # one run
    python -m tsa.train --data data --out runs_none --copies none          # Table II ablations:
    python -m tsa.train --data data --out runs_vote --pooling vote         #   --copies / --pooling

Each run writes <out>/f<fold>_s<seed>.pt; score them with ``python -m tsa.evaluate``.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import (COPY_SETS, Bags, CopySwap, augment, epoch_batches, generators, load_splits,
                   recording_bags)
from .model import Model

EPOCHS = 100
AVERAGE_FROM = 51          # weights averaged over epochs 51..100 (1-indexed)
LR = 1.9322544016001175e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
COPY_P = 0.5               # probability that a recording is replaced by one of its copies
BN_BATCHES = 500           # batches used to re-estimate the batch-norm statistics


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def reestimate_bn(model, store, seed: int):
    """Batch-norm statistics recomputed for the averaged weights: fresh cumulative averages over
    BN_BATCHES augmented training batches (the sampler continuing after the last epoch)."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    model.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    done, epoch = 0, EPOCHS
    while done < BN_BATCHES:
        if isinstance(store, CopySwap):
            store.set_epoch(epoch)
        gens = generators(seed, epoch)
        for idx in epoch_batches(store.groups, seed, epoch):
            b = augment(store.batch(idx), store, gens, clamp_to_recording=True)
            model.embed(model.encoder(b["x"]), b["bag"], b["pos"], b["n_bags"])
            done += 1
            if done == BN_BATCHES:
                break
        epoch += 1
    for m in bns:
        m.momentum = 0.1
    model.eval()


def train_run(run: dict, data: Path, copies: str = "mix", pooling: str = "attention",
              log=print) -> Model:
    seed = int(run["seed"])
    seed_all(seed)
    torch.use_deterministic_algorithms(True)
    discb, _ = load_splits()
    bags = recording_bags(run["train"], discb["recordings"], data / "maps" / "discb")
    store = (Bags(bags) if copies == "none"
             else CopySwap(bags, data / "copies", copies, seed, COPY_P))
    model = Model(len(discb["classes"]), pooling)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    avg, n_avg = None, 0
    for epoch in range(EPOCHS):
        t0 = time.time()
        if isinstance(store, CopySwap):
            store.set_epoch(epoch)
        gens = generators(seed, epoch)
        plan = epoch_batches(store.groups, seed, epoch)
        # The released weights were trained with a loop that drew 1 (first epoch) or 2 (later
        # epochs) seeds from the global generator per epoch; the same draws keep dropout identical.
        for _ in range(1 if epoch == 0 else 2):
            torch.empty((), dtype=torch.int64).random_()
        model.train()
        total = 0.0
        for idx in plan:
            loss = model.loss(augment(store.batch(idx), store, gens))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total += float(loss)
        if epoch + 1 >= AVERAGE_FROM:
            with torch.no_grad():
                if avg is None:
                    avg = {k: p.detach().double().clone() for k, p in model.named_parameters()}
                else:
                    for k, p in model.named_parameters():
                        avg[k] += p.detach().double()
            n_avg += 1
        log(f"epoch {epoch + 1:3d}  loss {total / len(plan):.4f}  {time.time() - t0:.1f}s")
    with torch.no_grad():
        for k, p in model.named_parameters():
            p.copy_((avg[k] / float(n_avg)).to(p.dtype))
    reestimate_bn(model, store, seed)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="output folder of tsa.prepare")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", nargs="*", default=None, help="e.g. f1_s42 (default: all 20)")
    ap.add_argument("--copies", default="mix", choices=["none", *COPY_SETS])
    ap.add_argument("--pooling", default="attention", choices=["attention", "mean", "vote"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    discb, _ = load_splits()
    for run in discb["runs"]:
        name = f"f{run['fold']}_s{run['seed']}"
        if a.runs and name not in a.runs:
            continue
        with open(a.out / f"{name}.log", "w") as fh:
            def log(msg):
                fh.write(msg + "\n")
                fh.flush()
            t0 = time.time()
            model = train_run(run, a.data, a.copies, a.pooling, log)
            torch.save(model.state_dict(), a.out / f"{name}.pt")
            print(f"{name}: done in {time.time() - t0:.0f}s -> {a.out / (name + '.pt')}")


if __name__ == "__main__":
    main()
