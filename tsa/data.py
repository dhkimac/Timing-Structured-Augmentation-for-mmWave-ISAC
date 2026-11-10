"""Maps -> bags of windows, the training-copy swap, the batch sampler and the window augmentation.

A recording is one bag: its map is min-max normalised, cut into windows of 200 frames (~1.7 s)
with a stride of 100, and each window is resampled to 64 x 64 (Doppler x time).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

WINDOW, STRIDE, SIZE = 200, 100, 64

# Training copies of each recording, as (copy set directory, copy index) for k = 0..15.
COPY_SETS = {
    "mix": [("replay", k) for k in range(8)] + [("perturb", k) for k in range(8)],
    "replay": [("replay", k) for k in range(16)],
    "removal": [("removal", k) for k in range(16)],
    "shuffled": [("shuffled", k) for k in range(16)],
}


def load_map(path: Path) -> np.ndarray:
    x = np.load(path).astype(np.float64)
    lo, hi = x.min(), x.max()
    return ((x - lo) / max(hi - lo, 1e-8)).astype(np.float32)


def resample(win: np.ndarray) -> np.ndarray:
    """(200, 64) -> (64 Doppler, 64 time) by linear interpolation along time."""
    pos = np.linspace(0.0, win.shape[0] - 1, SIZE)
    i0 = np.floor(pos).astype(np.int64)
    w = (pos - i0)[:, None]
    i1 = np.minimum(i0 + 1, win.shape[0] - 1)
    return np.ascontiguousarray((win[i0] * (1.0 - w) + win[i1] * w).T, dtype=np.float32)


@dataclass
class Bag:
    stem: str
    group: str
    label: int
    x: np.ndarray            # (windows, 64, 64)
    source: np.ndarray       # (frames, 64) the normalised map, for speed scaling
    starts: np.ndarray       # first frame of each window


def make_bag(stem: str, group: str, label: int, m: np.ndarray) -> Bag:
    """Short recordings give one zero-padded window."""
    starts = list(range(0, m.shape[0] - WINDOW + 1, STRIDE))
    if not starts:
        starts = [0]
        m = np.pad(m, ((0, WINDOW - m.shape[0]), (0, 0)))
    x = np.stack([resample(m[s:s + WINDOW]) for s in starts])
    return Bag(stem, group, int(label), x, m, np.asarray(starts, np.int64))


class Bags:
    """All bags of one set, collated into tensors; ``batch`` gathers bags by index."""

    def __init__(self, bags, device="cpu"):
        self.device = torch.device(device)
        self.set(bags)

    def set(self, bags):
        self.bags = list(bags)
        self.stems = [b.stem for b in self.bags]
        self.groups = [b.group for b in self.bags]
        dev = self.device
        nw = [b.x.shape[0] for b in self.bags]
        self.n_windows = torch.tensor(nw, dtype=torch.long, device=dev)
        self.offsets = torch.cumsum(torch.tensor([0] + nw[:-1]), 0).to(dev)
        self.x = torch.from_numpy(np.concatenate([b.x for b in self.bags])).to(dev)
        self.labels = torch.tensor([b.label for b in self.bags], dtype=torch.long, device=dev)
        self.src_len = torch.tensor([b.source.shape[0] for b in self.bags], dtype=torch.long)
        src = np.zeros((len(self.bags), int(self.src_len.max()), SIZE), np.float32)
        for i, b in enumerate(self.bags):
            src[i, :b.source.shape[0]] = b.source
        self.src = torch.from_numpy(src).to(dev)
        self.starts = torch.from_numpy(np.concatenate([b.starts for b in self.bags])).to(dev)

    def __len__(self):
        return len(self.bags)

    def batch(self, idx):
        idx = torch.as_tensor(list(idx), dtype=torch.long, device=self.device)
        nw = self.n_windows[idx]
        bag = torch.repeat_interleave(torch.arange(len(idx), device=self.device), nw)
        pos = torch.arange(int(nw.sum()), device=self.device) - torch.repeat_interleave(
            torch.cumsum(nw, 0) - nw, nw)
        rows = torch.repeat_interleave(self.offsets[idx], nw) + pos
        return {"x": self.x[rows].unsqueeze(1), "bag": bag, "pos": pos, "n_bags": len(idx),
                "labels": self.labels[idx], "owner": idx[bag], "starts": self.starts[rows]}


class CopySwap(Bags):
    """Training bags where, per (seed, epoch, recording), the recording is replaced with
    probability p by one of its 16 training copies (drawn uniformly)."""

    def __init__(self, bags, copies: Path, copy_set: str, seed: int, p: float = 0.5,
                 device="cpu"):
        self.orig, self.copies, self.seed, self.p = list(bags), Path(copies), int(seed), float(p)
        self.copy_list = COPY_SETS[copy_set]
        self.cache, self.epoch = {}, None
        self.device = torch.device(device)
        self.set_epoch(0)

    def draw(self, stem: str, epoch: int):
        rng = np.random.default_rng([self.seed, int(epoch), zlib.crc32(stem.encode()) ^ 0x5A5])
        return float(rng.random()) < self.p, int(rng.integers(0, len(self.copy_list)))

    def set_epoch(self, epoch: int):
        if epoch == self.epoch:
            return
        self.epoch = epoch
        eff = []
        for b in self.orig:
            swap, k = self.draw(b.stem, epoch)
            if swap:
                if (b.stem, k) not in self.cache:
                    d, kk = self.copy_list[k]
                    m = load_map(self.copies / d / f"{b.stem}__k{kk}.npy")
                    self.cache[(b.stem, k)] = make_bag(b.stem, b.group, b.label, m)
                b = self.cache[(b.stem, k)]
            eff.append(b)
        self.set(eff)


# ------------------------------------------------------------------------------- sampler
def epoch_batches(groups, seed: int, epoch: int, n_groups: int = 4, per_group: int = 4):
    """Batches of 16 recordings: 4 (room, day) groups x 4 recordings, every recording once per
    epoch in expectation (per-group shuffled queues)."""
    by_group = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    names = sorted(by_group)
    n_groups = min(n_groups, len(names))
    rng = np.random.default_rng(seed + 10_007 * epoch)
    order = {g: [ix[i] for i in rng.permutation(len(ix))] for g, ix in by_group.items()}
    cursor = dict.fromkeys(names, 0)

    def take(g, k):
        pool, out = order[g], []
        while len(out) < min(k, len(pool)):
            if cursor[g] >= len(pool):
                order[g] = pool = [pool[i] for i in rng.permutation(len(pool))]
                cursor[g] = 0
            n = min(k - len(out), len(pool) - cursor[g])
            out.extend(pool[cursor[g]:cursor[g] + n])
            cursor[g] += n
        return out

    batches = []
    for _ in range(max(1, int(np.ceil(len(groups) / (n_groups * per_group))))):
        chosen = sorted(rng.choice(names, size=n_groups, replace=False).tolist())
        batches.append([i for g in chosen for i in take(g, per_group)])
    return batches


# -------------------------------------------------------------------------- augmentation
P = 0.5
SCALE = (0.8, 1.25)


def generators(seed: int, epoch: int):
    """Per-epoch generators of the two augmentation stages."""
    g1 = torch.Generator().manual_seed(seed + 7919 * epoch)
    h = zlib.crc32(f"a1-dscale:{seed}:{epoch}".encode()) & 0xFFFFFFFF
    g2 = torch.Generator().manual_seed((h ^ 0x5E5) & 0x7FFFFFFF)
    return g1, g2


def shift_and_mask(x, gen):
    """Each with p = 0.5: time shift by up to +-8 frames (edge repeated), one time band of up to
    8 frames set to 0, one Doppler band of up to 6 bins set to 0."""
    n, _, rows, cols = x.shape
    u = torch.rand(n, 8, generator=gen).to(device=x.device, dtype=torch.float64)
    u_s, r_s, u_t, r_t, r_t0, u_d, r_d, r_d0 = u.unbind(1)
    s = torch.where(u_s < P, (r_s * 17).floor().long() - 8, torch.zeros_like(r_s, dtype=torch.long))
    c = torch.arange(cols, device=x.device)
    idx = (c.unsqueeze(0) - s.unsqueeze(1)).clamp(0, cols - 1)
    out = x.gather(3, idx.view(n, 1, 1, cols).expand(n, 1, rows, cols))
    t = (r_t * 9).floor().long()
    t0 = (r_t0 * (cols - t + 1).clamp_min(1).double()).floor().long()
    band = (c >= t0.unsqueeze(1)) & (c < (t0 + t).unsqueeze(1)) & ((u_t < P) & (t > 0)).unsqueeze(1)
    out = out.masked_fill(band.view(n, 1, 1, cols), 0.0)
    f = (r_d * 7).floor().long()
    f0 = (r_d0 * (rows - f + 1).clamp_min(1).double()).floor().long()
    r = torch.arange(rows, device=x.device).unsqueeze(0)
    band = (r >= f0.unsqueeze(1)) & (r < (f0 + f).unsqueeze(1)) & ((u_d < P) & (f > 0)).unsqueeze(1)
    return out.masked_fill(band.view(n, 1, rows, 1), 0.0)


def _bilinear(img, rows, cols, fill):
    """img (N, R, C) sampled at rows (N, R) x cols (N, C'); rows off the grid -> fill (N,)."""
    n, R, C = img.shape
    inside = (rows >= 0) & (rows <= R - 1)
    rc, cc = rows.clamp(0, R - 1), cols.clamp(0, C - 1)
    r0, c0 = rc.floor().long(), cc.floor().long()
    r1, c1 = (r0 + 1).clamp(max=R - 1), (c0 + 1).clamp(max=C - 1)
    fr, fc = (rc - r0.double()).to(img.dtype), (cc - c0.double()).to(img.dtype)

    def at(rr, cc_):
        a = img.gather(1, rr.unsqueeze(2).expand(n, R, C))
        return a.gather(2, cc_.unsqueeze(1).expand(n, R, cc_.shape[1]))

    top = (1 - fc).unsqueeze(1) * at(r0, c0) + fc.unsqueeze(1) * at(r0, c1)
    bot = (1 - fc).unsqueeze(1) * at(r1, c0) + fc.unsqueeze(1) * at(r1, c1)
    out = (1 - fr).unsqueeze(2) * top + fr.unsqueeze(2) * bot
    return torch.where(inside.unsqueeze(2), out, fill.to(out.dtype).view(n, 1, 1).expand_as(out))


def _scale(x, s, fill, src=None, starts=None, src_last=None):
    """Doppler axis x s about DC; with ``src`` also time x 1/s, read from the whole recording."""
    n, _, R, C = x.shape
    s = s.double()
    rows = R // 2 + (torch.arange(R, device=x.device).double() - R // 2).unsqueeze(0) / s.unsqueeze(1)
    if src is None:
        out = _bilinear(x[:, 0], rows, torch.arange(C, device=x.device).double().expand(n, C), fill)
    else:
        t = torch.linspace(0.0, 1.0, C, device=x.device, dtype=torch.float64)
        cols = starts.double().unsqueeze(1) + t.unsqueeze(0) * s.unsqueeze(1) * (WINDOW - 1.0)
        if src_last is not None:
            cols = torch.minimum(cols, src_last.double().unsqueeze(1))
        out = _bilinear(src.permute(0, 2, 1).contiguous(), rows, cols, fill)
    return torch.where((s == 1.0).view(n, 1, 1), x[:, 0], out).unsqueeze(1)


def scale(x, gen, b, store, clamp_to_recording=False):
    """Each with p = 0.5 and its own factor s, log-uniform in [0.8, 1.25]: velocity scaling
    (Doppler x s), then speed scaling (Doppler x s, time x 1/s). Speed scaling re-reads the window
    from the clean recording, so where it fires the earlier steps are not kept. Doppler bins pushed
    off the grid are filled with the window's 1st percentile."""
    n = x.shape[0]
    fill = torch.quantile(x.reshape(n, -1).double(), 0.01, dim=1)
    u = torch.rand(n, 4, generator=gen).to(device=x.device, dtype=torch.float64)
    u_v, r_v, u_s, r_s = u.unbind(1)
    lo, hi = SCALE
    s_v = torch.where(u_v < P, lo * (hi / lo) ** r_v, torch.ones_like(r_v))
    s_s = torch.where(u_s < P, lo * (hi / lo) ** r_s, torch.ones_like(r_s))
    x = _scale(x, s_v, fill)
    owner = b["owner"]
    lens = store.src_len[owner.cpu()]
    last = (lens.to(x.device) - 1) if clamp_to_recording else None
    return _scale(x, s_s, fill, store.src[owner, :int(lens.max())], b["starts"], last)


def augment(b, store, gens, clamp_to_recording=False):
    x = shift_and_mask(b["x"], gens[0])
    return {**b, "x": scale(x, gens[1], b, store, clamp_to_recording)}


# --------------------------------------------------------------------------------- splits
SPLITS = Path(__file__).resolve().parents[1] / "splits"


def load_splits():
    import json
    return (json.loads((SPLITS / "discb_folds.json").read_text()),
            json.loads((SPLITS / "discc_test.json").read_text()))


def recording_bags(stems, info: dict, map_dir: Path):
    """Bags in the given order; DISC-B recordings are grouped by (room, day) for the sampler."""
    out = []
    for s in stems:
        i = info[s]
        group = f"{i['room']}-day{i['day']}" if "room" in i else i["traffic"]
        out.append(make_bag(s, group, i["label"], load_map(Path(map_dir) / f"{s}.npy")))
    return out
