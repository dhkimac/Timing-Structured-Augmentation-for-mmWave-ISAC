"""Timing-structured augmentation: packet timings for the training copies of a uniform recording.

A copy keeps, for every requested instant, the nearest unused packet of the uniformly sampled
recording, and then goes through the same front end as a real capture.

  traffic replay      a random segment of a recorded Wi-Fi traffic capture plus the sensing
                      packets an ISAC transmitter injects to keep at least M per frame (the
                      injection schedule of SPARCS, Pegoraro et al., IPSN 2022, Alg. 2)
  trace perturbation  the same, with the traffic stretched in time, cut by bursty on/off gating
                      and the floor of M sensing packets enforced only in some frames
  random removal      (ablation) a fixed share of slots kept at random positions per half frame
  shuffled slots      (ablation) a traffic copy's measured slots moved to random positions
                      inside each half frame (same count per half frame)

Each copy k of recording ``stem`` is seeded with [crc32(stem), k, salt]: copies are identical
across machines and runs.
"""
from __future__ import annotations

import zlib

import numpy as np

from .frontend import FRAME, HOP, SLOT, default_n_slots, slot_index

FAMILIES = ("psu_cs", "library")       # training traffic only; ug stays unseen
BUDGETS = (4, 8, 16, 32)               # M
STRETCH = (1.0 / 3.0, 3.0)             # perturbation: log-uniform time stretch
GATE_P = 0.5                           # perturbation: probability of on/off gating
GATE_OFF = (5.0, 150.0)                # mean off duration, slots
GATE_ON = (10.0, 300.0)                # mean on duration, slots
REMOVAL_KEEP = (0.125, 0.25, 0.5)      # random removal: share of slots kept, copy k -> k mod 3
SALT_PERTURB, SALT_REMOVAL = 7, 51
SALT_SHUFFLE = zlib.crc32(b"shuf")


def copy_rng(stem: str, k: int, salt: int | None = None) -> np.random.Generator:
    seq = [zlib.crc32(stem.encode()), int(k)] + ([] if salt is None else [int(salt)])
    return np.random.default_rng(seq)


# ------------------------------------------------------------------------- traffic timings
def _on_off(rng, n_slots: int, mean_on: float, mean_off: float) -> np.ndarray:
    """Gilbert-Elliott mask over slots, starting on, geometric durations."""
    out = np.empty(n_slots, dtype=bool)
    i, on = 0, True
    while i < n_slots:
        n = min(int(rng.geometric(1.0 / (mean_on if on else mean_off))), n_slots - i)
        out[i:i + n] = on
        i += n
        on = not on
    return out


def traffic_times(trace: np.ndarray, M: int, duration: float, rng, perturb: bool) -> np.ndarray:
    """Packet instants (s, from 0) of a ``duration``-long capture: traffic + injected sensing."""
    span = float(trace[-1] - trace[0])
    n_slots = int(np.floor(duration / SLOT)) + 1
    floor_p = 1.0
    if not perturb:
        offset = float(trace[0]) + float(rng.uniform(0.0, span - duration))
        traffic = trace[(trace >= offset) & (trace <= offset + duration)] - offset
    else:
        stretch = float(np.exp(rng.uniform(np.log(STRETCH[0]), np.log(STRETCH[1]))))
        offset = float(trace[0]) + float(rng.uniform(0.0, span))
        gated = bool(rng.random() < GATE_P)
        if gated:
            mean_off = float(rng.uniform(*GATE_OFF))
            mean_on = float(rng.uniform(*GATE_ON))
        floor_p = float(rng.random())
        seg = duration / stretch
        if int(np.ceil((offset - trace[0] + seg) / span)) <= 1:   # fits in the capture
            traffic = (trace[(trace >= offset) & (trace <= offset + seg)] - offset) * stretch
        else:                                       # it wraps around the capture's end
            rel = (trace - offset) % span
            reps = np.maximum(np.ceil((seg - rel) / span), 0).astype(np.int64)
            idx = np.repeat(np.arange(trace.size), reps)
            j = np.arange(idx.size) - np.repeat(np.cumsum(reps) - reps, reps)
            traffic = np.sort(rel[idx] + j * span) * stretch
        if gated:
            on = _on_off(rng, n_slots, mean_on, mean_off)
            s = slot_index(traffic, n_slots)
            keep = (s >= 0) & (s < n_slots)
            keep[keep] = on[s[keep]]
            traffic = traffic[keep]
    if traffic.size < 2:
        raise ValueError("traffic segment holds fewer than 2 packets")

    # Sensing injection, frame by frame (SPARCS Alg. 2): after the first half of a frame, book
    # M - (packets seen) units in the last slots; a traffic packet in a booked slot replaces it.
    busy = np.zeros(n_slots, dtype=bool)
    s = slot_index(traffic, n_slots)
    busy[s[(s >= 0) & (s < n_slots)]] = True
    used = busy.copy()
    injected = []
    n_frames = (n_slots - FRAME) // HOP + 1 if n_slots >= FRAME else 0
    for f in range(n_frames):
        a = f * HOP
        need = max(M - int(used[a:a + HOP].sum()), 0)
        if floor_p < 1.0 and rng.random() >= floor_p:
            need = 0
        booked = set(range(a + FRAME - 1, a + FRAME - 1 - need, -1))
        for q in range(a + HOP, a + FRAME):
            if q in booked:
                booked.discard(q)
                if not busy[q]:
                    injected.append(q)
                    used[q] = True
            elif busy[q] and booked:
                booked.discard(min(booked))
    return np.sort(np.concatenate([traffic, np.asarray(injected, dtype=np.float64) * SLOT]))


# --------------------------------------------------------------------------- ablation timings
def removal_times(n_slots: int, keep: float, rng) -> np.ndarray:
    """Slot centres (s): round(32 * keep) random slots per half frame (the last, partial half
    frame keeps max(1, round(len * keep)))."""
    per_half = int(round(HOP * keep))
    out = []
    for lo in range(0, n_slots, HOP):
        n = min(HOP, n_slots - lo)
        k = per_half if n == HOP else max(1, int(round(n * keep)))
        out.append(lo + np.sort(rng.choice(n, size=min(k, n), replace=False)))
    return np.concatenate(out).astype(np.float64) * SLOT


def shuffled_times(kept: np.ndarray, allowed: np.ndarray, rng) -> np.ndarray:
    """Slot centres (s): each half frame keeps its number of measured slots, at random positions
    among the slots where the uniform recording has a packet (``allowed``)."""
    n_slots = allowed.size
    lo = np.arange(0, n_slots, HOP)
    counts = np.bincount(kept // HOP, minlength=lo.size)
    out = []
    for w, a in enumerate(lo):
        cand = np.flatnonzero(allowed[a:a + HOP]) + a
        out.append(cand[np.sort(rng.choice(cand.size, size=int(counts[w]), replace=False))])
    return np.concatenate(out).astype(np.float64) * SLOT


# ------------------------------------------------------------------------------ copies
def occupied_slots(times: np.ndarray, n_slots: int) -> np.ndarray:
    """Slots holding at least one packet."""
    out = np.zeros(n_slots, dtype=bool)
    s = slot_index(times, n_slots)
    out[s[(s >= 0) & (s < n_slots)]] = True
    return out


def nearest_packets(cir: np.ndarray, t: np.ndarray, want: np.ndarray):
    """For every requested instant (ascending) the nearest packet not taken yet (ties: earlier)."""
    used = np.zeros(t.size, dtype=bool)
    out = np.empty(want.size, dtype=np.int64)
    for k, tt in enumerate(np.sort(want)):
        lo = int(np.searchsorted(t, tt)) - 1
        hi = lo + 1
        while True:
            cand = [i for i in (hi, lo) if 0 <= i < t.size and not used[i]]
            if cand:
                best = min(cand, key=lambda i: (abs(t[i] - tt), i))
                out[k], used[best] = best, True
                break
            lo, hi = lo - 1, hi + 1
            if lo < 0 and hi >= t.size:
                raise ValueError("ran out of packets")
    order = np.sort(out)
    return cir[..., order, :], t[order]


def traffic_copy(stem: str, k: int, cir, times, traffic: dict, perturb: bool):
    """Copy k of a recording under recorded traffic (replay) or perturbed traffic."""
    rng = copy_rng(stem, k, SALT_PERTURB if perturb else None)
    family = FAMILIES[int(rng.integers(0, len(FAMILIES)))]
    M = int(BUDGETS[int(rng.integers(0, len(BUDGETS)))])
    want = traffic_times(traffic[family], M, float(times[-1] - times[0]), rng, perturb)
    return nearest_packets(cir, times, want + times[0])


def removal_copy(stem: str, k: int, cir, times):
    rng = copy_rng(stem, k, SALT_REMOVAL)
    want = removal_times(int(times[-1] / SLOT), REMOVAL_KEEP[k % 3], rng)
    return nearest_packets(cir, times, want + times[0])


def shuffled_copy(stem: str, k: int, cir, times, traffic: dict):
    """Shuffled slots of training copy k of the proposed set (k < 8: replay, else perturbation).
    Returns (cir, times, n_slots): the copy keeps its source copy's slot grid."""
    _, src_t = traffic_copy(stem, k % 8, cir, times, traffic, perturb=k >= 8)
    n_slots = default_n_slots(src_t)
    measured = occupied_slots(src_t, n_slots)
    allowed = occupied_slots(times, n_slots)
    want = shuffled_times(np.flatnonzero(measured), allowed, copy_rng(stem, k, SALT_SHUFFLE))
    c, t = nearest_packets(cir, times, want)
    return c, t, n_slots
