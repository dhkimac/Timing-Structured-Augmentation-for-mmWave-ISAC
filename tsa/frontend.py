"""Front end: one recording's CIR packets -> a micro-Doppler map (T frames x 64 Doppler bins, dB).

    CIR packets (beams, packets, range bins) at their recorded times
      -> clutter removal: 4th-order Butterworth high-pass along the packet axis
      -> slot grid: 0.27 ms slots, each takes the nearest packet, empty slots stay zero
      -> range/beam gate per frame of 64 slots (hop 32): strongest moving reflection
      -> spectrum per frame by sparse recovery on the measured slots only (IHT)
      -> dB, a fixed 60 dB range below the recording's peak, DC in the middle

The sparse-recovery step and the slot grid follow SPARCS (Pegoraro et al., IPSN 2022) and the
DISC dataset (Pegoraro et al., IEEE Commun. Mag. 2025); this file is our own implementation.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter

SLOT = 0.27e-3          # s, slot duration T of the 802.11ay testbed
FRAME = 64              # slots per frame (one spectrum)
HOP = 32                # slots between frames
N_RANGE = 120           # range bins kept from FRAMES/CIR
MTI_ORDER = 4
# High-pass cut-off 0.097 m/s (39.1 Hz at 60.48 GHz), as a fraction of the slot-rate Nyquist.
MTI_WN = 0.02111586206896552
GATE_LO = 24            # first range bin searched by the gate (~1 m)
GATE_HI = 119           # last range bin searched
GATE_HALF = 4           # gate = [centre - 4, centre + 4) range bins
GATE_MEDIAN = 5         # centred running median over frames of the gate centre
IHT_SPARSITY = 3        # non-zero Doppler bins per range bin
IHT_MIN_SLOTS = 3       # fewer measured slots in a frame -> empty spectrum
IHT_MAX_IT = 300
IHT_TOL = 1e-4
DB_EPS = 1e-8
DB_RANGE = 60.0


def clutter_removal(cir: np.ndarray) -> np.ndarray:
    """High-pass every (beam, range bin) series along the packet axis."""
    b, a = butter(MTI_ORDER, MTI_WN, "high")
    return lfilter(b, a, cir, axis=1)


def slot_index(times: np.ndarray, n_slots: int) -> np.ndarray:
    """Slot of each instant: slot j holds [(j - 0.5) T, (j + 0.5) T), slot 0 holds [0, 0.5 T).
    Values outside [0, n_slots) mean the instant falls off the grid."""
    edges = np.concatenate(([0.0], (np.arange(n_slots) + 0.5) * SLOT, [n_slots * SLOT]))
    return np.digitize(np.asarray(times, dtype=np.float64), edges) - 1


def default_n_slots(times: np.ndarray) -> int:
    return int(times[-1] / SLOT)


def slot_grid(samples: np.ndarray, times: np.ndarray, n_slots: int | None = None):
    """(..., packets, bins) at ``times`` -> (..., n_slots, bins) on the slot grid, and the mask of
    measured slots. Each slot takes its nearest packet (ties: the earlier one); no interpolation."""
    t = np.asarray(times, dtype=np.float64)
    n_slots = default_n_slots(t) if n_slots is None else int(n_slots)
    slot = slot_index(t, n_slots)
    inside = np.flatnonzero((slot >= 0) & (slot < n_slots))
    grid = np.zeros(samples.shape[:-2] + (n_slots, samples.shape[-1]), dtype=np.complex128)
    mask = np.zeros(n_slots, dtype=bool)
    if inside.size:
        s = slot[inside]
        gap = np.abs(s * SLOT - t[inside])
        order = np.argsort(s, kind="stable")
        s, gap, pk = s[order], gap[order], inside[order]
        slots, first, count = np.unique(s, return_index=True, return_counts=True)
        nearest = gap == np.repeat(np.minimum.reduceat(gap, first), count)
        group = np.repeat(np.arange(slots.size), count)
        hit = np.flatnonzero(nearest)
        chosen = pk[hit[np.unique(group[hit], return_index=True)[1]]]
        grid[..., slots, :] = np.take(samples, chosen, axis=-2)
        mask[slots] = True
    return grid, mask


def gate(grid: np.ndarray):
    """Per frame (start slot, beam, first range bin, last range bin + 1) of the strongest moving
    reflection: the argmax of the mean high-passed power over beams x range bins [24, 119], its
    range bin smoothed by a centred running median over 5 frames, then the strongest beam there."""
    n_beams, n_slots, n_bins = grid.shape
    starts = list(range(0, n_slots - FRAME + 1, HOP))
    if not starts:
        raise ValueError(f"recording shorter than one frame ({n_slots} slots)")
    power = [np.mean(np.abs(grid[:, i:i + FRAME, :]) ** 2, axis=1) for i in starts]
    raw = []
    for p in power:
        sub = p[:, GATE_LO:GATE_HI + 1]
        raw.append(GATE_LO + int(np.unravel_index(int(np.argmax(sub)), sub.shape)[1]))
    raw = np.asarray(raw, dtype=np.int64)
    half = GATE_MEDIAN // 2
    frames = []
    for f, (i, p) in enumerate(zip(starts, power)):
        centre = int(round(float(np.median(raw[max(0, f - half):f + half + 1]))))
        beam = int(np.argmax(p[:, centre]))
        frames.append((i, beam, max(0, centre - GATE_HALF), min(centre + GATE_HALF, n_bins)))
    return frames


# Unitary inverse DFT, built as powers of the root of unity.
_W = np.exp(-2j * np.pi * np.arange(FRAME) / FRAME).reshape(-1, 1)
INV_DFT = (_W ** np.arange(FRAME) / np.sqrt(FRAME)).conj()


def iht(A: np.ndarray, y: np.ndarray, s: int) -> np.ndarray:
    """Iterative hard thresholding for y = A x with x s-sparse (unit step size)."""
    x = np.zeros(A.shape[1], dtype=np.complex128)
    for it in range(1, IHT_MAX_IT + 1):
        x_new = x + A.conj().T @ (y - A @ x)
        x_new[np.argpartition(np.abs(x_new), -s)[:-s]] = 0.0
        done = np.linalg.norm(x_new - x) < IHT_TOL * np.linalg.norm(x_new)
        x = x_new
        if done:
            break
    return x


def frame_spectrum(paths: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """(64 slots, range bins) of one frame -> (64,) Doppler magnitude summed over range bins,
    recovered from the measured slots only: y = rows of the unitary inverse DFT at those slots."""
    if int(measured.sum()) < IHT_MIN_SLOTS:
        return np.zeros(FRAME)
    A = INV_DFT[measured]
    y = paths[measured]
    return sum(np.abs(iht(A, y[:, j], IHT_SPARSITY)) for j in range(paths.shape[1]))


def micro_doppler(cir: np.ndarray, times: np.ndarray, n_slots: int | None = None) -> np.ndarray:
    """(beams, packets, range bins) complex at ``times`` (s) -> (T, 64) float32 dB map."""
    grid, mask = slot_grid(clutter_removal(cir), times, n_slots)
    rows = []
    for i, beam, lo, hi in gate(grid):
        spec = frame_spectrum(grid[beam, i:i + FRAME, lo:hi], mask[i:i + FRAME])
        rows.append(np.fft.fftshift(20.0 * np.log10(spec + DB_EPS)))
    rows = np.stack(rows)
    return np.maximum(rows, rows.max() - DB_RANGE).astype(np.float32)
