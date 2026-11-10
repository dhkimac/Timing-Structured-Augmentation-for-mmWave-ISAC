"""Reading the DISC .mat files (MAT v7.3 / HDF5, as released on IEEE DataPort)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .frontend import N_RANGE

# Some DISC-B files carry no FRAMES/TIME; their packets are uniform at the median spacing.
DISCB_UNIFORM_SPACING_US = 277.70000000004734

# Recordings of DISC-C without sensing packets: the recorded Wi-Fi traffic used for augmentation.
TRAFFIC_RECORDINGS = {"library": "TEST148_1_M44_F2", "psu_cs": "TEST156_1_M46_F2"}


def read_cir(path: Path):
    """-> CIR (beams, packets, 120 range bins) complex128, packet times (packets,) s or None."""
    with h5py.File(path, "r") as f:
        frames = f["FRAMES"]
        cir = np.asarray(frames["CIR"][0:N_RANGE, :, :]).view("complex")
        times = (np.asarray(frames["TIME"], dtype=np.float64).squeeze()
                 if "TIME" in frames else None)
    return np.ascontiguousarray(np.transpose(cir, (2, 1, 0))), times


def read_discb(path: Path):
    cir, times = read_cir(path)
    if times is None:
        times = np.arange(cir.shape[1], dtype=np.float64) * DISCB_UNIFORM_SPACING_US * 1e-6
    return cir, times


def read_discc(path: Path):
    cir, times = read_cir(path)
    if times is None:
        raise ValueError(f"{path}: DISC-C recording without FRAMES/TIME")
    return cir, times


def read_traffic(discc_dir: Path) -> dict:
    """Packet times of the two recorded traffic captures, {family: (packets,) s}."""
    out = {}
    for family, stem in TRAFFIC_RECORDINGS.items():
        with h5py.File(Path(discc_dir) / f"{stem}.mat", "r") as f:
            out[family] = np.asarray(f["FRAMES"]["TIME"], dtype=np.float64).ravel()
    return out
