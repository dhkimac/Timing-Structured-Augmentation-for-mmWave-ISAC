# Timing-Structured Augmentation for Activity Recognition under Irregular Wi-Fi Traffic in mmWave ISAC

An ISAC receiver only sees the channel at the
instants packets are sent, so under real Wi-Fi traffic its micro-Doppler maps are built from bursty,
irregular samples. We train an activity classifier on uniformly sampled recordings and expose it to
that irregularity by training on copies of every recording whose packet timing replays and perturbs
recorded Wi-Fi traffic. The model is a small convolutional window encoder with attention pooling
over the windows of a recording.

## Setup

```bash
pip install -r requirements.txt          # Python 3.12
```

Download DISC from IEEE DataPort ([DOI 10.21227/2gm7-9z72](https://doi.org/10.21227/2gm7-9z72)):
`disc_b.tar.gz` and `disc_c.tar.gz`, and extract them. The code reads their `raw_data/*.mat` files.
The DISC data are subject to their own terms on IEEE DataPort.

## Reproduce

```bash
# 1. micro-Doppler maps of DISC-B and DISC-C, and 16 training copies per DISC-B recording
python -m tsa.prepare --discb DISC_B/raw_data --discc DISC_C/raw_data --out data --workers 16

# 2a. evaluate the released weights (weights/: the 20 runs of the paper)
python -m tsa.evaluate --data data --weights weights

# 2b. or train the 20 runs yourself and evaluate them
python -m tsa.train --data data --out runs
python -m tsa.evaluate --data data --weights runs
```

The released weights are those of the paper's runs. Training is deterministic for a given machine
and software stack, but floating-point differences between machines change the trained weights, so
retrained results agree with the released ones within the spread over runs. One run takes a few
minutes on CPU, while `prepare` is the expensive step (one sparse recovery per frame of every map).

`evaluate` prints the recording accuracy on the held-out DISC-B fold and on DISC-C (overall, per
traffic family and per sensing budget M), the macro F1 on DISC-C, and the per-class F1 pooled over
the runs, as mean ± standard deviation over the runs found.

### Ablations (Table II)

| Variant | prepare | train / evaluate |
|---|---|---|
| No timing copies | | `--copies none` |
| Random sample removal | `--copies removal` | `--copies removal` |
| Same density, shuffled slots | `--copies shuffled` | `--copies shuffled` |
| Traffic replay only | `--copies replay` | `--copies replay` |
| Majority vote over windows | | `--pooling vote` |
| Uniform weights | | `--pooling mean` |

Use a separate `--out` folder per variant, and pass the same `--pooling` to `evaluate`.

## Method in the code

- `tsa/frontend.py`: CIR packets -> micro-Doppler map. Butterworth high-pass along the packets
  (cut-off 0.097 m/s), nearest packet per 0.27 ms slot (empty slots stay empty), a range and beam
  gate per frame of 64 slots (hop 32) on the strongest moving reflection, sparse recovery of the
  Doppler spectrum from the measured slots only (IHT, 3 non-zeros per range bin, summed over the
  8 gated range bins), dB with a fixed 60 dB range below the recording's peak.
- `tsa/timing.py`: the training copies. Copies 0-7 replay a random segment of a recorded traffic
  capture (psu_cs or library; the ug family is never used for training) plus the sensing packets
  an ISAC transmitter injects to keep at least M ∈ {4, 8, 16, 32} per frame. Copies 8-15 perturb it:
  the traffic is stretched in time by a factor in [1/3, 3], cut by bursty on/off gating with
  probability 1/2, and the floor of M sensing packets is enforced only in a random share of the
  frames. Each copy keeps the nearest packets of the uniform recording and goes through the same
  front end.
- `tsa/data.py`: each map is min-max normalised and cut into windows of 200 frames (~1.7 s) with a
  stride of 100, resampled to 64 x 64. During training each recording is replaced with probability
  1/2 by one of its 16 copies (fixed per run, epoch and recording). Batches hold 16 recordings,
  4 from each of 4 (room, day) groups. Each training window is then augmented, in this order:
  1. a time shift of up to ±8 frames, one time band of up to 8 frames and one Doppler band of up
     to 6 bins set to zero, each with probability 1/2;
  2. velocity scaling (Doppler axis x s), with probability 1/2;
  3. speed scaling (Doppler x s and time x 1/s, read from the whole recording), with probability
     1/2. Where it applies, the window is re-read from the clean recording, so steps 1-2 are not
     kept on that window.

  Both factors s are log-uniform in [0.8, 1.25]; Doppler bins pushed off the grid take the window's
  1st percentile.
- `tsa/model.py`: the encoder (3x3 convolutions, batch norm, leaky ReLU; 16/32/64/128 channels at
  stride 2, 128 at stride 1, a 1x1 convolution to 64) and gated attention pooling (hidden size 32),
  dropout 0.2, linear classifier; 0.32 M parameters.
- `tsa/train.py`: Adam, learning rate 1.93e-3, weight decay 1e-4, gradient-norm clipping at 1.0,
  100 epochs. The final weights are the average of epochs 51-100, with the batch-norm statistics
  re-estimated on 500 training batches.

## Data splits

`splits/discb_folds.json`: the 162 DISC-B recordings of walking, running, waving hands and sitting
down / standing up, in 5 folds stratified by class and subject group, each with 4 seeds (20 runs).
Each run holds out 15 % of its training recordings (`val`, monitoring only). The DISC-B accuracy
leaves out 11 recordings that have no released micro-Doppler map, so that it covers the same 151
recordings as the DISC CNN baseline. `splits/discc_test.json`: the 39 DISC-C recordings used for
testing, with their traffic family and sensing budget M.

## Third-party components

The front end follows the processing of the DISC dataset and of SPARCS by Pegoraro et al.: the slot
grid with the nearest packet per slot, the sensing-packet injection schedule (SPARCS, Alg. 2), the
sparse recovery of the micro-Doppler spectrum with IHT, and the random sample removal of the
ablation. The implementation in this repository is our own. The baselines of the paper are not part
of this repository; see the authors' code at https://git2.networks.imdea.org/wng/disc_dataset.

- J. Pegoraro, J. O. Lacruz, M. Rossi, J. Widmer, "SPARCS: A Sparse Recovery Approach for Integrated
  Communication and Human Sensing in mmWave Systems," ACM/IEEE IPSN, 2022.
- J. Pegoraro, P. Saucedo, J. O. Lacruz, M. Rossi, J. Widmer, "DISC: A Dataset for Integrated Sensing
  and Communications in mmWave Systems," IEEE Communications Magazine, 2025.
