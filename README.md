# Rocko magnetic-link research

Rocko is a research codebase for a low-frequency magnetic communication link.
A QNX Raspberry Pi drives a coil through an L298N H-bridge; two magnetic
sensors feed a Raspberry Pi Pico, which streams `t,x,y` samples to a macOS
receiver at approximately 200 Hz.

This repository contains only the research system:

- safe calibration and coded-frame transmitters;
- serial acquisition and offline receiver analysis;
- Hamming and Reed–Solomon protocols and decoders;
- coherent combining, Gao cancellation, Duong whitening, and temporal models;
- reproducible experiment manifests, checksums, plots, and selected captures;
- the project wiki under [`docs/wiki/`](docs/wiki/Home.md).

The original hackathon audio, emergency-intent, photo/CNN, team-automation, and
demo layers have been removed from the research branch. Their history remains
available through Git.

## Repository map

- [`transmitter/`](transmitter/README.md) — QNX GPIO/coil control, calibration
  sweeps, frozen experimental protocols, and physical schedules.
- [`receiver/`](receiver/README.md) — Pico acquisition and experiment
  orchestration, plus the preserved historical pipeline under
  `receiver/legacy_decoder/` while an explainable receiver is rebuilt.
- [`tests/`](tests/) — deterministic protocol, decoder, safety, runner, and
  analysis tests.
- [`docs/wiki/`](docs/wiki/Home.md) — canonical research wiki.
- [`data/captures/`](data/captures/README.md) — selected physical captures,
  manifests, derived results, and a repository-wide checksum inventory.

## Quick start

```bash
python3 -m venv receiver/.venv
receiver/.venv/bin/pip install -r receiver/requirements.txt
receiver/.venv/bin/python -m unittest discover -s tests -q
```

Capture without energizing the transmitter:

```bash
receiver/.venv/bin/python receiver/capture.py \
  --port /dev/cu.usbmodem1201 --out capture.csv --duration 60
```

All physical transmitter tools default to a dry run and require explicit
`--execute`. Follow the safety procedure in
[`docs/wiki/Operations-and-Troubleshooting.md`](docs/wiki/Operations-and-Troubleshooting.md).

## Current research status

The accepted datasets compare uncoded, Hamming(15,11), RS(5,3), RS(12,6), and
RS(18,6) frames at 11 V and 3 m. The strongest accepted RS(18,6) result decoded
all five 100% frames with sensor-y-only or frozen off-RMS processing, but no
frames at 50% or below. Results are small-sample physical measurements, not
claims of general channel performance.

See [`docs/wiki/Context-Handoff.md`](docs/wiki/Context-Handoff.md) for exact
run identifiers, analysis qualifications, and immediate work.
