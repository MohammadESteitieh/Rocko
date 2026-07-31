# Receiver research pipeline

The receiver acquires two magnetic-sensor channels from a Pico over USB serial,
records immutable CSV captures, and performs live or offline decoding.

## Data contract

```text
t,x,y
0.000000,812,1571
```

- `t` is the Pico sample clock in seconds.
- `x` and `y` are the two ADC channels.
- Expected rate is approximately 200 Hz at 115200 baud.
- Existing datasets contain both 12-bit-range and `read_u16`-range captures;
  analyzers record the inferred full scale and report clipping explicitly.

Install dependencies:

```bash
python3 -m venv receiver/.venv
receiver/.venv/bin/pip install -r receiver/requirements.txt
```

## Acquisition

- `pico_main.py` — MicroPython dual-ADC streamer.
- `capture.py` — authoritative serial capture owner.
- `serial_source.py` — serial and replay source abstraction.
- `watch_capture.py` / `monitor_dataset.py` — read-only monitoring.
- `live_receiver.py` / `rocko_receiver.py` — Hamming-protocol live display.

## Protocol and decoding

- `coded_protocol.py` — Hamming(7,4) alphabet protocol used by calibration.
- `layered_decoder.py` / `hybrid_decoder.py` / `slnn_decoder.py` — Hamming
  matched-filter and codebook decoders.
- `analyze_final_experiment.py` — uncoded, Hamming(15,11), RS(5,3), and
  RS(12,6) analysis.
- `analyze_rs18_experiment.py` — accepted RS(18,6) analysis.
- `compare_final_rs_frontends.py` — hard/GMD, no-whitening, Gao, and Duong
  exploratory comparisons.

## Noise and frontend research

- `duong_whitener.py` — instantaneous gain-modulated IQ whitening.
- `temporal_whitening.py` — VAR, Kalman, GRU, and TCN prediction-error models.
- `benchmark_temporal_whitening.py` — historical physical benchmark.

Frontend parameters must be fitted only on declared transmitter-off data and
frozen during active frames. Do not fit boundaries or decoders using payload
truth.

## Physical runners

- `run_calibration.py`
- `run_final_experiment.py`
- `run_rs18_experiment.py`
- `run_rs18_quick_screen.py`

These scripts own capture, remote launch, manifest transfer, validation,
checksums, and independent GPIO-low cleanup. They require a separate hidden
password prompt or an ephemeral `SSHPASS` environment variable; never store a
password in the repository.

## Tests

```bash
receiver/.venv/bin/python -m unittest discover -s tests -q
```
