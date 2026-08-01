# Receiver research workspace

The receiver directory is now deliberately divided into two parts:

1. acquisition/orchestration code that remains at `receiver/`;
2. the complete historical decoding and analysis pipeline under
   [`receiver/legacy_decoder/`](legacy_decoder/README.md).

The legacy move does not assert that the previous code is defective. It creates
a stable reference while a smaller, explainable receiver is developed.

## Acquisition and orchestration

- `pico_main.py` — MicroPython dual-ADC streamer.
- `serial_source.py` — serial and replay source abstraction.
- `capture.py` — authoritative serial capture owner.
- `watch_capture.py` — read-only capture watcher.
- `run_calibration.py` — calibration capture and QNX orchestration.
- `run_final_experiment.py` — final-comparison orchestration.
- `run_rs18_experiment.py` — accepted RS18 orchestration.
- `run_rs18_quick_screen.py` — five-frame exploratory orchestration.

Plot-only and table-generation scripts remain at the receiver root because they
consume frozen analysis outputs rather than decide protocol bits.

## Legacy decoding pipeline

`legacy_decoder/` contains the previous:

- Hamming protocols and layered/SLNN/hybrid decoders;
- hard and GMD Reed–Solomon decoders;
- coherent combining and covariance processing;
- Gao and Duong frontends;
- temporal whitening models;
- calibration, final-experiment, and RS18 analyzers;
- live decoder and direct raw-capture visualization.

Historical reproduction command paths now include `legacy_decoder`, for
example:

```bash
receiver/.venv/bin/python receiver/legacy_decoder/analyze_rs18_experiment.py --help
receiver/.venv/bin/python receiver/legacy_decoder/rocko_receiver.py --help
```

## Data contract

```text
t,x,y
0.000000,812,1571
```

- `t` is the Pico sample clock in seconds.
- `x` and `y` are the two ADC channels.
- Expected rate is approximately 200 Hz at 115200 baud.
- Existing datasets contain both 12-bit-range and `read_u16`-range captures;
  analysis must state the assumed full scale and clipping behavior explicitly.

## Rewrite objective

The replacement receiver should expose one inspectable pipeline with explicit
intermediate artifacts for:

- input validation and sample continuity;
- carrier presence per sensor;
- synchronization evidence;
- channel estimation;
- per-bit likelihoods and hard decisions;
- symbol likelihoods and error locations;
- decoder syndrome, correction, failure, and candidate rationale;
- final payload acceptance.

A failed frame must produce an explanation of which stage failed and why—not
only a generic decoder exception or wrong payload.

## Tests

```bash
receiver/.venv/bin/python -m unittest discover -s tests -q
```

Legacy tests remain active so the preserved implementation can serve as a
regression reference during the rewrite.
