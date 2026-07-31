# Research Transmitter

The research transmitter runs on QNX and controls the L298N through text nodes
under `/dev/gpio`.

## Components

- `transmitter/hardware.py` supplies the shared GPIO backend, coil driver,
  Manchester primitives, simulation backend, and process lock.
- `transmitter/calibration_sweep.py` sends controlled duty/rate calibration
  schedules.
- `transmitter/duty_pair_test.py` provides short duty diagnostics.
- `transmitter/hamming_sweep.py` sends the historical Hamming training sweep.
- `transmitter/final_experiment.py` sends the frozen 100-frame comparison.
- `transmitter/rs18_experiment.py` sends the frozen 45-frame RS18 follow-up.
- `transmitter/rs18_quick_screen.py` sends one frame at 100%, 50%, 40%, 30%,
  and 10%.

Protocol constants and golden vectors are separated into
`final_experiment_protocol.py` and `rs18_experiment_protocol.py`.

## Execution controls

Every finite physical tool prints its schedule without touching GPIO unless
`--execute` is supplied. An exclusive pidfile prevents concurrent coil owners.
The manifest records scheduled condition, payload, frame, timestamps, observed
duration, and pulse statistics.

The macOS runners deploy complete transitive file sets, start the single serial
capture owner, wait for capture readiness, preserve any declared prelaunch-off
baseline, launch the QNX process, poll it, retrieve the manifest/log, validate
the schedule, and checksum every artifact.

## Cleanup

All experiment exit paths attempt to force GPIO27, GPIO18, GPIO22, and GPIO17
low. The macOS runner performs a second independent remote cleanup and records
whether those writes were verified.

## Credentials

Do not place a password in a command line, file, manifest, shell history, or
repository. Use a hidden macOS prompt and pass the credential ephemerally to the
SSH process. Stop on DNS, routing, SSH, or remote-host failure; do not
repeatedly retry a physical operation without explicit authorization.
