# QNX magnetic-link transmitter research

The transmitter runs on a QNX Raspberry Pi and drives an L298N/coil through
`/dev/gpio`. Research frames use 8 Hz polarity switching, OOK Manchester
modulation, and explicit duty control.

## Wiring

| Raspberry Pi GPIO | L298N |
|---|---|
| GPIO22 | IN3 |
| GPIO17 | IN4 |
| GPIO27 | ENB |
| GND | Shared ground |

GPIO18 is also forced low defensively by every physical experiment cleanup.
The coil connects to OUT3/OUT4. Confirm the exact L298N logic-power/jumper
arrangement before any future physical run.

## Files

- `hardware.py` — shared QNX GPIO, coil-driver, Manchester, simulation, and
  process-lock primitives used by the research experiments.
- `alphabet_transmitter.py` — Hamming alphabet transmitter dependency.
- `calibration_sweep.py` — controlled duty/rate calibration schedules.
- `duty_pair_test.py` — duty-pair diagnostics.
- `hamming_sweep.py` — Hamming training/test sweep.
- `final_experiment_protocol.py` / `final_experiment.py` — frozen uncoded,
  Hamming(15,11), RS(5,3), and RS(12,6) experiment.
- `rs18_experiment_protocol.py` / `rs18_experiment.py` — frozen RS(18,6)
  follow-up.
- `rs18_quick_screen.py` — five-frame exploratory screen.

## Safety contract

All physical tools:

1. default to a dry run;
2. require `--execute` before energizing the coil;
3. hold a single-instance lock;
4. finish an active pulse/frame before normal shutdown where applicable;
5. attempt to configure GPIO27, GPIO18, GPIO22, and GPIO17 as outputs and force
   all four low on success, failure, or interruption.

A process return code alone is not proof of a safe hardware state. Physical
runners record `safe_shutdown_outcome=VERIFIED_WRITES` only after independent
remote GPIO writes succeed.

## Deployment

The macOS runners under `receiver/run_*.py` deploy the exact transitive Python
files needed by each schedule. Use those runners rather than manually copying
individual modules; an earlier failed run demonstrated that a missing
transitive import can invalidate an experiment.

Never commit or relay a QNX password. Use a hidden local prompt and keep any
credential only in the launching process environment.
