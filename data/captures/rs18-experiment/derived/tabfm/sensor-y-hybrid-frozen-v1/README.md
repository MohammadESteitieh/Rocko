# Frozen Sensor-Y/TabFM hybrid result

This directory preserves the preregistered follow-up to the initial exploratory
TabFM sequence-24 result.

## Frozen contract

- Rocko experiment code: `2f95468b2c9662571909535514a91568a385dea1`.
- TabFM: `google/tabfm-1.0.0-jax`, checkpoint revision
  `d5e74033fcf257699fab013e2cdd7edf424ff904`.
- Backend/device: JAX 0.10.2 on CPU; both 16 GiB RTX 4080 devices were too small
  to restore the model and no parameter-sharded path was available.
- Features: aligned phasors plus Sensor-Y coherent LLR and absolute confidence.
- Context: 100 rows, seed 2026; each query payload repetition excluded.
- Gate: retain Sensor Y unless TabFM assigns at least 0.75 probability to one
  class.
- Exploratory control: sequence 24, repetition 3, 100% duty.
- Confirmatory queries: sequences 2, 10, 28, and 45; repetitions 1, 2, 4, and 5;
  50%, 45%, 25%, and 10% duty.

## Result

The exploratory 100% frame decoded with Sensor Y, TabFM alone, and the gate.
TabFM alone reduced it from 6 bit / 5 symbol errors to 5 bit / 4 symbol errors;
the 0.75 gate made no decision changes and matched Sensor Y.

Across the four confirmatory frames:

| Frontend | Bit errors | Symbol errors | Correct frames |
|---|---:|---:|---:|
| Primary coherent | 107 | 54 | 0/4 |
| Sensor Y | 97 | 55 | 0/4 |
| TabFM alone | 103 | 56 | 0/4 |
| Sensor-Y/TabFM 0.75 gate | 98 | 54 | 0/4 |

The gate changed only three Sensor-Y decisions: one correct repair on sequence
10 and two newly introduced bit errors on sequence 28. It did not recover a new
payload. This is a negative confirmatory result for the frozen configuration,
not evidence that every possible TabFM representation or soft-decoding strategy
will fail.

`batch-summary.json` is canonical for the aggregate and includes the experiment
contract, software/device provenance, source hashes, per-frame results, and
prediction hashes. Every artifact has a sibling SHA-256 sidecar.
