# Frozen TabFM soft-list confirmation

This directory preserves the prospective confirmation of the exploratory TabFM
soft-list decoder.

## Frozen contract

- Method-development frames: sequences 24, 2, 10, 28, and 45.
- Confirmatory frames: sequences 3, 18, 32, and 39; payload repetitions 1, 2,
  4, and 5 at 45%, 50%, 50%, and 50% duty.
- Rocko experiment code: `1d8008ca801722072c6870ed2aa16c4a1bd4e3f8`.
- TabFM checkpoint revision: `d5e74033fcf257699fab013e2cdd7edf424ff904`.
- Features: aligned phasors plus Sensor-Y coherent LLR and absolute confidence.
- Context: 100 rows, seed 2026; each query payload repetition excluded.
- Decoder: hard RS first; after hard failure, enumerate all 4,096 erasure
  subsets of the 12 least-reliable TabFM symbols and rank unique codewords by
  TabFM soft score.
- Acceptance: top-two score margin at least 20.
- Runtime: JAX 0.10.2 on CPU.

## Confirmatory result

The soft-list decoder accepted zero of four frames and therefore recovered no
new payloads. It rejected the top candidates with margins 1.024, 13.715, 5.799,
and 5.955 for sequences 3, 18, 32, and 39. A truth-assisted diagnostic performed
only after the frozen result showed that every rejected top candidate was wrong
by 10, 12, 10, and 7 payload bits respectively. The true codeword did not occur
anywhere in the enumerated candidate list for any of the four frames. Thus the
margin rule prevented four potential silent miscorrections, but the exploratory
sequence-2 recovery did not replicate.

Hard-decision aggregate:

| Frontend | Bit errors | Symbol errors | Correct frames |
|---|---:|---:|---:|
| Primary coherent | 85 | 51 | 0/4 |
| Sensor Y | 85 | 51 | 0/4 |
| TabFM | 91 | 58 | 0/4 |
| Sensor-Y/TabFM 0.75 gate | 86 | 51 | 0/4 |

`soft-list-batch-summary.json` is canonical and contains the frozen contract,
software/device provenance, hashes, per-frame outcomes, and aggregate. Every
preserved artifact has a sibling SHA-256 sidecar.
