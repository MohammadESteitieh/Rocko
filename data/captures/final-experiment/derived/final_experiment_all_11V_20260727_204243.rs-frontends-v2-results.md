# Soft GMD and Gao/Duong exploratory comparison

Run: `final_experiment_all_11V_20260727_204243`  
Scope: the 50 physical RS frames only. This is post-hoc exploratory reanalysis,
not untouched confirmatory evidence.

## Frozen comparison

| Method | Description |
|---|---|
| coherent-hard | Covariance-aware coherent LLR plus hard algebraic RS |
| coherent-GMD | Same LLRs plus fixed least-reliable-symbol GMD |
| Gao-GMD | Frozen y-primary/x-reference complex cancellation plus GMD |
| Duong-GMD | Frozen four-dimensional instantaneous I/Q equilibrium whitening plus GMD |
| Gao-Duong-GMD | Gao residual followed by frozen two-dimensional I/Q equilibrium whitening plus GMD |

All methods reused the same manifest boundaries and -0.070-second correction.
Per-frame Gao/Duong models used only the central 10 seconds of the post-frame
off gap and were frozen during the active frame. This is an offline, noncausal
post-gap contract. Payload truth was used only after decoding for scoring.

GMD hard-decides with zero mapped to zero, ranks each five-bit symbol by its
minimum absolute bit LLR, tries the hard attempt plus erasure prefixes one
through the parity-symbol count, algebraically validates and deduplicates every
candidate, and chooses maximum signed-LLR correlation. RS(5,3) candidates with
nonzero fixed application padding are rejected before ranking.

## Overall RS results

| Method | Payload frame errors | Decoder failures | Wrong-codeword miscorrections |
|---|---:|---:|---:|
| Coherent hard | 31/50 | 31 | 0 |
| Coherent GMD | 31/50 | 13 | 18 |
| Gao GMD | 30/50 | 13 | 17 |
| Duong GMD | 31/50 | 13 | 18 |
| Gao-Duong GMD | 30/50 | 13 | 17 |

Soft GMD produced zero paired payload improvements and zero regressions versus
hard decoding: the same 19 frames were correct and the same 31 were wrong. It
converted 18 explicit hard-decoder failures into undetected wrong valid
codewords. Therefore the lower GMD failure count is not a performance gain.

## RS(5,3) payload frame errors

| Duty | Coherent hard | Coherent GMD | Gao GMD | Duong GMD | Gao-Duong GMD |
|---:|---:|---:|---:|---:|---:|
| 100% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 50% | 1/5 | 1/5 | 1/5 | 1/5 | 1/5 |
| 25% | 4/5 | 4/5 | 3/5 | 4/5 | 3/5 |
| 10% | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| 1% | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

Gao and Gao-Duong recovered one additional 25% RS(5,3) frame. This is one
paired frame in a five-frame cell and is not strong evidence of a general gain.

## RS(12,6) payload frame errors

| Duty | Coherent hard | Coherent GMD | Gao GMD | Duong GMD | Gao-Duong GMD |
|---:|---:|---:|---:|---:|---:|
| 100% | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 50% | 1/5 | 1/5 | 1/5 | 1/5 | 1/5 |
| 25% | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| 10% | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| 1% | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

All front ends and GMD produced identical RS(12,6) FER. At 25%, 10%, and 1%,
GMD returned wrong valid codewords for all 15 frames instead of reporting the
hard decoder's 15 failures.

## Front-end diagnostics and limitations

- Median Gao off coherence was 0.00374 and median fitted off-noise reduction was
  only 0.0163 dB. Median sync-only desired attenuation was 0.0112 dB. The weak
  reference coherence gives little mechanism for a large cancellation gain.
- Duong-GMD had the same FER as coherent-GMD. Eleven of 50 four-dimensional
  Duong fits reached the frozen 60000-iteration limit; these are conservatively
  flagged as not converged before the limit. All Gao-residual two-dimensional
  Duong fits stopped before the limit.
- The Duong result is analytical equilibrium instantaneous spatial/I/Q
  whitening, not temporal whitening. Explicit recurrent execution and an
  ordinary ZCA control were not scored, so no neural-specific benefit can be
  claimed.
- Models were fit on raw analytic post-frame off samples rather than an
  untouched pre-final training session. A future predeclared held-out session
  is required to confirm any frontend improvement.
- Physical SNR, clipping, timing, and boundaries are properties of the shared
  capture and were not recomputed or selected per frontend.
