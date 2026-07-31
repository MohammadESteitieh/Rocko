# Final experiment decoding results

Run: `final_experiment_all_11V_20260727_204243`  
Setup: 11 V, 3 m, two coded bits/s, 100 complete physical frames.

## Analysis contract

- Primary boundaries are each manifest `started_utc` plus one common
  -0.070-second correction estimated from only the first 100% frame's known
  16-bit sync within the frozen +/-1.25-second search radius.
- Later boundaries were not optimized against sync, body, or payload truth.
- The 7.25-8.75 Hz filter is used identically for active/off SNR and first-sync
  correlation. Power-subtracted SNR is absent when active power does not exceed
  off power.
- Two-bit/s half-symbol LLRs use an 8 Hz matched carrier projection on
  DC-centered raw analytic channels. A 1.5 Hz-wide prefilter would remove the
  Manchester envelope sidebands at this rate. Per-frame channel weights use
  only the known sync; covariance uses the central 10 seconds of the declared
  transmitter-off gap and remains frozen during the frame.
- Primary frame error means decoder failure or any wrong decoded payload bit at
  the scheduled boundary. Sync errors are diagnostic and do not independently
  fail a scheduled-boundary frame.
- Autonomous acquisition was not evaluated because the collection used the raw
  capture owner and manifest-known scheme/length.

## Overall

| Metric | Result |
|---|---:|
| Primary decoded-payload frame errors | 59/100 |
| Wilson 95% interval for FER | 49.20%-68.13% |
| Decoder failures | 31/100 |
| Coded-body frame errors | 73/100 |
| Raw coded-body bit errors | 641/2775 (23.10%) |
| Raw systematic-payload bit errors | 358/1575 (22.73%) |
| Conditional decoded-payload bit errors | 131/930 (14.09%) |
| Postdecode BER bounds including failures | 131/1575 to 776/1575 (8.32%-49.27%) |
| Frames with positive in-band signal estimate | 89/100 |
| Frames with no positive signal estimate | 11/100 |
| Active frames with ADC clipping | 0/100 |

The conditional decoded BER excludes 31 decoder-failure frames and is therefore
optimistic; use FER and the stated BER bounds for complete evidence.

## Short comparison

Each cell reports decoded-payload frame errors out of five, followed by the
conditional median pooled in-band SNR in parentheses. The median is over only
frames with a positive power-subtracted signal estimate.

| Duty | Uncoded 11 | Hamming(15,11) | RS(5,3) |
|---:|---:|---:|---:|
| 100% | 1/5 (10.65 dB) | 0/5 (10.54 dB) | 0/5 (11.10 dB) |
| 50% | 2/5 (8.71 dB) | 0/5 (8.39 dB) | 1/5 (7.37 dB) |
| 25% | 3/5 (3.10 dB) | 2/5 (0.84 dB) | 4/5 (1.79 dB) |
| 10% | 5/5 (-5.44 dB; 1 no-estimate) | 5/5 (-4.02 dB) | 5/5 (-2.04 dB) |
| 1% | 5/5 (-7.06 dB) | 5/5 (-5.05 dB; 4 no-estimate) | 5/5 (-7.23 dB; 3 no-estimate) |

With only five frames per cell, zero observed errors is not proof of zero true
FER; the Wilson 95% upper bound for 0/5 is 43.45%.

## Final RS(12,6)

| Duty | Frame errors | Decoder failures | Raw symbol errors | Median in-band SNR |
|---:|---:|---:|---:|---:|
| 100% | 0/5 | 0/5 | 2/60 | 9.86 dB |
| 50% | 1/5 | 1/5 | 10/60 | 6.64 dB |
| 25% | 5/5 | 5/5 | 41/60 | 2.32 dB |
| 10% | 5/5 | 5/5 | 55/60 | -1.84 dB; 2 no-estimate |
| 1% | 5/5 | 5/5 | 60/60 | -2.66 dB; 1 no-estimate |

RS(12,6) corrected all five 100% frames and four of five 50% frames. It did not
recover any 25%, 10%, or commanded-1% frame under this frozen hard-decision
receiver.

## Timing and limitations

- Commanded 1% remained timing-limited: 1751 of 3500 pulse starts were late.
  The median of per-frame median pulse widths was about 759 us versus the 625 us
  target.
- Nominal frame durations were exceeded by approximately 1.2 ms, with complete
  manifests and gaps.
- No analyzed active frame clipped, although isolated saturation samples exist
  elsewhere in the complete capture.
- Every scheme used one fixed payload/codeword. Results apply to these bit
  positions and codewords, not all possible messages.
- Frame lengths and transmitted energies differ across schemes. Comparisons are
  paired by duty and repetition but are not equal-energy code comparisons.
