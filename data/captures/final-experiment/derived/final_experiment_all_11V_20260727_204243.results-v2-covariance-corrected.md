# Final experiment decoding results — covariance-corrected v2

Run: `final_experiment_all_11V_20260727_204243`  
Setup: 11 V, 3 m, two coded bits/s, 100 complete physical frames.

This version corrects complex covariance orientation to the required
row-observation form E[z z^H]. The original v1 analysis is invalidated.
Boundaries remain each manifest timestamp plus one frozen -0.070-second
correction estimated only from the first frame's known sync. Autonomous
acquisition was not evaluated.

## Overall

| Metric | Result |
|---|---:|
| Primary decoded-payload frame errors | 59/100 |
| Wilson 95% interval for FER | 49.20%-68.13% |
| Decoder failures | 31/100 |
| Coded-body frame errors | 73/100 |
| Raw coded-body bit errors | 634/2775 (22.85%) |
| Raw systematic-payload bit errors | 343/1575 (21.78%) |
| Conditional decoded-payload bit errors | 125/930 (13.44%) |
| Postdecode BER bounds including failures | 125/1575 to 770/1575 (7.94%-48.89%) |
| Frames with positive in-band signal estimate | 89/100 |
| Frames with no positive signal estimate | 11/100 |
| Active frames with ADC clipping | 0/100 |

## Short comparison

Each cell is decoded-payload frame errors out of five with conditional median
7.25-8.75 Hz pooled SNR in parentheses.

| Duty | Uncoded 11 | Hamming(15,11) | RS(5,3) |
|---:|---:|---:|---:|
| 100% | 1/5 (10.65 dB) | 0/5 (10.54 dB) | 0/5 (11.10 dB) |
| 50% | 2/5 (8.71 dB) | 0/5 (8.39 dB) | 1/5 (7.37 dB) |
| 25% | 4/5 (3.10 dB) | 1/5 (0.84 dB) | 4/5 (1.79 dB) |
| 10% | 5/5 (-5.44 dB; 1 no-estimate) | 5/5 (-4.02 dB) | 5/5 (-2.04 dB) |
| 1% | 5/5 (-7.06 dB) | 5/5 (-5.05 dB; 4 no-estimate) | 5/5 (-7.23 dB; 3 no-estimate) |

## Final RS(12,6)

| Duty | Frame errors | Decoder failures | Median in-band SNR |
|---:|---:|---:|---:|
| 100% | 0/5 | 0/5 | 9.86 dB |
| 50% | 1/5 | 1/5 | 6.64 dB |
| 25% | 5/5 | 5/5 | 2.32 dB |
| 10% | 5/5 | 5/5 | -1.84 dB; 2 no-estimate |
| 1% | 5/5 | 5/5 | -2.66 dB; 1 no-estimate |

Zero observed errors in a five-frame cell is not proof of zero true FER; its
Wilson 95% upper bound is 43.45%. Results use one fixed payload per scheme,
unequal frame lengths/energies, and scheduled manifest-known decoding.
