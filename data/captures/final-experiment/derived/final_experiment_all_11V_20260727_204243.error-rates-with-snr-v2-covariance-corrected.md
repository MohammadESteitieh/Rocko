# Final error rates with carrier SNR bandwidths

The full-noise SNR keeps the signal numerator restricted to the
power-subtracted 7.25-8.75 Hz carrier and uses all transmitter-off
noise from zero to 10 Hz in the denominator. It therefore cannot exceed
the in-band SNR. Medians exclude frames with no positive carrier-power
estimate and report their count explicitly.

| Scheme | Duty | Payload FER | Raw body BER | Carrier SNR vs 0-10 Hz noise | In-band SNR (7.25-8.75 Hz) |
|---|---:|---:|---:|---:|---:|
| Uncoded-11 | 100% | 1/5 (20.0%) | 16.4% | -12.99 dB | 10.65 dB |
| Uncoded-11 | 50% | 2/5 (40.0%) | 3.6% | -14.52 dB | 8.71 dB |
| Uncoded-11 | 25% | 4/5 (80.0%) | 14.5% | -22.03 dB | 3.10 dB |
| Uncoded-11 | 10% | 5/5 (100.0%) | 45.5% | -31.73 dB; 1 no-estimate | -5.44 dB; 1 no-estimate |
| Uncoded-11 | 1% | 5/5 (100.0%) | 54.5% | -33.68 dB | -7.06 dB |
| Hamming(15,11) | 100% | 0/5 (0.0%) | 1.3% | -16.54 dB | 10.54 dB |
| Hamming(15,11) | 50% | 0/5 (0.0%) | 2.7% | -15.58 dB | 8.39 dB |
| Hamming(15,11) | 25% | 1/5 (20.0%) | 10.7% | -25.02 dB | 0.84 dB |
| Hamming(15,11) | 10% | 5/5 (100.0%) | 36.0% | -25.74 dB | -4.02 dB |
| Hamming(15,11) | 1% | 5/5 (100.0%) | 45.3% | -25.64 dB; 4 no-estimate | -5.05 dB; 4 no-estimate |
| RS(5,3) | 100% | 0/5 (0.0%) | 0.8% | -15.47 dB | 11.10 dB |
| RS(5,3) | 50% | 1/5 (20.0%) | 3.2% | -14.45 dB | 7.37 dB |
| RS(5,3) | 25% | 4/5 (80.0%) | 14.4% | -18.76 dB | 1.79 dB |
| RS(5,3) | 10% | 5/5 (100.0%) | 35.2% | -25.40 dB | -2.04 dB |
| RS(5,3) | 1% | 5/5 (100.0%) | 56.0% | -31.57 dB; 3 no-estimate | -7.23 dB; 3 no-estimate |
| RS(12,6) | 100% | 0/5 (0.0%) | 0.7% | -11.93 dB | 9.86 dB |
| RS(12,6) | 50% | 1/5 (20.0%) | 4.0% | -14.88 dB | 6.64 dB |
| RS(12,6) | 25% | 5/5 (100.0%) | 23.0% | -19.69 dB | 2.32 dB |
| RS(12,6) | 10% | 5/5 (100.0%) | 39.3% | -24.61 dB; 2 no-estimate | -1.84 dB; 2 no-estimate |
| RS(12,6) | 1% | 5/5 (100.0%) | 50.0% | -25.94 dB; 1 no-estimate | -2.66 dB; 1 no-estimate |
