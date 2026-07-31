# Experiments and Data

## Repository archive

Selected research data is committed under:

```text
data/captures/
├── current-hamming/
├── final-experiment/
├── rs18-experiment/
├── rs18-quick-screen/
└── SHA256SUMS
```

The original working archive remains at `~/Desktop/CU-hakcing-captures/`.
Raw captures and transmitter manifests are immutable. Regenerated analysis must
use a new derived filename and checksum.

## Accepted principal datasets

| Dataset | Purpose | Frames |
|---|---|---:|
| `final_experiment_all_11V_20260727_204243` | Uncoded/Hamming/RS comparison | 100 |
| `rs18_experiment_11V_20260728_091828` | RS(18,6) follow-up | 45 |
| `rs18_quick_screen_11V_20260728_120220` | Corrected-wiring exploratory screen | 5 |

The quick screen completed safely but contained insufficient desired signal;
it is diagnostic rather than evidence of improved decoding.

## Dataset rules

- Preserve failed, interrupted, and invalidated runs as provenance.
- Split fitting and validation by complete frames or sessions, not bits.
- Fit frontends only on declared transmitter-off observations.
- Freeze a fitted frontend during active frames.
- Do not optimize boundaries or candidate codewords using payload truth.
- Report autonomous acquisition separately from manifest-boundary decoding.
- Retain decoder failures and wrong-codeword miscorrections as frame errors.
- Do not claim a true zero error rate from a five-frame zero-error cell.

## SNR convention

Apply the identical 7.25–8.75 Hz filter to active and transmitter-off windows.
Subtract off power from active power before treating any component as signal.
If active power does not exceed off power, report no positive signal estimate.
Never substitute a floor or infer SNR from commanded duty.

Carrier-versus-0–10 Hz SNR retains the carrier-only, power-subtracted numerator
and uses the zero-to-10 Hz transmitter-off denominator. Out-of-band active
excess is not signal.

## Accepted results summary

The final comparison produced 59/100 payload frame errors overall. Hamming
(15,11) decoded all five frames at 100% and 50%, while RS(12,6) decoded all five
at 100% and four of five at 50%.

The RS18 session produced 41/45 primary hard frame errors: one of five at 100%
and five of five at every lower duty. Frozen off-RMS and sensor-y-only improved
the 100% cell to zero of five but did not recover lower duties. The RS18 session
had materially lower physical SNR than the earlier RS(12,6) session, so it is
not a fair code-only comparison.
