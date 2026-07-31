# Context Handoff

## Checkpoint

- Repository: `~/Desktop/CU-hakcing-2026`, branch `task/receiver-v2`.
- `HEAD` and `origin/task/receiver-v2` remain `c7935d8`; `upstream` is read-only.
- Persistent artifacts are under
  `~/Desktop/CU-hakcing-captures/current-hamming/{raw,manifests,derived}`.
- No transmitter, capture, or analysis process is running. The final physical
  collection completed with `safe_shutdown_outcome=VERIFIED_WRITES`.
- The working tree is intentionally uncommitted. Do not describe the latest
  runner changes or revised protocol plan as integrated or pushed.

## Integrated versus uncommitted work

Integrated through `c7935d8`: the safe calibration runner, frozen 12-frame
100/50/25/10 schedule, transient-SSH handling, persistent transfer/checksums,
offline analysis, and optional live decoder capture.

Uncommitted local changes extend the existing pipeline rather than replacing
it:

- `calibration_sweep.py` accepts an explicit one-frame-per-duty sequence and
  pilot-only one- or half-second coded-bit durations while preserving its frozen
  defaults.
- `run_calibration.py` computes the declared duration, deploys the existing
  transmitter, owns capture/transfer/checksums, and passes custom timing to the
  live and offline receivers.
- `analyze_calibration.py`, `coded_protocol.py`, `slnn_decoder.py`, and the live
  receiver support process-local timing for analysis.
- Tests cover the new schedule and timing paths. The full current suite passed:
  175 tests, one skipped.

The earlier isolated branches remain unmerged:

| Area | Branch / commits | Status |
|---|---|---|
| Frozen 50-bit research docs | `task/protocol-docs` / `c0bc2b8` | Superseded by the provisional plan below; reconcile deliberately. |
| Uncoded + shortened Hamming | `task/codecs-basic` / `9ae947c`, `8b02032` | Different payload/code contract; do not bulk-merge. |
| RS(29,10) | `task/codec-rs` / `4d4861a`, `23f296f` | Useful GF(32) implementation evidence, but different code parameters. |
| Convolutional/Fano | `task/codec-fano` / `11649b0`, `6b01b97` | Threshold-restarted DFS is not a genuine sequential Fano decoder. |
| Gao/Duong front ends | `task/sensor-frontends` / `725936f`, `368fa29` | Still unmerged; includes the Gao overflow fix. |

## Current physical condition matrix

Use 11 V at 3 m. Duty is the control; measure in-band SNR independently for
every future frame. The latest matrix is the corrected-receiver 2-coded-bit/s
sweep `duty_sweep_11V_20260727_200236`:

| Commanded duty | One-frame pooled in-band SNR | Scheduled-boundary decode |
|---:|---:|---|
| 100% | 12.72 dB | Payload `A`; one header-bit error |
| 50% | 8.78 dB | Full `~A` accepted |
| 25% | 3.51 dB | Full `~A` accepted |
| 10% | -6.31 dB | Failed |
| 1% | -3.87 dB | Failed |

These are legitimate negative SNR estimates because active power exceeded off
power, but the inferred signal component was smaller than off-noise power. The
10%/1% inversion must be retained as observed near-floor variation. Skip 5% in
the main matrix; the earlier 0.5-bit/s sweep found that its one-frame estimate
also inverted with 10% near the noise floor.

The 2-bit/s initial broad correlation analysis selected the wrong frame windows.
The regenerated matrix uses each manifest `started_utc` plus one common
-0.990-second clock correction estimated from the first 100% frame within a
fixed 1.25-second search radius. That correction was frozen for later frames;
no later boundary was optimized against the expected payload. Use the
`manifest-boundary` derived CSV, not the initial derived CSV.

Label 1% `commanded 1%, timing-limited`: 56 of 112 pulse starts were late, and
its median software pulse was 753 us versus the 625 us target. The matrix has
only one frame per duty and is a control-selection pilot, not a precise BER
dataset. Future plots and tables must use each frame's measured SNR and retain
acquisition failures, decode failures, and no-positive-power rows.

## Provisional protocol test plan; not implemented

Every candidate uses the 16-bit two-tilde synchronization word
`0111111001111110`, regular Manchester modulation, and two coded bits/s. Primary
scoring uses declared manifest boundaries with autonomous acquisition reported
separately; this convention was explicitly approved after the rate pilots.

### Short comparison

Use the approved exact 11-bit payload `10001011100`, the first 11 bits of the
final payload, with five repetitions at each selected duty control:

| Candidate | Coded body | Total frame |
|---|---:|---:|
| Uncoded | 11 bits | 27 bits |
| Standard Hamming(15,11) | 15 bits | 31 bits |
| Systematic RS(5,3) over GF(32) | 25 bits | 41 bits |

RS(5,3) carries the 11 meaningful bits in three five-bit symbols with four
fixed padding bits and adds two parity symbols. Fano is theoretical context
only: the zero-terminated NASA constraint-length-24 rate-1/2 construction would
produce an 84-bit frame for the same 11-bit payload. Do not claim physical Fano
results.

Scheduled collection time for the short comparison at two coded bits/s is about
39 minutes 43 seconds, excluding setup and retries.

### Final RS test

Use the approved exact 30-bit payload
`100010111000010010100111011111`, represented by the six five-bit symbols
`[17, 14, 2, 10, 14, 31]`. Systematic RS(12,6) adds six parity symbols,
producing a 60-bit coded body and a 76-bit total frame. Run five repetitions at
every selected duty control. Scheduled collection time is about 22 minutes 25
seconds, excluding setup and retries.

The approved field convention uses primitive-polynomial identifier `0x25`,
MSB-first symbols, systematic data followed by parity, and generator roots
starting at the first nonzero field power. RS(5,3) pads the 11-bit payload with
four trailing zero bits. Freeze independently reviewed golden frames before
physical use; the new code parameters and vectors do not yet exist in the
integrated branch.

## Rate pilots

The receiver pin was disconnected during the first one-bit/s attempt, so
`duty_sweep_11V_20260727_194249` is hardware-invalidated rather than evidence
against the rate. `duty_sweep_11V_20260727_193737` failed before transmitter PID
acknowledgement and contains no accepted transmission. Both are preserved, and
both have verified follow-up GPIO-low cleanup.

After the receiver pin was restored, two new 11 V, 3 m, 100% `~A` pilots
completed with persistent artifacts and verified shutdown:

| Run | Rate | Scheduled-boundary result | Pooled in-band SNR | Clipping |
|---|---:|---|---:|---:|
| `duty_sweep_11V_20260727_195255` | 1 coded bit/s | Decoded `~A` | 11.71 dB | None |
| `duty_sweep_11V_20260727_195416` | 2 coded bits/s | Decoded `~A` | 9.45 dB | None |

Both physical rates therefore worked with the known scheduled frame boundary.
Neither passed the stricter autonomous-acquisition gate: the live receiver
reported signal activity but no valid tilde preamble. The one-bit/s live window
started before the actual transmission, consistent with movement contamination;
the two-bit/s window also failed autonomous synchronization. Do not yet declare
either faster rate frozen for the main experiment. Fix or characterize live
acquisition and repeat the chosen rate, or retain the established 0.5 coded
bits/s rate.

The prior 0.5-bit/s pilot `pilot_11V_3m_100p_20260727_190519` decoded `~A`
autonomously at 14.54 dB pooled in-band SNR with no clipping.

## Final physical collection and planned decoding

The accepted final dataset is
`final_experiment_all_11V_20260727_204243` under
`~/Desktop/CU-hakcing-captures/final-experiment`. It contains 100/100 validated
frames, 743701 receiver samples at 200 Hz, complete transmitter logs/manifests,
and passing checksums. The preceding `..._204029` launch transmitted nothing
because a transitive Python import was not deployed; preserve it as a failed,
pre-transmission run. Follow-up GPIO-low writes were verified.

Planned decoding is complete in `receiver/analyze_final_experiment.py`. It uses
manifest timestamps plus one frozen -0.070-second correction estimated only
from the first 100% frame's sync. Autonomous acquisition was not evaluated.
Primary SNR remains 7.25-8.75 Hz. Half-symbol LLRs use an 8 Hz matched carrier
projection on DC-centered raw analytic channels because a 1.5 Hz-wide filter
removes the two-bit/s Manchester envelope sidebands; sync-only channel training
and central-10-second off covariance remain frozen per frame.

During the Gao/Duong review, the original v1 analyzer was found to use the
transpose/conjugate of the required complex covariance. V1 and its dependent
tables/plots are explicitly invalidated. Canonical v2 uses row-observation
covariance `centered.T @ centered.conj()`. The correction leaves overall FER
unchanged but changes raw counts and the 25% uncoded/Hamming cells.

Key results:

| Test | 100% | 50% | 25% | 10% | 1% |
|---|---:|---:|---:|---:|---:|
| Uncoded-11 frame errors | 1/5 | 2/5 | 4/5 | 5/5 | 5/5 |
| Hamming(15,11) frame errors | 0/5 | 0/5 | 1/5 | 5/5 | 5/5 |
| RS(5,3) frame errors | 0/5 | 1/5 | 4/5 | 5/5 | 5/5 |
| Final RS(12,6) frame errors | 0/5 | 1/5 | 5/5 | 5/5 | 5/5 |

Overall primary decoded-payload FER was 59/100, with Wilson 95% interval
49.20%-68.13%; 31 frames were decoder failures. There were 634/2775 raw coded
body bit errors, 343/1575 raw systematic-payload bit errors, 89 positive-SNR
frames, 11 no-positive-power frames, and no clipped analyzed active frames.
For final RS(12,6), median in-band SNR by duty was 9.86, 6.64, 2.32, -1.84,
and -2.66 dB. It decoded 5/5 at 100%, 4/5 at 50%, and none below 50%.

Canonical results contain `v2-covariance-corrected` in their names. The v1
invalidation reason is preserved in `analysis-v1.INVALID.txt`.

Post-hoc soft-decoder/front-end comparison is complete in
`receiver/compare_final_rs_frontends.py` for the 50 RS frames. Coherent hard
and coherent soft GMD both produced 31/50 frame errors. GMD reduced explicit
failures from 31 to 13 only by producing 18 wrong-codeword miscorrections; it
had zero paired payload improvements. Gao-style cancellation plus GMD produced
30/50 errors by gaining one RS(5,3) frame at 25%; with one changed frame and no
untouched confirmation session, this is exploratory only. Duong-GMD remained
31/50; Gao-Duong-GMD matched Gao at 30/50. Median Gao off coherence was 0.00374
and noise reduction only 0.0163 dB. Eleven of 50 four-dimensional Duong fits
reached the 60000-iteration limit. Models used each frame's post-frame central
off gap, so this is explicitly offline/noncausal analysis, not a deployable
causal result. See `rs-frontends-v2-fit-diagnostics*` and
`rs-frontends-v2-results.md`.

The full suite passes 205 tests with one skipped.

## Other recent exploratory data

- `exploratory_6V_3m_5p_1p_20260727_184555`: explicitly invalidated because a
  lab member entered during the active frames; preserve it only as invalid data.
- `exploratory_6V_3m_5p_1p_20260727_185315`: clean exploratory capture. The 5%
  and commanded-1% scheduled windows measured -7.87 and -1.09 dB in-band, but
  neither acquired a valid preamble, the ordering was nonmonotonic, and pulse
  timing exceeded targets. It demonstrates noise-floor operation, not calibrated
  negative-SNR conditions.
- Earlier clean selection datasets remain
  `calibration_11V_20260724_192522` and
  `calibration_10V_20260724_195004`; all 24 `~A` frames decoded with no clipping.

## Safety and analysis invariants

Verified transmitter wiring is GPIO22 to IN3, GPIO17 to IN4, and GPIO27 to ENB.
After success, interruption, failure, or stop, configure GPIO27, GPIO18, GPIO22,
and GPIO17 as outputs and force all four low. Never infer physical transmission
or safe hardware state from a Python return code alone. The exact L298N module,
jumper state, and logic-power arrangement still need a canonical hardware label.

Keep raw captures immutable and archive manifests, logs, derived outputs, and
checksums. Apply the identical 7.25-8.75 Hz filter to active and off windows:

```text
Psignal = Pframe - Pnoise
SNR = 10 log10(Psignal / Pnoise)
```

If active power does not exceed off power, report no positive power estimate.
Never substitute a floor or duty-derived SNR. For carrier SNR against zero-to-10
Hz noise, keep the signal numerator restricted to power-subtracted 7.25-8.75 Hz
carrier power and use all zero-to-10 Hz transmitter-off power as the noise
denominator. Never treat out-of-band active-window excess as signal. The first
`snr-0-10hz` table for `duty_sweep_11V_20260727_200236` violated this rule and is
explicitly invalidated; use `snr-carrier-vs-0-10hz-noise` instead.

Use complete frames or complete sessions as fit, validation, and held-out units;
never split individual samples or bits. Fit means, covariance, Gao parameters,
Duong gains, and H0 thresholds only on declared transmitter-off data and freeze
them during active frames. Reuse identical observations and boundaries across
algorithm comparisons.

## Immediate next steps

1. Preserve and deliberately commit the tested protocol, runner, analyzer,
   tests, handoff, and final-result artifacts; the working tree remains dirty.
2. Use the final per-frame CSV for figures and statistical interpretation; do
   not claim zero true FER from any 0/5 cell or omit decoder failures.
3. Keep autonomous acquisition explicitly not evaluated and do not re-optimize
   boundaries or decoding against payload truth.
4. Document the exact L298N module and logic-power arrangement.
