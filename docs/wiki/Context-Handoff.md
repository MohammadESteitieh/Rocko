# Research Context Handoff

## Repository

- Local worktree: `~/Desktop/CU-hakcing-2026`
- GitHub origin: `https://github.com/MohammadESteitieh/Rocko.git`
- Research branch: `research-main`
- Wiki source: `docs/wiki/`
- Repository capture backup: `data/captures/`
- Original capture archive: `~/Desktop/CU-hakcing-captures/`

The repository has been reduced to the magnetic transmitter/receiver research,
algorithms, tests, wiki, and relevant datasets. Hackathon audio, emergency
intent, photo/CNN, team automation, and demo code were removed from the
research branch while normal Git history was retained.

## Safety invariants

- Wiring: GPIO22→IN3, GPIO17→IN4, GPIO27→ENB; GPIO18 is defensively low.
- Force GPIO27, GPIO18, GPIO22, and GPIO17 low after every physical attempt.
- Require independent verified writes; a return code is not proof of safety.
- Never store or relay a QNX password.
- Stop on network, DNS, SSH, relay, or remote-host failure; do not auto-retry.
- Preserve raw captures, manifests, logs, metadata, invalidations, and hashes.

## Frozen analysis conventions

- Primary final scoring uses manifest timestamps plus one correction derived
  from the first 100% sync and frozen for all later frames.
- Autonomous acquisition was not evaluated for either accepted final session.
- Physical SNR uses power-subtracted 7.25–8.75 Hz carrier power.
- Report no positive estimate when active power does not exceed off power.
- Fit frontend parameters only from declared transmitter-off data and freeze
  them during active frames.
- Treat decoder failures and wrong valid codewords as frame errors.
- Split by complete frames or sessions rather than bits.
- Gao, Duong, and GMD reanalysis is exploratory because outcomes were already
  inspected and there is no untouched confirmation session.

## Accepted final comparison

Dataset: `final_experiment_all_11V_20260727_204243`

- 100 validated frames at 11 V, 3 m, and two coded bits/s.
- Schemes: uncoded-11, Hamming(15,11), RS(5,3), and RS(12,6).
- Duties: 100%, 50%, 25%, 10%, and commanded 1%.
- Overall primary payload errors: 59/100.
- Decoder failures: 31.
- Raw coded-body errors: 634/2775.
- No clipping in analyzed active frames.
- Hamming cell errors: 0/5, 0/5, 1/5, 5/5, 5/5.
- RS(12,6) cell errors: 0/5, 1/5, 5/5, 5/5, 5/5.

Canonical analysis names contain `v2-covariance-corrected`. The v1 covariance
orientation is explicitly invalidated.

Frontend exploration on the 50 RS frames found:

- Coherent hard: 31/50 errors, 31 failures.
- Coherent GMD: 31/50 errors, with 18 miscorrections.
- Gao GMD: 30/50 errors, a one-frame post-hoc change.
- Duong GMD: 31/50 errors.
- Gao→Duong GMD: 30/50 errors.

GMD mainly converted detected failures into wrong codewords. Median Gao
transmitter-off coherence and noise reduction were negligible. Several Duong
fits reached the iteration limit.

## Accepted RS(18,6) follow-up

Dataset: `rs18_experiment_11V_20260728_091828`

- 45 validated frames across nine duties and five repetitions.
- 120.005 seconds of captured prelaunch transmitter-off baseline.
- Frozen raw-coherent first-sync correction: approximately −0.075 seconds.
- Primary hard result: 41/45 frame errors.
- 100%: 1/5 primary errors; every duty at or below 50%: 5/5.
- Exploratory GMD: 40/45 errors, with all lower-duty outputs wrong codewords.
- Frozen off-RMS and sensor-y-only: 40/45 errors, including 0/5 at 100%.
- Sensor-x-only: 45/45 errors.
- Per-frame post-gap covariance: 41/45 errors.
- No tangible whitening gain below 100%.

The RS18 session had substantially lower SNR than the earlier RS(12,6)
session. Do not present it as a fair code-only comparison.

## Corrected-wiring diagnostics and quick screen

The cleanest final background capture was
`corrected_wiring_final_placement_background_20260728_154237`, with no clipping
and low carrier-band noise.

The subsequent active dataset
`rs18_quick_screen_11V_20260728_120220` sent one RS18 frame at 100%, 50%, 40%,
30%, and 10%. It completed safely but did not contain a reliable sync or enough
desired signal. Every tested frontend failed all five frames. Even a prohibited
payload-truth timing oracle left 14–18 erroneous RS symbols per frame, far above
the six-symbol hard-correction capability. The quick screen is a diagnostic,
not evidence about decoder improvements.

## Receiver rewrite status

The historical protocol decoders, experimental frontends, live integration,
and analyzers have been moved intact to `receiver/legacy_decoder/`. This is an
organizational checkpoint, not a finding that the code is defective. The goal
is to build a replacement receiver whose synchronization, channel estimate,
bit evidence, symbol errors, correction attempt, and failure reason can each be
inspected and explained independently.

## TabFM frontend result

A pinned TabFM 1.0.0 experiment tested aligned phasors plus Sensor-Y coherent
soft evidence. Sequence 24 was exploratory and already decodable without
TabFM. Confirmatory sequences 2, 10, 28, and 45 covered distinct payload
repetitions at 50%, 45%, 25%, and 10% duty. Neither TabFM alone nor a frozen
0.75-confidence Sensor-Y/TabFM gate decoded any new frame. Across the four
confirmatory frames, Sensor Y had 97 bit / 55 symbol errors, TabFM had 103 / 56,
and the gate had 98 / 54. Preserve this as a negative result for that exact
representation and gate, not a universal conclusion about tabular models.
Artifacts and full provenance are under
`data/captures/rs18-experiment/derived/tabfm/sensor-y-hybrid-frozen-v1/`.

## Current offline research opportunities

- Sync-trained time-domain and harmonic waveform extraction.
- Frequency-selective, transmitter-off-fitted reference cancellation.
- Constrained sensor-y-primary combining with shrinkage.
- Lag-aware temporal processing with untouched-session validation.
- True soft symbol-likelihood or Chase/list RS decoding.
- Strict capture continuity validation and reusable experiment schemas.

These methods cannot create information absent from the quick-screen capture.
All additional work on inspected data must remain labelled exploratory.

## Maintenance priorities

1. Keep the research branch, wiki, code, and capture archive committed and
   pushed to the MohammadESteitieh origin.
2. Implement the new explainable receiver outside `legacy_decoder/` and retain
   stage-by-stage regression comparisons.
3. Consolidate duplicated calibration/final/RS18 runner machinery.
4. Replace mutable protocol globals and `sys.path` imports with a package.
5. Add strict timestamp/gap validation to capture loading.
6. Document the exact L298N revision and logic-power arrangement.
