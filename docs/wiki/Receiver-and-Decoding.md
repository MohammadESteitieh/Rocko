# Receiver and Decoding

## Acquisition

The Pico streams `t,x,y` rows at approximately 200 Hz. `receiver/capture.py` is
the authoritative serial owner. Live viewers must be read-only and must not
compete for the serial device.

Analysis validates required columns, estimates sample rate, reports clipping,
and aligns frame windows to transmitter manifest timestamps. Primary final
analyses use one synchronization-derived clock correction from the first 100%
frame and freeze it for every later boundary. Autonomous acquisition is
reported separately and was not evaluated for the accepted final datasets.

## Signal extraction

The receiver uses 8 Hz complex matched projections over Manchester
half-symbols. The 7.25–8.75 Hz band is used consistently for physical carrier
power and noise measurements, while raw coherent projections are retained for
two-coded-bit/s decisions because an overly narrow preprocessing filter can
remove Manchester sidebands.

Channel response is learned from the known sync only. Adaptive noise or
frontend parameters must be learned from declared transmitter-off samples and
frozen during the active frame.

## Decoders

- Hard uncoded payload decisions.
- Hamming(15,11) syndrome correction.
- Generic systematic Reed–Solomon correction over GF(32).
- Truth-free exploratory GMD using least-reliable-symbol erasure prefixes.
- Historical Hamming(7,4) layered Gaussian and fixed-codebook decoders.

A decoder exception or invalid application padding is a frame error. Conditional
BER is always accompanied by failure counts and failure-inclusive bounds.
Wrong valid codewords are miscorrections, not successful decodes.

## Frontends

- Sync-trained coherent two-sensor combining.
- Identity/no-whitening control.
- Frozen off-RMS diagonal normalization.
- Complex covariance combining.
- Gao-style sensor-y-primary reference cancellation.
- Duong gain-modulated instantaneous spatial/IQ whitening.
- Gao followed by Duong.
- Historical VAR, Kalman, GRU, and TCN prediction-error filters.

Current Duong whitening is instantaneous; it is not temporal recurrence.
Earlier temporal benchmarks found that VAR reduced held-out autocorrelation but
also cancelled beacon evidence. GRU and TCN retained accuracy without proving
held-out whitening.

## Key interpretation

The accepted RS18 data showed no meaningful gain from whitening. Sensor-y-only,
no whitening, and frozen off-RMS each produced 40/45 frame errors; per-frame
post-gap covariance produced 41/45. Gao and Duong had negligible effect because
the sensors shared little useful transmitter-off noise.
