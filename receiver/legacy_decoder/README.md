# Legacy decoder and analysis pipeline

This directory preserves the complete receiver decoding, experimental frontend,
and analysis implementation used for the recorded physical results.

It was moved here intact as an architectural checkpoint, **not because it was
proven defective**. The original receiver grew into many overlapping scripts,
which made the assumptions, signal path, and failure mechanism difficult to
explain and audit. A new receiver will be built separately with one explicit,
traceable pipeline.

## Contents

- `coded_protocol.py` — historical Hamming alphabet constants and codebook.
- `layered_decoder.py`, `hybrid_decoder.py`, `slnn_decoder.py` — Hamming
  decoders.
- `decode_*.py` — historical decoder command-line tools.
- `analyze_calibration.py` — calibration analysis.
- `analyze_final_experiment.py` — uncoded/Hamming/RS(5,3)/RS(12,6) analysis.
- `analyze_rs18_experiment.py` — accepted RS(18,6) analysis.
- `compare_final_rs_frontends.py` — hard/GMD/Gao/Duong comparison.
- `duong_whitener.py`, `temporal_whitening.py`, and
  `benchmark_temporal_whitening.py` — experimental frontends.
- `live_receiver.py`, `rocko_receiver.py`, `monitor_dataset.py`, and
  `plot_receiver.py` — legacy live/offline integration and visualization.

## Status

The code remains executable for reproduction and regression comparison. New
receiver work must not silently modify these modules or overwrite their derived
artifacts. If a result changes, record the new implementation, input hashes,
boundaries, and output under a distinct name.

The central question for the rewrite is not merely whether a decoder returns a
payload. Every stage must expose enough evidence to explain:

1. whether the carrier is present;
2. whether the frame boundary is supported by sync alone;
3. what each sensor contributes;
4. how bit and symbol likelihoods are formed;
5. whether the observed error pattern is within the code's correction region;
6. why a decoder failed or selected a particular codeword.
