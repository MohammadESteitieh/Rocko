# Initial TabFM aligned-feature result: sequence 24

This directory preserves the first completed pretrained TabFM inference before
the Sensor-Y hybrid was designed.

- Model: `google/tabfm-1.0.0`, JAX release, one ensemble member.
- Query: RS18 sequence 24, repetition 3, 100% duty.
- Features: 29 aligned two-sensor phasor features; the coherent LLR was excluded.
- Context: 100 rows; repetition 3 excluded from historical context.
- Execution: remote Linux host with 94 GiB RAM. CUDA initialization could not
  locate cuSPARSE, so JAX explicitly warned and completed on CPU.
- Result: the primary coherent frontend had 8 bit / 7 symbol errors and failed;
  TabFM had 10 bit / 8 symbol errors and failed.

The predictions showed that TabFM corrected four coherent bit errors but added
six. A post-hoc confidence gate motivated the separately preregistered
Sensor-Y-hybrid threshold of 0.75. Sequence 24 is therefore exploratory and
must not be counted as confirmatory evidence for that threshold.

Every preserved artifact has a sibling SHA-256 sidecar. The summary also binds
the source capture, manifest, exported prompt, evaluator truth, and prediction
hashes.
