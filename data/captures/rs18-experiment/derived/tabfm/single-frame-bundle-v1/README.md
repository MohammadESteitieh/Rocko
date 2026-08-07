# Single-frame TabFM validation bundle v1

Each signal CSV contains one 53-second RS18 message beginning at its first sample, followed by a 2.5-second guard and 10-second transmitter-off noise interval required by the frozen coherent frontend. `relative` files start at timestamp zero; `absolute` files preserve the original capture clock. The decoder must produce identical features and decisions for both.

The fixed 84-row labelled TabFM reference context comes only from the 100%-duty frames for payload repetitions 1–3. Query CSVs contain only repetitions 4–5, so their payloads are absent from the reference context. Some query frames informed earlier method development, so this bundle is retrospective pipeline validation, not method-level confirmation. `expected-results.csv` is evaluator-only ground truth and must never be read by the decoder. Autonomous synchronization is outside this bundle's contract.
