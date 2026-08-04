# TabFM smoke-test record

Run date: 2026-08-04 UTC

## Environment

- Apple A18 Pro
- 8 GB unified memory
- Python 3.14.6
- TabFM 1.0.1
- JAX 0.11.0
- JAX CPU backend

## Results

1. Installing the official TabFM JAX package in an isolated temporary virtual
   environment succeeded.
2. The official classification checkpoint download completed: 134 files and
   approximately 5.7 GB in the Hugging Face cache.
3. The official classification example did not finish checkpoint restoration
   and inference before a 60-minute command limit. About 22 minutes were spent
   downloading; the remaining time was spent restoring or preparing the model.
   The process was terminated by the command timeout and produced no
   probabilities. No pretrained-model accuracy result is claimed.
4. The RS18 exporter completed for held-out sequence 24. It produced 100
   labeled context rows and 90 unlabeled query rows. The evaluator-only truth
   reproduces the frozen baseline for that frame: eight body-bit errors, seven
   symbol errors, and decoder failure.
5. A miniature randomly initialized TabFM with the same official classifier API
   successfully consumed the exported table and returned a finite 90-by-2
   probability array whose rows summed to one. This validates table shape,
   categorical/numeric preprocessing, context/query separation, and API
   plumbing only; it is not a model-quality result.

## Conclusion

The data representation and software integration pass their smoke tests. The
released pretrained checkpoint is not practical on this 8 GB machine. A real
comparison requires a higher-memory host; start with one ensemble member and a
single held-out frame before expanding to frame-level cross-validation.
