# TabFM RS18 bit-frontend pilot

This experiment tests Google TabFM as a probabilistic **bit frontend** for the
accepted RS(18,6) capture. TabFM does not replace Reed–Solomon decoding.

## Scope

Each table row represents one received coded bit. The context contains:

- all 16 known sync bits from the query frame;
- a deterministic balanced sample of labeled body bits from historical frames;
- no historical rows with the query frame's payload repetition.

The query contains the 90 unknown body-bit rows. Query labels are blank in the
mixed table and stored separately in an evaluator-only truth file.

The features contain normalized two-sensor Manchester-half phasors, their
contrasts, and sync-derived channel alignment. Aggregate error counts, decoded
payloads, correction outcomes, and other truth-derived analysis fields are not
features. The legacy coherent LLR is retained only as a reported baseline and
is excluded from `FEATURE_COLUMNS`.

This first pilot reuses the accepted experiment's manifest-clock frame-boundary
contract. It evaluates only frontend discrimination; it is not an autonomous
receiver test.

## Export a prompt

```bash
receiver/.venv/bin/python experiments/tabfm_decoder/export_rs18_table.py \
  --query-sequence 24 \
  --output-dir /tmp/rocko-tabfm
```

This writes:

- `*.context-query.csv` — mixed known and unknown rows for TabFM;
- `*.truth.csv` — evaluator-only labels;
- `*.metadata.json` — scope, feature list, split description, source paths,
  recorded source hashes, boundary diagnostics, and output hashes.

## TabFM environment

The pretrained weights have a non-commercial, non-production license. Use a
separate environment because TabFM's JAX/PyTorch stack is substantially larger
than the receiver runtime:

```bash
python3 -m venv .venv
.venv/bin/pip install -r experiments/tabfm_decoder/requirements-tabfm.txt
```

A practical machine needs more than 8 GB of memory. The JAX classification
checkpoint alone occupies approximately 5.7 GB on disk; model restoration and
inference require additional memory.

## Remote high-memory quick start

On the Linux compute server, the repository-provided bootstrap installs an isolated
Python environment and runs the frozen Sensor-Y hybrid batch. It includes the
exploratory sequence 24 control and confirmatory sequences 2, 10, 28, and 45,
spanning 50%, 45%, 25%, and 10% duty and payload repetitions 1, 2, 4, and 5.
Their legacy outcomes are known, but their TabFM outputs have not been inspected.
The confidence gate is frozen at 0.75 and the TabFM checkpoint at revision
`d5e74033fcf257699fab013e2cdd7edf424ff904` before those new results are run.
Each 16 GiB RTX 4080 failed while restoring the model, and TabFM does not
parameter-shard it across GPUs, so the bootstrap uses the host's 94 GiB CPU RAM:

```bash
curl -fsSL https://raw.githubusercontent.com/MohammadESteitieh/Rocko/research-main/experiments/tabfm_decoder/run_remote.sh \
  -o "$HOME/run-rocko-tabfm.sh"
nohup bash "$HOME/run-rocko-tabfm.sh" > "$HOME/rocko-tabfm.log" 2>&1 &
```

## Run the pretrained model manually

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
  experiments/tabfm_decoder/run_tabfm.py \
  /tmp/rocko-tabfm/rs18-sequence-24.context-query.csv \
  /tmp/rocko-tabfm/rs18-sequence-24.truth.csv \
  /tmp/rocko-tabfm/rs18-sequence-24.metadata.json \
  --backend jax \
  --output /tmp/rocko-tabfm/rs18-sequence-24.predictions.csv \
  --summary /tmp/rocko-tabfm/rs18-sequence-24.summary.json
```

The pilot deliberately uses one ensemble member. The official default of 32
would multiply an already expensive first evaluation and is inappropriate for
the initial feasibility screen.

The `sensor-y-hybrid` feature set gives TabFM the Sensor-Y coherent LLR and its
absolute confidence in addition to the aligned phasor features. The gated result
keeps the Sensor-Y decision unless TabFM assigns at least 0.75 probability to
one class. Sequence 24 motivated that fixed threshold and is exploratory; the
four confirmatory frames test it without per-frame adjustment.

## Frozen batch result

The completed batch is preserved under
`data/captures/rs18-experiment/derived/tabfm/sensor-y-hybrid-frozen-v1/`.
Sequence 24 decoded with every Sensor-Y-based variant, as it already did before
TabFM. None of the four confirmatory frames decoded. Across those frames, Sensor
Y had 97 bit / 55 symbol errors, TabFM alone had 103 / 56, and the fixed gate
had 98 / 54. The gate changed three decisions—one repair and two new errors—so
it did not add a decodable payload.

## Soft-list follow-up

A post-hoc development pass on sequences 24, 2, 10, 28, and 45 enumerated all
erasure subsets among the 12 least-reliable TabFM symbols. A candidate is
accepted only when its soft-score margin over the runner-up is at least 20.
This recovered sequence 2, which every hard-decision frontend had failed, while
rejecting the wrong candidates on sequences 10, 28, and 45. Because the pool
size and margin were selected after those outputs were inspected, that recovery
is exploratory.

The frozen confirmation uses previously uninspected TabFM outputs for sequences
3, 18, 32, and 39: payload repetitions 1, 2, 4, and 5 at 45%, 50%, 50%, and 50%
duty. The model revision, 100-row context, feature set, 12-symbol list pool, and
margin 20 are fixed before execution. A score margin is not an error-detection
code; any accepted wrong payload remains a critical miscorrection.

## Validation policy

A useful follow-up must evaluate complete held-out frames and payload
repetitions. Never randomly split bit rows from the same encoded frame across
training and evaluation. Compare bit errors, symbol errors, decoder failures,
and payload errors against the frozen coherent frontend on exactly the same
boundaries. The runner rejects query/truth pairs whose frame identity or full
body-bit index set does not match.
