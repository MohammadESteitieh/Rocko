# Current Hamming experiment data

- `raw/`: immutable receiver `t,x,y` CSV captures.
- `manifests/`: transmitter manifests and run notes.
- `derived/`: segmentation, features, plots, and reports; raw files are never overwritten.
- `models/`: models fitted only from designated training frames.

Current frame protocol: encoded `~` plus A-Z, Hamming(7,4), 28 coded bits,
0.5 coded bit/s, 8 Hz OOK Manchester, and 15-second transmitter-off gaps.
Train/test partitioning is by complete physical frame.
