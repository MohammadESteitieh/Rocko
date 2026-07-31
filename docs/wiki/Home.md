# Rocko Magnetic-Link Research Wiki

Rocko is a low-frequency magnetic communication research platform. A QNX
Raspberry Pi drives an L298N/coil transmitter; a Pico streams two magnetic
sensor channels to a macOS receiver at approximately 200 Hz.

This wiki documents the research pipeline only. The original hackathon audio,
emergency-intent, photo/CNN, and demo layers are not part of the research
branch.

## Current research protocols

- Carrier: 8 Hz polarity-switched magnetic field.
- Modulation: OOK Manchester.
- Primary experimental rate: two coded bits/s.
- Synchronization: `0111111001111110`.
- Codes: uncoded, Hamming(15,11), RS(5,3), RS(12,6), and RS(18,6) over GF(32).
- Physical benchmark condition: 11 V, 3 m, measured SNR per frame.

## Pages

- [Hardware](Hardware.md)
- [Protocols](Protocol.md)
- [Transmitter](Transmitter.md)
- [Receiver and decoding](Receiver-and-Decoding.md)
- [Experiments and data](Experiments-and-Data.md)
- [Historical 2026-07-16 benchmark](Results-2026-07-16.md)
- [Temporal-whitening benchmark](Temporal-Whitening-Benchmark.md)
- [Operations and troubleshooting](Operations-and-Troubleshooting.md)
- [Current context handoff](Context-Handoff.md)

## Canonical locations

- Repository: `~/Desktop/CU-hakcing-2026`
- GitHub origin: `https://github.com/MohammadESteitieh/Rocko.git`
- Repository capture backup: `data/captures/`
- Original capture archive: `~/Desktop/CU-hakcing-captures/`
- Wiki source: `docs/wiki/`

The Markdown files in `docs/wiki/` are the canonical wiki source and are backed
up with the repository. They are not a separate local database.
