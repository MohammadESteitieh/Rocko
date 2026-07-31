# Research capture archive

This directory is a repository backup of the physical data still relevant to
the magnetic-link research as of 2026-07-28.

| Directory | Contents |
|---|---|
| `current-hamming/` | Calibration, rate, orientation, background, and historical Hamming captures. |
| `final-experiment/` | Accepted 100-frame uncoded/Hamming/RS comparison and derived frontend results. |
| `rs18-experiment/` | Accepted 45-frame RS(18,6) experiment plus interrupted provenance. |
| `rs18-quick-screen/` | Five-frame corrected-wiring exploratory screen. |

`SHA256SUMS` covers every archived file except itself. Individual experiment
folders also retain their original checksums, manifests, metadata, logs,
invalidations, CSV tables, JSON summaries, and plots.

The original working copies remain under
`~/Desktop/CU-hakcing-captures/`. Treat both copies as immutable evidence:
never edit a raw CSV or transmitter manifest in place. Regenerated work belongs
under the appropriate `derived/` directory and must receive a new filename and
checksum.

The archive intentionally excludes unrelated reference material,
`legacy-naive-bayes`, and temporary source caches. Historical or invalidated
runs inside the retained research folders remain present because they document
hardware and analysis provenance.
