# Research Protocols

The repository retains several explicit protocol families rather than one
mutable “current frame.” Each capture manifest names its scheme, timing, coded
body, and complete frame.

## Calibration Hamming protocol

- Payload: MSB-first tilde header plus one uppercase letter.
- Code: four standard even-parity Hamming(7,4) groups.
- Body: 28 coded bits.
- Default coded-bit duration: 2 seconds.
- Gap: 15 seconds transmitter-off.

Pilot-only timing can be configured process-locally for one or two coded bits/s.
The frozen default remains unchanged.

## Final comparison protocol

Shared synchronization word: `0111111001111110`.

| Scheme | Payload | Coded body | Total frame |
|---|---:|---:|---:|
| Uncoded | 11 bits | 11 bits | 27 bits |
| Hamming(15,11) | 11 bits | 15 bits | 31 bits |
| RS(5,3) | 11 meaningful bits plus padding | 25 bits | 41 bits |
| RS(12,6) | 30 bits | 60 bits | 76 bits |

The experiment uses two coded bits/s, 15-second gaps, and duties 100%, 50%,
25%, 10%, and commanded 1%, with five repetitions per cell.

## RS(18,6) follow-up

- Field: GF(32).
- Data: six symbols / 30 bits.
- Parity: twelve symbols / 60 bits.
- Body: 18 symbols / 90 bits.
- Frame: 16 sync bits plus 90 body bits.
- Hard unique-decoding capability: six arbitrary symbol errors.
- Duties: 100%, 50%, 45%, 40%, 35%, 30%, 25%, 10%, and commanded 1%.
- Five preregistered payloads, fixed by repetition across duties.

## GF(32) convention

- Primitive-polynomial identifier: `0x25`.
- Primitive element: 2.
- Five-bit symbols are MSB-first.
- Systematic data symbols precede parity symbols.
- Generator roots start at the first nonzero field power.

Exact payloads and golden frames live in:

- `transmitter/final_experiment_protocol.py`
- `transmitter/rs18_experiment_protocol.py`

## Manchester modulation

A coded one is tone then silence; a coded zero is silence then tone. Physical
experiments use an 8 Hz polarity-switched carrier. Commanded 1% is explicitly
labelled timing-limited because software pulse deadlines can exceed the target.
