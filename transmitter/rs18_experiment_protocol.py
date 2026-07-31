#!/usr/bin/env python3
"""Frozen RS(18,6) protocol for the separate follow-up experiment.

Payloads were prospectively fixed before collection as the first 30 MSB-first
bits of SHA-256("CU-hacking-RS18-repetition-{n}"), n=1..5. The hash is not
computed at runtime: exact payloads and frames are stored below.
"""

from __future__ import annotations

import final_experiment_protocol as gf

SYNC_TEXT = "0111111001111110"
BIT_SECONDS = 0.5
DATA_SYMBOLS = 6
PARITY_SYMBOLS = 12
CODE_SYMBOLS = 18
CODE_BITS = 90
FRAME_BITS = 106
DUTIES = (100.0, 50.0, 45.0, 40.0, 35.0, 30.0, 25.0, 10.0, 1.0)
REPETITIONS = 5
PAYLOAD_DERIVATION = (
    "first 30 MSB-first bits of SHA-256 ASCII label "
    "CU-hacking-RS18-repetition-{1..5}"
)
PAYLOADS = {
    1: "001110011010001100001001111101",
    2: "001111110011110100101001011110",
    3: "100001000001010001101010001010",
    4: "111001000111101100010011100010",
    5: "100110101110100100011000100010",
}


def encode_payload(payload: str) -> tuple[int, ...]:
    bits = gf.require_bits(payload, 30, "RS(18,6) payload")
    data = gf.bits_to_symbols(bits)
    if len(data) != DATA_SYMBOLS:
        raise ValueError("RS(18,6) payload must map to six symbols")
    return gf.rs_encode_symbols(data, PARITY_SYMBOLS)


def build_frame(payload: str) -> str:
    body = gf.bits_text(gf.symbols_to_bits(encode_payload(payload)))
    frame = SYNC_TEXT + body
    if len(frame) != FRAME_BITS:
        raise AssertionError("RS(18,6) frame length contract violated")
    return frame


CODEWORDS = {repetition: encode_payload(payload)
             for repetition, payload in PAYLOADS.items()}
FRAMES = {repetition: build_frame(payload)
          for repetition, payload in PAYLOADS.items()}

assert gf.PRIMITIVE_POLYNOMIAL == 0x25
assert gf.PRIMITIVE_ELEMENT == 2
assert len(set(PAYLOADS.values())) == REPETITIONS
assert all(len(payload) == 30 for payload in PAYLOADS.values())
assert all(len(codeword) == CODE_SYMBOLS for codeword in CODEWORDS.values())
assert all(len(frame) == FRAME_BITS for frame in FRAMES.values())
