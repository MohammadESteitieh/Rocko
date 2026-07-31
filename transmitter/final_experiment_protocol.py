#!/usr/bin/env python3
"""Frozen frames for the approved 2-coded-bit/s final experiment.

Wire convention: 16-bit two-tilde sync, then an uncoded, standard even-parity
Hamming(15,11), or systematic data-first Reed--Solomon body. RS symbols are
MSB-first GF(32) values using primitive polynomial 0x25, alpha=2, and generator
roots alpha**1 onward. The short RS payload is padded with four trailing zeros.
"""

from __future__ import annotations

from functools import reduce
from typing import Sequence

SYNC_TEXT = "0111111001111110"
SHORT_PAYLOAD_TEXT = "10001011100"
FINAL_PAYLOAD_TEXT = "100010111000010010100111011111"
BIT_SECONDS = 0.5  # two coded bits per second
DUTIES = (100.0, 50.0, 25.0, 10.0, 1.0)
REPETITIONS = 5

FIELD_BITS = 5
FIELD_SIZE = 32
FIELD_ORDER = 31
PRIMITIVE_POLYNOMIAL = 0x25
PRIMITIVE_ELEMENT = 2

_GF_EXP = [0] * (2 * FIELD_ORDER)
_GF_LOG = [0] * FIELD_SIZE
_value = 1
for _power in range(FIELD_ORDER):
    _GF_EXP[_power] = _value
    _GF_LOG[_value] = _power
    _value <<= 1
    if _value & FIELD_SIZE:
        _value ^= PRIMITIVE_POLYNOMIAL
for _power in range(FIELD_ORDER, 2 * FIELD_ORDER):
    _GF_EXP[_power] = _GF_EXP[_power - FIELD_ORDER]


def require_bits(bits: Sequence[int] | str, length: int, name: str = "bits") -> tuple[int, ...]:
    if isinstance(bits, str):
        if any(bit not in "01" for bit in bits):
            raise ValueError(f"{name} must contain only 0 and 1")
        values = tuple(int(bit) for bit in bits)
    else:
        values = tuple(bits)
        if any(type(bit) is not int or bit not in (0, 1) for bit in values):
            raise ValueError(f"{name} must contain binary integers")
    if len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} bits")
    return values


def bits_text(bits: Sequence[int]) -> str:
    return "".join(str(int(bit)) for bit in bits)


def hamming15_11(payload: Sequence[int] | str) -> tuple[int, ...]:
    """Standard even-parity Hamming coordinates 1..15, parity at 1,2,4,8."""
    data = require_bits(payload, 11, "Hamming payload")
    parity_positions = frozenset((1, 2, 4, 8))
    code = [0] * 16
    for position, bit in zip(
        (p for p in range(1, 16) if p not in parity_positions), data
    ):
        code[position] = bit
    for parity in parity_positions:
        code[parity] = sum(
            code[position] for position in range(1, 16)
            if position & parity and position != parity
        ) & 1
    return tuple(code[1:])


def _gf_mul(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return _GF_EXP[_GF_LOG[left] + _GF_LOG[right]]


def _gf_pow(power: int) -> int:
    return _GF_EXP[power % FIELD_ORDER]


def _poly_mul(left: Sequence[int], right: Sequence[int]) -> list[int]:
    product = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            product[i + j] ^= _gf_mul(a, b)
    return product


def rs_generator(parity_symbols: int) -> tuple[int, ...]:
    if not 1 <= parity_symbols < FIELD_ORDER:
        raise ValueError("parity symbol count must be in 1..30")
    ascending = reduce(
        _poly_mul,
        ([_gf_pow(root), 1] for root in range(1, parity_symbols + 1)),
        [1],
    )
    return tuple(reversed(ascending))


def symbols_to_bits(symbols: Sequence[int]) -> tuple[int, ...]:
    output: list[int] = []
    for symbol in symbols:
        if type(symbol) is not int or not 0 <= symbol < FIELD_SIZE:
            raise ValueError("RS symbols must be integers in 0..31")
        output.extend((symbol >> shift) & 1 for shift in range(4, -1, -1))
    return tuple(output)


def bits_to_symbols(bits: Sequence[int] | str) -> tuple[int, ...]:
    length = len(bits)
    data = require_bits(bits, length, "RS data")
    if length == 0 or length % FIELD_BITS:
        raise ValueError("RS data must contain a positive multiple of five bits")
    return tuple(
        sum(data[start + offset] << (4 - offset) for offset in range(5))
        for start in range(0, length, 5)
    )


def rs_encode_symbols(data_symbols: Sequence[int], parity_symbols: int) -> tuple[int, ...]:
    data = tuple(data_symbols)
    symbols_to_bits(data)  # strict value validation
    if not data:
        raise ValueError("RS data must not be empty")
    generator = rs_generator(parity_symbols)
    work = list(data) + [0] * parity_symbols
    for index in range(len(data)):
        coefficient = work[index]
        if coefficient:
            for offset, generator_value in enumerate(generator):
                work[index + offset] ^= _gf_mul(coefficient, generator_value)
    return data + tuple(work[len(data):])


def build_frames() -> dict[str, str]:
    short = require_bits(SHORT_PAYLOAD_TEXT, 11, "short payload")
    final = require_bits(FINAL_PAYLOAD_TEXT, 30, "final payload")
    short_rs_data = short + (0, 0, 0, 0)
    bodies = {
        "uncoded": bits_text(short),
        "hamming15_11": bits_text(hamming15_11(short)),
        "rs5_3": bits_text(symbols_to_bits(rs_encode_symbols(bits_to_symbols(short_rs_data), 2))),
        "rs12_6": bits_text(symbols_to_bits(rs_encode_symbols(bits_to_symbols(final), 6))),
    }
    return {name: SYNC_TEXT + body for name, body in bodies.items()}


FRAMES = build_frames()
SHORT_SCHEMES = ("uncoded", "hamming15_11", "rs5_3")
FINAL_SCHEMES = ("rs12_6",)
EXPECTED_FRAME_LENGTHS = {
    "uncoded": 27,
    "hamming15_11": 31,
    "rs5_3": 41,
    "rs12_6": 76,
}
assert {name: len(frame) for name, frame in FRAMES.items()} == EXPECTED_FRAME_LENGTHS
assert bits_to_symbols(SHORT_PAYLOAD_TEXT + "0000") == (17, 14, 0)
assert bits_to_symbols(FINAL_PAYLOAD_TEXT) == (17, 14, 2, 10, 14, 31)
