"""Tests for the two-byte Hamming(7,4) alphabet protocol."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path[:0] = [
    str(ROOT / "receiver" / "legacy_decoder"),
    str(ROOT / "receiver"),
    str(ROOT / "transmitter"),
]

import coded_protocol as protocol  # noqa: E402
import layered_decoder  # noqa: E402
import alphabet_transmitter  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_known_group_encoding(self):
        self.assertEqual(alphabet_transmitter.encode_group("0111"), "0001111")
        self.assertEqual(alphabet_transmitter.encode_group("1110"), "0010110")

    def test_hamming_code_has_minimum_distance_three(self):
        words = protocol.GROUP_CODEBOOK
        distances = [
            np.count_nonzero(words[left] != words[right])
            for left in range(16) for right in range(left + 1, 16)
        ]
        self.assertEqual(min(distances), 3)

    def test_half_baud_timing_contract(self):
        self.assertEqual(protocol.BIT_SECONDS, 2.0)
        self.assertEqual(protocol.HALF_SYMBOL_SECONDS, 1.0)

    def test_process_local_pilot_timing(self):
        try:
            protocol.configure_bit_seconds(1.0)
            self.assertEqual(protocol.BIT_SECONDS, 1.0)
            self.assertEqual(protocol.HALF_SYMBOL_SECONDS, 0.5)
            protocol.configure_bit_seconds(0.5)
            self.assertEqual(protocol.BIT_SECONDS, 0.5)
            self.assertEqual(protocol.HALF_SYMBOL_SECONDS, 0.25)
        finally:
            protocol.configure_bit_seconds(2.0)

    def test_tilde_and_letter_are_msb_first_and_28_bits(self):
        coded = alphabet_transmitter.build_message("A")
        self.assertEqual(len(coded), 28)
        self.assertEqual(coded, "".join(map(str, protocol.encode_message("A"))))
        self.assertTrue(protocol.parity_valid([int(bit) for bit in coded]))

    def test_alphabet_input_is_strict(self):
        for invalid in ("a", "AA", "1", ""):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                protocol.encode_message(invalid)


class LayeredDecoderTests(unittest.TestCase):
    def test_all_layers_decode_clean_capture(self):
        t, x, y = layered_decoder.synthesize_capture("A", noise_std=0.02)
        result = layered_decoder.decode_capture(t, x, y)
        self.assertEqual(result.selected.header, 0x7E)
        self.assertEqual(result.selected.letter, "A")
        self.assertEqual(result.selected.layer, "L4")
        self.assertIn("naive-max", result.successful_layers)
        self.assertIn("L4", result.successful_layers)

    def test_layered_decoder_handles_noise(self):
        for letter, seed in (("B", 1), ("M", 2), ("Z", 3)):
            with self.subTest(letter=letter):
                t, x, y = layered_decoder.synthesize_capture(
                    letter, noise_std=0.12, seed=seed
                )
                result = layered_decoder.decode_capture(t, x, y)
                self.assertEqual(result.selected.letter, letter)
                self.assertTrue(result.selected.success)

    def test_naive_parity_failure_is_visible(self):
        coded = protocol.encode_message("Q")
        r0 = np.where(coded == 1, 0.9, 0.1).astype(float)
        r1 = np.where(coded == 0, 0.9, 0.1).astype(float)
        # Flip one observation; constrained layers should still expose valid
        # candidates while naive-max reports its failed Hamming checks.
        r0[2], r1[2] = r1[2], r0[2]
        layers = layered_decoder.decode_observations(r0, r1)
        naive = layers[0]
        self.assertFalse(naive.parity_ok)
        self.assertFalse(naive.success)
        self.assertTrue(any(layer.success for layer in layers[1:]))


if __name__ == "__main__":
    unittest.main()
