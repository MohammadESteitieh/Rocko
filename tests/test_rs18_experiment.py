"""Frozen protocol, schedule, safety, and runner tests for RS(18,6)."""

import csv
from datetime import datetime
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT / "transmitter"), str(ROOT / "receiver")]
import final_experiment_protocol as gf  # noqa: E402
import rs18_experiment as experiment  # noqa: E402
import rs18_experiment_protocol as protocol  # noqa: E402
import run_rs18_experiment as runner  # noqa: E402
import transmitter as hw  # noqa: E402

GOLDEN_REPETITION_1_SYMBOLS = (
    7, 6, 17, 16, 19, 29, 25, 9, 20, 2, 23, 19, 0, 17, 10, 19, 23, 17,
)
GOLDEN_REPETITION_1_FRAME = (
    "0111111001111110"
    "001110011010001100001001111101110010100110100000101011110011000001000101010100111011110001"
)


class ProtocolTests(unittest.TestCase):
    def test_prospective_payload_derivation_is_exact_and_distinct(self):
        expected = {}
        for repetition in range(1, 6):
            digest = hashlib.sha256(
                f"CU-hacking-RS18-repetition-{repetition}".encode("ascii")
            ).digest()
            value = int.from_bytes(digest[:4], "big") >> 2
            expected[repetition] = f"{value:030b}"
        self.assertEqual(protocol.PAYLOADS, expected)
        self.assertEqual(len(set(expected.values())), 5)

    @staticmethod
    def _independent_mul(left, right):
        # Bitwise polynomial multiplication; deliberately does not use the
        # production exponent/log tables or encoder helpers.
        product = 0
        for _ in range(5):
            if right & 1:
                product ^= left
            right >>= 1
            left <<= 1
            if left & 0x20:
                left ^= 0x25
        return product & 0x1F

    @classmethod
    def _independent_encode(cls, data):
        alpha_powers = [1]
        for _ in range(1, 32):
            alpha_powers.append(cls._independent_mul(alpha_powers[-1], 2))
        generator = [1]
        for root in range(1, 13):
            factor = [1, alpha_powers[root]]  # descending x + alpha**root
            product = [0] * (len(generator) + 1)
            for i, left in enumerate(generator):
                for j, right in enumerate(factor):
                    product[i + j] ^= cls._independent_mul(left, right)
            generator = product
        work = list(data) + [0] * 12
        for index in range(6):
            coefficient = work[index]
            for offset, value in enumerate(generator):
                work[index + offset] ^= cls._independent_mul(coefficient, value)
        return tuple(data) + tuple(work[6:])

    def test_golden_repetition_one_matches_independent_bitwise_rs_encoder(self):
        independently_encoded = self._independent_encode((7, 6, 17, 16, 19, 29))
        self.assertEqual(independently_encoded, GOLDEN_REPETITION_1_SYMBOLS)
        self.assertEqual(protocol.CODEWORDS[1], independently_encoded)
        self.assertEqual(protocol.FRAMES[1], GOLDEN_REPETITION_1_FRAME)
        self.assertEqual(len(protocol.FRAMES[1]), 106)

    def test_all_codewords_are_systematic_and_have_twelve_zero_syndromes(self):
        for repetition, payload in protocol.PAYLOADS.items():
            data = gf.bits_to_symbols(payload)
            word = protocol.CODEWORDS[repetition]
            self.assertEqual(word[:6], data)
            self.assertEqual(len(word[6:]), 12)
            for root in range(1, 13):
                point = gf._gf_pow(root)
                value = 0
                for coefficient in word:
                    value = gf._gf_mul(value, point) ^ coefficient
                self.assertEqual(value, 0)

    def test_frame_contract_is_16_sync_plus_90_body_at_two_bits_per_second(self):
        self.assertEqual(protocol.SYNC_TEXT, "0111111001111110")
        self.assertEqual((protocol.DATA_SYMBOLS, protocol.PARITY_SYMBOLS), (6, 12))
        self.assertEqual((protocol.CODE_BITS, protocol.FRAME_BITS), (90, 106))
        self.assertEqual(protocol.BIT_SECONDS, 0.5)
        for frame in protocol.FRAMES.values():
            self.assertEqual(frame[:16], protocol.SYNC_TEXT)
            self.assertEqual(len(frame[16:]), 90)


class ScheduleTests(unittest.TestCase):
    def test_counterbalanced_schedule_has_45_frames_and_fixed_payload_per_repetition(self):
        schedule = experiment.experiment_schedule()
        self.assertEqual(len(schedule), 45)
        self.assertEqual(experiment.DUTY_ROTATIONS, (0, 2, 4, 6, 8))
        for repetition in range(1, 6):
            selected = [trial for trial in schedule if trial.repetition == repetition]
            self.assertEqual(len(selected), 9)
            self.assertEqual({trial.duty_percent for trial in selected},
                             set(protocol.DUTIES))
            self.assertEqual({trial.payload_bits for trial in selected},
                             {protocol.PAYLOADS[repetition]})
            self.assertEqual({trial.frame_bits for trial in selected},
                             {protocol.FRAMES[repetition]})
        for duty in protocol.DUTIES:
            self.assertEqual(sum(trial.duty_percent == duty for trial in schedule), 5)

    def test_duration_includes_initial_every_gap_and_final(self):
        self.assertEqual(experiment.estimated_seconds(), 3080.0)
        self.assertEqual(runner.PRELAUNCH_OFF_SECONDS, 120.0)
        self.assertEqual(experiment.estimated_seconds() + runner.PRELAUNCH_OFF_SECONDS,
                         3200.0)


class SafetyTests(unittest.TestCase):
    class FailingBackend:
        def __init__(self):
            self.calls = []

        def write_pin(self, pin, value):
            self.calls.append((pin, value))
            if pin == 18:
                raise OSError("injected")

    def test_cleanup_attempts_all_four_pins_after_individual_failure(self):
        backend = self.FailingBackend()
        self.assertFalse(experiment.safe_all_off(backend, hw.Config()))
        self.assertEqual(backend.calls, [(27, 0), (18, 0), (22, 0), (17, 0)])
        with self.assertRaises(hw.BeaconError):
            experiment.require_all_off(backend, hw.Config())

    def test_runner_independent_safe_command_contains_every_pin(self):
        for pin in (27, 18, 22, 17):
            self.assertIn(str(pin), runner.SAFE_GPIO_COMMAND)
        command = runner.transmitter_command(
            "/remote", "manifest.csv", "run.log", "bench A"
        )
        self.assertIn(runner.SAFE_GPIO_COMMAND, command)
        self.assertIn("./rs18_experiment.py", command)
        self.assertIn("--execute", command)
        self.assertIn("'bench A'", command)


class PrelaunchCaptureTests(unittest.TestCase):
    class Capture:
        def __init__(self, exit_after=None):
            self.polls = 0
            self.exit_after = exit_after

        def poll(self):
            self.polls += 1
            return 1 if self.exit_after is not None and self.polls >= self.exit_after else None

    def test_baseline_clock_starts_only_after_raw_capture_header_exists(self):
        capture = self.Capture()
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.csv"
            raw.write_text("t,x,y\n", encoding="ascii")
            elapsed = runner.wait_capture_ready(capture, raw)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertGreater(capture.polls, 0)

    def test_capture_readiness_has_a_hard_timeout(self):
        clock = [0.0]
        capture = self.Capture()
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.csv"
            with self.assertRaisesRegex(RuntimeError, "ready"):
                runner.wait_capture_ready(
                    capture, missing, timeout=1.0,
                    monotonic=lambda: clock[0],
                    sleep=lambda duration: clock.__setitem__(0, clock[0] + duration),
                )

    def test_120_second_prelaunch_is_measured_while_capture_stays_alive(self):
        clock = [10.0]
        capture = self.Capture()

        def monotonic():
            return clock[0]

        def sleep(duration):
            clock[0] += duration

        elapsed = runner.prelaunch_capture_wait(
            capture, 120.0, monotonic=monotonic, sleep=sleep, poll_seconds=7.0
        )
        self.assertEqual(elapsed, 120.0)
        self.assertGreater(capture.polls, 10)

    def test_capture_failure_aborts_prelaunch_before_transmitter(self):
        clock = [0.0]
        capture = self.Capture(exit_after=3)
        with self.assertRaisesRegex(RuntimeError, "prelaunch"):
            runner.prelaunch_capture_wait(
                capture, 120.0, monotonic=lambda: clock[0],
                sleep=lambda duration: clock.__setitem__(0, clock[0] + duration),
                poll_seconds=1.0,
            )


class ManifestAndRunnerTests(unittest.TestCase):
    def test_deployment_includes_transitive_alphabet_module(self):
        self.assertIn("alphabet_transmitter.py", runner.DEPLOY_NAMES)
        self.assertEqual(len(runner.DEPLOY_NAMES), len(set(runner.DEPLOY_NAMES)))

    def _write_manifest(self, path):
        fields = [
            "sequence", "scheme", "repetition", "duty_position", "duty_percent",
            "voltage_v", "distance_m", "payload_derivation", "payload_bits",
            "data_symbols", "parity_symbols", "coded_body_bits", "frame_bits",
            "bit_seconds", "started_utc", "finished_utc", "duration_s", "gap_after_s",
        ]
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for sequence, trial in enumerate(experiment.experiment_schedule(), 1):
                word = protocol.CODEWORDS[trial.repetition]
                writer.writerow({
                    "sequence": sequence, "scheme": "rs18_6",
                    "repetition": trial.repetition,
                    "duty_position": trial.duty_position,
                    "duty_percent": trial.duty_percent, "voltage_v": 11,
                    "distance_m": 3, "payload_derivation": protocol.PAYLOAD_DERIVATION,
                    "payload_bits": trial.payload_bits,
                    "data_symbols": " ".join(map(str, word[:6])),
                    "parity_symbols": " ".join(map(str, word[6:])),
                    "coded_body_bits": trial.frame_bits[16:],
                    "frame_bits": trial.frame_bits, "bit_seconds": 0.5,
                    "started_utc": "2026-01-01T00:00:00Z",
                    "finished_utc": "2026-01-01T00:00:53Z",
                    "duration_s": 53, "gap_after_s": 15,
                })

    def test_manifest_validator_enforces_every_declared_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            self._write_manifest(path)
            runner.validate_manifest(path)
            with path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            rows[22]["payload_bits"] = "0" * 30
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "golden contract"):
                runner.validate_manifest(path)

    def test_runner_name_and_subprocess_timeout(self):
        self.assertEqual(
            runner.run_name(datetime(2026, 7, 28, 12, 34, 56)),
            "rs18_experiment_11V_20260728_123456",
        )
        command = [sys.executable, "-c", "import time; time.sleep(1)"]
        with self.assertRaises(subprocess.TimeoutExpired):
            runner.checked_run(command, timeout=0.01)
        self.assertIsNone(runner.unchecked_run(command, timeout=0.01))


if __name__ == "__main__":
    unittest.main()
