"""Frozen vectors, schedule, persistence, and safety tests for the final experiment."""

import csv
from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT / "transmitter"), str(ROOT / "receiver")]

import final_experiment as experiment  # noqa: E402
import final_experiment_protocol as protocol  # noqa: E402
import run_final_experiment as runner  # noqa: E402
import transmitter as hw  # noqa: E402


GOLDEN_FRAMES = {
    "uncoded": "011111100111111010001011100",
    "hamming15_11": "0111111001111110001000001011100",
    "rs5_3": "01111110011111101000101110000001110011111",
    "rs12_6": (
        "0111111001111110100010111000010010100111011111000011100001001010001010001001"
    ),
}


class ProtocolVectorTests(unittest.TestCase):
    def test_independently_derived_golden_frames_match_exactly(self):
        self.assertEqual(protocol.FRAMES, GOLDEN_FRAMES)
        self.assertEqual(
            {name: len(frame) for name, frame in protocol.FRAMES.items()},
            {"uncoded": 27, "hamming15_11": 31, "rs5_3": 41, "rs12_6": 76},
        )

    def test_payload_and_rs_symbol_conventions_are_frozen(self):
        self.assertEqual(protocol.SYNC_TEXT, "0111111001111110")
        self.assertEqual(protocol.SHORT_PAYLOAD_TEXT, "10001011100")
        self.assertEqual(protocol.FINAL_PAYLOAD_TEXT,
                         "100010111000010010100111011111")
        self.assertEqual(protocol.PRIMITIVE_POLYNOMIAL, 0x25)
        self.assertEqual(protocol.bits_to_symbols(protocol.SHORT_PAYLOAD_TEXT + "0000"),
                         (17, 14, 0))
        self.assertEqual(protocol.rs_encode_symbols((17, 14, 0), 2),
                         (17, 14, 0, 28, 31))
        self.assertEqual(protocol.rs_encode_symbols((17, 14, 2, 10, 14, 31), 6),
                         (17, 14, 2, 10, 14, 31, 1, 24, 9, 8, 20, 9))

    def test_hamming_body_has_zero_even_parity_syndrome(self):
        body = tuple(map(int, protocol.FRAMES["hamming15_11"][16:]))
        syndrome = 0
        for position, bit in enumerate(body, 1):
            if bit:
                syndrome ^= position
        self.assertEqual(syndrome, 0)


class FrozenScheduleTests(unittest.TestCase):
    def test_full_collection_is_exactly_one_hundred_frames(self):
        short = experiment.short_schedule()
        final = experiment.final_schedule()
        self.assertEqual((len(short), len(final), len(short + final)), (75, 25, 100))
        for duty in protocol.DUTIES:
            for scheme in protocol.SHORT_SCHEMES:
                selected = [trial for trial in short
                            if trial.duty_percent == duty and trial.scheme == scheme]
                self.assertEqual(len(selected), 5)
                self.assertEqual({trial.repetition for trial in selected}, set(range(1, 6)))
            selected_final = [trial for trial in final if trial.duty_percent == duty]
            self.assertEqual(len(selected_final), 5)

    def test_duty_order_is_counterbalanced_across_repetitions(self):
        duties = protocol.DUTIES
        short = experiment.short_schedule()
        final = experiment.final_schedule()
        for repetition in range(1, 6):
            expected = duties[repetition - 1:] + duties[:repetition - 1]
            short_order = tuple(
                trial.duty_percent
                for index, trial in enumerate(
                    [item for item in short if item.repetition == repetition]
                )
                if index % len(protocol.SHORT_SCHEMES) == 0
            )
            final_order = tuple(
                trial.duty_percent
                for trial in final if trial.repetition == repetition
            )
            self.assertEqual(short_order, expected)
            self.assertEqual(final_order, expected)

    def test_approved_timing_and_collection_estimates_are_frozen(self):
        self.assertEqual(protocol.BIT_SECONDS, 0.5)
        self.assertEqual(experiment.estimated_seconds(experiment.short_schedule()), 2382.5)
        self.assertEqual(experiment.estimated_seconds(experiment.final_schedule()), 1345.0)
        self.assertEqual(experiment.estimated_seconds(experiment.experiment_schedule("all")),
                         3707.5)

    def test_one_percent_trials_retain_timing_limited_label_source(self):
        self.assertEqual(protocol.DUTIES[-1], 1.0)
        self.assertEqual(sum(t.duty_percent == 1 for t in experiment.short_schedule()), 15)
        self.assertEqual(sum(t.duty_percent == 1 for t in experiment.final_schedule()), 5)


class SafetyAndRunnerTests(unittest.TestCase):
    class FailingBackend:
        def __init__(self):
            self.calls = []

        def write_pin(self, pin, value):
            self.calls.append((pin, value))
            if pin == 18:
                raise OSError("injected")

    def test_safe_shutdown_attempts_all_four_pins_despite_failure(self):
        backend = self.FailingBackend()
        self.assertFalse(experiment.safe_all_off(backend, hw.Config()))
        self.assertEqual(backend.calls, [(27, 0), (18, 0), (22, 0), (17, 0)])

    def test_runner_deploys_every_transitive_qnx_import(self):
        self.assertEqual(
            runner.DEPLOY_NAMES,
            (
                "transmitter.py", "alphabet_transmitter.py", "duty_pair_test.py",
                "final_experiment_protocol.py", "final_experiment.py",
            ),
        )

    def test_runner_names_duration_and_remote_command(self):
        when = datetime(2026, 7, 27, 20, 0, 0)
        self.assertEqual(runner.run_name("all", when),
                         "final_experiment_all_11V_20260727_200000")
        self.assertEqual(runner.expected_seconds("short"), 2382.5)
        command = runner.transmitter_command(
            "/remote", "all", "manifest.csv", "run.log", "bench A"
        )
        self.assertIn(runner.SAFE_GPIO_COMMAND, command)
        self.assertIn("./final_experiment.py", command)
        self.assertIn("--phase all", command)
        self.assertIn("--execute", command)
        self.assertIn("'bench A'", command)
        self.assertNotIn("sshpass", command)

    def _write_manifest(self, path, phase):
        fields = [
            "sequence", "phase", "scheme", "repetition", "duty_percent",
            "voltage_v", "distance_m", "payload_bits", "coded_body_bits",
            "frame_bits", "bit_seconds", "started_utc", "finished_utc", "duration_s",
        ]
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for sequence, trial in enumerate(runner.expected_trials(phase), 1):
                frame = protocol.FRAMES[trial.scheme]
                writer.writerow({
                    "sequence": sequence, "phase": trial.phase,
                    "scheme": trial.scheme, "repetition": trial.repetition,
                    "duty_percent": trial.duty_percent, "voltage_v": 11,
                    "distance_m": 3,
                    "payload_bits": (protocol.FINAL_PAYLOAD_TEXT
                                     if trial.phase == "final"
                                     else protocol.SHORT_PAYLOAD_TEXT),
                    "coded_body_bits": frame[16:], "frame_bits": frame,
                    "bit_seconds": 0.5, "started_utc": "2026-01-01T00:00:00Z",
                    "finished_utc": "2026-01-01T00:00:01Z", "duration_s": 1,
                })

    def test_manifest_validator_enforces_all_100_golden_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            self._write_manifest(path, "all")
            runner.validate_manifest(path, "all")
            with path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            rows[72]["frame_bits"] = rows[72]["frame_bits"][:-1] + "0"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "golden frame"):
                runner.validate_manifest(path, "all")


if __name__ == "__main__":
    unittest.main()
