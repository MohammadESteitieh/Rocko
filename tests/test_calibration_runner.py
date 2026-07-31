"""Tests for the macOS calibration-session orchestrator helpers."""

from datetime import datetime
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "receiver"))

import run_calibration as runner  # noqa: E402


class CalibrationRunnerTests(unittest.TestCase):
    def test_run_name_is_stable_and_filesystem_safe(self):
        when = datetime(2026, 7, 25, 12, 34, 56)
        self.assertEqual(
            runner.run_name(10, when), "calibration_10V_20260725_123456"
        )
        self.assertEqual(
            runner.run_name(7.5, when), "calibration_7p5V_20260725_123456"
        )

    def test_custom_duty_sequence_has_safe_name_and_duration(self):
        duties = runner.parse_duty_sequence("100,50,25,10,5,1")
        self.assertEqual(duties, (100.0, 50.0, 25.0, 10.0, 5.0, 1.0))
        self.assertEqual(runner.expected_session_seconds(duties), 446.0)
        self.assertEqual(
            runner.expected_session_seconds((100.0,), bit_seconds=1.0),
            63.0,
        )
        self.assertEqual(
            runner.expected_session_seconds((100.0,), bit_seconds=0.5),
            49.0,
        )
        when = datetime(2026, 7, 25, 12, 34, 56)
        self.assertEqual(
            runner.run_name(11, when, prefix="duty_sweep"),
            "duty_sweep_11V_20260725_123456",
        )
        for value in ("", "0", "101", "5,5", "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.parse_duty_sequence(value)

    def test_hardware_note_must_be_nonempty_single_line(self):
        self.assertEqual(runner.validate_note(" bench-v1 "), "bench-v1")
        for note in ("", "  ", "line1\nline2", "line1\rline2"):
            with self.subTest(note=note), self.assertRaises(ValueError):
                runner.validate_note(note)

    def test_remote_directory_uses_conservative_absolute_qnx_path(self):
        valid = "/data/home/qnxuser/calibration-runner"
        self.assertEqual(runner.validate_remote_dir(valid), valid)
        for value in ("relative", "/data/with space", "/data/../tmp", "/data/$HOME"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.validate_remote_dir(value)

    def test_serial_selection_accepts_one_or_requires_explicit_choice(self):
        self.assertEqual(
            runner.select_serial_port(None, ["/dev/cu.usbmodem1201"]),
            "/dev/cu.usbmodem1201",
        )
        for candidates in ([], ["one", "two"]):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                runner.select_serial_port(None, candidates)
        with tempfile.TemporaryDirectory() as directory:
            port = Path(directory) / "port"
            port.touch()
            self.assertEqual(runner.select_serial_port(str(port), []), str(port))

    def test_remote_command_has_safe_start_and_no_embedded_password(self):
        command = runner.transmitter_command(
            remote_dir="/data/home/qnxuser/calibration-runner",
            manifest_name="run.transmitter.csv",
            log_name="run.transmitter.log",
            voltage=11,
            distance_m=1.25,
            hardware_note="bench configuration A",
            allow_six_volts=False,
        )
        self.assertIn(runner.SAFE_GPIO_COMMAND, command)
        self.assertIn("./calibration_sweep.py", command)
        self.assertIn("--voltage 11", command)
        self.assertIn("'bench configuration A'", command)
        self.assertIn("nohup", command)
        self.assertNotIn("SSHPASS", command)
        self.assertNotIn("sshpass", command)

    def test_remote_command_can_request_existing_custom_sweep_mode(self):
        command = runner.transmitter_command(
            remote_dir="/remote",
            manifest_name="manifest.csv",
            log_name="run.log",
            voltage=11,
            distance_m=3,
            hardware_note="bench",
            allow_six_volts=False,
            duty_sequence=(100.0, 50.0, 25.0, 10.0, 5.0, 1.0),
        )
        self.assertIn("--duty-sequence 100,50,25,10,5,1", command)
        self.assertIn(runner.SAFE_GPIO_COMMAND, command)

    def test_one_bit_per_second_is_passed_to_transmitter_and_dashboard(self):
        command = runner.transmitter_command(
            remote_dir="/remote",
            manifest_name="manifest.csv",
            log_name="run.log",
            voltage=11,
            distance_m=3,
            hardware_note="bench",
            allow_six_volts=False,
            duty_sequence=(100.0,),
            bit_seconds=1.0,
        )
        self.assertIn("--bit-seconds 1", command)
        dashboard = runner.capture_command(
            Path("/repo"), "/venv/python", "/dev/port", 115200,
            Path("/data/run.csv"), live_dashboard=True, bit_seconds=1.0,
        )
        self.assertIn("--bit-seconds", dashboard)
        self.assertIn("1", dashboard)

    def test_six_volt_remote_command_includes_explicit_opt_in(self):
        command = runner.transmitter_command(
            remote_dir="/remote",
            manifest_name="manifest.csv",
            log_name="run.log",
            voltage=6,
            distance_m=1,
            hardware_note="logic documented",
            allow_six_volts=True,
        )
        self.assertIn("--allow-six-volts", command)

    def test_visible_dashboard_can_be_the_single_capture_owner(self):
        repo = Path("/repo")
        raw = Path("/data/run.csv")
        visible = runner.capture_command(
            repo, "/venv/python", "/dev/cu.usbmodem1201", 115200, raw,
            live_dashboard=True,
        )
        headless = runner.capture_command(
            repo, "/venv/python", "/dev/cu.usbmodem1201", 115200, raw,
            live_dashboard=False,
        )
        self.assertIn("/repo/receiver/rocko_receiver.py", visible)
        self.assertIn("--output", visible)
        self.assertNotIn("/repo/receiver/capture.py", visible)
        self.assertIn("/repo/receiver/capture.py", headless)
        self.assertIn("--out", headless)

    def test_subprocess_helpers_enforce_timeout_without_hanging_cleanup(self):
        command = [sys.executable, "-c", "import time; time.sleep(1)"]
        with self.assertRaises(subprocess.TimeoutExpired):
            runner.checked_run(command, timeout=0.01)
        self.assertIsNone(runner.unchecked_run(command, timeout=0.01))

    def test_transient_remote_contact_loss_has_bounded_grace(self):
        grace = runner.REMOTE_CONTACT_GRACE_SECONDS
        self.assertFalse(runner.remote_contact_expired(100.0, 100.0 + grace))
        self.assertTrue(runner.remote_contact_expired(100.0, 100.0 + grace + 0.001))

    def test_sha256_is_computed_without_mutating_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "capture.csv"
            artifact.write_bytes(b"abc")
            self.assertEqual(
                runner.sha256_file(artifact),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
            self.assertEqual(artifact.read_bytes(), b"abc")


if __name__ == "__main__":
    unittest.main()
