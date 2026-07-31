"""Tests for the counterbalanced voltage/duty calibration transmitter."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "transmitter"))

import calibration_sweep as calibration  # noqa: E402
import transmitter as hw  # noqa: E402


class RecordingBackend:
    def __init__(self, fail_pin=None):
        self.fail_pin = fail_pin
        self.events = []

    def write_pin(self, pin, value):
        self.events.append((pin, value))
        if pin == self.fail_pin:
            raise RuntimeError("injected GPIO failure")


class CalibrationScheduleTests(unittest.TestCase):
    def test_counterbalanced_schedule_is_frozen(self):
        self.assertEqual(
            calibration.calibration_schedule(),
            (
                (1, 1, 100.0, "A"), (1, 2, 50.0, "A"),
                (1, 3, 25.0, "A"), (1, 4, 10.0, "A"),
                (2, 1, 10.0, "A"), (2, 2, 25.0, "A"),
                (2, 3, 50.0, "A"), (2, 4, 100.0, "A"),
                (3, 1, 50.0, "A"), (3, 2, 10.0, "A"),
                (3, 3, 100.0, "A"), (3, 4, 25.0, "A"),
            ),
        )

    def test_every_duty_occurs_three_times(self):
        duties = [entry[2] for entry in calibration.calibration_schedule()]
        for duty in (100.0, 50.0, 25.0, 10.0):
            self.assertEqual(duties.count(duty), 3)
        self.assertNotIn(1.0, duties)

    def test_duration_includes_initial_every_gap_and_final_wait(self):
        self.assertEqual(calibration.estimated_seconds(), 872.0)

    def test_explicit_duty_sequence_is_one_declared_round(self):
        duties = calibration.parse_duty_sequence("100,50,25,10,5,1")
        schedule = calibration.exploratory_schedule(duties)
        self.assertEqual(
            schedule,
            (
                (1, 1, 100.0, "A"), (1, 2, 50.0, "A"),
                (1, 3, 25.0, "A"), (1, 4, 10.0, "A"),
                (1, 5, 5.0, "A"), (1, 6, 1.0, "A"),
            ),
        )
        self.assertEqual(calibration.estimated_seconds(schedule=schedule), 446.0)
        self.assertEqual(
            calibration.estimated_seconds(schedule=schedule[:1], bit_seconds=1.0),
            63.0,
        )
        self.assertEqual(
            calibration.estimated_seconds(schedule=schedule[:1], bit_seconds=0.5),
            49.0,
        )

    def test_invalid_explicit_duty_sequences_are_rejected(self):
        for value in ("", "100,0", "101", "5,5", "five"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                calibration.parse_duty_sequence(value)

    def test_negative_timing_is_rejected(self):
        for kwargs in (
            {"initial_wait": -1}, {"gap": -1}, {"final_wait": -1}
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                calibration.estimated_seconds(**kwargs)

    def test_normal_voltage_grid_and_conditional_six_volts(self):
        for voltage in (7, 8, 9, 10, 11):
            self.assertEqual(calibration.validate_voltage(voltage), float(voltage))
        with self.assertRaisesRegex(ValueError, "allow-six-volts"):
            calibration.validate_voltage(6)
        self.assertEqual(calibration.validate_voltage(6, True), 6.0)
        for voltage in (5, 6.5, 12):
            with self.subTest(voltage=voltage), self.assertRaises(ValueError):
                calibration.validate_voltage(voltage)


class SafeShutdownTests(unittest.TestCase):
    def test_all_current_and_legacy_pins_are_forced_low_in_safe_order(self):
        backend = RecordingBackend()
        self.assertTrue(calibration.safe_all_off(backend, hw.Config()))
        self.assertEqual(
            backend.events,
            [(27, 0), (18, 0), (22, 0), (17, 0)],
        )

    def test_one_gpio_failure_does_not_skip_other_cleanup_pins(self):
        backend = RecordingBackend(fail_pin=18)
        self.assertFalse(calibration.safe_all_off(backend, hw.Config()))
        self.assertEqual(
            backend.events,
            [(27, 0), (18, 0), (22, 0), (17, 0)],
        )
        backend = RecordingBackend(fail_pin=18)
        with self.assertRaisesRegex(hw.BeaconError, "forced low"):
            calibration.require_all_off(backend, hw.Config())


if __name__ == "__main__":
    unittest.main()
