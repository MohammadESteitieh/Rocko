"""Tests for the finite 100%/1% transmitter diagnostic."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "transmitter"))

import duty_pair_test as diagnostic  # noqa: E402
import hardware as hw  # noqa: E402


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeDriver:
    def __init__(self):
        self.enabled = False
        self.enable_events = []
        self.polarities = []
        self.off_calls = 0

    def set_polarity(self, forward):
        self.polarities.append(forward)

    def enable(self, on):
        self.enabled = on
        self.enable_events.append(on)

    def all_off(self):
        self.enabled = False
        self.off_calls += 1


class ScheduleTests(unittest.TestCase):
    def test_exact_paired_schedule(self):
        self.assertEqual(
            diagnostic.test_schedule(),
            (
                (100.0, "A"), (1.0, "A"),
                (100.0, "B"), (1.0, "B"),
                (100.0, "C"), (1.0, "C"),
                (100.0, "D"), (1.0, "D"),
                (100.0, "E"), (1.0, "E"),
            ),
        )

    def test_estimate_uses_current_half_baud_frame_duration(self):
        self.assertEqual(diagnostic.estimated_seconds(), 695.0)

    def test_descending_dataset_has_balanced_train_and_held_out_frames(self):
        schedule = diagnostic.dataset_schedule()
        self.assertEqual(len(schedule), 30)
        for index, duty in enumerate(diagnostic.DATASET_DUTIES):
            level = schedule[index * 6:(index + 1) * 6]
            self.assertEqual([item[0] for item in level], [duty] * 6)
            self.assertEqual(
                "".join(item[1] for item in level),
                diagnostic.LETTERS + diagnostic.HELD_OUT_LETTERS[index],
            )
        self.assertEqual(
            diagnostic.estimated_seconds(schedule=schedule), 2115.0
        )

    def test_single_frame_schedule_and_duration(self):
        schedule = diagnostic.requested_schedule(0.1, "e")
        self.assertEqual(schedule, ((0.1, "E"),))
        self.assertEqual(diagnostic.estimated_seconds(schedule=schedule), 56.0)

    def test_dataset_and_single_frame_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            diagnostic.requested_schedule(10, "A", dataset=True)

    def test_single_frame_arguments_are_strict(self):
        for duty, letter in ((0, "A"), (101, "A"), (0.1, "AA"), (0.1, "1")):
            with self.subTest(duty=duty, letter=letter), self.assertRaises(ValueError):
                diagnostic.requested_schedule(duty, letter)


class PulseTests(unittest.TestCase):
    def test_one_percent_at_eight_hz_uses_625_microsecond_pulses(self):
        clock = FakeClock()
        driver = FakeDriver()
        config = hw.Config()
        transmitter = diagnostic.DutyFrameTransmitter(
            driver, config, monotonic=clock.monotonic, sleep=clock.sleep
        )
        stats = transmitter.transmit_frame("1", 1.0)
        self.assertAlmostEqual(clock.now, 1.0)
        self.assertEqual(len(stats.widths), 8)
        for width in stats.widths:
            self.assertAlmostEqual(width, 0.000625)
        self.assertEqual(stats.late_starts, 0)
        self.assertFalse(driver.enabled)
        self.assertGreaterEqual(driver.off_calls, 1)

    def test_point_one_percent_at_eight_hz_targets_62_point_5_microseconds(self):
        clock = FakeClock()
        driver = FakeDriver()
        transmitter = diagnostic.DutyFrameTransmitter(
            driver, hw.Config(), monotonic=clock.monotonic, sleep=clock.sleep
        )
        stats = transmitter.transmit_frame("1", 0.1)
        self.assertEqual(len(stats.widths), 8)
        self.assertAlmostEqual(stats.target_seconds, 0.0000625)
        for width in stats.widths:
            self.assertAlmostEqual(width, 0.0000625)

    def test_invalid_duty_is_rejected_before_gpio(self):
        clock = FakeClock()
        driver = FakeDriver()
        transmitter = diagnostic.DutyFrameTransmitter(
            driver, hw.Config(), monotonic=clock.monotonic, sleep=clock.sleep
        )
        for duty in (0, -1, 101):
            with self.subTest(duty=duty), self.assertRaises(ValueError):
                transmitter.transmit_frame("1", duty)
        self.assertEqual(driver.enable_events, [])


if __name__ == "__main__":
    unittest.main()
