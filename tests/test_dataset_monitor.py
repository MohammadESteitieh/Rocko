"""Tests for read-only growing-capture dataset utilities."""

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
sys.path[:0] = [
    str(ROOT / "receiver" / "legacy_decoder"), str(ROOT / "receiver")
]

import monitor_dataset  # noqa: E402
import watch_capture  # noqa: E402


class DatasetMonitorTests(unittest.TestCase):
    def test_schedule_has_five_train_and_one_held_out_per_duty(self):
        self.assertEqual(len(monitor_dataset.SCHEDULE), 30)
        for index, duty in enumerate(monitor_dataset.DUTIES):
            level = monitor_dataset.SCHEDULE[index * 6:(index + 1) * 6]
            self.assertEqual([row[0] for row in level], [duty] * 6)
            self.assertEqual([row[2] for row in level], ["train"] * 5 + ["test"])
            self.assertEqual(level[-1][1], monitor_dataset.TEST[index])

    def test_metadata_parser_and_utc_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.txt"
            path.write_text("alpha=one\nstarted=2026-07-16T18:00:00Z\n")
            parsed = monitor_dataset.metadata(path)
            self.assertEqual(parsed["alpha"], "one")
            self.assertEqual(
                monitor_dataset.utc_seconds(parsed["started"]),
                monitor_dataset.utc_seconds("2026-07-16T18:00:00+00:00"),
            )

    def test_growing_viewer_row_parser_is_strict(self):
        self.assertEqual(watch_capture.parse_row("1.5,20,30\n"), (1.5, 20.0, 30.0))
        self.assertIsNone(watch_capture.parse_row("t,x,y\n"))
        self.assertIsNone(watch_capture.parse_row("1,2\n"))


if __name__ == "__main__":
    unittest.main()
