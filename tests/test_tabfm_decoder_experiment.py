import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path[:0] = [
    str(ROOT / "experiments" / "tabfm_decoder"),
    str(ROOT / "receiver" / "legacy_decoder"),
    str(ROOT / "receiver"),
    str(ROOT / "transmitter"),
]

import export_rs18_table as exporter  # noqa: E402
import rs18_experiment_protocol as protocol  # noqa: E402
import run_tabfm  # noqa: E402


class TabFMTableTests(unittest.TestCase):
    @staticmethod
    def rows():
        rows = []
        for sequence, repetition in enumerate((1, 2, 3), 1):
            frame = protocol.FRAMES[repetition]
            for frame_index, character in enumerate(frame):
                section = "sync" if frame_index < len(protocol.SYNC_TEXT) else "body"
                body_index = frame_index - len(protocol.SYNC_TEXT)
                rows.append({
                    "sequence": sequence,
                    "repetition": repetition,
                    "duty_percent": 100.0,
                    "section": section,
                    "frame_bit_index": frame_index,
                    "body_bit_index": body_index if section == "body" else "",
                    "target_bit": int(character),
                    "symbol_index": body_index // 5 if section == "body" else "sync",
                    "bit_in_symbol": body_index % 5 if section == "body" else frame_index,
                    "baseline_llr": 1.0 if character == "1" else -1.0,
                    "baseline_bit": int(character),
                })
        return rows

    def test_prompt_mixes_known_context_with_blank_query_truth(self):
        prompt, truth = exporter.make_prompt(self.rows(), 1, context_rows=100)
        context = [row for row in prompt if row["role"] != "query"]
        query = [row for row in prompt if row["role"] == "query"]
        self.assertEqual(len(context), 100)
        self.assertEqual(len(query), protocol.CODE_BITS)
        self.assertEqual(len(truth), protocol.CODE_BITS)
        self.assertEqual({row["target_bit"] for row in context}, {0, 1})
        self.assertTrue(all(row["target_bit"] == "" for row in query))
        self.assertTrue(all(row["repetition"] != 1
                            for row in context if row["role"] == "historical"))
        self.assertEqual(truth[0]["payload_bits"], protocol.PAYLOADS[1])

    def test_perfect_probabilities_decode_the_payload(self):
        prompt, truth = exporter.make_prompt(self.rows(), 1, context_rows=100)
        query = [row for row in prompt if row["role"] == "query"]
        probability_one = np.asarray([
            0.999 if int(row["target_bit"]) else 0.001 for row in truth
        ])
        result = run_tabfm.evaluate(probability_one, query, truth)
        self.assertEqual(result["tabfm_bit_errors"], 0)
        self.assertEqual(result["tabfm_symbol_errors"], 0)
        self.assertEqual(result["tabfm_decoder_failure"], 0)
        self.assertEqual(result["tabfm_payload_bit_errors_conditional_on_decode"], 0)
        self.assertEqual(result["baseline_decoder_failure"], 0)
        self.assertEqual(result["baseline_payload_bit_errors_conditional_on_decode"], 0)

    def test_evaluator_rejects_truth_from_another_frame(self):
        prompt, truth = exporter.make_prompt(self.rows(), 1, context_rows=100)
        query = [row for row in prompt if row["role"] == "query"]
        truth[0]["sequence"] = 2
        with self.assertRaisesRegex(ValueError, "exactly one identical frame"):
            run_tabfm.evaluate(np.full(protocol.CODE_BITS, 0.5), query, truth)

    def test_evaluator_rejects_mixed_frames_even_when_rows_match_pairwise(self):
        prompt, truth = exporter.make_prompt(self.rows(), 1, context_rows=100)
        query = [row for row in prompt if row["role"] == "query"]
        for row in query[45:]:
            row["sequence"] = 2
        for row in truth[45:]:
            row["sequence"] = 2
        with self.assertRaisesRegex(ValueError, "exactly one identical frame"):
            run_tabfm.evaluate(np.full(protocol.CODE_BITS, 0.5), query, truth)

    def test_runner_rejects_historical_context_from_query_repetition(self):
        prompt, truth = exporter.make_prompt(self.rows(), 1, context_rows=100)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "table.csv"
            truth_path = root / "truth.csv"
            metadata_path = root / "metadata.json"
            exporter.write_csv(table, exporter.TABLE_COLUMNS, prompt)
            exporter.write_csv(truth_path, exporter.TRUTH_COLUMNS, truth)
            metadata = {
                "experiment": "tabfm-rs18-bit-frontend-pilot",
                "query_sequence": 1,
                "query_repetition_held_out_from_historical_context": 1,
                "context_rows": 100,
                "query_rows": protocol.CODE_BITS,
                "feature_columns": exporter.FEATURE_COLUMNS,
                "table_sha256": run_tabfm.sha256_file(table),
                "truth_sha256": run_tabfm.sha256_file(truth_path),
                "source": {
                    "capture_path": "/capture.csv",
                    "capture_sha256": "0" * 64,
                    "manifest_path": "/manifest.csv",
                    "manifest_sha256": "1" * 64,
                    "metadata_path": "/source-metadata.txt",
                    "metadata_sha256": "2" * 64,
                    "manifest_clock_correction_s": 0.0,
                    "boundary_sync_score": 1.0,
                    "boundary_sync_errors": 0,
                },
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            serialized_prompt = run_tabfm.read_rows(table)
            run_tabfm.validate_export(
                table, truth_path, metadata_path, serialized_prompt
            )

            leaked = [dict(row) for row in serialized_prompt]
            next(row for row in leaked if row["role"] == "historical")["repetition"] = "1"
            exporter.write_csv(table, exporter.TABLE_COLUMNS, leaked)
            metadata["table_sha256"] = run_tabfm.sha256_file(table)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leaks"):
                run_tabfm.validate_export(table, truth_path, metadata_path, leaked)

    def test_accepted_sequence_24_extraction_reproduces_frozen_baseline(self):
        rows, provenance = exporter.extract_rows(
            exporter.DEFAULT_CAPTURE,
            exporter.DEFAULT_MANIFEST,
            exporter.DEFAULT_METADATA,
        )
        body = [row for row in rows
                if row["sequence"] == 24 and row["section"] == "body"]
        self.assertEqual(len(body), protocol.CODE_BITS)
        self.assertEqual(sum(row["baseline_bit"] != row["target_bit"] for row in body), 8)
        self.assertEqual(sum(
            any(row["baseline_bit"] != row["target_bit"]
                for row in body[start:start + 5])
            for start in range(0, protocol.CODE_BITS, 5)
        ), 7)
        self.assertEqual(
            provenance["capture_sha256"],
            "ee730cb3ef0998d6ab7399a3a99580380fd3afe8e16c80ba59c9f162294b8fba",
        )


if __name__ == "__main__":
    unittest.main()
