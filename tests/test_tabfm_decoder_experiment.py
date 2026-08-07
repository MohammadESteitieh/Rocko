import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path[:0] = [
    str(ROOT / "experiments" / "tabfm_decoder"),
    str(ROOT / "receiver" / "legacy_decoder"),
    str(ROOT / "receiver"),
    str(ROOT / "transmitter"),
]

import decode_capture  # noqa: E402
import export_rs18_table as exporter  # noqa: E402
import plot_meeting_summary  # noqa: E402
import rs18_experiment_protocol as protocol  # noqa: E402
import run_batch  # noqa: E402
import run_soft_list_batch  # noqa: E402
import run_tabfm  # noqa: E402
import soft_list_decoder  # noqa: E402


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
                    "sensor_y_llr": 1.0 if character == "1" else -1.0,
                    "sensor_y_abs_llr": 1.0,
                    "sensor_y_bit": int(character),
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
        self.assertEqual(result["sensor_y_decoder_failure"], 0)
        self.assertEqual(result["sensor_y_tabfm_gated_decoder_failure"], 0)

    def test_confidence_gate_retains_sensor_y_on_uncertain_tabfm_outputs(self):
        prompt, truth = exporter.make_prompt(self.rows(), 1, context_rows=100)
        query = [row for row in prompt if row["role"] == "query"]
        uncertain_wrong = np.asarray([
            0.51 if int(row["target_bit"]) == 0 else 0.49 for row in truth
        ])
        result = run_tabfm.evaluate(
            uncertain_wrong, query, truth, confidence_threshold=0.75
        )
        self.assertEqual(result["tabfm_bit_errors"], protocol.CODE_BITS)
        self.assertEqual(result["sensor_y_tabfm_gated_bit_errors"], 0)
        self.assertEqual(result["confidence_gate_applied_bits"], 0)

    def test_confidence_gate_applies_inclusively_at_both_boundaries(self):
        prompt, truth = exporter.make_prompt(self.rows(), 1, context_rows=100)
        query = [row for row in prompt if row["role"] == "query"]
        probabilities = np.asarray([
            0.5 if int(row["target_bit"]) == 0 else 0.5 for row in truth
        ])
        zero_index = next(i for i, row in enumerate(truth) if row["target_bit"] == 0)
        one_index = next(i for i, row in enumerate(truth) if row["target_bit"] == 1)
        probabilities[zero_index] = 0.75
        probabilities[one_index] = 0.25
        result = run_tabfm.evaluate(
            probabilities, query, truth, confidence_threshold=0.75
        )
        self.assertEqual(result["confidence_gate_applied_bits"], 2)
        self.assertEqual(result["sensor_y_tabfm_gated_bit_errors"], 2)

    def test_frozen_batch_separates_exploratory_payload_repetition(self):
        self.assertEqual(run_batch.EXPLORATORY_SEQUENCE, 24)
        self.assertEqual(run_batch.CONFIRMATORY_SEQUENCES, (2, 10, 28, 45))
        self.assertEqual(run_batch.FROZEN_CONFIDENCE_THRESHOLD, 0.75)
        self.assertEqual(
            run_batch.FROZEN_CHECKPOINT_REVISION,
            "d5e74033fcf257699fab013e2cdd7edf424ff904",
        )
        rows = exporter._manifest(exporter.DEFAULT_MANIFEST)
        by_sequence = {int(row["sequence"]): row for row in rows}
        exploratory_repetition = int(by_sequence[24]["repetition"])
        confirmatory_repetitions = {
            int(by_sequence[sequence]["repetition"])
            for sequence in run_batch.CONFIRMATORY_SEQUENCES
        }
        self.assertNotIn(exploratory_repetition, confirmatory_repetitions)
        self.assertEqual(confirmatory_repetitions, {1, 2, 4, 5})
        self.assertEqual(
            [float(by_sequence[sequence]["duty_percent"])
             for sequence in run_batch.CONFIRMATORY_SEQUENCES],
            [50.0, 45.0, 25.0, 10.0],
        )

    def test_soft_list_frozen_development_outcomes(self):
        root = (
            ROOT / "data/captures/rs18-experiment/derived/tabfm"
            / "sensor-y-hybrid-frozen-v1"
        )
        expected = {
            24: (0, 0, "hard"),
            2: (0, 0, "list"),
            10: (1, None, "list-rejected"),
            28: (1, None, "list-rejected"),
            45: (1, None, "list-rejected"),
        }
        for sequence, (failure, payload_errors, mode) in expected.items():
            predictions = run_tabfm.read_rows(
                root / f"rs18-sequence-{sequence:02d}.predictions.csv"
            )
            truth = run_tabfm.read_rows(
                root / f"rs18-sequence-{sequence:02d}.truth.csv"
            )
            llrs = soft_list_decoder.probability_llrs([
                float(row["tabfm_probability_one"]) for row in predictions
            ])
            decoded = soft_list_decoder.decode(llrs)
            self.assertEqual(decoded["failure"], failure, sequence)
            self.assertEqual(decoded["mode"], mode, sequence)
            actual_errors = None if decoded["failure"] else sum(
                left != right
                for left, right in zip(
                    decoded["payload"], map(int, truth[0]["payload_bits"])
                )
            )
            self.assertEqual(actual_errors, payload_errors, sequence)

    def test_simple_decoder_reproduces_development_recovery_without_truth(self):
        root = (
            ROOT / "data/captures/rs18-experiment/derived/tabfm"
            / "sensor-y-hybrid-frozen-v1"
        )
        prompt = run_tabfm.read_rows(root / "rs18-sequence-02.context-query.csv")
        predictions = run_tabfm.read_rows(root / "rs18-sequence-02.predictions.csv")
        probabilities = np.asarray([
            float(row["tabfm_probability_one"]) for row in predictions
        ])
        result = decode_capture.decode_prompt(
            None, prompt,
            predictor=lambda _model, _rows, _features: probabilities,
        )
        truth = run_tabfm.read_rows(root / "rs18-sequence-02.truth.csv")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["mode"], "list")
        self.assertEqual(result["payload_bits"], truth[0]["payload_bits"])
        self.assertAlmostEqual(result["score_margin"], 53.31597703494161)

    def test_simple_decoder_does_not_attest_an_injected_model_as_pinned(self):
        decoded = {
            "accepted": False,
            "payload_bits": None,
            "tabfm_probability_one": [0.5] * protocol.CODE_BITS,
        }
        with (
            mock.patch.object(
                decode_capture.exporter, "extract_rows",
                return_value=([], {"capture_sha256": "example"}),
            ),
            mock.patch.object(
                decode_capture.exporter, "make_prompt",
                return_value=([{"role": "placeholder"}], []),
            ),
            mock.patch.object(
                decode_capture, "decode_prompt", return_value=decoded,
            ),
        ):
            result = decode_capture.decode_capture(2, model=object())
        self.assertFalse(result["pinned_checkpoint_verified"])
        self.assertIsNone(result["checkpoint_revision"])
        self.assertEqual(result["model"], "injected model object")

    def test_simple_decoder_rejects_unconfirmed_candidate(self):
        root = (
            ROOT / "data/captures/rs18-experiment/derived/tabfm"
            / "soft-list-confirmatory-v1"
        )
        prompt = run_tabfm.read_rows(root / "rs18-sequence-03.context-query.csv")
        predictions = run_tabfm.read_rows(root / "rs18-sequence-03.predictions.csv")
        probabilities = np.asarray([
            float(row["tabfm_probability_one"]) for row in predictions
        ])
        result = decode_capture.decode_prompt(
            None, prompt,
            predictor=lambda _model, _rows, _features: probabilities,
        )
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["payload_bits"])
        self.assertEqual(result["mode"], "list-rejected")

    def test_meeting_plot_data_matches_frozen_results(self):
        rows = plot_meeting_summary.build_plot_rows(
            plot_meeting_summary.DEFAULT_ANALYSIS,
            plot_meeting_summary.DEFAULT_DEVELOPMENT,
            plot_meeting_summary.DEFAULT_CONFIRMATION,
        )
        self.assertEqual(
            [row["sequence"] for row in rows],
            list(plot_meeting_summary.SEQUENCES),
        )
        self.assertEqual(
            [row["sequence"] for row in rows if row["soft_gmd_success"]],
            [24, 2],
        )
        self.assertFalse(any(
            row["soft_gmd_success"]
            for row in rows if row["phase"] == "confirmation"
        ))
        self.assertAlmostEqual(
            next(row for row in rows if row["sequence"] == 2)[
                "sensor_y_physical_inband_snr_db"
            ],
            4.574236827,
        )

    def test_soft_list_confirmation_uses_new_frame_outputs(self):
        self.assertEqual(
            run_soft_list_batch.CONFIRMATORY_SEQUENCES, (3, 18, 32, 39)
        )
        self.assertEqual(soft_list_decoder.FROZEN_POOL_SIZE, 12)
        self.assertEqual(soft_list_decoder.FROZEN_ACCEPTANCE_MARGIN, 20.0)
        self.assertTrue(
            set(run_soft_list_batch.CONFIRMATORY_SEQUENCES).isdisjoint(
                run_soft_list_batch.METHOD_DEVELOPMENT_SEQUENCES
            )
        )
        rows = exporter._manifest(exporter.DEFAULT_MANIFEST)
        by_sequence = {int(row["sequence"]): row for row in rows}
        self.assertEqual(
            {int(by_sequence[sequence]["repetition"])
             for sequence in run_soft_list_batch.CONFIRMATORY_SEQUENCES},
            {1, 2, 4, 5},
        )
        self.assertEqual(
            [float(by_sequence[sequence]["duty_percent"])
             for sequence in run_soft_list_batch.CONFIRMATORY_SEQUENCES],
            [45.0, 50.0, 50.0, 50.0],
        )

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
        self.assertEqual(sum(row["sensor_y_bit"] != row["target_bit"] for row in body), 6)
        sensor_y_decode = run_tabfm.rs18.hard_rs18_decode(
            [row["sensor_y_llr"] for row in body]
        )
        self.assertEqual(sensor_y_decode["failure"], 0)
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
