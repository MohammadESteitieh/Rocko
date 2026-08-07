#!/usr/bin/env python3
"""Interactive Matplotlib viewer for the accepted RS18 capture and decoding data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, RangeSlider, Slider
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(HERE),
    str(ROOT / "receiver" / "legacy_decoder"),
]

import analyze_final_experiment as base  # noqa: E402
import analyze_rs18_experiment as rs18  # noqa: E402
import export_rs18_table as exporter  # noqa: E402

DEFAULT_ANALYSIS = (
    ROOT / "data/captures/rs18-experiment/derived"
    / "rs18_experiment_11V_20260728_091828.analysis.csv"
)
DEFAULT_TABFM = (
    ROOT / "data/captures/rs18-experiment/derived/tabfm/meeting-summary-v1"
    / "tabfm-soft-gmd-meeting.plot-data.csv"
)
ACCEPTED_RUN = "rs18_experiment_11V_20260728_091828"
FROZEN_SHA256 = {
    "capture": "ee730cb3ef0998d6ab7399a3a99580380fd3afe8e16c80ba59c9f162294b8fba",
    "metadata": "7ef3c9546d41781cf5a1650423bdd327a8a51aaaaaed38e8698f298710c9b4dc",
    "analysis": "54dbc36e9ab23d3de0537445c2bc4dba3ca5bb8bd1f3e075b59687ead291ef53",
    "tabfm": "d30a8e65df9eae4da6daaa22986bb20e972749bc2b5a9eee8f439e0b973d8ac8",
}
TABFM_SEQUENCES = {2, 3, 10, 18, 24, 28, 32, 39, 45}
DUTY_ORDER = (100, 50, 45, 40, 35, 30, 25, 10, 1)
METHODS = (
    ("Primary", "baseline_symbol_errors", "baseline_success"),
    ("Sensor Y", "sensor_y_symbol_errors", "sensor_y_success"),
    ("TabFM hard", "tabfm_symbol_errors", "tabfm_success"),
    ("Gated hybrid", "sensor_y_tabfm_gated_symbol_errors",
     "sensor_y_tabfm_gated_success"),
)


def verify_frozen_file(path: Path, expected_sha256: str) -> None:
    actual = base.sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{path} is not the frozen accepted-run artifact: "
            f"expected {expected_sha256}, got {actual}"
        )
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        raise ValueError(f"missing SHA-256 sidecar for {path}")
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[0] != actual or Path(fields[1]).name != path.name:
        raise ValueError(f"invalid SHA-256 sidecar for {path}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


class RunViewer:
    """Stateful Matplotlib widget for frame, SNR, waveform, and decoder inspection."""

    def __init__(self, *, capture: Path, metadata: Path, analysis: Path,
                 tabfm: Path, initial_sequence: int = 2):
        for key, path in (
            ("capture", capture), ("metadata", metadata),
            ("analysis", analysis), ("tabfm", tabfm),
        ):
            verify_frozen_file(path, FROZEN_SHA256[key])
        self.analysis_rows = read_rows(analysis)
        self.analysis = {int(row["sequence"]): row for row in self.analysis_rows}
        if set(self.analysis) != set(range(1, 46)):
            raise ValueError("analysis must contain sequences 1 through 45")
        self.tabfm = {int(row["sequence"]): row for row in read_rows(tabfm)}
        if set(self.tabfm) != TABFM_SEQUENCES:
            raise ValueError("TabFM data must contain the exact nine evaluated frames")
        if initial_sequence not in self.analysis:
            raise ValueError("initial sequence is absent from analysis")

        self.t, self.raw_x, self.raw_y = base.load_capture(capture)
        self.fs = base.sample_rate(self.t)
        info = base.metadata(metadata)
        rs18.validate_metadata(info)
        off_start, off_stop = rs18.prelaunch_indices(
            info, self.fs, len(self.t)
        )
        means, scales = rs18.fit_off_rms(
            self.raw_x, self.raw_y, off_start, off_stop
        )
        self.normalized_x = (np.asarray(self.raw_x, float) - means[0]) / scales[0]
        self.normalized_y = (np.asarray(self.raw_y, float) - means[1]) / scales[1]
        self.normalized = True
        self.channel_mode = "Both"
        self.sequence = initial_sequence

        self.figure = plt.figure(figsize=(15, 9))
        self.figure.subplots_adjust(
            left=0.08, right=0.97, top=0.88, bottom=0.29,
            hspace=0.48, wspace=0.30,
        )
        grid = self.figure.add_gridspec(2, 2, height_ratios=(1.15, 1.0))
        self.wave_axis = self.figure.add_subplot(grid[0, :])
        self.snr_axis = self.figure.add_subplot(grid[1, 0])
        self.decode_axis = self.figure.add_subplot(grid[1, 1])

        self.wave_x_line, = self.wave_axis.plot(
            [], [], color="#4C78A8", linewidth=0.8, label="Sensor X"
        )
        self.wave_y_line, = self.wave_axis.plot(
            [], [], color="#E45756", linewidth=0.8, alpha=0.85,
            label="Sensor Y"
        )
        self.wave_axis.legend(loc="upper right", ncol=2)
        self.wave_axis.grid(alpha=0.22)
        self.wave_axis.set_xlabel("Time from frozen frame boundary (s)")

        snr_values = np.asarray([
            float(row["physical_inband_snr_db_y"])
            if row["physical_inband_snr_db_y"] else np.nan
            for row in self.analysis_rows
        ])
        duties = np.asarray([float(row["duty_percent"]) for row in self.analysis_rows])
        repetitions = np.asarray([int(row["repetition"]) for row in self.analysis_rows])
        self.scatter_sequences = np.asarray([
            int(row["sequence"]) for row in self.analysis_rows
        ])
        self.snr_scatter = self.snr_axis.scatter(
            duties, snr_values, c=repetitions, cmap="viridis", vmin=1, vmax=5,
            s=45, alpha=0.8, picker=True,
        )
        self.selected_snr, = self.snr_axis.plot(
            [], [], marker="o", markersize=13, markerfacecolor="none",
            markeredgecolor="black", markeredgewidth=2, linestyle="none",
        )
        self.snr_axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        self.snr_axis.set_xlim(105, -4)
        self.snr_axis.set_xticks(DUTY_ORDER)
        self.snr_axis.set_xlabel("Commanded duty (%)")
        self.snr_axis.set_ylabel("Sensor-Y physical in-band SNR (dB)")
        self.snr_axis.set_title("All 45 frames — color indicates repetition 1–5")
        self.snr_axis.grid(alpha=0.22)

        self.sequence_axis = self.figure.add_axes((0.12, 0.135, 0.62, 0.025))
        self.sequence_slider = Slider(
            self.sequence_axis, "Sequence", 1, 45,
            valinit=initial_sequence, valstep=1,
        )
        self.window_axis = self.figure.add_axes((0.12, 0.085, 0.62, 0.025))
        self.window_slider = RangeSlider(
            self.window_axis, "Frame window (s)", 0.0, 53.0,
            valinit=(0.0, 8.0), valstep=0.1,
        )
        self.channel_axis = self.figure.add_axes((0.79, 0.075, 0.085, 0.10))
        self.channel_buttons = RadioButtons(
            self.channel_axis, ("Both", "Sensor X", "Sensor Y"), active=0
        )
        self.options_axis = self.figure.add_axes((0.89, 0.105, 0.09, 0.055))
        self.options = CheckButtons(self.options_axis, ("Normalize",), (True,))
        self.reset_axis = self.figure.add_axes((0.89, 0.055, 0.085, 0.035))
        self.reset_button = Button(self.reset_axis, "Reset window")

        self.sequence_slider.on_changed(self._sequence_changed)
        self.window_slider.on_changed(lambda _value: self.update_waveform())
        self.channel_buttons.on_clicked(self._channel_changed)
        self.options.on_clicked(self._normalization_changed)
        self.reset_button.on_clicked(self._reset_window)
        self.figure.canvas.mpl_connect("pick_event", self._picked_snr)
        self.figure.canvas.mpl_connect("key_press_event", self._key_pressed)
        self.update_sequence(initial_sequence)

    def _frame_values(self) -> tuple[np.ndarray, np.ndarray]:
        if self.normalized:
            return self.normalized_x, self.normalized_y
        return np.asarray(self.raw_x, float), np.asarray(self.raw_y, float)

    def update_waveform(self) -> None:
        row = self.analysis[self.sequence]
        frame_start = round(float(row["start_offset_s"]) * self.fs)
        low, high = self.window_slider.val
        start = frame_start + round(low * self.fs)
        stop = min(len(self.t), frame_start + round(high * self.fs))
        relative = np.arange(max(0, stop - start), dtype=float) / self.fs + low
        x, y = self._frame_values()
        self.wave_x_line.set_data(relative, x[start:stop])
        self.wave_y_line.set_data(relative, y[start:stop])
        self.wave_x_line.set_visible(self.channel_mode in ("Both", "Sensor X"))
        self.wave_y_line.set_visible(self.channel_mode in ("Both", "Sensor Y"))
        self.wave_axis.set_xlim(low, high)
        self.wave_axis.relim(visible_only=True)
        self.wave_axis.autoscale_view(scalex=False, scaley=True)
        unit = "prelaunch-off RMS units" if self.normalized else "raw ADC counts"
        self.wave_axis.set_ylabel(unit)
        self.figure.canvas.draw_idle()

    def update_sequence(self, sequence: int) -> None:
        self.sequence = int(sequence)
        row = self.analysis[self.sequence]
        snr_text = row["physical_inband_snr_db_y"]
        snr_value = float(snr_text) if snr_text else np.nan
        snr_label = f"{snr_value:.2f} dB" if np.isfinite(snr_value) else "N/A"
        self.selected_snr.set_data(
            [float(row["duty_percent"])], [snr_value]
        )
        self._update_decoder_panel(row)
        status = "decoded" if not int(row["frame_error"]) else "failed"
        self.figure.suptitle(
            f"Accepted RS18 run {ACCEPTED_RUN} — interactive frame viewer\n"
            f'Sequence {self.sequence} | repetition {row["repetition"]} | '
            f'duty {float(row["duty_percent"]):g}% | '
            f'Sensor-Y SNR {snr_label} | primary hard RS {status}',
            fontsize=14,
        )
        self.wave_axis.set_title(
            f'Actual 200 Hz dual-sensor samples — sequence {self.sequence}'
        )
        self.update_waveform()

    def _update_decoder_panel(self, analysis_row: dict[str, str]) -> None:
        self.decode_axis.clear()
        selected = self.tabfm.get(self.sequence)
        errors: list[float] = []
        successes: list[int] = []
        labels: list[str] = []
        for index, (label, error_key, success_key) in enumerate(METHODS):
            labels.append(label)
            if selected is not None:
                errors.append(float(selected[error_key]))
                successes.append(int(selected[success_key]))
            elif index == 0:
                errors.append(float(analysis_row["raw_symbol_errors"]))
                successes.append(1 - int(analysis_row["frame_error"]))
            else:
                errors.append(np.nan)
                successes.append(0)
        positions = np.arange(len(labels))
        for position, error, success in zip(positions, errors, successes):
            if np.isfinite(error):
                self.decode_axis.bar(
                    position, error, color="#59A14F" if success else "#E15759"
                )
                self.decode_axis.text(
                    position, error + 0.35, f"{int(error)}", ha="center", fontsize=9
                )
            else:
                self.decode_axis.text(position, 1.0, "N/A", ha="center", color="gray")
        self.decode_axis.axhline(
            6, color="black", linestyle="--", linewidth=1,
            label="Hard RS correction limit",
        )
        self.decode_axis.set_xticks(positions, labels, rotation=0, ha="center")
        self.decode_axis.tick_params(axis="x", labelsize=9)
        self.decode_axis.set_ylabel("Pre-RS body symbol errors")
        self.decode_axis.set_ylim(0, 20)
        self.decode_axis.grid(axis="y", alpha=0.22)
        self.decode_axis.legend(loc="upper right", fontsize=8)
        if selected is None:
            soft_text = "TabFM soft GMD/list: not evaluated on this frame"
        else:
            phase = selected["phase"]
            soft_text = f'TabFM soft GMD/list: {selected["soft_gmd_label"]} ({phase})'
        self.decode_axis.set_title(soft_text)

    def _sequence_changed(self, value: float) -> None:
        self.update_sequence(round(value))

    def _channel_changed(self, label: str) -> None:
        self.channel_mode = label
        self.update_waveform()

    def _normalization_changed(self, _label: str) -> None:
        self.normalized = not self.normalized
        self.update_waveform()

    def _reset_window(self, _event) -> None:
        self.window_slider.set_val((0.0, 8.0))

    def _picked_snr(self, event) -> None:
        if event.artist is self.snr_scatter and len(event.ind):
            sequence = int(self.scatter_sequences[int(event.ind[0])])
            self.sequence_slider.set_val(sequence)

    def _key_pressed(self, event) -> None:
        if event.key == "left":
            self.sequence_slider.set_val(max(1, self.sequence - 1))
        elif event.key == "right":
            self.sequence_slider.set_val(min(45, self.sequence + 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=exporter.DEFAULT_CAPTURE)
    parser.add_argument("--metadata", type=Path, default=exporter.DEFAULT_METADATA)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--tabfm", type=Path, default=DEFAULT_TABFM)
    parser.add_argument("--sequence", type=int, default=2)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    viewer = RunViewer(
        capture=args.capture,
        metadata=args.metadata,
        analysis=args.analysis,
        tabfm=args.tabfm,
        initial_sequence=args.sequence,
    )
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        viewer.figure.savefig(args.snapshot, dpi=160, bbox_inches="tight")
        print(f"Wrote {args.snapshot}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(viewer.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
