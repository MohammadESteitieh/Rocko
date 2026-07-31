#!/usr/bin/env python3
"""Rocko live receiver dashboard.

Three stacked panes plus a compact event panel:

    top     sensor 1 + sensor 2 raw ADC
    middle  sensor 1 + sensor 2 bandpass around the 8 Hz carrier
    bottom  combined carrier amplitude + adaptive tone threshold
    right   Rocko status header + recent numbered events

Samples arrive from a serial port (or a CSV replay) via :mod:`serial_source`.
A causal tone detector arms on a confirmed carrier and, after enough continuous
silence (long enough to span the 3 s inter-repeat gaps but not the 120 s
heartbeat), decodes the recent window in-process with :mod:`decoder` and marks
the decode point on the amplitude pane.

The panes start EMPTY — nothing is drawn until real samples arrive.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import queue
import textwrap
import threading

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy import signal

import coded_protocol as protocol
import layered_decoder
from eventlog import EventLog

BANDWIDTH_HZ = protocol.BANDWIDTH_HZ
CARRIER_HZ = protocol.CARRIER_HZ
DEFAULT_SAMPLE_RATE_HZ = protocol.DEFAULT_SAMPLE_RATE_HZ
FRAME_BITS = protocol.CODED_BITS
HALF_SYMBOL_SECONDS = protocol.HALF_SYMBOL_SECONDS
PREAMBLE_BITS = tuple(map(int, protocol.ENCODED_HEADER))
REPEAT_GAP_SECONDS = protocol.INTERFRAME_GAP_SECONDS
from serial_source import ReplaySource, SerialSource

# Visual identity — muted, readable, works on a projector.
COLOR_RAW = "#2563eb"
COLOR_RAW_2 = "#7c3aed"
COLOR_BAND = "#0891b2"
COLOR_BAND_2 = "#9333ea"
COLOR_AMP = "#059669"
COLOR_THRESH = "#dc2626"
COLOR_MARK = "#f59e0b"
COLOR_PANEL_BG = "#0f172a"
COLOR_PANEL_FG = "#e2e8f0"

# Reject decodes whose best tilde-preamble correlation (summed over the two ADC
# channels, so it maxes at 2.0) falls below this absolute floor. Real captures —
# clean or heavily noised — score ~1.7; pure 8 Hz-band noise tops out near 0.3.
# Below the floor the envelope tripped on interference, not a beacon: surface it
# as WARN, never as a confident emergency shown to rescuers.
MIN_PREAMBLE_CONFIDENCE = 0.8


class LiveReceiver:
    def __init__(self, args, event_log: EventLog | None = None):
        self.args = args
        self.sample_queue: "queue.Queue" = queue.Queue()
        self.stop_event = threading.Event()

        if getattr(args, "replay", None):
            self.source = ReplaySource(
                Path(args.replay), self.sample_queue, self.stop_event,
                speed=getattr(args, "speed", 1.0),
            )
            source_label = f"replay {Path(args.replay).name}"
        else:
            self.source = SerialSource(
                args.port, args.baud, self.sample_queue, self.stop_event
            )
            source_label = args.port
        self.source_label = source_label

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = args.output or (Path("captures") / f"live_{stamp}.csv")
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.csv = self.output.open("w", buffering=1)
        self.csv.write("t,x,y\n")

        self.log = event_log or EventLog(Path("captures") / f"rocko_{stamp}.log")

        self.max_plot_samples = round(args.plot_seconds * args.sample_rate)
        self.t = deque(maxlen=self.max_plot_samples)
        self.x = deque(maxlen=self.max_plot_samples)
        self.y = deque(maxlen=self.max_plot_samples)
        self.fx = deque(maxlen=self.max_plot_samples)
        self.fy = deque(maxlen=self.max_plot_samples)
        self.envelope = deque(maxlen=self.max_plot_samples)
        self.detector_history = deque(maxlen=round(60 * args.sample_rate))

        low = args.carrier - args.bandwidth / 2
        high = args.carrier + args.bandwidth / 2
        self.band_label = f"{low:g}-{high:g} Hz bandpass"
        self.sos = signal.butter(
            4, [low, high], btype="bandpass", fs=args.sample_rate, output="sos"
        )
        self.zi_x = None
        self.zi_y = None
        self.envelope_state = 0.0
        self.envelope_alpha = 1 - np.exp(-1 / (args.sample_rate * 0.25))

        self.threshold = 0.0
        self.noise_floor = 0.0
        self.signal_high = 0.0
        self.separation = 0.0
        self.current_amplitude = 0.0
        self.tone_now = False
        self.current_silence = 0.0
        self.decoder_trace = deque(maxlen=100)
        self.log_scroll_offset = 0
        self.log_view_rows = 8
        self.next_live_decode_time = None
        self.live_frame_start_time = None
        self.live_preamble_score = 0.0
        self.live_flag_bits: list[int] = []
        self.live_frame_reported = False
        self.live_decoder_state = "SEARCHING FOR CODED TILDE"
        self.tone_run = 0.0
        self.seen_tone = False
        self.signal_logged = False
        self.last_tone_time = None
        self.decode_pending = False
        self.last_decode_time = None  # boundary: decode only samples newer than this
        self.total_samples = 0        # unlike the rolling plot deque, never resets
        self._closed = False          # close() must be idempotent (see close())
        self.status_line = "Waiting for the first signal…"
        self._marker_artists: list = []

        self._build_figure()
        self.animation = None

    # --- figure ------------------------------------------------------------

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 9.5), facecolor="#f8fafc")
        try:
            self.fig.canvas.manager.set_window_title("Rocko — cave beacon receiver")
        except Exception:  # headless / Agg backend has no window manager
            pass
        # The event panel is operationally as important as the traces: give it
        # over a third of the window instead of squeezing logs into a sidebar.
        grid = GridSpec(
            3, 2, width_ratios=[2.35, 1.35], height_ratios=[1, 1, 1.15],
            hspace=0.28, wspace=0.055, left=0.055, right=0.985,
            top=0.9, bottom=0.08,
        )
        self.ax_raw = self.fig.add_subplot(grid[0, 0])
        self.ax_band = self.fig.add_subplot(grid[1, 0], sharex=self.ax_raw)
        self.ax_amp = self.fig.add_subplot(grid[2, 0], sharex=self.ax_raw)
        self.ax_side = self.fig.add_subplot(grid[:, 1])
        self.ax_side.set_facecolor(COLOR_PANEL_BG)
        self.ax_side.set_xticks([])
        self.ax_side.set_yticks([])
        for spine in self.ax_side.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.2)

        (self.raw_line,) = self.ax_raw.plot(
            [], [], lw=0.9, color=COLOR_RAW, label="Sensor 1"
        )
        (self.raw_y_line,) = self.ax_raw.plot(
            [], [], lw=0.8, color=COLOR_RAW_2, alpha=0.72, label="Sensor 2"
        )
        (self.band_line,) = self.ax_band.plot(
            [], [], lw=0.9, color=COLOR_BAND, label="Sensor 1"
        )
        (self.band_y_line,) = self.ax_band.plot(
            [], [], lw=0.8, color=COLOR_BAND_2, alpha=0.72, label="Sensor 2"
        )
        (self.env_line,) = self.ax_amp.plot(
            [], [], lw=1.3, color=COLOR_AMP, label="carrier amplitude"
        )
        (self.threshold_line,) = self.ax_amp.plot(
            [], [], "--", lw=1.2, color=COLOR_THRESH, label="tone threshold"
        )

        self.ax_raw.set_ylabel("Raw ADC")
        self.ax_raw.set_title("Sensors — raw", loc="left", fontsize=10, color="#334155")
        self.ax_raw.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.85)
        self.ax_band.set_ylabel("Bandpass")
        self.ax_band.set_title(
            f"Sensors — {self.band_label}", loc="left", fontsize=10, color="#334155"
        )
        self.ax_band.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.85)
        self.ax_amp.set_ylabel("Carrier amplitude")
        self.ax_amp.set_xlabel("Receiver time (s)")
        self.ax_amp.set_title(
            "Carrier amplitude + adaptive tone threshold",
            loc="left", fontsize=10, color="#334155",
        )
        self.ax_amp.legend(loc="upper right", fontsize=8, framealpha=0.85)
        for axis in (self.ax_raw, self.ax_band, self.ax_amp):
            axis.grid(alpha=0.22)
            axis.margins(x=0)

        self.fig.suptitle(
            "ROCKO  ·  cave explorer safety beacon — surface receiver",
            fontsize=15, fontweight="bold", x=0.06, ha="left",
        )
        self.status_text = self.ax_side.text(
            0.045, 0.965, "WAITING", transform=self.ax_side.transAxes,
            va="top", ha="left", fontsize=11, fontweight="bold", color="#ffffff",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#475569",
                  "edgecolor": "none"},
        )
        self.side_text = self.ax_side.text(
            0.045, 0.90, "", transform=self.ax_side.transAxes, va="top", ha="left",
            family="monospace", fontsize=9.1, linespacing=1.32,
            color=COLOR_PANEL_FG,
        )
        self.ax_side.text(
            0.045, 0.025, "Wheel: scroll logs  •  Q / Esc: close  •  toolbar: zoom / pan",
            transform=self.ax_side.transAxes, va="bottom", ha="left",
            fontsize=8, color="#94a3b8",
        )
        self.fig.canvas.mpl_connect("close_event", lambda _evt: self.close())
        self.fig.canvas.mpl_connect(
            "key_press_event",
            lambda event: plt.close(self.fig) if event.key in ("q", "escape") else None,
        )
        self.fig.canvas.mpl_connect("scroll_event", self._scroll_logs)

    # --- sample intake -----------------------------------------------------

    def _drain_samples(self):
        rows = []
        while True:
            try:
                rows.append(self.sample_queue.get_nowait())
            except queue.Empty:
                break
        if not rows:
            return

        values = np.asarray(rows, dtype=float)
        times, raw_x, raw_y = values.T
        for timestamp, xv, yv in rows:
            self.csv.write(f"{timestamp:.6f},{xv:g},{yv:g}\n")

        if self.zi_x is None:
            # Seed each bandpass at the ADC's initial DC level so the filter
            # startup transient does not masquerade as a tone.
            self.zi_x = signal.sosfilt_zi(self.sos) * raw_x[0]
            self.zi_y = signal.sosfilt_zi(self.sos) * raw_y[0]
        filtered_x, self.zi_x = signal.sosfilt(self.sos, raw_x, zi=self.zi_x)
        filtered_y, self.zi_y = signal.sosfilt(self.sos, raw_y, zi=self.zi_y)
        instantaneous = np.sqrt(filtered_x ** 2 + filtered_y ** 2)
        smoothed = np.empty_like(instantaneous)
        for index, amplitude in enumerate(instantaneous):
            self.envelope_state += self.envelope_alpha * (amplitude - self.envelope_state)
            smoothed[index] = self.envelope_state

        self.total_samples += len(rows)
        self.t.extend(times)
        self.x.extend(raw_x)
        self.y.extend(raw_y)
        self.fx.extend(filtered_x)
        self.fy.extend(filtered_y)
        self.envelope.extend(smoothed)
        self.detector_history.extend(smoothed)
        self._update_detector(times, smoothed)
        if not self._closed and not hasattr(self, "_close_timer"):
            self._update_live_decoder(float(times[-1]))

    # --- tone / end-of-message detector -----------------------------------

    def _update_detector(self, times, amplitudes):
        history = np.asarray(self.detector_history)
        if len(history) < self.args.sample_rate:
            return
        floor, high = np.percentile(history, [10, 90])
        self.noise_floor = float(floor)
        self.signal_high = float(high)
        self.separation = high - floor
        self.threshold = floor + 0.30 * self.separation
        self.current_amplitude = float(amplitudes[-1])
        detector_ready = high > max(floor * 3.0, floor + self.args.min_separation)

        for timestamp, amplitude in zip(times, amplitudes):
            tone = detector_ready and amplitude > self.threshold
            self.tone_now = bool(tone)
            if tone:
                self.tone_run += 1 / self.args.sample_rate
                if self.tone_run >= self.args.tone_confirm:
                    if not self.seen_tone and not self.signal_logged:
                        self.log.emit("SIGNAL", "carrier detected — recording beacon")
                        self.signal_logged = True
                    self.seen_tone = True
                if self.seen_tone:
                    self.last_tone_time = timestamp
            else:
                self.tone_run = 0.0

        if self.last_tone_time is not None:
            self.current_silence = max(0.0, float(times[-1] - self.last_tone_time))
        else:
            self.current_silence = 0.0
        if self.seen_tone and self.last_tone_time is not None:
            silence = self.current_silence
            self.status_line = (
                f"Tone captured — silence {silence:.1f}/{self.args.silence:g}s"
            )
            if silence >= self.args.silence and not self.decode_pending:
                # A startup transient or short interference burst can satisfy the
                # silence timer before a complete 12-bit frame exists. Defer the
                # decoder rather than displaying that expected condition as an
                # analyzer ERROR.
                boundary = self.last_decode_time
                if boundary is None and self.t:
                    boundary = float(self.t[0])
                available = float(times[-1] - boundary) if boundary is not None else 0.0
                frame_seconds = FRAME_BITS * protocol.BIT_SECONDS
                if available < frame_seconds:
                    self.status_line = (
                        f"Signal ended — buffering full frame "
                        f"{available:.1f}/{frame_seconds:.1f}s"
                    )
                    return
                confidence_floor = getattr(
                    self.args, "min_confidence", MIN_PREAMBLE_CONFIDENCE
                )
                if (self.live_frame_start_time is None
                        or self.live_preamble_score < confidence_floor):
                    self.log.emit(
                        "WARN", "signal ended without a valid tilde preamble — ignored"
                    )
                    self.status_line = "Signal ignored — no valid preamble"
                    self._reset_after_decode()
                    return
                frame_age = float(times[-1] - self.live_frame_start_time)
                if frame_age < frame_seconds:
                    self.status_line = (
                        f"Preamble locked — receiving frame "
                        f"{frame_age:.1f}/{frame_seconds:.1f}s"
                    )
                    return
                self.decode_pending = True
                self.csv.flush()
                self._decode()

    def _decoder_log(self, kind: str, message: str):
        line = f"{kind:<5} {message}"
        self.decoder_trace.append(line)
        self.log.emit(kind, message)

    def _update_live_decoder(self, now: float):
        """Lock the coded tilde and expose 28-bit frame progress in real time."""
        if self.next_live_decode_time is not None and now < self.next_live_decode_time:
            return
        self.next_live_decode_time = now + 0.75
        t = np.asarray(self.t)
        header_seconds = len(protocol.ENCODED_HEADER) * protocol.BIT_SECONDS
        if len(t) < round(header_seconds * self.args.sample_rate):
            self.live_decoder_state = (
                f"BUFFERING CODED HEADER {len(t)/self.args.sample_rate:.1f}/"
                f"{header_seconds:.1f}s"
            )
            return
        try:
            fs = layered_decoder.sample_rate(t)
            channels = layered_decoder.analytic_channels(
                np.asarray(self.x), np.asarray(self.y), fs
            )
            half = round(fs * protocol.HALF_SYMBOL_SECONDS)
            template = protocol.complex_template(protocol.ENCODED_HEADER, half, fs)
            correlation = sum(
                layered_decoder.sliding_correlation(channel, template)
                for channel in channels
            )
        except Exception as exc:
            self.live_decoder_state = f"DSP WAIT: {exc}"
            return

        best = float(np.max(correlation))
        floor = getattr(self.args, "min_confidence", MIN_PREAMBLE_CONFIDENCE)
        peaks, _ = signal.find_peaks(
            correlation, height=floor,
            distance=max(
                1, round(fs * len(protocol.ENCODED_HEADER) * protocol.BIT_SECONDS * 0.75)
            ),
        )
        if not len(peaks):
            self.live_decoder_state = f"SEARCHING CODED ~ — best {best:.2f}/{floor:.2f}"
            return

        candidate = int(peaks[-1])
        candidate_time = float(t[candidate])
        new_frame = (
            self.live_frame_start_time is None
            or candidate_time > (
                self.live_frame_start_time
                + protocol.CODED_BITS * protocol.BIT_SECONDS * 0.75
            )
            or self.live_frame_start_time < float(t[0])
        )
        if new_frame:
            self.live_frame_start_time = candidate_time
            self.live_preamble_score = float(correlation[candidate])
            self.live_flag_bits = []
            self.live_frame_reported = False
            self._decoder_log(
                "SYNC", f"coded tilde locked t={candidate_time:.2f}s "
                f"score={self.live_preamble_score:.2f}/2.00",
            )

        age = max(0.0, now - self.live_frame_start_time)
        received = min(protocol.CODED_BITS, int(age / protocol.BIT_SECONDS))
        self.live_decoder_state = (
            f"HEADER ~ LOCKED {self.live_preamble_score:.2f}/2.00  "
            f"CODED {received:02d}/{protocol.CODED_BITS}  LETTER pending"
        )
        if received < protocol.CODED_BITS or self.live_frame_reported:
            return

        try:
            keep = t >= self.live_frame_start_time - 0.1
            result = layered_decoder.decode_capture(
                t[keep], np.asarray(self.x)[keep], np.asarray(self.y)[keep]
            )
        except Exception as exc:
            self.live_decoder_state = f"LAYER DECODE WAIT: {exc}"
            return
        self.live_frame_reported = True
        for layer in result.layers:
            marker = "OK" if layer.success else "--"
            self._decoder_log(
                "LAYER", f"{layer.layer:<9} {marker} header=0x{layer.header:02X} "
                f"letter={layer.letter} parity={'ok' if layer.parity_ok else 'bad'} "
                f"confidence={layer.confidence:.2f}",
            )
        chosen = result.selected
        self.live_decoder_state = (
            f"HEADER ~  LETTER {chosen.letter}  LAYER {chosen.layer}  "
            f"successful={','.join(result.successful_layers) or 'none'}"
        )

    def _decode(self):
        times, xs, ys = np.asarray(self.t), np.asarray(self.x), np.asarray(self.y)
        if self.last_decode_time is not None:
            fresh = times > self.last_decode_time
            times, xs, ys = times[fresh], xs[fresh], ys[fresh]
        try:
            result = layered_decoder.decode_capture(times, xs, ys)
        except Exception as exc:
            self.log.emit("WARN", f"layered decode rejected signal: {exc}")
            self.status_line = f"Signal rejected: {exc}"
            self._reset_after_decode()
            return
        floor = getattr(self.args, "min_confidence", MIN_PREAMBLE_CONFIDENCE)
        if result.preamble_score < floor or not result.successful_layers:
            self.log.emit(
                "WARN", f"coded frame rejected — preamble={result.preamble_score:.2f} "
                f"successful layers={result.successful_layers or 'none'}",
            )
            self.status_line = "Coded frame rejected"
            self._reset_after_decode()
            return
        if not self.live_frame_reported:
            for layer in result.layers:
                marker = "OK" if layer.success else "--"
                self.log.emit(
                    "LAYER", f"{layer.layer} {marker} header=0x{layer.header:02X} "
                    f"letter={layer.letter} parity={'ok' if layer.parity_ok else 'bad'}",
                )
        chosen = result.selected
        self.log.emit(
            "DECODE", f"header=~ letter={chosen.letter} layer={chosen.layer} "
            f"successful={','.join(result.successful_layers)}",
        )
        self.status_line = (
            f"Decoded: header=~ letter={chosen.letter} layer={chosen.layer}"
        )
        self._add_decode_marker(result)
        self._reset_after_decode()
        if self.args.stop_after_decode:
            # Closing Matplotlib inside its animation callback invalidates the
            # callback's timer. Defer shutdown to a separate one-shot timer.
            self._close_timer = self.fig.canvas.new_timer(interval=1)
            self._close_timer.single_shot = True
            self._close_timer.add_callback(self._close_window)
            self._close_timer.start()

    def _close_window(self):
        self.close()
        plt.close(self.fig)

    def _reset_after_decode(self):
        self.seen_tone = False
        self.signal_logged = False
        self.last_tone_time = None
        self.decode_pending = False
        self.live_frame_start_time = None
        self.live_preamble_score = 0.0
        self.live_flag_bits = []
        self.live_frame_reported = False
        self.live_decoder_state = "SEARCHING FOR CODED TILDE"
        if self.t:
            self.last_decode_time = float(self.t[-1])

    def _add_decode_marker(self, result):
        top = max(self.envelope) if self.envelope else 1.0
        when = result.start_time
        chosen = result.selected
        line = self.ax_amp.axvline(
            when, color=COLOR_MARK, lw=1.4, alpha=0.8, zorder=4
        )
        star = self.ax_amp.scatter(
            [when], [top], marker="*", s=320, color=COLOR_MARK,
            edgecolor="#7c2d12", linewidth=0.8, zorder=6,
        )
        label = self.ax_amp.annotate(
            f"~ {chosen.letter}\n{chosen.layer}", xy=(when, top), xytext=(4, -6),
            textcoords="offset points", fontsize=8, fontweight="bold",
            color="#7c2d12", zorder=6,
        )
        self._marker_artists.append((when, [line, star, label]))

    def _prune_markers(self, left_edge):
        keep = []
        for start_time, artists in self._marker_artists:
            if start_time < left_edge:
                for artist in artists:
                    artist.remove()
            else:
                keep.append((start_time, artists))
        self._marker_artists = keep

    # --- rendering ---------------------------------------------------------

    @staticmethod
    def _set_limits(axis, x_values, y_values):
        if len(x_values) < 2:
            return
        axis.set_xlim(x_values[0], x_values[-1])
        low, high = np.percentile(y_values, [1, 99])
        margin = max((high - low) * 0.12, 1.0)
        axis.set_ylim(low - margin, high + margin)

    _PANEL_WIDTH = 49

    @classmethod
    def _wrap_panel_line(cls, text: str, subsequent_indent: str = "  ") -> list[str]:
        return textwrap.wrap(
            text, width=cls._PANEL_WIDTH, subsequent_indent=subsequent_indent,
            break_long_words=False, break_on_hyphens=False,
        ) or [""]

    def _scroll_logs(self, event):
        if event.inaxes is not self.ax_side:
            return
        events = self.log.recent()
        maximum = max(0, len(events) - self.log_view_rows)
        if event.button == "up":
            self.log_scroll_offset = min(maximum, self.log_scroll_offset + 3)
        elif event.button == "down":
            self.log_scroll_offset = max(0, self.log_scroll_offset - 3)
        self._render_side_panel()
        self.fig.canvas.draw_idle()

    def _status_style(self):
        status = self.status_line.lower()
        if "error" in status or "failed" in status:
            return "ERROR", "#b91c1c"
        if "low-confidence" in status or "ignored" in status:
            return "CHECK SIGNAL", "#b45309"
        if status.startswith("decoded"):
            return "DECODED", "#047857"
        if "tone" in status or "silence" in status:
            return "RECEIVING", "#0369a1"
        return "READY", "#475569"

    def _render_side_panel(self):
        log_name = self.log.path.name if self.log.path else "-"
        receiver_time = self.t[-1] if self.t else 0.0
        buffer_seconds = (self.t[-1] - self.t[0]) if len(self.t) >= 2 else 0.0
        header = [
            "ROCKO RECEIVER",
            "─" * self._PANEL_WIDTH,
            f"Source       {self.source_label}",
            f"Sample rate  {self.args.sample_rate:g} Hz",
            f"Samples      {self.total_samples:,}",
            f"Receiver t   {receiver_time:.1f} s",
            f"Plot buffer  {buffer_seconds:.1f} s",
            f"Amplitude    {self.current_amplitude:.2f}",
            f"Threshold    {self.threshold:.2f}",
            f"Noise floor  {self.noise_floor:.2f}",
            f"Signal high  {self.signal_high:.2f}",
            f"Separation   {self.separation:.2f}",
            f"Tone state   {'ON' if self.tone_now else 'quiet'}",
            f"Silence      {self.current_silence:.1f} s",
            f"Log file     {log_name}",
            "",
            self.status_line,
            "",
            "",
            "LIVE DECODER",
            "─" * self._PANEL_WIDTH,
            self.live_decoder_state,
            "",
            "DECODER TRACE",
            "─" * self._PANEL_WIDTH,
        ]
        lines: list[str] = []
        for line in header:
            lines.extend(self._wrap_panel_line(line))
        if self.decoder_trace:
            for entry in list(self.decoder_trace)[-4:]:
                lines.extend(self._wrap_panel_line(entry, "      "))
        else:
            lines.append("Waiting for preamble correlation…")
        all_events = self.log.recent()
        end = max(0, len(all_events) - self.log_scroll_offset)
        start = max(0, end - self.log_view_rows)
        position = "newest" if self.log_scroll_offset == 0 else f"-{self.log_scroll_offset}"
        lines.extend([
            "", f"EVENT LOG ({position}; mouse wheel to scroll)",
            "─" * self._PANEL_WIDTH,
        ])
        events = all_events[start:end]
        if events:
            for event in events:
                lines.extend(self._wrap_panel_line(event.compact(), "       "))
        else:
            lines.extend(self._wrap_panel_line("No events yet — waiting for a real signal."))
        self.side_text.set_text("\n".join(lines))
        badge, color = self._status_style()
        self.status_text.set_text(badge)
        self.status_text.get_bbox_patch().set_facecolor(color)

    def update_plot(self, _frame):
        if self._closed:
            return ()
        self._drain_samples()
        source_error = getattr(self.source, "error", None)
        if source_error is not None:
            message = f"source error: {source_error}"
            if self.status_line != message:
                self.log.emit("ERROR", message)
            self.status_line = message

        if len(self.t) >= 2:
            t = np.asarray(self.t)
            self.raw_line.set_data(t, np.asarray(self.x))
            self.raw_y_line.set_data(t, np.asarray(self.y))
            self.band_line.set_data(t, np.asarray(self.fx))
            self.band_y_line.set_data(t, np.asarray(self.fy))
            self.env_line.set_data(t, np.asarray(self.envelope))
            self.threshold_line.set_data(t, np.full_like(t, self.threshold))
            self._set_limits(self.ax_raw, t, np.r_[np.asarray(self.x), np.asarray(self.y)])
            self._set_limits(self.ax_band, t, np.r_[np.asarray(self.fx), np.asarray(self.fy)])
            self._set_limits(
                self.ax_amp, t, np.r_[np.asarray(self.envelope), self.threshold]
            )
            self._prune_markers(t[0])

        self._render_side_panel()
        return ()

    # --- lifecycle ---------------------------------------------------------

    def run(self):
        self.log.emit(
            "CAPTURE",
            f"listening on {self.source_label} -> {self.output}",
        )
        self.log.emit(
            "READY",
            f"layered Hamming(7,4) decode after {self.args.silence:g}s silence "
            f"(28 bits at {1 / protocol.BIT_SECONDS:g} bit/s, 8 Hz bandpass, "
            f"{REPEAT_GAP_SECONDS:g}s gap)",
        )
        self.source.start()
        from matplotlib.animation import FuncAnimation

        self.animation = FuncAnimation(
            self.fig, self.update_plot, interval=100, blit=False,
            cache_frame_data=False,
        )
        plt.show()

    def close(self):
        # Idempotent: close() fires from the window close_event, the
        # --stop-after-decode path, and the main() finally — often twice in one
        # shutdown. Re-running it would emit to an already-closed log handle.
        if self._closed:
            return
        self._closed = True
        if not self.stop_event.is_set():
            self.source.close()
        if not self.csv.closed:
            self.csv.flush()
            self.csv.close()
        self.log.emit("CAPTURE", f"stopped — saved {self.output}", echo=True)
        self.log.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-p", "--port", help="serial port, e.g. /dev/cu.usbmodem1201")
    source.add_argument("--replay", help="re-stream a recorded t,x,y CSV (no hardware)")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    parser.add_argument("-b", "--baud", type=int, default=115200)
    parser.add_argument("-o", "--output")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--carrier", type=float, default=CARRIER_HZ)
    parser.add_argument("--bandwidth", type=float, default=BANDWIDTH_HZ)
    parser.add_argument("--silence", type=float, default=5.0)
    parser.add_argument("--tone-confirm", type=float, default=0.3)
    parser.add_argument("--min-separation", type=float, default=2.0)
    parser.add_argument("--min-confidence", type=float, default=MIN_PREAMBLE_CONFIDENCE,
                        help="reject decodes below this preamble score (0-2 scale)")
    parser.add_argument("--plot-seconds", type=float, default=90.0)
    parser.add_argument("--stop-after-decode", action="store_true")
    return parser.parse_args()


def main():
    app = LiveReceiver(parse_args())
    try:
        app.run()
    finally:
        app.close()


if __name__ == "__main__":
    main()
