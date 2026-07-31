#!/usr/bin/env python3
"""Cave Beacon coil transmitter daemon (QNX 8 / Raspberry Pi 5).

Drives an L298N H-bridge coil through the QNX rpi_gpio resource manager.

Physical layer
    "tone"    = 8 Hz square wave made by flipping coil polarity on IN3/IN4
                (62.5 ms per half-cycle), ENB high.
    "no tone" = ENB low (bridge gated off).

Encoding
    Regular Manchester per data bit: 1 -> tone/no-tone, 0 -> no-tone/tone.
    Bit time 1.0 s (0.5 s half-symbols -> 4 carrier cycles per tone half).

Frame (12 bits, ~12 s)
    preamble 01111110 (tilde), then 4 flag bits MSB-first:
        bit3=fire  bit2=trapped  bit1=lost  bit0=injured
    0000 = heartbeat ("alive, no emergency"), 1111 = SOS (the classifier's
    "help" keyword override), combinations legal (0101 = trapped+injured).

Behavior
    Long-running beacon. Transmits NOTHING at launch: the first heartbeat
    fires one full period (120 s) after start; emergencies may transmit any
    time. Emergency triggers are read from a spool file (class names the
    listener writes); a frame that is mid-transmission is always finished
    first, then the pending flags (OR-merged, duplicates debounced) go out
    3x with 3 s gaps and the heartbeat schedule resets. Spool tokens
    stop/cancel/clear/ok finish the current frame, abort remaining repeats,
    and clear the queue. Stale spool + stale pidfile are cleared at startup.
    Also a one-shot CLI mode for bench tests: transmitter.py --send injured

Wiring (BCM): GPIO22 -> IN3, GPIO17 -> IN4, GPIO27 -> ENB, coil on OUT3/OUT4.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Sequence

LOG = logging.getLogger("beacon")
LOG.addHandler(logging.NullHandler())

# --- frame contract (docs/equipment-codes.md; change only by telling everyone)
PREAMBLE_BITS = "01111110"
FLAG_FIELD_WIDTH = 4
FLAG_FIRE = 0b1000
FLAG_TRAPPED = 0b0100
FLAG_LOST = 0b0010
FLAG_INJURED = 0b0001
HEARTBEAT_FLAGS = 0b0000
SOS_FLAGS = 0b1111

# F19: accept ONLY the tokens the listener actually writes to the spool. The
# single-bit emergency classes, plus "sos" and the stop synonyms. No raw 4-bit
# strings, no "heartbeat", no "help" alias - no writer produces those, and dead
# grammar in the transmit path is a defect on a cannot-fail device.
CLASS_FLAGS = {
    "fire": FLAG_FIRE,
    "trapped": FLAG_TRAPPED,
    "lost": FLAG_LOST,
    "injured": FLAG_INJURED,
}
SOS_ALIASES = ("sos",)
NO_OP_CLASSES = ("none",)  # classifier's "none" never triggers a transmission
STOP_TOKENS = ("stop", "cancel", "clear", "ok")  # voice off-switch

LOG_MAX_BYTES = 128 * 1024
LOG_BACKUP_COUNT = 2
SPOOL_MAX_BYTES = 64 * 1024  # cap on a single spool read
SPOOL_WORK_SUFFIX = ".consuming"  # atomic consume: rename first, then read
MAX_FRAME_HISTORY = 64  # F18: bound in-memory frame log (recent frames only)


class BeaconError(RuntimeError):
    """Fatal beacon condition with a message fit for the operator."""


@dataclass(frozen=True)
class Config:
    """All tunables in one place - no magic numbers below this line."""

    in3_gpio: int = 22  # BCM 22 -> L298N IN3 (polarity A)
    in4_gpio: int = 17  # BCM 17 -> L298N IN4 (polarity B)
    enb_gpio: int = 27  # BCM 27 -> L298N ENB (bridge on/off)
    carrier_hz: float = 8.0  # polarity-flip square wave ("the tone")
    bit_seconds: float = 1.0  # one Manchester data bit
    heartbeat_interval_s: float = 120.0
    emergency_repeats: int = 3
    emergency_gap_s: float = 3.0
    poll_interval_s: float = 0.5  # spool check cadence while idle
    spool_settle_s: float = 0.1  # F16: pause after rename so a racing writer finishes
    spool_path: str = "/tmp/beacon_trigger"
    pidfile_path: str = "/tmp/beacon.pid"
    log_path: str = "/tmp/beacon.log"
    gpio_dev: str = "/dev/gpio"

    @property
    def half_symbol_seconds(self) -> float:
        return self.bit_seconds / 2.0

    def validate(self) -> None:
        if len({self.in3_gpio, self.in4_gpio, self.enb_gpio}) != 3:
            raise ValueError("IN3, IN4, and ENB must use different GPIO pins")
        if self.carrier_hz <= 0 or self.bit_seconds <= 0:
            raise ValueError("carrier_hz and bit_seconds must be positive")
        if self.heartbeat_interval_s <= 0 or self.poll_interval_s <= 0:
            raise ValueError("heartbeat and poll intervals must be positive")
        if self.spool_settle_s < 0:
            raise ValueError("spool_settle_s must not be negative")
        if self.emergency_repeats < 1 or self.emergency_gap_s < 0:
            raise ValueError("emergency repeats/gap out of range")
        cycles_per_half = self.carrier_hz * self.half_symbol_seconds
        if cycles_per_half != int(cycles_per_half):
            raise ValueError(
                "carrier_hz * bit_seconds/2 must be a whole number of cycles"
            )


# --- encoding -----------------------------------------------------------


def regular_manchester(bits: str) -> list[int]:
    """Encode bits into half-symbols: 0 -> OFF/ON and 1 -> ON/OFF."""
    if not bits or any(bit not in "01" for bit in bits):
        raise ValueError("message must be a non-empty binary string")
    symbols: list[int] = []
    for bit in bits:
        symbols.extend((0, 1) if bit == "0" else (1, 0))
    return symbols


def flags_to_field(flags: int) -> str:
    """4 flag bits, MSB-first: fire, trapped, lost, injured."""
    if not 0 <= flags <= (1 << FLAG_FIELD_WIDTH) - 1:
        raise ValueError(f"flags out of range: {flags}")
    return format(flags, f"0{FLAG_FIELD_WIDTH}b")


def build_frame(flags: int) -> str:
    """Full 12-bit frame: tilde preamble + flag field."""
    return PREAMBLE_BITS + flags_to_field(flags)


def flags_label(flags: int) -> str:
    if flags == HEARTBEAT_FLAGS:
        return "heartbeat"
    if flags == SOS_FLAGS:
        return "sos"
    names = [name for name, bit in CLASS_FLAGS.items() if bit and flags & bit]
    return "+".join(names) if names else f"flags:{flags_to_field(flags)}"


def coded_label(flags: int) -> str:
    """Human label + its 4-bit code, e.g. 'injured (0001)', 'SOS (1111)'.

    The frozen contract (docs/equipment-codes.md, E4): every log line that names
    an emergency carries its code so the AI side and the signals side agree.
    """
    label = "SOS" if flags == SOS_FLAGS else flags_label(flags)
    return f"{label} ({flags_to_field(flags)})"


def _wall_clock() -> str:
    """Local wall-clock stamp for human-facing completion events (E6)."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


# --- trigger parsing ----------------------------------------------------


@dataclass(frozen=True)
class TriggerBatch:
    """One parsed batch of trigger tokens."""

    flags: int = 0  # OR of every recognized class flag
    stop: bool = False  # a stop/cancel/clear/ok token was present
    recognized: bool = False  # at least one class/alias/bits token matched
    tokens: int = 0  # total non-blank tokens seen (stale-discard logging)
    unknown: tuple[str, ...] = ()


def parse_trigger_text(text: str) -> TriggerBatch:
    """Fold trigger tokens IN ORDER; report the unknown ones.

    Case-insensitive. Accepts class names (fire/injured/lost/trapped), "sos",
    the stop synonyms (stop/cancel/clear/ok), and "none" (ignored). A partially
    written or malformed token is simply reported as unknown - never a crash.

    F4: tokens are processed left-to-right (the spool's write order IS the
    temporal order). A stop token clears only the flags queued BEFORE it in this
    batch; class tokens written AFTER a stop survive. So "stop fire" yields
    flags=fire (with stop set), while "fire stop" yields flags=0 - the batch's
    net intent, never a lost trigger.
    """
    flags = 0
    stop = False
    recognized = False
    unknown: list[str] = []
    tokens = text.replace(",", " ").lower().split()
    for token in tokens:
        if token in NO_OP_CLASSES:
            continue
        if token in STOP_TOKENS:
            stop = True
            flags = 0  # F4: a stop clears only what was queued before it
        elif token in SOS_ALIASES:
            flags |= SOS_FLAGS
            recognized = True
        elif token in CLASS_FLAGS:
            flags |= CLASS_FLAGS[token]
            recognized = True
        else:
            unknown.append(token[:32])  # truncate: log hygiene for junk data
    return TriggerBatch(flags, stop, recognized, len(tokens), tuple(unknown))


def consume_spool(path: str, settle_delay: float = 0.0) -> TriggerBatch | None:
    """Atomically consume the trigger spool (rename, then read+delete).

    The rename detaches the batch from writers still appending to the
    original path; the read is capped so a runaway writer cannot wedge the
    daemon. Returns None when there is no spool file.

    F16: `settle_delay` (>0) inserts a short pause AFTER the rename and before
    the read. It mitigates the writer-open-before-rename race: a listener that
    opened the spool by name a hair before our os.replace still holds an fd to
    the now-renamed inode and may be mid-append; the pause lets that short
    write finish so we don't read+delete a half-written batch.
    """
    work = path + SPOOL_WORK_SUFFIX
    try:
        os.replace(path, work)
    except FileNotFoundError:
        return None
    except OSError as exc:
        LOG.warning("spool rename failed (%s): %s", path, exc)
        return None
    if settle_delay > 0:
        time.sleep(settle_delay)
    text = ""
    try:
        with open(work, "r", encoding="utf-8", errors="replace") as spool:
            text = spool.read(SPOOL_MAX_BYTES)
    except OSError as exc:
        LOG.warning("spool read failed (%s): %s", work, exc)
    finally:
        try:
            os.remove(work)
        except OSError:
            pass
    batch = parse_trigger_text(text)
    if batch.unknown:
        LOG.warning("spool: skipping unknown trigger tokens %s", list(batch.unknown))
    return batch


# --- GPIO backends ------------------------------------------------------


class SimBackend:
    """Records (time, pin, value) events; used by tests and --sim runs."""

    def __init__(self, monotonic: Callable[[], float] = time.monotonic):
        self.monotonic = monotonic
        self.events: list[tuple[float, int, int]] = []

    def open(self) -> None:
        pass

    def write_pin(self, pin: int, value: int) -> None:
        self.events.append((self.monotonic(), pin, value))

    def close(self) -> None:
        pass

    def last_value(self, pin: int) -> int:
        for _, event_pin, value in reversed(self.events):
            if event_pin == pin:
                return value
        return 0


class QnxGpioBackend:
    """GPIO via the QNX rpi_gpio resource manager (per-pin text nodes).

    Interface verified on qnxpi 2026-07-12 via `use rpi_gpio`: the resmgr
    mounts one node per GPIO under /dev/gpio, driven by text commands
    written with NO trailing newline (the documented usage is `echo -n`):

        echo -n out > /dev/gpio/17    # program as output
        echo -n on  > /dev/gpio/17    # drive high
        echo -n off > /dev/gpio/17    # drive low

    Each command is a fresh open+write+close, mirroring the documented
    usage exactly; at an 8 Hz carrier (~34 commands/s) that is cheap.
    (A binary /dev/gpio/msg node also exists for rpi_gpio_msg_t messages
    if we ever need more speed.)
    """

    DIRECTION_OUT = b"out"
    VALUE_COMMANDS = (b"off", b"on")  # indexed by pin value 0/1
    RETRY_DELAY_S = 0.05  # F9: one short retry absorbs a transient resmgr hiccup

    def __init__(
        self,
        dev_path: str,
        pins: Sequence[int],
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.dev_path = dev_path
        self.pins = tuple(pins)
        self._sleep = sleep

    def _node_path(self, pin: int) -> str:
        return os.path.join(self.dev_path, str(pin))

    def _command(self, pin: int, command: bytes) -> None:
        fd = os.open(self._node_path(pin), os.O_WRONLY)
        try:
            os.write(fd, command)
        finally:
            os.close(fd)

    def open(self) -> None:
        for pin in self.pins:
            try:
                self._command(pin, self.DIRECTION_OUT)
            except OSError as exc:
                raise BeaconError(
                    f"cannot program GPIO {pin} as output via "
                    f"{self._node_path(pin)}: {exc}. Is the rpi_gpio resource "
                    "manager running (pidin | grep -i gpio), and does this "
                    "user have write access to the gpio group nodes?"
                ) from exc

    def write_pin(self, pin: int, value: int) -> None:
        command = self.VALUE_COMMANDS[1 if value else 0]
        try:
            self._command(pin, command)
            return
        except OSError:
            pass  # F9: transient? give the resmgr one short retry before failing
        self._sleep(self.RETRY_DELAY_S)
        try:
            self._command(pin, command)
        except OSError as exc:
            raise BeaconError(
                f"GPIO write failed (pin {pin}) after retry: {exc}"
            ) from exc

    def close(self) -> None:
        pass  # no persistent handles - every command opens and closes


# --- coil driver + frame transmitter -------------------------------------


class CoilDriver:
    """Maps polarity/enable intent onto the three L298N input pins."""

    def __init__(self, backend, config: Config):
        self.backend = backend
        self.config = config

    def set_polarity(self, forward: bool) -> None:
        self.backend.write_pin(self.config.in3_gpio, 1 if forward else 0)
        self.backend.write_pin(self.config.in4_gpio, 0 if forward else 1)

    def enable(self, on: bool) -> None:
        self.backend.write_pin(self.config.enb_gpio, 1 if on else 0)

    def all_off(self) -> None:
        """Coil safe: gate off first, then both polarity inputs low."""
        self.enable(False)
        self.backend.write_pin(self.config.in3_gpio, 0)
        self.backend.write_pin(self.config.in4_gpio, 0)


class FrameTransmitter:
    """Sends one Manchester frame; always leaves the coil off."""

    def __init__(
        self,
        driver: CoilDriver,
        config: Config,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        config.validate()
        self.driver = driver
        self.config = config
        self.monotonic = monotonic
        self.sleep = sleep

    def _sleep_until(self, deadline: float) -> None:
        delay = deadline - self.monotonic()
        if delay > 0:
            self.sleep(delay)

    def _tone_until(self, symbol_start: float, deadline: float) -> None:
        """8 Hz square: flip IN3/IN4 every half-cycle with ENB high."""
        half_cycle = 1.0 / (2.0 * self.config.carrier_hz)
        self.driver.enable(True)
        edge = 0
        while symbol_start + edge * half_cycle < deadline:
            self.driver.set_polarity(edge % 2 == 0)
            edge += 1
            self._sleep_until(min(symbol_start + edge * half_cycle, deadline))

    def transmit_frame(self, bits: str) -> None:
        half_symbols = regular_manchester(bits)
        half_seconds = self.config.half_symbol_seconds
        start = self.monotonic()
        try:
            for index, tone in enumerate(half_symbols):
                symbol_start = start + index * half_seconds
                deadline = symbol_start + half_seconds
                if tone:
                    self._tone_until(symbol_start, deadline)
                else:
                    self.driver.enable(False)
                    self._sleep_until(deadline)
        finally:
            self.driver.all_off()


# --- beacon daemon --------------------------------------------------------


class Beacon:
    """Heartbeat scheduler + spool-triggered emergency sequences.

    The spool is only read between frames, so a frame that is already going
    out is always finished (never corrupted); the worst-case trigger wait is
    one frame (~12 s). Triggers that arrive while transmitting accumulate
    into a pending set (flags OR-merged, duplicates of the active/pending
    set debounced) and go out as the NEXT sequence - merge-then-queue, never
    interleaved. A heartbeat that comes due during an emergency sequence is
    skipped (the emergency proves aliveness) and the timer resets afterwards.
    Startup transmits nothing: stale spool is discarded and the first
    heartbeat fires one full period after launch.
    """

    def __init__(
        self,
        transmitter: FrameTransmitter,
        config: Config,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.transmitter = transmitter
        self.config = config
        self.monotonic = monotonic
        self.sleep = sleep
        # F18: keep only the most recent frames so a long run cannot grow memory
        # without bound. Tests and bench checks only ever read the recent tail.
        self.frame_history: deque[tuple[float, str, str]] = deque(
            maxlen=MAX_FRAME_HISTORY
        )
        self.pending_flags = 0  # OR-merged classes waiting for the next sequence
        self.active_flags: int | None = None  # sequence on air right now

    def send_frame(self, flags: int, kind: str, progress: str | None = None) -> bool:
        """Transmit one frame, logging its start and completion (E6).

        `progress` is "n/m" for a repeat within an emergency sequence; the
        heartbeat and one-shot paths leave it None. Returns True when the frame
        went out fully, False when a GPIO error aborted it.

        F9: a GPIO write error aborts THIS frame safely (coil driven off,
        best-effort) but is NEVER allowed to escape and kill the daemon loop -
        a crash that stops heartbeats is worse than a dropped frame.
        """
        bits = build_frame(flags)
        started = self.monotonic()
        self.frame_history.append((started, bits, kind))
        where = f" frame {progress}" if progress else ""
        LOG.info("tx start: %s %s%s bits=%s", kind, coded_label(flags), where, bits)
        try:
            self.transmitter.transmit_frame(bits)
        except BeaconError as exc:
            LOG.error(
                "tx ABORTED (GPIO): %s %s%s: %s", kind, coded_label(flags), where, exc
            )
            self._safe_coil_off()
            return False
        LOG.info("tx done: %s %s%s", kind, coded_label(flags), where)
        return True

    def _safe_coil_off(self) -> None:
        """Best-effort coil-off after a GPIO abort; never raises (F9)."""
        try:
            self.transmitter.driver.all_off()
        except BeaconError as exc:
            LOG.error("coil-off after abort also failed: %s", exc)

    def _discard_stale_spool(self) -> None:
        """The trigger queue never survives across runs."""
        try:  # a crash may also have left a half-consumed work file behind
            os.remove(self.config.spool_path + SPOOL_WORK_SUFFIX)
        except OSError:
            pass
        batch = consume_spool(self.config.spool_path)
        if batch is not None and batch.tokens:
            LOG.info("discarded %d stale trigger(s) from a previous run", batch.tokens)

    def _poll_triggers(self) -> bool:
        """Consume the spool into pending state; True if stop was requested."""
        batch = consume_spool(self.config.spool_path, self.config.spool_settle_s)
        if batch is None:
            return False
        if batch.stop:
            LOG.info(
                "stop received: clearing queue (pending=%s active=%s)",
                coded_label(self.pending_flags) if self.pending_flags else "-",
                coded_label(self.active_flags) if self.active_flags is not None else "-",
            )
            self.pending_flags = 0
            if batch.flags:
                # F4: class tokens written AFTER the stop in this batch survive.
                # The active sequence (if any) is being aborted by this same
                # stop, so queue the survivors fresh - no debounce against it.
                self.pending_flags = batch.flags
                LOG.info("queued after stop: %s", coded_label(batch.flags))
            return True
        duplicate = batch.flags & (self.pending_flags | (self.active_flags or 0))
        fresh = batch.flags & ~duplicate
        if duplicate:
            LOG.info(
                "debounced trigger(s) already active/pending: %s",
                coded_label(duplicate),
            )
        if fresh:
            self.pending_flags |= fresh
            LOG.info(
                "queued: %s (pending now %s)",
                coded_label(fresh),
                coded_label(self.pending_flags),
            )
        return False

    def _run_sequence(self, flags: int, budget: int | None) -> tuple[int, bool]:
        """Send the emergency sequence; poll for stop between frames only.

        Returns (frames sent, stopped early). The in-flight frame always
        completes; stop only cancels repeats that have not started.
        """
        self.active_flags = flags
        sent = 0
        stopped = False
        failed = False  # F9: a GPIO abort on any repeat suppresses SIGNAL SENT
        try:
            for repeat in range(self.config.emergency_repeats):
                if budget is not None and sent >= budget:
                    break
                if repeat > 0:
                    self.sleep(self.config.emergency_gap_s)
                    if self._poll_triggers():
                        stopped = True
                        break
                if not self.send_frame(
                    flags,
                    "emergency",
                    progress=f"{repeat + 1}/{self.config.emergency_repeats}",
                ):
                    failed = True
                sent += 1
                if repeat + 1 < self.config.emergency_repeats and self._poll_triggers():
                    stopped = True
                    break
        finally:
            self.active_flags = None
        if stopped:
            LOG.info(
                "emergency sequence aborted: %s after %d of %d repeats",
                coded_label(flags),
                sent,
                self.config.emergency_repeats,
            )
        elif failed:
            # Never claim SIGNAL SENT when a frame did not go out (silence must
            # stay honest so the surface can trust the log).
            LOG.error(
                "emergency sequence had GPIO failures: %s (%d/%d frames attempted)",
                coded_label(flags),
                sent,
                self.config.emergency_repeats,
            )
        elif sent >= self.config.emergency_repeats:
            # E6: the whole repeat-set is on the air. Announce completion only
            # AFTER the final frame fully finishes, stamped with the wall clock.
            LOG.info(
                "SIGNAL SENT: %s x%d complete at %s",
                coded_label(flags),
                sent,
                _wall_clock(),
            )
        return sent, stopped

    def run(self, max_frames: int | None = None) -> None:
        """Main loop; max_frames bounds the run for tests/bench checks."""
        LOG.info(
            "beacon up: silent start, first heartbeat in %gs, spool %s",
            self.config.heartbeat_interval_s,
            self.config.spool_path,
        )
        self._discard_stale_spool()
        interval = self.config.heartbeat_interval_s
        next_heartbeat = self.monotonic() + interval  # NO transmission at launch
        sent = 0
        try:
            while max_frames is None or sent < max_frames:
                # Poll the spool for its side effects only. A stop clears the
                # pending queue (and aborts any active sequence via the polls
                # inside _run_sequence). F3: a stop must NEVER touch the
                # heartbeat clock - resetting it on a stop could nearly double
                # the radio silence and break silence-is-alarm.
                self._poll_triggers()
                now = self.monotonic()
                if self.pending_flags:
                    flags = self.pending_flags
                    self.pending_flags = 0
                    budget = None if max_frames is None else max_frames - sent
                    frames, _ = self._run_sequence(flags, budget)
                    sent += frames
                    after = self.monotonic()
                    if next_heartbeat <= after:
                        LOG.info("heartbeat skipped (emergency sequence was active)")
                    next_heartbeat = after + interval
                elif now >= next_heartbeat:
                    self.send_frame(HEARTBEAT_FLAGS, "heartbeat")
                    sent += 1
                    next_heartbeat = now + interval
                else:
                    self.sleep(min(self.config.poll_interval_s, next_heartbeat - now))
        finally:
            self.transmitter.driver.all_off()


# --- single-instance lock -------------------------------------------------


class SingleInstanceLock:
    """Pidfile lock so two processes can never fight over the coil."""

    def __init__(self, path: str):
        self.path = path
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self) -> None:
        for _ in range(2):  # second try after clearing a stale pidfile
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                with os.fdopen(fd, "w") as pidfile:
                    pidfile.write(str(os.getpid()))
                self._held = True
                return
            except FileExistsError:
                other = self._read_pid()
                if other is not None and _pid_alive(other):
                    raise BeaconError(
                        f"another beacon (pid {other}) already owns the coil "
                        f"({self.path}); stop it first"
                    )
                try:
                    os.remove(self.path)  # stale pidfile
                except OSError:
                    pass
        raise BeaconError(f"could not acquire pidfile {self.path}")

    def _read_pid(self) -> int | None:
        try:
            with open(self.path, "r", encoding="ascii") as pidfile:
                return int(pidfile.read().strip())
        except (OSError, ValueError):
            return None

    def release(self) -> None:
        if self._held:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self._held = False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OverflowError):
        return True
    return True


# --- CLI ------------------------------------------------------------------


def _raise_exit(signum, _frame):
    raise SystemExit(128 + signum)


def _setup_logging(log_path: str, plain: bool = False) -> None:
    LOG.setLevel(logging.INFO)
    for handler in list(LOG.handlers):  # idempotent; close to avoid leaking fds
        LOG.removeHandler(handler)
        try:
            handler.close()
        except OSError:
            pass
    # rocko.sh owns numbering + timestamps for the unified stream, so it runs us
    # with --log-plain (bare messages); standalone we keep asctime + level.
    fmt = logging.Formatter(
        "%(message)s" if plain else "%(asctime)s %(levelname)s %(message)s"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    LOG.addHandler(stream)
    try:
        rotating = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
        )
        rotating.setFormatter(fmt)
        LOG.addHandler(rotating)
    except OSError as exc:
        LOG.warning("cannot open log file %s: %s (logging to stderr only)", log_path, exc)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cave Beacon coil transmitter (QNX). No args = daemon mode."
    )
    parser.add_argument(
        "--send",
        action="append",
        metavar="CLASS",
        help="one-shot: transmit a single frame for CLASS(es) "
        "(fire/injured/lost/trapped/sos) and exit",
    )
    parser.add_argument("--sim", action="store_true", help="simulated GPIO (no hardware)")
    parser.add_argument(
        "--log-plain",
        action="store_true",
        help="bare message log format (rocko.sh owns numbering + timestamps)",
    )
    # F7: the four knobs below change the frozen frame contract (timing, carrier,
    # heartbeat period, GPIO node). They are accepted ONLY with --sim; on real
    # hardware the contract is not tunable from the CLI.
    parser.add_argument(
        "--heartbeat-interval", type=float, default=None, help="(--sim only)"
    )
    parser.add_argument("--bit-seconds", type=float, default=None, help="(--sim only)")
    parser.add_argument("--carrier", type=float, default=None, help="(--sim only)")
    parser.add_argument("--spool", default=None)
    parser.add_argument("--pidfile", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--gpio-dev", default=None, help="(--sim only)")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> Config:
    overrides = {
        "heartbeat_interval_s": args.heartbeat_interval,
        "bit_seconds": args.bit_seconds,
        "carrier_hz": args.carrier,
        "spool_path": args.spool,
        "pidfile_path": args.pidfile,
        "log_path": args.log_file,
        "gpio_dev": args.gpio_dev,
    }
    config = replace(
        Config(), **{key: value for key, value in overrides.items() if value is not None}
    )
    if args.sim:
        # F12: a --sim run must NEVER remove or race the live daemon's files.
        # Unless the operator overrode them explicitly, point the spool and
        # pidfile at sim-suffixed defaults so a bench run leaves the real ones
        # (and thus the real coil's queue) untouched.
        defaults = Config()
        if args.spool is None:
            config = replace(config, spool_path=defaults.spool_path + ".sim")
        if args.pidfile is None:
            config = replace(config, pidfile_path=defaults.pidfile_path + ".sim")
    config.validate()
    return config


def _reject_off_contract_knobs(args: argparse.Namespace) -> list[str]:
    """F7: contract-changing knobs are legal only under --sim. Returns the
    offending flag names (empty when the invocation is contract-safe)."""
    if args.sim:
        return []
    knobs = {
        "--heartbeat-interval": args.heartbeat_interval,
        "--bit-seconds": args.bit_seconds,
        "--carrier": args.carrier,
        "--gpio-dev": args.gpio_dev,
    }
    return [name for name, value in knobs.items() if value is not None]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # F7: reject frame-contract knobs on real hardware BEFORE touching any GPIO.
    off_contract = _reject_off_contract_knobs(args)
    if off_contract:
        print(
            "refusing off-contract knob(s) "
            f"{off_contract} without --sim: the frame contract is frozen "
            "(docs/equipment-codes.md). No GPIO was touched.",
            file=sys.stderr,
        )
        return 2

    try:
        config = _config_from_args(args)
    except ValueError as exc:
        print(f"bad config: {exc}", file=sys.stderr)
        return 2

    send_flags: int | None = None
    if args.send:
        batch = parse_trigger_text(" ".join(args.send))
        if batch.unknown or batch.stop or not batch.recognized:
            print(f"unknown class(es): {list(batch.unknown) or args.send}", file=sys.stderr)
            return 2
        send_flags = batch.flags

    _setup_logging(config.log_path, plain=args.log_plain)
    signal.signal(signal.SIGINT, _raise_exit)
    signal.signal(signal.SIGTERM, _raise_exit)

    pins = (config.in3_gpio, config.in4_gpio, config.enb_gpio)
    backend = SimBackend() if args.sim else QnxGpioBackend(config.gpio_dev, pins)
    lock = None if args.sim else SingleInstanceLock(config.pidfile_path)
    driver = CoilDriver(backend, config)
    try:
        if lock is not None:
            lock.acquire()
        backend.open()
        transmitter = FrameTransmitter(driver, config)
        beacon = Beacon(transmitter, config)
        if send_flags is not None:
            # F9: send_frame swallows GPIO errors to keep the daemon alive, but
            # a one-shot bench send must still report failure via its exit code.
            if not beacon.send_frame(send_flags, "one-shot"):
                return 1
        else:
            beacon.run()
        return 0
    except BeaconError as exc:
        LOG.error("%s", exc)
        return 1
    finally:
        try:
            driver.all_off()  # coil off and ENB low, ALWAYS
        except BeaconError as exc:
            LOG.error("cleanup GPIO write failed: %s", exc)
        except Exception as exc:  # F17: never silently swallow the unexpected
            LOG.error("unexpected error during coil-off cleanup: %s", exc)
        backend.close()
        if lock is None or lock.held:
            # We owned the coil: the trigger queue dies with us. (If the
            # lock was NOT ours, another beacon is live - leave its spool.)
            for stale in (config.spool_path, config.spool_path + SPOOL_WORK_SUFFIX):
                try:
                    os.remove(stale)
                except OSError:
                    pass
        if lock is not None:
            lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
