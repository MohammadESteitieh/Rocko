#!/usr/bin/env python3
"""Shared QNX GPIO and coil primitives for magnetic-link research transmitters."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Callable, Sequence


class BeaconError(RuntimeError):
    """Fatal transmitter hardware or ownership error."""


@dataclass(frozen=True)
class Config:
    in3_gpio: int = 22
    in4_gpio: int = 17
    enb_gpio: int = 27
    carrier_hz: float = 8.0
    bit_seconds: float = 1.0
    pidfile_path: str = "/tmp/beacon.pid"
    gpio_dev: str = "/dev/gpio"

    @property
    def half_symbol_seconds(self) -> float:
        return self.bit_seconds / 2.0

    def validate(self) -> None:
        if len({self.in3_gpio, self.in4_gpio, self.enb_gpio}) != 3:
            raise ValueError("IN3, IN4, and ENB must use different GPIO pins")
        if self.carrier_hz <= 0 or self.bit_seconds <= 0:
            raise ValueError("carrier_hz and bit_seconds must be positive")
        cycles_per_half = self.carrier_hz * self.half_symbol_seconds
        if cycles_per_half != int(cycles_per_half):
            raise ValueError("carrier_hz * bit_seconds/2 must be whole carrier cycles")


def regular_manchester(bits: str) -> list[int]:
    if not bits or any(bit not in "01" for bit in bits):
        raise ValueError("message must be a non-empty binary string")
    return [level for bit in bits for level in ((0, 1) if bit == "0" else (1, 0))]


class SimBackend:
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
    """QNX rpi_gpio per-pin text-node backend."""

    DIRECTION_OUT = b"out"
    VALUE_COMMANDS = (b"off", b"on")
    RETRY_DELAY_S = 0.05

    def __init__(self, dev_path: str, pins: Sequence[int],
                 sleep: Callable[[float], None] = time.sleep):
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
                    f"cannot program GPIO {pin} as output via {self._node_path(pin)}: "
                    f"{exc}; verify rpi_gpio and gpio-group access"
                ) from exc

    def write_pin(self, pin: int, value: int) -> None:
        command = self.VALUE_COMMANDS[1 if value else 0]
        try:
            self._command(pin, command)
            return
        except OSError:
            self._sleep(self.RETRY_DELAY_S)
        try:
            self._command(pin, command)
        except OSError as exc:
            raise BeaconError(f"GPIO write failed (pin {pin}) after retry: {exc}") from exc

    def close(self) -> None:
        pass


class CoilDriver:
    def __init__(self, backend, config: Config):
        self.backend = backend
        self.config = config

    def set_polarity(self, forward: bool) -> None:
        self.backend.write_pin(self.config.in3_gpio, 1 if forward else 0)
        self.backend.write_pin(self.config.in4_gpio, 0 if forward else 1)

    def enable(self, on: bool) -> None:
        self.backend.write_pin(self.config.enb_gpio, 1 if on else 0)

    def all_off(self) -> None:
        self.enable(False)
        self.backend.write_pin(self.config.in3_gpio, 0)
        self.backend.write_pin(self.config.in4_gpio, 0)


class FrameTransmitter:
    def __init__(self, driver: CoilDriver, config: Config,
                 monotonic: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
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
        half_cycle = 1.0 / (2.0 * self.config.carrier_hz)
        self.driver.enable(True)
        edge = 0
        while symbol_start + edge * half_cycle < deadline:
            self.driver.set_polarity(edge % 2 == 0)
            edge += 1
            self._sleep_until(min(symbol_start + edge * half_cycle, deadline))

    def transmit_frame(self, bits: str) -> None:
        levels = regular_manchester(bits)
        half_seconds = self.config.half_symbol_seconds
        start = self.monotonic()
        try:
            for index, tone in enumerate(levels):
                symbol_start = start + index * half_seconds
                deadline = symbol_start + half_seconds
                if tone:
                    self._tone_until(symbol_start, deadline)
                else:
                    self.driver.enable(False)
                    self._sleep_until(deadline)
        finally:
            self.driver.all_off()


class SingleInstanceLock:
    def __init__(self, path: str):
        self.path = path
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self) -> None:
        for _ in range(2):
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
                        f"another process (pid {other}) owns the coil ({self.path})"
                    )
                try:
                    os.remove(self.path)
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


def _raise_exit(signum, _frame):
    raise SystemExit(128 + signum)
