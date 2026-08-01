#!/usr/bin/env python3
"""Rocko receiver event log — monotonic numbers, timestamps, file + on-screen.

Every notable receiver event (capture start, tone observed, decoded emergency,
error) becomes a numbered line ``[#0001] 2026-07-12T14:03:11 DECODE  fire
(1000)``. The same stream goes to a size-capped log file and to an in-memory
deque the GUI renders as a compact recent-events list. No serial or GUI
dependencies, so it is trivially testable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading
from typing import Deque, List, Optional

MAX_LOG_BYTES = 2_000_000  # rotate once the log passes ~2 MB (disk-full guard)
ON_SCREEN_EVENTS = 100     # scrollable in-memory history for the GUI panel


@dataclass(frozen=True)
class Event:
    number: int
    timestamp: str
    kind: str
    message: str

    def format(self) -> str:
        return f"[#{self.number:04d}] {self.timestamp}  {self.kind:<7} {self.message}"

    def compact(self) -> str:
        """Shorter form (clock time only) for the on-screen list."""
        clock = self.timestamp[11:19] if len(self.timestamp) >= 19 else self.timestamp
        return f"#{self.number:04d} {clock}  {self.kind:<7} {self.message}"


class EventLog:
    """Thread-safe numbered event log to file and an in-memory ring buffer."""

    def __init__(self, path: Optional[Path] = None, on_screen: int = ON_SCREEN_EVENTS):
        self._lock = threading.Lock()
        self._counter = 0
        self._recent: Deque[Event] = deque(maxlen=on_screen)
        self.path = Path(path) if path else None
        self._handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed()
            # Best effort: a failing log file must never crash the receiver.
            try:
                self._handle = self.path.open("a", buffering=1)
            except OSError as exc:  # pragma: no cover - disk/permission edge
                self._handle = None
                print(f"warning: cannot open log file {self.path}: {exc}")

    def _rotate_if_needed(self) -> None:
        try:
            if self.path and self.path.exists() and self.path.stat().st_size > MAX_LOG_BYTES:
                self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))
        except OSError:
            pass

    def emit(self, kind: str, message: str, *, echo: bool = True) -> Event:
        """Record one event; returns it. Never raises on a log-file failure."""
        with self._lock:
            self._counter += 1
            event = Event(
                number=self._counter,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                kind=kind.upper(),
                message=message,
            )
            self._recent.append(event)
            if self._handle is not None and not self._handle.closed:
                try:
                    self._handle.write(event.format() + "\n")
                except (OSError, ValueError) as exc:  # disk-full or closed-handle edge
                    print(f"warning: log write failed: {exc}")
        if echo:
            print(event.format(), flush=True)
        return event

    def recent(self) -> List[Event]:
        with self._lock:
            return list(self._recent)

    @property
    def count(self) -> int:
        with self._lock:
            return self._counter

    def close(self) -> None:
        with self._lock:
            if self._handle is not None and not self._handle.closed:
                self._handle.flush()
                self._handle.close()
