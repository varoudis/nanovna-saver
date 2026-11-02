"""Shared fake serial interface for driver-level tests.

No hardware is involved: the fake answers the ASCII commands the drivers
actually send, so a driver can be constructed and swept in-process.
"""

from __future__ import annotations

from threading import RLock

HELP = "Commands: help version frequencies data scan sweep"


def help_lines(*extra: str) -> str:
    return " ".join((HELP, *extra))


class FakeInterface:
    """Minimal stand-in for :class:`NanoVNASaver.Hardware.Serial.Interface`."""

    def __init__(self, replies=None, responder=None):
        self.is_open = True
        self.timeout = 0.05
        self.lock = RLock()
        self.writes: list[str] = []
        self.replies = dict(replies or {})
        self.responder = responder
        self._pending: list[str] = []

    # -- serial surface used by VNA ---------------------------------
    def read(self, _size: int = 1) -> bytes:
        # No unsolicited bytes: every reply is produced by a write.
        return b""

    def write(self, data: bytes) -> int:
        command = data.decode("ascii").strip()
        self.writes.append(command)
        lines = self._answer(command)
        self._pending = [f"{line}\r\n" for line in lines]
        self._pending.append("ch>\r\n")
        return len(data)

    def readline(self) -> bytes:
        if not self._pending:
            return b""
        return self._pending.pop(0).encode("ascii")

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def __str__(self) -> str:
        return "fake"

    # -- replies ----------------------------------------------------
    def _answer(self, command: str) -> list[str]:
        if self.responder is not None:
            answer = self.responder(command)
            if answer is not None:
                return list(answer)
        head = command.split(" ", 1)[0]
        if command in self.replies:
            return self.replies[command]
        if head in self.replies:
            return self.replies[head]
        return []

    def commands(self, prefix: str) -> list[str]:
        return [c for c in self.writes if c.startswith(prefix)]
