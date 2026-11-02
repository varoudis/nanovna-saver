#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020 Rune B. Broberg
#  Copyright (C) 2020ff NanoVNA-Saver Authors
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""An optional local REST/WebSocket service for sweep automation (AUTO-04).

Nothing here is imported by the application, nothing happens on import, and
no Qt module is in this module's import graph: a script (or the integrator's
GUI hook) constructs a :class:`MeasurementService` and calls
:meth:`MeasurementService.start_server` explicitly.

Protocol
--------
Every request is versioned twice: the path carries the version
(``/v1/<command>``) and a ``POST`` body must carry ``schema_version``.
A missing, unparsable or unsupported version is a structured error, and so
is an unknown command -- an old client talking to a new server never gets a
silent success.

===  ==================================================================
path command
===  ==================================================================
GET  ``/v1/info``          server, limits and whether a device is open
GET  ``/v1/status``        server state, or one run's state by ``run_id``
GET  ``/v1/last_result``   the most recent completed result
GET  ``/v1/checks``        the frozen limit evaluations of one run
GET  ``/v1/events``        WebSocket event stream
POST ``/v1/start``         start a sweep (refused while one is live)
POST ``/v1/cancel``        cancel the run named by ``run_id``
POST ``/v1/export``        write a run's result into the export directory
===  ==================================================================

Every JSON response uses the same envelope::

    {"schema_version": 1, "ok": true, "command": "info", "result": {...}}
    {"schema_version": 1, "ok": false, "error": {"code": ..., ...}}

Design rules
------------
* **Loopback by default, and guarded.**  A non-loopback bind needs both
  ``allow_remote=True`` and a token of at least
  :data:`MINIMUM_TOKEN_LENGTH` characters, or the configuration is refused
  before a socket is created.
* **Not reachable from a web page.**  Loopback does not keep out a page open
  in the operator's browser, so any request with an ``Origin`` header, any
  ``Host`` that is not a loopback name on a loopback bind, and any body that
  is not ``application/json`` is refused.
* **One run at a time.**  A second ``start`` while a run is live is a
  ``busy`` error carrying the live run id: there is no queue and no race.
  Two clients therefore never read another run's data, and every result is
  frozen and addressable by its own ``run_id``.
* **Bounded input.**  Request bodies, client connections, subscribers and
  their event queues are all capped, and publishing an event never blocks:
  a subscriber that cannot keep up is dropped.
* **No client-chosen paths.**  ``export`` only writes inside the directory
  the server was configured with; the request names a relative filename
  that is resolved with ``realpath`` and checked with ``commonpath``.
* **Server-owned runs.**  A client that disconnects cannot strand or
  corrupt a run: the run lives in the service, not in the connection.

Known limitations
-----------------
The WebSocket implementation is a deliberately small, standard-library
subset of RFC 6455: text frames, close, ping and pong, no fragmentation on
send, no permessage-deflate, and no TLS (put the service behind a reverse
proxy if it ever leaves loopback).  The REST half has no such limitation.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import queue
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import parse_qs, urlsplit

from .Acquisition import (
    AcquisitionError,
    CancelledError,
    CancelToken,
    SweepRequest,
    UnsupportedError,
    open_device,
)
from .Limits import LimitError, LimitEvaluation, LimitSet, Verdict
from .Measurement import MeasurementSnapshot
from .Touchstone import TouchstoneExportError

__all__ = [
    "SCHEMA_VERSION",
    "MeasurementService",
    "ServiceConfig",
    "ServiceError",
    "Subscriber",
    "is_loopback_host",
]

LOGGER = logging.getLogger(__name__)

#: Protocol version carried by every request and response.
SCHEMA_VERSION = 1
SUPPORTED_VERSIONS: tuple[int, ...] = (1,)
PROTOCOL_NAME = "nanovna-saver-service"

#: A non-loopback bind requires a token at least this long; a short shared
#: secret is not an access control.
MINIMUM_TOKEN_LENGTH = 16

#: Hosts that are treated as loopback without a DNS lookup.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

#: Channels a sweep may name.  The device decides which of them exist.
_KNOWN_CHANNELS = ("S11", "S21")

#: Bounds on the work one request may ask for.
MINIMUM_POINTS = 2
MAXIMUM_POINTS = 100_000
MAXIMUM_AVERAGE = 1_000
MAXIMUM_NAME_LENGTH = 200


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
class ServiceError(Exception):
    """A structured failure that maps to one HTTP status and error code."""

    code = "service_error"
    status = 500

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        if code is not None:
            self.code = code
        if status is not None:
            self.status = status
        self.details = dict(details) if details else {}

    def error_body(self) -> dict:
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            body["details"] = self.details
        return body

    def document(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "error": self.error_body(),
        }


class ConfigError(ServiceError, ValueError):
    """The service cannot be configured this way; nothing was bound."""

    code = "invalid_configuration"
    status = 400


class InvalidRequestError(ServiceError):
    code = "invalid_request"
    status = 400


class InvalidJsonError(InvalidRequestError):
    code = "invalid_json"


class UnsupportedVersionError(InvalidRequestError):
    code = "unsupported_version"


class UnknownCommandError(InvalidRequestError):
    code = "unknown_command"
    status = 404


class MethodNotAllowedError(InvalidRequestError):
    code = "method_not_allowed"
    status = 405


class UnauthorizedError(ServiceError):
    code = "unauthorized"
    status = 401


class PayloadTooLargeError(ServiceError):
    code = "payload_too_large"
    status = 413


class TooManyClientsError(ServiceError):
    code = "too_many_clients"
    status = 503


class TooManySubscribersError(ServiceError):
    code = "too_many_subscribers"
    status = 503


class RunBusyError(ServiceError):
    """A run is already live; the caller must not queue a second one."""

    code = "busy"
    status = 409


class RunNotFoundError(ServiceError):
    code = "run_not_found"
    status = 404


class StaleRunError(ServiceError):
    """The named run is no longer live, so the command cannot act on it."""

    code = "stale_run"
    status = 409


class RunIncompleteError(ServiceError):
    """The named run exists but never produced a result to read or export."""

    code = "run_incomplete"
    status = 409


class NoResultError(ServiceError):
    code = "no_result"
    status = 404


class PathRefusedError(ServiceError):
    code = "path_refused"
    status = 403


class ExportDisabledError(ServiceError):
    code = "export_disabled"
    status = 409


class ExportExistsError(ServiceError):
    code = "export_exists"
    status = 409


class ExportFailedError(ServiceError):
    code = "export_failed"
    status = 500


class ExportRefusedError(ServiceError):
    """This run's data cannot be written in the requested format."""

    code = "export_refused"
    status = 409


class DeviceUnavailableError(ServiceError):
    code = "no_device"
    status = 503


class ForbiddenRequestError(ServiceError):
    """The request came from a web page, or through a rebound host name."""

    code = "forbidden_request"
    status = 403


class UnsupportedMediaTypeError(ServiceError):
    code = "unsupported_media_type"
    status = 415


class InternalServiceError(ServiceError):
    code = "internal_error"
    status = 500


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
def is_loopback_host(host: str) -> bool:
    """True when ``host`` can only be reached from this machine."""
    if not isinstance(host, str):
        return False
    name = host.strip().lower()
    if name in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(name.strip("[]")).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class ServiceConfig:
    """Everything the service may touch, decided before it is started.

    Raising here means nothing has been bound: the refusal is cheaper than
    a socket nobody asked for, and a test can assert it without listening.
    """

    #: Loopback by default.  Anything else is an explicit, guarded choice.
    host: str = "127.0.0.1"
    #: ``0`` asks the operating system for a free ephemeral port.
    port: int = 0
    #: Directory every ``export`` writes into; empty disables export.
    export_dir: str = ""
    #: Shared secret required from every request when non-empty, and always
    #: required for a non-loopback bind.
    token: str = ""
    #: Must be set deliberately before a non-loopback bind is accepted.
    allow_remote: bool = False
    max_body_bytes: int = 64 * 1024
    max_clients: int = 8
    max_subscribers: int = 4
    subscriber_queue: int = 32
    max_retained_runs: int = 8
    #: Seconds a single socket operation may block before it is abandoned.
    socket_timeout: float = 5.0

    def __post_init__(self) -> None:
        self.validate()

    @property
    def loopback(self) -> bool:
        return is_loopback_host(self.host)

    @property
    def authentication_required(self) -> bool:
        return bool(self.token)

    def validate(self) -> None:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ConfigError("the bind host must not be empty")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise ConfigError("the bind port must be an integer")
        if not 0 <= self.port <= 65535:
            raise ConfigError(f"the bind port {self.port} is out of range")
        if not self.loopback:
            if not self.allow_remote:
                raise ConfigError(
                    f"refusing to bind {self.host!r}: only loopback is"
                    " allowed unless allow_remote=True is passed"
                    " deliberately"
                )
            if not self.token:
                raise ConfigError(
                    f"refusing to bind {self.host!r} without a token:"
                    " non-loopback access always requires a shared secret"
                )
        if self.token and len(self.token) < 8:
            raise ConfigError("the token must be at least 8 characters")
        if self.token and not self.loopback:
            if len(self.token) < MINIMUM_TOKEN_LENGTH:
                raise ConfigError(
                    f"a non-loopback bind needs a token of at least"
                    f" {MINIMUM_TOKEN_LENGTH} characters"
                )
        for name, value, floor in (
            ("max_body_bytes", self.max_body_bytes, 256),
            ("max_clients", self.max_clients, 1),
            ("max_subscribers", self.max_subscribers, 1),
            ("subscriber_queue", self.subscriber_queue, 1),
            ("max_retained_runs", self.max_retained_runs, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{name} must be an integer")
            if value < floor:
                raise ConfigError(f"{name} must be at least {floor}")
        if self.socket_timeout <= 0:
            raise ConfigError("socket_timeout must be positive")
        if self.export_dir:
            root = os.path.abspath(self.export_dir)
            if not os.path.isdir(root):
                raise ConfigError(
                    f"export_dir {self.export_dir!r} is not an existing"
                    " directory"
                )


# ----------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------
def _require_object(value: object, where: str) -> dict:
    if not isinstance(value, dict):
        raise InvalidRequestError(f"{where} must be a JSON object")
    return value


def _reject_unknown(value: Mapping, allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise InvalidRequestError(
            f"{where}: unknown key(s) {', '.join(unknown)};"
            f" known keys are {', '.join(allowed)}"
        )


def _require_int(
    value: object, where: str, *, minimum: int, maximum: int
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidRequestError(f"{where} must be an integer")
    if not minimum <= value <= maximum:
        raise InvalidRequestError(
            f"{where} must be between {minimum} and {maximum}, got {value}"
        )
    return value


@dataclass(frozen=True)
class SweepSpec:
    """One validated sweep request; identical in meaning to the API's."""

    start_hz: int
    stop_hz: int
    points: int
    average: int = 1
    channels: tuple[str, ...] | None = None
    name: str = ""

    def document(self) -> dict:
        return {
            "start_hz": self.start_hz,
            "stop_hz": self.stop_hz,
            "points": self.points,
            "average": self.average,
            "channels": list(self.channels) if self.channels else None,
            "name": self.name,
        }

    def request(self, channels: Sequence[str]) -> SweepRequest:
        return SweepRequest(
            start_hz=self.start_hz,
            stop_hz=self.stop_hz,
            points=self.points,
            channels=tuple(channels),
            average=self.average,
            name=self.name or "service sweep",
        )


def parse_sweep_spec(value: object) -> SweepSpec:
    sweep = _require_object(value, "sweep")
    _reject_unknown(
        sweep,
        ("start_hz", "stop_hz", "points", "average", "channels", "name"),
        "sweep",
    )
    for required in ("start_hz", "stop_hz"):
        if required not in sweep:
            raise InvalidRequestError(f"sweep: {required} is required")
    start = _require_int(
        sweep["start_hz"], "sweep.start_hz", minimum=1, maximum=10**12
    )
    stop = _require_int(
        sweep["stop_hz"], "sweep.stop_hz", minimum=1, maximum=10**12
    )
    if start >= stop:
        raise InvalidRequestError(
            f"sweep.start_hz {start} must be below sweep.stop_hz {stop}"
        )
    points = _require_int(
        sweep.get("points", 101),
        "sweep.points",
        minimum=MINIMUM_POINTS,
        maximum=MAXIMUM_POINTS,
    )
    average = _require_int(
        sweep.get("average", 1),
        "sweep.average",
        minimum=1,
        maximum=MAXIMUM_AVERAGE,
    )
    name = sweep.get("name", "")
    if not isinstance(name, str):
        raise InvalidRequestError("sweep.name must be a string")
    if len(name) > MAXIMUM_NAME_LENGTH:
        raise InvalidRequestError(
            f"sweep.name must be at most {MAXIMUM_NAME_LENGTH} characters"
        )
    channels = sweep.get("channels")
    if channels is not None:
        if not isinstance(channels, list) or not channels:
            raise InvalidRequestError(
                "sweep.channels must be a non-empty list of S11/S21"
            )
        ordered: list[str] = []
        for entry in channels:
            if entry not in _KNOWN_CHANNELS:
                raise InvalidRequestError(
                    f"sweep.channels: unknown channel {entry!r};"
                    f" known channels are {', '.join(_KNOWN_CHANNELS)}"
                )
            if entry in ordered:
                raise InvalidRequestError(
                    f"sweep.channels: {entry} is listed twice"
                )
            ordered.append(entry)
        channels = tuple(ordered)
    return SweepSpec(
        start_hz=start,
        stop_hz=stop,
        points=points,
        average=average,
        channels=channels,
        name=name,
    )


def parse_limits(value: object) -> LimitSet | None:
    """Accept an inline limit document and nothing else.

    A filename is refused explicitly: the service never reads a file a
    client names, so a limit set has to arrive as data.
    """
    if value is None:
        return None
    if isinstance(value, str):
        raise InvalidRequestError(
            "limits must be an inline limit-set document; this service"
            " never opens a path named by a client"
        )
    document = _require_object(value, "limits")
    try:
        return LimitSet.from_dict(document)
    except LimitError as error:
        raise InvalidRequestError(f"limits: {error}") from error


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _encode_time(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat()


def summarise_checks(
    checks: Sequence[LimitEvaluation],
) -> dict:
    """Counts and the worst verdict, with UNAVAILABLE never reading as pass.

    ``verdict`` is ``None`` when nothing was checked: an unchecked run is
    not a pass either.
    """
    passed = sum(1 for item in checks if item.verdict is Verdict.PASS)
    failed = sum(1 for item in checks if item.verdict is Verdict.FAIL)
    unavailable = sum(
        1 for item in checks if item.verdict is Verdict.UNAVAILABLE
    )
    if not checks:
        verdict = None
    elif failed:
        verdict = Verdict.FAIL.value
    elif unavailable:
        verdict = Verdict.UNAVAILABLE.value
    else:
        verdict = Verdict.PASS.value
    return {
        "checked": bool(checks),
        "verdict": verdict,
        "passed": passed,
        "failed": failed,
        "unavailable": unavailable,
    }


@dataclass(frozen=True)
class RunResult:
    """One completed measurement; immutable and addressed by ``run_id``."""

    run_id: str
    spec: SweepSpec
    channels: tuple[str, ...]
    snapshot: MeasurementSnapshot
    limits: LimitSet | None
    checks: tuple[LimitEvaluation, ...]
    started_at: datetime
    finished_at: datetime

    @property
    def summary(self) -> dict:
        return summarise_checks(self.checks)

    def checks_document(self) -> dict:
        return {
            "run_id": self.run_id,
            "checked": bool(self.checks),
            "summary": summarise_checks(self.checks),
            "limits": [item.to_dict() for item in self.checks],
        }

    def document(self) -> dict:
        """A fresh plain-data view; callers cannot mutate the stored one."""
        return {
            "run_id": self.run_id,
            "sweep": {**self.spec.document(), "channels": list(self.channels)},
            "started_at": _encode_time(self.started_at),
            "finished_at": _encode_time(self.finished_at),
            "snapshot": self.snapshot.to_dict(),
            "summary": self.summary,
            "limits": [item.to_dict() for item in self.checks],
        }


# ----------------------------------------------------------------------
# Event subscribers
# ----------------------------------------------------------------------
class Subscriber:
    """A bounded event queue drained by one connection's writer thread.

    The publisher calls :meth:`offer`, which never blocks: a full queue
    means the reader is too slow and the subscription is closed with the
    reason ``lagged``.  Acquisition therefore cannot be held up by a client
    that stopped reading its socket.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("a subscriber queue needs a positive capacity")
        self.capacity = capacity
        self.dropped = 0
        self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        self._closed = threading.Event()
        self._reason = ""

    # -- publisher side ------------------------------------------------
    def offer(self, event: Mapping) -> bool:
        """Queue ``event``; return False when this subscriber was dropped."""
        if self._closed.is_set():
            return False
        try:
            self._queue.put_nowait(dict(event))
        except queue.Full:
            self.dropped += 1
            self.close("lagged")
            return False
        return True

    def close(self, reason: str = "closed") -> None:
        """Mark the subscription finished; safe to call from any thread."""
        if self._closed.is_set():
            return
        self._reason = reason
        self._closed.set()

    # -- writer side ---------------------------------------------------
    def poll(self, timeout: float = 0.25) -> dict | None:
        """One event, or ``None`` when the timeout expired."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def pending(self) -> int:
        return self._queue.qsize()


# ----------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------
STATE_RUNNING = "running"
STATE_CANCELLING = "cancelling"
STATE_COMPLETED = "completed"
STATE_CANCELLED = "cancelled"
STATE_FAILED = "failed"


@dataclass
class _RunState:
    """Mutable bookkeeping for one run; guarded by the service lock."""

    run_id: str
    spec: SweepSpec
    limits: LimitSet | None
    started_at: datetime
    state: str = STATE_RUNNING
    finished_at: datetime | None = None
    result: RunResult | None = None
    failure: dict | None = None
    cancel_requested: bool = False
    cancel: CancelToken = field(default_factory=CancelToken)
    thread: threading.Thread | None = None

    def status_document(self) -> dict:
        return {
            "run_id": self.run_id,
            "state": self.state,
            "sweep": self.spec.document(),
            "started_at": _encode_time(self.started_at),
            "finished_at": _encode_time(self.finished_at),
            "has_result": self.result is not None,
            "error": dict(self.failure) if self.failure else None,
            "summary": self.result.summary if self.result else None,
        }


class DeviceFactory(Protocol):
    """Callable producing one fresh, unopened device per run."""

    def __call__(self) -> Any: ...


# ----------------------------------------------------------------------
# WebSocket framing (RFC 6455 subset)
# ----------------------------------------------------------------------
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONTINUATION = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

CLOSE_NORMAL = 1000
CLOSE_GOING_AWAY = 1001
CLOSE_PROTOCOL_ERROR = 1002
CLOSE_TOO_LARGE = 1009
CLOSE_INTERNAL_ERROR = 1011


def websocket_accept(key: str) -> str:
    """The RFC 6455 ``Sec-WebSocket-Accept`` value for a client key."""
    digest = hashlib.sha1(
        (key + _WS_GUID).encode("ascii"), usedforsecurity=False
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_frame(opcode: int, payload: bytes = b"") -> bytes:
    """Encode one unmasked, unfragmented server frame."""
    header = bytearray()
    header.append(0x80 | opcode)
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 1 << 16:
        header.append(126)
        header.extend(length.to_bytes(2, "big"))
    else:
        header.append(127)
        header.extend(length.to_bytes(8, "big"))
    return bytes(header) + payload


class _FrameError(Exception):
    """The peer sent a frame this server will not accept."""

    def __init__(self, message: str, code: int = CLOSE_PROTOCOL_ERROR) -> None:
        super().__init__(message)
        self.code = code


def decode_frames(buffer: bytearray, max_bytes: int) -> list[tuple[int, bytes]]:
    """Pull complete client frames out of ``buffer``, leaving any remainder.

    Client frames must be masked (RFC 6455 section 5.1); an unmasked frame,
    a fragmented stream or an oversized frame raises :class:`_FrameError`,
    so a peer cannot make the server buffer without bound.
    """
    frames: list[tuple[int, bytes]] = []
    while True:
        if len(buffer) < 2:
            return frames
        first, second = buffer[0], buffer[1]
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        offset = 2
        if length == 126:
            if len(buffer) < 4:
                return frames
            length = int.from_bytes(buffer[2:4], "big")
            offset = 4
        elif length == 127:
            if len(buffer) < 10:
                return frames
            length = int.from_bytes(buffer[2:10], "big")
            offset = 10
        if length > max_bytes:
            raise _FrameError("frame too large", CLOSE_TOO_LARGE)
        if not masked:
            raise _FrameError("client frames must be masked")
        if not fin:
            raise _FrameError("fragmented frames are not supported")
        if len(buffer) < offset + 4 + length:
            return frames
        key = bytes(buffer[offset : offset + 4])
        start = offset + 4
        payload = bytearray(buffer[start : start + length])
        for index in range(length):
            payload[index] ^= key[index % 4]
        del buffer[: start + length]
        frames.append((opcode, bytes(payload)))


# ----------------------------------------------------------------------
# The service
# ----------------------------------------------------------------------
class MeasurementService:
    """Sweep ownership, results and event fan-out, independent of HTTP.

    The methods named after protocol commands take validated plain data and
    return plain data.  The HTTP/WebSocket layer above them only parses
    requests, applies access policy and writes responses, so a GUI hook or a
    test can drive exactly the same behaviour without a socket.
    """

    def __init__(
        self,
        config: ServiceConfig | None = None,
        *,
        device_factory: DeviceFactory | None = None,
        app_version: str = "",
    ) -> None:
        self._config = config if config is not None else ServiceConfig()
        self._config.validate()
        self._factory = device_factory
        self.app_version = app_version
        self._lock = threading.RLock()
        self._runs: dict[str, _RunState] = {}
        self._order: list[str] = []
        self._live: str | None = None
        self._last_result_id: str | None = None
        self._subscribers: list[Subscriber] = []
        self._clients = 0
        self.refused_clients = 0
        self.dropped_subscribers = 0
        self._server: "_ServiceServer | None" = None
        self._thread: threading.Thread | None = None
        self._address: tuple[str, int] | None = None

    # -- lifecycle -----------------------------------------------------
    @property
    def config(self) -> ServiceConfig:
        return self._config

    @property
    def address(self) -> tuple[str, int] | None:
        """``(host, port)`` once started, else ``None``."""
        return self._address

    @property
    def port(self) -> int:
        return self._address[1] if self._address else 0

    @property
    def running(self) -> bool:
        return self._server is not None

    def start_server(self) -> tuple[str, int]:
        """Bind and serve; an explicit call, never an import side effect."""
        with self._lock:
            if self._server is not None:
                raise ConfigError("the service is already running")
            config = self._config
            if not config.loopback and not (
                config.allow_remote and config.token
            ):
                raise ConfigError(
                    f"refusing to bind {config.host!r} without"
                    " allow_remote=True and a token"
                )
            server = _ServiceServer(
                (config.host, config.port), _ServiceHandler, self
            )
            self._server = server
            self._address = (
                str(server.server_address[0]),
                int(server.server_address[1]),
            )
            thread = threading.Thread(
                target=server.serve_forever,
                kwargs={"poll_interval": 0.05},
                name="nanovna-service",
                daemon=True,
            )
            self._thread = thread
        thread.start()
        return self._address

    def stop(self, timeout: float = 5.0) -> None:
        """Cancel a live run, drop subscribers and release the port.

        Idempotent: a second call does nothing, so a ``finally`` is safe.
        """
        with self._lock:
            server, thread = self._server, self._thread
            self._server = None
            self._thread = None
            self._address = None
            live = self._runs.get(self._live) if self._live else None
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.close("server stopping")
        if live is not None:
            with self._lock:
                if live.state == STATE_RUNNING:
                    live.cancel_requested = True
                    live.state = STATE_CANCELLING
            live.cancel.cancel()
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout)
        if live is not None and live.thread is not None:
            live.thread.join(timeout)

    def __enter__(self) -> "MeasurementService":
        self.start_server()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # -- client accounting ---------------------------------------------
    def begin_client(self) -> bool:
        """Claim a client slot; False when the service is at its limit."""
        with self._lock:
            if self._clients >= self._config.max_clients:
                self.refused_clients += 1
                return False
            self._clients += 1
            return True

    def end_client(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)

    @property
    def active_clients(self) -> int:
        with self._lock:
            return self._clients

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    # -- events --------------------------------------------------------
    def subscribe(self) -> Subscriber:
        with self._lock:
            if len(self._subscribers) >= self._config.max_subscribers:
                raise TooManySubscribersError(
                    "this service already has the maximum number of event"
                    " subscribers",
                    details={"max_subscribers": self._config.max_subscribers},
                )
            subscriber = Subscriber(self._config.subscriber_queue)
            self._subscribers.append(subscriber)
            return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)
        subscriber.close("unsubscribed")

    def _publish(self, event: Mapping) -> None:
        """Fan out one event without ever blocking the caller."""
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            if subscriber.offer(event):
                continue
            with self._lock:
                if subscriber in self._subscribers:
                    self._subscribers.remove(subscriber)
                self.dropped_subscribers += 1
            LOGGER.info(
                "dropped event subscriber (%s)", subscriber.reason or "closed"
            )

    # -- commands ------------------------------------------------------
    def info(self) -> dict:
        with self._lock:
            live = self._live
            completed = sum(
                1 for run in self._runs.values() if run.state == STATE_COMPLETED
            )
            retained = len(self._runs)
        config = self._config
        return {
            "service": PROTOCOL_NAME,
            "protocol": {
                "schema_version": SCHEMA_VERSION,
                "supported_versions": list(SUPPORTED_VERSIONS),
                "app_version": self.app_version,
                "transport": "http+websocket",
            },
            "server": {
                "host": self.address[0] if self.address else config.host,
                "port": self.port,
                "loopback": config.loopback,
                "authentication_required": config.authentication_required,
                "clients": self._clients,
                "max_clients": config.max_clients,
                "refused_clients": self.refused_clients,
                "subscribers": len(self._subscribers),
                "max_subscribers": config.max_subscribers,
            },
            "device": {
                "configured": self._factory is not None,
                "open": live is not None,
            },
            "runs": {
                "live": live,
                "completed": completed,
                "retained": retained,
                "max_retained": config.max_retained_runs,
                "last_result": self._last_result_id,
            },
            "limits": {
                "max_body_bytes": config.max_body_bytes,
                "max_points": MAXIMUM_POINTS,
                "max_average": MAXIMUM_AVERAGE,
            },
            "export": {
                "enabled": bool(config.export_dir),
                "directory": (
                    os.path.abspath(config.export_dir)
                    if config.export_dir
                    else None
                ),
                "formats": ["touchstone", "csv", "html", "json"],
            },
        }

    def start(self, document: Mapping) -> dict:
        """Start one run; refuse rather than queue while another is live."""
        _reject_unknown(
            document, ("schema_version", "command", "sweep", "limits"), "start"
        )
        if "sweep" not in document:
            raise InvalidRequestError("start: sweep is required")
        spec = parse_sweep_spec(document["sweep"])
        limits = parse_limits(document.get("limits"))
        factory = self._factory
        if factory is None:
            raise DeviceUnavailableError(
                "this service was started without a device source"
            )
        with self._lock:
            if self._live is not None:
                raise RunBusyError(
                    f"a run is already in progress ({self._live});"
                    " wait for it to finish or cancel it",
                    details={"run_id": self._live},
                )
            run_id = uuid.uuid4().hex
            run = _RunState(
                run_id=run_id,
                spec=spec,
                limits=limits,
                started_at=_now(),
            )
            self._runs[run_id] = run
            self._order.append(run_id)
            self._live = run_id
            self._prune()
            thread = threading.Thread(
                target=self._execute,
                args=(run,),
                name=f"nanovna-service-run-{run_id[:8]}",
                daemon=True,
            )
            run.thread = thread
        self._publish(
            {
                "type": "started",
                "run_id": run_id,
                "sweep": spec.document(),
            }
        )
        thread.start()
        return {"run_id": run_id, "state": STATE_RUNNING}

    def cancel(self, run_id: object) -> dict:
        """Cancel the named live run; a finished run id is a stale id."""
        identifier = _require_run_id(run_id, "cancel")
        with self._lock:
            run = self._runs.get(identifier)
            if run is None:
                raise RunNotFoundError(f"no run with id {identifier!r}")
            if run.state != STATE_RUNNING:
                raise StaleRunError(
                    f"run {identifier!r} is {run.state} and cannot be"
                    " cancelled",
                    details={"run_id": identifier, "state": run.state},
                )
            run.cancel_requested = True
            run.state = STATE_CANCELLING
            run.cancel.cancel()
        self._publish({"type": "cancelling", "run_id": identifier})
        return {"run_id": identifier, "state": STATE_CANCELLING}

    def status(self, run_id: object = None) -> dict:
        """One run's state, or the whole server's state without an id."""
        if run_id is None:
            with self._lock:
                live = self._live
                completed = sum(
                    1
                    for run in self._runs.values()
                    if run.state == STATE_COMPLETED
                )
                return {
                    "state": "busy" if live else "idle",
                    "run_id": live,
                    "completed": completed,
                    "last_result": self._last_result_id,
                }
        identifier = _require_run_id(run_id, "status")
        with self._lock:
            run = self._runs.get(identifier)
            if run is None:
                raise RunNotFoundError(f"no run with id {identifier!r}")
            document = run.status_document()
            document["thread_alive"] = bool(
                run.thread is not None and run.thread.is_alive()
            )
            return document

    def last_result(self) -> dict:
        with self._lock:
            identifier = self._last_result_id
            run = self._runs.get(identifier) if identifier else None
            if run is None or run.result is None:
                raise NoResultError("no run has produced a result yet")
            return run.result.document()

    def checks(self, run_id: object) -> dict:
        identifier = _require_run_id(run_id, "checks")
        return self._result_for(identifier).checks_document()

    def export(self, document: Mapping) -> dict:
        """Write one frozen result inside the configured directory only."""
        _reject_unknown(
            document,
            (
                "schema_version",
                "command",
                "run_id",
                "format",
                "filename",
                "overwrite",
            ),
            "export",
        )
        identifier = _require_run_id(document.get("run_id"), "export")
        result = self._result_for(identifier)
        export_format = document.get("format")
        if export_format not in EXPORT_FORMATS:
            raise InvalidRequestError(
                f"export: unknown format {export_format!r};"
                f" known formats are {', '.join(EXPORT_FORMATS)}"
            )
        filename = document.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            raise InvalidRequestError(
                "export: filename must be a non-empty string"
            )
        overwrite = document.get("overwrite", False)
        if not isinstance(overwrite, bool):
            raise InvalidRequestError("export: overwrite must be true or false")
        path = self.resolve_export_path(filename)
        expected = expected_extension(export_format, result)
        if expected and not path.lower().endswith(expected):
            raise InvalidRequestError(
                f"this run's data fits a {expected} file, not"
                f" {os.path.basename(path)}"
            )
        if os.path.exists(path) and not overwrite:
            raise ExportExistsError(
                f"{path} already exists; pass overwrite to replace it",
                details={"path": path},
            )
        try:
            written = _write_export(export_format, result, path)
        except OSError as error:
            raise ExportFailedError(f"cannot write {path}: {error}") from error
        except (TouchstoneExportError, ValueError) as error:
            # The file was never written: the data does not fit the format
            # (no S11 for a Touchstone file, non-finite samples).
            raise ExportRefusedError(
                f"this run cannot be exported as {export_format}: {error}",
                details={"run_id": identifier},
            ) from error
        return {
            "run_id": identifier,
            "format": export_format,
            "path": path,
            "bytes": written,
        }

    def resolve_export_path(self, filename: str) -> str:
        """Resolve ``filename`` inside the export directory or refuse it."""
        root = self._config.export_dir
        if not root:
            raise ExportDisabledError(
                "this service was started without an export directory,"
                " so it cannot write anything"
            )
        root_real = os.path.realpath(root)
        if "\x00" in filename:
            raise PathRefusedError(
                f"refusing {filename!r}: it contains a null byte",
                details={"requested": filename, "allowed_root": root_real},
            )
        if os.path.isabs(filename):
            raise PathRefusedError(
                f"refusing the absolute path {filename!r};"
                f" only names inside {root_real} are allowed",
                details={"requested": filename, "allowed_root": root_real},
            )
        resolved = os.path.realpath(os.path.join(root_real, filename))
        try:
            inside = os.path.commonpath([root_real, resolved]) == root_real
        except ValueError:
            inside = False
        if not inside or resolved == root_real:
            raise PathRefusedError(
                f"refusing {filename!r}: it resolves to {resolved}, which is"
                f" outside the allowed export directory {root_real}",
                details={
                    "requested": filename,
                    "allowed_root": root_real,
                    "resolved": resolved,
                },
            )
        return resolved

    def wait_for_result(self, run_id: str, timeout: float = 30.0) -> dict:
        """Block until ``run_id`` finishes (or the timeout) and describe it."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                run = self._runs.get(run_id)
                if run is None:
                    raise RunNotFoundError(f"no run with id {run_id!r}")
                alive = run.thread is not None and run.thread.is_alive()
                if not alive:
                    return run.status_document()
            time.sleep(0.005)
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise RunNotFoundError(f"no run with id {run_id!r}")
            return run.status_document()

    # -- internals -----------------------------------------------------
    def _result_for(self, run_id: str) -> RunResult:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise RunNotFoundError(f"no run with id {run_id!r}")
            if run.result is None:
                raise RunIncompleteError(
                    f"run {run_id!r} is {run.state} and has no result",
                    details={"run_id": run_id, "state": run.state},
                )
            return run.result

    def _prune(self) -> None:
        """Forget old finished runs; the live one and the last result stay."""
        while len(self._order) > self._config.max_retained_runs:
            for candidate in list(self._order):
                if candidate in (self._live, self._last_result_id):
                    continue
                self._order.remove(candidate)
                self._runs.pop(candidate, None)
                break
            else:  # only the protected ids are left
                return

    def _finish(
        self,
        run: _RunState,
        *,
        state: str,
        result: RunResult | None = None,
        failure: dict | None = None,
    ) -> None:
        with self._lock:
            run.state = state
            run.finished_at = _now()
            run.result = result
            run.failure = failure
            if self._live == run.run_id:
                self._live = None
            if result is not None:
                self._last_result_id = run.run_id

    def _execute(self, run: _RunState) -> None:
        """The run body: one device, one sweep, one frozen result."""
        started = run.started_at
        try:
            device = self._factory()  # type: ignore[misc]
        except Exception as error:
            self._finish(
                run,
                state=STATE_FAILED,
                failure={
                    "code": "no_device",
                    "message": f"cannot prepare a device: {error}",
                },
            )
            self._publish_failure(run)
            return
        try:
            with open_device(device) as session:
                channels = run.spec.channels or session.supported_channels
                if not channels:
                    raise UnsupportedError(
                        "this device provides no measurable channels"
                    )
                request = run.spec.request(channels)
                session.on_progress = (
                    lambda fraction, identifier=run.run_id: self._publish(
                        {
                            "type": "progress",
                            "run_id": identifier,
                            "fraction": round(float(fraction), 4),
                        }
                    )
                )
                sweep = session.sweep(request, cancel=run.cancel)
        except CancelledError:
            self._finish(run, state=STATE_CANCELLED)
            self._publish({"type": "cancelled", "run_id": run.run_id})
            return
        except UnsupportedError as error:
            self._fail(run, "unsupported", str(error))
            return
        except AcquisitionError as error:
            self._fail(run, "device_error", str(error))
            return
        except ValueError as error:
            self._fail(run, "invalid_request", str(error))
            return
        except Exception as error:
            LOGGER.exception("service run %s failed", run.run_id)
            self._fail(run, "internal_error", str(error))
            return

        with self._lock:
            cancelled = run.cancel_requested
        if cancelled:
            # A cancel that arrived before publication wins: a cancelled run
            # never publishes data, so two clients cannot disagree about it.
            self._finish(run, state=STATE_CANCELLED)
            self._publish({"type": "cancelled", "run_id": run.run_id})
            return

        checks = (
            run.limits.evaluate(sweep.snapshot, trace_id=run.run_id)
            if run.limits is not None
            else ()
        )
        result = RunResult(
            run_id=run.run_id,
            spec=run.spec,
            channels=tuple(channels),
            snapshot=sweep.snapshot,
            limits=run.limits,
            checks=tuple(checks),
            started_at=started,
            finished_at=_now(),
        )
        self._finish(run, state=STATE_COMPLETED, result=result)
        self._publish(
            {
                "type": "completed",
                "run_id": run.run_id,
                "result": result.document(),
            }
        )

    def _fail(self, run: _RunState, code: str, message: str) -> None:
        self._finish(
            run, state=STATE_FAILED, failure={"code": code, "message": message}
        )
        self._publish_failure(run)

    def _publish_failure(self, run: _RunState) -> None:
        self._publish(
            {
                "type": "failed",
                "run_id": run.run_id,
                "error": dict(run.failure or {}),
            }
        )


def _require_run_id(value: object, command: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidRequestError(
            f"{command}: run_id must be a non-empty string naming the run"
            " this command acts on"
        )
    if len(value) > 128:
        raise InvalidRequestError(f"{command}: run_id is too long")
    return value.strip()


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------
EXPORT_FORMATS: tuple[str, ...] = ("touchstone", "csv", "html", "json")


def expected_extension(format_name: str, result: RunResult) -> str:
    """The extension a format requires, given the channels actually held."""
    if format_name == "touchstone":
        return ".s2p" if "S21" in result.channels else ".s1p"
    return {"csv": ".csv", "html": ".html", "json": ".json"}[format_name]


def _export_payload(format_name: str, result: RunResult) -> str:
    """Serialise one frozen result; nothing reads live state."""
    if format_name == "json":
        return json.dumps(result.document(), indent=2, sort_keys=True) + "\n"
    from .Reports import build_report, csv_summary, render_html
    from .Traces import Trace

    trace = Trace(
        id=result.run_id,
        name=result.snapshot.name or result.run_id,
        snapshot=result.snapshot,
    )
    model = build_report(
        f"Run {result.run_id}",
        [trace],
        app_version="",
        generated=result.finished_at,
        limits=result.limits,
    )
    return csv_summary(model) if format_name == "csv" else render_html(model)


def _touchstone_payload(result: RunResult) -> str:
    from .Measurement import snapshot_to_touchstone

    count = 4 if "S21" in result.channels else 1
    return snapshot_to_touchstone(result.snapshot).saves(count)


def _write_export(format_name: str, result: RunResult, path: str) -> int:
    """Write ``path`` atomically and return the number of bytes written."""
    from .cli import write_atomically

    text = (
        _touchstone_payload(result)
        if format_name == "touchstone"
        else _export_payload(format_name, result)
    )

    def writer(temporary: str) -> None:
        with open(temporary, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_atomically(path, writer)
    return len(text.encode("utf-8"))


# ----------------------------------------------------------------------
# HTTP transport
# ----------------------------------------------------------------------
_PATH = re.compile(r"^(?P<base>/v(?P<version>[0-9]+)/(?P<command>[a-z_]+))/?$")

_GET_COMMANDS = ("info", "status", "last_result", "checks", "events")
_POST_COMMANDS = ("start", "cancel", "export")


def split_path(path: str) -> tuple[int, str]:
    """Version and command from ``/v1/<command>``; never guesses."""
    match = _PATH.match(path)
    if match is None:
        raise UnknownCommandError(
            f"{path!r} is not a known endpoint; use /v1/<command> where"
            " command is one of"
            f" {', '.join(sorted(_GET_COMMANDS + _POST_COMMANDS))}"
        )
    version = int(match["version"])
    if version not in SUPPORTED_VERSIONS:
        supported = ", ".join(f"v{item}" for item in SUPPORTED_VERSIONS)
        raise UnsupportedVersionError(
            f"schema version {version} is not supported (supported:"
            f" {supported})",
            details={"supported_versions": list(SUPPORTED_VERSIONS)},
        )
    return version, match["command"]


class _ServiceServer(ThreadingHTTPServer):
    """Threaded server that owns exactly one service instance."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, service) -> None:
        host = str(address[0])
        self.address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        self.service = service
        super().__init__(address, handler)


class _ServiceHandler(BaseHTTPRequestHandler):
    """Parses one request, applies access policy, calls the service."""

    protocol_version = "HTTP/1.1"
    server_version = "NanoVNASaver-Service/1"
    #: Bounds an idle connection so a client cannot hold a slot forever.
    timeout = 15.0

    server: _ServiceServer
    #: Set by ``_dispatch`` once the request body has been consumed.
    _body_read = False

    # -- plumbing ------------------------------------------------------
    def log_message(self, format: str, *args) -> None:
        LOGGER.debug("http %s - %s", self.address_string(), format % args)

    def handle(self) -> None:
        service = self.server.service
        if not service.begin_client():
            self._write_raw(
                TooManyClientsError(
                    "this service already has the maximum number of clients",
                    details={"max_clients": service.config.max_clients},
                )
            )
            return
        try:
            super().handle()
        finally:
            service.end_client()

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    # -- request handling ----------------------------------------------
    def _dispatch(self, method: str) -> None:
        # The connection stays open unless a refusal says otherwise: a body
        # that was never read cannot be skipped, and a client that reuses a
        # connection would then read one response's headers as another's.
        self._body_read = False
        if method == "GET" and self._declared_body_length():
            # A GET body is ignored, but it would remain on the stream and
            # be read as the next request, so that connection cannot be
            # reused.
            self.close_connection = True
        try:
            split = urlsplit(self.path)
            version, command = split_path(split.path)
            self._check_request_origin()
            self._authorise()
            query = {
                key: values[-1] for key, values in parse_qs(split.query).items()
            }
            if command == "events":
                self._serve_events(version, query)
                return
            if command in _GET_COMMANDS:
                if method != "GET":
                    raise MethodNotAllowedError(
                        f"{command} is read-only; use GET"
                    )
                result = self._get_command(command, version, query)
            elif command in _POST_COMMANDS:
                if method != "POST":
                    raise MethodNotAllowedError(
                        f"{command} changes state; use POST"
                    )
                result = self._post_command(command, version)
            else:  # pragma: no cover - split_path already refused it
                raise UnknownCommandError(f"unknown command {command!r}")
            self._send_json(
                HTTPStatus.OK,
                {
                    "schema_version": SCHEMA_VERSION,
                    "ok": True,
                    "command": command,
                    "result": result,
                },
            )
        except ServiceError as error:
            # A POST whose body was never read leaves unread bytes on the
            # stream, so that connection must not be reused.
            if method == "POST" and not self._body_read:
                self.close_connection = True
            self._send_json(error.status, error.document(), error=error)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            # The protocol promises a structured error, so a defect must not
            # surface to the client as a dropped connection.
            LOGGER.exception("service request %s failed", self.path)
            self.close_connection = True
            error = InternalServiceError(
                "the service failed to handle this request; see its log"
            )
            self._send_json(error.status, error.document(), error=error)

    def _declared_body_length(self) -> int:
        """The announced body length, or 0 when there is none to skip."""
        if self.headers.get("Transfer-Encoding"):
            return -1
        raw = self.headers.get("Content-Length")
        if raw is None:
            return 0
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return -1

    def _check_request_origin(self) -> None:
        """Refuse what a web page in the operator's browser could send.

        Binding to loopback keeps other machines out, but not a page the
        operator happens to have open: it can POST to 127.0.0.1 without a
        preflight, or rebind its own host name to reach every endpoint.
        Browsers always name the page in ``Origin`` on those requests and
        cannot forge it, and scripts do not send it at all.
        """
        origin = self.headers.get("Origin")
        if origin is not None:
            raise ForbiddenRequestError(
                f"requests from a web page ({origin}) are not accepted;"
                " use a script or the command line"
            )
        config = self.server.service.config
        host = self.headers.get("Host")
        if config.loopback and host is not None:
            name = urlsplit(f"//{host}").hostname or ""
            if not is_loopback_host(name):
                raise ForbiddenRequestError(
                    f"this service only answers to a loopback name, not"
                    f" {host!r}"
                )

    def _check_content_type(self) -> None:
        declared = self.headers.get("Content-Type", "")
        if declared.split(";")[0].strip().lower() != "application/json":
            raise UnsupportedMediaTypeError(
                "send the request body as Content-Type: application/json,"
                f" not {declared or 'nothing'!r}"
            )

    def _authorise(self) -> None:
        expected = self.server.service.config.token
        if not expected:
            return
        header = self.headers.get("Authorization", "")
        supplied = ""
        if header.lower().startswith("bearer "):
            supplied = header[7:].strip()
        if not supplied:
            supplied = self.headers.get("X-NanoVNA-Token", "").strip()
        if not hmac.compare_digest(supplied, expected):
            raise UnauthorizedError(
                "this service requires a token (Authorization: Bearer <token>)"
            )

    def _read_body(self) -> dict:
        config = self.server.service.config
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            raise InvalidRequestError(
                "chunked request bodies are not supported; send Content-Length"
            )
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise InvalidRequestError(
                "a request body is required; send Content-Length and a JSON"
                " document"
            )
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            raise InvalidRequestError(
                f"Content-Length {raw_length!r} is not a number"
            ) from None
        if length < 0:
            raise InvalidRequestError("Content-Length must not be negative")
        if length > config.max_body_bytes:
            # Refuse before reading: an oversized body must never be
            # buffered, and the client is told the real limit.
            self.close_connection = True
            raise PayloadTooLargeError(
                f"a request body may be at most {config.max_body_bytes}"
                f" bytes; this one announced {length}",
                details={"max_body_bytes": config.max_body_bytes},
            )
        self._check_content_type()
        self.connection.settimeout(config.socket_timeout)
        raw = self.rfile.read(length)
        # The body is off the stream from here on, whatever it contained, so
        # even a rejected request leaves the connection reusable.
        self._body_read = True
        if len(raw) != length:
            raise InvalidRequestError(
                f"the request body ended after {len(raw)} of {length} bytes"
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise InvalidJsonError(f"the body is not UTF-8: {error}") from error
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise InvalidJsonError(
                f"the body is not JSON: {error.msg} at line {error.lineno}"
                f" column {error.colno}"
            ) from error
        return _require_object(document, "the request body")

    def _post_command(self, command: str, version: int) -> dict:
        document = self._read_body()
        _check_document_version(document, version)
        declared = document.get("command")
        if declared is not None and declared != command:
            raise InvalidRequestError(
                f"the body names command {declared!r} but the path names"
                f" {command!r}"
            )
        service = self.server.service
        if command == "start":
            return service.start(document)
        if command == "cancel":
            return service.cancel(document.get("run_id"))
        _reject_unknown(
            document,
            (
                "schema_version",
                "command",
                "run_id",
                "format",
                "filename",
                "overwrite",
            ),
            "export",
        )
        return service.export(document)

    def _get_command(
        self, command: str, version: int, query: Mapping[str, str]
    ) -> dict:
        _reject_unknown(
            query, ("schema_version", "run_id"), f"GET {command} parameters"
        )
        if "schema_version" in query:
            _check_version_text(query["schema_version"], version, query=True)
        service = self.server.service
        if command == "info":
            return service.info()
        if command == "status":
            return service.status(query.get("run_id"))
        if command == "last_result":
            return service.last_result()
        return service.checks(query.get("run_id"))

    # -- responses -----------------------------------------------------
    def _send_json(
        self,
        status: int,
        payload: Mapping,
        *,
        error: ServiceError | None = None,
    ) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        try:
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if error is not None and error.status == 401:
                self.send_header("WWW-Authenticate", "Bearer")
            self.send_header(
                "Connection", "close" if self.close_connection else "keep-alive"
            )
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _write_raw(self, error: ServiceError) -> None:
        """Answer a connection that is over the client limit, then close."""
        body = json.dumps(error.document(), sort_keys=True).encode("utf-8")
        try:
            reason = HTTPStatus(error.status).phrase
        except ValueError:  # pragma: no cover - every status here is known
            reason = "Error"
        head = (
            f"HTTP/1.1 {error.status} {reason}\r\n"
            "Content-Type: application/json; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("latin-1")
        try:
            self.wfile.write(head + body)
            self.wfile.flush()
        except OSError:
            pass
        self.close_connection = True

    # -- WebSocket event stream ----------------------------------------
    def _serve_events(self, version: int, query: Mapping[str, str]) -> None:
        if query.get("schema_version") is not None:
            _check_version_text(query["schema_version"], version, query=True)
        _reject_unknown(query, ("schema_version",), "GET events parameters")
        self._check_upgrade()
        service = self.server.service
        subscriber = service.subscribe()
        try:
            self._accept_upgrade()
            self._stream(subscriber)
        except OSError:
            return
        finally:
            service.unsubscribe(subscriber)

    def _check_upgrade(self) -> None:
        upgrade = (self.headers.get("Upgrade") or "").lower()
        connection = (self.headers.get("Connection") or "").lower()
        if upgrade != "websocket" or "upgrade" not in connection:
            raise InvalidRequestError(
                "this endpoint is a WebSocket; request it with"
                " Upgrade: websocket and Sec-WebSocket-Version: 13"
            )
        if (self.headers.get("Sec-WebSocket-Version") or "") != "13":
            raise InvalidRequestError("only WebSocket version 13 is supported")
        key = self.headers.get("Sec-WebSocket-Key") or ""
        try:
            decoded = base64.b64decode(key, validate=True)
        except Exception:
            decoded = b""
        if len(decoded) != 16:
            raise InvalidRequestError(
                "Sec-WebSocket-Key must be 16 base64-encoded bytes"
            )

    def _accept_upgrade(self) -> None:
        key = self.headers.get("Sec-WebSocket-Key", "")
        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", websocket_accept(key))
        self.end_headers()
        self.wfile.flush()
        self.close_connection = True

    def _stream(self, subscriber: Subscriber) -> None:
        config = self.server.service.config
        if not self._send_event(
            {
                "type": "hello",
                "schema_version": SCHEMA_VERSION,
                "service": PROTOCOL_NAME,
                "server_version": self.server_version,
                "supported_versions": list(SUPPORTED_VERSIONS),
                "max_frame_bytes": config.max_body_bytes,
                "subscriber_queue": config.subscriber_queue,
            }
        ):
            return
        buffer = bytearray()
        deadline = time.monotonic() + 1.0
        while True:
            if subscriber.closed:
                reason = subscriber.reason or "closed"
                code = (
                    CLOSE_INTERNAL_ERROR
                    if reason == "lagged"
                    else CLOSE_GOING_AWAY
                )
                self._send_close(code, reason)
                return
            event = subscriber.poll(0.1)
            if event is not None:
                if not self._send_event(event):
                    return
                continue
            if not self._drain(buffer, config):
                return
            now = time.monotonic()
            if now >= deadline:
                if not self._send_frame(OP_PING, b""):
                    return
                deadline = now + 1.0

    def _drain(self, buffer: bytearray, config: ServiceConfig) -> bool:
        """Read whatever the client sent; False means the stream is over."""
        try:
            self.connection.settimeout(0.01)
            chunk = self.connection.recv(4096)
        except TimeoutError:
            return True
        except OSError:
            return False
        finally:
            try:
                self.connection.settimeout(config.socket_timeout)
            except OSError:  # pragma: no cover - socket already gone
                return False
        if not chunk:
            return False
        buffer.extend(chunk)
        try:
            frames = decode_frames(buffer, config.max_body_bytes)
        except _FrameError as error:
            self._send_close(error.code, str(error))
            return False
        for opcode, payload in frames:
            if opcode == OP_CLOSE:
                self._send_close(CLOSE_NORMAL, "")
                return False
            if opcode == OP_PING:
                if not self._send_frame(OP_PONG, payload):
                    return False
                continue
            if opcode == OP_PONG:
                continue
            # The stream is server-to-client.  Anything else is answered
            # with a structured error event rather than ignored.
            if not self._send_event(
                {
                    "type": "error",
                    "schema_version": SCHEMA_VERSION,
                    "error": {
                        "code": "unsupported_websocket_message",
                        "message": (
                            "the event stream does not accept commands;"
                            " send requests over HTTP"
                        ),
                        "details": {
                            "bytes": len(payload),
                            "opcode": opcode,
                        },
                    },
                }
            ):
                return False
        return True

    def _send_event(self, event: Mapping) -> bool:
        return self._send_frame(OP_TEXT, json.dumps(event).encode("utf-8"))

    def _send_frame(self, opcode: int, payload: bytes) -> bool:
        try:
            self.wfile.write(encode_frame(opcode, payload))
            self.wfile.flush()
        except OSError:
            return False
        return True

    def _send_close(self, code: int, reason: str) -> None:
        body = code.to_bytes(2, "big") + reason.encode("utf-8")[:120]
        self._send_frame(OP_CLOSE, body)


def _check_version_text(value: object, version: int, *, query: bool) -> None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        where = "schema_version" if query else "the body's schema_version"
        raise InvalidRequestError(f"{where} must be an integer") from None
    if parsed != version:
        raise UnsupportedVersionError(
            f"schema_version {parsed} does not match the path version"
            f" {version}",
            details={"supported_versions": list(SUPPORTED_VERSIONS)},
        )


def _check_document_version(document: Mapping, version: int) -> None:
    if "schema_version" not in document:
        raise UnsupportedVersionError(
            "the request body must carry schema_version",
            details={"supported_versions": list(SUPPORTED_VERSIONS)},
        )
    raw = document["schema_version"]
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise InvalidRequestError("schema_version must be an integer")
    if raw not in SUPPORTED_VERSIONS:
        raise UnsupportedVersionError(
            f"schema version {raw} is not supported; this server speaks"
            f" {', '.join(str(item) for item in SUPPORTED_VERSIONS)}",
            details={"supported_versions": list(SUPPORTED_VERSIONS)},
        )
    if raw != version:
        raise UnsupportedVersionError(
            f"schema_version {raw} does not match the path version {version}"
        )


#: Frozen view used by callers that want the protocol description only.
PROTOCOL_DESCRIPTION = MappingProxyType(
    {
        "name": PROTOCOL_NAME,
        "schema_version": SCHEMA_VERSION,
        "supported_versions": SUPPORTED_VERSIONS,
        "get": _GET_COMMANDS,
        "post": _POST_COMMANDS,
    }
)


def serve_forever(
    config: ServiceConfig | None = None,
    *,
    device_factory: DeviceFactory | None = None,
) -> None:  # pragma: no cover - the GUI hook calls start_server instead
    """Blocking helper for a standalone service; never called on import."""
    service = MeasurementService(config, device_factory=device_factory)
    service.start_server()
    try:
        while service.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
