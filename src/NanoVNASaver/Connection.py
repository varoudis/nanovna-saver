#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020  Rune B. Broberg
#  Copyright (C) 2020,2021 NanoVNA-Saver Authors
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
"""Device connection conversation, off the GUI thread (PERF-01).

Connecting to a NanoVNA is a long blocking conversation: the port is
opened, the firmware is queried (``get_comment``/``get_VNA``: several
``drain_serial`` passes, ``RETRIES`` retries and, on generation two
firmware, ``NanoVNA_V2._read_version``'s hard-coded ``sleep(2.0)`` three
times), then the sweep frequencies are read.  Doing that on the GUI thread
freezes the window for seconds (minutes in the worst case, see
``docs``-free notes in :mod:`NanoVNASaver.Controls.SerialControl`).

This module owns that conversation:

* :class:`ConnectionWorker` - a ``QThread`` that performs exactly one
  attempt and publishes an immutable :class:`ConnectionOutcome`;
* :class:`ConnectionService` - the single owner of a worker.  It refuses a
  second concurrent attempt, bounds the attempt with a timeout (closing the
  partially opened interface so a stuck device read cannot outlive it) and
  lets a caller cancel.

Nothing here touches widgets: results are delivered through Qt signals and
the GUI thread decides what to publish.  Every outcome carries the token of
the attempt that produced it, so an outcome that arrives after the attempt
was cancelled or superseded can be recognised and discarded (the same
protocol as :meth:`NanoVNASaver.SweepWorker.SweepWorker.bind_next_run`).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock, RLock
from time import monotonic, sleep

import serial
from PySide6.QtCore import QObject, QThread, Signal

from .Hardware.Hardware import get_comment, get_VNA

logger = logging.getLogger(__name__)

#: Upper bound for one attempt.  A generation one device identifies in well
#: under a second; a generation two device needs about three seconds per
#: version register plus its firmware query.  The bound exists so a dead or
#: wedged port cannot leave "Connecting ..." on screen forever.
DEFAULT_CONNECT_TIMEOUT = 60.0

#: Settling delay after the port was opened, matching the delay the GUI used
#: to sleep before it started talking to a freshly opened device.
SETTLE_DELAY = 0.1

#: Granularity of the interruptible waits (seconds).
POLL_INTERVAL = 0.01

#: Outcome statuses.
CONNECTED = "connected"
FAILED = "failed"
CANCELLED = "cancelled"
TIMED_OUT = "timed out"

#: Status of a port scan that enumerated the ports.
SCANNED = "scanned"


class DeviceConnectionError(Exception):
    """A connection attempt failed; the message is meant for the user."""


class ConnectionBusyError(RuntimeError):
    """A connection attempt is already owned; a second one is refused."""


class _AttemptCancelledError(Exception):
    """Internal: the attempt was cancelled."""


class _AttemptTimedOutError(Exception):
    """Internal: the attempt ran past its timeout."""


# ----------------------------------------------------------------------
# Interface helpers (moved here so the worker can open and close them)
# ----------------------------------------------------------------------
def close_quietly(interface) -> None:
    """Close ``interface`` ignoring errors; safe for ``None``."""
    if interface is None:
        return
    try:
        interface.close()
    except Exception:
        logger.debug("Unable to close %s", interface, exc_info=True)


def open_tcp_interface(
    address: str, *, comment_probe: Callable | None = None
) -> object:
    """Open a TCP interface for address, closing it again on any failure."""
    probe = comment_probe or _hardware_comment
    url = f"socket://{address}"
    logger.info("Connection TCP %s", url)
    try:
        interface = serial.serial_for_url(url, do_not_open=True)
        interface.type = "network"
        interface.lock = RLock()
        interface.timeout = 0.05
        interface.open()
        interface.comment = probe(interface)
        interface.close()
        interface.open()
    except Exception as exc:
        close_quietly(interface)
        raise DeviceConnectionError(
            f"Unable to connect to {address}: {exc}"
        ) from exc
    return interface


def open_serial_interface(interface) -> object:
    """Open the selected serial interface, closing it on any failure."""
    logger.info("Connection %s", interface)
    try:
        interface.open()
        is_open = interface.is_open
    except Exception as exc:
        close_quietly(interface)
        raise DeviceConnectionError(
            f"Unable to open {interface}: {exc}"
        ) from exc
    if not is_open:
        close_quietly(interface)
        raise DeviceConnectionError(f"Unable to open port {interface}")
    interface.timeout = 0.05
    return interface


def _hardware_comment(interface) -> str:
    """Late-bound default so callers can patch the lookup."""
    return get_comment(interface)


def _hardware_get_vna(interface):
    """Late-bound default so callers can patch the lookup."""
    return get_VNA(interface)


def validate_frequencies(frequencies) -> list[int]:
    """Return the frequencies as integers, rejecting unusable responses."""
    if not frequencies:
        raise DeviceConnectionError(
            "The device reported no sweep frequencies. Check that the "
            "correct port and connection type is selected, then try again."
        )
    normalized = []
    for frequency in frequencies:
        if isinstance(frequency, bool) or not isinstance(
            frequency, (int, float)
        ):
            raise DeviceConnectionError(
                "The device reported an invalid sweep frequency "
                f"({frequency!r}). Check the device firmware and connection."
            )
        if frequency < 0:
            raise DeviceConnectionError(
                "The device reported a negative sweep frequency "
                f"({frequency!r}). Check the device firmware and connection."
            )
        normalized.append(int(frequency))
    return normalized


def identify_device(
    interface,
    get_vna: Callable | None = None,
    *,
    validate_input: bool = False,
    comment_probe: Callable | None = None,
) -> tuple[object, list[int]]:
    """Identify the device on an open interface and read its frequencies.

    Blocking; callers run this off the GUI thread.  A port that was only
    enumerated (:func:`list_serial_ports`) still has to be asked what it is,
    so the firmware comment is read here when ``interface.probed`` is
    false.  Any failure raises :class:`DeviceConnectionError` carrying a
    user facing message; the interface is closed by the caller (the
    worker), which also owns it.
    """
    creator = get_vna or _hardware_get_vna
    stage = "identify the connected device"
    try:
        if not getattr(interface, "probed", True):
            # Enumerated but never opened: ask the firmware what it is.
            probe = comment_probe or _hardware_comment
            interface.comment = probe(interface)
            interface.probed = True
        vna = creator(interface)
        vna.validateInput = validate_input
        stage = "read the sweep frequencies from the connected device"
        frequencies = validate_frequencies(list(vna.read_frequencies()))
    except DeviceConnectionError:
        raise
    except Exception as exc:
        raise DeviceConnectionError(f"Unable to {stage}: {exc}") from exc
    return vna, frequencies


# ----------------------------------------------------------------------
# Request / outcome
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ConnectionRequest:
    """Everything one attempt needs, resolved on the GUI thread.

    ``open`` and ``identify`` are callables so the worker never has to know
    about widgets, settings or the control that started it.  Both are
    invoked in the worker thread; anything they need must already be
    captured here.
    """

    open: Callable[[], object]
    identify: Callable[[object], tuple[object, list[int]]]
    note: str = ""
    settle: float = SETTLE_DELAY
    timeout: float = DEFAULT_CONNECT_TIMEOUT


@dataclass(frozen=True)
class ConnectionOutcome:
    """Immutable result of one attempt.

    ``token`` identifies the attempt that produced it; a listener compares
    it with the token it bound before applying anything.
    """

    token: object
    status: str = FAILED
    note: str = ""
    message: str = ""
    interface: object = None
    vna: object = None
    frequencies: tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == CONNECTED

    @property
    def cancelled(self) -> bool:
        return self.status in {CANCELLED, TIMED_OUT}

    @property
    def timed_out(self) -> bool:
        return self.status == TIMED_OUT


@dataclass(frozen=True)
class ScanOutcome:
    """Immutable result of one port scan.

    ``interfaces`` is only meaningful for :data:`SCANNED`; a failed or
    cancelled scan never carries a partial list, so a listener can apply the
    result unconditionally without publishing half a scan.
    """

    token: object
    status: str = FAILED
    message: str = ""
    interfaces: tuple[object, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == SCANNED


class ConnectionSignals(QObject):
    """Signals of one attempt (emitted from the worker thread).

    ``succeeded``/``failed``/``finished`` are the public conversation;
    ``attempt_finished`` carries the whole immutable outcome including the
    correlation token and is therefore the authoritative one.
    """

    #: ``(vna, frequencies, note)`` after a fully successful attempt.
    succeeded = Signal(object, object, str)
    #: One user facing message when the attempt failed.
    failed = Signal(str)
    #: Emitted once per attempt, after ``succeeded``/``failed``.
    finished = Signal()
    #: Emitted once per attempt with the :class:`ConnectionOutcome`.
    attempt_finished = Signal(object)


class ScanSignals(QObject):
    """Signals of one port scan (emitted from the worker thread)."""

    #: Emitted once per scan with the :class:`ScanOutcome`, token included.
    results = Signal(object)
    #: Emitted once the scan thread has left its run loop, so a listener can
    #: restore its "scanning" state even if the result arrived earlier.
    finished = Signal()


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------
class ConnectionWorker(QThread):
    """Performs one whole blocking connection conversation."""

    def __init__(
        self,
        request: ConnectionRequest,
        token: object = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._request = request
        self._token = object() if token is None else token
        self._active_token: object = None
        self.signals = ConnectionSignals()
        self._cancel = Event()
        self._expired = Event()
        self._lock = Lock()
        self._interface: object = None
        self._closed = False

    # -- caller API ----------------------------------------------------
    def bind_next_run(self, token: object) -> None:
        """Correlate the next :meth:`run_once` with ``token``."""
        self._token = token

    @property
    def request_token(self) -> object:
        """Token of the attempt being performed (``None`` when idle)."""
        return self._active_token

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def timed_out(self) -> bool:
        return self._expired.is_set()

    def cancel(self) -> None:
        """Ask the attempt to stop and close whatever it opened.

        Safe to call from any thread: closing the interface is what makes a
        blocked ``read`` on a real device return.
        """
        logger.debug("Connection attempt cancel requested")
        self._abort()

    def expire(self) -> None:
        """Cancel the attempt because it exceeded its timeout."""
        logger.warning("Connection attempt exceeded its timeout")
        self._expired.set()
        self._abort()

    # -- QThread -------------------------------------------------------
    def run(self) -> None:
        self.run_once()

    def run_once(self) -> ConnectionOutcome:
        """Synchronous entry point used by the service and by tests."""
        self._active_token = self._token
        self._token = None
        try:
            outcome = self._attempt()
            self._publish(outcome)
            return outcome
        finally:
            self._active_token = None

    # -- internals -----------------------------------------------------
    def _attempt(self) -> ConnectionOutcome:
        request = self._request
        deadline = (
            monotonic() + request.timeout
            if request.timeout and request.timeout > 0
            else None
        )
        interface = None
        try:
            self._check(deadline)
            interface = request.open()
            self._track(interface)
            self._settle(request.settle, deadline)
            self._check(deadline)
            vna, frequencies = request.identify(interface)
            self._check(deadline)
            return ConnectionOutcome(
                token=self._active_token,
                status=CONNECTED,
                note=request.note,
                interface=interface,
                vna=vna,
                frequencies=tuple(validate_frequencies(frequencies)),
            )
        except _AttemptTimedOutError:
            self._close(interface)
            return self._failure(
                TIMED_OUT,
                f"Connecting to the device timed out after "
                f"{request.timeout:g} s. Check the port and the device, then"
                " try again.",
            )
        except _AttemptCancelledError:
            self._close(interface)
            return self._failure(CANCELLED, "Connection attempt cancelled.")
        except DeviceConnectionError as exc:
            self._close(interface)
            return self._failure(FAILED, str(exc))
        except Exception as exc:  # never let a worker die silently
            logger.exception("Unexpected connection failure")
            self._close(interface)
            return self._failure(
                FAILED, f"Unable to identify the connected device: {exc}"
            )

    def _failure(self, status: str, message: str) -> ConnectionOutcome:
        return ConnectionOutcome(
            token=self._active_token,
            status=status,
            note=self._request.note,
            message=message,
        )

    def _publish(self, outcome: ConnectionOutcome) -> None:
        if outcome.ok:
            self.signals.succeeded.emit(
                outcome.vna, outcome.frequencies, outcome.note
            )
        else:
            self.signals.failed.emit(outcome.message)
        self.signals.finished.emit()
        self.signals.attempt_finished.emit(outcome)

    def _check(self, deadline: float | None) -> None:
        if self._expired.is_set():
            raise _AttemptTimedOutError()
        if self._cancel.is_set():
            raise _AttemptCancelledError()
        if deadline is not None and monotonic() > deadline:
            self._expired.set()
            raise _AttemptTimedOutError()

    def _settle(self, delay: float, deadline: float | None) -> None:
        end = monotonic() + max(float(delay), 0.0)
        while True:
            self._check(deadline)
            remaining = end - monotonic()
            if remaining <= 0:
                return
            sleep(min(remaining, POLL_INTERVAL))

    def _track(self, interface) -> None:
        with self._lock:
            self._interface = interface
            self._closed = False
        self._check(None)

    def _close(self, interface) -> None:
        """Close ``interface`` at most once, whichever path got here first."""
        with self._lock:
            if interface is None or self._closed:
                return
            self._closed = True
            self._interface = None
        close_quietly(interface)

    def _abort(self) -> None:
        """Set the abort flags and close the interface if there is one."""
        self._cancel.set()
        with self._lock:
            interface = self._interface
            close = interface is not None and not self._closed
            if close:
                self._closed = True
                self._interface = None
        if close:
            close_quietly(interface)


# ----------------------------------------------------------------------
# Service: the single owner of an attempt
# ----------------------------------------------------------------------
class ConnectionService(QObject):
    """Owns at most one :class:`ConnectionWorker` at a time.

    ``begin`` refuses while an attempt is still running, ``cancel`` asks the
    running attempt to stop and ``shutdown`` waits for it (used when the
    window closes mid-attempt).  Outcomes of an attempt that was cancelled
    or superseded are dropped here, so a stale result can never reach a
    newer attempt's listener.
    """

    #: Milliseconds a caller may wait for a worker to stop.
    DEFAULT_STOP_TIMEOUT_MS = 5000

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.signals = ConnectionSignals()
        self._worker: ConnectionWorker | None = None
        self._pending_token: object = None
        self._timer: threading.Timer | None = None

    # -- state ---------------------------------------------------------
    @property
    def busy(self) -> bool:
        worker = self._worker
        return worker is not None and not worker.isFinished()

    @property
    def pending_token(self) -> object:
        return self._pending_token

    @property
    def worker(self) -> ConnectionWorker | None:
        """The current (or last) worker; exposed for diagnostics and tests."""
        return self._worker

    # -- lifecycle -----------------------------------------------------
    def begin(self, request: ConnectionRequest, token: object = None) -> object:
        """Start one attempt and return its correlation token."""
        if self.busy:
            raise ConnectionBusyError(
                "A connection attempt is already in progress."
            )
        self._release_finished()
        token = object() if token is None else token
        worker = ConnectionWorker(request, token)
        worker.signals.attempt_finished.connect(self._on_attempt_finished)
        worker.finished.connect(self._on_thread_finished)
        self._worker = worker
        self._pending_token = token
        if request.timeout and request.timeout > 0:
            timer = threading.Timer(
                request.timeout, self._on_timeout, (worker,)
            )
            timer.daemon = True
            self._timer = timer
            timer.start()
        logger.info("Starting connection attempt (%s)", request.note or "-")
        worker.start()
        return token

    def cancel(self) -> None:
        """Ask the running attempt to stop; returns immediately.

        The pending token is dropped first, so the outcome of the cancelled
        attempt is discarded instead of being published as a result nobody
        is waiting for any more (the same rule the port scan follows).
        """
        self._pending_token = None
        worker = self._worker
        if worker is not None and not worker.isFinished():
            logger.info("Cancelling connection attempt")
            worker.cancel()

    def wait(self, timeout_ms: int = DEFAULT_STOP_TIMEOUT_MS) -> bool:
        """Wait for the current worker to leave its run loop."""
        worker = self._worker
        if worker is None:
            return True
        if worker.isFinished():
            return True
        return bool(worker.wait(int(timeout_ms)))

    def shutdown(self, timeout_ms: int = DEFAULT_STOP_TIMEOUT_MS) -> bool:
        """Cancel, wait and discard: used when the window closes."""
        self._stop_timer()
        worker = self._worker
        self._pending_token = None
        if worker is None:
            return True
        stopped = True
        if not worker.isFinished():
            worker.cancel()
            stopped = bool(worker.wait(int(timeout_ms)))
            if not stopped:
                logger.warning(
                    "Connection worker did not stop within %d ms",
                    timeout_ms,
                )
        return stopped

    # -- worker slots --------------------------------------------------
    def _on_attempt_finished(self, outcome: ConnectionOutcome) -> None:
        self._stop_timer()
        if outcome.token is not self._pending_token:
            logger.info("Discarding outcome of a superseded connection attempt")
            return
        self._pending_token = None
        # Authoritative, token correlated result first; the convenience
        # signals follow so a listener of those sees the applied state.
        self.signals.attempt_finished.emit(outcome)
        if outcome.ok:
            self.signals.succeeded.emit(
                outcome.vna, outcome.frequencies, outcome.note
            )
        else:
            self.signals.failed.emit(outcome.message)
        self.signals.finished.emit()

    def _on_thread_finished(self) -> None:
        # The worker is kept referenced until the next attempt starts (or
        # shutdown runs) so a finishing thread is never deleted underfoot.
        logger.debug("Connection worker finished")

    def _on_timeout(self, worker: ConnectionWorker) -> None:
        if worker.isFinished():
            return
        worker.expire()

    # -- helpers -------------------------------------------------------
    def _stop_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def _release_finished(self) -> None:
        worker = self._worker
        if worker is None or not worker.isFinished():
            return
        try:
            worker.signals.attempt_finished.disconnect(
                self._on_attempt_finished
            )
            worker.finished.disconnect(self._on_thread_finished)
        except (RuntimeError, TypeError):
            logger.debug("Worker connections already gone", exc_info=True)
        worker.deleteLater()
        self._worker = None


# ----------------------------------------------------------------------
# Port scan: the same conversation protocol for pure enumeration
# ----------------------------------------------------------------------
class ScanWorker(QThread):
    """Enumerates the serial ports off the GUI thread.

    Enumeration probes candidate ports (open, ask the firmware, close), so
    it is hardware bound even though it publishes nothing but a list.  The
    token protocol matches :class:`ConnectionWorker`.
    """

    def __init__(
        self,
        scan: Callable[[], object],
        token: object = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._scan = scan
        self._token = object() if token is None else token
        self._active_token: object = None
        self._cancelled = Event()
        self.signals = ScanSignals()

    # -- caller API ----------------------------------------------------
    def bind_next_run(self, token: object) -> None:
        self._token = token

    @property
    def request_token(self) -> object:
        return self._active_token

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        """Ask the scan to stop; a running port probe still has to return."""
        self._cancelled.set()

    # -- QThread -------------------------------------------------------
    def run(self) -> None:
        self.run_once()

    def run_once(self) -> ScanOutcome:
        """Synchronous entry point used by the service and by tests."""
        self._active_token = self._token
        self._token = None
        try:
            outcome = self._scan_once()
            self.signals.results.emit(outcome)
            return outcome
        finally:
            self._active_token = None

    def _scan_once(self) -> ScanOutcome:
        if self._cancelled.is_set():
            return ScanOutcome(token=self._active_token, status=CANCELLED)
        try:
            interfaces = tuple(self._scan() or ())
        except Exception as exc:
            logger.exception("Unable to enumerate the serial ports")
            return ScanOutcome(
                token=self._active_token,
                status=FAILED,
                message=f"Unable to list the serial ports: {exc}",
            )
        if self._cancelled.is_set():
            return ScanOutcome(token=self._active_token, status=CANCELLED)
        return ScanOutcome(
            token=self._active_token,
            status=SCANNED,
            interfaces=interfaces,
        )


class PortScanService(QObject):
    """Owns at most one :class:`ScanWorker` at a time.

    Scans are cheap compared to a connection attempt but they still open
    ports, so one runs at a time and a cancelled scan's result is dropped
    instead of repopulating the port list behind the user's back.
    """

    #: Milliseconds a caller may wait for a scan to stop.
    DEFAULT_STOP_TIMEOUT_MS = 2000

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.signals = ScanSignals()
        self._worker: ScanWorker | None = None
        self._pending_token: object = None

    # -- state ---------------------------------------------------------
    @property
    def busy(self) -> bool:
        worker = self._worker
        return worker is not None and not worker.isFinished()

    @property
    def pending_token(self) -> object:
        return self._pending_token

    @property
    def worker(self) -> ScanWorker | None:
        return self._worker

    # -- lifecycle -----------------------------------------------------
    def begin(self, scan: Callable[[], object], token: object = None) -> object:
        """Start one scan and return its correlation token."""
        if self.busy:
            raise ConnectionBusyError("A port scan is already in progress.")
        self._release_finished()
        token = object() if token is None else token
        worker = ScanWorker(scan, token)
        worker.signals.results.connect(self._on_results)
        worker.finished.connect(self._on_thread_finished)
        self._worker = worker
        self._pending_token = token
        logger.debug("Starting serial port scan")
        worker.start()
        return token

    def cancel(self) -> None:
        """Ask the running scan to stop; returns immediately."""
        worker = self._worker
        self._pending_token = None
        if worker is not None and not worker.isFinished():
            worker.cancel()

    def wait(self, timeout_ms: int = DEFAULT_STOP_TIMEOUT_MS) -> bool:
        worker = self._worker
        if worker is None or worker.isFinished():
            return True
        return bool(worker.wait(int(timeout_ms)))

    def shutdown(self, timeout_ms: int = DEFAULT_STOP_TIMEOUT_MS) -> bool:
        """Cancel and wait for the scan; used when the window closes."""
        worker = self._worker
        self.cancel()
        if worker is None or worker.isFinished():
            return True
        stopped = bool(worker.wait(int(timeout_ms)))
        if not stopped:
            # Detach instead of letting the owning widget delete a thread
            # that is still probing a port.
            logger.warning("Port scan did not stop within %d ms", timeout_ms)
            worker.setParent(None)
            _ABANDONED_WORKERS.append(worker)
        return stopped

    # -- worker slots --------------------------------------------------
    def _on_results(self, outcome: ScanOutcome) -> None:
        if outcome.token is not self._pending_token:
            logger.info("Discarding a superseded serial port scan")
            return
        self._pending_token = None
        self.signals.results.emit(outcome)

    def _on_thread_finished(self) -> None:
        logger.debug("Serial port scan finished")
        self.signals.finished.emit()

    # -- helpers -------------------------------------------------------
    def _release_finished(self) -> None:
        worker = self._worker
        if worker is None or not worker.isFinished():
            return
        try:
            worker.signals.results.disconnect(self._on_results)
            worker.finished.disconnect(self._on_thread_finished)
        except (RuntimeError, TypeError):
            logger.debug("Scan connections already gone", exc_info=True)
        worker.deleteLater()
        self._worker = None


#: Threads that outlived their owner; a reference keeps PySide from
#: destroying a running QThread (only when a scan overruns ``shutdown``).
_ABANDONED_WORKERS: list[QThread] = []

__all__ = [
    "CANCELLED",
    "CONNECTED",
    "DEFAULT_CONNECT_TIMEOUT",
    "FAILED",
    "SCANNED",
    "SETTLE_DELAY",
    "TIMED_OUT",
    "ConnectionBusyError",
    "ConnectionOutcome",
    "ConnectionRequest",
    "ConnectionService",
    "ConnectionSignals",
    "ConnectionWorker",
    "DeviceConnectionError",
    "PortScanService",
    "ScanOutcome",
    "ScanSignals",
    "ScanWorker",
    "close_quietly",
    "identify_device",
    "open_serial_interface",
    "open_tcp_interface",
    "validate_frequencies",
]
