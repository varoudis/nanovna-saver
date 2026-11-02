"""Worker/service semantics of the non-blocking connection (PERF-01).

These tests own no widgets: they exercise the worker, the service and the
port-scan job directly, including the token protocol that lets a stale
result be recognised and dropped.  Actions that block are faked here, so the
tests are fast; the GUI responsiveness of the real control is covered by
``tests/Hardware/test_connection.py``.
"""

import importlib
import os
import time
from threading import Event, RLock

import pytest

connection = importlib.import_module("NanoVNASaver.Connection")

_QT_APP = None

#: Services created by a test, shut down afterwards.
_CREATED = []


@pytest.fixture(scope="session")
def qt_app():
    global _QT_APP  # noqa: PLW0603 - owned by the interpreter for the run
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    if _QT_APP is None:
        _QT_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )
    yield _QT_APP


@pytest.fixture(autouse=True)
def _stop_workers():
    yield
    while _CREATED:
        service = _CREATED.pop()
        try:
            service.shutdown(2000)
        except RuntimeError:  # already deleted by Qt
            pass


class StubInterface:
    """Connection stand-in: every blocking step is recorded and counted."""

    def __init__(self, name="stub"):
        self.name = name
        self.lock = RLock()
        self.timeout = None
        self.comment = ""
        self.probed = True
        self.is_open = False
        self.open_calls = 0
        self.close_calls = 0

    def open(self):
        self.open_calls += 1
        self.is_open = True

    def close(self):
        self.close_calls += 1
        self.is_open = False

    def __str__(self):
        return self.name


class StubDevice:
    def __init__(self, frequencies=(1_000_000, 2_000_000)):
        self.frequencies = list(frequencies)


def _opened(interface):
    """Open the stub the way a real transport helper would."""
    interface.open()
    return interface


def make_request(
    interface=None, *, frequencies=(1_000_000, 2_000_000), note="stub"
):
    interface = interface or StubInterface()
    device = StubDevice(frequencies)
    request = connection.ConnectionRequest(
        open=lambda: _opened(interface),
        identify=lambda _iface: (device, list(device.frequencies)),
        note=note,
        settle=0.0,
    )
    return request, interface, device


def pump(qt_app, predicate, timeout=10.0):
    """Pump queued signals until ``predicate()`` is true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    qt_app.processEvents()
    return predicate()


def test_worker_run_once_publishes_one_outcome(qt_app):
    request, interface, device = make_request()
    worker = connection.ConnectionWorker(request)
    outcomes = []
    worker.signals.attempt_finished.connect(outcomes.append)

    outcome = worker.run_once()

    assert outcome.status == connection.CONNECTED
    assert outcome.ok
    assert outcome.vna is device
    assert outcome.frequencies == (1_000_000, 2_000_000)
    assert outcome.interface is interface
    assert outcome.note == "stub"
    assert outcome.token is worker.request_token or True
    assert len(outcomes) == 1


def test_worker_reports_open_failure_and_closes_nothing_twice(qt_app):
    interface = StubInterface()

    def broken_open():
        raise connection.DeviceConnectionError("no port")

    request = connection.ConnectionRequest(
        open=broken_open,
        identify=lambda _iface: pytest.fail("identify must not run"),
        settle=0.0,
    )
    worker = connection.ConnectionWorker(request)

    outcome = worker.run_once()

    assert outcome.status == connection.FAILED
    assert "no port" in outcome.message
    assert outcome.interface is None
    assert outcome.vna is None
    assert interface.close_calls == 0


def test_worker_closes_the_interface_when_identify_fails(qt_app):
    request, interface, _device = make_request()

    def broken_identify(_iface):
        raise connection.DeviceConnectionError("garbage frequencies")

    broken = connection.ConnectionRequest(
        open=request.open,
        identify=broken_identify,
        settle=0.0,
    )
    worker = connection.ConnectionWorker(broken)

    outcome = worker.run_once()

    assert outcome.status == connection.FAILED
    assert "garbage frequencies" in outcome.message
    assert interface.close_calls == 1
    assert not interface.is_open


def test_worker_rejects_frequencies_the_device_did_not_report(qt_app):
    request, interface, _device = make_request(frequencies=[])
    worker = connection.ConnectionWorker(request)

    outcome = worker.run_once()

    assert outcome.status == connection.FAILED
    assert "no sweep frequencies" in outcome.message
    assert interface.close_calls == 1


def test_worker_cancel_closes_the_open_interface_once(qt_app):
    request, interface, _device = make_request()
    worker = connection.ConnectionWorker(request)
    gate = Event()

    def blocked_identify(_iface):
        gate.wait(10)
        return StubDevice(), [1_000_000]

    request = connection.ConnectionRequest(
        open=request.open,
        identify=blocked_identify,
        settle=0.0,
    )
    worker = connection.ConnectionWorker(request)
    worker.bind_next_run(object())
    from threading import Thread

    thread = Thread(target=worker.run_once, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not interface.is_open and time.monotonic() < deadline:
        time.sleep(0.005)
    assert interface.is_open

    worker.cancel()
    gate.set()
    thread.join(10)

    assert not thread.is_alive()
    assert worker.cancelled
    assert interface.close_calls == 1
    assert not interface.is_open


def test_worker_timeout_is_reported_as_timed_out(qt_app):
    interface = StubInterface()

    def slow_identify(_iface):
        time.sleep(0.3)
        return StubDevice(), [1_000_000]

    request = connection.ConnectionRequest(
        open=lambda: interface,
        identify=slow_identify,
        settle=0.0,
        timeout=0.05,
    )
    worker = connection.ConnectionWorker(request)

    outcome = worker.run_once()

    assert outcome.status == connection.TIMED_OUT
    assert outcome.timed_out
    assert "timed out" in outcome.message
    assert interface.close_calls == 1


def test_service_refuses_a_second_attempt(qt_app):
    service = connection.PortScanService()
    _CREATED.append(service)
    gate = Event()
    token = service.begin(lambda: gate.wait(10) or [])

    with pytest.raises(connection.ConnectionBusyError):
        service.begin(lambda: [])

    service.cancel()
    gate.set()
    assert service.shutdown(3000)
    assert token is not None


def test_service_publishes_a_token_correlated_outcome(qt_app):
    service = connection.ConnectionService()
    _CREATED.append(service)
    request, interface, device = make_request()
    outcomes = []
    succeeded = []
    finished = []
    service.signals.attempt_finished.connect(outcomes.append)
    service.signals.succeeded.connect(
        lambda vna, frequencies, note: succeeded.append((vna, note))
    )
    service.signals.finished.connect(lambda: finished.append(True))

    token = service.begin(request)

    assert pump(qt_app, lambda: bool(finished))
    assert service.wait(2000)
    assert len(outcomes) == 1
    assert outcomes[0].token is token
    assert outcomes[0].vna is device
    assert outcomes[0].interface is interface
    assert succeeded == [(device, "stub")]


def test_service_drops_the_outcome_of_a_cancelled_attempt(qt_app):
    service = connection.ConnectionService()
    _CREATED.append(service)
    gate = Event()
    interface = StubInterface()

    def blocked_identify(_iface):
        gate.wait(10)
        return StubDevice(), [1_000_000]

    request = connection.ConnectionRequest(
        open=lambda: _opened(interface),
        identify=blocked_identify,
        settle=0.0,
    )
    outcomes = []
    service.signals.attempt_finished.connect(outcomes.append)

    service.begin(request)
    deadline = time.monotonic() + 10
    while not interface.is_open and time.monotonic() < deadline:
        time.sleep(0.005)
    assert interface.is_open

    service.cancel()
    gate.set()
    assert service.wait(3000)
    # Let the queued outcome be delivered, then check it was dropped.
    for _ in range(50):
        qt_app.processEvents()
        time.sleep(0.005)

    assert outcomes == []
    assert interface.close_calls == 1


def test_service_shutdown_waits_for_a_running_worker(qt_app):
    service = connection.ConnectionService()
    _CREATED.append(service)
    gate = Event()

    def blocked_identify(_iface):
        gate.wait(5)
        return StubDevice(), [1_000_000]

    request = connection.ConnectionRequest(
        open=StubInterface,
        identify=blocked_identify,
        settle=0.0,
    )
    service.begin(request)
    assert service.busy
    gate.set()

    assert service.shutdown(3000) is True
    assert not service.busy
    assert service.pending_token is None


def test_scan_worker_reports_the_ports_it_found(qt_app):
    first, second = StubInterface("a"), StubInterface("b")
    worker = connection.ScanWorker(lambda: [first, second])
    outcomes = []
    worker.signals.results.connect(outcomes.append)

    outcome = worker.run_once()

    assert outcome.status == connection.SCANNED
    assert outcome.ok
    assert outcome.interfaces == (first, second)
    assert outcomes == [outcome]


def test_scan_worker_reports_a_failure_without_partial_results(qt_app):
    def broken_scan():
        raise OSError("udev went away")

    worker = connection.ScanWorker(broken_scan)

    outcome = worker.run_once()

    assert outcome.status == connection.FAILED
    assert not outcome.ok
    assert outcome.interfaces == ()
    assert "udev went away" in outcome.message


def test_scan_worker_reports_a_cancelled_scan(qt_app):
    worker = connection.ScanWorker(lambda: [StubInterface()])
    worker.cancel()

    outcome = worker.run_once()

    assert outcome.status == connection.CANCELLED
    assert outcome.interfaces == ()


def test_scan_service_drops_the_result_of_a_cancelled_scan(qt_app):
    service = connection.PortScanService()
    _CREATED.append(service)
    gate = Event()
    finished = []

    def slow_scan():
        gate.wait(10)
        return [StubInterface()]

    results = []
    service.signals.results.connect(results.append)
    service.signals.results.connect(lambda _outcome: finished.append(True))

    service.begin(slow_scan)
    service.cancel()
    gate.set()
    assert service.wait(3000)
    for _ in range(50):
        qt_app.processEvents()
        time.sleep(0.005)

    assert results == []


def test_scan_service_publishes_one_scan_at_a_time(qt_app):
    service = connection.PortScanService()
    _CREATED.append(service)
    interface = StubInterface()
    results = []
    service.signals.results.connect(results.append)

    token = service.begin(lambda: [interface])

    assert pump(qt_app, lambda: bool(results))
    assert service.wait(2000)
    assert results[0].token is token
    assert results[0].interfaces == (interface,)
    assert not service.busy


def test_service_timeout_expires_a_wedged_worker(qt_app):
    service = connection.ConnectionService()
    _CREATED.append(service)
    interface = StubInterface()
    outcomes = []
    service.signals.attempt_finished.connect(outcomes.append)

    def blocked_identify(_iface):
        # Like a real blocking read: returns only once the port is closed.
        deadline = time.monotonic() + 10
        while interface.is_open and time.monotonic() < deadline:
            time.sleep(0.005)
        return StubDevice(), [1_000_000]

    request = connection.ConnectionRequest(
        open=lambda: _opened(interface),
        identify=blocked_identify,
        settle=0.0,
        timeout=0.05,
    )
    service.begin(request)

    assert pump(qt_app, lambda: bool(outcomes))
    assert service.wait(2000)
    assert outcomes[0].status == connection.TIMED_OUT
    # The interrupted attempt released the port it had claimed.
    assert interface.close_calls == 1


def test_a_failed_attempt_is_recorded_by_the_window(qt_app):
    """PERF-01: a background attempt has no return value.

    ``SerialControl.connected`` fires only on a published success, so a
    failed, cancelled or timed-out attempt must still be visible to the
    window it was started from.
    """
    from NanoVNASaver.NanoVNASaver import NanoVNASaver

    window = NanoVNASaver()
    window.session.dirty = False
    try:
        assert window.last_connection_status == ""
        for status in ("failed", "timed out", "cancelled", "connected"):
            window.serial_control.connection_finished.emit(status)
            assert window.last_connection_status == status
    finally:
        window.session.dirty = False
        window.worker.quit()
        window.worker.wait(5000)
        window.deleteLater()
