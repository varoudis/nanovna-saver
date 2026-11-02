"""Connection handling for SerialControl (FIX-01, PERF-01).

Connecting is no longer synchronous: :meth:`SerialControl.connect_device`
starts a worker and the result is applied on the GUI thread through a
queued signal.  Every test therefore pumps the event loop with
:func:`wait_for_connection` before asserting on the published state.
"""

import importlib
import os
import time
from threading import Event, RLock
from types import SimpleNamespace

import pytest
from serial import SerialException

serial_control = importlib.import_module("NanoVNASaver.Controls.SerialControl")
connection = importlib.import_module("NanoVNASaver.Connection")
DeviceConnectionError = serial_control.DeviceConnectionError
SerialControl = serial_control.SerialControl

_QT_APP = None

#: Controls created by the tests, stopped again after each test so no
#: worker thread outlives the run.
_CREATED = []


@pytest.fixture(scope="session")
def qt_app():
    """Keep one offscreen QApplication alive and referenced."""
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
    """Never let a worker thread (or an open port) survive a test."""
    yield
    while _CREATED:
        control = _CREATED.pop()
        try:
            control.shutdown(2000)
        except RuntimeError:  # already deleted by Qt
            pass


class FakeInterface:
    """Minimal serial interface stand-in that records open/close."""

    def __init__(self, comment="NanoVNA"):
        self.lock = RLock()
        self.timeout = None
        self.port = "fake-port"
        self.comment = comment
        self.is_open = False
        self.open_calls = 0
        self.close_calls = 0
        self.open_error = None
        self.type = "serial"

    def open(self):
        self.open_calls += 1
        if self.open_error:
            raise self.open_error
        self.is_open = True

    def close(self):
        self.close_calls += 1
        self.is_open = False

    def __str__(self):
        return self.port


class FakeVNA:
    """Candidate device whose initialization is fully under test control."""

    datapoints = 101

    def __init__(self, interface, frequencies=None, error=None):
        self.serial = interface
        self.validateInput = None
        self._frequencies = (101, 202) if frequencies is None else frequencies
        self._error = error

    def read_frequencies(self):
        if self._error is not None:
            raise self._error
        return list(self._frequencies)

    def connected(self):
        return self.serial.is_open

    def disconnect(self):
        self.serial.close()


class FakeSweepControl:
    def __init__(self):
        self.calls = []
        self._segments = 1

    def set_start(self, value):
        self.calls.append(("start", value))

    def set_end(self, value):
        self.calls.append(("end", value))

    def get_segments(self):
        return self._segments

    def set_segments(self, value):
        self._segments = value
        self.calls.append(("segments", value))

    def update_center_span(self):
        self.calls.append(("center_span", None))

    def update_step_size(self):
        self.calls.append(("step_size", None))

    def value_of(self, name):
        return [value for call, value in self.calls if call == name]


class FakeSettings:
    def value(self, key, default=None, type_=None):
        return default


class FakeSweepSettingsWindow:
    def __init__(self):
        self.vna_connected_calls = 0

    def vna_connected(self):
        self.vna_connected_calls += 1


class FakeApp:
    def __init__(self, vna):
        self.vna = vna
        self.settings = FakeSettings()
        self.sweep_control = FakeSweepControl()
        self.windows = {"sweep_settings": FakeSweepSettingsWindow()}
        self.sweep_starts = 0
        self.errors = []

    def sweep_start(self):
        self.sweep_starts += 1

    def showError(self, text):
        self.errors.append(text)


def make_control(qt_app, monkeypatch, app):
    monkeypatch.setattr(serial_control, "get_interfaces", lambda: [])
    control = SerialControl(app)
    _CREATED.append(control)
    emissions = []
    control.connected.connect(emissions.append)
    return control, emissions


def select_serial_port(control, interface):
    control.inp_port.currentData = lambda: interface


def wait_for_connection(qt_app, control, timeout=10.0):
    """Pump the GUI event loop until the attempt result was applied.

    The worker publishes through a queued signal, so a plain
    ``processEvents()`` spin is what delivers it; the worker itself already
    ended when the slot runs.
    """
    deadline = time.monotonic() + timeout
    while control.is_connecting() and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.005)
    qt_app.processEvents()
    return not control.is_connecting()


def finish_rescan(qt_app, control, timeout=10.0):
    """Pump the GUI event loop until a running port scan was applied."""
    deadline = time.monotonic() + timeout
    while control.is_scanning() and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.005)
    qt_app.processEvents()
    return not control.is_scanning()


def test_get_vna_failure_does_not_publish_connection(qt_app, monkeypatch):
    previous = FakeVNA(FakeInterface())
    app = FakeApp(previous)
    control, emissions = make_control(qt_app, monkeypatch, app)
    interface = FakeInterface()
    select_serial_port(control, interface)

    def broken_get_vna(_iface):
        raise IOError("device did not answer")

    monkeypatch.setattr(serial_control, "get_VNA", broken_get_vna)

    control.connect_device()
    assert wait_for_connection(qt_app, control)

    assert emissions == []
    assert app.vna is previous
    assert interface.close_calls == 1
    assert not interface.is_open
    assert app.sweep_starts == 0
    assert control.btn_toggle.text() == "Connect to device"
    assert not control.btn_settings.isEnabled()
    assert app.errors and "device did not answer" in app.errors[0]


def test_empty_frequencies_fail_connection(qt_app, monkeypatch):
    previous = FakeVNA(FakeInterface())
    app = FakeApp(previous)
    control, emissions = make_control(qt_app, monkeypatch, app)
    interface = FakeInterface()
    select_serial_port(control, interface)
    new_vna = FakeVNA(interface, frequencies=[])
    monkeypatch.setattr(serial_control, "get_VNA", lambda _iface: new_vna)

    control.connect_device()
    assert wait_for_connection(qt_app, control)

    assert emissions == []
    assert app.vna is previous
    assert interface.close_calls == 1
    assert not interface.is_open
    assert app.sweep_control.calls == []
    assert app.sweep_starts == 0
    assert control.btn_toggle.text() == "Connect to device"
    assert app.errors and "no sweep frequencies" in app.errors[0]


def test_read_frequencies_error_is_a_connection_failure(qt_app, monkeypatch):
    previous = FakeVNA(FakeInterface())
    app = FakeApp(previous)
    control, emissions = make_control(qt_app, monkeypatch, app)
    interface = FakeInterface()
    select_serial_port(control, interface)
    new_vna = FakeVNA(interface, error=ValueError("garbage"))
    monkeypatch.setattr(serial_control, "get_VNA", lambda _iface: new_vna)

    control.connect_device()
    assert wait_for_connection(qt_app, control)

    assert emissions == []
    assert app.vna is previous
    assert interface.close_calls == 1
    assert control.btn_toggle.text() == "Connect to device"
    assert app.errors and "frequencies" in app.errors[0]


def test_malformed_frequencies_fail_without_type_error(qt_app, monkeypatch):
    app = FakeApp(FakeVNA(FakeInterface()))
    control, emissions = make_control(qt_app, monkeypatch, app)
    interface = FakeInterface()
    select_serial_port(control, interface)
    new_vna = FakeVNA(interface, frequencies=["not-a-frequency"])
    monkeypatch.setattr(serial_control, "get_VNA", lambda _iface: new_vna)

    control.connect_device()
    assert wait_for_connection(qt_app, control)

    assert emissions == []
    assert interface.close_calls == 1
    assert app.errors


def test_single_frequency_still_configures_sweep(qt_app, monkeypatch):
    app = FakeApp(FakeVNA(FakeInterface()))
    control, emissions = make_control(qt_app, monkeypatch, app)
    interface = FakeInterface()
    select_serial_port(control, interface)
    new_vna = FakeVNA(interface, frequencies=[5_000_000])
    monkeypatch.setattr(serial_control, "get_VNA", lambda _iface: new_vna)

    control.connect_device()
    assert wait_for_connection(qt_app, control)

    assert emissions == [True]
    assert app.vna is new_vna
    assert app.sweep_control.value_of("start") == [5_000_000]
    # single (and descending) responses must not index past the ends
    assert app.sweep_control.value_of("end") == [5_000_000 + 101]
    assert app.sweep_starts == 1


def test_successful_connect_publishes_exactly_once(qt_app, monkeypatch):
    previous = FakeVNA(FakeInterface())
    app = FakeApp(previous)
    control, emissions = make_control(qt_app, monkeypatch, app)
    interface = FakeInterface()
    select_serial_port(control, interface)
    new_vna = FakeVNA(interface, frequencies=[1_000_000, 30_000_000])
    monkeypatch.setattr(serial_control, "get_VNA", lambda _iface: new_vna)

    control.connect_device()
    assert wait_for_connection(qt_app, control)

    assert emissions == [True]
    assert app.vna is new_vna
    assert new_vna.validateInput is False
    assert interface.is_open
    assert control.btn_toggle.text() == "Disconnect"
    assert control.btn_settings.isEnabled()
    assert not control.btn_rescan.isEnabled()
    assert app.sweep_control.value_of("start") == [1_000_000]
    assert app.sweep_control.value_of("end") == [30_000_000]
    assert app.windows["sweep_settings"].vna_connected_calls == 1
    assert app.sweep_starts == 1
    assert app.errors == []


def test_retry_after_failure_succeeds(qt_app, monkeypatch):
    previous = FakeVNA(FakeInterface())
    app = FakeApp(previous)
    control, emissions = make_control(qt_app, monkeypatch, app)

    attempts = []

    def flaky_get_vna(iface):
        if not attempts:
            attempts.append(iface)
            raise IOError("first attempt fails")
        created = FakeVNA(iface, frequencies=[1_000_000, 2_000_000])
        attempts.append(created)
        return created

    monkeypatch.setattr(serial_control, "get_VNA", flaky_get_vna)

    first = FakeInterface()
    select_serial_port(control, first)
    control.connect_device()
    assert wait_for_connection(qt_app, control)

    assert emissions == []
    assert app.vna is previous
    assert first.close_calls == 1

    second = FakeInterface()
    select_serial_port(control, second)
    control.connect_device()
    assert wait_for_connection(qt_app, control)

    assert emissions == [True]
    assert app.vna is attempts[-1]
    assert app.vna is not previous
    assert second.is_open
    assert control.btn_toggle.text() == "Disconnect"
    assert app.sweep_starts == 1


def test_tcp_comment_failure_closes_socket(qt_app, monkeypatch):
    app = FakeApp(FakeVNA(FakeInterface()))
    control, emissions = make_control(qt_app, monkeypatch, app)
    socket_interface = FakeInterface(comment=None)
    control.cb_tcp.setChecked(True)
    control.inp_port.setEditText("127.0.0.1:5000")

    monkeypatch.setattr(
        connection,
        "serial",
        SimpleNamespace(
            serial_for_url=lambda url, do_not_open=True: socket_interface
        ),
    )

    def broken_comment(_iface):
        raise ValueError("no answer")

    monkeypatch.setattr(serial_control, "get_comment", broken_comment)

    control.connect_device()
    assert wait_for_connection(qt_app, control)

    assert emissions == []
    assert socket_interface.close_calls == 1
    assert not socket_interface.is_open
    assert app.errors
    assert control.btn_toggle.text() == "Connect to device"


def test_tcp_success_uses_comment_to_identify_device(qt_app, monkeypatch):
    previous = FakeVNA(FakeInterface())
    app = FakeApp(previous)
    control, emissions = make_control(qt_app, monkeypatch, app)
    socket_interface = FakeInterface(comment=None)
    control.cb_tcp.setChecked(True)
    control.inp_port.setEditText("127.0.0.1:5000")

    monkeypatch.setattr(
        connection,
        "serial",
        SimpleNamespace(
            serial_for_url=lambda url, do_not_open=True: socket_interface
        ),
    )
    monkeypatch.setattr(serial_control, "get_comment", lambda _iface: "NanoVNA")
    new_vna = FakeVNA(socket_interface, frequencies=[1_000_000, 2_000_000])
    monkeypatch.setattr(serial_control, "get_VNA", lambda _iface: new_vna)

    control.connect_device()
    assert wait_for_connection(qt_app, control)

    assert emissions == [True]
    assert app.vna is new_vna
    assert socket_interface.comment == "NanoVNA"
    assert socket_interface.is_open
    # opened for the comment probe, closed, and opened again for the device
    assert socket_interface.open_calls == 2


def test_open_serial_interface_failure_is_reported():
    interface = FakeInterface()
    interface.open_error = IOError("port busy")

    with pytest.raises(DeviceConnectionError) as error:
        serial_control.open_serial_interface(interface)

    assert "port busy" in str(error.value)
    assert interface.close_calls == 1


def test_initialize_vna_closes_interface_on_identify_failure(monkeypatch):
    app = FakeApp(FakeVNA(FakeInterface()))
    interface = FakeInterface()

    monkeypatch.setattr(
        serial_control,
        "get_VNA",
        lambda _iface: (_ for _ in ()).throw(IOError("boom")),
    )

    with pytest.raises(DeviceConnectionError):
        serial_control.initialize_vna(app, interface)

    assert interface.close_calls == 1
    assert app.vna.connected() is False


def test_validate_frequencies_normalizes_and_rejects_bad_values():
    assert serial_control.validate_frequencies([1]) == [1]
    assert serial_control.validate_frequencies([1.6, 2]) == [1, 2]
    with pytest.raises(DeviceConnectionError):
        serial_control.validate_frequencies([])
    with pytest.raises(DeviceConnectionError):
        serial_control.validate_frequencies([-1])
    with pytest.raises(DeviceConnectionError):
        serial_control.validate_frequencies(["not-a-frequency"])
    with pytest.raises(DeviceConnectionError):
        serial_control.validate_frequencies([True])


def test_failure_while_preparing_sweep_keeps_disconnected(qt_app, monkeypatch):
    previous = FakeVNA(FakeInterface())
    app = FakeApp(previous)
    control, emissions = make_control(qt_app, monkeypatch, app)
    interface = FakeInterface()
    select_serial_port(control, interface)
    monkeypatch.setattr(
        serial_control,
        "get_VNA",
        lambda _iface: FakeVNA(interface, frequencies=[1_000_000, 2_000_000]),
    )

    def broken_range(*_args):
        raise RuntimeError("sweep control exploded")

    monkeypatch.setattr(serial_control, "apply_frequency_range", broken_range)

    control.connect_device()
    assert wait_for_connection(qt_app, control)

    assert emissions == []
    assert app.vna is previous
    assert interface.close_calls == 1
    assert app.sweep_starts == 0
    assert control.btn_toggle.text() == "Connect to device"
    assert app.errors and "sweep control exploded" in app.errors[0]


def test_interface_that_fails_to_open_reports_and_stays_disconnected(
    qt_app, monkeypatch
):
    previous = FakeVNA(FakeInterface())
    app = FakeApp(previous)
    control, emissions = make_control(qt_app, monkeypatch, app)
    interface = FakeInterface()
    interface.open_error = IOError("port busy")
    select_serial_port(control, interface)
    monkeypatch.setattr(
        serial_control,
        "get_VNA",
        lambda _iface: pytest.fail("get_VNA must not run"),
    )

    control.connect_device()
    assert wait_for_connection(qt_app, control)

    assert emissions == []
    assert app.vna is previous
    assert interface.close_calls == 1
    assert control.btn_toggle.text() == "Connect to device"
    assert app.errors and "port busy" in app.errors[0]


def test_connect_device_returns_before_the_device_answers(qt_app, monkeypatch):
    """The whole point of PERF-01: no blocking conversation on the GUI."""
    app = FakeApp(FakeVNA(FakeInterface()))
    control, _emissions = make_control(qt_app, monkeypatch, app)
    interface = FakeInterface()
    select_serial_port(control, interface)
    release = Event()

    def slow_get_vna(_iface):
        release.wait(10)
        return FakeVNA(interface, frequencies=[1_000_000, 2_000_000])

    monkeypatch.setattr(serial_control, "get_VNA", slow_get_vna)

    started = time.monotonic()
    control.connect_device()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert control.is_connecting()
    release.set()
    assert wait_for_connection(qt_app, control)
    assert app.vna is not None


def test_connecting_state_disables_conflicting_controls(qt_app, monkeypatch):
    app = FakeApp(FakeVNA(FakeInterface()))
    control, emissions = make_control(qt_app, monkeypatch, app)
    interface = FakeInterface()
    select_serial_port(control, interface)
    release = Event()

    def slow_get_vna(_iface):
        release.wait(10)
        return FakeVNA(interface, frequencies=[1_000_000, 2_000_000])

    monkeypatch.setattr(serial_control, "get_VNA", slow_get_vna)

    control.connect_device()
    try:
        assert control.is_connecting()
        assert control.btn_toggle.text() == serial_control.CONNECTING_LABEL
        assert not control.btn_toggle.isEnabled()
        assert not control.inp_port.isEnabled()
        assert not control.cb_tcp.isEnabled()
        assert not control.btn_rescan.isEnabled()
        assert not control.cb_simulated.isEnabled()
        # A rescan while the device is being claimed would open the same port.
        assert control.rescanSerialPort() is False
        # A second connect request must not start a second conversation.
        control.connect_device()
        assert control.is_connecting()
    finally:
        release.set()
    assert wait_for_connection(qt_app, control)
    assert emissions == [True]
    assert control.btn_toggle.text() == "Disconnect"
    assert control.inp_port.isEnabled()


def test_cancel_closes_the_port_and_never_publishes(qt_app, monkeypatch):
    previous = FakeVNA(FakeInterface())
    app = FakeApp(previous)
    control, emissions = make_control(qt_app, monkeypatch, app)
    interface = FakeInterface()
    select_serial_port(control, interface)
    release = Event()

    def slow_get_vna(_iface):
        release.wait(10)
        return FakeVNA(interface, frequencies=[1_000_000, 2_000_000])

    monkeypatch.setattr(serial_control, "get_VNA", slow_get_vna)

    statuses = []
    control.connection_finished.connect(statuses.append)

    control.connect_device()
    deadline = time.monotonic() + 10
    while not interface.is_open and time.monotonic() < deadline:
        time.sleep(0.005)
    assert interface.is_open
    assert control.is_connecting()
    control.cancel_connection()
    assert not control.is_connecting()
    # The port the worker opened was closed to unblock it, exactly once.
    assert interface.close_calls == 1
    assert not interface.is_open

    release.set()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if control._service.worker is not None and (
            control._service.worker.isFinished()
        ):
            break
        time.sleep(0.005)
    qt_app.processEvents()

    assert emissions == []
    assert statuses == []
    assert app.vna is previous
    assert app.sweep_starts == 0
    assert control.btn_toggle.text() == "Connect to device"


def test_shutdown_mid_connect_stops_the_worker(qt_app, monkeypatch):
    app = FakeApp(FakeVNA(FakeInterface()))
    control, emissions = make_control(qt_app, monkeypatch, app)
    interface = FakeInterface()
    select_serial_port(control, interface)
    release = Event()

    def slow_get_vna(_iface):
        release.wait(5)
        raise IOError("device unplugged")

    monkeypatch.setattr(serial_control, "get_VNA", slow_get_vna)

    control.connect_device()
    deadline = time.monotonic() + 10
    while not interface.is_open and time.monotonic() < deadline:
        time.sleep(0.005)
    assert interface.is_open
    assert control.is_connecting()
    release.set()
    assert control.shutdown(3000) is True
    qt_app.processEvents()

    assert emissions == []
    assert app.errors == []
    assert interface.close_calls == 1
    assert not interface.is_open


def test_unplugged_device_reports_and_restores_previous_device(
    qt_app, monkeypatch
):
    previous = FakeVNA(FakeInterface())
    app = FakeApp(previous)
    control, emissions = make_control(qt_app, monkeypatch, app)
    interface = FakeInterface()
    select_serial_port(control, interface)
    new_vna = FakeVNA(
        interface, error=SerialException("device reports readiness to read")
    )
    monkeypatch.setattr(serial_control, "get_VNA", lambda _iface: new_vna)

    control.connect_device()
    assert wait_for_connection(qt_app, control)

    assert app.vna is previous
    assert emissions == []
    assert interface.close_calls == 1
    assert not interface.is_open
    assert control.btn_toggle.text() == "Connect to device"
    assert app.errors and "frequencies" in app.errors[0]


def test_second_connect_while_connecting_is_ignored(qt_app, monkeypatch):
    app = FakeApp(FakeVNA(FakeInterface()))
    control, emissions = make_control(qt_app, monkeypatch, app)
    interface = FakeInterface()
    select_serial_port(control, interface)
    release = Event()
    calls = []

    def slow_get_vna(_iface):
        calls.append(1)
        release.wait(10)
        return FakeVNA(interface, frequencies=[1_000_000, 2_000_000])

    monkeypatch.setattr(serial_control, "get_VNA", slow_get_vna)

    control.connect_device()
    control.connect_device()
    control.serialButtonClick()
    release.set()
    assert wait_for_connection(qt_app, control)

    assert calls == [1]
    assert emissions == [True]


def test_rescan_is_non_blocking_and_applies_its_result(qt_app, monkeypatch):
    app = FakeApp(FakeVNA(FakeInterface()))
    monkeypatch.setattr(serial_control, "get_interfaces", lambda: [])
    control = SerialControl(app)
    _CREATED.append(control)

    first = FakeInterface(comment="NanoVNA V2")
    first.port = "/dev/ttyACM0"
    second = FakeInterface(comment="NanoVNA-H")
    second.port = "/dev/ttyACM1"

    def slow_scan():
        time.sleep(0.2)
        return [first, second]

    monkeypatch.setattr(serial_control, "get_interfaces", slow_scan)

    started = time.monotonic()
    assert control.rescanSerialPort() is True
    elapsed = time.monotonic() - started

    assert elapsed < 0.1
    assert control.is_scanning()
    assert control.btn_rescan.text() == "Scanning"
    assert not control.btn_rescan.isEnabled()
    # A second click must not start a second scan over the same ports.
    assert control.rescanSerialPort() is False

    assert finish_rescan(qt_app, control)
    assert control.inp_port.count() == 2
    assert control.btn_rescan.text() == "Rescan"
    assert control.btn_rescan.isEnabled()


def test_rescan_keeps_the_selected_port(qt_app, monkeypatch):
    app = FakeApp(FakeVNA(FakeInterface()))
    monkeypatch.setattr(serial_control, "get_interfaces", lambda: [])
    control = SerialControl(app)
    _CREATED.append(control)

    first = FakeInterface(comment="NanoVNA-H")
    first.port = "/dev/ttyACM0"
    second = FakeInterface(comment="NanoVNA-H")
    second.port = "/dev/ttyACM1"
    monkeypatch.setattr(
        serial_control, "get_interfaces", lambda: [first, second]
    )

    assert control.rescanSerialPort() is True
    assert finish_rescan(qt_app, control)
    control.inp_port.setCurrentIndex(control.inp_port.findData(second))
    assert control.inp_port.currentData() is second

    assert control.rescanSerialPort() is True
    assert finish_rescan(qt_app, control)
    assert control.inp_port.count() == 2
    assert control.inp_port.currentData() is second


def test_failed_rescan_keeps_the_previous_port_list(qt_app, monkeypatch):
    app = FakeApp(FakeVNA(FakeInterface()))
    monkeypatch.setattr(serial_control, "get_interfaces", lambda: [])
    control = SerialControl(app)
    _CREATED.append(control)

    interface = FakeInterface(comment="NanoVNA-H")
    monkeypatch.setattr(serial_control, "get_interfaces", lambda: [interface])
    assert control.rescanSerialPort() is True
    assert finish_rescan(qt_app, control)
    assert control.inp_port.count() == 1

    def broken_scan():
        raise OSError("udev went away")

    monkeypatch.setattr(serial_control, "get_interfaces", broken_scan)
    assert control.rescanSerialPort() is True
    assert finish_rescan(qt_app, control)

    assert control.inp_port.count() == 1
    assert control.btn_rescan.text() == "Rescan"
    assert control.btn_rescan.isEnabled()


def test_startup_enumeration_does_not_open_any_port(qt_app, monkeypatch):
    monkeypatch.setattr(
        serial_control,
        "list_serial_ports",
        lambda: [_unprobed("NanoVNA-H", "/dev/ttyACM0")],
    )
    monkeypatch.setattr(
        serial_control,
        "get_interfaces",
        lambda: pytest.fail("startup must not probe the ports"),
    )
    app = FakeApp(FakeVNA(FakeInterface()))
    control = SerialControl(app)
    _CREATED.append(control)

    assert control.inp_port.count() == 1
    assert control.inp_port.currentData().port == "/dev/ttyACM0"
    assert control.inp_port.currentData().probed is False


def test_connecting_probes_deferred_port_comment(qt_app, monkeypatch):
    """A port listed at startup is identified once, inside the worker."""
    app = FakeApp(FakeVNA(FakeInterface()))
    monkeypatch.setattr(serial_control, "get_interfaces", lambda: [])
    control = SerialControl(app)
    _CREATED.append(control)

    interface = _unprobed("", "fake-port")
    select_serial_port(control, interface)
    probed = []

    def probe(iface):
        probed.append(iface)
        return "NanoVNA-H"

    monkeypatch.setattr(serial_control, "get_comment", probe)
    monkeypatch.setattr(
        serial_control,
        "get_VNA",
        lambda iface: FakeVNA(iface, frequencies=[1_000_000, 2_000_000]),
    )

    control.connect_device()
    assert wait_for_connection(qt_app, control)

    assert probed == [interface]
    assert interface.comment == "NanoVNA-H"
    assert interface.probed is True
    assert app.vna.serial is interface


def _unprobed(comment, port):
    interface = FakeInterface(comment=comment)
    interface.port = port
    interface.probed = False
    return interface


class _PortInfo:
    """Minimal ``serial.tools.list_ports`` entry."""

    def __init__(self, device, vid=0x0483, pid=0x5740):
        self.device = device
        self.vid = vid
        self.pid = pid
        self.hwid = f"USB VID:PID={vid:04X}:{pid:04X}"


def test_list_serial_ports_never_opens_a_port(monkeypatch):
    hardware = importlib.import_module("NanoVNASaver.Hardware.Hardware")
    monkeypatch.setattr(
        hardware.list_ports,
        "comports",
        lambda: [_PortInfo("/dev/ttyACM0"), _PortInfo("/dev/ttyACM1")],
    )
    opened = []

    class NeverOpened(FakeInterface):
        def open(self):  # pragma: no cover - must not be called
            opened.append(self.port)
            raise AssertionError("enumeration must not open a port")

    monkeypatch.setattr(
        hardware, "get_comment", lambda iface: opened.append(iface.port)
    )
    monkeypatch.setattr(
        hardware, "Interface", lambda kind, name: NeverOpened(name)
    )

    ports = hardware.list_serial_ports()

    assert [iface.port for iface in ports] == [
        "/dev/ttyACM0",
        "/dev/ttyACM1",
    ]
    assert all(iface.probed is False for iface in ports)
    assert opened == []


def test_scan_budget_stops_probing_and_leaves_ports_closed(monkeypatch):
    hardware = importlib.import_module("NanoVNASaver.Hardware.Hardware")
    monkeypatch.setattr(
        hardware.list_ports,
        "comports",
        lambda: [_PortInfo("/dev/ttyACM0")],
    )
    created = []

    class SlowOpen(FakeInterface):
        def open(self):
            super().open()
            time.sleep(0.05)

    monkeypatch.setattr(
        hardware,
        "Interface",
        lambda kind, name: created.append(SlowOpen(name)) or created[-1],
    )
    monkeypatch.setattr(hardware, "get_comment", lambda iface: "NanoVNA-H")

    found = hardware.get_interfaces(budget=0.01)

    assert found == []
    assert created
    assert all(iface.is_open is False for iface in created)
    assert all(iface.close_calls == 1 for iface in created)
