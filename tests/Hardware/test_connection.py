"""Connection failure handling for SerialControl (FIX-01)."""

import importlib
import os
from threading import RLock

import pytest

serial_control = importlib.import_module("NanoVNASaver.Controls.SerialControl")
DeviceConnectionError = serial_control.DeviceConnectionError
SerialControl = serial_control.SerialControl

_QT_APP = None


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
    emissions = []
    control.connected.connect(emissions.append)
    return control, emissions


def select_serial_port(control, interface):
    control.inp_port.currentData = lambda: interface


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

    assert emissions == []
    assert app.vna is previous
    assert first.close_calls == 1

    second = FakeInterface()
    select_serial_port(control, second)
    control.connect_device()

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
        serial_control.serial,
        "serial_for_url",
        lambda url, do_not_open=True: socket_interface,
    )

    def broken_comment(_iface):
        raise ValueError("no answer")

    monkeypatch.setattr(serial_control, "get_comment", broken_comment)

    control.connect_device()

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
        serial_control.serial,
        "serial_for_url",
        lambda url, do_not_open=True: socket_interface,
    )
    monkeypatch.setattr(serial_control, "get_comment", lambda _iface: "NanoVNA")
    new_vna = FakeVNA(socket_interface, frequencies=[1_000_000, 2_000_000])
    monkeypatch.setattr(serial_control, "get_VNA", lambda _iface: new_vna)

    control.connect_device()

    assert emissions == [True]
    assert app.vna is new_vna
    assert socket_interface.comment == "NanoVNA"
    assert socket_interface.is_open
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

    assert emissions == []
    assert app.vna is previous
    assert interface.close_calls == 1
    assert control.btn_toggle.text() == "Connect to device"
    assert app.errors and "port busy" in app.errors[0]
