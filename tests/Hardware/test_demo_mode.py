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
"""SIM-02: demonstration mode via the normal connection workflow.

These tests drive ``SerialControl`` with a stub application, so no sweep
thread starts and no hardware or serial port is required.
"""

import importlib
import os

import pytest

serial_control = importlib.import_module("NanoVNASaver.Controls.SerialControl")
SerialControl = serial_control.SerialControl

_QT_APP = None


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

    def update_center_span(self):
        return None

    def update_step_size(self):
        return None


class FakeWindow:
    def __init__(self):
        self.vna_connected_calls = 0

    def vna_connected(self):
        self.vna_connected_calls += 1


class FakeApp:
    def __init__(self):
        self.vna = None
        self.sweep_control = FakeSweepControl()
        self.windows = {"sweep_settings": FakeWindow()}
        self.sweep_starts = 0
        self.errors = []

    def sweep_start(self):
        self.sweep_starts += 1

    def showError(self, text):
        self.errors.append(text)

    def display_window(self, _name):
        return None


def _control(qt_app, monkeypatch):
    monkeypatch.setattr(serial_control, "get_interfaces", lambda: [])
    app = FakeApp()
    control = SerialControl(app)
    emissions = []
    control.connected.connect(emissions.append)
    return control, app, emissions


class TestScenarioChoices:
    @staticmethod
    def test_scenarios_are_offered_by_name(qt_app, monkeypatch):
        control, _app, _ = _control(qt_app, monkeypatch)
        labels = [
            control.cb_scenario.itemText(index)
            for index in range(control.cb_scenario.count())
        ]
        assert labels
        assert any("Antenna" in label for label in labels)

    @staticmethod
    def test_selection_maps_to_a_scenario_name(qt_app, monkeypatch):
        control, _app, _ = _control(qt_app, monkeypatch)
        control.cb_scenario.setCurrentIndex(1)
        assert control.selected_scenario() in serial_control.SCENARIO_NAMES
        assert control.selected_scenario() != serial_control.SCENARIO_NAMES[0]

    @staticmethod
    def test_scenario_controls_only_matter_in_demo_mode(qt_app, monkeypatch):
        control, _app, _ = _control(qt_app, monkeypatch)
        assert not control.cb_simulated.isChecked()
        assert control.inp_port.isEnabled()
        control.cb_simulated.setChecked(True)
        assert control.cb_scenario.isEnabled()
        assert not control.inp_port.isEnabled()
        control.cb_simulated.setChecked(False)
        assert control.inp_port.isEnabled()

    @staticmethod
    def test_connect_button_is_usable_without_a_port_in_demo_mode(
        qt_app, monkeypatch
    ):
        control, _app, _ = _control(qt_app, monkeypatch)
        control.cb_simulated.setChecked(True)
        assert control.btn_toggle.isEnabled()


class TestSimulatedConnection:
    @staticmethod
    def test_demo_mode_connects_a_simulated_device(qt_app, monkeypatch):
        control, app, emissions = _control(qt_app, monkeypatch)
        control.cb_simulated.setChecked(True)
        control.connect_device()

        assert emissions == [True]
        assert app.vna.simulated is True
        assert app.vna.connected()
        assert app.sweep_starts == 1
        assert control.btn_toggle.text() == "Disconnect"

    @staticmethod
    def test_selected_scenario_is_used(qt_app, monkeypatch):
        control, app, _ = _control(qt_app, monkeypatch)
        control.cb_simulated.setChecked(True)
        control.cb_scenario.setCurrentIndex(1)
        expected = control.selected_scenario()
        control.connect_device()
        assert app.vna.scenario_key == expected

    @staticmethod
    def test_simulated_device_produces_data_through_the_normal_path(
        qt_app, monkeypatch
    ):
        control, app, _ = _control(qt_app, monkeypatch)
        control.cb_simulated.setChecked(True)
        control.connect_device()

        vna = app.vna
        vna.setSweep(13_000_000, 15_000_000)
        frequencies = vna.read_frequencies()
        values = vna.readValues("data 0")
        assert len(frequencies) == len(values) == vna.datapoints
        assert all(abs(value) > 0 for value in values)

    @staticmethod
    def test_no_serial_port_is_required(qt_app, monkeypatch):
        control, app, emissions = _control(qt_app, monkeypatch)

        def forbidden(_interface):
            raise AssertionError("demo mode must not open a serial port")

        monkeypatch.setattr(serial_control, "open_serial_interface", forbidden)
        monkeypatch.setattr(serial_control, "open_tcp_interface", forbidden)
        control.cb_simulated.setChecked(True)
        control.connect_device()
        assert emissions == [True]
        assert app.errors == []


class TestSwitchingBack:
    @staticmethod
    def test_disconnect_ends_the_simulated_session(qt_app, monkeypatch):
        control, app, emissions = _control(qt_app, monkeypatch)
        control.cb_simulated.setChecked(True)
        control.connect_device()
        control.disconnect_device()
        assert emissions == [True, False]
        assert not app.vna.connected()
        assert control.btn_toggle.text() == "Connect to device"

    @staticmethod
    def test_leaving_demo_mode_restores_port_controls(qt_app, monkeypatch):
        control, _app, _ = _control(qt_app, monkeypatch)
        control.cb_simulated.setChecked(True)
        control.connect_device()
        control.disconnect_device()
        control.cb_simulated.setChecked(False)
        assert control.inp_port.isEnabled()
        assert control.cb_tcp.isEnabled()
        assert not control.cb_scenario.isEnabled()


class TestIdentifiability:
    @staticmethod
    def test_simulated_device_is_self_identifying(qt_app, monkeypatch):
        control, app, _ = _control(qt_app, monkeypatch)
        control.cb_simulated.setChecked(True)
        control.connect_device()
        assert app.vna.simulated is True
        assert "simulat" in app.vna.readFirmware().lower()

    @staticmethod
    def test_firmware_text_names_the_scenario(qt_app, monkeypatch):
        control, app, _ = _control(qt_app, monkeypatch)
        control.cb_simulated.setChecked(True)
        control.connect_device()
        scenario = app.vna.scenario_name
        assert scenario in app.vna.readFirmware()
