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
"""SESSION-03 / TRACE-02.1 / CAL-02.4: experiment boundaries.

Three defects reported together, all about state that leaked between
actions:

* starting a new session kept the previous session's captured traces;
* importing a Touchstone file as a trace used a settings field that does not
  exist and a ``read()`` method that does not exist, so it always crashed;
* the calibration status label issued a serial command on every redraw,
  freezing the GUI thread during a sweep.
"""

import os

import pytest

from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.RFTools import Datapoint

_APP = None


@pytest.fixture(scope="session")
def qt_app():
    global _APP  # noqa: PLW0603 - one application per interpreter
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    if _APP is None:
        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


def _fill(window, count=5, value=0.4):
    window.data.s11 = [
        Datapoint(13_000_000 + index * 1_000, value, 0.1)
        for index in range(count)
    ]
    window.data.s21 = []


class _Errors(list):
    """Collects reported errors; the real ``showError`` is a modal dialog."""

    def __init__(self, owner):
        super().__init__()
        self._owner = owner

    def __call__(self, text):
        self.append(text)


@pytest.fixture
def window(qt_app, monkeypatch):
    from NanoVNASaver.NanoVNASaver import NanoVNASaver

    win = NanoVNASaver()
    win.errors = _Errors(win)
    monkeypatch.setattr(win, "showError", win.errors)
    yield win
    win.worker.quit()
    win.deleteLater()


class TestNewSessionStartsEmpty:
    @staticmethod
    def test_new_session_clears_captured_traces(window, monkeypatch):
        _fill(window)
        window.trace_control.capture_current()
        window.trace_control.capture_current()
        assert window.trace_control.collection.ordered_ids()

        monkeypatch.setattr(window, "_confirm_discard", lambda: True)
        window.action_new_session()

        assert window.trace_control.collection.ordered_ids() == []
        assert window.trace_control.list_traces.count() == 0

    @staticmethod
    def test_new_session_clears_displayed_data(window, monkeypatch):
        _fill(window)
        monkeypatch.setattr(window, "_confirm_discard", lambda: True)
        window.action_new_session()
        assert window.data.s11 == []
        assert window.data.s21 == []

    @staticmethod
    def test_new_session_is_not_dirty(window, monkeypatch):
        _fill(window)
        window.mark_session_changed("sweep")
        monkeypatch.setattr(window, "_confirm_discard", lambda: True)
        window.action_new_session()
        assert not window.session.dirty

    @staticmethod
    def test_new_session_clears_remembered_sweep(window, monkeypatch):
        _fill(window)
        window._remember_complete_sweep()
        assert window._last_complete is not None
        monkeypatch.setattr(window, "_confirm_discard", lambda: True)
        window.action_new_session()
        assert window._last_complete is None
        assert window._live_snapshot is None

    @staticmethod
    def test_new_session_clears_the_source_label(window, monkeypatch):
        _fill(window)
        window.sweepSource = "previous experiment"
        monkeypatch.setattr(window, "_confirm_discard", lambda: True)
        window.action_new_session()
        assert window.sweepSource == ""

    @staticmethod
    def test_cancelling_the_confirmation_changes_nothing(window, monkeypatch):
        _fill(window)
        window.trace_control.capture_current()
        captured = window.trace_control.collection.ordered_ids()

        monkeypatch.setattr(window, "_confirm_discard", lambda: False)
        window.action_new_session()

        assert window.trace_control.collection.ordered_ids() == captured
        assert window.data.s11


class TestTraceImport:
    def test_a_touchstone_file_becomes_a_trace(self, window, tmp_path):
        path = tmp_path / "one.s1p"
        path.write_text(
            "# Hz S RI R 50\n1000000 0.5 0.1\n2000000 0.4 0.2\n",
            encoding="utf-8",
        )
        assert window.trace_control.import_path(str(path)) is True
        trace_id = window.trace_control.collection.ordered_ids()[-1]
        trace = window.trace_control.collection.get(trace_id)
        assert trace.name == "one.s1p"
        assert len(trace.channel("S11")) == 2

    def test_a_two_port_file_imports_both_channels(self, window, tmp_path):
        path = tmp_path / "two.s2p"
        path.write_text(
            "# Hz S RI R 50\n"
            "1000000 0.5 0.1 0.1 0.0 0.1 0.0 0.5 0.1\n"
            "2000000 0.4 0.2 0.1 0.0 0.1 0.0 0.4 0.2\n",
            encoding="utf-8",
        )
        assert window.trace_control.import_path(str(path)) is True
        trace_id = window.trace_control.collection.ordered_ids()[-1]
        trace = window.trace_control.collection.get(trace_id)
        assert trace.has_channel("S11")
        assert trace.has_channel("S21")

    def test_importing_does_not_need_a_save_directory_setting(
        self, window, tmp_path
    ):
        """The dialog read ``app.app_config.save_directory``: no such field."""
        assert not hasattr(window, "app_config")
        path = tmp_path / "one.s1p"
        path.write_text("# Hz S RI R 50\n1000000 0.5 0.1\n", encoding="utf-8")
        assert window.trace_control.import_path(str(path)) is True

    def test_an_empty_file_adds_no_trace(self, window, tmp_path):
        path = tmp_path / "empty.s1p"
        path.write_text("! nothing here\n", encoding="utf-8")
        before = window.trace_control.collection.ordered_ids()
        assert window.trace_control.import_path(str(path)) is False
        assert window.trace_control.collection.ordered_ids() == before
        assert window.errors

    def test_a_missing_file_adds_no_trace(self, window, tmp_path):
        before = window.trace_control.collection.ordered_ids()
        assert (
            window.trace_control.import_path(str(tmp_path / "nope.s1p"))
            is False
        )
        assert window.trace_control.collection.ordered_ids() == before
        assert window.errors

    def test_import_marks_the_session_dirty(self, window, tmp_path):
        path = tmp_path / "one.s1p"
        path.write_text("# Hz S RI R 50\n1000000 0.5 0.1\n", encoding="utf-8")
        window.session.dirty = False
        window.trace_control.import_path(str(path))
        assert window.session.dirty


class TestCalibrationStatusDoesNotDoIO:
    @staticmethod
    def test_reading_the_label_issues_no_device_command(window):
        calls = []
        window.vna.getCalibration = lambda: calls.append(1) or "none"
        window.vna.connected = lambda: True

        for _ in range(10):
            window._correction_source_label()

        assert calls == [], "the label queried the device on a redraw"

    @staticmethod
    def test_calibration_state_is_refreshed_once_per_connection(window):
        calls = []
        window.vna.getCalibration = lambda: calls.append(1) or "none"
        window.vna.connected = lambda: True

        window._on_connection_changed(True)
        assert len(calls) == 1
        for _ in range(5):
            window.refreshCoverage()
        assert len(calls) == 1

    @staticmethod
    def test_a_device_reported_calibration_is_labelled_device(window):
        window.vna.getCalibration = lambda: "cal: 0"
        window.vna.connected = lambda: True
        window.refresh_device_calibration_state()
        assert window._correction_source_label() == "device"

    @staticmethod
    def test_an_empty_device_calibration_is_labelled_application(window):
        window.vna.getCalibration = lambda: "none"
        window.vna.connected = lambda: True
        window.refresh_device_calibration_state()
        assert window._correction_source_label() == "application"

    @staticmethod
    def test_a_disconnected_device_is_labelled_unknown(window):
        window.vna.connected = lambda: False
        window.refresh_device_calibration_state()
        assert window._correction_source_label() == "unknown"

    @staticmethod
    def test_an_incomplete_calibration_does_not_break_the_label(window):
        window.calibration = Calibration()
        window.vna.connected = lambda: False
        window.refresh_device_calibration_state()
        assert window._correction_source_label() in (
            "unknown",
            "application",
            "device",
        )
