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
"""SESSION-01/03: a saved session really brings the traces back.

Written after a real defect: saving only serialized the currently displayed
sweep, so reopening a session produced an empty trace list and the captured
traces were lost.
"""

import os

import pytest

from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Measurement import snapshot_from_datapoints
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


def _sweep_points(scenario="antenna", points=51):
    device = make_simulated_vna(scenario)
    device.datapoints = points
    device.setSweep(13_000_000, 15_000_000)
    frequencies = device.read_frequencies()
    s11_values = device.readValues("data 0")
    s21_values = [] if scenario == "antenna" else device.readValues("data 1")
    s11 = [
        Datapoint(freq, value.real, value.imag)
        for freq, value in zip(frequencies, s11_values, strict=True)
    ]
    s21 = (
        [
            Datapoint(freq, value.real, value.imag)
            for freq, value in zip(frequencies, s21_values, strict=True)
        ]
        if s21_values
        else None
    )
    return s11, s21


@pytest.fixture
def window(qt_app):
    from NanoVNASaver.NanoVNASaver import NanoVNASaver

    win = NanoVNASaver()
    yield win
    win.worker.quit()
    win.deleteLater()


class TestSessionTraceRoundTrip:
    @staticmethod
    def test_captured_traces_survive_save_and_reopen(window, tmp_path):
        """Four named traces must come back, SESSION-01.4."""
        for index, scenario in enumerate(
            ("antenna", "filter", "cable", "attenuator")
        ):
            s11, s21 = _sweep_points(scenario)
            window.data.s11 = s11
            window.data.s21 = s21 or []
            window.data.simulated = True
            window.data.simulated_scenario = scenario
            window.trace_control.capture_current()
            window.trace_control.collection.rename(
                window.trace_control.collection.ordered_ids()[-1],
                f"trace-{index}",
            )
        captured = window.trace_control.list_traces.count()
        assert captured == 4

        filename = tmp_path / "four.nvsession"
        window._write_session(str(filename))

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        reopened = NanoVNASaver()
        try:
            assert reopened.open_session_file(str(filename))
            assert reopened.trace_control.list_traces.count() == 4
            names = [
                reopened.trace_control.collection.get(trace_id).name
                for trace_id in reopened.trace_control.collection.ordered_ids()
            ]
            assert names == [f"trace-{index}" for index in range(4)]
            # The reopened traces carry their own data, not an empty list.
            for trace_id in reopened.trace_control.collection.ordered_ids():
                trace = reopened.trace_control.collection.get(trace_id)
                assert trace.has_channel("S11")
                assert len(trace.channel("S11")) == 51
        finally:
            reopened.worker.quit()
            reopened.deleteLater()

    @staticmethod
    def test_reopened_traces_keep_simulation_provenance(window, tmp_path):
        s11, s21 = _sweep_points("antenna")
        window.data.s11 = s11
        window.data.s21 = s21 or []
        window.data.simulated = True
        window.data.simulated_scenario = "antenna"
        window.trace_control.capture_current()

        filename = tmp_path / "sim.nvsession"
        window._write_session(str(filename))

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        reopened = NanoVNASaver()
        try:
            assert reopened.open_session_file(str(filename))
            trace_id = reopened.trace_control.collection.ordered_ids()[0]
            trace = reopened.trace_control.collection.get(trace_id)
            assert trace.snapshot.simulated is True
            assert trace.snapshot.provenance.simulator_scenario == "antenna"
        finally:
            reopened.worker.quit()
            reopened.deleteLater()

    @staticmethod
    def test_restoring_never_connects_or_sweeps(window, tmp_path):
        """SESSION-03.4: reopening a session must stay offline."""
        s11, _s21 = _sweep_points()
        window.data.s11 = s11
        window.data.s21 = []
        window.trace_control.capture_current()
        filename = tmp_path / "offline.nvsession"
        window._write_session(str(filename))

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        reopened = NanoVNASaver()
        try:
            assert reopened.open_session_file(str(filename))
            assert not reopened.vna.connected()
            assert not reopened.worker.isRunning()
        finally:
            reopened.worker.quit()
            reopened.deleteLater()


class TestCurrentSnapshots:
    @staticmethod
    def test_captured_traces_are_saved_without_duplicating_the_display(
        window,
    ):
        s11, s21 = _sweep_points("filter")
        window.data.s11 = s11
        window.data.s21 = s21
        window.data.simulated = True
        window.trace_control.capture_current()
        captured = window.trace_control.collection.ordered_ids()

        snapshots, trace_ids = window._current_snapshots()
        assert trace_ids == captured
        assert len(snapshots) == 1
        assert snapshots[0].simulated is True

    @staticmethod
    def test_an_uncaptured_sweep_is_still_saved(window):
        s11, s21 = _sweep_points("filter")
        window.data.s11 = s11
        window.data.s21 = s21
        window.data.simulated = True
        window.trace_control.collection.clear()

        snapshots, trace_ids = window._current_snapshots()
        assert trace_ids == ["trace-0"]
        assert len(snapshots) == 1

    @staticmethod
    def test_no_data_yields_no_snapshots(window):
        window.data.s11 = []
        window.data.s21 = []
        window.trace_control.collection.clear()
        snapshots, trace_ids = window._current_snapshots()
        assert snapshots == []
        assert trace_ids == []

    @staticmethod
    def test_captured_trace_alone_is_saved(window):
        """A cleared display must not discard already-captured traces."""
        s11, _s21 = _sweep_points()
        window.data.s11 = s11
        window.data.s21 = []
        window.trace_control.capture_current()
        window.data.s11 = []

        snapshots, trace_ids = window._current_snapshots()
        assert len(snapshots) == 1
        assert trace_ids == [window.trace_control.collection.ordered_ids()[0]]


class TestTraceCollectionClear:
    @staticmethod
    def test_clear_empties_everything():
        from NanoVNASaver.Traces import TraceCollection

        s11, _s21 = _sweep_points()
        collection = TraceCollection()
        collection.add(snapshot_from_datapoints(s11, None))
        collection.set_reference(collection.ordered_ids()[0])
        collection.clear()

        assert collection.ordered_ids() == []
        assert collection.reference is None
        assert collection.live_trace is None
        assert collection.visible() == []


class TestCaptureAndTitleFixes:
    """Regressions from the session and trace integration."""

    @staticmethod
    def test_second_capture_captures_the_displayed_data_not_the_first_trace(
        window,
    ):
        from NanoVNASaver.RFTools import Datapoint

        window.data.s11 = [
            Datapoint(1_000_000 + index * 1_000, 0.5, 0.2) for index in range(5)
        ]
        window.data.s21 = []
        window.trace_control.capture_current()
        window.data.s11 = [
            Datapoint(point.freq, point.re * 0.99, point.im)
            for point in window.data.s11
        ]
        window.trace_control.capture_current()

        collection = window.trace_control.collection
        ids = collection.ordered_ids()
        first = collection.get(ids[0])
        second = collection.get(ids[1])
        assert first.snapshot is not second.snapshot
        assert first.channel("S11")[0].re == pytest.approx(0.5)
        assert second.channel("S11")[0].re == pytest.approx(0.495)

    @staticmethod
    def test_the_simulation_marker_survives_a_session_title_update(window):
        window.data.simulated = True
        window.data.simulated_scenario = "antenna"
        window.update_session_title()

        title = window.windowTitle()
        assert "[SIMULATED: antenna]" in title
        assert "Untitled" in title
