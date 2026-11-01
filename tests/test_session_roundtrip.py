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
from NanoVNASaver.Traces import TraceKind

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

        from NanoVNASaver.NanoVNASaver import LIVE_TRACE_ID

        snapshots, trace_ids = window._current_snapshots()
        assert trace_ids == [LIVE_TRACE_ID]
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


class TestDisplayedMeasurementIsNotLost:
    """SESSION-06: a sweep after the last capture still belongs to the session.

    Written after a real defect: as soon as one trace was captured, the
    measurement on screen was dropped from the saved session, so sweeping
    again and saving silently lost the newest acquisition.
    """

    @staticmethod
    def test_a_newer_displayed_sweep_is_saved_alongside_traces(window):
        from NanoVNASaver.NanoVNASaver import LIVE_TRACE_ID

        s11, s21 = _sweep_points("antenna")
        window.data.s11 = s11
        window.data.s21 = s21 or []
        window.trace_control.capture_current()
        captured = window.trace_control.collection.ordered_ids()
        assert len(captured) == 1

        # A different sweep arrives after the capture.
        newer, _newer21 = _sweep_points("filter")
        window.data.s11 = newer
        window.data.s21 = []

        snapshots, trace_ids = window._current_snapshots()
        assert trace_ids == [*captured, LIVE_TRACE_ID]
        assert len(snapshots) == 2
        assert snapshots[-1].channel("S11")[0].freq == newer[0].freq

    @staticmethod
    def test_the_same_displayed_data_is_not_saved_twice(window):
        s11, s21 = _sweep_points("filter")
        window.data.s11 = s11
        window.data.s21 = s21
        window.trace_control.capture_current()
        captured = window.trace_control.collection.ordered_ids()

        snapshots, trace_ids = window._current_snapshots()
        assert trace_ids == captured
        assert len(snapshots) == 1

    @staticmethod
    def test_the_displayed_sweep_survives_a_round_trip(window, tmp_path):
        s11, s21 = _sweep_points("antenna")
        window.data.s11 = s11
        window.data.s21 = s21 or []
        window.trace_control.capture_current()
        newer, _ = _sweep_points("filter")
        window.data.s11 = newer
        window.data.s21 = []

        filename = tmp_path / "two.nvsession"
        window._write_session(str(filename))

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        reopened = NanoVNASaver()
        try:
            assert reopened.open_session_file(str(filename))
            names = [
                reopened.trace_control.collection.get(trace_id).name
                for trace_id in reopened.trace_control.collection.ordered_ids()
            ]
            assert len(names) == 2
        finally:
            reopened.worker.quit()


class TestSweepStateRoundTrip:
    """SESSION-01.2: the sweep setup itself is part of the experiment."""

    @staticmethod
    def test_sweep_settings_are_saved_and_restored(window, tmp_path):
        window.sweep.update(
            start=27_000_000, end=29_000_000, segments=2, points=51
        )
        window.data.s11 = [
            Datapoint(27_000_000 + index * 1_000_000, 0.4, 0.1)
            for index in range(3)
        ]
        document = window.current_session_document()
        assert document.sweep["start"] == 27_000_000
        assert document.sweep["end"] == 29_000_000
        assert document.sweep["points"] == 51
        assert document.sweep["segments"] == 2

        filename = tmp_path / "sweep.nvsession"
        window._write_session(str(filename))

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        reopened = NanoVNASaver()
        try:
            assert reopened.open_session_file(str(filename))
            assert reopened.sweep.start == 27_000_000
            assert reopened.sweep.end == 29_000_000
            assert reopened.sweep.segments == 2
            assert reopened.sweep.points == 51
            assert reopened.vna.datapoints == 51
        finally:
            reopened.worker.quit()

    @staticmethod
    def test_markers_are_saved_and_restored(window, tmp_path):
        window.data.s11 = [
            Datapoint(27_000_000 + index * 1_000_000, 0.4, 0.1)
            for index in range(3)
        ]
        window.markers[0].frequencyInput.setText("29000000")
        window.markers[0].freq = 29_000_000
        window.markers[0].findLocation(window.data.s11)

        document = window.current_session_document()
        assert document.markers[0]["frequency"] == 29_000_000

        filename = tmp_path / "markers.nvsession"
        window._write_session(str(filename))

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        reopened = NanoVNASaver()
        try:
            assert reopened.open_session_file(str(filename))
            assert reopened.markers[0].freq == 29_000_000
        finally:
            reopened.worker.quit()

    @staticmethod
    def test_calibration_is_saved_and_restored(window, tmp_path):
        from NanoVNASaver.Calibration import Calibration

        calibration = Calibration()
        calibration.load("tests/data/sol_27_30.cal")
        calibration.calc_corrections()
        window.calibration = calibration
        window.data.s11 = [Datapoint(27_500_000, 0.4, 0.1)]

        document = window.current_session_document()
        assert document.calibration["valid_one_port"] is True
        assert "loaded.cal" in document.calibration_files

        filename = tmp_path / "cal.nvsession"
        window._write_session(str(filename))

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        reopened = NanoVNASaver()
        try:
            assert reopened.open_session_file(str(filename))
            assert reopened.calibration.isValid1Port()
            assert reopened.calibration.dataset.frequencies()
        finally:
            reopened.worker.quit()


class TestCapturedSnapshotKeepsItsProvenance:
    """SESSION-06: capture must not re-derive a snapshot from the display.

    Written after a real defect: ``current_snapshot()`` rebuilt the snapshot
    from the displayed datapoints, so a capture always claimed to be
    uncorrected and complete - discarding both the calibration identity and
    the real completion state of a partial sweep.
    """

    @staticmethod
    def test_capture_keeps_the_workers_snapshot(window):
        from NanoVNASaver.Measurement import (
            CompletionState,
            snapshot_from_datapoints,
        )

        s11, _s21 = _sweep_points("antenna")
        snapshot = snapshot_from_datapoints(
            s11,
            None,
            name="worker",
            source="worker",
            completion=CompletionState.PARTIAL,
        )
        window._live_snapshot = snapshot
        window.data.s11 = s11
        window.data.s21 = []

        current = window.current_snapshot()
        assert current is snapshot
        assert current.provenance.completion is CompletionState.PARTIAL

    @staticmethod
    def test_a_completed_sweep_marks_the_snapshot_complete(window):
        from NanoVNASaver.Measurement import (
            CompletionState,
            snapshot_from_datapoints,
        )

        s11, _s21 = _sweep_points("antenna")
        window._live_snapshot = snapshot_from_datapoints(
            s11,
            None,
            name="worker",
            source="worker",
            completion=CompletionState.PARTIAL,
        )
        window.data.s11 = s11
        window.data.s21 = []
        window._remember_complete_sweep()

        assert (
            window.current_snapshot().provenance.completion
            is CompletionState.COMPLETE
        )

    @staticmethod
    def test_two_captures_of_different_data_are_distinct(window):
        """The collapsed snapshot must not be reused after the data changes."""
        s11, _s21 = _sweep_points("antenna")
        window.data.s11 = s11
        window.data.s21 = []
        window.worker.run_once()
        window.trace_control.capture_current()

        window.data.s11 = [
            Datapoint(point.freq, point.re * 0.99, point.im) for point in s11
        ]
        window.trace_control.capture_current()

        collection = window.trace_control.collection
        ids = collection.ordered_ids()
        first = collection.get(ids[0])
        second = collection.get(ids[1])
        assert first.snapshot is not second.snapshot
        assert first.channel("S11")[0].re != second.channel("S11")[0].re

    @staticmethod
    def test_without_an_acquisition_the_display_is_snapshotted(window):
        s11, _s21 = _sweep_points("antenna")
        window._live_snapshot = None
        window.data.s11 = s11
        window.data.s21 = []

        current = window.current_snapshot()
        assert current is not None
        assert len(current.channel("S11")) == len(s11)


class TestRepeatedSaveDoesNotLoseMeasurements:
    """SESSION-07: save -> reopen -> sweep -> save again.

    Written after a real defect: the displayed measurement was always filed
    under ``trace-live``, so a session reopened from a file that already held
    a ``trace-live`` entry produced two entries with the same id.  The archive
    is a mapping keyed by trace id, so the older measurement was overwritten.
    """

    @staticmethod
    def test_a_second_save_keeps_both_measurements(window, tmp_path):
        first = tmp_path / "first.nvsession"
        window.data.s11 = [
            Datapoint(100 + index * 100, 0.4, 0.1) for index in range(3)
        ]
        window.data.s21 = []
        assert window._write_session(str(first))

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        reopened = NanoVNASaver()
        try:
            assert reopened.open_session_file(str(first))
            reopened.data.s11 = [
                Datapoint(500 + index * 100, 0.3, 0.2) for index in range(3)
            ]
            reopened.data.s21 = []
            reopened._live_snapshot = None

            second = tmp_path / "second.nvsession"
            assert reopened._write_session(str(second))

            from NanoVNASaver.Session import (
                document_snapshots,
                load_session,
            )

            document = load_session(str(second))
            ids = [trace["trace_id"] for trace in document.traces]
            assert len(ids) == len(set(ids)), f"duplicate trace ids: {ids}"
            snapshots = document_snapshots(document)
            starts = sorted(
                snapshot.channel("S11")[0].freq
                for snapshot in snapshots.values()
            )
            assert starts == [100, 500]
        finally:
            reopened.worker.quit()

    @staticmethod
    def test_a_third_save_is_stable(window, tmp_path):
        window.data.s11 = [
            Datapoint(100 + index * 100, 0.4, 0.1) for index in range(3)
        ]
        window.data.s21 = []
        first = tmp_path / "one.nvsession"
        window._write_session(str(first))

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        reopened = NanoVNASaver()
        try:
            reopened.open_session_file(str(first))
            reopened.data.s11 = [
                Datapoint(500 + index * 100, 0.3, 0.2) for index in range(3)
            ]
            reopened.data.s21 = []
            reopened._live_snapshot = None
            second = tmp_path / "two.nvsession"
            reopened._write_session(str(second))
            reopened._write_session(str(second))

            from NanoVNASaver.Session import load_session

            ids = [
                trace["trace_id"] for trace in load_session(str(second)).traces
            ]
            assert len(ids) == len(set(ids))
            assert len(ids) == 2
        finally:
            reopened.worker.quit()


class TestReopenedSessionShowsTheDisplayedMeasurement:
    """SESSION-07: reopening must show the saved *displayed* sweep.

    Written after a real defect: restoration always took the first trace in
    the archive, so reopening a session whose displayed sweep started at
    500 Hz showed the earlier captured sweep starting at 100 Hz.
    """

    @staticmethod
    def test_the_live_measurement_is_displayed(window, tmp_path):
        window.data.s11 = [
            Datapoint(100 + index * 10, 0.4, 0.1) for index in range(3)
        ]
        window.data.s21 = []
        window.trace_control.capture_current()
        window.data.s11 = [
            Datapoint(500 + index * 10, 0.3, 0.2) for index in range(3)
        ]
        window.data.s21 = []

        filename = tmp_path / "live.nvsession"
        assert window._write_session(str(filename))

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        reopened = NanoVNASaver()
        try:
            assert reopened.open_session_file(str(filename))
            assert reopened.data.s11[0].freq == 500
            ids = reopened.trace_control.collection.ordered_ids()
            live = [
                trace_id
                for trace_id in ids
                if reopened.trace_control.collection.get(trace_id).kind
                is TraceKind.LIVE
            ]
            assert len(live) == 1
            assert (
                reopened.trace_control.collection.get(live[0])
                .channel("S11")[0]
                .freq
                == 500
            )
        finally:
            reopened.worker.quit()


class TestReopenedSessionCanBeSavedAgain:
    """SESSION-07: restoration must produce a saveable in-memory state.

    Written after a real defect: analysis kinds were restored verbatim as
    strings while saving expected enum members, so saving a reopened preset
    session raised ``AttributeError: 'str' object has no attribute 'name'``.
    """

    @staticmethod
    def test_restored_analysis_kinds_are_enum_members(window, tmp_path):
        from NanoVNASaver.Analysis.Results import AnalysisKind

        window._preset_analyses = (AnalysisKind.VSWR, AnalysisKind.CABLE)
        first = tmp_path / "kinds.nvsession"
        window.data.s11 = [
            Datapoint(1_000_000, 0.4, 0.1),
            Datapoint(2_000_000, 0.3, 0.1),
        ]
        window.data.s21 = []
        window._write_session(str(first))

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        reopened = NanoVNASaver()
        try:
            assert reopened.open_session_file(str(first))
            assert all(
                isinstance(kind, AnalysisKind)
                for kind in reopened._preset_analyses
            )
            assert reopened._preset_analyses == (
                AnalysisKind.VSWR,
                AnalysisKind.CABLE,
            )
        finally:
            reopened.worker.quit()

    @staticmethod
    def test_save_reopen_save_does_not_raise(window, tmp_path):
        from NanoVNASaver.Analysis.Results import AnalysisKind

        window._preset_analyses = (AnalysisKind.RESONANCE,)
        window.data.s11 = [
            Datapoint(1_000_000, 0.4, 0.1),
            Datapoint(2_000_000, 0.3, 0.1),
        ]
        window.data.s21 = []
        first = tmp_path / "first.nvsession"
        window._write_session(str(first))

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        reopened = NanoVNASaver()
        try:
            reopened.open_session_file(str(first))
            second = tmp_path / "second.nvsession"
            assert reopened._write_session(str(second))
            assert reopened._write_session(str(second))

            from NanoVNASaver.Session import load_session

            document = load_session(str(second))
            assert document.analysis["kinds"] == ["RESONANCE"]
        finally:
            reopened.worker.quit()

    @staticmethod
    def test_unknown_analysis_kinds_are_ignored_not_crashed(window, tmp_path):
        from NanoVNASaver.Session import make_document, save_session

        document = make_document(
            [],
            analysis={"kinds": ["NOT_A_KIND", "VSWR"]},
        )
        filename = tmp_path / "unknown.nvsession"
        save_session(document, str(filename))

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        reopened = NanoVNASaver()
        try:
            assert reopened.open_session_file(str(filename))
            from NanoVNASaver.Analysis.Results import AnalysisKind

            assert reopened._preset_analyses == (AnalysisKind.VSWR,)
            assert reopened._write_session(str(tmp_path / "again.nvsession"))
        finally:
            reopened.worker.quit()


class TestSweepAndLayoutPersistence:
    """SESSION-07: the saved sweep setup and chart layout are complete."""

    @staticmethod
    def test_logarithmic_and_truncation_are_saved(window):
        window.sweep.set_logarithmic(True)
        window.sweep.set_averages(5, 2)
        information = window.sweep_info_dict()
        assert information["logarithmic"] is True
        assert information["truncates"] == 2
        assert information["averages"] == 5

    @staticmethod
    def test_logarithmic_and_truncation_survive_a_round_trip(window, tmp_path):
        window.sweep.update(
            start=1_000_000, end=2_000_000, segments=1, points=51
        )
        window.sweep.set_logarithmic(True)
        window.sweep.set_averages(5, 2)
        window.data.s11 = [
            Datapoint(1_000_000 + index * 1_000, 0.4, 0.1) for index in range(3)
        ]
        window.data.s21 = []
        filename = tmp_path / "sweep.nvsession"
        window._write_session(str(filename))

        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        reopened = NanoVNASaver()
        try:
            assert reopened.open_session_file(str(filename))
            assert reopened.sweep.properties.logarithmic is True
            assert reopened.sweep.properties.averages == (5, 2)
        finally:
            reopened.worker.quit()

    @staticmethod
    def test_the_layout_records_the_charts_on_screen(window):
        state = window.chart_state_dict()
        assert state["cells"], "no chart cells recorded"
        shown = {cell["name"] for cell in state["cells"]}
        placed = {
            window.charts_layout.itemAt(index).widget().name
            for index in range(window.charts_layout.count())
            if window.charts_layout.itemAt(index).widget() is not None
        }
        assert shown == placed

    @staticmethod
    def test_the_layout_round_trips(window, tmp_path):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        window.apply_chart_state(
            {
                "cells": [
                    {"row": 0, "column": 0, "name": "S11 Return Loss"},
                    {"row": 0, "column": 1, "name": "S21 Gain"},
                    {"row": 1, "column": 0, "name": "TDR"},
                ]
            }
        )
        window.data.s11 = [
            Datapoint(1_000_000 + index * 1_000, 0.4, 0.1) for index in range(3)
        ]
        window.data.s21 = []
        before = window.chart_state_dict()
        filename = tmp_path / "layout.nvsession"
        window._write_session(str(filename))

        reopened = NanoVNASaver()
        try:
            assert reopened.open_session_file(str(filename))
            assert reopened.chart_state_dict() == before
        finally:
            reopened.worker.quit()
