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
"""TRACE-02/TRACE-04: multi-trace charts, legends and comparison cost."""

import os
import time
import timeit

import pytest

from NanoVNASaver.Measurement import snapshot_from_datapoints
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Traces import (
    ComparisonCache,
    ComparisonKind,
    Trace,
    TraceCollection,
    compare_traces,
)

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


def _snapshot(shift=0.0, points=21, name="t", two_port=True):
    s11 = [
        Datapoint(1_000_000 + index * 100_000, 0.5 + shift, 0.2)
        for index in range(points)
    ]
    s21 = list(s11) if two_port else None
    return snapshot_from_datapoints(s11, s21, name=name)


def _collection(count=4):
    collection = TraceCollection()
    return collection, [
        collection.add(_snapshot(0.01 * index, name=f"T{index}"))
        for index in range(count)
    ]


class TestChartTraces:
    @staticmethod
    def test_a_chart_accepts_several_named_traces(qt_app):
        from NanoVNASaver.Charts.Frequency import FrequencyChart

        collection, traces = _collection()
        chart = FrequencyChart("S11 Smith Chart")
        series = [
            (trace.name, trace.color, list(trace.channel("S11")))
            for trace in collection.visible()
        ]
        chart.setTraces(series)

        assert len(chart.traces) == 4
        assert [name for name, _color, _points in chart.traces] == [
            trace.name for trace in traces
        ]

    @staticmethod
    def test_empty_series_are_dropped(qt_app):
        from NanoVNASaver.Charts.Frequency import FrequencyChart

        chart = FrequencyChart("S11 VSWR")
        chart.setTraces([("empty", "#ffffff", []), ("full", "#000000", [1])])
        assert [name for name, _c, _p in chart.traces] == ["full"]

    @staticmethod
    def test_the_chart_renders_with_four_traces(qt_app):
        from NanoVNASaver.Charts.Frequency import FrequencyChart

        collection, traces = _collection()
        chart = FrequencyChart("S11 Return Loss")
        chart.resize(320, 320)
        chart.setData(list(traces[0].channel("S11")))
        chart.setTraces(
            [
                (trace.name, trace.color, list(trace.channel("S11")))
                for trace in traces
            ]
        )
        image = chart.grab()
        assert not image.isNull()
        assert image.width() > 0

    @staticmethod
    def test_legend_lists_the_primary_and_extra_traces(qt_app):
        from NanoVNASaver.Charts.Frequency import FrequencyChart

        chart = FrequencyChart("S21 Gain")
        chart.setTraces([("A", "#ff0000", [1]), ("B", "#00ff00", [1])])
        legend = chart.trace_legend()
        assert legend[0][0] == "S21 Gain"
        assert [name for name, _color in legend[1:]] == ["A", "B"]

    @staticmethod
    def test_hidden_traces_are_not_offered(qt_app):
        collection, traces = _collection()
        collection.set_visible(traces[1].id, False)
        names = [trace.name for trace in collection.visible()]
        assert traces[1].name not in names
        assert len(names) == 3

    @staticmethod
    def test_copy_preserves_traces(qt_app):
        from NanoVNASaver.Charts.Frequency import FrequencyChart

        chart = FrequencyChart("S11 VSWR")
        chart.setTraces([("A", "#ff0000", [1])])
        clone = chart.copy()
        assert clone.traces == chart.traces


class TestChannelSelection:
    @staticmethod
    def test_comparison_uses_a_channel_both_traces_carry():
        """TRACE-02.4: never compare on a channel the reference lacks."""
        collection = TraceCollection()
        one_port = collection.add(_snapshot(name="one-port", two_port=False))
        two_port = collection.add(_snapshot(0.01, name="two-port"))
        assert one_port.has_channel("S11")
        assert not one_port.has_channel("S21")
        assert two_port.has_channel("S21")

        result = compare_traces(
            two_port, one_port, parameter="S11", kind=ComparisonKind.COMPLEX
        )
        assert result.available

        unavailable = compare_traces(
            two_port, one_port, parameter="S21", kind=ComparisonKind.COMPLEX
        )
        assert not unavailable.available
        assert "S21" in unavailable.summary()

    @staticmethod
    def test_label_reports_both_names_and_units():
        collection = TraceCollection()
        first = collection.add(_snapshot(name="Source"))
        second = collection.add(_snapshot(0.02, name="Reference"))
        result = compare_traces(
            first, second, parameter="S11", kind=ComparisonKind.MAGNITUDE_DB
        )
        summary = result.summary()
        assert first.name in summary
        assert second.name in summary
        assert "dB" in summary


class TestComparisonPerformance:
    @staticmethod
    def test_ten_thousand_point_comparison_is_fast():
        """TRACE-04.1: 10,000 points per trace must not take seconds."""
        points = 10_000
        source = _snapshot(points=points, name="src")
        reference = _snapshot(0.01, points=points, name="ref")
        source_trace = Trace(id="a", name="A", snapshot=source)
        reference_trace = Trace(id="b", name="B", snapshot=reference)

        result = compare_traces(
            source_trace,
            reference_trace,
            parameter="S11",
            kind=ComparisonKind.MAGNITUDE_DB,
        )
        assert result.available
        assert len(result.frequencies()) == points
        # Measure comparison CPU cost rather than scheduler delays on a
        # shared CI runner.  timeit excludes cyclic GC of objects left by
        # other tests; the best of three also excludes transient overhead.
        elapsed = min(
            timeit.repeat(
                lambda: compare_traces(
                    source_trace,
                    reference_trace,
                    parameter="S11",
                    kind=ComparisonKind.MAGNITUDE_DB,
                ),
                timer=time.process_time,
                number=1,
                repeat=3,
            )
        )
        # Keep the 400 ms budget to catch quadratic comparison regressions.
        assert elapsed < 0.4, f"comparison took {elapsed * 1000:.0f} ms"

    @staticmethod
    def test_repeated_comparisons_hit_the_cache():
        collection, traces = _collection()
        cache = ComparisonCache()
        cache.get(traces[1], traces[0], "S11", ComparisonKind.MAGNITUDE_DB)

        started = time.perf_counter()
        for _ in range(200):
            cache.get(traces[1], traces[0], "S11", ComparisonKind.MAGNITUDE_DB)
        elapsed = time.perf_counter() - started
        assert elapsed < 0.05

    @staticmethod
    def test_cache_is_invalidated_by_a_new_trace_revision():
        collection, traces = _collection()
        cache = ComparisonCache()
        first = cache.get(
            traces[1], traces[0], "S11", ComparisonKind.MAGNITUDE_DB
        )
        traces[1].replace_snapshot(_snapshot(0.5, name="T1"))
        second = cache.get(
            traces[1], traces[0], "S11", ComparisonKind.MAGNITUDE_DB
        )
        assert first.summary() != second.summary()

    @staticmethod
    def test_cache_invalidate_by_trace_id():
        _unused, traces = _collection()
        cache = ComparisonCache()
        cache.get(traces[1], traces[0], "S11", ComparisonKind.MAGNITUDE_DB)
        cache.invalidate(traces[1].id)
        assert cache._entries == {}


class TestChartAudit:
    @staticmethod
    def test_single_trace_charts_are_documented():
        """TRACE-02.3: every chart has explicitly declared support."""
        from NanoVNASaver.Controls.TraceControl import (
            MULTI_TRACE_CHARTS,
            SINGLE_TRACE_CHARTS,
        )

        assert "TDR" in SINGLE_TRACE_CHARTS
        assert "TDR" not in MULTI_TRACE_CHARTS
        assert MULTI_TRACE_CHARTS

    @staticmethod
    def test_limitation_text_names_the_deferred_chart(qt_app):
        from NanoVNASaver.Controls.TraceControl import TraceControl

        class FakeApp:
            def showError(self, _text):
                return None

        control = TraceControl(FakeApp())
        assert "TDR" in control.label_limit.text()


class TestTraceUpdatesReachTheCharts:
    """TRACE-02.1: a trace action repaints immediately.

    Written after a real defect: the trace control's ``updated`` signal was
    never connected, so capturing, hiding or importing a trace changed no
    chart until the next sweep arrived.
    """

    @staticmethod
    def _window(qt_app):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        return NanoVNASaver()

    @staticmethod
    def test_capturing_a_trace_pushes_it_into_the_charts(qt_app):
        window = TestTraceUpdatesReachTheCharts._window(qt_app)
        try:
            window.data.s11 = [
                Datapoint(1_000_000 + index * 1_000, 0.4, 0.1)
                for index in range(5)
            ]
            window.data.s21 = []
            window.trace_control.capture_current()

            assert window.trace_control.collection.ordered_ids()
            drawn = [
                series
                for chart in window.s11charts
                for series in getattr(chart, "traces", [])
            ]
            assert drawn, "captured trace never reached the charts"
            assert len(drawn[0][2]) == 5
        finally:
            window.worker.quit()

    @staticmethod
    def test_the_signal_is_connected_once(qt_app):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        window = NanoVNASaver()
        try:
            handled = []
            window.trace_control.updated.disconnect()
            window.trace_control.updated.connect(
                lambda _control=None: handled.append(1)
            )
            window.trace_control.refresh()
            window.trace_control.updated.emit(window.trace_control)
            assert handled == [1]
        finally:
            window.worker.quit()


class TestOverlaysOnEverySquareChart:
    """TRACE-02.2: Smith and polar plots draw overlays too.

    Written after a real defect: ``SquareChart.drawValues`` returned early
    when the primary data was empty, so a Smith chart with only overlays
    rendered nothing - the one plot where overlays matter most.
    """

    @staticmethod
    def _render(chart):
        from PySide6 import QtGui

        image = QtGui.QImage(
            max(chart.width(), 64),
            max(chart.height(), 64),
            QtGui.QImage.Format.Format_ARGB32,
        )
        image.fill(QtGui.QColor("black"))
        painter = QtGui.QPainter(image)
        chart.drawValues(painter)
        painter.end()
        return image

    @staticmethod
    def _non_black(image):
        count = 0
        for y in range(image.height()):
            for x in range(image.width()):
                if image.pixelColor(x, y) != image.pixelColor(0, 0):
                    count += 1
        return count

    @staticmethod
    def test_smith_draws_overlays_without_primary_data(qt_app):
        from NanoVNASaver.Charts.Smith import SmithChart

        chart = SmithChart("S11")
        chart.resize(200, 200)
        chart.data = []
        chart.reference = []
        chart.setTraces(
            [
                (
                    "trace-1",
                    "#ff0000",
                    [Datapoint(1e6 + i * 1e5, 0.4, 0.1) for i in range(20)],
                )
            ]
        )
        assert TestOverlaysOnEverySquareChart._non_black(
            TestOverlaysOnEverySquareChart._render(chart)
        )
        chart.deleteLater()

    @staticmethod
    def test_polar_draws_overlays_without_primary_data(qt_app):
        from NanoVNASaver.Charts.Polar import PolarChart

        chart = PolarChart("S11")
        chart.resize(200, 200)
        chart.data = []
        chart.reference = []
        chart.setTraces(
            [
                (
                    "trace-1",
                    "#00ff00",
                    [Datapoint(1e6 + i * 1e5, 0.4, 0.1) for i in range(20)],
                )
            ]
        )
        assert TestOverlaysOnEverySquareChart._non_black(
            TestOverlaysOnEverySquareChart._render(chart)
        )
        chart.deleteLater()
