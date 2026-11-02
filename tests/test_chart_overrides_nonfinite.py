#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020  Rune B. Broberg
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
"""Every chart must paint a non-finite sample as a gap, not a crash.

``Charts/Frequency.py`` established the rule for its own scaling: a value
that is not a number does not affect the range, leaves a gap with no line
through it, and is never drawn as zero or clamped to an edge.  Charts that
override ``drawValues``/``getYPosition`` each carried their own copy of the
scaling loop and their own arithmetic, so each one has to be checked on its
own -- the base-class fix cannot reach them.

``tests/test_chart_nonfinite.py`` covers the base chart and the capacitance,
inductance and quality-factor charts.  This file covers the overrides.
"""

import math

import pytest

from NanoVNASaver.Charts.Chart import Chart
from NanoVNASaver.Charts.CLogMag import CombinedLogMagChart
from NanoVNASaver.Charts.GroupDelay import GroupDelayChart
from NanoVNASaver.Charts.LogMag import LogMagChart
from NanoVNASaver.Charts.Magnitude import MagnitudeChart
from NanoVNASaver.Charts.MagnitudeZ import MagnitudeZChart
from NanoVNASaver.Charts.Permeability import PermeabilityChart
from NanoVNASaver.Charts.Phase import PhaseChart
from NanoVNASaver.Charts.Polar import PolarChart
from NanoVNASaver.Charts.RIMu import RealImaginaryMuChart
from NanoVNASaver.Charts.RIZ import RealImaginaryZChart
from NanoVNASaver.Charts.RIZSeries import RealImaginaryZSeriesChart
from NanoVNASaver.Charts.RIZShunt import RealImaginaryZShuntChart
from NanoVNASaver.Charts.Smith import SmithChart
from NanoVNASaver.Charts.SParam import SParameterChart
from NanoVNASaver.Charts.VSWR import VSWRChart
from NanoVNASaver.Marker.Widget import Marker
from NanoVNASaver.RFTools import Datapoint

FREQS = [1_000_000 + index * 100_000 for index in range(11)]
BAD = 5

#: Every chart with its own scaling or drawing path.  ``SquareChart`` and
#: ``RealImaginaryChart`` are abstract (``drawChart``/``value`` raise), so
#: the concrete subclasses are listed instead.
OVERRIDING_CHARTS = [
    MagnitudeChart,
    SParameterChart,
    PhaseChart,
    GroupDelayChart,
    PermeabilityChart,
    VSWRChart,
    LogMagChart,
    CombinedLogMagChart,
    MagnitudeZChart,
    RealImaginaryMuChart,
    RealImaginaryZChart,
    RealImaginaryZSeriesChart,
    RealImaginaryZShuntChart,
    SmithChart,
    PolarChart,
]


def _axes(chart) -> dict[str, tuple[float, float]]:
    """The axis ranges a chart computed for its last paint.

    Each override keeps its own pair of attributes, so the ranges are
    collected per family rather than assumed.
    """
    ranges = {}
    for name in ("Value", "Angle", "Delay"):
        low, high = (
            getattr(chart, f"min{name}", None),
            getattr(chart, f"max{name}", None),
        )
        if low is not None or high is not None:
            ranges[name] = (low, high)
    if hasattr(chart, "max_real"):
        for name in ("real", "imag"):
            high = getattr(chart, f"max_{name}")
            span = getattr(chart, f"span_{name}")
            ranges[name] = (high - span, high)
    if hasattr(chart, "maxVSWR"):
        ranges["vswr"] = (chart.maxVSWR - chart.span, chart.maxVSWR)
    return ranges


def _samples(bad=None, bad_real=0.0, bad_imag=0.0):
    """A varying sweep inside the unit circle, one sample replaceable.

    The values have to vary: a constant sweep collapses several of these
    charts to a single row, and a trace drawn on top of an axis line proves
    nothing about the gap rule.
    """
    points = [
        Datapoint(FREQS[index], 0.10 + 0.05 * index, 0.10 * index - 0.25)
        for index in range(len(FREQS))
    ]
    if bad is not None:
        points[bad] = Datapoint(FREQS[bad], bad_real, bad_imag)
    return points


_CHARTS: list = []


def _make(chart_class, name="probe"):
    """Create a chart that outlives the test body.

    A chart wrapper freed while Qt still has it queued for painting takes
    the interpreter down, so every chart stays referenced until the session
    ends (``conftest`` then destroys the widgets while Qt is still alive).
    """
    chart = chart_class(name)
    _CHARTS.append(chart)
    return chart


#: Charts plotted on a fixed unit plane, which have no axis of their own.
UNIT_PLANE_CHARTS = [SmithChart, PolarChart]


def _load(chart, points):
    """Give ``chart`` the sweep in the field it reads.

    A combined log-magnitude chart has two traces of its own; giving it the
    same points twice would draw the second one exactly over the first.
    """
    if isinstance(chart, CombinedLogMagChart):
        # The second trace has to differ, or it hides the first one.
        chart.setCombinedData(
            points,
            [Datapoint(d.freq, d.re / 2, d.im / 2) for d in points],
        )
    else:
        chart.setData(points)
    return chart


def _render(chart):
    chart.resize(320, 320)
    image = chart.grab().toImage()
    assert not image.isNull()
    return image


def _color(name) -> int:
    from PySide6 import QtGui

    return QtGui.QColor(name).rgb() & 0xFFFFFF


def _plot_pixels(image, chart, name) -> set[tuple[int, int]]:
    """Pixels of colour ``name`` inside the plot area, title row excluded."""
    wanted = _color(name)
    return {
        (x, y)
        for x in range(chart.leftMargin, chart.leftMargin + chart.dim.width)
        for y in range(chart.topMargin + 20, chart.topMargin + chart.dim.height)
        if image.pixel(x, y) & 0xFFFFFF == wanted
    }


def _sweep(image, chart) -> set[tuple[int, int]]:
    return _plot_pixels(image, chart, Chart.color.sweep)


def _bad_variants():
    return [
        pytest.param(math.nan, math.nan, id="nan"),
        pytest.param(math.inf, 0.3, id="real-inf"),
        pytest.param(0.3, -math.inf, id="imag-minus-inf"),
    ]


class TestEveryOverridePaintsNonFiniteSamples:
    @pytest.mark.parametrize("chart_class", OVERRIDING_CHARTS)
    @pytest.mark.parametrize(("real", "imag"), _bad_variants())
    def test_a_non_finite_sample_does_not_raise(
        self, qt_app, chart_class, real, imag
    ):
        chart = _load(
            _make(chart_class), _samples(bad=BAD, bad_real=real, bad_imag=imag)
        )

        image = _render(chart)

        assert image.width() > 0

    @pytest.mark.parametrize("chart_class", OVERRIDING_CHARTS)
    def test_a_finite_trace_still_paints(self, qt_app, chart_class):
        chart = _load(_make(chart_class), _samples())

        image = _render(chart)

        assert _sweep(image, chart), "a finite trace drew nothing"

    @pytest.mark.parametrize("chart_class", OVERRIDING_CHARTS)
    def test_an_undefined_sample_does_not_widen_the_axis(
        self, qt_app, chart_class
    ):
        mixed = _load(
            _make(chart_class),
            _samples(bad=BAD, bad_real=math.nan, bad_imag=math.nan),
        )
        finite = _load(_make(chart_class), _samples())

        mixed_axes = _axes(mixed)
        finite_axes = _axes(finite)
        if isinstance(mixed, tuple(UNIT_PLANE_CHARTS)):
            # A Smith or polar chart plots on the fixed unit circle: it has
            # no axis to stretch, only positions to drop.
            assert not mixed_axes and not finite_axes
            return
        assert mixed_axes.keys() == finite_axes.keys(), "scaling did not run"
        for name, (low, high) in finite_axes.items():
            mixed_low, mixed_high = mixed_axes[name]
            # Dropping one sample can only narrow a range.  An axis outside
            # the one without the bad sample was stretched by the
            # undefined value, and a NaN bound fails this outright.
            assert low <= mixed_low <= mixed_high <= high, (
                f"{name} axis moved: {mixed_axes[name]} vs {(low, high)}"
            )

    @pytest.mark.parametrize("chart_class", OVERRIDING_CHARTS)
    def test_every_position_of_the_undefined_sample_is_none(
        self, qt_app, chart_class
    ):
        chart = _load(
            _make(chart_class),
            _samples(bad=BAD, bad_real=math.nan, bad_imag=math.nan),
        )

        # Painting is what computes the axis; the positions are only
        # meaningful afterwards.
        _render(chart)

        positions = {}
        for name in dir(chart_class):
            if not (name.startswith("get") and name.endswith("YPosition")):
                continue
            function = getattr(chart, name)
            if function.__qualname__.startswith("FrequencyChart."):
                # The base implementation is checked in
                # tests/test_chart_nonfinite.py; the charts here that keep
                # it draw through another function.
                continue
            sample = (
                chart.data11[BAD]
                if isinstance(chart, CombinedLogMagChart)
                else chart.data[BAD]
            )
            positions[name] = function(sample)
        assert positions, "no position function to check"
        # ``None`` is what makes Frequency.drawData break the line; zero,
        # an edge value or NaN would paint a wrong point instead.
        for name, position in positions.items():
            assert position is None, f"{name} placed a non-finite sample"


class TestScalarPositionHelpers:
    """The two charts that map a bare number, not a datapoint."""

    def test_a_vswr_without_a_value_has_no_position(self, qt_app):
        chart = _make(VSWRChart)

        assert chart.getYPositionFromValue(math.inf) is None
        assert chart.getYPositionFromValue(math.nan) is None
        assert chart.getYPositionFromValue(2.0) is not None

    def test_a_delay_without_a_value_has_no_position(self, qt_app):
        chart = _make(GroupDelayChart)
        chart.setData(_samples())
        _render(chart)

        assert chart.getYPositionFromDelay(math.nan) is None
        assert chart.getYPositionFromDelay(math.inf) is None
        assert chart.getYPositionFromDelay(1.0) is not None


class TestMarkerOnAnUndefinedPoint:
    @pytest.mark.parametrize("chart_class", OVERRIDING_CHARTS)
    def test_a_marker_on_the_gap_is_not_drawn(self, qt_app, chart_class):
        from PySide6 import QtGui

        chart = _load(
            _make(chart_class),
            _samples(bad=BAD, bad_real=math.nan, bad_imag=math.nan),
        )
        marker = Marker("M")
        marker.location = BAD
        marker.color = QtGui.QColor(255, 0, 255)
        chart.setMarkers([marker])

        image = _render(chart)

        assert not _plot_pixels(image, chart, marker.color)


class TestSegmentedChartsDropBadSamples:
    """Charts that draw real and imaginary parts on separate passes."""

    def test_only_the_undefined_component_loses_its_position(self, qt_app):
        # A raw S parameter has two independent components, so one can be
        # undefined while the other is not.  The impedance charts cannot
        # separate them: both come from the same reflection coefficient.
        chart = _load(
            _make(SParameterChart),
            _samples(bad=BAD, bad_real=math.nan, bad_imag=0.1),
        )

        _render(chart)

        sample = chart.data[BAD]
        assert chart.getReYPosition(sample) is None
        assert chart.getImYPosition(sample) is not None

    @pytest.mark.parametrize(
        "chart_class",
        [
            PermeabilityChart,
            RealImaginaryMuChart,
            RealImaginaryZChart,
            RealImaginaryZSeriesChart,
            RealImaginaryZShuntChart,
        ],
    )
    def test_an_undefined_reflection_coefficient_hides_both_parts(
        self, qt_app, chart_class
    ):
        chart = _load(
            _make(chart_class),
            _samples(bad=BAD, bad_real=math.nan, bad_imag=0.1),
        )

        _render(chart)

        sample = chart.data[BAD]
        assert chart.getReYPosition(sample) is None
        assert chart.getImYPosition(sample) is None
