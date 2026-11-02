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
"""Charts must paint when a plotted value is undefined.

A 50 ohm load has Gamma = 0, so X = 0 and ``impedance_to_capacitance``
returns ``inf``.  Converting that to a pixel used to raise inside
``paintEvent`` - ``ValueError`` for NaN (``inf - inf`` while scaling) and
``OverflowError`` for ``inf`` (``round()`` in ``getYPosition``).  An
undefined value is not a measurement: it must be a gap, it must not move the
axis, and it must never be drawn as zero or clamped to an edge.
"""

import math

import pytest
from PySide6 import QtGui

from NanoVNASaver.Charts.Capacitance import CapacitanceChart
from NanoVNASaver.Charts.Chart import Chart
from NanoVNASaver.Charts.Inductance import InductanceChart
from NanoVNASaver.Charts.QFactor import QualityFactorChart
from NanoVNASaver.Marker.Widget import Marker
from NanoVNASaver.RFTools import Datapoint

FREQS = [1_000_000 + index * 100_000 for index in range(11)]


def _load(freqs=FREQS):
    """Gamma = 0: a 50 ohm load, X exactly 0, capacitance ``inf``."""
    return [Datapoint(freq, 0.0, 0.0) for freq in freqs]


def _reactive(bad=None, freqs=FREQS):
    """Finite capacitive samples, with one optional load at ``bad``."""
    return [
        Datapoint(
            freq,
            0.0 if index == bad else 0.3,
            0.0 if index == bad else 0.4,
        )
        for index, freq in enumerate(freqs)
    ]


def _render(chart):
    chart.resize(320, 320)
    image = chart.grab().toImage()
    assert not image.isNull()
    return image


def _capacitance(*, fixed=False):
    chart = CapacitanceChart("S11 Serial C")
    if fixed:
        chart.minDisplayValue = 0
        chart.maxDisplayValue = 100
        chart.setFixedValues(True)
    return chart


def _color(name) -> int:
    return QtGui.QColor(name).rgb() & 0xFFFFFF


def _painting_area(chart):
    """Pixel coordinates inside the plot, excluding axes and labels."""
    width = chart.leftMargin + chart.dim.width
    height = chart.topMargin + chart.dim.height
    for x in range(chart.leftMargin + 1, width):
        for y in range(chart.topMargin + 1, height):
            yield x, y


def _pixels(image, chart, color) -> set[tuple[int, int]]:
    """Pixels of ``color`` inside the plot, meaning below the trace legend.

    The legend draws a short colour swatch for each trace near the top
    edge, so the plot proper starts below it.
    """
    wanted = _color(color)
    top = chart.topMargin + 20
    return {
        (x, y)
        for x, y in _painting_area(chart)
        if y >= top and image.pixel(x, y) & 0xFFFFFF == wanted
    }


def _sweep(image, chart) -> set[tuple[int, int]]:
    return _pixels(image, chart, Chart.color.sweep)


class TestCapacitanceUndefinedValues:
    def test_all_load_paints_nothing_with_an_automatic_span(self, qt_app):
        chart = _capacitance()
        chart.setData(_load())
        assert all(d.capacitiveEquivalent() == math.inf for d in chart.data)

        image = _render(chart)

        # The default display range, unchanged by an undefined value.
        assert chart.minValue == 0.0
        assert chart.maxValue == pytest.approx(100 / 10e11)
        assert _sweep(image, chart) == set()

    def test_all_load_paints_nothing_with_a_fixed_span(self, qt_app):
        chart = _capacitance(fixed=True)
        chart.setData(_load())

        image = _render(chart)

        assert chart.span > 0
        assert _sweep(image, chart) == set()

    def test_one_undefined_point_does_not_move_the_scale(self, qt_app):
        mixed = _capacitance()
        mixed.setData(_reactive(bad=5))
        finite = _capacitance()
        finite.setData(_reactive())

        assert mixed._find_scaling() == finite._find_scaling()
        image = _render(mixed)
        finite_image = _render(finite)
        assert (mixed.minValue, mixed.maxValue) == (
            finite.minValue,
            finite.maxValue,
        )

        # Every finite sample keeps its position.
        for index, point in enumerate(mixed.data):
            if index == 5:
                continue
            assert mixed.getYPosition(point) == finite.getYPosition(point)
        # The only pixels the finite chart has and this one lacks are the
        # two columns of the undefined point's dot; nothing else moved.
        removed = _sweep(finite_image, finite) - _sweep(image, mixed)
        x = mixed.getXPosition(mixed.data[5])
        assert x in {px for px, _ in removed}
        assert {px for px, _ in removed} <= {x - 2, x - 1, x}
        assert _sweep(image, mixed) <= _sweep(finite_image, finite)

    def test_the_undefined_point_is_not_drawn(self, qt_app):
        chart = _capacitance()
        chart.setData(_reactive(bad=5))

        image = _render(chart)

        x = chart.getXPosition(chart.data[5])
        assert chart.getYPosition(chart.data[5]) is None
        assert not [(px, py) for px, py in _sweep(image, chart) if px == x]

    def test_no_line_is_drawn_through_the_gap(self, qt_app):
        chart = _capacitance()
        chart.setData(_reactive(bad=5))
        chart.setDrawLines(True)

        image = _render(chart)
        sweep = _sweep(image, chart)

        left = chart.getXPosition(chart.data[4])
        right = chart.getXPosition(chart.data[6])
        assert right > left + 20
        # Lines exist on both sides, so the gap below is not vacuous.
        assert [px for px, _ in sweep if left - 30 < px < left]
        assert [px for px, _ in sweep if right < px < right + 30]
        assert not [px for px, _ in sweep if left + 2 < px < right - 2]

    def test_open_and_short_are_undefined_and_still_paint(self, qt_app):
        open_point = Datapoint(FREQS[0], 1.0, 0.0)
        short_point = Datapoint(FREQS[0], -1.0, 0.0)
        assert open_point.impedance() == math.inf
        assert short_point.impedance() == complex(-0.0, 0.0)
        assert abs(short_point.impedance()) == 0.0
        assert open_point.capacitiveEquivalent() == math.inf
        assert short_point.capacitiveEquivalent() == math.inf

        for gamma in (1.0, -1.0):
            chart = _capacitance()
            chart.setData([Datapoint(freq, gamma, 0.0) for freq in FREQS])
            image = _render(chart)
            assert chart.maxValue == pytest.approx(100 / 10e11)
            assert _sweep(image, chart) == set()

    def test_zero_frequency_is_undefined_but_paints(self, qt_app):
        points = [Datapoint(0, 0.3, 0.4) for _ in FREQS]
        assert points[0].capacitiveEquivalent() == -math.inf

        chart = _capacitance()
        chart.setData(points)

        image = _render(chart)
        assert _sweep(image, chart) == set()

    def test_nan_and_minus_infinity_mix_with_finite_values(self, qt_app):
        points = _reactive()
        points[3] = Datapoint(FREQS[3], math.nan, 0.0)
        points[7] = Datapoint(0, 0.3, 0.4)
        assert math.isnan(points[3].capacitiveEquivalent())
        assert points[7].capacitiveEquivalent() == -math.inf

        chart = _capacitance()
        chart.setData(points)
        finite = _capacitance()
        finite.setData(_reactive())

        image = _render(chart)
        finite_image = _render(finite)
        assert (chart.minValue, chart.maxValue) == (
            finite.minValue,
            finite.maxValue,
        )
        measured = _sweep(image, chart)
        assert measured  # the finite samples still draw
        for index, point in enumerate(points):
            x = chart.getXPosition(point)
            if index in (3, 7):
                assert chart.getYPosition(point) is None
                assert not [(px, _) for px, _ in measured if px == x]
                continue
            assert chart.getYPosition(point) == finite.getYPosition(point)
        # Nothing appears that the finite chart does not have.
        assert measured <= _sweep(finite_image, finite)


class TestUndefinedReferenceAndTraces:
    def test_a_load_reference_trace_leaves_the_scale_alone(self, qt_app):
        chart = _capacitance()
        chart.setData(_reactive())
        alone = _capacitance()
        alone.setData(_reactive())
        chart.setReference(_load())

        image = _render(chart)
        alone_image = _render(alone)

        assert (chart.minValue, chart.maxValue) == (
            alone.minValue,
            alone.maxValue,
        )
        # The reference contributes no pixels at all: every one is undefined.
        assert _sweep(image, chart) == _sweep(alone_image, alone)
        assert _pixels(image, chart, Chart.color.reference) == set()

    def test_a_load_extra_trace_paints_and_leaves_the_scale_alone(self, qt_app):
        chart = _capacitance()
        chart.setData(_reactive())
        alone = _capacitance()
        alone.setData(_reactive())
        chart.setTraces([("L", "#ff0000", _load())])

        image = _render(chart)
        alone_image = _render(alone)

        assert (chart.minValue, chart.maxValue) == (
            alone.minValue,
            alone.maxValue,
        )
        assert _sweep(image, chart) == _sweep(alone_image, alone)
        assert _pixels(image, chart, "#ff0000") == set()

    def test_a_marker_on_an_undefined_point_is_not_drawn(self, qt_app):
        chart = _capacitance()
        chart.setData(_load())
        marker = Marker("M")
        marker.location = 3
        marker.color = QtGui.QColor(255, 0, 255)
        chart.setMarkers([marker])

        image = _render(chart)

        assert chart.getYPosition(chart.data[3]) is None
        assert _pixels(image, chart, marker.color) == set()

    def test_a_dragged_marker_skips_an_undefined_point(self, qt_app):
        chart = _capacitance()
        chart.setData(_load())
        chart.resize(320, 320)
        marker = Marker("M")
        marker.location = 3
        chart.setMarkers([marker])

        assert chart.getNearestMarker(100, 100) is None


class TestOtherChartsWithAnAllLoadTrace:
    @pytest.mark.parametrize(
        "chart_class", [InductanceChart, QualityFactorChart]
    )
    def test_an_all_load_trace_still_paints(self, qt_app, chart_class):
        chart = chart_class("all load")
        chart.setData(_load())

        image = _render(chart)

        assert not image.isNull()
        assert image.width() > 0


class TestQualityFactorOwnYPosition:
    def test_one_undefined_point_is_a_gap_there_too(self, qt_app):
        """QFactor overrides getYPosition; the same rule must hold."""
        points = _reactive()
        points[5] = Datapoint(FREQS[5], math.nan, 0.0)
        assert math.isnan(points[5].qFactor())
        chart = QualityFactorChart("S11 Quality Factor")
        chart.setData(points)

        image = _render(chart)

        assert chart.getYPosition(points[5]) is None
        x = chart.getXPosition(points[5])
        assert not [(px, _) for px, _ in _sweep(image, chart) if px == x]
        assert chart.maxQ == 2
