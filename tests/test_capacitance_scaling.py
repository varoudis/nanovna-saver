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
"""A nearly perfect load must not flatten the capacitance chart.

A capacitance chart divides by reactance: ``C = -1 / (2 pi f X)``.  As
``X`` approaches zero the quotient runs away, and a load a hair away from
resonance produces a value hundreds of times larger than the rest of the
sweep.  The chart then stretched its axis to reach that one value, and every
other sample — the part the user is looking at — collapsed onto a single
pixel row.

``tests/test_chart_nonfinite.py`` covers the exactly-zero case, which is an
infinity and is already treated as undefined.  This file covers the finite
but unrepresentative one, which cannot be filtered by a finiteness test.
"""

import math

import pytest

from NanoVNASaver.Charts.Capacitance import CapacitanceChart
from NanoVNASaver.RFTools import Datapoint, impedance_to_capacitance

FREQS = [1_000_000 + index * 100_000 for index in range(21)]
CAPACITANCE = 100e-12
#: The reactance that turns into a value far outside the sweep's own band.
HUGE = 1e-9


def _datapoint(freq: float, impedance: complex) -> Datapoint:
    """The sweep point a Touchstone file would carry for ``impedance``."""
    gamma = (impedance - 50.0) / (impedance + 50.0)
    return Datapoint(round(freq), gamma.real, gamma.imag)


def _capacitor(freq: float, capacitance: float = CAPACITANCE) -> Datapoint:
    reactance = -1 / (2 * math.pi * freq * capacitance)
    return _datapoint(freq, complex(5.0, reactance))


def _sweep() -> list[Datapoint]:
    return [_capacitor(freq) for freq in FREQS]


@pytest.fixture
def chart(qt_app):
    chart = CapacitanceChart("C")
    yield chart
    chart.hide()
    chart.deleteLater()


def _paint(chart, data) -> None:
    chart.setData(data)
    chart.resize(320, 320)
    image = chart.grab().toImage()
    assert not image.isNull()


class TestOutlierDoesNotSetTheAxis:
    def test_a_plain_capacitor_keeps_its_axis(self, chart):
        _paint(chart, _sweep())

        bulk = impedance_to_capacitance(
            complex(5.0, -1 / (2 * math.pi * FREQS[0] * CAPACITANCE)),
            FREQS[0],
        )
        assert chart.maxValue == pytest.approx(bulk)

    def test_a_near_zero_reactance_does_not_widen_the_axis(self, chart):
        plain = CapacitanceChart("plain")
        try:
            _paint(plain, _sweep())
            plain_range = (plain.minValue, plain.maxValue)
        finally:
            plain.hide()
            plain.deleteLater()

        sweep = _sweep()
        sweep[10] = _datapoint(FREQS[10], complex(50.0, HUGE))
        outlier_capacitance = impedance_to_capacitance(
            complex(50.0, HUGE), FREQS[10]
        )
        assert abs(outlier_capacitance) > 100 * abs(plain_range[1]), (
            "the fixture does not contain an outlier"
        )

        _paint(chart, sweep)

        assert (chart.minValue, chart.maxValue) == pytest.approx(plain_range)

    def test_the_bulk_of_the_sweep_stays_visible(self, chart):
        sweep = _sweep()
        sweep[10] = _datapoint(FREQS[10], complex(50.0, HUGE))

        _paint(chart, sweep)

        visible = [
            d
            for d in sweep
            if chart.minValue
            <= chart.getYPosition(d)
            <= (chart.topMargin + chart.dim.height)
        ]
        assert len(visible) == len(sweep) - 1

    def test_the_outlier_reads_its_true_value(self, chart):
        sweep = _sweep()
        sweep[10] = _datapoint(FREQS[10], complex(50.0, HUGE))
        _paint(chart, sweep)

        # The axis was narrowed, but the value itself is a measurement and
        # is not altered: only the plot refuses to stretch for it.
        assert sweep[10].capacitiveEquivalent(50) == pytest.approx(
            impedance_to_capacitance(complex(50.0, HUGE), FREQS[10])
        )

    def test_a_fixed_span_is_still_honoured(self, chart):
        chart.setFixedValues(True)
        sweep = _sweep()
        sweep[10] = _datapoint(FREQS[10], complex(50.0, HUGE))

        _paint(chart, sweep)

        assert (chart.minValue, chart.maxValue) == pytest.approx(
            (chart.minDisplayValue / 10e11, chart.maxDisplayValue / 10e11)
        )

    def test_a_sweep_with_only_two_points_is_still_scaled(self, chart):
        # Too few samples to call any of them an outlier.
        sweep = [_capacitor(FREQS[0]), _sweep()[10]]
        sweep[1] = _datapoint(FREQS[1], complex(50.0, HUGE))

        _paint(chart, sweep)

        assert chart.maxValue > chart.minValue
