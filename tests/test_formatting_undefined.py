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
"""An undefined readout says so; an infinite one says infinity.

A chart leaves an undefined sample as a gap, which is the right thing to
draw, but the marker panel next to it printed the raw float: ``nan``,
``inf``, ``-inf dB``, ``nan°`` or a bare ``-`` with a unit behind it.  A
user cannot tell an instrument that measured nothing from one that measured
a value the program could not format.

The rule now lives in one place, ``Formatting.UNDEFINED``, and every
readout goes through it.  ``inf`` keeps its own sign and unit: it is a real
limit of the model (a perfect match has infinite return loss), not a
missing measurement.
"""

import math

import pytest

from NanoVNASaver import Formatting as fmt
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.SITools import Value

INFINITY = "\N{INFINITY}"
DEGREE = "\N{DEGREE SIGN}"
OHM = "\N{OHM SIGN}"

#: Every scalar readout with the text it gives a finite value and the
#: suffix ``_non_finite`` appends to the infinity sign.
SCALAR_FORMATTERS = [
    ("format_gain", 1.5, "1.500 dB", " dB"),
    ("format_vswr", 1.5, "1.500", ""),
    ("format_magnitude", 1.5, "1.500", ""),
    ("format_resistance", 1.5, f"1.5 {OHM}", f" {OHM}"),
    ("format_capacitance", 1.5, "1.5 F", " F"),
    ("format_inductance", 1.5, "1.5 H", " H"),
    ("format_group_delay", 1.5, "1.5000 s", " s"),
    ("format_phase", 0.0, f"0.00{DEGREE}", DEGREE),
    ("format_wavelength", 1.5, "1.500 m", " m"),
    ("format_q_factor", 1.5, "1.5", ""),
]


class TestUndefinedIsNamed:
    @pytest.mark.parametrize(
        ("name", "value", "finite", "suffix"), SCALAR_FORMATTERS
    )
    def test_a_finite_value_is_formatted_as_before(
        self, name, value, finite, suffix
    ):
        assert getattr(fmt, name)(value) == finite

    @pytest.mark.parametrize(
        ("name", "value", "finite", "suffix"), SCALAR_FORMATTERS
    )
    def test_a_missing_measurement_is_undefined(
        self, name, value, finite, suffix
    ):
        text = getattr(fmt, name)(math.nan)

        assert text == fmt.UNDEFINED
        assert "nan" not in text.lower()

    @pytest.mark.parametrize(
        ("name", "value", "finite", "suffix"), SCALAR_FORMATTERS
    )
    def test_an_infinite_value_keeps_its_unit_and_sign(
        self, name, value, finite, suffix
    ):
        formatter = getattr(fmt, name)

        assert formatter(math.inf) == f"{INFINITY}{suffix}"
        # A quality factor is a magnitude estimate: its sign carries no
        # meaning, so it is reported without one.
        expected_sign = "" if name == "format_q_factor" else "-"
        assert formatter(-math.inf) == f"{expected_sign}{INFINITY}{suffix}"

    def test_a_complex_impedance_with_a_missing_half_is_undefined(self):
        assert fmt.format_complex_imp(complex(math.nan, 1)) == fmt.UNDEFINED
        assert fmt.format_complex_imp(complex(1, math.nan)) == fmt.UNDEFINED
        assert fmt.format_complex_imp(complex(math.inf, 1)) == fmt.UNDEFINED

    def test_an_admittance_with_a_missing_half_is_undefined(self):
        assert fmt.format_complex_adm(complex(math.nan, 1)) == fmt.UNDEFINED

    def test_a_finite_complex_value_still_reads_as_one(self):
        assert fmt.format_complex_imp(complex(1, 2)) == f"1+j2 {OHM}"

    def test_a_polar_readout_is_undefined_rather_than_a_pair_of_nans(self):
        assert fmt.format_polar(math.nan, 0) == fmt.UNDEFINED
        assert fmt.format_polar(math.inf, 0) == INFINITY
        assert fmt.format_polar(0.5, 0) == f"0.50\N{ANGLE}0.00{DEGREE}"

    def test_the_undefined_word_is_not_a_unitless_dash(self):
        # ``_`` and ``-`` are used elsewhere for "clamped at zero" and for
        # "negative, refused"; neither says the value was not measured.
        assert fmt.UNDEFINED not in {"-", "", "nan"}


class TestMarkerReadout:
    """The panel the user actually reads."""

    @staticmethod
    def _readout(sample):
        from PySide6 import QtWidgets

        from NanoVNASaver.Marker.Widget import Marker

        assert QtWidgets.QApplication.instance() is not None
        marker = Marker("probe")
        marker.location = 0
        marker.updateLabels([sample], [])
        return {name: label.text() for name, label in marker.label.items()}

    def test_a_measured_point_shows_no_undefined_word(self, qt_app):
        labels = self._readout(Datapoint(10_000_000, 0.3, 0.2))

        assert fmt.UNDEFINED not in labels.values()
        assert all("nan" not in text.lower() for text in labels.values())
        assert all("inf" not in text.lower() for text in labels.values())

    def test_an_undefined_point_is_labelled_everywhere(self, qt_app):
        labels = self._readout(Datapoint(10_000_000, math.nan, math.nan))

        for name in (
            "impedance",
            "admittance",
            "s11mag",
            "s11phase",
            "s11polar",
            "s11q",
            "serr",
            "serl",
            "serc",
            "vswr",
            "returnloss",
        ):
            assert labels[name] == fmt.UNDEFINED, name
        for text in labels.values():
            assert "nan" not in text.lower(), text

    def test_a_perfect_match_reports_infinity_not_undefined(self, qt_app):
        # Gamma exactly zero is a 50 ohm load: infinite return loss, and a
        # VSWR of 1 — the one place ``1 - |Gamma|`` is legitimately zero.
        labels = self._readout(Datapoint(10_000_000, 1.0, 0.0))

        assert labels["vswr"] == INFINITY
        assert labels["returnloss"] == "0.000 dB"

    def test_a_zero_reflection_coefficient_is_not_infinite_vswr(self, qt_app):
        labels = self._readout(Datapoint(10_000_000, 0.0, 0.0))

        assert labels["vswr"] == "1.000"
        assert labels["returnloss"] == f"-{INFINITY} dB"
        assert fmt.UNDEFINED not in labels.values()


class TestDeltaReadout:
    """A difference needs both ends; one missing end is not a number."""

    def _delta(self, first, second):
        from PySide6 import QtWidgets

        from NanoVNASaver.Marker.Delta import DeltaMarker
        from NanoVNASaver.Marker.Widget import Marker

        assert QtWidgets.QApplication.instance() is not None
        data = [first, second]
        marker_a = Marker("1")
        marker_a.location = 0
        marker_a.updateLabels(data, [])
        marker_b = Marker("2")
        marker_b.location = 1
        marker_b.updateLabels(data, [])
        delta = DeltaMarker("delta")
        delta.set_markers(marker_a, marker_b)
        delta.updateLabels()
        return {name: label.text() for name, label in delta.label.items()}

    def test_two_measured_points_give_a_difference(self, qt_app):
        labels = self._delta(
            Datapoint(10_000_000, 0.3, 0.2),
            Datapoint(20_000_000, 0.2, 0.1),
        )

        assert fmt.UNDEFINED not in labels.values()
        assert all("nan" not in text.lower() for text in labels.values())

    def test_one_missing_point_makes_the_difference_undefined(self, qt_app):
        labels = self._delta(
            Datapoint(10_000_000, 0.3, 0.2),
            Datapoint(20_000_000, math.nan, math.nan),
        )

        assert labels["impedance"] == fmt.UNDEFINED
        assert labels["s11mag"] == fmt.UNDEFINED
        assert labels["s11phase"] == fmt.UNDEFINED
        assert labels["s11polar"] == fmt.UNDEFINED
        for text in labels.values():
            assert "nan" not in text.lower(), text

    def test_two_missing_points_are_not_subtracted_into_a_number(self, qt_app):
        # ``inf - inf`` is NaN and ``nan - nan`` is NaN; both must read as
        # undefined rather than as a plausible zero or a crash.
        labels = self._delta(
            Datapoint(10_000_000, math.nan, math.nan),
            Datapoint(20_000_000, math.nan, math.nan),
        )

        for name in ("impedance", "s11mag", "s11phase", "s11polar"):
            assert labels[name] == fmt.UNDEFINED, name


def test_the_value_type_still_renders_nan_as_a_dash():
    # ``SITools.Value`` is also the parser and the axis label source; its
    # own rendering is asserted by tests/test_sitools.py and is deliberately
    # unchanged, because chart axes never receive an undefined bound.
    assert str(Value(math.nan)) == "-"
