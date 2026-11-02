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
"""EFHW analysis: comparing successive sweeps when bands come and go.

The comparison pairs the resonances of the two sweeps by position and
reports the delta of the paired ones.  Two things must hold however the
band count changes: every resonance of the new sweep gets a delta entry
(the results table indexes it directly), and a band that really did line
up is reported as a delta rather than as "no comparison".
"""

import pytest

from NanoVNASaver.Analysis.EFHWAnalysis import EFHWAnalysis
from NanoVNASaver.RFTools import Datapoint

FIELDS = [("freq", str)]
MHZ = 1_000_000

#: Two resonances 12 MHz apart: closer than any pairing could be, so one
#: band of the old sweep can be skipped unambiguously.
WIDE = 12 * MHZ


@pytest.fixture
def analysis(qt_app):
    class _App:
        reference_impedance = 50.0

    return EFHWAnalysis(_App())


def _bands(*freqs):
    """The old/new mapping shape ``compare`` consumes."""
    return {index: {"freq": float(freq)} for index, freq in enumerate(freqs)}


class TestComparePairsBands:
    @staticmethod
    def test_equal_sweeps_compare_every_band(analysis):
        diff = analysis.compare(
            _bands(1 * MHZ, 100 * MHZ),
            _bands(1 * MHZ, 100 * MHZ),
            fields=FIELDS,
        )
        assert sorted(diff) == [0, 1]
        assert diff[0]["freq"] == "0.0"
        assert diff[1]["freq"] == "0.0"

    @staticmethod
    def test_a_vanished_band_leaves_a_comparison_for_the_next_one(analysis):
        """Bug 10: the shifted pairing never wrote an entry for that index."""
        diff = analysis.compare(
            _bands(1 * MHZ, 13 * MHZ, 25 * MHZ),
            _bands(1 * MHZ, 25 * MHZ),
            fields=FIELDS,
        )
        assert sorted(diff) == [0, 1]
        assert diff[0]["freq"] == "0.0"
        assert diff[1]["freq"] == "0.0"

    @staticmethod
    def test_two_vanished_bands_are_skipped(analysis):
        """The FIXME's own case: two (here three) bands missing in the new
        sweep must not stop the next band from being paired."""
        diff = analysis.compare(
            _bands(1 * MHZ, 13 * MHZ, 25 * MHZ, 50 * MHZ),
            _bands(1 * MHZ, 50 * MHZ),
            fields=FIELDS,
        )
        assert sorted(diff) == [0, 1]
        assert diff[1]["freq"] == "0.0"

    @staticmethod
    def test_an_extra_band_does_not_erase_the_last_comparison(analysis):
        """The filler loop used to overwrite a matched entry with '-'."""
        diff = analysis.compare(
            _bands(1 * MHZ, 100 * MHZ),
            _bands(1 * MHZ, 50 * MHZ, 100 * MHZ),
            fields=FIELDS,
        )
        assert sorted(diff) == [0, 1, 2]
        assert diff[0]["freq"] == "0.0"
        assert diff[1]["freq"] == "-"
        assert diff[2]["freq"] == "0.0"

    @staticmethod
    def test_an_unrelated_band_is_not_paired(analysis):
        diff = analysis.compare(
            _bands(1 * MHZ, 100 * MHZ),
            _bands(1 * MHZ, 250 * MHZ),
            fields=FIELDS,
        )
        assert diff[0]["freq"] == "0.0"
        assert diff[1]["freq"] == "-"

    @staticmethod
    def test_the_first_sweep_has_nothing_to_compare_against(analysis):
        diff = analysis.compare({}, _bands(1 * MHZ, 100 * MHZ), fields=FIELDS)
        assert diff == {0: {"freq": "-"}, 1: {"freq": "-"}}


def _s11(zero_indices, points=40):
    """A sweep whose phase is zero at ``zero_indices``."""
    data = []
    for index in range(points):
        imag = 0.0 if index in zero_indices else 0.1
        data.append(Datapoint(1 * MHZ + index * MHZ, 0.1, imag))
    return data


def _rows(analysis):
    """Text of the result rows added to the analysis form."""
    from PySide6 import QtWidgets

    texts = []
    for row in range(analysis.layout.rowCount()):
        item = analysis.layout.itemAt(
            row, QtWidgets.QFormLayout.ItemRole.FieldRole
        )
        widget = item.widget() if item is not None else None
        if isinstance(widget, QtWidgets.QLabel) and widget.text().startswith(
            " ("
        ):
            texts.append(widget.text())
    return texts


class TestReport:
    @staticmethod
    def test_a_band_disappearing_between_sweeps_keeps_the_report(analysis):
        analysis.app.data = type("Data", (), {"s11": _s11({8, 20, 32})})()
        analysis.crossings = [8, 20, 32]
        analysis.do_resonance_analysis()
        first = len(_rows(analysis))
        assert first == 3

        analysis.app.data = type("Data", (), {"s11": _s11({8, 32})})()
        analysis.crossings = [8, 32]
        # Used to raise KeyError(1) out of the results loop.
        analysis.do_resonance_analysis()

        second = _rows(analysis)[first:]
        assert len(second) == 2
        assert all("(0.000Hz)" in text for text in second)
