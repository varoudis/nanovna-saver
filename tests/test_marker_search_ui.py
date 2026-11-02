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
"""ANALYSIS-01 at the window level: the marker panel's search buttons.

These drive the real ``NanoVNASaver`` window against an analytic resonance, so
they prove the wiring (button -> engine -> marker widget) rather than the
engine, which ``test_marker_search.py`` covers on its own.
"""

import math
import os

import pytest

MHZ = 1_000_000
CENTRE = 100 * MHZ
#: Gaussian peak depth and width: -6 dB at the centre, -20 dB |S11| floor.
DEPTH_DB = 14.0
SIGMA_HZ = 8 * MHZ

_APP = None


@pytest.fixture(scope="session")
def qt_app():
    global _APP  # noqa: PLW0603 - one application per interpreter
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    if _APP is None:
        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


def _resonance(points=101, span=80 * MHZ):
    """|S11| peaking at -6 dB on CENTRE and falling to -20 dB at the edges."""
    from NanoVNASaver.RFTools import Datapoint

    start = CENTRE - span // 2
    step = span // (points - 1)
    out = []
    for index in range(points):
        freq = start + index * step
        db = -20.0 + DEPTH_DB * math.exp(
            -((freq - CENTRE) ** 2) / (2 * SIGMA_HZ**2)
        )
        out.append(Datapoint(freq, 10 ** (db / 20.0), 0.0))
    return out


@pytest.fixture
def window(qt_app):
    from NanoVNASaver.NanoVNASaver import NanoVNASaver

    window = NanoVNASaver()
    window.data.s11 = _resonance()
    window.data.s21 = []
    yield window
    window.close()


class TestMarkerPanelSearch:
    @staticmethod
    def test_a_peak_search_moves_the_first_marker_to_the_peak(window):
        assert window.markers, "the window should create markers at startup"
        assert window.place_marker_from_search() is True
        assert window.markers[0].freq == CENTRE
        # The peak falls between samples here, so the honest wording is
        # "estimated"; either way the marker lands on the measured peak.
        assert "estimated" in window.label_search.text()

    @staticmethod
    def test_the_marker_snaps_to_a_measured_sample_not_the_fit(window):
        window.place_marker_from_search()
        measured = {point.freq for point in window.data.s11}
        assert window.markers[0].freq in measured

    @staticmethod
    def test_bandwidth_matches_the_analytic_half_power_width(window):
        assert window.show_marker_bandwidth() is True
        text = window.label_search.text()
        # The curve is already in dB, so a 3 dB drop from the -6 dB peak
        # leaves 11 of its 14 dB of depth: exp(-x^2/2sigma^2) = 11/14.
        half_width = SIGMA_HZ * math.sqrt(
            2 * math.log(DEPTH_DB / (DEPTH_DB - 3.0))
        )
        expected = 2 * half_width
        reported = float(text.split("bandwidth ")[1].split(" Hz")[0])
        assert reported == pytest.approx(expected, rel=0.05)
        assert "Q" in text

    @staticmethod
    def test_a_crossing_search_uses_the_target_field(window):
        index = window.search_kind.findData("crossing_above")
        assert index >= 0
        window.search_kind.setCurrentIndex(index)
        window.search_target.setValue(-10.0)
        assert window.place_marker_from_search() is True
        assert "estimated" in window.label_search.text()

    @staticmethod
    def test_an_empty_sweep_says_so_and_does_not_move_the_marker(window):
        before = [marker.freq for marker in window.markers]
        window.data.s11 = []
        window.data.s21 = []
        assert window.place_marker_from_search() is False
        assert "No captured data" in window.label_search.text()
        assert [marker.freq for marker in window.markers] == before

    @staticmethod
    def test_bandwidth_without_data_says_so(window):
        window.data.s11 = []
        window.data.s21 = []
        assert window.show_marker_bandwidth() is False
        assert "No captured data" in window.label_search.text()

    @staticmethod
    def test_every_search_choice_is_reachable_from_the_engine(window):
        from NanoVNASaver.MarkerSearch import SEARCH_CHOICES

        offered = {
            window.search_kind.itemData(index)
            for index in range(window.search_kind.count())
        }
        assert offered == {kind.value for kind, _label in SEARCH_CHOICES}
