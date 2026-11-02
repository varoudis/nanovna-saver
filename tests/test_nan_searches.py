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
"""Minimum and maximum searches over undefined samples.

``Datapoint.vswr`` and ``Datapoint.gain`` answer NaN for a sample that was
never measured, and that is deliberate: NaN means "undefined", while ``inf``
means "total reflection" and is a real measurement.  Python's ``min``/``max``
compare NaN as unordered, so a NaN won a search whenever it came first and
lost whenever it came last -- the reported best match depended on the sweep
order.  Every search therefore covers the measured samples only, keeps an
infinite value as a candidate, and falls back to its existing empty state
when nothing was measured.
"""

import math
from typing import ClassVar

import pytest

from NanoVNASaver import AnalyticTools as At
from NanoVNASaver.Analysis.BandStopAnalysis import BandStopAnalysis
from NanoVNASaver.Analysis.Results import (
    AnalysisKind,
    AnalysisStatus,
    analyse_cable,
)
from NanoVNASaver.Analysis.SimplePeakSearchAnalysis import (
    SimplePeakSearchAnalysis,
)
from NanoVNASaver.Measurement import snapshot_from_datapoints
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Traces import (
    ComparisonKind,
    TraceCollection,
    compare_traces,
)

NAN = math.nan
INF = math.inf

#: VSWR sweep with one deep match at index 3 (1.105) and a shallow one at
#: index 9 (3.0).
VSWR_SWEEP = [9.0, 4.0, 2.3333, 1.1053, 2.3333, 4.0, 9.0, 5.6667, 4.0, 3.0]
#: The same sweep as S11 magnitudes, for the analysis that reads ``vswr``.
S11_MAGNITUDES = [0.8, 0.6, 0.4, 0.05, 0.4, 0.6, 0.8, 0.7, 0.6, 0.5]
#: Two separated peaks at indices 5 and 15.
TWO_PEAKS = [
    0.0,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    3.0,
    2.0,
    1.0,
    0.5,
    0.0,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    3.0,
    2.0,
    1.0,
    0.5,
    0.0,
]

#: Positions a NaN is planted at: first, a middle sample and the last one.
POSITIONS = [
    pytest.param(0, id="first"),
    pytest.param(5, id="middle"),
    pytest.param(9, id="last"),
]


def _planted(values, position, value=NAN):
    """``values`` with ``value`` at ``position``."""
    planted = list(values)
    planted[position] = value
    return planted


def _points(values, start=1_000_000):
    return [
        Datapoint(start + index * 1_000, value, 0.0)
        for index, value in enumerate(values)
    ]


def _snapshot(s11, s21=None):
    return snapshot_from_datapoints(
        s11, s21, name="nan", reference_impedance=50.0
    )


class _Marker:
    name = "M1"
    location = 0

    def __init__(self):
        self.frequency = ""

    def setFrequency(self, frequency):
        self.frequency = frequency


class _Data:
    def __init__(self, s11, s21=()):
        self.s11 = list(s11)
        self.s21 = list(s21)


class _App:
    """The parts of the main window an analysis reads."""

    reference_impedance = 50.0

    def __init__(self, s11, s21=()):
        self.data = _Data(s11, s21)
        self.markers = [_Marker() for _ in range(4)]


@pytest.fixture
def qt_app(qt_app):
    """Depend on the session-wide application for widget construction."""
    return qt_app


class TestAnalyticPeakSearch:
    @staticmethod
    def test_a_gap_does_not_hide_a_real_minimum():
        # The dip at index 3 is the only one; a gap elsewhere must not hide it.
        for position in (0, 1, 10 - 1):
            assert At.minima(_planted(VSWR_SWEEP, position), 3.0) == [3]

    @staticmethod
    def test_a_gap_is_never_reported_as_a_minimum():
        assert At.minima(_planted(VSWR_SWEEP, 3), 3.0) == []

    @staticmethod
    @pytest.mark.parametrize(
        ("position", "expected"),
        [(0, [5, 15]), (10, [5, 15]), (20, [5, 15]), (5, [15])],
    )
    def test_a_gap_does_not_hide_a_real_maximum(position, expected):
        assert At.maxima(_planted(TWO_PEAKS, position), 4.0) == expected

    @staticmethod
    def test_a_gap_is_never_reported_as_a_maximum():
        assert At.maxima(_planted(TWO_PEAKS, 5), 4.0) == [15]
        assert At.maxima(_planted(TWO_PEAKS, 15), 4.0) == [5]

    @staticmethod
    def test_an_infinite_value_is_never_dropped_as_undefined():
        # ``inf`` is a real measurement.  It is not a peak *shape* (its width
        # is undefined, so ``find_peaks`` refuses it) but it must not be
        # treated like a gap either: the neighbouring dip still stands.
        assert At.minima(_planted(VSWR_SWEEP, 9, INF), 3.0) == [3]

    @staticmethod
    def test_nothing_measured_finds_nothing():
        assert At.minima([NAN] * 5, 3.0) == []
        assert At.maxima([NAN] * 5, 3.0) == []


class TestAnalyticCentre:
    @staticmethod
    def test_a_gap_outside_the_window_does_not_move_the_centre():
        gains = [-3.0, -2.0, -1.0, -2.0, -3.0, -4.0, -5.0]
        for position in (0, 5, 6):
            assert At.center_from_idx(_planted(gains, position), 2) == 2

    @staticmethod
    def test_an_unusable_window_reports_not_found():
        gains = [-3.0, -2.0, NAN, -2.0, -3.0]
        assert At.center_from_idx(gains, 2) == -1
        assert At.center_from_idx([NAN] * 5, 2) == -1


class TestCrossings:
    @staticmethod
    def test_a_gap_does_not_hide_a_distant_crossing():
        values = [-1.0, -0.5, 0.5, 1.0, -1.0]
        assert At.zero_crossings(_planted(values, 0)) == [2, 4]
        assert At.zero_crossings(_planted(values, 4)) == [2]

    @staticmethod
    def test_a_crossing_inside_the_gap_is_not_invented():
        # The NaN replaces the positive sample, so index 2-3 was never a
        # measured sign change and must not be reported.
        values = [-1.0, -0.5, 0.5, 1.0, -1.0]
        assert At.zero_crossings(_planted(values, 2)) == [4]
        assert At.zero_crossings(_planted(values, 3)) == [2]
        assert At.zero_crossings([NAN] * 4) == []


class TestSimplePeakSearch:
    @pytest.mark.parametrize("position", POSITIONS)
    def test_the_match_is_found_from_any_gap_position(self, qt_app, position):
        analysis = SimplePeakSearchAnalysis(
            _App(_points(_planted(S11_MAGNITUDES, position)))
        )
        analysis.button["vswr"].setChecked(True)
        analysis.button["peak_l"].setChecked(True)
        analysis.runAnalysis()
        assert analysis.label["peak_freq"].text() == "1.00300MHz"
        assert analysis.label["peak_db"].text() == "1.105"

    def test_an_infinite_reading_still_wins_the_high_search(self, qt_app):
        analysis = SimplePeakSearchAnalysis(
            _App(_points(_planted(S11_MAGNITUDES, 2, 2.0)))
        )
        analysis.button["vswr"].setChecked(True)
        analysis.button["peak_h"].setChecked(True)
        analysis.runAnalysis()
        assert analysis.label["peak_freq"].text() == "1.00200MHz"
        assert analysis.label["peak_db"].text() == "\N{INFINITY}"

    def test_the_gain_search_ignores_a_gap(self, qt_app):
        gains = [0.5] * 9
        gains[4] = 0.05
        analysis = SimplePeakSearchAnalysis(
            _App(_points([0.3] * 9), _points(_planted(gains, 0)))
        )
        analysis.button["gain"].setChecked(True)
        analysis.button["peak_l"].setChecked(True)
        analysis.runAnalysis()
        assert analysis.label["peak_freq"].text() == "1.00400MHz"

    def test_nothing_measured_leaves_the_labels_empty(self, qt_app):
        analysis = SimplePeakSearchAnalysis(_App(_points([NAN] * 5)))
        analysis.button["vswr"].setChecked(True)
        analysis.runAnalysis()
        assert analysis.label["peak_freq"].text() == ""
        assert analysis.label["peak_db"].text() == ""
        assert (
            analysis.label["result"].text() == "No measured samples to search"
        )


class TestBandStopCentre:
    #: Highest gain at index 3 (-3 dB), next at index 2 (-6 dB).
    GAINS: ClassVar[list[float]] = [-20.0, -10.0, -6.0, -3.0, -9.0]

    @pytest.mark.parametrize(
        ("position", "expected"), [(0, 3), (2, 3), (4, 3), (3, 2)]
    )
    def test_the_highest_measured_gain_is_the_centre(
        self, qt_app, position, expected
    ):
        analysis = BandStopAnalysis(_App([]))
        assert analysis.find_center(_planted(self.GAINS, position)) == expected

    def test_nothing_measured_reports_no_centre(self, qt_app):
        analysis = BandStopAnalysis(_App([]))
        assert analysis.find_center([NAN] * 4) == -1
        assert analysis.label["result"].text() == "No measured S21 magnitudes"


class TestCableAnalysis:
    #: |Z| rises with the reflection magnitude, so 0.1 is the minimum.
    S11: ClassVar[list[float]] = [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.parametrize(
        ("position", "lowest", "highest"),
        [(1, 0, 4), (2, 0, 4), (0, 1, 4), (4, 0, 3)],
    )
    def test_the_impedance_range_ignores_an_undefined_sample(
        self, position, lowest, highest
    ):
        points = _points(self.S11)
        planted = list(points)
        planted[position] = Datapoint(planted[position].freq, NAN, NAN)
        result = analyse_cable(_snapshot(planted))
        assert result.available
        assert result.value("Min |Z|").frequency == points[lowest].freq
        assert result.value("Max |Z|").frequency == points[highest].freq
        assert result.parameters["points"] == len(self.S11) - 1

    def test_the_mean_covers_only_measured_samples(self):
        planted = _points(self.S11)
        planted[2] = Datapoint(planted[2].freq, NAN, NAN)
        clean = analyse_cable(_snapshot(_points(self.S11)))
        result = analyse_cable(_snapshot(planted))
        # |Z| of the dropped sample is 50*(1+0.3)/(1-0.3) = 92.86 ohm,
        # below the mean, so dropping it *raises* the mean of the rest.
        remaining = [
            abs(Datapoint(0, magnitude, 0.0).impedance(50.0))
            for magnitude in (0.1, 0.2, 0.4, 0.5)
        ]
        assert result.value("Mean |Z|").value == pytest.approx(
            sum(remaining) / len(remaining)
        )
        assert result.value("Mean |Z|").value != pytest.approx(
            clean.value("Mean |Z|").value
        )

    def test_every_sample_undefined_is_unavailable(self):
        result = analyse_cable(
            _snapshot([Datapoint(1_000_000 + i, NAN, NAN) for i in range(4)])
        )
        assert not result.available
        assert result.status is AnalysisStatus.UNAVAILABLE
        assert result.kind is AnalysisKind.CABLE
        assert "no measured S11 magnitudes" in result.reason


class TestComparisonSummary:
    @staticmethod
    def _comparison(values, reference=0.5):
        collection = TraceCollection()
        left = collection.add(_snapshot(_points(values)))
        right = collection.add(_snapshot(_points([reference] * len(values))))
        return compare_traces(left, right, kind=ComparisonKind.MAGNITUDE_DB)

    @pytest.mark.parametrize("position", POSITIONS)
    def test_the_range_names_only_measured_differences(self, position):
        # Reference 0.5: 0.25 is -6.021 dB, 1.0 is +6.021 dB, 0.5 is 0 dB.
        # Every planted position holds a 0 dB sample, so the extremes stand.
        values = [0.5, 0.25, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5]
        summary = self._comparison(_planted(values, position)).summary()
        assert "min -6.021 dB" in summary
        assert "max 6.021 dB" in summary
        assert "nan" not in summary

    @staticmethod
    def test_the_extremes_survive_a_gap_between_them():
        values = [0.25, 0.5, NAN, 0.5, 1.0, 0.5]
        summary = TestComparisonSummary._comparison(values).summary()
        assert "min -6.021 dB" in summary
        assert "max 6.021 dB" in summary

    @staticmethod
    def test_nothing_measured_keeps_the_span_and_says_so():
        result = TestComparisonSummary._comparison([NAN] * 4)
        assert result.available
        assert result.summary().endswith(
            "no measured values, 4 points, 1-1.003 MHz"
        )


class TestMainWindowReadouts:
    """The summary labels above the charts name the best measured sample."""

    @staticmethod
    @pytest.fixture
    def window(qt_app):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        app = NanoVNASaver()
        try:
            yield app
        finally:
            app.session.dirty = False
            app.worker.quit()
            app.worker.wait(5000)
            app.deleteLater()

    @staticmethod
    def _update(window, magnitudes, transmission=()):
        window.data.s11 = _points(magnitudes)
        window.data.s21 = _points(transmission)
        window.dataUpdated()

    @pytest.mark.parametrize("position", POSITIONS)
    def test_the_minimum_vswr_names_the_measured_match(self, window, position):
        self._update(window, _planted(S11_MAGNITUDES, position))
        assert window.s11_min_swr_label.text() == "1.105 @ 1.00300MHz"

    def test_the_return_loss_belongs_to_the_same_sample(self, window):
        self._update(window, S11_MAGNITUDES)
        # The matched sample is a real 0.05 reflection: 26.02 dB of return
        # loss, not the -inf of a reflection coefficient of exactly zero.
        assert window.s11_min_rl_label.text() == "-26.021 dB"

    def test_a_total_reflection_is_reported_not_dropped(self, window):
        # Every sample is a total reflection: the search has no undefined
        # samples to exclude, and the honest answer is infinity, not a blank.
        self._update(window, [2.0] * 4)
        assert window.s11_min_swr_label.text() == "\N{INFINITY} @ 1.00000MHz"

    @pytest.mark.parametrize("position", POSITIONS)
    def test_the_gain_readouts_name_measured_samples(self, window, position):
        gains = [0.5] * 10
        gains[2] = 0.25
        gains[6] = 2.0
        self._update(
            window, [0.3] * 10, transmission=_planted(gains, position, NAN)
        )
        # 0.25 is -12.041 dB, 2.0 is 6.021 dB; the NaN is neither.
        assert window.s21_min_gain_label.text() == "-12.041 dB @ 1.00200MHz"
        assert window.s21_max_gain_label.text() == "6.021 dB @ 1.00600MHz"

    def test_an_infinite_gain_is_reported_not_dropped(self, window):
        # A zero transmission is 0 dB, so the maximum of a sweep of zeros is
        # the measurement itself; gain is -inf for silence and stays so.
        window.data.s11 = _points([0.3] * 3)
        window.data.s21 = [
            Datapoint(1_000_000, 0.5, 0.0),
            Datapoint(1_001_000, NAN, NAN),
            Datapoint(1_002_000, 0.0, 0.0),
        ]
        window.dataUpdated()
        assert window.s21_min_gain_label.text() == (
            "-\N{INFINITY} dB @ 1.00200MHz"
        )
        assert window.s21_max_gain_label.text() == "-6.021 dB @ 1.00000MHz"

    def test_nothing_measured_clears_the_labels(self, window):
        self._update(window, [NAN] * 4, transmission=[NAN] * 4)
        assert window.s11_min_swr_label.text() == ""
        assert window.s11_min_rl_label.text() == ""
        assert window.s21_min_gain_label.text() == ""
        assert window.s21_max_gain_label.text() == ""
