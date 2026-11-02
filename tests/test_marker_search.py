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
"""ANALYSIS-01: marker search, tracking, bandwidth and Q.

Every expected value here comes from an analytic curve written out in the
test, not from re-running the implementation: a Gaussian resonance has a known
-3 dB width, a straight line has one known crossing, and a constant has none.
"""

import math

import pytest

from NanoVNASaver.MarkerSearch import (
    BandwidthResult,
    LocationKind,
    MarkerRule,
    SearchKind,
    SearchStatus,
    bandwidth_3db,
    search_crossing,
    search_extreme,
    search_local,
    track_markers,
    unwrapped_phase,
)
from NanoVNASaver.RFTools import Datapoint

MHZ = 1_000_000
CENTRE = 100 * MHZ
STEP = 100_000


def _from_db(values):
    """Reflection samples whose |S11| in dB matches ``values``.

    The grid is centred on :data:`CENTRE`, so the middle entry of an odd
    list sits exactly there.
    """
    start = CENTRE - (len(values) // 2) * STEP
    points = []
    for index, db in enumerate(values):
        magnitude = 10 ** (db / 20.0)
        points.append(Datapoint(start + index * STEP, magnitude, 0.0))
    return points


def _gaussian(centre_db=-1.0, sigma_hz=3 * STEP, points=101):
    """A Gaussian response in dB with a known half-power width.

    ``-3 dB`` from the peak happens at ``sigma * sqrt(2 ln 2)``, which is the
    analytic reference the tests compare against.
    """
    values = []
    for index in range(points):
        offset = (index - points // 2) * STEP
        values.append(centre_db - 6.0206 * (offset / sigma_hz) ** 2 / 2.0)
    return _from_db(values)


def freq_of(points, index):
    """Frequency of sample ``index``: expectations come from the grid."""
    return points[index].freq


class TestGlobalExtremes:
    @staticmethod
    def test_the_maximum_is_a_measured_sample():
        points = _from_db([-10, -5, -1, -5, -10])
        result = search_extreme(points, maximum=True)
        assert result.ok
        assert result.location is LocationKind.MEASURED
        assert result.frequency == freq_of(points, 2)
        assert result.value == pytest.approx(-1.0)

    @staticmethod
    def test_the_minimum_is_a_measured_sample():
        points = _from_db([-1, -5, -20, -5, -1])
        result = search_extreme(points, maximum=False)
        assert result.frequency == freq_of(points, 2)
        assert result.value == pytest.approx(-20.0)

    @staticmethod
    def test_a_flat_curve_picks_the_lowest_frequency_deterministically():
        points = _from_db([-3, -3, -3, -3])
        result = search_extreme(points, maximum=True)
        assert result.frequency == freq_of(points, 0)

    @staticmethod
    def test_non_finite_samples_are_skipped_not_treated_as_extremes():
        points = _from_db([-10, -1, -5])
        broken = Datapoint(freq_of(points, 0), math.nan, 0.0)
        result = search_extreme(
            [points[0], broken, points[1], points[2]], maximum=True
        )
        assert result.frequency == freq_of(points, 1)

    @staticmethod
    def test_an_empty_band_is_unavailable_with_a_reason():
        points = _from_db([-1, -2, -3])
        result = search_extreme(points, start=10**12, stop=2 * 10**12)
        assert result.status is SearchStatus.UNAVAILABLE
        assert "no finite samples" in result.reason

    @staticmethod
    def test_a_band_restricts_the_search():
        points = _from_db([-1, -10, -20, -30])
        widened = search_extreme(points, maximum=True)
        narrowed = search_extreme(
            points, maximum=True, start=freq_of(points, 2)
        )
        assert widened.value == pytest.approx(-1.0)
        assert narrowed.value == pytest.approx(-20.0)


class TestEstimatedLocations:
    @staticmethod
    def test_a_parabolic_vertex_between_samples_is_estimated():
        """The fitted vertex lies toward the shallower shoulder.

        Samples ``-10, -2, -5`` at equal spacing fit a parabola with
        ``a = -5.5``, ``b = 2.5``, so its vertex is at ``x = 0.227``: just
        past the peak, toward the sample that is only 3 dB lower.  That is a
        property of the three samples, not of the implementation, and the
        result must mark it as an estimate rather than claim a measurement.
        """
        points = _from_db([-10, -2, -5, -12])
        result = search_extreme(points, maximum=True, estimate=True)
        assert result.ok
        assert result.location is LocationKind.ESTIMATED
        assert freq_of(points, 1) < result.frequency < freq_of(points, 2)
        # The fitted peak is higher than the best measured sample, which is
        # exactly what "estimated" means here.
        assert result.value > -2.0

    @staticmethod
    def test_an_endpoint_extremum_is_never_extrapolated():
        points = _from_db([-1, -2, -3, -4])
        result = search_extreme(points, maximum=True, estimate=True)
        assert result.location is LocationKind.MEASURED
        assert result.frequency == freq_of(points, 0)

    @staticmethod
    def test_a_plateau_is_not_turned_into_an_invented_peak():
        points = _from_db([-1, -1, -1, -5])
        result = search_extreme(points, maximum=True, estimate=True)
        assert result.location is LocationKind.MEASURED


class TestLocalSearches:
    @staticmethod
    def test_the_next_peak_to_the_right_is_found():
        points = _from_db([-20, -10, -2, -10, -3, -10])
        result = search_local(
            points, freq_of(points, 0), maximum=True, direction=1
        )
        assert result.frequency == freq_of(points, 2)
        assert result.location is LocationKind.MEASURED

    @staticmethod
    def test_the_next_notch_to_the_left_is_found():
        points = _from_db([-10, -30, -10, -2, -20])
        result = search_local(
            points, freq_of(points, 2), maximum=False, direction=-1
        )
        assert result.frequency == freq_of(points, 1)

    @staticmethod
    def test_a_monotonic_run_reports_its_end_not_an_invented_peak():
        points = _from_db([-10, -8, -6, -4, -2])
        result = search_local(
            points, freq_of(points, 0), maximum=True, direction=1
        )
        assert result.frequency == freq_of(points, -1)

    @staticmethod
    def test_a_search_past_the_last_sample_is_unavailable():
        points = _from_db([-1, -2, -3])
        result = search_local(
            points, freq_of(points, -1), maximum=True, direction=1
        )
        assert result.status is SearchStatus.UNAVAILABLE
        assert "beyond the reference" in result.reason

    @staticmethod
    def test_an_invalid_direction_is_rejected():
        with pytest.raises(ValueError):
            search_local(_from_db([-1, -2]), CENTRE, direction=2)


class TestCrossings:
    @staticmethod
    def test_a_crossing_between_samples_is_estimated_linearly():
        # A straight line rising 2 dB per 10 MHz crosses 1 dB exactly half way
        # between the first two samples: an analytic value, not a fitted one.
        points = _from_db([0, 2, 4, 6, 8])
        result = search_crossing(points, 1.0, above=True)
        assert result.ok
        assert result.frequency == pytest.approx(
            (freq_of(points, 0) + freq_of(points, 1)) / 2
        )
        assert result.location is LocationKind.ESTIMATED

    @staticmethod
    def test_a_sample_exactly_on_the_target_is_measured():
        points = _from_db([0, 5, 10])
        result = search_crossing(points, 5.0, above=True)
        assert result.location is LocationKind.MEASURED
        assert result.frequency == freq_of(points, 1)

    @staticmethod
    def test_no_crossing_reports_why():
        points = _from_db([0, 1, 2])
        result = search_crossing(points, 50.0, above=True)
        assert result.status is SearchStatus.UNAVAILABLE
        assert "no crossing" in result.reason

    @staticmethod
    def test_a_downward_crossing_is_found_separately():
        points = _from_db([10, 7.5, 5, 2.5, 0])
        up = search_crossing(points, 5.0, above=True)
        down = search_crossing(points, 5.0, above=False)
        assert up.status is SearchStatus.UNAVAILABLE
        assert down.ok

    @staticmethod
    def test_a_single_sample_cannot_produce_a_crossing():
        result = search_crossing(_from_db([1]), 0.5, above=True)
        assert result.status is SearchStatus.UNAVAILABLE
        assert "at least two samples" in result.reason


class TestBandwidth:
    @staticmethod
    def test_a_gaussian_notch_bandwidth_matches_the_analytic_width():
        sigma = 5 * STEP
        points = _gaussian(sigma_hz=sigma)
        result = bandwidth_3db(points, channel="S11")
        assert result.status is SearchStatus.OK
        analytic = 2.0 * sigma * math.sqrt(2.0 * math.log(2.0))
        # The estimate comes from linear interpolation between two samples, so
        # it is bounded by the sample spacing on each side.
        assert result.bandwidth == pytest.approx(analytic, abs=2 * STEP)
        assert result.lower < result.centre < result.upper

    @staticmethod
    def test_q_is_centre_over_bandwidth():
        points = _gaussian(sigma_hz=5 * STEP)
        result = bandwidth_3db(points)
        assert result.quality_factor == pytest.approx(
            result.centre / result.bandwidth
        )

    @staticmethod
    def test_one_missing_edge_is_unavailable_not_completed_with_a_band_edge():
        # A monotonic rise never has a lower -3 dB edge.
        points = _from_db([-60, -40, -20, -2])
        result = bandwidth_3db(points)
        assert result.status is SearchStatus.UNAVAILABLE
        assert "edge is not inside" in result.reason

    @staticmethod
    def test_too_few_samples_is_unavailable():
        result = bandwidth_3db(_from_db([-1, -2]))
        assert result.status is SearchStatus.UNAVAILABLE
        assert "three finite samples" in result.reason

    @staticmethod
    def test_a_flat_curve_has_no_bandwidth():
        result = bandwidth_3db(_from_db([-3] * 21))
        assert result.status is SearchStatus.UNAVAILABLE


class TestPhaseUnwrapping:
    @staticmethod
    def test_a_wrap_is_removed_monotonically():
        angles = [170, -170, -150]
        points = [
            Datapoint(
                CENTRE + index * STEP,
                math.cos(math.radians(a)),
                math.sin(math.radians(a)),
            )
            for index, a in enumerate(angles)
        ]
        values = [value for _freq, value in unwrapped_phase(points)]
        assert values[1] - values[0] == pytest.approx(20.0)
        assert values == sorted(values)

    @staticmethod
    def test_an_unwrapped_series_keeps_its_original_angles():
        points = [
            Datapoint(
                CENTRE, math.cos(math.radians(a)), math.sin(math.radians(a))
            )
            for a in (0, 10, 20)
        ]
        values = [value for _freq, value in unwrapped_phase(points)]
        assert values == pytest.approx([0.0, 10.0, 20.0])


class TestTracking:
    @staticmethod
    def test_every_rule_reports_against_its_own_channel():
        s11 = _from_db([-10, -1, -10])
        s21 = _from_db([-30, -20, -10])
        rules = [
            MarkerRule("s11 peak", SearchKind.MAXIMUM, channel="S11"),
            MarkerRule("s21 peak", SearchKind.MAXIMUM, channel="S21"),
        ]
        results = track_markers(rules, {"S11": s11, "S21": s21})
        assert results["s11 peak"].frequency == freq_of(s11, 1)
        assert results["s21 peak"].frequency == freq_of(s21, 2)

    @staticmethod
    def test_a_missing_channel_is_unavailable_not_borrowed():
        rules = [MarkerRule("s21 peak", SearchKind.MAXIMUM, channel="S21")]
        results = track_markers(rules, {"S11": _from_db([-1, -2])})
        assert results["s21 peak"].status is SearchStatus.UNAVAILABLE
        assert "no S21 channel" in results["s21 peak"].reason

    @staticmethod
    def test_a_local_rule_without_a_reference_is_unavailable():
        rule = MarkerRule("left", SearchKind.PEAK_LEFT)
        result = rule.apply(_from_db([-1, -2, -3]))
        assert result.status is SearchStatus.UNAVAILABLE
        assert "reference frequency" in result.reason

    @staticmethod
    def test_a_crossing_rule_without_a_target_is_unavailable():
        rule = MarkerRule("cross", SearchKind.CROSSING_ABOVE)
        result = rule.apply(_from_db([-1, -2, -3]))
        assert result.status is SearchStatus.UNAVAILABLE
        assert "target value" in result.reason

    @staticmethod
    def test_each_search_kind_is_reachable_from_a_rule():
        points = _from_db([-20, -2, -10, -30, -5])
        cases = {
            SearchKind.MAXIMUM: None,
            SearchKind.MINIMUM: None,
            SearchKind.PEAK_LEFT: freq_of(points, 2),
            SearchKind.PEAK_RIGHT: freq_of(points, 0),
            SearchKind.NOTCH_LEFT: freq_of(points, 2),
            SearchKind.NOTCH_RIGHT: freq_of(points, 0),
            SearchKind.CROSSING_ABOVE: None,
            SearchKind.CROSSING_BELOW: None,
        }
        for kind, reference in cases.items():
            rule = MarkerRule(
                kind.value,
                kind,
                reference=reference,
                target=-15.0,
                maximum=kind in (SearchKind.PEAK_LEFT, SearchKind.PEAK_RIGHT),
            )
            result = rule.apply(points)
            # Every kind must produce a decision, even when it is "no".
            assert isinstance(result.status, SearchStatus)


class TestResultDescriptions:
    @staticmethod
    def test_a_measured_result_says_so():
        result = search_extreme(_from_db([-1, -5, -10]))
        assert "measured" in result.describe()

    @staticmethod
    def test_an_unavailable_result_says_why():
        result = search_extreme([], start=0, stop=1)
        text = result.describe()
        assert "unavailable" in text
        assert "no finite samples" in text

    @staticmethod
    def test_a_bandwidth_description_carries_q_and_edges():
        result = bandwidth_3db(_gaussian())
        text = result.describe()
        assert "Q" in text
        assert "bandwidth" in text

    @staticmethod
    def test_an_unavailable_bandwidth_description_says_why():
        result = BandwidthResult(
            status=SearchStatus.UNAVAILABLE, reason="nothing to measure"
        )
        assert "nothing to measure" in result.describe()
