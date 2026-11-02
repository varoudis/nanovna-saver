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
"""ANALYSIS-04: group delay and phase curvature.

Every expectation here is an analytic reference written out in the test:

* a delay line is ``S21 = exp(-j*omega*tau)``, so ``-d(phi)/d(omega) = tau``;
* a constant phase has neither slope nor curvature;
* a phase ``phi(omega) = K*omega**2/2`` has curvature ``K`` and group delay
  ``-K*omega`` exactly;
* a linear phase has zero curvature.

A complex sample can only carry a phase modulo ``2*pi``, so "the unwrapped
version of the same data" means the analytic phase the samples were built
from: the wrap test compares a wrapping sweep with a sweep of the same line
that never leaves ``(-pi, pi)``, and it shows that a raw difference of the
wrapped samples would be wrong by orders of magnitude.
"""

import itertools
import math
import random
import statistics
from dataclasses import FrozenInstanceError

import pytest

from NanoVNASaver.PhaseDerivatives import (
    METHOD,
    SMOOTHING,
    SUPPORTED_CHANNELS,
    DerivativePoint,
    DerivativeStatus,
    PhaseDerivative,
    PhaseDerivativeError,
    PhaseQuantity,
    available_channels,
    diagnostics,
    group_delay,
    phase_curvature,
    unwrap_phase,
)
from NanoVNASaver.RFTools import Datapoint

TWO_PI = 2.0 * math.pi
TAU = 3.7e-9
#: phi(omega) = K*omega**2/2  =>  curvature K, group delay -K*omega.
K = 2.0e-16
QUADRATIC_FREQS = [10_000_000 + index * 400_000 for index in range(101)]


def omega_of(freq):
    """Angular frequency, as the module computes it (same expression)."""
    return TWO_PI * float(freq)


def curve(freqs, phase_of_omega):
    """Unit-magnitude samples, phase modulo 2*pi equal to ``phase_of_omega``."""
    points = []
    for freq in freqs:
        phase = phase_of_omega(omega_of(freq))
        points.append(Datapoint(freq, math.cos(phase), math.sin(phase)))
    return points


def linear_grid(count=101, start=10_000_000, step=5_000_000):
    return [start + index * step for index in range(count)]


def delay_line(freqs, tau=TAU):
    """Pure delay: ``phi = -omega*tau``, a decreasing linear phase."""
    return curve(freqs, lambda omega: -omega * tau)


def quadratic_line(freqs=QUADRATIC_FREQS):
    return curve(freqs, lambda omega: 0.5 * K * omega**2)


def irregular_grid(count=101):
    """Strictly increasing but non-uniform: 2.25 MHz and 2.00 MHz steps."""
    return [
        100_000_000 + 2_000_000 * index + (index % 4) * 250_000
        for index in range(count)
    ]


class TestPureDelay:
    @staticmethod
    def test_group_delay_is_the_delay_at_every_sample():
        result = group_delay({"S21": delay_line(linear_grid())})
        assert result.status is DerivativeStatus.OK
        assert len(result.points) == 101
        for point in result.points:
            assert point.value == pytest.approx(TAU, abs=1e-15)

    @staticmethod
    def test_a_line_delays_with_a_positive_sign():
        result = group_delay({"S21": delay_line(linear_grid())})
        assert result.median == pytest.approx(TAU, abs=1e-15)
        assert min(result.values) > 0.0

    @staticmethod
    def test_every_sample_matches_a_central_difference_of_the_analytic_phase():
        freqs = linear_grid()
        result = group_delay({"S21": delay_line(freqs)})
        for index in range(1, len(freqs) - 1):
            low = omega_of(freqs[index - 1])
            high = omega_of(freqs[index + 1])
            # central difference of phi = -tau*omega with equal steps
            expected = -((-TAU * high) - (-TAU * low)) / (high - low)
            assert expected == pytest.approx(TAU, abs=1e-18)
            assert result.value_at(freqs[index]) == pytest.approx(
                expected, abs=1e-15
            )

    @staticmethod
    def test_the_sweep_actually_wraps():
        """Guards the wrap test: data that never wraps would prove nothing."""
        result = group_delay({"S21": delay_line(linear_grid())})
        assert result.wraps_corrected > 0

    @staticmethod
    def test_a_wrapping_curve_matches_one_that_never_wraps():
        # Same line, two sweeps.  The narrow one advances only 0.023 rad per
        # step and stays inside (-pi, pi), so its unwrapped phase is its own
        # phase; the wide one sweeps 2.35 rad in total and wraps.
        narrow = [1_000_000 * (index + 1) for index in range(101)]
        assert TWO_PI * 101_000_000 * TAU < math.pi
        narrow_result = group_delay({"S21": delay_line(narrow)})
        wide_result = group_delay({"S21": delay_line(linear_grid())})
        assert narrow_result.wraps_corrected == 0
        assert wide_result.wraps_corrected > 0
        for narrow_value, wide_value in zip(
            narrow_result.values, wide_result.values, strict=True
        ):
            assert narrow_value == pytest.approx(TAU, abs=1e-15)
            assert wide_value == pytest.approx(narrow_value, abs=1e-15)

    @staticmethod
    def test_unwrapping_recovers_the_analytic_phase():
        freqs = linear_grid()
        phase = unwrap_phase(delay_line(freqs))
        for freq, value in zip(freqs, phase, strict=True):
            assert value == pytest.approx(-omega_of(freq) * TAU, abs=1e-9)

    @staticmethod
    def test_ignoring_the_wrap_would_be_wrong():
        """A raw difference across the branch is nowhere near the delay."""
        freqs = linear_grid()
        points = delay_line(freqs)
        step = omega_of(freqs[1]) - omega_of(freqs[0])
        # The sweep advances 0.116 rad per sample, so the branch is crossed
        # at a handful of sample pairs; those are the ones a raw difference
        # gets wrong by a whole 2*pi.  At least one must exist for the test
        # to mean anything.
        crossings = [
            index
            for index in range(len(points) - 1)
            if abs(points[index + 1].phase - points[index].phase) > math.pi
        ]
        assert crossings
        index = crossings[0]
        raw = -(points[index + 1].phase - points[index].phase) / step
        assert abs(raw - TAU) > 1e-9
        assert group_delay({"S21": points}).value_at(
            freqs[index]
        ) == pytest.approx(TAU, abs=1e-15)

    @staticmethod
    def test_two_samples_still_give_a_two_point_difference():
        freqs = [10_000_000, 20_000_000]
        result = group_delay({"S21": delay_line(freqs)})
        assert result.status is DerivativeStatus.OK
        assert [point.value for point in result.points] == pytest.approx(
            [TAU, TAU], abs=1e-15
        )


class TestConstantPhase:
    # 0.7 rad, so the analytic phase is not a multiple of pi/2.
    POINTS = curve(linear_grid(), lambda omega: 0.7)

    @staticmethod
    def test_group_delay_is_zero():
        result = group_delay({"S21": TestConstantPhase.POINTS})
        assert result.status is DerivativeStatus.OK
        for point in result.points:
            assert point.value == pytest.approx(0.0, abs=1e-18)

    @staticmethod
    def test_curvature_is_zero():
        result = phase_curvature({"S21": TestConstantPhase.POINTS})
        assert result.status is DerivativeStatus.OK
        for point in result.points:
            assert point.value == pytest.approx(0.0, abs=1e-24)


class TestCurvature:
    @staticmethod
    def test_a_linear_phase_has_no_curvature():
        result = phase_curvature({"S21": delay_line(linear_grid())})
        assert result.status is DerivativeStatus.OK
        for point in result.points:
            assert point.value == pytest.approx(0.0, abs=1e-24)

    @staticmethod
    def test_quadratic_phase_has_the_analytic_curvature():
        result = phase_curvature({"S21": quadratic_line()})
        assert result.status is DerivativeStatus.OK
        assert len(result.points) == 101
        for point in result.points:
            assert point.value == pytest.approx(K, rel=1e-9)

    @staticmethod
    def test_quadratic_phase_has_the_analytic_group_delay():
        result = group_delay({"S21": quadratic_line()})
        for point, freq in zip(result.points, QUADRATIC_FREQS, strict=True):
            expected = -K * omega_of(freq)
            assert point.value == pytest.approx(expected, rel=1e-9, abs=1e-20)

    @staticmethod
    def test_curvature_is_labelled_a_diagnostic_and_carries_its_unit():
        result = phase_curvature({"S21": quadratic_line()})
        assert result.quantity is PhaseQuantity.PHASE_CURVATURE
        assert result.unit == "s^2"
        assert "diagnostic" in result.describe()
        assert group_delay({"S21": quadratic_line()}).unit == "s"


class TestIrregularGrid:
    @staticmethod
    def test_known_delay_is_recovered_with_the_local_steps():
        freqs = irregular_grid()
        result = group_delay({"S21": delay_line(freqs)})
        assert result.status is DerivativeStatus.OK
        assert len(result.points) == len(freqs)
        for point in result.points:
            assert point.value == pytest.approx(TAU, abs=1e-15)

    @staticmethod
    def test_assuming_a_uniform_step_would_be_wrong():
        # The grid steps cycle 2.25, 2.25, 2.25, 1.25 MHz.  A nominal step
        # taken from the first pair is 2.25 MHz, and the 3 -> 4 pair is
        # 1.25 MHz, so a uniform-step difference is wrong by 80% there.
        freqs = irregular_grid()
        points = delay_line(freqs)
        assert freqs[4] - freqs[3] == 1_250_000
        nominal = TWO_PI * (freqs[1] - freqs[0])
        uniform = -(points[4].phase - points[3].phase) / nominal
        local = -(
            (points[4].phase - points[3].phase)
            / (omega_of(freqs[4]) - omega_of(freqs[3]))
        )
        assert abs(uniform - TAU) > 1e-10
        assert local == pytest.approx(TAU, abs=1e-15)

    @staticmethod
    def test_quadratic_phase_on_an_irregular_grid_keeps_its_curvature():
        freqs = irregular_grid()
        result = phase_curvature({"S21": quadratic_line(freqs)})
        assert result.status is DerivativeStatus.OK
        for point in result.points:
            assert point.value == pytest.approx(K, rel=1e-9)

    @staticmethod
    def test_quadratic_phase_on_an_irregular_grid_keeps_its_group_delay():
        # phi(omega) = K*omega**2/2 has -d(phi)/d(omega) = -K*omega at every
        # sample, and the three-point stencil is exact for a quadratic whatever
        # the two local steps are.  Use a K small enough that the phase moves
        # far less than pi between samples, so the unwrapping assumption holds
        # and the comparison is about the stencil alone.
        k_small = 1.0e-18
        freqs = irregular_grid()
        points = curve(freqs, lambda omega: 0.5 * k_small * omega**2)
        result = group_delay({"S21": points})
        assert result.status is DerivativeStatus.OK
        assert not result.wrap_ambiguous
        for point in result.points:
            analytic = -k_small * omega_of(point.freq)
            assert point.value == pytest.approx(analytic, rel=1e-9)
            assert point.omega == pytest.approx(omega_of(point.freq))

    @staticmethod
    def test_a_delay_beyond_the_grid_ambiguity_is_flagged_not_trusted():
        # The same irregular grid has a 1.25 MHz minimum step, so any phase
        # moving by more than pi between those samples is aliased: a delay
        # longer than pi/d_omega = 1/(2*1.25e6) = 400 ns cannot be told from a
        # shorter one.  With K large the quadratic's delay reaches 377 ns at
        # the top of this band while its phase step reaches 5.29 rad > pi, so
        # the unwrapped curve is a different phase and the numbers must not be
        # presented as measured.
        freqs = irregular_grid()
        result = group_delay({"S21": quadratic_line(freqs)})
        smallest_step = TWO_PI * min(
            later - earlier for earlier, later in itertools.pairwise(freqs)
        )
        assert smallest_step == pytest.approx(TWO_PI * 1_250_000)
        assert result.ambiguous_delay == pytest.approx(math.pi / smallest_step)
        assert result.ambiguous_delay == pytest.approx(400e-9)
        # The analytic phase step somewhere in this band exceeds pi.
        steps = [
            0.5 * K * (omega_of(later) ** 2 - omega_of(earlier) ** 2)
            for earlier, later in itertools.pairwise(freqs)
        ]
        assert max(steps) > math.pi
        assert max(steps) == pytest.approx(5.2874, abs=1e-4)
        assert result.wrap_ambiguous
        assert "wrap ambiguity of this grid" in result.describe()

    @staticmethod
    def test_an_honest_delay_is_not_flagged_as_ambiguous():
        result = group_delay({"S21": delay_line(irregular_grid())})
        # 3.7 ns against a 400 ns ambiguity: no warning, and the bound is
        # still reported so a caller can compare without recomputing the grid.
        assert not result.wrap_ambiguous
        assert result.ambiguous_delay == pytest.approx(400e-9)
        assert "wrap ambiguity" not in result.describe()

    @staticmethod
    def test_a_duplicate_frequency_is_refused():
        freqs = irregular_grid(7)
        freqs[3] = freqs[2]
        points = curve(freqs, lambda omega: -omega * TAU)
        for maker in (group_delay, phase_curvature):
            result = maker({"S21": points})
            assert result.status is DerivativeStatus.UNAVAILABLE
            assert "duplicate" in result.reason
            assert result.points == ()
            assert result.median is None
            assert "unavailable" in result.describe()

    @staticmethod
    def test_a_decreasing_frequency_is_refused():
        freqs = irregular_grid(7)
        freqs[4], freqs[5] = freqs[5], freqs[4]
        result = group_delay({"S21": curve(freqs, lambda w: -w * TAU)})
        assert result.status is DerivativeStatus.UNAVAILABLE
        assert "not monotonic" in result.reason


class TestSmoothing:
    @staticmethod
    def test_raw_is_labelled_raw_and_the_aperture_is_reported():
        points = delay_line(linear_grid())
        raw = group_delay({"S21": points})
        assert raw.aperture_points == 1
        assert not raw.smoothed
        assert "raw curve, aperture 1" in raw.describe()
        assert "noise" in raw.window

        smoothed = group_delay({"S21": points}, aperture_points=9)
        assert smoothed.aperture_points == 9
        assert smoothed.smoothed
        assert smoothed.window.startswith("smoothed, aperture 9 samples")
        assert "aperture 9 samples" in smoothed.describe()

    @staticmethod
    def test_smoothing_preserves_a_pure_delay():
        """A symmetric average of a linear-phase line keeps the slope."""
        points = delay_line(linear_grid())
        result = group_delay({"S21": points}, aperture_points=9)
        assert result.status is DerivativeStatus.OK
        for point in result.points:
            assert point.value == pytest.approx(TAU, abs=1e-15)

    @staticmethod
    def test_the_off_centre_values_are_reported_and_land_on_their_centroid():
        # Aperture 9 = half-width 4: the first and last 4 windows are
        # truncated, so their frequency centroid is not the sample's own
        # frequency.  Those values are differentiated against the centroid
        # they actually belong to, and the count says so.
        freqs = linear_grid()
        result = group_delay({"S21": delay_line(freqs)}, aperture_points=9)
        assert result.off_centre == 8
        assert "8 value(s) at window centroids" in result.window
        assert result.undefined_count == 0
        # The reported abscissa of an end point is the window's mean
        # frequency: samples 0..4 for index 0.
        expected = math.fsum(omega_of(freq) for freq in freqs[:5]) / 5
        assert result.points[0].omega == pytest.approx(expected)
        assert result.points[50].omega == pytest.approx(omega_of(freqs[50]))

    @staticmethod
    def test_smoothing_suppresses_curvature_noise():
        generator = random.Random(1234)
        points = []
        for freq in linear_grid():
            phase = -omega_of(freq) * TAU + generator.uniform(-0.01, 0.01)
            points.append(Datapoint(freq, math.cos(phase), math.sin(phase)))
        raw = phase_curvature({"S21": points})
        smooth = phase_curvature({"S21": points}, aperture_points=9)
        raw_dev = statistics.pstdev(raw.values)
        smooth_dev = statistics.pstdev(smooth.values)
        assert raw_dev > 0.0
        assert smooth_dev < raw_dev / 2.0

    @staticmethod
    def test_an_aperture_wider_than_the_record_is_unavailable():
        points = delay_line(irregular_grid(5))
        result = group_delay({"S21": points}, aperture_points=7)
        assert result.status is DerivativeStatus.UNAVAILABLE
        assert "whole 5-sample record" in result.reason
        assert result.aperture_points == 7
        assert result.points == ()

    @staticmethod
    def test_an_aperture_covering_the_whole_record_is_unavailable():
        # Five samples with aperture 5 smooths every sample with every other,
        # so all five values are identical and the phase is flat; a "group
        # delay of zero" would be a fabricated answer, so it is refused.
        points = delay_line(irregular_grid(5))
        assert group_delay({"S21": points}).status is DerivativeStatus.OK
        result = group_delay({"S21": points}, aperture_points=5)
        assert result.status is DerivativeStatus.UNAVAILABLE
        assert "flatten the phase" in result.reason
        assert result.points == ()

    @staticmethod
    def test_smoothing_a_short_record_still_works_when_it_leaves_data():
        # The grid is 100.00, 102.25, 104.50, 106.75, 108.00 MHz.  Aperture 3
        # averages each sample with one neighbour on each side, so the windows
        # at samples 1 and 2 (2.25 MHz steps on both sides) are centred
        # exactly, while samples 0, 3 and 4 are truncated and their centroids
        # differ from their own frequencies.  Every value stays correct
        # because the derivative follows the centroid.
        points = delay_line(irregular_grid(5))
        result = group_delay({"S21": points}, aperture_points=3)
        assert result.status is DerivativeStatus.OK
        assert result.off_centre == 3
        # The residual is the second-order error of averaging phasors across a
        # 0.05 rad window, far below the ~10% a wrong abscissa would cost.
        for point in result.points:
            assert point.value == pytest.approx(TAU, rel=1e-3)

    @staticmethod
    def test_a_useless_aperture_raises():
        points = delay_line(linear_grid())
        for aperture in (0, -3, 4):
            with pytest.raises(PhaseDerivativeError):
                group_delay({"S21": points}, aperture_points=aperture)
        with pytest.raises(PhaseDerivativeError):
            phase_curvature({"S21": points}, aperture_points=2)

    @staticmethod
    def test_diagnostics_share_one_aperture():
        points = delay_line(linear_grid())
        delay, curvature = diagnostics({"S21": points}, aperture_points=5)
        assert delay.aperture_points == curvature.aperture_points == 5
        assert delay.channel == curvature.channel == "S21"


class TestChannelAccess:
    @staticmethod
    def test_s11_and_s21_both_work():
        points = delay_line(linear_grid())
        for channel in ("S11", "S21"):
            delay, curvature = diagnostics({channel: points}, channel)
            assert delay.status is DerivativeStatus.OK
            assert curvature.status is DerivativeStatus.OK
            assert delay.channel == channel
            assert delay.median == pytest.approx(TAU, abs=1e-15)

    @staticmethod
    def test_an_absent_channel_is_refused():
        result = group_delay({"S21": delay_line(linear_grid())}, "S11")
        assert result.status is DerivativeStatus.UNAVAILABLE
        assert "S11" in result.reason
        assert "absent" in result.reason
        assert "S21" in result.reason
        assert not result.ok
        assert result.median is None
        assert "unavailable" in result.describe()

    @staticmethod
    def test_an_empty_mapping_is_refused():
        result = phase_curvature({}, "S11")
        assert result.status is DerivativeStatus.UNAVAILABLE
        assert "none" in result.reason

    @staticmethod
    def test_available_channels_lists_only_what_is_present():
        points = delay_line(linear_grid())
        assert available_channels({"S21": points}) == ("S21",)
        assert available_channels({"S11": points, "S21": points}) == (
            SUPPORTED_CHANNELS
        )
        assert available_channels({"S21": ()}) == ()
        assert available_channels({}) == ()


class TestUndefinedSamples:
    @staticmethod
    def test_a_zero_magnitude_sample_is_undefined_not_zero():
        points = delay_line(linear_grid())
        points[50] = Datapoint(points[50].freq, 0.0, 0.0)
        delay = group_delay({"S21": points})
        curvature = phase_curvature({"S21": points})

        assert delay.status is DerivativeStatus.PARTIAL
        assert delay.value_at(points[50].freq) is None
        assert delay.value_at(points[49].freq) is None
        assert delay.value_at(points[51].freq) is None
        assert delay.undefined_count == 3
        assert delay.value_at(points[48].freq) == pytest.approx(TAU, abs=1e-15)

        # A second difference at index i needs i-1, i, i+1 (or a one-sided
        # trio at the ends), so the single dead sample in the interior kills
        # exactly its own sample: 49, 50 and 51 all have a dead member of
        # their three-point window.
        assert curvature.status is DerivativeStatus.PARTIAL
        assert curvature.undefined_count == 3
        assert curvature.value_at(points[50].freq) is None
        assert curvature.value_at(points[52].freq) is not None
        assert curvature.value_at(points[52].freq) == pytest.approx(
            0.0, abs=1e-24
        )
        assert "undefined sample(s)" in delay.describe()

    @staticmethod
    def test_no_value_is_ever_nan_or_infinite():
        points = delay_line(linear_grid())
        points[10] = Datapoint(points[10].freq, math.nan, 0.0)
        for result in diagnostics({"S21": points}):
            for point in result.points:
                assert point.value is None or math.isfinite(point.value)
            assert all(math.isfinite(value) for value in result.values)

    @staticmethod
    def test_a_single_sample_cannot_be_differentiated():
        delay = group_delay({"S21": delay_line([10_000_000])})
        assert delay.status is DerivativeStatus.UNAVAILABLE
        assert "two samples" in delay.reason

    @staticmethod
    def test_curvature_needs_three_samples():
        curvature = phase_curvature(
            {"S21": delay_line([10_000_000, 20_000_000])}
        )
        assert curvature.status is DerivativeStatus.UNAVAILABLE
        assert "three samples" in curvature.reason
        assert curvature.points == ()

    @staticmethod
    def test_a_probably_impossible_value_is_rejected_at_construction():
        with pytest.raises(ValueError):
            PhaseDerivative(
                quantity=PhaseQuantity.GROUP_DELAY,
                channel="S21",
                status=DerivativeStatus.OK,
                points=(
                    DerivativePoint(1, 1.0, math.nan),
                    DerivativePoint(2, 2.0, math.inf),
                ),
            )


class TestResultShape:
    @staticmethod
    def test_results_are_immutable():
        result = group_delay({"S21": delay_line(linear_grid())})
        with pytest.raises(FrozenInstanceError):
            result.aperture_points = 3
        with pytest.raises(FrozenInstanceError):
            result.points[0].value = 0.0

    @staticmethod
    def test_values_are_not_interpolated():
        freqs = linear_grid()
        result = group_delay({"S21": delay_line(freqs)})
        assert result.value_at(freqs[3] + 1) is None
        assert result.value_at(freqs[3]) == pytest.approx(TAU, abs=1e-15)

    @staticmethod
    def test_median_is_the_middle_of_the_defined_values():
        result = phase_curvature({"S21": delay_line(linear_grid())})
        assert result.median == statistics.median(result.values)
        assert len(result.values) == 101
        assert result.undefined_count == 0

    @staticmethod
    def test_describe_names_the_quantity_unit_and_count():
        result = group_delay({"S21": delay_line(linear_grid())})
        text = result.describe()
        assert text.startswith("group delay of S21: 3.7e-09 s (median, ")
        assert "101 of 101 samples" in text
        assert "2 wrap correction(s)" in text
        assert result.method == METHOD
        assert result.smoothing == SMOOTHING
        assert "smooth_channel" in result.smoothing
