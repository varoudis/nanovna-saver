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
"""DSP-02: complex frequency smoothing.

Every expected value here is computed from the definition of a centred
moving average in plain Python; none of it is read back out of the
implementation.  The tests pin the *meaning* of smoothing, not just that it
runs: a smoothed trace is a different measurement, never a more accurate
one, and the arithmetic is done on the complex reflection coefficient so
that magnitude, phase and Smith views cannot disagree.
"""

import cmath
import math

import pytest

from NanoVNASaver.Processing import (
    STAGE_SMOOTHING,
    ProcessingConfig,
    ProcessingError,
    ProcessingStage,
    apply_processing,
    channel_is_uniform,
    effective_smoothing_aperture,
    smooth_channel,
    smoothing_aperture_for,
    smoothing_stage,
    smoothing_stage_for,
    validate_smoothing_aperture,
)
from NanoVNASaver.RFTools import Datapoint

STEP = 1_000_000


def _channel(values, start=10_000_000, step=STEP):
    return tuple(
        Datapoint(start + index * step, value.real, value.imag)
        for index, value in enumerate(values)
    )


def _reference_mean(points, index, aperture):
    """Centred window mean, window clipped to the samples that exist."""
    half = aperture // 2
    low = max(0, index - half)
    high = min(len(points), index + half + 1)
    window = [point.z for point in points[low:high]]
    return sum(window) / len(window)


def _resonance(count=101, depth=0.1, half_width=4.0):
    """Analytic reflection dip: |Gamma| falls to ``1 - depth`` at the centre."""
    return [
        1.0 - depth / (1.0 + 1j * (index - (count - 1) / 2) / half_width)
        for index in range(count)
    ]


class TestApertureValidation:
    @staticmethod
    def test_a_positive_odd_number_of_points_is_accepted():
        assert validate_smoothing_aperture(1) == 1
        assert validate_smoothing_aperture(9) == 9
        assert validate_smoothing_aperture(9.0) == 9

    @staticmethod
    @pytest.mark.parametrize("aperture", [0, -1, 2, 4, -3])
    def test_a_non_positive_or_even_aperture_is_refused(aperture):
        with pytest.raises(ProcessingError):
            validate_smoothing_aperture(aperture)

    @staticmethod
    @pytest.mark.parametrize("aperture", [None, "wide", 2.5, True])
    def test_a_meaningless_aperture_is_refused(aperture):
        with pytest.raises(ProcessingError):
            validate_smoothing_aperture(aperture)

    @staticmethod
    def test_non_finite_apertures_are_refused():
        with pytest.raises(ProcessingError):
            validate_smoothing_aperture(math.inf)
        with pytest.raises(ProcessingError):
            validate_smoothing_aperture(math.nan)

    @staticmethod
    def test_an_aperture_wider_than_the_data_is_clamped_to_the_data():
        assert effective_smoothing_aperture(10, 99) == 9
        assert effective_smoothing_aperture(11, 99) == 11
        assert effective_smoothing_aperture(10, 3) == 3

    @staticmethod
    def test_too_few_samples_can_only_support_no_smoothing():
        assert effective_smoothing_aperture(0, 5) == 1
        assert effective_smoothing_aperture(1, 5) == 1
        assert effective_smoothing_aperture(2, 5) == 1
        assert effective_smoothing_aperture(3, 5) == 3

    @staticmethod
    def test_a_percentage_aperture_is_odd_and_never_exceeds_the_data():
        assert smoothing_aperture_for(101, 10) % 2 == 1
        assert smoothing_aperture_for(101, 100) == 101
        assert smoothing_aperture_for(100, 100) == 99
        assert smoothing_aperture_for(11, 0.5) == 1

    @staticmethod
    @pytest.mark.parametrize("percent", [0, -5, 101, math.nan])
    def test_an_impossible_percentage_is_refused(percent):
        with pytest.raises(ProcessingError):
            smoothing_aperture_for(101, percent)

    @staticmethod
    def test_a_percentage_that_collapses_to_one_sample_means_no_stage():
        assert smoothing_stage_for(11, 0.5) is None
        stage = smoothing_stage_for(101, 10)
        assert stage is not None
        assert stage.get("aperture_points") % 2 == 1


class TestCentredComplexMean:
    @staticmethod
    def test_a_constant_trace_is_returned_unchanged():
        constant = _channel([complex(0.5, -0.25)] * 21)
        smoothed = smooth_channel(constant, 7)
        assert len(smoothed) == len(constant)
        for before, after in zip(smoothed, constant, strict=True):
            assert after.z == pytest.approx(before.z, abs=1e-12)

    @staticmethod
    def test_every_interior_point_is_the_complex_window_mean():
        values = [
            cmath.rect(0.2 + 0.03 * index, 0.4 * index) for index in range(21)
        ]
        points = _channel(values)
        smoothed = smooth_channel(points, 5)
        for index in (2, 5, 10, 15, 18):
            assert smoothed[index].z == pytest.approx(
                _reference_mean(points, index, 5), abs=1e-12
            )

    @staticmethod
    def test_the_frequency_grid_is_preserved_sample_by_sample():
        points = _channel([cmath.rect(0.3, 0.1 * index) for index in range(15)])
        smoothed = smooth_channel(points, 3)
        assert [point.freq for point in smoothed] == [
            point.freq for point in points
        ]

    @staticmethod
    def test_windows_at_the_ends_shrink_instead_of_wrapping():
        points = _channel([complex(0.1 * index, 0.0) for index in range(9)])
        smoothed = smooth_channel(points, 5)
        # First sample: mean of the first three samples only.
        assert smoothed[0].re == pytest.approx((0.0 + 0.1 + 0.2) / 3)
        assert smoothed[1].re == pytest.approx((0.0 + 0.1 + 0.2 + 0.3) / 4)
        assert smoothed[2].re == pytest.approx(
            (0.0 + 0.1 + 0.2 + 0.3 + 0.4) / 5
        )
        assert smoothed[8].re == pytest.approx((0.6 + 0.7 + 0.8) / 3)

    @staticmethod
    def test_the_last_window_is_the_mirror_of_the_first():
        points = _channel([complex(0.1 * index, 0.0) for index in range(9)])
        smoothed = smooth_channel(points, 5)
        for offset in range(3):
            assert smoothed[offset].re + smoothed[8 - offset].re == (
                pytest.approx(0.8)
            )

    @staticmethod
    def test_a_ten_point_channel_clamped_to_nine_matches_nine_directly():
        points = _channel(_resonance(count=10))
        assert smooth_channel(points, 99) == smooth_channel(points, 9)

    @staticmethod
    def test_the_uniform_grid_check_recognises_a_linear_sweep():
        assert channel_is_uniform(_channel([complex(0.1, 0.0)] * 5)) is True
        grid = _channel([complex(0.1, 0.0)] * 5)
        log_like = (
            grid[0],
            grid[1],
            grid[2],
            Datapoint(grid[3].freq + 5 * STEP, 0.1, 0.0),
            Datapoint(grid[4].freq + 20 * STEP, 0.1, 0.0),
        )
        assert channel_is_uniform(log_like) is False


class TestDegenerateInputs:
    @staticmethod
    def test_an_empty_channel_returns_an_empty_tuple():
        assert smooth_channel((), 5) == ()

    @staticmethod
    def test_one_sample_is_returned_unchanged():
        point = Datapoint(STEP, 0.3, -0.2)
        assert smooth_channel((point,), 21) == (point,)

    @staticmethod
    def test_two_samples_are_returned_unchanged():
        points = (Datapoint(STEP, 0.3, 0.0), Datapoint(2 * STEP, 0.1, 0.0))
        assert smooth_channel(points, 21) == points

    @staticmethod
    def test_three_samples_average_across_the_whole_channel():
        points = _channel(
            [complex(0.0, 0.0), complex(0.3, 0.0), complex(0.6, 0.0)]
        )
        smoothed = smooth_channel(points, 3)
        assert smoothed[1].re == pytest.approx(0.3)

    @staticmethod
    def test_an_invalid_aperture_raises_rather_than_being_repaired():
        points = _channel([complex(0.1, 0.0)] * 5)
        for aperture in (0, 2, None):
            with pytest.raises(ProcessingError):
                smooth_channel(points, aperture)

    @staticmethod
    def test_a_non_finite_sample_raises_instead_of_spreading_nan():
        points = _channel(
            [complex(0.1, 0.0), complex(math.nan, 0.0), complex(0.3, 0.0)]
        )
        with pytest.raises(ProcessingError):
            smooth_channel(points, 3)

    @staticmethod
    def test_the_input_sequence_is_not_modified():
        values = [complex(0.1 * index, 0.0) for index in range(7)]
        points = _channel(values)
        before = tuple(points)
        smooth_channel(points, 5)
        assert points == before


class TestSmoothingChangesTheMeasurement:
    @staticmethod
    def test_a_narrow_resonance_is_shallower_and_wider_after_smoothing():
        points = _channel(_resonance())
        smoothed = smooth_channel(points, 11)
        raw_min = min(abs(point.z) for point in points)
        smooth_min = min(abs(point.z) for point in smoothed)
        assert raw_min == pytest.approx(0.9, abs=1e-9)
        # A wider, shallower feature: the depth is measurably reduced.
        assert smooth_min > raw_min + 1e-3

        def width(series, level):
            return sum(1 for point in series if abs(point.z) < level)

        level = 0.95
        assert width(smoothed, level) > width(points, level)

    @staticmethod
    def test_the_smoothed_point_is_not_the_mean_of_the_magnitudes():
        points = _channel(_resonance(depth=0.3))
        smoothed = smooth_channel(points, 9)
        centre = (len(points) - 1) // 2
        window = [abs(point.z) for point in points[centre - 4 : centre + 5]]
        # |mean(z)| <= mean(|z|) by the triangle inequality, and here the gap
        # is measurable: averaging magnitudes would report a *shallower* null
        # (a larger |Gamma|) than the complex mean actually used.
        assert sum(window) / len(window) > abs(smoothed[centre].z) + 0.005

    @staticmethod
    def test_phase_follows_the_smoothed_complex_value_across_a_wrap():
        # Phases run past +-180 degrees, so averaging them directly would be
        # nonsense; averaging the complex value is the only defined answer.
        points = _channel(
            [
                cmath.rect(0.7, math.radians(150 + 12 * index))
                for index in range(11)
            ]
        )
        smoothed = smooth_channel(points, 5)
        for index, point in enumerate(smoothed):
            expected = _reference_mean(points, index, 5)
            assert point.z == pytest.approx(expected, abs=1e-12)
            assert point.phase == pytest.approx(
                cmath.phase(expected), abs=1e-12
            )

    @staticmethod
    def test_magnitude_phase_and_smith_views_all_come_from_one_value():
        points = _channel(_resonance())
        smoothed = smooth_channel(points, 7)
        for point in smoothed:
            expected = point.z
            assert point.gain == pytest.approx(
                20 * math.log10(abs(expected)), abs=1e-12
            )
            assert point.phase == pytest.approx(
                cmath.phase(expected), abs=1e-12
            )
            assert point.impedance(50.0) == pytest.approx(
                ((-expected - 1) / (expected - 1)) * 50.0, abs=1e-9
            )

    @staticmethod
    def test_a_one_port_measurement_smooths_without_a_second_channel():
        points = _channel(_resonance(count=21))
        result = apply_processing(
            {"S11": points}, ProcessingConfig.build(smoothing_stage(5))
        )
        assert set(result.channels) == {"S11"}
        assert result.channels["S11"] == smooth_channel(points, 5)


class TestSmoothingThroughThePipeline:
    @staticmethod
    def test_the_raw_channel_is_never_overwritten():
        points = _channel(_resonance(count=21))
        before = tuple(points)
        apply_processing(
            {"S11": points}, ProcessingConfig.build(smoothing_stage(5))
        )
        assert points == before

    @staticmethod
    def test_an_identity_configuration_returns_the_input_unchanged():
        points = _channel(_resonance(count=21))
        result = apply_processing({"S11": points}, None)
        assert result.channels["S11"] == points
        assert result.config.is_identity is True

    @staticmethod
    @pytest.mark.parametrize("aperture", [0, None, 2, -4])
    def test_a_stage_that_cannot_be_honoured_raises(aperture):
        stage = ProcessingStage.build(
            STAGE_SMOOTHING, {"aperture_points": aperture}
        )
        with pytest.raises(ProcessingError):
            apply_processing(
                {"S11": _channel([complex(0.1, 0.0)] * 9)},
                ProcessingConfig.build(stage),
            )

    @staticmethod
    def test_a_stage_without_an_aperture_is_refused():
        stage = ProcessingStage.build(STAGE_SMOOTHING, {})
        with pytest.raises(ProcessingError):
            apply_processing(
                {"S11": _channel([complex(0.1, 0.0)] * 9)},
                ProcessingConfig.build(stage),
            )

    @staticmethod
    def test_the_notes_report_the_applied_point_count():
        result = apply_processing(
            {"S11": _channel([complex(0.1, 0.0)] * 21)},
            ProcessingConfig.build(smoothing_stage(5)),
        )
        assert any("5 point(s)" in note for note in result.notes)

    @staticmethod
    def test_the_notes_report_a_clamped_aperture_as_the_one_used():
        result = apply_processing(
            {"S11": _channel([complex(0.1, 0.0)] * 11)},
            ProcessingConfig.build(smoothing_stage(99)),
        )
        assert any("11 point(s)" in note for note in result.notes)

    @staticmethod
    def test_a_non_uniform_grid_is_reported_rather_than_hidden():
        uniform = _channel([complex(0.1, 0.0)] * 9)
        uneven = (
            *uniform[:4],
            Datapoint(uniform[4].freq + 7 * STEP, 0.1, 0.0),
            *uniform[5:],
        )
        result = apply_processing(
            {"S11": uneven}, ProcessingConfig.build(smoothing_stage(3))
        )
        assert any("non-uniform" in note for note in result.notes)

    @staticmethod
    def test_a_uniform_grid_produces_no_warning_note():
        result = apply_processing(
            {"S11": _channel([complex(0.1, 0.0)] * 9)},
            ProcessingConfig.build(smoothing_stage(3)),
        )
        assert not any("non-uniform" in note for note in result.notes)

    @staticmethod
    def test_each_channel_reports_the_aperture_it_could_support():
        long_channel = _channel([complex(0.1, 0.0)] * 21)
        short_channel = _channel([complex(0.2, 0.0)] * 7)
        result = apply_processing(
            {"S11": long_channel, "S21": short_channel},
            ProcessingConfig.build(smoothing_stage(11)),
        )
        note = next(
            note for note in result.notes if "Frequency smoothing" in note
        )
        assert "11 point(s)" in note
        assert "7 point(s)" in note
        assert result.channels["S21"] == smooth_channel(short_channel, 7)

    @staticmethod
    def test_smoothing_stage_validates_at_construction():
        with pytest.raises(ProcessingError):
            smoothing_stage(4)
        with pytest.raises(ProcessingError):
            smoothing_stage(0)

    @staticmethod
    def test_a_smoothing_stage_survives_a_configuration_round_trip():
        config = ProcessingConfig.build(smoothing_stage(7))
        restored = ProcessingConfig.from_dict(config.to_dict())
        assert restored == config
        assert restored.identity() == config.identity()
