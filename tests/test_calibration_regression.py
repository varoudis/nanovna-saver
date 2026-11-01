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
"""CAL-03.2: coverage reporting must not change corrected values.

Coverage is a *display* feature.  These tests pin the numerical contract:
asking where calibration applies returns metadata only, and every corrected
sample is bit-for-bit identical to what the pre-coverage code produced.
"""

import pytest

from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.CalibrationCoverage import (
    CoverageState,
    coverage_from_calibration,
    coverage_identity,
    state_at_frequency,
)
from NanoVNASaver.RFTools import Datapoint

CALIBRATION_FILES = (
    "tests/data/sol_27_30.cal",
    "tests/data/test_2port_long.cal",
    "tests/data/test_1port_fixed.cal",
)


def _calibration(filename: str) -> Calibration:
    calibration = Calibration()
    calibration.load(filename)
    calibration.calc_corrections()
    return calibration


def _span(calibration: Calibration) -> tuple[int, int]:
    frequencies = calibration.dataset.frequencies()
    return frequencies[0], frequencies[-1]


def _samples(calibration: Calibration, count: int = 17) -> list[Datapoint]:
    """Samples spanning the whole calibrated range, including its edges."""
    start, stop = _span(calibration)
    step = (stop - start) / (count - 1)
    return [
        Datapoint(round(start + index * step), 0.4, -0.2)
        for index in range(count)
    ]


def _corrected_digest(
    calibration: Calibration,
) -> list[tuple[int, float, float]]:
    return [
        (point.freq, point.re, point.im)
        for point in map(calibration.correct11, _samples(calibration))
    ]


class TestCorrectionIsUnaffectedByCoverage:
    @staticmethod
    @pytest.mark.parametrize("filename", CALIBRATION_FILES)
    def test_corrected_values_are_bit_identical(filename):
        """Correcting before and after a coverage query gives the same data."""
        calibration = _calibration(filename)
        before = _corrected_digest(calibration)

        # Query coverage aggressively: the whole range, both endpoints, and
        # points deliberately outside it.
        samples = _samples(calibration, count=64)
        frequencies = [point.freq for point in samples]
        start, stop = _span(calibration)
        frequencies.extend([start - 1_000_000, stop + 1_000_000])
        result = coverage_from_calibration(calibration, frequencies)
        for frequency in frequencies:
            state_at_frequency(result, frequency)

        after = _corrected_digest(calibration)
        assert after == before
        assert all(
            first == second for first, second in zip(before, after, strict=True)
        )

    @staticmethod
    @pytest.mark.parametrize("filename", CALIBRATION_FILES)
    def test_coverage_does_not_mutate_the_calibration(filename):
        calibration = _calibration(filename)
        start, stop = _span(calibration)
        identity_before = coverage_identity(calibration)

        coverage_from_calibration(calibration, [start, stop])

        assert _span(calibration) == (start, stop)
        assert coverage_identity(calibration) == identity_before

    @staticmethod
    def test_repeated_coverage_queries_are_idempotent():
        """A second query cannot shift a boundary by a rounding step."""
        calibration = _calibration("tests/data/sol_27_30.cal")
        samples = [point.freq for point in _samples(calibration)]
        first = coverage_from_calibration(calibration, samples)
        second = coverage_from_calibration(calibration, samples)
        assert first == second
        assert first.calibrated_start == second.calibrated_start
        assert first.calibrated_end == second.calibrated_end


class TestEndpointsAndOutsideSpan:
    @staticmethod
    def test_exact_endpoints_are_covered_not_edge_held():
        """A sample exactly on a calibration endpoint is inside the domain."""
        calibration = _calibration("tests/data/sol_27_30.cal")
        start, stop = _span(calibration)
        result = coverage_from_calibration(calibration, [start, stop])
        assert result.state_at(start) is CoverageState.FULL
        assert result.state_at(stop) is CoverageState.FULL
        assert result.edge_held_points == 0

    @staticmethod
    def test_one_hz_outside_is_outside_not_interpolated():
        calibration = _calibration("tests/data/sol_27_30.cal")
        start, stop = _span(calibration)
        result = coverage_from_calibration(calibration, [start - 1, stop + 1])
        assert result.state_at(start - 1) is CoverageState.OUTSIDE
        assert result.state_at(stop + 1) is CoverageState.OUTSIDE
        # Neither point is interpolated: nothing between real calibration
        # frequencies is invented for them.
        assert result.interpolated_points == 0
        # They do still receive the existing endpoint-held correction, and
        # are reported as such rather than being silently claimed as covered.
        assert result.edge_held_points == 2
        assert not result.covered

    @staticmethod
    def test_a_span_wider_than_calibration_reports_partial():
        calibration = _calibration("tests/data/sol_27_30.cal")
        start, stop = _span(calibration)
        middle = (start + stop) // 2
        result = coverage_from_calibration(
            calibration, [start - 1_000_000, middle, stop + 1_000_000]
        )
        assert result.state is CoverageState.PARTIAL
        assert result.state_at(middle) is CoverageState.FULL


class TestCoverageCannotEstablish:
    @staticmethod
    def test_coverage_reports_no_uncertainty():
        """CAL-03.3: coverage is a span statement, never an error estimate."""
        calibration = _calibration("tests/data/sol_27_30.cal")
        result = coverage_from_calibration(calibration, [*_span(calibration)])
        fields = set(vars(result))
        for forbidden in ("uncertainty", "tolerance", "error", "confidence"):
            assert forbidden not in fields

    @staticmethod
    def test_coverage_describe_states_only_what_it_knows():
        calibration = _calibration("tests/data/sol_27_30.cal")
        result = coverage_from_calibration(calibration, [*_span(calibration)])
        description = result.describe()
        # It may name what is covered and when the file was made, never a
        # judgement about whether the calibration is still valid.
        for forbidden in (
            "uncertainty",
            "accurate to",
            "connector wear",
            "expired",
        ):
            assert forbidden not in description.lower()
