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
"""CAL-01/CAL-03: coverage classification without touching correction maths."""

import math

from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.CalibrationCoverage import (
    CoverageState,
    coverage_from_calibration,
    coverage_identity,
    state_at_frequency,
)
from NanoVNASaver.Measurement import CorrectionSource
from NanoVNASaver.RFTools import Datapoint


def _one_port_calibration() -> Calibration:
    calibration = Calibration()
    calibration.load("tests/data/sol_27_30.cal")
    calibration.calc_corrections()
    return calibration


def _two_port_calibration() -> Calibration:
    calibration = Calibration()
    calibration.load("tests/data/test_2port_long.cal")
    calibration.calc_corrections()
    return calibration


class TestCoverageStates:
    @staticmethod
    def test_no_calibration_is_uncalibrated():
        result = coverage_from_calibration(Calibration(), [1_000_000])
        assert result.state is CoverageState.UNCALIBRATED
        assert not result.has_calibration
        assert not result.covered
        assert "No calibration" in result.describe()

    @staticmethod
    def test_full_span_is_fully_covered():
        result = coverage_from_calibration(
            _one_port_calibration(), [27_500_000, 28_000_000, 29_500_000]
        )
        assert result.state is CoverageState.FULL
        assert result.covered
        assert result.edge_held_points == 0
        assert result.interpolated_points == 3

    @staticmethod
    def test_exact_boundary_frequencies_are_covered():
        calibration = _one_port_calibration()
        start = calibration.dataset.freq_min()
        end = calibration.dataset.freq_max()
        result = coverage_from_calibration(calibration, [start, end])
        assert result.state is CoverageState.FULL
        assert result.state_at(start) is CoverageState.FULL
        assert result.state_at(end) is CoverageState.FULL
        assert result.calibrated_start == start
        assert result.calibrated_end == end

    @staticmethod
    def test_just_outside_the_boundary_is_not_covered():
        calibration = _one_port_calibration()
        start = calibration.dataset.freq_min()
        end = calibration.dataset.freq_max()
        result = coverage_from_calibration(calibration, [start - 1, end + 1])
        assert result.state is CoverageState.OUTSIDE
        assert result.state_at(start - 1) is CoverageState.OUTSIDE
        assert result.state_at(end + 1) is CoverageState.OUTSIDE

    @staticmethod
    def test_partial_overlap_is_partial_with_spans():
        calibration = _one_port_calibration()
        start = calibration.dataset.freq_min()
        end = calibration.dataset.freq_max()
        result = coverage_from_calibration(
            calibration, [start - 1_000_000, start, end, end + 1_000_000]
        )
        assert result.state is CoverageState.PARTIAL
        assert result.interpolated_points == 2
        assert result.edge_held_points == 2
        assert [span.state for span in result.spans] == [
            CoverageState.OUTSIDE,
            CoverageState.FULL,
            CoverageState.OUTSIDE,
        ]
        # Contiguous runs are merged into spans with real boundaries.
        assert result.spans[0].start == start - 1_000_000
        assert result.spans[-1].end == end + 1_000_000

    @staticmethod
    def test_no_overlap_at_all_is_outside():
        calibration = _one_port_calibration()
        end = calibration.dataset.freq_max()
        result = coverage_from_calibration(calibration, [end + 1, end + 2])
        assert result.state is CoverageState.OUTSIDE
        assert result.interpolated_points == 0
        assert result.edge_held_points == 2

    @staticmethod
    def test_endpoint_held_corrections_are_reported_explicitly():
        calibration = _one_port_calibration()
        start = calibration.dataset.freq_min()
        result = coverage_from_calibration(calibration, [start - 5_000_000])
        assert result.state is CoverageState.OUTSIDE
        assert result.edge_held
        assert "held at edge" in result.describe()

    @staticmethod
    def test_empty_grid_is_invalid_not_crash():
        result = coverage_from_calibration(_one_port_calibration(), [])
        assert result.state is CoverageState.INVALID
        assert result.reason == "empty frequency grid"

    @staticmethod
    def test_missing_correction_channel_is_invalid_for_that_channel():
        """S21 needs two-port corrections; one-port data cannot supply them."""
        result = coverage_from_calibration(
            _one_port_calibration(), [28_000_000], parameters=("S21",)
        )
        assert result.state is CoverageState.INVALID
        assert result.uncorrected_parameters == ("S21",)
        assert "S21" in result.describe()

    @staticmethod
    def test_two_port_calibration_covers_both_channels():
        calibration = _two_port_calibration()
        frequencies = calibration.dataset.frequencies()
        result = coverage_from_calibration(
            calibration,
            [frequencies[0], frequencies[len(frequencies) // 2]],
            parameters=("S11", "S21"),
        )
        assert result.state is CoverageState.FULL
        assert set(result.corrected_parameters) == {"S11", "S21"}
        assert result.uncorrected_parameters == ()

    @staticmethod
    def test_zero_denominator_safe_with_single_sample_calibration():
        """One-sample calibration must not produce bogus interpolation."""
        calibration = Calibration()
        for name in ("short", "open", "load"):
            calibration.insert(name, [Datapoint(1_000_000, 0.1, 0.1)])
        result = coverage_from_calibration(calibration, [1_000_000, 2_000_000])
        assert result.state is CoverageState.PARTIAL
        assert result.calibrated_start == result.calibrated_end == 1_000_000


class TestMarkerLevelCoverage:
    @staticmethod
    def test_state_at_frequency_uses_the_live_marker_position():
        calibration = _one_port_calibration()
        frequencies = [27_000_000, 28_000_000, 30_000_000]
        result = coverage_from_calibration(calibration, frequencies)
        assert result.state_at(27_000_000) is CoverageState.FULL
        assert result.state_at(30_000_000) is CoverageState.FULL
        assert state_at_frequency(result, 26_999_999) is CoverageState.OUTSIDE
        assert state_at_frequency(result, 30_000_001) is CoverageState.OUTSIDE

    @staticmethod
    def test_uncalibrated_state_at_any_frequency():
        result = coverage_from_calibration(Calibration(), [1_000_000])
        assert result.state_at(1_000_000) is CoverageState.UNCALIBRATED


class TestCoverageIdentity:
    @staticmethod
    def test_identity_captures_span_channels_and_source():
        calibration = _one_port_calibration()
        frequencies = calibration.dataset.frequencies()
        identity = coverage_identity(
            calibration,
            frequencies,
            parameters=("S11",),
            correction_source=CorrectionSource.APPLICATION,
        )
        assert identity.one_port
        assert not identity.two_port
        assert identity.coverage_spans == ((frequencies[0], frequencies[-1]),)
        assert identity.correction_source is CorrectionSource.APPLICATION
        assert identity.identity

    @staticmethod
    def test_device_correction_source_is_recorded_distinctly():
        calibration = _one_port_calibration()
        identity = coverage_identity(
            calibration, correction_source=CorrectionSource.DEVICE
        )
        assert identity.correction_source is CorrectionSource.DEVICE

    @staticmethod
    def test_unknown_device_metadata_is_reported_as_unknown():
        calibration = _one_port_calibration()
        identity = coverage_identity(calibration)
        assert identity.correction_source is CorrectionSource.UNKNOWN

    @staticmethod
    def test_empty_calibration_has_no_spans():
        identity = coverage_identity(Calibration())
        assert identity.coverage_spans == ()
        assert identity.identity == ""


class TestNumericalCompatibility:
    """CAL-03: coverage must not alter corrected values."""

    @staticmethod
    def test_loading_existing_calibration_files_still_works():
        for name in (
            "test_1port_fixed.cal",
            "sol_27_30.cal",
            "full_v2_200_300.cal",
        ):
            calibration = Calibration()
            calibration.load(f"tests/data/{name}")
            if name.endswith(".cal"):
                calibration.calc_corrections()
                assert calibration.isValid1Port()
            assert calibration.size() > 0

    @staticmethod
    def test_corrected_values_are_unchanged_by_coverage_queries():
        calibration = _one_port_calibration()
        raw = [
            Datapoint(freq, 0.3, -0.2)
            for freq in (27_000_000, 28_000_000, 30_000_000)
        ]
        before = [calibration.correct11(dp).z for dp in raw]
        # Query coverage in a way that covers inside, at and outside the span.
        coverage_from_calibration(
            calibration,
            [26_000_000, 27_000_000, 30_000_000, 31_000_000],
            parameters=("S11", "S21"),
        )
        after = [calibration.correct11(dp).z for dp in raw]
        assert before == after

    @staticmethod
    def test_out_of_span_correction_is_still_applied_as_before():
        """Endpoint holding is existing numerical behaviour, not an error."""
        calibration = _one_port_calibration()
        outside = Datapoint(
            calibration.dataset.freq_max() + 5_000_000, 0.3, -0.2
        )
        corrected = calibration.correct11(outside)
        assert math.isfinite(corrected.re)
        assert math.isfinite(corrected.im)
        coverage = coverage_from_calibration(
            calibration, [outside.freq], parameters=("S11",)
        )
        # Coverage reports the limitation; the value is unchanged.
        assert coverage.state is CoverageState.OUTSIDE

    @staticmethod
    def test_two_port_corrected_values_unchanged():
        calibration = _two_port_calibration()
        frequencies = calibration.dataset.frequencies()
        freq = frequencies[len(frequencies) // 2]
        raw11 = Datapoint(freq, 0.2, 0.1)
        raw21 = Datapoint(freq, 0.7, -0.1)
        before = calibration.correct21(raw21, raw11).z
        coverage_from_calibration(calibration, frequencies, ("S11", "S21"))
        assert calibration.correct21(raw21, raw11).z == before
