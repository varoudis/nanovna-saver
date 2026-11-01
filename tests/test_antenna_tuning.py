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
"""ANT-01: minimum search, sweep/zoom planning and honest S1P export.

Pure numerical tests: no widgets, no device, no Qt application.
"""

import math
import os
from datetime import datetime, timezone

import pytest

from NanoVNASaver.AntennaTuning import (
    MIN_POINTS,
    AntennaExportError,
    DeviceLimits,
    SweepPlanError,
    auto_zoom_span,
    choose_points,
    find_vswr_minimum,
    normalize_s1p_path,
    plan_sweep,
    plan_zoom,
    return_loss,
    sweep_grid,
    sweep_vswr,
    tuning_filename,
    unique_path,
    write_s1p,
)
from NanoVNASaver.Measurement import (
    CompletionState,
    snapshot_from_datapoints,
)
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Touchstone import Touchstone

LIMITS = DeviceLimits(
    min_frequency=50_000,
    max_frequency=300_000_000,
    points=(11, 51, 101, 201, 401),
    default_points=101,
)


def _reflection(magnitude: float) -> complex:
    """A pure-resistance reflection coefficient of the given magnitude."""
    return complex(magnitude, 0.0)


def _samples(values, start=14_000_000, step=1_000):
    return [
        Datapoint(start + index * step, value.real, value.imag)
        for index, value in enumerate(values)
    ]


class TestMinimumSearch:
    @staticmethod
    def test_lowest_sample_is_selected_by_value_not_by_index():
        # The dip sits at index 3; a hardcoded index would report another
        # frequency entirely.
        samples = _samples(
            [
                _reflection(0.60),
                _reflection(0.45),
                _reflection(0.30),
                _reflection(0.02),
                _reflection(0.20),
                _reflection(0.50),
            ]
        )
        result = find_vswr_minimum(samples)
        assert result.found
        minimum = result.minimum
        assert minimum.index == 3
        assert minimum.freq == samples[3].freq
        assert minimum.vswr == pytest.approx(1.040816, rel=1e-6)
        assert minimum.return_loss == pytest.approx(-20 * math.log10(0.02))
        assert minimum.samples == 6
        assert not minimum.at_edge

    @staticmethod
    def test_shape_of_the_curve_does_not_matter():
        # A minimum that is *not* at a local extremum of index order must
        # still be found: plain min over measured VSWR.
        values = [0.9, 0.1, 0.8, 0.05, 0.7]
        result = find_vswr_minimum(_samples(values))
        assert result.minimum.index == 3

    @staticmethod
    def test_empty_data_reports_why():
        result = find_vswr_minimum([])
        assert not result.found
        assert "no S11 samples" in result.describe()
        assert find_vswr_minimum(None).samples == 0

    @staticmethod
    def test_single_point_is_found_and_flagged_as_an_edge():
        result = find_vswr_minimum(_samples([_reflection(0.10)]))
        assert result.found
        assert result.minimum.index == 0
        assert result.minimum.at_edge
        assert "sweep edge" in result.minimum.describe()

    @staticmethod
    def test_first_and_last_samples_are_edge_minima():
        first = find_vswr_minimum(_samples([0.01, 0.5, 0.6]))
        last = find_vswr_minimum(_samples([0.6, 0.5, 0.01]))
        assert first.minimum.at_edge and first.minimum.index == 0
        assert last.minimum.at_edge and last.minimum.index == 2
        assert "sweep edge" in last.minimum.describe()

    @staticmethod
    def test_non_finite_samples_are_skipped_not_adopted():
        samples = [
            Datapoint(14_000_000, math.nan, math.inf),
            Datapoint(14_001_000, 0.40, 0.0),
            Datapoint(14_002_000, math.inf, 0.0),
            Datapoint(14_003_000, 0.05, 0.0),
        ]
        result = find_vswr_minimum(samples)
        assert result.samples == 4
        assert result.skipped == 2
        assert result.minimum.index == 3
        assert "2 non-finite" in result.describe()

    @staticmethod
    def test_all_non_finite_reports_no_minimum():
        samples = [
            Datapoint(14_000_000, math.nan, 0.0),
            Datapoint(14_001_000, 0.0, math.inf),
        ]
        result = find_vswr_minimum(samples)
        assert not result.found
        assert "non-finite" in result.describe()

    @staticmethod
    def test_total_reflection_yields_no_finite_minimum():
        samples = _samples([complex(1.0, 0.0), complex(0.0, 1.2)])
        result = find_vswr_minimum(samples)
        assert not result.found
        assert "infinite" in result.describe()

    @staticmethod
    def test_one_perfect_match_beside_total_reflection_wins():
        samples = _samples([complex(0.9, 0.9), complex(0.0, 0.0)])
        result = find_vswr_minimum(samples)
        assert result.minimum.index == 1
        assert result.minimum.vswr == pytest.approx(1.0)

    @staticmethod
    def test_vswr_and_return_loss_helpers():
        assert sweep_vswr(Datapoint(1, 0.5, 0.0)) == pytest.approx(3.0)
        assert sweep_vswr(Datapoint(1, 1.5, 0.0)) == math.inf
        assert sweep_vswr(Datapoint(1, math.nan, 0.0)) is None
        # Standard return loss is -20*log10(|S11|): positive for a passive
        # match, zero at total reflection, infinite for a perfect match.
        assert return_loss(Datapoint(1, 0.1, 0.0)) == pytest.approx(20.0)
        assert return_loss(Datapoint(1, 0.01, 0.0)) == pytest.approx(40.0)
        # 0 dB, and printed as "0.000" rather than "-0.000".
        total = return_loss(Datapoint(1, 1.0, 0.0))
        assert total == pytest.approx(0.0)
        assert math.copysign(1.0, total) == 1.0
        assert return_loss(Datapoint(1, 0.0, 0.0)) == math.inf
        assert return_loss(Datapoint(1, 2.0, 0.0)) == pytest.approx(-6.0206)
        assert return_loss(Datapoint(1, math.inf, 0.0)) is None

    @staticmethod
    def test_return_loss_agrees_with_the_vswr_of_the_same_sample():
        # A passive match must be a *positive* dB figure consistent with
        # VSWR = (1+|G|)/(1-|G|) for the same reflection coefficient.
        for magnitude in (0.001, 0.05, 0.2, 0.6, 0.9):
            sample = Datapoint(14_000_000, magnitude, 0.0)
            vswr = sweep_vswr(sample)
            loss = return_loss(sample)
            magnitude_back = (vswr - 1) / (vswr + 1)
            assert loss == pytest.approx(-20 * math.log10(magnitude_back))
            assert loss > 0


class TestPointSelection:
    @staticmethod
    def test_unsupported_count_falls_back_to_the_nearest_supported():
        assert choose_points(LIMITS, 100) == 101
        # 151 is exactly between 101 and 201; the larger count wins.
        assert choose_points(LIMITS, 151) == 201
        assert choose_points(LIMITS, 7) == 11
        assert choose_points(LIMITS, None) == 101

    @staticmethod
    def test_unknown_device_points_keeps_the_request():
        limits = DeviceLimits(points=(), default_points=101)
        assert choose_points(limits, 301) == 301
        assert choose_points(limits, None) == 101


class TestOverviewPlanning:
    @staticmethod
    def test_valid_span_is_brought_into_device_range():
        plan = plan_sweep(
            10_000, 400_000_000, limits=LIMITS, points=101, segments=1
        )
        assert (plan.start, plan.end) == (50_000, 300_000_000)
        assert plan.notes  # the clamping is reported, never silent
        assert any("device minimum" in note for note in plan.notes)
        assert any("device maximum" in note for note in plan.notes)

    @staticmethod
    def test_reversed_and_empty_spans_are_rejected():
        with pytest.raises(SweepPlanError):
            plan_sweep(15_000_000, 15_000_000, limits=LIMITS)
        with pytest.raises(SweepPlanError):
            plan_sweep(16_000_000, 15_000_000, limits=LIMITS)

    @staticmethod
    def test_span_entirely_outside_the_device_is_rejected():
        limits = DeviceLimits(min_frequency=50_000, max_frequency=1_000_000)
        with pytest.raises(SweepPlanError, match="outside"):
            plan_sweep(2_000_000, 3_000_000, limits=limits)

    @staticmethod
    def test_segments_are_reduced_when_the_span_cannot_fit_them():
        plan = plan_sweep(
            14_000_000, 14_000_100, limits=LIMITS, points=101, segments=64
        )
        assert plan.segments < 64
        assert plan.step >= 1
        assert any("segments" in note for note in plan.notes)

    @staticmethod
    def test_device_with_unknown_ceiling_keeps_the_upper_bound():
        limits = DeviceLimits(min_frequency=1, max_frequency=None)
        plan = plan_sweep(1_000, 900_000_000, limits=limits, points=11)
        assert plan.end == 900_000_000

    @staticmethod
    def test_logarithmic_plan_is_carried_and_needs_multi_segments():
        plan = plan_sweep(
            1_000_000,
            30_000_000,
            limits=LIMITS,
            points=101,
            segments=8,
            logarithmic=True,
        )
        assert plan.logarithmic
        assert plan.segments == 8

    @staticmethod
    def test_planned_grid_never_leaves_a_narrow_device_range():
        # Regression: the nominal 1..201 Hz span with two segments made the
        # real Sweep grid stop at 202 Hz, one step beyond the device maximum.
        limits = DeviceLimits(
            min_frequency=1,
            max_frequency=201,
            points=(101,),
            default_points=101,
        )
        plan = plan_sweep(1, 201, limits=limits, points=101, segments=2)
        grid = sweep_grid(plan.start, plan.end, plan.points, plan.segments)
        assert max(grid) <= limits.max_frequency
        assert min(grid) >= limits.min_frequency
        assert grid[0] == plan.start
        assert grid[-1] <= plan.end
        assert len(set(grid)) >= MIN_POINTS

    @staticmethod
    def test_planned_grid_never_leaves_a_realistic_narrow_range():
        # Same defect at 1 MHz: 1_000_000..1_000_200 with 2 segments.
        limits = DeviceLimits(
            min_frequency=1_000_000,
            max_frequency=1_000_200,
            points=(101,),
            default_points=101,
        )
        plan = plan_sweep(
            1_000_000, 1_000_200, limits=limits, points=101, segments=2
        )
        grid = sweep_grid(plan.start, plan.end, plan.points, plan.segments)
        assert max(grid) <= 1_000_200
        assert grid[-1] - grid[0] == plan.span
        assert plan.segments in (1, 2)

    @staticmethod
    def test_both_device_edges_are_respected_by_the_planned_grid():
        for start, end, segments in (
            (LIMITS.min_frequency, LIMITS.min_frequency + 5_000, 4),
            (LIMITS.max_frequency - 5_000, LIMITS.max_frequency, 4),
            (LIMITS.max_frequency - 1_001, LIMITS.max_frequency, 7),
        ):
            plan = plan_sweep(
                start, end, limits=LIMITS, points=101, segments=segments
            )
            grid = sweep_grid(plan.start, plan.end, plan.points, plan.segments)
            assert grid[0] == plan.start == start
            assert grid[-1] <= plan.end
            assert LIMITS.min_frequency <= grid[-1] <= LIMITS.max_frequency
            assert all(
                LIMITS.min_frequency <= f <= LIMITS.max_frequency for f in grid
            )
            # The measured stop is the device's own rounded step, so it may
            # sit up to one step away from the request - never further.
            assert abs(plan.end - end) <= max(plan.step, 1)

    @staticmethod
    def test_device_maximum_is_never_exceeded_by_the_measured_grid():
        # 101 points across 1..200 Hz need a 1.99 Hz step, which the device
        # rounds to 2 Hz: it would sample 201 Hz, one Hz past the maximum.
        # That cannot be repaired by shortening the stop, so it is refused.
        limits = DeviceLimits(
            min_frequency=1,
            max_frequency=200,
            points=(101,),
            default_points=101,
        )
        with pytest.raises(SweepPlanError, match="device range|step grid"):
            plan_sweep(1, 200, limits=limits, points=101, segments=1)
        # A request the device's own grid represents exactly is accepted
        # and stays inside the device range end to end.
        plan = plan_sweep(1, 101, limits=limits, points=101, segments=1)
        grid = sweep_grid(plan.start, plan.end, plan.points, plan.segments)
        assert grid[0] == 1 and grid[-1] == 101
        assert max(grid) <= 200
        # A span the 1 Hz grid cannot represent at all is refused rather
        # than measured far short of the requested stop.
        with pytest.raises(SweepPlanError, match="step grid|range"):
            plan_sweep(1, 150, limits=limits, points=101, segments=1)

    @staticmethod
    def test_impossible_narrow_span_is_refused_not_rescaled():
        # Ten samples' worth of distinct frequencies cannot be placed in a
        # 3 Hz span, so the plan is refused rather than widened silently.
        limits = DeviceLimits(
            min_frequency=1_000,
            max_frequency=1_003,
            points=(11,),
            default_points=11,
        )
        with pytest.raises(SweepPlanError):
            plan_sweep(1_000, 1_003, limits=limits, points=11)

    @staticmethod
    def test_plan_keeps_the_requested_stop_the_device_can_reach():
        # 4_000_000 / 401 rounds to a 9975 Hz step, so the measured grid
        # stops 25 Hz short of 16 MHz.  That is inside one step, so the
        # requested stop is kept: the device reproduces exactly this grid.
        plan = plan_sweep(
            12_000_000, 16_000_000, limits=LIMITS, points=201, segments=2
        )
        grid = sweep_grid(plan.start, plan.end, plan.points, plan.segments)
        assert plan.end == 16_000_000
        assert grid[-1] == 15_999_975
        assert 16_000_000 - grid[-1] <= plan.step

    @staticmethod
    def test_plan_refuses_a_stop_the_device_grid_cannot_represent():
        # 3 Hz over 101 points rounds to a 0 Hz step, which cannot hold 101
        # distinct samples; the request is refused rather than silently
        # measured as a shorter sweep.
        limits = DeviceLimits(
            min_frequency=1_000_000,
            max_frequency=2_000_000,
            points=(101,),
            default_points=101,
        )
        with pytest.raises(SweepPlanError):
            plan_sweep(1_000_000, 1_000_003, limits=limits, points=101)
        # With a span the step grid represents, the requested stop is kept.
        plan = plan_sweep(1_000_000, 1_000_100, limits=limits, points=101)
        assert plan.end == 1_000_100


class TestZoomPlanning:
    @staticmethod
    def test_zoom_is_centred_and_keeps_supported_points():
        plan = plan_zoom(
            14_150_000, 200_000, limits=LIMITS, points=201, segments=1
        )
        assert (plan.start, plan.end) == (14_050_000, 14_250_000)
        assert plan.points == 201
        assert not plan.notes

    @staticmethod
    def test_zero_and_negative_spans_are_rejected():
        for span in (0, -1):
            with pytest.raises(SweepPlanError, match="greater than zero"):
                plan_zoom(14_150_000, span, limits=LIMITS)

    @staticmethod
    def test_span_too_narrow_for_the_point_count_is_rejected():
        with pytest.raises(SweepPlanError, match="too narrow"):
            plan_zoom(14_150_000, 50, limits=LIMITS, points=101)

    @staticmethod
    def test_span_wider_than_the_device_is_rejected():
        with pytest.raises(SweepPlanError, match="wider than the device"):
            plan_zoom(14_150_000, 400_000_000, limits=LIMITS, points=101)

    @staticmethod
    def test_minimum_outside_the_device_is_rejected():
        with pytest.raises(SweepPlanError, match="outside the device"):
            plan_zoom(400_000_000, 200_000, limits=LIMITS)

    @staticmethod
    def test_low_edge_zoom_is_clamped_and_still_contains_the_minimum():
        plan = plan_zoom(50_500, 100_000, limits=LIMITS, points=101)
        assert plan.start == LIMITS.min_frequency
        assert plan.end - plan.start == 100_000
        assert plan.start <= 50_500 <= plan.end
        assert any("device minimum" in note for note in plan.notes)

    @staticmethod
    def test_high_edge_zoom_is_clamped_and_still_contains_the_minimum():
        plan = plan_zoom(
            LIMITS.max_frequency - 1_000, 100_000, limits=LIMITS, points=101
        )
        assert plan.end == LIMITS.max_frequency
        assert plan.start <= LIMITS.max_frequency - 1_000 <= plan.end
        assert any("device maximum" in note for note in plan.notes)

    @staticmethod
    def test_zoom_segments_that_cannot_fit_are_rejected():
        # 250 Hz fits two 100-step segments but not four: the zoom is
        # refused rather than quietly changing the step resolution.
        with pytest.raises(SweepPlanError, match="segments"):
            plan_zoom(14_150_000, 250, limits=LIMITS, points=101, segments=4)
        assert (
            plan_zoom(
                14_150_000, 250, limits=LIMITS, points=101, segments=2
            ).segments
            == 2
        )

    @staticmethod
    def test_default_zoom_span_is_usable_and_narrower():
        measured = 16_000_000
        span = auto_zoom_span(measured, 401)
        assert 400 <= span < measured
        plan_zoom(14_150_000, span, limits=LIMITS, points=401)


class TestFilenames:
    @staticmethod
    def test_name_states_timestamp_and_resonance():
        name = tuning_filename(14_150_000, datetime(2026, 11, 1, 12, 30, 15))
        assert name == "antenna_14p150MHz_20261101-123015.s1p"

    @staticmethod
    def test_name_without_a_minimum_still_has_a_timestamp():
        name = tuning_filename(None, datetime(2026, 1, 2, 3, 4, 5))
        assert name == "antenna_20260102-030405.s1p"

    @staticmethod
    def test_unique_path_never_returns_an_existing_file(tmp_path):
        first = tmp_path / "antenna_14p150MHz_20261101-123015.s1p"
        first.write_text("! existing\n", encoding="utf-8")
        second = unique_path(first)
        assert second != first
        assert second.name.endswith("_2.s1p")
        second.write_text("! existing\n", encoding="utf-8")
        assert unique_path(first).name.endswith("_3.s1p")


def _snapshot(
    completion=CompletionState.COMPLETE,
    magnitude=0.1,
    simulated=False,
    scenario="",
    points=5,
):
    samples = _samples([_reflection(magnitude)] * points)
    return snapshot_from_datapoints(
        samples,
        None,
        name="antenna",
        simulated=simulated,
        simulator_scenario=scenario or None,
        source="simulated" if simulated else "worker",
        completion=completion,
        reference_impedance=75.0,
    )


class TestExport:
    @staticmethod
    def test_measured_values_and_impedance_are_written_unchanged(tmp_path):
        snapshot = _snapshot(magnitude=0.1)
        search = find_vswr_minimum(snapshot.channel("S11"))
        target = write_s1p(
            snapshot, tmp_path / "antenna.s1p", minimum=search.minimum
        )
        text = target.read_text(encoding="utf-8")
        assert "# HZ S RI R 75" in text
        reloaded = Touchstone(str(target))
        reloaded.load()
        assert len(reloaded.s11) == 5
        assert reloaded.opts.resistance == 75
        assert reloaded.s11[0].freq == 14_000_000
        assert reloaded.s11[0].re == pytest.approx(0.1)
        assert reloaded.s11[-1].freq == 14_004_000

    @staticmethod
    def test_export_states_the_measured_minimum_and_completion(tmp_path):
        snapshot = _snapshot()
        search = find_vswr_minimum(snapshot.channel("S11"))
        target = write_s1p(snapshot, tmp_path / "m.s1p", minimum=search.minimum)
        text = target.read_text(encoding="utf-8")
        assert "antenna tuning measurement" in text
        assert "minimum VSWR 1.222" in text
        assert "14000000 Hz on the measured sample #0" in text
        assert "completion: complete sweep" in text
        assert "INCOMPLETE" not in text

    @staticmethod
    def test_simulated_provenance_survives_a_reload(tmp_path):
        snapshot = _snapshot(simulated=True, scenario="antenna")
        target = write_s1p(snapshot, tmp_path / "sim.s1p")
        reloaded = Touchstone(str(target))
        reloaded.load()
        assert reloaded.simulated
        assert reloaded.simulated_scenario == "antenna"
        assert "SIMULATED DATA" in target.read_text(encoding="utf-8")

    @staticmethod
    def test_partial_data_is_refused_unless_explicitly_allowed(tmp_path):
        snapshot = _snapshot(completion=CompletionState.PARTIAL)
        with pytest.raises(AntennaExportError, match="not complete"):
            write_s1p(snapshot, tmp_path / "partial.s1p")
        assert not (tmp_path / "partial.s1p").exists()
        target = write_s1p(
            snapshot, tmp_path / "partial.s1p", allow_partial=True
        )
        text = target.read_text(encoding="utf-8")
        assert "INCOMPLETE: partial sweep" in text

    @staticmethod
    def test_a_file_created_concurrently_is_preserved(tmp_path, monkeypatch):
        """The no-clobber publish must not destroy a racing creator.

        Regression: the old code checked ``exists()`` and then called
        ``os.replace`` unconditionally, so a file created between the check
        and the rename was silently overwritten.  The publish step itself
        must refuse.
        """
        snapshot = _snapshot()
        target = tmp_path / "race.s1p"
        real_measure = write_s1p.__globals__["measurement_touchstone"]

        def racing_measure(*args, **kwargs):
            # A concurrent writer wins the race after validation but before
            # the payload is published.
            target.write_text("KEEP CONCURRENT FILE\n", encoding="utf-8")
            return real_measure(*args, **kwargs)

        monkeypatch.setitem(
            write_s1p.__globals__, "measurement_touchstone", racing_measure
        )
        with pytest.raises(AntennaExportError, match="already exists"):
            write_s1p(snapshot, target)
        assert target.read_text(encoding="utf-8") == "KEEP CONCURRENT FILE\n"
        # No temporary file is left behind either.
        assert sorted(p.name for p in tmp_path.iterdir()) == ["race.s1p"]

    @staticmethod
    def test_the_copy_fallback_preserves_a_racing_file(tmp_path, monkeypatch):
        """Filesystems without hard links must not clobber either."""

        def no_link(*args, **kwargs):
            raise OSError("hard links unsupported")

        # Only the link primitive is unavailable; the rest of ``os`` is used
        # by the fallback itself.
        monkeypatch.setattr(os, "link", no_link)
        snapshot = _snapshot()
        target = tmp_path / "fallback.s1p"
        written = write_s1p(snapshot, target)
        assert written == target
        assert "antenna tuning measurement" in target.read_text(
            encoding="utf-8"
        )
        assert sorted(p.name for p in tmp_path.iterdir()) == ["fallback.s1p"]

        target.write_text("KEEP CONCURRENT FILE\n", encoding="utf-8")
        with pytest.raises(AntennaExportError, match="already exists"):
            write_s1p(snapshot, target)
        assert target.read_text(encoding="utf-8") == "KEEP CONCURRENT FILE\n"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["fallback.s1p"]

    @staticmethod
    def test_normalized_path_is_the_one_written(tmp_path):
        # The effective path (not the requested name) is what the caller must
        # check and confirm; the writer never writes an unrecognised suffix.
        assert normalize_s1p_path(tmp_path / "named.s2p") == (
            tmp_path / "named.s1p"
        )
        assert normalize_s1p_path(tmp_path / "named.S1P") == (
            tmp_path / "named.S1P"
        )
        assert normalize_s1p_path(tmp_path / "named") == tmp_path / "named.s1p"

    @staticmethod
    def test_overwrite_authorises_only_the_target_it_names(tmp_path):
        # Replacement permission for one file must not touch another.
        snapshot = _snapshot()
        s1p = tmp_path / "named.s1p"
        s2p = tmp_path / "named.s2p"
        s1p.write_text("KEEP S1P\n", encoding="utf-8")
        s2p.write_text("KEEP S2P\n", encoding="utf-8")
        with pytest.raises(AntennaExportError, match="already exists"):
            write_s1p(snapshot, s2p)
        assert s1p.read_text(encoding="utf-8") == "KEEP S1P\n"
        assert s2p.read_text(encoding="utf-8") == "KEEP S2P\n"

    @staticmethod
    def test_existing_file_is_never_overwritten_silently(tmp_path):
        snapshot = _snapshot()
        target = tmp_path / "keep.s1p"
        target.write_text("! hand-written\n", encoding="utf-8")
        with pytest.raises(AntennaExportError, match="already exists"):
            write_s1p(snapshot, target)
        assert target.read_text(encoding="utf-8") == "! hand-written\n"
        write_s1p(snapshot, target, overwrite=True)
        assert "antenna tuning measurement" in target.read_text(
            encoding="utf-8"
        )

    @staticmethod
    def test_empty_measurement_cannot_be_exported(tmp_path):
        snapshot = snapshot_from_datapoints(None, None, name="empty")
        with pytest.raises(AntennaExportError, match="no S11"):
            write_s1p(snapshot, tmp_path / "empty.s1p")

    @staticmethod
    def test_extension_is_forced_to_s1p(tmp_path):
        target = write_s1p(_snapshot(), tmp_path / "antenna.txt")
        assert target.suffix == ".s1p"
        assert target.exists()

    @staticmethod
    def test_created_timestamp_is_utc():
        # The exported note must not depend on the machine's timezone.
        snapshot = _snapshot()
        assert snapshot.timestamp.tzinfo is not None
        assert snapshot.timestamp.tzinfo == timezone.utc
