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
"""DSP-03: the continuous running average.

The reference values are computed straight from the definitions: the
arithmetic mean of every complete sweep for ``cumulative``, and the explicit
``avg_n = (1 - alpha) * avg_(n-1) + alpha * x_n`` recurrence for
``exponential``.  Nothing here is read back out of the implementation, so a
wrong recurrence cannot hide behind itself.
"""

import math

import pytest

from NanoVNASaver.Processing import (
    AVERAGE_CUMULATIVE,
    AVERAGE_EXPONENTIAL,
    AVERAGE_MODES,
    STAGE_AVERAGE,
    ProcessingConfig,
    ProcessingError,
    ProcessingStage,
    RunningAverageState,
    apply_processing,
    average_stage,
    running_average_blend,
    validate_average_mode,
)
from NanoVNASaver.RFTools import Datapoint

STEP = 1_000_000
START = 10_000_000
POINTS = 5


def _sweep(offsets, *, start=START, step=STEP, points=POINTS):
    """One sweep whose sample ``i`` is ``offsets[i]`` (real or complex)."""
    return {
        "S11": tuple(
            Datapoint(start + index * step, value.real, value.imag)
            for index, value in enumerate(
                complex(value) for value in offsets[:points]
            )
        )
    }


def _reference_cumulative(sweeps):
    """Arithmetic mean of every sweep, per (channel, frequency)."""
    totals = {}
    counts = {}
    for sweep in sweeps:
        for name, points in sweep.items():
            for point in points:
                key = (name, point.freq)
                totals[key] = totals.get(key, 0j) + point.z
                counts[key] = counts.get(key, 0) + 1
    return {key: total / counts[key] for key, total in totals.items()}


def _reference_exponential(samples, alpha):
    """EWMA recurrence starting at the first sweep, as a closed form."""
    first = samples[0]
    value = first
    for sample in samples[1:]:
        value = (1.0 - alpha) * value + alpha * sample
    # Independent check of the same number:
    folded = (1.0 - alpha) ** (len(samples) - 1) * first
    for index, sample in enumerate(samples[1:], start=2):
        folded += alpha * (1.0 - alpha) ** (len(samples) - index) * sample
    assert value == pytest.approx(folded, abs=1e-12)
    return value


class TestPolicyValidation:
    @staticmethod
    def test_the_two_modes_are_named():
        assert AVERAGE_MODES == ("cumulative", "exponential")
        assert validate_average_mode(AVERAGE_CUMULATIVE) == ("cumulative", None)
        assert validate_average_mode(AVERAGE_EXPONENTIAL, 0.25) == (
            "exponential",
            0.25,
        )

    @staticmethod
    def test_a_missing_mode_means_cumulative():
        assert validate_average_mode(None) == (AVERAGE_CUMULATIVE, None)

    @staticmethod
    @pytest.mark.parametrize("mode", ["mean", "", "CUMULATIVE", 3])
    def test_an_unknown_mode_is_refused(mode):
        with pytest.raises(ProcessingError):
            validate_average_mode(mode)

    @staticmethod
    def test_cumulative_refuses_an_alpha():
        with pytest.raises(ProcessingError):
            validate_average_mode(AVERAGE_CUMULATIVE, 0.5)

    @staticmethod
    @pytest.mark.parametrize("alpha", [None, 0.0, -0.1, 1.5, math.nan, "half"])
    def test_exponential_needs_an_alpha_in_range(alpha):
        with pytest.raises(ProcessingError):
            validate_average_mode(AVERAGE_EXPONENTIAL, alpha)

    @staticmethod
    def test_an_alpha_of_one_means_only_the_newest_sweep():
        assert validate_average_mode(AVERAGE_EXPONENTIAL, 1.0) == (
            "exponential",
            1.0,
        )


class TestBlendRecurrence:
    @staticmethod
    def test_the_cumulative_recurrence_is_the_arithmetic_mean():
        samples = [1.0, 3.0, 5.0, 7.0, 11.0]
        value = samples[0]
        for count, sample in enumerate(samples[1:], start=2):
            value = running_average_blend(value, sample, count=count)
            expected = sum(samples[:count]) / count
            assert value == pytest.approx(expected, abs=1e-12)

    @staticmethod
    def test_the_cumulative_recurrence_matches_plain_python_for_complex_data():
        samples = [complex(0.1 * index, -0.05 * index) for index in range(1, 9)]
        value = samples[0]
        for count, sample in enumerate(samples[1:], start=2):
            value = running_average_blend(value, sample, count=count)
            window = samples[:count]
            assert value == pytest.approx(sum(window) / len(window), abs=1e-12)

    @staticmethod
    def test_the_exponential_recurrence_matches_its_closed_form():
        alpha = 0.3
        samples = [
            complex(0.2 + 0.1 * index, 0.7 - 0.05 * index) for index in range(6)
        ]
        value = samples[0]
        for index, sample in enumerate(samples[1:], start=1):
            value = running_average_blend(
                value,
                sample,
                count=index + 1,
                mode=AVERAGE_EXPONENTIAL,
                alpha=alpha,
            )
            assert value == pytest.approx(
                _reference_exponential(samples[: index + 1], alpha), abs=1e-12
            )

    @staticmethod
    def test_an_alpha_of_one_returns_the_newest_sample_unchanged():
        value = running_average_blend(
            1.0, 4.0, count=7, mode=AVERAGE_EXPONENTIAL, alpha=1.0
        )
        assert value == pytest.approx(4.0)

    @staticmethod
    @pytest.mark.parametrize("count", [0, -3])
    def test_a_count_below_one_is_refused(count):
        with pytest.raises(ProcessingError):
            running_average_blend(0.0, 1.0, count=count)

    @staticmethod
    def test_a_fractional_count_is_refused():
        with pytest.raises(ProcessingError):
            running_average_blend(0.0, 1.0, count=2.5)

    @staticmethod
    def test_the_blend_keeps_one_complex_value_per_point():
        value = running_average_blend(
            complex(0.3, -0.4), complex(0.1, 0.2), count=2
        )
        assert isinstance(value, complex)
        assert value == pytest.approx(complex(0.2, -0.1), abs=1e-12)


class TestStateAccumulation:
    @staticmethod
    def test_repeated_sweeps_average_to_the_plain_mean():
        sweeps = [
            _sweep([float(n + index) for index in range(POINTS)])
            for n in range(4)
        ]
        state = RunningAverageState()
        published = None
        for sweep in sweeps:
            state, published = state.blend(sweep, context="device")
        expected = _reference_cumulative(sweeps)
        assert state.count == 4
        assert state.effective_count == 4
        for point in published["S11"]:
            assert point.z == pytest.approx(
                expected[("S11", point.freq)], abs=1e-12
            )

    @staticmethod
    def test_the_first_sweep_is_published_unchanged():
        sweep = _sweep([1.0, 2.0, 3.0, 4.0, 5.0])
        state, published = RunningAverageState().blend(sweep, context="device")
        assert state.count == 1
        assert published["S11"] == sweep["S11"]

    @staticmethod
    def test_a_two_sweep_average_is_the_midpoint():
        first = _sweep([0.0] * POINTS)
        second = _sweep([2.0] * POINTS)
        state = RunningAverageState()
        state, _ = state.blend(first, context="device")
        _state, published = state.blend(second, context="device")
        assert [point.re for point in published["S11"]] == pytest.approx(
            [1.0] * POINTS
        )

    @staticmethod
    def test_the_exponential_mode_matches_the_closed_form_per_point():
        alpha = 0.4
        sweeps = [
            _sweep(
                [complex(0.1 * n + index, 0.02 * n) for index in range(POINTS)]
            )
            for n in range(5)
        ]
        reference = {
            point.freq: _reference_exponential(
                [sweep["S11"][index].z for sweep in sweeps], alpha
            )
            for index, point in enumerate(sweeps[0]["S11"])
        }
        state = RunningAverageState(mode=AVERAGE_EXPONENTIAL, alpha=alpha)
        published = None
        for sweep in sweeps:
            state, published = state.blend(sweep, context="device")
        assert state.count == 5
        for point in published["S11"]:
            assert point.z == pytest.approx(reference[point.freq], abs=1e-12)

    @staticmethod
    def test_the_state_is_immutable_and_the_previous_state_is_untouched():
        sweep = _sweep([1.0] * POINTS)
        empty = RunningAverageState()
        blended, _ = empty.blend(sweep, context="device")
        assert empty.count == 0
        assert empty.values == ()
        assert blended.count == 1
        second, _ = blended.blend(sweep, context="device")
        assert blended.count == 1
        assert second.count == 2

    @staticmethod
    def test_one_channel_on_its_own_can_be_averaged():
        state = RunningAverageState()
        for value in (1.0, 3.0):
            sweep = {"S11": (Datapoint(START, value, 0.0),)}
            state, published = state.blend(sweep, context="device")
        assert published["S11"][0].re == pytest.approx(2.0)


class TestResetPolicy:
    @staticmethod
    def test_a_context_change_starts_a_new_accumulation():
        sweep = _sweep([1.0] * POINTS)
        state, _ = RunningAverageState().blend(sweep, context="device-a")
        state, _ = state.blend(sweep, context="device-a")
        assert state.count == 2
        restarted, _ = state.blend(sweep, context="device-b")
        assert restarted.count == 1
        assert restarted.context == "device-b"

    @staticmethod
    def test_a_grid_change_starts_a_new_accumulation():
        state, _ = RunningAverageState().blend(
            _sweep([1.0] * POINTS), context="device"
        )
        state, _ = state.blend(_sweep([1.0] * POINTS), context="device")
        assert state.count == 2
        moved = _sweep([1.0] * POINTS, start=START + 13 * STEP)
        restarted, _ = state.blend(moved, context="device")
        assert restarted.count == 1

    @staticmethod
    def test_a_channel_set_change_starts_a_new_accumulation():
        state, _ = RunningAverageState().blend(
            {"S11": (Datapoint(START, 0.5, 0.0),)}, context="device"
        )
        state, _ = state.blend(
            {"S11": (Datapoint(START, 0.5, 0.0),)}, context="device"
        )
        assert state.count == 2
        two_port = {
            "S11": (Datapoint(START, 0.5, 0.0),),
            "S21": (Datapoint(START, 0.1, 0.0),),
        }
        restarted, published = state.blend(two_port, context="device")
        assert restarted.count == 1
        assert set(published) == {"S11", "S21"}

    @staticmethod
    def test_a_grid_change_does_not_blend_across_the_boundary():
        state, _ = RunningAverageState().blend(
            _sweep([9.0] * POINTS), context="device"
        )
        moved = _sweep([1.0] * POINTS, start=START + 7 * STEP)
        _restarted, published = state.blend(moved, context="device")
        assert [point.re for point in published["S11"]] == pytest.approx(
            [1.0] * POINTS
        )

    @staticmethod
    def test_switching_the_weighting_discards_the_old_average():
        sweep = _sweep([1.0] * POINTS)
        state, _ = RunningAverageState().blend(sweep, context="device")
        state, _ = state.blend(sweep, context="device")
        assert state.count == 2
        reweighted = state.with_policy(mode=AVERAGE_EXPONENTIAL, alpha=0.5)
        assert reweighted.count == 0
        assert reweighted.mode == AVERAGE_EXPONENTIAL
        assert reweighted.alpha == 0.5

    @staticmethod
    def test_keeping_the_same_weighting_keeps_the_average():
        state, _ = RunningAverageState().blend(
            _sweep([1.0] * POINTS), context="device"
        )
        assert state.with_policy(mode=AVERAGE_CUMULATIVE) is state

    @staticmethod
    def test_reset_clears_the_average_but_keeps_the_policy():
        state = RunningAverageState(
            mode=AVERAGE_EXPONENTIAL, alpha=0.25, count=4
        )
        cleared = state.reset()
        assert cleared.count == 0
        assert cleared.values == ()
        assert cleared.mode == AVERAGE_EXPONENTIAL
        assert cleared.alpha == 0.25

    @staticmethod
    def test_invalidate_records_why_the_average_was_dropped():
        state, _ = RunningAverageState().blend(
            _sweep([1.0] * POINTS), context="device"
        )
        dropped = state.invalidate("sweep cancelled")
        assert dropped.count == 0
        assert dropped.discarded == "sweep cancelled"
        assert state.count == 1

    @staticmethod
    def test_a_negative_count_is_refused():
        with pytest.raises(ProcessingError):
            RunningAverageState(count=-1)

    @staticmethod
    def test_an_empty_average_reports_no_weighting():
        state = RunningAverageState()
        assert state.active is False
        assert state.effective_count == 0
        assert state.weighting == "no sweeps blended"


class TestCompleteness:
    @staticmethod
    def test_an_empty_sweep_raises():
        with pytest.raises(ProcessingError):
            RunningAverageState().blend({}, context="device")

    @staticmethod
    def test_a_partial_sweep_raises_instead_of_averaging_a_short_grid():
        partial = {
            "S11": (Datapoint(START, 0.5, 0.0),),
            "S21": (),
        }
        with pytest.raises(ProcessingError):
            RunningAverageState().blend(partial, context="device")

    @staticmethod
    def test_a_non_finite_sample_raises_and_leaves_the_state_useful():
        state, _ = RunningAverageState().blend(
            _sweep([1.0] * POINTS), context="device"
        )
        poisoned = {"S11": (Datapoint(START, math.nan, 0.0),)}
        with pytest.raises(ProcessingError):
            state.blend(poisoned, context="device")
        assert state.count == 1
        clean, published = state.blend(_sweep([3.0] * POINTS), context="device")
        assert clean.count == 2
        assert published["S11"][0].re == pytest.approx(2.0)

    @staticmethod
    def test_an_infinite_sample_raises():
        poisoned = {"S11": (Datapoint(START, math.inf, 0.0),)}
        with pytest.raises(ProcessingError):
            RunningAverageState().blend(poisoned, context="device")


class TestEffectiveCount:
    @staticmethod
    def test_cumulative_reports_the_number_of_sweeps():
        state = RunningAverageState(count=6, values=((1, 0.5 + 0j),))
        assert state.effective_count == 6

    @staticmethod
    def test_exponential_reports_the_equivalent_number_of_sweeps():
        state = RunningAverageState(
            mode=AVERAGE_EXPONENTIAL,
            alpha=0.25,
            count=6,
            values=((1, 0.5 + 0j),),
        )
        assert state.effective_count == 4

    @staticmethod
    def test_the_equivalent_count_never_exceeds_the_sweeps_seen():
        state = RunningAverageState(
            mode=AVERAGE_EXPONENTIAL,
            alpha=0.1,
            count=2,
            values=((1, 0.5 + 0j),),
        )
        assert state.effective_count == 2

    @staticmethod
    def test_the_equivalent_count_is_at_least_one():
        state = RunningAverageState(
            mode=AVERAGE_EXPONENTIAL,
            alpha=1.0,
            count=1,
            values=((1, 0.5 + 0j),),
        )
        assert state.effective_count == 1

    @staticmethod
    def test_the_weighting_string_states_what_is_in_force():
        cumulative = RunningAverageState(count=3, values=((1, 0j),))
        assert cumulative.weighting == "cumulative mean of 3 sweep(s)"
        exponential = RunningAverageState(
            mode=AVERAGE_EXPONENTIAL,
            alpha=0.5,
            count=3,
            values=((1, 0j),),
        )
        assert "exponential alpha=0.5" in exponential.weighting
        assert "equivalent to 2 equal sweeps" in exponential.weighting


class TestThroughThePipeline:
    @staticmethod
    def test_the_config_reports_a_running_state_to_carry_forward():
        sweep = _sweep([1.0] * POINTS)
        config = ProcessingConfig.build(average_stage())
        first = apply_processing(sweep, config, average_context="device")
        assert first.running is not None
        assert first.running.count == 1
        second = apply_processing(
            _sweep([3.0] * POINTS),
            config,
            running=first.running,
            average_context="device",
        )
        assert second.running.count == 2
        assert second.effective_average_count == 2
        assert [point.re for point in second.channels["S11"]] == pytest.approx(
            [2.0] * POINTS
        )

    @staticmethod
    def test_the_accumulated_sweeps_equal_the_plain_mean():
        config = ProcessingConfig.build(average_stage())
        sweeps = [
            _sweep([float(2 * n + index) for index in range(POINTS)])
            for n in range(3)
        ]
        state = None
        result = None
        for sweep in sweeps:
            result = apply_processing(
                sweep, config, running=state, average_context="device"
            )
            state = result.running
        expected = _reference_cumulative(sweeps)
        for point in result.channels["S11"]:
            assert point.z == pytest.approx(
                expected[("S11", point.freq)], abs=1e-12
            )

    @staticmethod
    def test_a_loop_that_drops_the_state_restarts_every_sweep():
        """The integrator must publish ``result.running``, not just read it."""
        config = ProcessingConfig.build(average_stage())
        sweep = _sweep([1.0] * POINTS)
        counts = [
            apply_processing(
                sweep, config, average_context="device"
            ).running.count
            for _ in range(3)
        ]
        assert counts == [1, 1, 1]

    @staticmethod
    def test_the_raw_channels_are_never_overwritten():
        sweep = _sweep([1.0] * POINTS)
        before = {name: tuple(points) for name, points in sweep.items()}
        apply_processing(
            sweep,
            ProcessingConfig.build(average_stage()),
            average_context="device",
        )
        assert sweep == before

    @staticmethod
    def test_the_notes_name_the_weighting():
        result = apply_processing(
            _sweep([1.0] * POINTS),
            ProcessingConfig.build(average_stage()),
            average_context="device",
        )
        assert any(
            "cumulative mean of 1 sweep(s)" in note for note in result.notes
        )

    @staticmethod
    def test_no_average_stage_means_no_state_to_carry():
        result = apply_processing(_sweep([1.0] * POINTS), None)
        assert result.running is None
        assert result.effective_average_count == 0

    @staticmethod
    def test_an_exponential_stage_carries_its_alpha_through_the_pipeline():
        config = ProcessingConfig.build(
            average_stage(mode=AVERAGE_EXPONENTIAL, alpha=0.5)
        )
        sweep = _sweep([1.0] * POINTS)
        first = apply_processing(sweep, config, average_context="device")
        second = apply_processing(
            _sweep([3.0] * POINTS),
            config,
            running=first.running,
            average_context="device",
        )
        # (1 - 0.5) * 1.0 + 0.5 * 3.0
        assert [point.re for point in second.channels["S11"]] == pytest.approx(
            [2.0] * POINTS
        )

    @staticmethod
    def test_an_average_stage_without_a_ready_state_is_seeded():
        config = ProcessingConfig.build(
            average_stage(mode=AVERAGE_EXPONENTIAL, alpha=0.5)
        )
        result = apply_processing(
            _sweep([1.0] * POINTS), config, average_context="device"
        )
        assert result.running.mode == AVERAGE_EXPONENTIAL
        assert result.running.alpha == 0.5

    @staticmethod
    def test_an_unusable_weighting_policy_is_refused():
        stage = ProcessingStage.build(STAGE_AVERAGE, {"mode": "bogus"})
        with pytest.raises(ProcessingError):
            apply_processing(
                _sweep([1.0] * POINTS),
                ProcessingConfig.build(stage),
                average_context="device",
            )

    @staticmethod
    def test_a_stored_alpha_on_cumulative_is_refused():
        stage = ProcessingStage.build(
            STAGE_AVERAGE, {"mode": AVERAGE_CUMULATIVE, "alpha": 0.25}
        )
        with pytest.raises(ProcessingError):
            apply_processing(
                _sweep([1.0] * POINTS),
                ProcessingConfig.build(stage),
                average_context="device",
            )

    @staticmethod
    def test_an_exponential_stage_without_alpha_is_refused():
        stage = ProcessingStage.build(
            STAGE_AVERAGE, {"mode": AVERAGE_EXPONENTIAL}
        )
        with pytest.raises(ProcessingError):
            apply_processing(
                _sweep([1.0] * POINTS),
                ProcessingConfig.build(stage),
                average_context="device",
            )

    @staticmethod
    def test_the_average_stage_survives_a_configuration_round_trip():
        for stage in (
            average_stage(),
            average_stage(mode=AVERAGE_EXPONENTIAL, alpha=0.125),
        ):
            config = ProcessingConfig.build(stage)
            restored = ProcessingConfig.from_dict(config.to_dict())
            assert restored == config
            assert restored.identity() == config.identity()
            assert restored.find(STAGE_AVERAGE).as_dict == stage.as_dict

    @staticmethod
    def test_average_stage_validates_at_construction():
        with pytest.raises(ProcessingError):
            average_stage(mode="median")
        with pytest.raises(ProcessingError):
            average_stage(alpha=0.5)
        with pytest.raises(ProcessingError):
            average_stage(mode=AVERAGE_EXPONENTIAL)


class TestSwitchingWeightingPolicy:
    """Changing the policy drops the average; it must never break sweeps."""

    @staticmethod
    def test_exponential_to_cumulative_is_accepted():
        state = RunningAverageState(mode="exponential", alpha=0.3)
        changed = state.with_policy(mode="cumulative")
        assert (changed.mode, changed.alpha) == ("cumulative", None)

    @staticmethod
    def test_cumulative_to_exponential_still_needs_an_alpha():
        with pytest.raises(ProcessingError):
            RunningAverageState().with_policy(mode="exponential")

    @staticmethod
    def test_the_same_mode_keeps_its_alpha():
        state = RunningAverageState(mode="exponential", alpha=0.3)
        assert state.with_policy(mode="exponential") is state

    @staticmethod
    def test_a_stage_switch_runs_after_the_ui_reset_keeps_the_old_policy():
        points = [
            Datapoint(1_000_000, 0.5, 0.1),
            Datapoint(2_000_000, 0.4, 0.2),
        ]
        exponential = ProcessingConfig.build(
            average_stage(mode="exponential", alpha=0.3)
        )
        running = apply_processing({"S11": points}, exponential).running
        # What the settings window does on any pipeline edit.
        running = running.reset()
        result = apply_processing(
            {"S11": points},
            ProcessingConfig.build(average_stage(mode="cumulative")),
            running=running,
        )
        assert result.running.mode == "cumulative"
        assert result.running.count == 1
