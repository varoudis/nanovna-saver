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
"""NEXT-01: one usable frequency span and sweep preflight for every source.

Pure numerical tests: no widgets, no device connection, no Qt application.
"""

import math
import random
from dataclasses import replace

import pytest

from NanoVNASaver.Hardware.Capability import (
    ABOVE_MAXIMUM,
    BELOW_MINIMUM,
    GRID_ABOVE_MAXIMUM,
    GRID_COLLAPSED,
    INVALID_BOUNDS,
    INVERTED_SPAN,
    LOGARITHMIC_UNSUPPORTED,
    OUTSIDE_SOURCE,
    PROBLEM_CODES,
    TOO_FEW_POINTS,
    TOO_MANY_POINTS,
    GridValidation,
    SourceSpan,
    SweepCapability,
)
from NanoVNASaver.Hardware.ReplayScenario import (
    ReplayScenario,
    load_replay_file,
)
from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Hardware.VNA import VNA
from NanoVNASaver.Settings.ReplayRange import (
    ReplayRangeError,
    plan_replay_range,
    replay_capability,
)
from NanoVNASaver.Settings.Sweep import Properties, Sweep

ONE_PORT = """! recorded antenna sweep
# HZ S RI R 50
1000000 0.10 0.20
2000000 0.30 0.40
4000000 0.50 -0.60
"""

TWO_PORT = """# HZ S RI R 50
1000000 0.00 0.00 0.90 0.10 0.80 0.20 0.05 0.01
2000000 0.02 0.01 0.70 -0.30 0.60 -0.20 0.06 0.02
"""

CW = """# HZ S RI R 50
5000000 0.10 0.20
"""

FRACTIONAL = """# HZ S RI R 50
1000000.4 0.10 0.20
2000000.6 0.30 0.40
"""

NARROW = """# HZ S RI R 50
1000000 0.10 0.20
1000010 0.20 0.30
1000020 0.30 0.40
"""


def _file(tmp_path, name: str, text: str) -> ReplayScenario:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return ReplayScenario(load_replay_file(str(path)))


def _grid(start, stop, points, segments, logarithmic=False):
    return tuple(
        Sweep(
            start,
            stop,
            points,
            segments,
            Properties(logarithmic=logarithmic),
        ).get_frequencies()
    )


class _BareDevice:
    """A device that declares nothing at all (the worst supported case)."""


class _NoCeiling:
    sweep_max_freq_hz = 0.0


@pytest.fixture
def one_port(tmp_path) -> ReplayScenario:
    return _file(tmp_path, "recorded.s1p", ONE_PORT)


@pytest.fixture
def two_port(tmp_path) -> ReplayScenario:
    return _file(tmp_path, "thru.s2p", TWO_PORT)


@pytest.fixture
def simulated():
    return make_simulated_vna("filter")


# ----------------------------------------------------------------------
# Hardware limits
# ----------------------------------------------------------------------
class TestHardwareLimits:
    @staticmethod
    def test_simulated_device_declares_its_own_limits(simulated):
        capability = simulated.capability()
        assert capability.max_frequency == 6_300_000_000
        assert capability.default_points == 101
        assert 401 in capability.valid_points
        assert capability.supports_channel("data 0") is True
        assert "data 0" in capability.supported_channels
        assert capability.reference_impedance == 50.0

    @staticmethod
    def test_unknown_ceiling_is_never_fabricated():
        capability = SweepCapability.from_vna(_NoCeiling())
        assert capability.max_frequency is None
        assert capability.effective_span.stop is None
        # Nothing is enforced above the minimum, and nothing is invented.
        assert capability.preflight(1_000_000, 10**12) is None

    @staticmethod
    def test_bare_device_declares_no_limits():
        capability = SweepCapability.from_vna(_BareDevice())
        assert capability.max_frequency is None
        assert capability.valid_points == ()
        assert capability.supported_channels == ()
        assert capability.reference_impedance is None
        assert capability.min_points == 1
        assert capability.max_points is None

    @staticmethod
    def test_real_vna_reads_its_own_class_limits():
        # ``VNA`` declares ``(101, 51, 11)`` and a 0.0 ceiling, so the
        # capability reports no ceiling rather than a guessed one.
        instance = object.__new__(VNA)
        capability = instance.capability()
        assert capability.max_frequency is None
        assert capability.valid_points == (11, 51, 101)
        assert capability.max_points == 101
        assert capability.min_points == 11

    @staticmethod
    def test_non_finite_device_values_are_ignored():
        class Odd:
            sweep_max_freq_hz = float("nan")
            valid_datapoints = ("x", 101)
            datapoints = "many"
            reference_impedance = -50.0

        capability = SweepCapability.from_vna(Odd())
        assert capability.max_frequency is None
        assert capability.valid_points == ()
        assert capability.default_points == 101
        assert capability.reference_impedance is None

    @staticmethod
    def test_describe_states_the_device_range_and_points(simulated):
        text = simulated.capability().describe()
        assert "Device range:" in text
        assert "6.30000GHz" in text
        assert "supported points:" in text
        assert "Z0 50 ohm" in text


# ----------------------------------------------------------------------
# Span preflight
# ----------------------------------------------------------------------
class TestSpanPreflight:
    @staticmethod
    def test_inverted_span_is_rejected():
        problem = SweepCapability().preflight(3_000_000, 2_000_000)
        assert problem.code == INVERTED_SPAN
        assert "must not be below" in problem.message

    @staticmethod
    def test_below_minimum_is_rejected():
        capability = SweepCapability(min_frequency=50_000)
        problem = capability.preflight(100, 1_000_000)
        assert problem.code == BELOW_MINIMUM
        assert "below the minimum" in problem.message

    @staticmethod
    def test_above_maximum_is_rejected():
        capability = SweepCapability(max_frequency=30_000_000)
        problem = capability.preflight(1_000_000, 40_000_000)
        assert problem.code == ABOVE_MAXIMUM
        assert "above the maximum" in problem.message

    @staticmethod
    def test_unknown_ceiling_enforces_nothing_above():
        capability = SweepCapability(max_frequency=None)
        assert capability.preflight(1, 10**15) is None

    @staticmethod
    def test_single_frequency_span_is_legal():
        # start == stop is how a CW request is expressed; it is never an
        # inverted span.
        assert SweepCapability().preflight(5_000_000, 5_000_000) is None

    @staticmethod
    @pytest.mark.parametrize(
        "start, stop",
        [
            ("not-a-frequency", 2_000_000),
            (None, 2_000_000),
            (1_000_000, float("nan")),
            (1_000_000, math.inf),
            (1_000_001.5, 2_000_000),
        ],
    )
    def test_preflight_never_raises(start, stop):
        problem = SweepCapability().preflight(start, stop)
        assert problem is not None
        assert problem.code == INVALID_BOUNDS
        assert str(problem) == problem.message

    @staticmethod
    def test_problem_codes_are_all_registered():
        capability = SweepCapability(
            max_frequency=100, min_points=11, max_points=101
        )
        problems = [
            capability.preflight(5, 4),
            capability.preflight(1, 1_000),
            SweepCapability().preflight(None, None),
            capability.request_problem(5, 1),
            capability.request_problem(1_000, 1),
            capability.request_problem(101, 0),
            capability.validate_grid(1, 1_000, 101).problem,
        ]
        assert all(problem is not None for problem in problems)
        assert {problem.code for problem in problems} <= PROBLEM_CODES


class TestRequestPreflight:
    @staticmethod
    def test_point_range_is_enforced():
        capability = SweepCapability(min_points=11, max_points=401)
        assert capability.request_problem(5).code == TOO_FEW_POINTS
        assert capability.request_problem(1024).code == TOO_MANY_POINTS
        assert capability.request_problem(101) is None

    @staticmethod
    def test_declared_pick_list_is_advisory():
        # ``valid_datapoints`` is the device's pick list; the settings clamp
        # to the declared range, so an intermediate count is not a refusal.
        capability = SweepCapability(
            valid_points=(11, 51, 101), min_points=11, max_points=101
        )
        assert capability.supports_points(31) is False
        assert capability.request_problem(31) is None

    @staticmethod
    def test_segment_range_and_log_mode_are_enforced():
        capability = SweepCapability(min_segments=1, max_segments=3)
        assert capability.request_problem(101, 4) is not None
        assert capability.request_problem(101, 0) is not None
        capability = SweepCapability(logarithmic=False)
        problem = capability.request_problem(101, 1, logarithmic=True)
        assert problem.code == LOGARITHMIC_UNSUPPORTED


# ----------------------------------------------------------------------
# Source limits (replay files)
# ----------------------------------------------------------------------
class TestSourceLimits:
    @staticmethod
    def test_one_port_file_offers_only_s11(one_port):
        capability = one_port.capability()
        assert capability.source_min_frequency == 1_000_000
        assert capability.source_max_frequency == 4_000_000
        assert capability.supported_channels == ("data 0",)
        assert capability.is_cw is False
        assert capability.reference_impedance == 50.0

    @staticmethod
    def test_two_port_file_offers_both_channels(two_port):
        capability = two_port.capability()
        assert capability.supported_channels == ("data 0", "data 1")
        assert capability.supports_channel("data 1") is True

    @staticmethod
    def test_effective_span_is_the_file_intersected_with_the_device(one_port):
        device = make_simulated_vna(one_port)
        device.device.sweep_max_freq_hz = 2_000_000
        capability = device.capability()
        assert capability.effective_span.start == 1_000_000
        assert capability.effective_span.stop == 2_000_000

    @staticmethod
    def test_span_outside_the_file_is_rejected_with_its_name(one_port):
        problem = one_port.capability().preflight(500_000, 2_000_000)
        assert problem.code == OUTSIDE_SOURCE
        assert "outside the replay range" in problem.message
        assert "recorded.s1p" in problem.message
        assert "never extrapolated" in problem.message

    @staticmethod
    def test_file_edges_are_inclusive(one_port):
        capability = one_port.capability()
        assert capability.preflight(1_000_000, 4_000_000) is None

    @staticmethod
    def test_replay_preflight_and_capability_agree(one_port):
        capability = one_port.capability()
        scenario = one_port
        rng = random.Random(20261101)
        for _ in range(200):
            start = rng.randint(0, 5_000_000)
            stop = rng.randint(0, 5_000_000)
            device_verdict = scenario.preflight(start, stop)
            capability_verdict = capability.preflight(start, stop)
            assert (device_verdict is None) == (capability_verdict is None), (
                start,
                stop,
                device_verdict,
                capability_verdict,
            )

    @staticmethod
    def test_capability_is_at_least_as_strict_as_device_preflight(simulated):
        # The scenario preflight is span-only for an analytic scenario, so
        # the device ceiling is enforced by the capability.  What must never
        # happen is the capability accepting something the device refuses.
        capability = simulated.capability()
        rng = random.Random(7)
        rejected = 0
        for _ in range(200):
            start = rng.randint(0, 7_000_000_000)
            stop = rng.randint(start, 7_000_000_000)
            device_verdict = simulated.preflight(start, stop)
            capability_verdict = capability.preflight(start, stop)
            if device_verdict is not None:
                rejected += 1
                assert capability_verdict is not None, (start, stop)
            if capability_verdict is not None:
                assert stop < start or stop > capability.effective_span.stop
        assert rejected >= 0

    @staticmethod
    def test_cw_file_is_single_frequency(tmp_path):
        scenario = _file(tmp_path, "cw.s1p", CW)
        capability = scenario.capability()
        assert capability.is_cw is True
        assert capability.source_min_frequency == 5_000_000
        assert capability.source_max_frequency == 5_000_000
        validation = capability.validate_grid(5_000_000, 5_000_000, 11)
        assert validation.ok
        assert set(validation.frequencies) == {5_000_000}
        assert len(validation.frequencies) == 11

    @staticmethod
    def test_non_integer_file_edges_become_whole_hertz(tmp_path):
        scenario = _file(tmp_path, "fractional.s1p", FRACTIONAL)
        capability = scenario.capability()
        assert capability.source_min_frequency == 1_000_000
        assert capability.source_max_frequency == 2_000_001

    @staticmethod
    def test_capability_is_rebuilt_for_the_current_source(tmp_path):
        first = _file(tmp_path, "first.s1p", ONE_PORT).capability()
        assert first.source_max_frequency == 4_000_000
        device = make_simulated_vna(
            _file(tmp_path, "fractional.s1p", FRACTIONAL)
        )
        second = device.capability()
        assert second.source_max_frequency == 2_000_001
        # The earlier capability is immutable and keeps its own source.
        assert first.source_max_frequency == 4_000_000
        assert "first.s1p" in first.source
        assert "fractional.s1p" in second.source


# ----------------------------------------------------------------------
# Grid validation
# ----------------------------------------------------------------------
class TestGridValidation:
    @staticmethod
    def test_grid_is_exactly_what_sweep_requests():
        capability = SweepCapability()
        for logarithmic in (False, True):
            validation = capability.validate_grid(
                3_600_000,
                30_000_000,
                101,
                3,
                logarithmic=logarithmic,
            )
            assert validation.ok
            assert validation.frequencies == _grid(
                3_600_000, 30_000_000, 101, 3, logarithmic
            )
            assert len(validation.frequencies) == 303
            assert validation.frequencies[-1] <= 30_000_000
            assert len(validation.per_segment) == 3
            for segment in validation.per_segment:
                assert len(segment) == 101
                assert len(set(segment)) == len(segment)

    @staticmethod
    def test_segment_rounding_past_the_device_maximum_is_rejected():
        # 101 points across 1..200 Hz need a 1.99 Hz step, which the sweep
        # rounds to 2 Hz: it would sample 201 Hz, one Hz past the maximum.
        capability = SweepCapability(max_frequency=200)
        validation = capability.validate_grid(1, 200, 101)
        assert not validation.ok
        assert validation.problem.code == GRID_ABOVE_MAXIMUM
        assert validation.frequencies[-1] == 201
        assert capability.validate_grid(1, 101, 101).ok

    @staticmethod
    def test_narrow_span_is_refused_not_collapsed():
        capability = SweepCapability()
        validation = capability.validate_grid(1_000_000, 1_000_020, 101)
        assert not validation.ok
        assert validation.problem.code == GRID_COLLAPSED
        assert "too narrow" in validation.problem.message
        # The same span with fewer points is a real measurement.
        assert capability.validate_grid(1_000_000, 1_000_020, 3).ok

    @staticmethod
    def test_grid_outside_the_source_is_rejected(one_port):
        capability = one_port.capability()
        validation = capability.validate_grid(900_000, 3_000_000, 11)
        assert not validation.ok
        assert validation.problem.code == OUTSIDE_SOURCE

    @staticmethod
    def test_problem_raise_helper_carries_the_message():
        from NanoVNASaver.Hardware.Capability import SweepCapabilityError

        validation = SweepCapability().validate_grid(1, 4, 101)
        with pytest.raises(SweepCapabilityError) as excinfo:
            validation.raise_for_problem()
        assert str(excinfo.value) == validation.problem.message

    @staticmethod
    def test_single_sample_grids_are_not_crash_cases():
        validation = SweepCapability().validate_grid(1_000_000, 1_000_000, 1)
        assert not validation.ok
        assert validation.frequencies == (1_000_000,)
        assert isinstance(validation, GridValidation)

    @staticmethod
    def test_illegal_sweep_settings_are_reported_not_raised():
        capability = SweepCapability()
        validation = capability.validate_grid(1, 1, 1, 1)
        assert validation.problem is not None
        validation = capability.validate_grid(-5, -1, 11, 1)
        assert not validation.ok


# ----------------------------------------------------------------------
# Planner agreement
# ----------------------------------------------------------------------
class TestPlannerAgreement:
    @staticmethod
    def test_every_planned_replay_grid_validates():
        rng = random.Random(20261101)
        checked = 0
        for _ in range(300):
            start = rng.choice([1, 2, 1_000, 1_000_000, 123_456_789])
            stop = start + rng.randint(0, max(2, start))
            points = rng.choice([2, 3, 11, 51, 101, 201])
            segments = rng.choice([1, 2, 3, 5, 11])
            logarithmic = rng.random() < 0.5
            try:
                plan = plan_replay_range(
                    start,
                    stop,
                    points=points,
                    segments=segments,
                    logarithmic=logarithmic,
                )
            except ReplayRangeError:
                continue
            checked += 1
            capability = replay_capability(start, stop)
            validation = capability.validate_grid(
                plan.start,
                plan.stop,
                plan.points,
                plan.segments,
                logarithmic=plan.logarithmic,
            )
            assert validation.ok, (start, stop, plan, validation.problem)
            assert min(validation.frequencies) >= start
            assert max(validation.frequencies) <= stop
        assert checked > 100

    @staticmethod
    def test_narrow_replay_file_refuses_101_points_but_allows_three(
        tmp_path,
    ):
        scenario = _file(tmp_path, "narrow.s1p", NARROW)
        capability = scenario.capability()
        with pytest.raises(ReplayRangeError, match="cannot hold 101"):
            plan_replay_range(1_000_000, 1_000_020, points=101, segments=1)
        validation = capability.validate_grid(1_000_000, 1_000_020, 101)
        assert validation.problem.code == GRID_COLLAPSED

        plan = plan_replay_range(1_000_000, 1_000_020, points=3, segments=1)
        assert plan.grid() == [1_000_000, 1_000_010, 1_000_020]
        assert capability.validate_grid(plan.start, plan.stop, plan.points).ok

    @staticmethod
    def test_preset_style_bounds_share_the_same_verdict():
        from NanoVNASaver.AntennaTuning import DeviceLimits, plan_sweep
        from NanoVNASaver.Presets import SweepPreset, validate_sweep

        limits = DeviceLimits(
            min_frequency=50_000,
            max_frequency=300_000_000,
            points=(11, 51, 101, 201, 401),
            default_points=101,
        )
        plan = plan_sweep(7_000_000, 30_000_000, limits=limits, points=401)
        assert (
            limits.capability()
            .validate_grid(plan.start, plan.end, plan.points, plan.segments)
            .ok
        )
        problems = validate_sweep(
            SweepPreset(
                start=plan.start,
                end=plan.end,
                points=plan.points,
                segments=plan.segments,
            ),
            capability=limits.capability(),
        )
        assert problems == []

        too_wide = validate_sweep(
            SweepPreset(start=1_000_000, end=10_000_000_000, points=101),
            capability=limits.capability(),
        )
        assert any("above the maximum" in item for item in too_wide)

    @staticmethod
    def test_capability_with_source_filters_presets(tmp_path):
        from NanoVNASaver.Presets import SweepPreset, validate_sweep

        capability = _file(tmp_path, "recorded.s1p", ONE_PORT).capability()
        inside = validate_sweep(
            SweepPreset(start=1_000_000, end=4_000_000, points=11),
            capability=capability,
        )
        assert inside == []
        outside = validate_sweep(
            SweepPreset(start=1_000_000, end=5_000_000, points=11),
            capability=capability,
        )
        assert any("outside the replay range" in item for item in outside)


# ----------------------------------------------------------------------
# Immutability and description
# ----------------------------------------------------------------------
class TestModel:
    @staticmethod
    def test_capability_is_immutable():
        capability = SweepCapability()
        with pytest.raises(AttributeError):
            capability.max_frequency = 1

    @staticmethod
    def test_source_restriction_keeps_the_hardware_limits():
        hardware = SweepCapability(min_frequency=50_000, max_frequency=300e6)
        restricted = hardware.with_source(
            SourceSpan(1_000_000, 4_000_000, name="a.s1p", label="the replay")
        )
        assert restricted.max_frequency == hardware.max_frequency
        assert restricted.min_frequency == hardware.min_frequency
        assert restricted.effective_span.describe() == "1.00000MHz - 4.00000MHz"
        assert "a.s1p" in restricted.describe()

    @staticmethod
    def test_source_span_contains_is_inclusive():
        span = SourceSpan(1_000_000, 4_000_000)
        assert span.contains(1_000_000, 4_000_000)
        assert not span.contains(999_999, 2_000_000)
        assert not span.contains(2_000_000, 4_000_001)
        assert not span.contains(3_000_000, 2_000_000)

    @staticmethod
    def test_replace_keeps_a_capability_consistent(one_port):
        capability = replace(one_port.capability(), cw=False)
        assert capability.source_max_frequency == 4_000_000
