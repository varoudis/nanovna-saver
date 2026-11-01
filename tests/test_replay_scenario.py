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
"""REPLAY-01: Touchstone replay as a first-class simulated scenario.

Covers the interpolation contract (exact sample agreement, complex linear
interpolation between samples, no extrapolation), validation of unusable
files, one-port honesty (S11 only, no fabricated S21), normal worker
acquisition and GUI selection with error preservation.
"""

from dataclasses import dataclass, field

import pytest

from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.Hardware.ReplayScenario import (
    MAX_REPLAY_BYTES,
    ReplayChannelError,
    ReplayData,
    ReplayError,
    ReplayOutOfSpanError,
    ReplayReferenceImpedanceError,
    ReplayScenario,
    load_replay_file,
)
from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Measurement import CompletionState, snapshot_to_touchstone
from NanoVNASaver.Settings.ReplayRange import (
    ReplayRangeError,
    plan_replay_range,
)
from NanoVNASaver.Settings.Sweep import Properties, Sweep
from NanoVNASaver.SweepState import SweepState
from NanoVNASaver.SweepWorker import SweepWorker
from NanoVNASaver.Touchstone import SIMULATED_MARKER, Touchstone

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


def write(tmp_path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def one_port(tmp_path) -> ReplayData:
    return load_replay_file(write(tmp_path, "recorded.s1p", ONE_PORT))


@pytest.fixture
def two_port(tmp_path) -> ReplayData:
    return load_replay_file(write(tmp_path, "thru.s2p", TWO_PORT))


# ----------------------------------------------------------------------
# Interpolation contract
# ----------------------------------------------------------------------
class TestInterpolation:
    @staticmethod
    def test_exact_frequencies_return_the_file_samples(one_port):
        assert one_port.value_at("S11", 1_000_000) == complex(0.10, 0.20)
        assert one_port.value_at("S11", 2_000_000) == complex(0.30, 0.40)
        assert one_port.value_at("S11", 4_000_000) == complex(0.50, -0.60)

    @staticmethod
    def test_exact_samples_are_bit_identical(one_port):
        # 0.1/0.2 are not exactly representable; an interpolating
        # implementation that recomputes them would drift by an ulp.
        dp = one_port.s11[0]
        assert one_port.value_at("S11", dp.freq) == complex(dp.re, dp.im)

    @staticmethod
    def test_between_samples_interpolates_real_and_imag_separately(one_port):
        # Halfway between 1 MHz (0.1, 0.2) and 2 MHz (0.3, 0.4).
        assert one_port.value_at("S11", 1_500_000) == pytest.approx(
            complex(0.2, 0.3)
        )
        # A quarter of the way from 2 MHz to 4 MHz.
        value = one_port.value_at("S11", 2_500_000)
        assert value.real == pytest.approx(0.30 + 0.25 * (0.50 - 0.30))
        assert value.imag == pytest.approx(0.40 + 0.25 * (-0.60 - 0.40))

    @staticmethod
    def test_interpolation_matches_linear_weights_at_every_step(one_port):
        for freq in (1_100_000, 1_250_000, 3_000_000, 3_990_000):
            index = next(
                i for i, dp in enumerate(one_port.s11) if dp.freq >= freq
            )
            if one_port.s11[index].freq == freq:
                continue
            low, high = one_port.s11[index - 1], one_port.s11[index]
            t = (freq - low.freq) / (high.freq - low.freq)
            expected = complex(
                low.re + t * (high.re - low.re),
                low.im + t * (high.im - low.im),
            )
            assert one_port.value_at("S11", freq) == pytest.approx(expected)

    @staticmethod
    def test_below_and_above_the_span_are_rejected(one_port):
        with pytest.raises(ReplayOutOfSpanError) as low:
            one_port.value_at("S11", 999_999)
        assert "does not extrapolate" in str(low.value) or (
            "outside the replay range" in str(low.value)
        )
        with pytest.raises(ReplayOutOfSpanError):
            one_port.value_at("S11", 4_000_001)

    @staticmethod
    def test_the_span_endpoints_are_inclusive(one_port):
        assert one_port.value_at("S11", one_port.start)
        assert one_port.value_at("S11", one_port.stop)

    @staticmethod
    def test_is_exact_flags_only_file_frequencies(one_port):
        assert one_port.is_exact(2_000_000)
        assert not one_port.is_exact(1_500_000)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
class TestFileValidation:
    @staticmethod
    def test_empty_file_is_rejected(tmp_path):
        with pytest.raises(ReplayError, match="empty"):
            load_replay_file(write(tmp_path, "empty.s1p", ""))

    @staticmethod
    def test_comment_only_file_is_rejected(tmp_path):
        with pytest.raises(ReplayError):
            load_replay_file(write(tmp_path, "comments.s1p", "! nothing\n"))

    @staticmethod
    def test_unsupported_file_type_is_rejected(tmp_path):
        path = write(tmp_path, "data.txt", ONE_PORT)
        with pytest.raises(ReplayError, match="unsupported file type"):
            load_replay_file(path)

    @staticmethod
    def test_missing_file_is_reported(tmp_path):
        with pytest.raises(ReplayError, match="does not exist"):
            load_replay_file(str(tmp_path / "absent.s1p"))

    @staticmethod
    def test_oversized_file_is_rejected_before_parsing(tmp_path, monkeypatch):
        path = write(tmp_path, "huge.s1p", ONE_PORT)
        monkeypatch.setattr(
            "NanoVNASaver.Hardware.ReplayScenario.MAX_REPLAY_BYTES", 10
        )
        with pytest.raises(ReplayError, match="replay"):
            load_replay_file(path)
        assert MAX_REPLAY_BYTES > 10

    @staticmethod
    def test_duplicate_frequencies_are_rejected(tmp_path):
        text = "# HZ S RI R 50\n1000000 0.1 0.0\n1000000 0.2 0.0\n"
        with pytest.raises(ReplayError, match="duplicate|non-ascending"):
            load_replay_file(write(tmp_path, "dupes.s1p", text))

    @staticmethod
    def test_unsorted_input_is_reordered_and_still_replayable(tmp_path):
        # ``Touchstone`` has always reordered out-of-order rows; replay
        # keeps that behaviour and validates the resulting grid.
        text = "# HZ S RI R 50\n2000000 0.3 0.4\n1000000 0.1 0.2\n"
        data = load_replay_file(write(tmp_path, "unsorted.s1p", text))
        assert data.frequencies == (1_000_000, 2_000_000)
        assert data.value_at("S11", 1_000_000) == complex(0.1, 0.2)

    @staticmethod
    def test_non_positive_frequency_is_rejected(tmp_path):
        text = "# HZ S RI R 50\n0 0.1 0.0\n1000000 0.2 0.0\n"
        with pytest.raises(ReplayError, match="non-positive"):
            load_replay_file(write(tmp_path, "zero.s1p", text))

    @staticmethod
    def test_non_finite_values_are_rejected(tmp_path):
        text = "# HZ S RI R 50\n1000000 nan 0.0\n2000000 0.2 0.0\n"
        with pytest.raises(ReplayError):
            load_replay_file(write(tmp_path, "nan.s1p", text))

    @staticmethod
    def test_malformed_data_line_is_rejected(tmp_path):
        text = "# HZ S RI R 50\n1000000 0.1\n"
        with pytest.raises(ReplayError, match="not a usable Touchstone"):
            load_replay_file(write(tmp_path, "odd.s1p", text))

    @staticmethod
    def test_non_s_parameter_file_is_rejected(tmp_path):
        text = "# HZ Z RI R 50\n1000000 50.0 0.0\n"
        with pytest.raises(ReplayError, match="only S-parameters"):
            load_replay_file(write(tmp_path, "z.s1p", text))

    @staticmethod
    def test_duplicate_columns_inconsistent_length_are_rejected(tmp_path):
        text = "# HZ S RI R 50\n1000000 0.1 0.0 0.2 0.0\n2000000 0.1 0.0\n"
        with pytest.raises(ReplayError):
            load_replay_file(write(tmp_path, "ragged.s2p", text))

    @staticmethod
    def test_reference_impedance_mismatch_is_actionable(tmp_path):
        text = "# HZ S RI R 75\n1000000 0.1 0.0\n"
        with pytest.raises(ReplayReferenceImpedanceError) as error:
            load_replay_file(write(tmp_path, "seventyfive.s1p", text))
        message = str(error.value)
        assert "75 ohm" in message and "50 ohm" in message
        assert "renormalize" in message

    @staticmethod
    def test_matching_non_integer_impedance_is_accepted(tmp_path):
        text = "# HZ S RI R 50.0\n1000000 0.1 0.0\n"
        data = load_replay_file(write(tmp_path, "fifty.s1p", text))
        assert data.reference_impedance == pytest.approx(50.0)


# ----------------------------------------------------------------------
# One-port / two-port honesty
# ----------------------------------------------------------------------
class TestChannelHonesty:
    @staticmethod
    def test_one_port_file_has_no_s21(one_port):
        assert one_port.has_s21 is False
        assert one_port.s21 == ()
        assert one_port.kind == "S1P"

    @staticmethod
    def test_one_port_requests_for_s21_fail(one_port):
        with pytest.raises(ReplayChannelError):
            one_port.value_at("S21", 2_000_000)

    @staticmethod
    def test_two_port_file_keeps_s11_and_s21(two_port):
        assert two_port.has_s21 is True
        assert two_port.value_at("S21", 1_000_000) == complex(0.90, 0.10)

    @staticmethod
    def test_two_port_reports_ignored_channels(two_port):
        assert two_port.ignored_parameters == ("S12", "S22")
        assert "not replayed" in two_port.provenance_text()

    @staticmethod
    def test_one_port_provenance_says_s11_only(one_port):
        text = one_port.provenance_text()
        assert "S11 only" in text
        assert "no S21" in text.replace("(no S21)", "no S21")

    @staticmethod
    def test_scenario_supports_only_present_channels(one_port, two_port):
        assert ReplayScenario(one_port).supported_channels == ("data 0",)
        assert ReplayScenario(two_port).supported_channels == (
            "data 0",
            "data 1",
        )

    @staticmethod
    def test_device_refuses_to_synthesize_s21(one_port):
        vna = make_simulated_vna(ReplayScenario(one_port))
        vna.setSweep(1_000_000, 4_000_000)
        assert vna.readValues("data 0")
        with pytest.raises(ReplayChannelError):
            vna.readValues("data 1")


# ----------------------------------------------------------------------
# Scenario preflight
# ----------------------------------------------------------------------
class TestPreflight:
    @staticmethod
    def test_span_inside_the_file_is_accepted(one_port):
        assert ReplayScenario(one_port).preflight(2_000_000, 3_000_000) is None

    @staticmethod
    def test_span_beyond_the_file_is_rejected(one_port):
        problem = ReplayScenario(one_port).preflight(1_000_000, 5_000_000)
        assert problem is not None
        assert "outside the replay range" in problem
        assert "never extrapolates" in problem

    @staticmethod
    def test_inverted_span_is_rejected(one_port):
        problem = ReplayScenario(one_port).preflight(3_000_000, 2_000_000)
        assert problem is not None
        assert "must not be below" in problem


# ----------------------------------------------------------------------
# Default replay range planning
# ----------------------------------------------------------------------
class TestDefaultRangePlan:
    """The default sweep must be legal for the grid ``Sweep`` produces."""

    @staticmethod
    def _grid(plan):
        return list(
            Sweep(
                plan.start,
                plan.stop,
                plan.points,
                plan.segments,
                Properties(logarithmic=plan.logarithmic),
            ).get_frequencies()
        )

    @staticmethod
    def _segments(plan):
        """Per-segment frequency lists, as the worker reads them.

        Adjacent segments may share a boundary frequency (that is the sweep
        model's own rounding); a single segment must never repeat one.
        """
        grid = TestDefaultRangePlan._grid(plan)
        if plan.cw:
            return [grid]
        return [
            grid[index : index + plan.points]
            for index in range(0, len(grid), plan.points)
        ]

    @staticmethod
    def test_sweep_rounding_overshoot_is_removed_not_ignored():
        # Reviewer cases: the old clamp returned stops whose grid ended at
        # 202 and 1_000_201 Hz, i.e. outside the file's last sample.
        for start, stop in ((1, 201), (1_000_000, 1_000_200)):
            plan = plan_replay_range(start, stop, points=101, segments=2)
            grid = TestDefaultRangePlan._grid(plan)
            assert max(grid) <= stop
            assert min(grid) >= start
            assert len(grid) == 101 * plan.segments
            for segment in TestDefaultRangePlan._segments(plan):
                assert len(set(segment)) == len(segment)
            # Two full segments do not fit, so one segment is used and said.
            assert plan.segments == 1
            assert plan.reduced
            assert "fits 1 segment(s)" in plan.describe()

    @staticmethod
    def test_narrow_positive_span_is_rejected_not_collapsed():
        # 50 Hz cannot hold 101 distinct whole-hertz frequencies.
        with pytest.raises(ReplayRangeError) as error:
            plan_replay_range(1_000_000, 1_000_050, points=101, segments=1)
        message = str(error.value)
        assert "cannot hold 101 distinct frequencies" in message
        assert "100 Hz" in message
        assert "fewer sweep points" in message

    @staticmethod
    def test_a_single_point_on_a_positive_span_is_rejected():
        with pytest.raises(ReplayRangeError, match="single point"):
            plan_replay_range(1_000_000, 1_000_100, points=1, segments=1)

    @staticmethod
    def test_too_many_segments_reduce_and_say_so():
        plan = plan_replay_range(1_000_000, 1_000_200, points=101, segments=5)
        assert plan.segments == 1
        assert plan.reduced
        assert "not the requested 5" in plan.describe()
        assert max(TestDefaultRangePlan._grid(plan)) <= 1_000_200

    @staticmethod
    def test_a_true_cw_file_keeps_the_requested_shape():
        plan = plan_replay_range(5_000_000, 5_000_000, points=101, segments=3)
        assert plan.cw is True
        assert plan.grid() == [5_000_000] * 303
        assert "start == stop" in plan.describe()

    @staticmethod
    @pytest.mark.parametrize(
        ("start", "stop", "points", "segments", "logarithmic"),
        [
            (1_000_000, 4_000_000, 101, 1, False),
            (1_000_000, 4_000_000, 101, 3, False),
            (1_000_000, 4_000_000, 51, 5, False),
            (13_000_000, 13_000_011, 11, 1, False),
            (1_000_000, 4_000_000, 101, 1, True),
            (1_000_000, 30_000_000, 51, 7, True),
            (100_000, 100_000_000, 201, 11, True),
        ],
    )
    def test_ordinary_spans_keep_their_grid_and_stay_inside(
        start, stop, points, segments, logarithmic
    ):
        plan = plan_replay_range(
            start,
            stop,
            points=points,
            segments=segments,
            logarithmic=logarithmic,
        )
        grid = TestDefaultRangePlan._grid(plan)
        assert plan.points == points
        assert plan.logarithmic is logarithmic
        assert len(grid) == points * segments
        for segment in TestDefaultRangePlan._segments(plan):
            assert len(set(segment)) == len(segment)
        assert min(grid) >= start
        assert max(grid) <= stop
        assert plan.describe() == ""

    @staticmethod
    def test_the_one_port_replay_file_default_is_inside_the_file(
        one_port,
    ):
        plan = plan_replay_range(
            one_port.start, one_port.stop, points=101, segments=1
        )
        assert plan.stop <= one_port.stop
        assert max(plan.grid()) <= one_port.stop

    @staticmethod
    def test_random_spans_never_produce_an_illegal_grid():
        import random

        rng = random.Random(20261101)
        for _ in range(200):
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
            grid = TestDefaultRangePlan._grid(plan)
            assert min(grid) >= start and max(grid) <= stop
            assert plan.points == points
            if not plan.cw:
                assert len(grid) == points * plan.segments
                for segment in TestDefaultRangePlan._segments(plan):
                    assert len(set(segment)) == len(segment)


# ----------------------------------------------------------------------
# Worker acquisition through the normal pipeline
# ----------------------------------------------------------------------
@dataclass
class Recorder:
    sample_ready: list = field(default_factory=list)
    run_finished: list = field(default_factory=list)
    run_request_finished: list = field(default_factory=list)
    state_changed: list = field(default_factory=list)
    progress: list = field(default_factory=list)
    finished: int = 0
    sweep_error: int = 0
    updated: int = 0


class Emitter:
    def __init__(self, recorder, name):
        self._recorder = recorder
        self._name = name

    def emit(self, *payload):
        # ``run_request_finished`` carries two arguments; record every
        # payload shape without discarding the correlation token.
        getattr(self._recorder, self._name).append(
            payload[0] if len(payload) == 1 else payload
        )


class Counter:
    def __init__(self, recorder, name):
        self._recorder = recorder
        self._name = name

    def emit(self, *_args):
        setattr(
            self._recorder,
            self._name,
            getattr(self._recorder, self._name) + 1,
        )


class Signals:
    def __init__(self, recorder):
        self._recorder = recorder

    state_changed = property(lambda s: Emitter(s._recorder, "state_changed"))
    progress = property(lambda s: Emitter(s._recorder, "progress"))
    sample_ready = property(lambda s: Emitter(s._recorder, "sample_ready"))
    run_finished = property(lambda s: Emitter(s._recorder, "run_finished"))
    run_request_finished = property(
        lambda s: Emitter(s._recorder, "run_request_finished")
    )
    finished = property(lambda s: Counter(s._recorder, "finished"))
    sweep_error = property(lambda s: Counter(s._recorder, "sweep_error"))
    updated = property(lambda s: Counter(s._recorder, "updated"))


@dataclass
class FakeApp:
    vna: object
    sweep: Sweep
    calibration: Calibration


def _worker(source: ReplayData, start: int, stop: int, points=11, segments=1):
    vna = make_simulated_vna(ReplayScenario(source))
    # The device's point count is what drives the acquired grid.
    vna.datapoints = points
    app = FakeApp(
        vna=vna,
        sweep=Sweep(start=start, end=stop, points=points, segments=segments),
        calibration=Calibration(),
    )
    worker = SweepWorker(app)
    recorder = Recorder()
    worker.signals = Signals(recorder)
    return worker, recorder


class TestWorkerAcquisition:
    @staticmethod
    def test_replay_sweep_completes_with_the_expected_samples(one_port):
        worker, recorder = _worker(one_port, 1_000_000, 4_000_000, points=4)
        worker.run_once()

        assert worker.last_terminal_state is SweepState.COMPLETED
        sample = recorder.sample_ready[-1]
        freqs = [dp.freq for dp in sample.s11]
        assert freqs == [1_000_000, 2_000_000, 3_000_000, 4_000_000]
        values = {dp.freq: dp.z for dp in sample.s11}
        assert values[1_000_000] == complex(0.10, 0.20)
        assert values[2_000_000] == complex(0.30, 0.40)
        assert values[4_000_000] == complex(0.50, -0.60)
        assert values[3_000_000] == pytest.approx(complex(0.40, -0.10))

    @staticmethod
    def test_replay_publishes_no_s21_for_a_one_port_file(one_port):
        worker, recorder = _worker(one_port, 1_000_000, 4_000_000)
        worker.run_once()

        sample = recorder.sample_ready[-1]
        assert sample.s21 == ()
        assert worker.data21 == []
        assert not sample.snapshot.has("S21")
        assert sample.snapshot.has("S11")

    @staticmethod
    def test_replay_snapshot_carries_full_provenance(one_port):
        worker, recorder = _worker(one_port, 1_000_000, 4_000_000)
        worker.run_once()

        provenance = recorder.sample_ready[-1].snapshot.provenance
        assert provenance.simulated is True
        assert provenance.simulator_scenario == "replay"
        assert provenance.source == "simulated"
        assert "recorded.s1p" in provenance.notes
        assert provenance.completion is CompletionState.COMPLETE

    @staticmethod
    def test_two_port_replay_samples_both_channels(two_port):
        worker, recorder = _worker(two_port, 1_000_000, 2_000_000, points=2)
        worker.run_once()

        sample = recorder.sample_ready[-1]
        s21 = {dp.freq: dp.z for dp in sample.s21}
        assert s21[1_000_000] == complex(0.90, 0.10)
        assert s21[2_000_000] == complex(0.70, -0.30)
        assert "S12, S22" in recorder.sample_ready[-1].snapshot.provenance.notes

    @staticmethod
    def test_out_of_span_sweep_fails_before_reading_any_sample(one_port):
        worker, recorder = _worker(one_port, 1_000_000, 5_000_000)
        worker.run_once()

        assert worker.last_terminal_state is SweepState.FAILED
        assert recorder.sample_ready == []
        assert worker.data11 == [] and worker.data21 == []
        assert "outside the replay range" in worker.error_message
        # The user sees the actionable reason, not a silent failure.
        assert recorder.sweep_error == 1

    @staticmethod
    def test_segmented_replay_covers_the_file_span(one_port):
        worker, recorder = _worker(
            one_port, 1_000_000, 4_000_000, points=3, segments=2
        )
        worker.run_once()

        assert worker.last_terminal_state is SweepState.COMPLETED
        freqs = [dp.freq for dp in recorder.sample_ready[-1].s11]
        assert freqs == sorted(set(freqs))
        assert min(freqs) >= one_port.start
        assert max(freqs) <= one_port.stop

    @staticmethod
    def test_the_default_segmented_span_needs_no_extrapolation(one_port):
        from NanoVNASaver.Settings.ReplayRange import plan_replay_range

        plan = plan_replay_range(
            one_port.start, one_port.stop, points=101, segments=3
        )
        assert plan.stop <= one_port.stop

        worker, recorder = _worker(
            one_port, plan.start, plan.stop, points=101, segments=plan.segments
        )
        worker.run_once()

        assert worker.error_message == ""
        assert worker.last_terminal_state is SweepState.COMPLETED
        freqs = [dp.freq for dp in recorder.sample_ready[-1].s11]
        assert max(freqs) <= one_port.stop

    @staticmethod
    def test_replay_is_deterministic_across_runs(one_port):
        first, recorder_first = _worker(one_port, 1_000_000, 4_000_000)
        first.run_once()
        second, recorder_second = _worker(one_port, 1_000_000, 4_000_000)
        second.run_once()
        assert recorder_first.sample_ready[-1].s11 == (
            recorder_second.sample_ready[-1].s11
        )

    @staticmethod
    def test_analytic_scenarios_still_publish_s21():
        app = FakeApp(
            vna=make_simulated_vna("filter"),
            sweep=Sweep(start=10_000_000, end=20_000_000, points=11),
            calibration=Calibration(),
        )
        worker = SweepWorker(app)
        recorder = Recorder()
        worker.signals = Signals(recorder)
        worker.run_once()

        sample = recorder.sample_ready[-1]
        assert sample.s21
        assert sample.snapshot.provenance.simulator_scenario == "filter"
        assert sample.snapshot.provenance.notes == ""


def _declared_app(vna, start=13_000_000, stop=15_000_000):
    return FakeApp(
        vna=vna,
        sweep=Sweep(start=start, end=stop, points=101, segments=1),
        calibration=Calibration(),
    )


def _run_worker(app):
    worker = SweepWorker(app)
    recorder = Recorder()
    worker.signals = Signals(recorder)
    worker.run_once()
    return worker, recorder


class TestDeclaredChannels:
    """A device that declares S21 must never be published as one-port."""

    @staticmethod
    def test_empty_declared_s21_fails_and_publishes_nothing():
        vna = make_simulated_vna("antenna")
        original = vna.readValues

        def read(channel):
            if channel == "data 1":
                return []
            return original(channel)

        vna.readValues = read
        worker, recorder = _run_worker(_declared_app(vna))

        assert ("data 0", "data 1") == vna.supported_channels
        assert worker.last_terminal_state is SweepState.FAILED
        assert recorder.sample_ready == []
        assert worker.data11 == [] and worker.data21 == []
        assert recorder.sweep_error == 1
        assert "S21" in worker.error_message
        assert "1-channel" not in worker.error_message

    @staticmethod
    def test_short_declared_s21_fails_and_publishes_nothing():
        vna = make_simulated_vna("antenna")
        original = vna.readValues

        def read(channel):
            values = original(channel)
            return values[:-1] if channel == "data 1" else values

        vna.readValues = read
        worker, recorder = _run_worker(_declared_app(vna))

        assert worker.last_terminal_state is SweepState.FAILED
        assert recorder.sample_ready == []
        assert "100 samples for 101 requested frequencies" in (
            worker.error_message
        )

    @staticmethod
    def test_a_retry_that_keeps_failing_never_publishes_partial_data():
        vna = make_simulated_vna("antenna")
        original = vna.readValues
        attempts = []

        def read(channel):
            if channel == "data 1":
                attempts.append(channel)
                return []
            return original(channel)

        vna.readValues = read
        worker, recorder = _run_worker(_declared_app(vna))

        # Retried (bounded), still zero published samples.
        assert len(attempts) > 1
        assert recorder.sample_ready == []
        assert worker.last_terminal_state is SweepState.FAILED

    @staticmethod
    def test_a_genuine_one_port_source_is_never_asked_for_s21(one_port):
        vna = make_simulated_vna(ReplayScenario(one_port))
        calls = []
        original = vna.readValues

        def read(channel):
            calls.append(channel)
            return original(channel)

        vna.readValues = read
        worker, recorder = _run_worker(
            _declared_app(vna, one_port.start, one_port.stop)
        )

        assert worker.last_terminal_state is SweepState.COMPLETED
        assert set(calls) == {"data 0"}
        sample = recorder.sample_ready[-1]
        assert sample.s21 == ()
        assert not sample.snapshot.has("S21")
        assert sample.snapshot.has("S11")

    @staticmethod
    def test_declared_channels_default_to_s11_and_s21_without_a_declaration():
        class Undeclared:
            name = "bare"
            validateInput = False  # noqa: N815 - device API name
            datapoints = 11

            def __init__(self):
                self.values = [complex(0.1, 0.2)] * 11

            def connected(self):
                return True

            def setSweep(self, start, stop):
                return None

            def read_frequencies(self):
                return list(range(1_000_000, 1_000_000 + 11_000, 1000))

            def readValues(self, channel):
                return list(self.values)

            def reconnect(self):
                return None

        worker, recorder = _run_worker(
            FakeApp(
                vna=Undeclared(),
                sweep=Sweep(1_000_000, 1_010_000, 11, 1),
                calibration=Calibration(),
            )
        )

        # Undeclared means "both", exactly as for any real VNA.
        assert worker.last_terminal_state is SweepState.COMPLETED
        assert recorder.sample_ready[-1].s21


class TestPlannedGridAcquisition:
    """The planned default is what the worker actually requests."""

    @staticmethod
    def _dense_file(tmp_path):
        text = "# HZ S RI R 50\n" + "".join(
            f"{freq} 0.1 0.2\n" for freq in range(1_000_000, 4_000_001, 1_000)
        )
        path = tmp_path / "dense.s1p"
        path.write_text(text, encoding="utf-8")
        return load_replay_file(str(path))

    @staticmethod
    @pytest.mark.parametrize("logarithmic", [False, True])
    def test_multi_segment_defaults_stay_inside_the_file(tmp_path, logarithmic):
        source = TestPlannedGridAcquisition._dense_file(tmp_path)
        plan = plan_replay_range(
            source.start,
            source.stop,
            points=11,
            segments=3,
            logarithmic=logarithmic,
        )
        vna = make_simulated_vna(ReplayScenario(source))
        vna.datapoints = 11
        worker = SweepWorker(
            FakeApp(
                vna=vna,
                sweep=Sweep(
                    plan.start,
                    plan.stop,
                    11,
                    3,
                    Properties(logarithmic=logarithmic),
                ),
                calibration=Calibration(),
            )
        )
        recorder = Recorder()
        worker.signals = Signals(recorder)
        worker.run_once()

        assert worker.last_terminal_state is SweepState.COMPLETED
        freqs = [dp.freq for dp in recorder.sample_ready[-1].s11]
        assert len(freqs) == 33
        assert min(freqs) >= source.start
        assert max(freqs) <= source.stop

    @staticmethod
    def test_a_cw_file_replays_through_the_worker(tmp_path):
        path = tmp_path / "cw.s1p"
        path.write_text("# HZ S RI R 50\n5000000 0.1 0.2\n", encoding="utf-8")
        source = load_replay_file(str(path))
        plan = plan_replay_range(
            source.start, source.stop, points=11, segments=1
        )

        vna = make_simulated_vna(ReplayScenario(source))
        vna.datapoints = 11
        worker = SweepWorker(
            FakeApp(
                vna=vna,
                sweep=Sweep(plan.start, plan.stop, 11, 1),
                calibration=Calibration(),
            )
        )
        recorder = Recorder()
        worker.signals = Signals(recorder)
        worker.run_once()

        assert worker.last_terminal_state is SweepState.COMPLETED
        freqs = [dp.freq for dp in recorder.sample_ready[-1].s11]
        assert freqs == [5_000_000] * 11


class TestExportProvenance:
    @staticmethod
    def test_replayed_snapshot_exports_simulated_marker_and_source(one_port):
        worker, recorder = _worker(one_port, 1_000_000, 4_000_000, points=4)
        worker.run_once()

        snapshot = recorder.sample_ready[-1].snapshot
        exported = snapshot_to_touchstone(snapshot)
        exported.simulated = snapshot.simulated
        exported.simulated_scenario = (
            snapshot.provenance.simulator_scenario or ""
        )
        exported.comments = [
            f"! {line}" for line in snapshot.provenance.notes.splitlines()
        ]
        text = exported.saves(1)

        assert SIMULATED_MARKER in text
        assert "scenario replay" in text
        assert "recorded.s1p" in text
        # A one-port replay never exports a two-port file.
        reloaded = Touchstone()
        reloaded.loads(text)
        assert reloaded.s11
        assert not reloaded.s21
        assert reloaded.simulated is True

    @staticmethod
    def test_two_port_replay_export_still_marks_placeholders(two_port):
        worker, recorder = _worker(two_port, 1_000_000, 2_000_000, points=2)
        worker.run_once()

        snapshot = recorder.sample_ready[-1].snapshot
        exported = snapshot_to_touchstone(snapshot)
        text = exported.saves(4)
        assert "zero placeholders, NOT measured: S12, S22" in text


# ----------------------------------------------------------------------
# Provenance of a reloaded replay file
# ----------------------------------------------------------------------
class TestReloadedMarker:
    @staticmethod
    def test_replay_export_survives_a_round_trip(one_port):
        dataset = Touchstone()
        dataset.opts.resistance = one_port.reference_impedance
        dataset.s11 = list(one_port.s11)
        dataset.simulated = True
        dataset.simulated_scenario = "replay"
        dataset.comments = ["! replay of recorded.s1p"]

        reloaded = Touchstone()
        reloaded.loads(dataset.saves(1))
        assert reloaded.simulated is True
        assert reloaded.simulated_scenario == "replay"
        assert "recorded.s1p" in "\n".join(reloaded.comments)

    @staticmethod
    def test_session_round_trip_keeps_replay_provenance(one_port, tmp_path):
        from NanoVNASaver.Session import (
            document_snapshots,
            load_session,
            make_document,
            save_session,
        )

        worker, recorder = _worker(one_port, 1_000_000, 4_000_000, points=4)
        worker.run_once()
        snapshot = recorder.sample_ready[-1].snapshot

        path = tmp_path / "replay.nvsession"
        save_session(make_document([snapshot]), str(path))
        restored = document_snapshots(load_session(str(path)))["trace-0"]

        assert restored.simulated is True
        assert restored.provenance.simulator_scenario == "replay"
        assert "recorded.s1p" in restored.provenance.notes
        assert restored.has("S11")
        assert not restored.has("S21")

    @staticmethod
    def test_report_names_the_replay_source(one_port, tmp_path):
        from NanoVNASaver.Reports import build_report, render_html
        from NanoVNASaver.Traces import TraceCollection

        worker, recorder = _worker(one_port, 1_000_000, 4_000_000, points=4)
        worker.run_once()
        snapshot = recorder.sample_ready[-1].snapshot

        collection = TraceCollection()
        trace = collection.add(snapshot, name="Replayed")
        model = build_report("Replay", [trace])
        document = render_html(model)

        assert "SIMULATED (replay)" in document
        assert "recorded.s1p" in document
        assert "replay of" in document


# ----------------------------------------------------------------------
# GUI selection and error preservation
# ----------------------------------------------------------------------
serial_control = pytest.importorskip("NanoVNASaver.Controls.SerialControl")


class FakeSweepControl:
    def __init__(self):
        self._segments = 1
        self._start = 0
        self._end = 0

    def set_start(self, value):
        self._start = value

    def set_end(self, value):
        self._end = value

    def get_segments(self):
        return self._segments

    def set_segments(self, value):
        self._segments = value

    def update_center_span(self):
        return None

    def update_step_size(self):
        return None


class FakeWindow:
    def vna_connected(self):
        return None


class FakeGuiApp:
    def __init__(self):
        from NanoVNASaver.Settings.Sweep import Sweep

        self.vna = make_simulated_vna("antenna")
        self.sweep_control = FakeSweepControl()
        self.windows = {"sweep_settings": FakeWindow()}
        self.sweep = Sweep()
        self.sweep_starts = 0
        self.errors = []

    def sweep_start(self):
        self.sweep_starts += 1

    def showError(self, text):
        self.errors.append(text)

    def display_window(self, _name):
        return None


@pytest.fixture
def control(qt_app, monkeypatch):
    monkeypatch.setattr(serial_control, "get_interfaces", lambda: [])
    app = FakeGuiApp()
    widget = serial_control.SerialControl(app)
    yielded = (widget, app)
    try:
        yield yielded
    finally:
        widget.deleteLater()


class TestGuiSelection:
    @staticmethod
    def test_replay_is_offered_as_a_scenario(control):
        widget, _app = control
        names = [
            widget.cb_scenario.itemData(index)
            for index in range(widget.cb_scenario.count())
        ]
        assert serial_control.REPLAY_SCENARIO in names
        assert len(names) == len(set(names))

    @staticmethod
    def test_choosing_a_valid_file_loads_and_selects_replay(
        control, one_port, tmp_path
    ):
        widget, _app = control
        assert widget.choose_replay_file(one_port.path) is True
        assert widget.replay_data() is not None
        assert widget.selected_scenario() == serial_control.REPLAY_SCENARIO
        assert "recorded.s1p" in widget.label_replay_source.text()

    @staticmethod
    def test_an_invalid_file_keeps_the_previous_source(
        control, one_port, tmp_path
    ):
        widget, app = control
        assert widget.choose_replay_file(one_port.path) is True
        previous = widget.replay_data()

        bad = tmp_path / "bad.s1p"
        bad.write_text("# HZ S RI R 75\n1000000 0.1 0.0\n", encoding="utf-8")
        assert widget.choose_replay_file(str(bad)) is False
        assert widget.replay_data() is previous
        assert len(app.errors) == 1
        assert "75 ohm" in app.errors[0]

    @staticmethod
    def test_a_malformed_file_reports_and_preserves(
        control, one_port, tmp_path
    ):
        widget, app = control
        widget.choose_replay_file(one_port.path)
        previous = widget.replay_data()

        broken = tmp_path / "broken.s1p"
        broken.write_text("this is not touchstone\n", encoding="utf-8")
        assert widget.choose_replay_file(str(broken)) is False
        assert widget.replay_data() is previous
        assert app.errors

    @staticmethod
    def test_connecting_replay_uses_the_normal_workflow(control, one_port):
        widget, app = control
        widget.cb_simulated.setChecked(True)
        widget.choose_replay_file(one_port.path)
        widget.connect_device()

        assert app.vna.scenario_key == "replay"
        assert app.vna.simulated is True
        assert app.vna.supported_channels == ("data 0",)
        assert app.sweep_starts == 1

    @staticmethod
    def test_connecting_replay_without_a_file_reports_and_keeps_the_device(
        control,
    ):
        widget, app = control
        widget.cb_simulated.setChecked(True)
        widget.cb_scenario.setCurrentIndex(
            widget.cb_scenario.findData(serial_control.REPLAY_SCENARIO)
        )
        previous = app.vna
        widget.connect_device()

        assert app.errors
        assert "No replay file is loaded" in app.errors[0]
        assert app.vna is previous
        assert app.sweep_starts == 0

    @staticmethod
    def test_a_connected_replay_refuses_another_file(control, one_port):
        widget, app = control
        widget.cb_simulated.setChecked(True)
        widget.choose_replay_file(one_port.path)
        widget.connect_device()
        previous = widget.replay_data()

        assert widget.choose_replay_file(one_port.path) is False
        assert widget.replay_data() is previous
        assert "Disconnect the replay device" in app.errors[-1]

    @staticmethod
    def test_disconnecting_allows_a_new_replay_file(control, one_port):
        widget, app = control
        widget.cb_simulated.setChecked(True)
        widget.choose_replay_file(one_port.path)
        widget.connect_device()
        widget.disconnect_device()

        assert widget.choose_replay_file(one_port.path) is True
        assert app.errors == []

    @staticmethod
    def test_replay_connection_defaults_the_sweep_into_the_file(
        control, one_port
    ):
        widget, app = control
        widget.cb_simulated.setChecked(True)
        widget.choose_replay_file(one_port.path)
        widget.connect_device()

        assert app.sweep_control._start == one_port.start
        assert one_port.start <= app.sweep_control._end <= one_port.stop

    @staticmethod
    def test_segmented_sweep_still_defaults_inside_the_file(control, one_port):
        widget, app = control
        app.sweep.update(
            start=one_port.start, end=one_port.stop, segments=3, points=101
        )
        app.sweep_control._segments = 3
        widget.cb_simulated.setChecked(True)
        widget.choose_replay_file(one_port.path)
        widget.connect_device()

        stop = app.sweep_control._end
        assert one_port.start <= stop < one_port.stop
        freqs = Sweep(
            start=one_port.start, end=stop, points=101, segments=3
        ).get_frequencies()
        assert max(freqs) <= one_port.stop

    @staticmethod
    def test_switching_replay_source_during_a_sweep_is_refused(
        control, one_port, monkeypatch
    ):
        widget, app = control
        widget.choose_replay_file(one_port.path)
        previous = widget.replay_data()
        monkeypatch.setattr(widget, "sweep_active", lambda: True)

        assert widget.choose_replay_file(one_port.path) is False
        assert widget.replay_data() is previous
        assert "A sweep is running" in app.errors[0]

    @staticmethod
    def test_connecting_another_device_during_a_sweep_is_refused(
        control, one_port, monkeypatch
    ):
        widget, app = control
        widget.cb_simulated.setChecked(True)
        monkeypatch.setattr(widget, "sweep_active", lambda: True)
        widget.connect_device()

        assert app.sweep_starts == 0
        assert "A sweep is running" in app.errors[0]

    @staticmethod
    def test_the_replay_picker_is_only_enabled_for_the_replay_scenario(control):
        widget, _app = control
        widget.cb_simulated.setChecked(True)
        assert widget.btn_replay_file.isEnabled() is False
        widget.cb_scenario.setCurrentIndex(
            widget.cb_scenario.findData(serial_control.REPLAY_SCENARIO)
        )
        assert widget.btn_replay_file.isEnabled() is True

    @staticmethod
    def test_a_narrow_file_that_cannot_fit_the_points_is_refused(
        control, tmp_path
    ):
        narrow = tmp_path / "narrow.s1p"
        narrow.write_text(
            "# HZ S RI R 50\n1000000 0.1 0.0\n1000010 0.2 0.0\n",
            encoding="utf-8",
        )
        widget, app = control
        previous_vna = app.vna
        assert widget.choose_replay_file(str(narrow)) is True
        widget.cb_simulated.setChecked(True)
        widget.connect_device()

        # The narrow file cannot hold 101 distinct frequencies: connection
        # is refused transactionally, with an actionable message.
        assert app.errors
        assert "cannot hold 101 distinct frequencies" in app.errors[-1]
        assert "wider span or fewer sweep points" in app.errors[-1]
        assert app.vna is previous_vna
        assert app.sweep_starts == 0
        assert widget.replay_data() is not None

    @staticmethod
    def test_reducing_the_segment_count_is_reported_to_the_user(
        control, tmp_path
    ):
        narrow = tmp_path / "two_segment.s1p"
        narrow.write_text(
            "# HZ S RI R 50\n1000000 0.1 0.0\n1000200 0.2 0.0\n",
            encoding="utf-8",
        )
        widget, app = control
        app.sweep.update(start=1_000_000, end=1_000_200, segments=5, points=101)
        app.sweep_control._segments = 5
        assert widget.choose_replay_file(str(narrow)) is True
        widget.cb_simulated.setChecked(True)
        widget.connect_device()

        assert app.errors == []
        assert app.sweep_control._segments == 1
        assert "Default sweep: the file span fits 1 segment(s)" in (
            widget.label_replay_source.text()
        )
        grid = list(
            Sweep(
                app.sweep_control._start,
                app.sweep_control._end,
                101,
                1,
            ).get_frequencies()
        )
        assert max(grid) <= 1_000_200


class TestRealWindowReplay:
    """The real application window adopts replay provenance on screen."""

    @staticmethod
    def test_applied_sample_marks_the_display_simulated_and_named(
        qt_app, one_port
    ):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        window = NanoVNASaver()
        try:
            worker, recorder = _worker(one_port, 1_000_000, 4_000_000, points=4)
            worker.run_once()
            sample = recorder.sample_ready[-1]

            window.worker = worker
            window.applySample(sample)

            assert window.data.simulated is True
            assert window.data.simulated_scenario == "replay"
            assert not window.data.s21
            comments = "\n".join(window.data.comments)
            assert "SIMULATED DATA (scenario replay)" in comments
            assert "recorded.s1p" in comments
            assert "SIMULATED" in window.windowTitle()
            # Exactly one generated marker, even after a second publication.
            window.applySample(sample)
            markers = [
                c for c in window.data.comments if c.startswith("SIMULATED")
            ]
            assert len(markers) == 1
        finally:
            window.worker.quit()
            window.deleteLater()

    @staticmethod
    def test_current_snapshot_of_displayed_replay_data_is_marked(
        qt_app, one_port
    ):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        window = NanoVNASaver()
        try:
            worker, recorder = _worker(one_port, 1_000_000, 4_000_000, points=4)
            worker.run_once()
            window.worker = worker
            window.applySample(recorder.sample_ready[-1])

            snapshot = window.current_snapshot()
            assert snapshot is not None
            assert snapshot.simulated is True
            assert snapshot.provenance.simulator_scenario == "replay"
            assert "recorded.s1p" in snapshot.provenance.notes
            assert not snapshot.has("S21")
        finally:
            window.worker.quit()
            window.deleteLater()
