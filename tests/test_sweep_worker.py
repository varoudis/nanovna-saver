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
"""SWEEP-01..04: sweep lifecycle, progress, cancellation and publication.

The worker is exercised against the simulated device with a stub application
object and synchronous signal collectors, so these tests need no widgets and
no hardware.
"""

import threading
import time
from dataclasses import dataclass, field

import pytest

from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.Hardware.Simulated import (
    FaultConfig,
    FaultMode,
    make_simulated_vna,
)
from NanoVNASaver.Measurement import CompletionState
from NanoVNASaver.Settings.Sweep import Sweep
from NanoVNASaver.SweepState import (
    SweepProgress,
    SweepState,
    SweepStateMachine,
    estimate_remaining,
)
from NanoVNASaver.SweepWorker import SweepSample, SweepWorker


@dataclass
class Recorder:
    """Collects signal emissions instead of connecting Qt signals."""

    state_changed: list = field(default_factory=list)
    progress: list = field(default_factory=list)
    sample_ready: list = field(default_factory=list)
    run_finished: list = field(default_factory=list)
    run_request_finished: list = field(default_factory=list)
    finished: int = 0
    sweep_error: int = 0


class Emitter:
    def __init__(self, recorder, name):
        self._recorder = recorder
        self._name = name

    def emit(self, *payload):
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
    """Synchronous stand-in for ``WorkerSignals``."""

    def __init__(self, recorder):
        self._recorder = recorder

    state_changed = property(
        lambda self: Emitter(self._recorder, "state_changed")
    )
    progress = property(lambda self: Emitter(self._recorder, "progress"))
    sample_ready = property(
        lambda self: Emitter(self._recorder, "sample_ready")
    )
    run_finished = property(
        lambda self: Emitter(self._recorder, "run_finished")
    )
    run_request_finished = property(
        lambda self: Emitter(self._recorder, "run_request_finished")
    )
    finished = property(lambda self: Counter(self._recorder, "finished"))
    sweep_error = property(lambda self: Counter(self._recorder, "sweep_error"))
    updated = property(lambda self: Counter(self._recorder, "updated"))


@dataclass
class FakeApp:
    vna: object
    sweep: Sweep
    calibration: Calibration


def _app(scenario="antenna", **sweep_kwargs):
    options = {
        "start": 13_000_000,
        "end": 15_000_000,
        "points": 101,
        "segments": 1,
    }
    options.update(sweep_kwargs)
    return FakeApp(
        vna=make_simulated_vna(scenario),
        sweep=Sweep(**options),
        calibration=Calibration(),
    )


def _make_worker(app):
    """Build a worker that is not started as a QThread."""
    worker = SweepWorker.__new__(SweepWorker)
    worker.app = app
    worker.sweep = Sweep()
    worker.percentage = 0.0
    worker.data11 = []
    worker.data21 = []
    worker.rawData11 = []
    worker.rawData21 = []
    worker.error_message = ""
    worker.offsetDelay = 0.0
    worker._cancel = threading.Event()
    worker._state_machine = SweepStateMachine()
    worker._capture = None
    worker._sweep_counter = 0
    worker._started_at = 0.0
    worker._retry_count = 0
    worker._retry_reason = ""
    worker._last_terminal = None
    recorder = Recorder()
    worker.signals = Signals(recorder)
    return worker, recorder


class TestCompletedSweep:
    @staticmethod
    def test_single_segment_sweep_completes():
        worker, record = _make_worker(_app())
        worker.run_once()
        assert record.run_finished == [SweepState.COMPLETED]
        assert worker.state is SweepState.COMPLETED
        assert record.finished == 1

    @staticmethod
    def test_published_samples_are_complete_and_labelled():
        worker, record = _make_worker(_app())
        worker.run_once()
        assert record.sample_ready
        final = record.sample_ready[-1]
        assert isinstance(final, SweepSample)
        assert final.completion is CompletionState.COMPLETE
        assert len(final.s11) == 101
        assert final.snapshot.available == ("S11", "S21")

    @staticmethod
    def test_no_preallocated_zeros_are_published():
        worker, record = _make_worker(_app(segments=4))
        worker.run_once()
        for sample in record.sample_ready:
            assert all(abs(dp.z) > 0 for dp in sample.s11)
        assert len(record.sample_ready[-1].s11) == 101 * 4

    @staticmethod
    def test_intermediate_segments_are_labelled_partial():
        worker, record = _make_worker(_app(segments=3))
        worker.run_once()
        completions = [s.completion for s in record.sample_ready]
        assert completions == [
            CompletionState.PARTIAL,
            CompletionState.PARTIAL,
            CompletionState.COMPLETE,
        ]

    @staticmethod
    def test_progress_reports_elapsed_and_estimate():
        worker, record = _make_worker(_app(segments=3))
        worker.run_once()
        final = record.progress[-1]
        assert isinstance(final, SweepProgress)
        assert final.elapsed_seconds >= 0.0
        assert final.sweep_id == worker.sweep_id
        assert final.percentage == pytest.approx(100.0)

    @staticmethod
    def test_state_sequence_follows_the_declared_machine():
        worker, record = _make_worker(_app())
        worker.run_once()
        assert record.state_changed[0] is SweepState.PREPARING
        assert SweepState.MEASURING in record.state_changed
        assert record.state_changed[-1] is SweepState.COMPLETED

    @staticmethod
    def test_capture_is_frozen_at_start():
        app = _app()
        worker, record = _make_worker(app)
        worker.run_once()
        capture = worker.capture
        assert capture is not None
        assert capture.sweep.start == 13_000_000
        assert capture.sweep_id == worker.sweep_id
        # The live sweep object is a separate copy.
        assert capture.sweep is not app.sweep

    @staticmethod
    def test_sweep_range_is_restored_after_a_multi_segment_run():
        app = _app(segments=3)
        worker, record = _make_worker(app)
        worker.run_once()
        assert record.run_finished == [SweepState.COMPLETED]
        assert app.vna.device.start == 13_000_000
        assert app.vna.device.stop == 15_000_000


class TestCancellation:
    @staticmethod
    def test_pending_cancel_finishes_as_cancelled():
        worker, record = _make_worker(_app())
        worker.cancel()
        worker.run_once()
        assert record.run_finished == [SweepState.CANCELLED]
        assert record.finished == 0
        assert record.sweep_error == 0

    @staticmethod
    def test_cancel_mid_sweep_discards_the_partial_segment():
        worker, record = _make_worker(_app(segments=6))

        original = worker.read_averaged_segment

        def cancel_at_third_segment(start, stop, averages=1, *, segment=0):
            if segment == 2:
                worker.cancel()
                return None
            return original(start, stop, averages, segment=segment)

        worker.read_averaged_segment = cancel_at_third_segment
        worker.run_once()
        assert record.run_finished[-1] is SweepState.CANCELLED
        assert len(record.sample_ready) == 2
        assert all(
            sample.completion is CompletionState.PARTIAL
            for sample in record.sample_ready
        )

    @staticmethod
    def test_cancellation_is_bounded_in_time():
        worker, record = _make_worker(_app())
        worker.cancel()
        started = time.monotonic()
        worker.run_once()
        assert time.monotonic() - started < 0.25
        assert record.run_finished == [SweepState.CANCELLED]

    @staticmethod
    def test_repeated_runs_each_have_exactly_one_terminal_outcome():
        worker, record = _make_worker(_app())
        worker.run_once()
        worker.run_once()
        assert record.run_finished == [
            SweepState.COMPLETED,
            SweepState.COMPLETED,
        ]
        assert record.finished == 2

    @staticmethod
    def test_each_run_uses_a_new_sweep_id():
        worker, record = _make_worker(_app())
        worker.run_once()
        first = worker.sweep_id
        worker.run_once()
        assert worker.sweep_id == first + 1


class TestFailures:
    @staticmethod
    def test_disconnected_device_fails_before_the_first_sample():
        app = _app()
        app.vna.disconnect()
        worker, record = _make_worker(app)
        worker.run_once()
        assert record.run_finished == [SweepState.FAILED]
        assert record.sample_ready == []

    @staticmethod
    def test_transport_fault_surfaces_as_a_failure_with_a_reason():
        app = _app()
        app.vna.device.fault = FaultConfig(
            mode=FaultMode.TIMEOUT, fail_after_samples=1
        )
        worker, record = _make_worker(app)
        worker.run_once()
        assert record.run_finished == [SweepState.FAILED]
        assert worker.error_message

    @staticmethod
    def test_retry_state_is_observed_during_recovery():
        app = _app()
        app.vna.device.fault = FaultConfig(
            mode=FaultMode.EMPTY_SAMPLES, fail_after_samples=0
        )
        worker, record = _make_worker(app)
        worker.run_once()
        assert SweepState.RETRYING in record.state_changed
        # The retry budget is finite: the run ends instead of looping.
        assert record.run_finished == [SweepState.FAILED]


class TestProgressModel:
    @staticmethod
    def test_estimate_is_unavailable_before_any_unit_finishes():
        assert estimate_remaining(1.0, 0, 4) is None

    @staticmethod
    def test_estimate_extrapolates_observed_durations():
        assert estimate_remaining(2.0, 1, 4) == pytest.approx(6.0)

    @staticmethod
    def test_estimate_is_zero_when_everything_finished():
        assert estimate_remaining(2.0, 4, 4) == 0.0

    @staticmethod
    def test_continuous_sweeps_report_no_percentage():
        progress = SweepProgress(
            sweep_id=1,
            state=SweepState.MEASURING,
            segments=2,
            averages=2,
            completed_segments=1,
            completed_averages=1,
            continuous=True,
        )
        assert progress.percentage == 0.0

    @staticmethod
    def test_percentage_accounts_for_segments_and_averages():
        progress = SweepProgress(
            sweep_id=1,
            state=SweepState.MEASURING,
            segments=2,
            averages=2,
            completed_segments=1,
            completed_averages=1,
        )
        assert progress.percentage == pytest.approx(75.0)


class TestCorrectionProvenance:
    """CAL-02: the snapshot says what actually happened to the samples.

    Written after a real defect: the worker produced corrected values while
    the published snapshot still reported ``correction.applied == False`` and
    carried no identity, so a corrected measurement was indistinguishable
    from an uncorrected one.
    """

    @staticmethod
    def _calibrated_worker():
        from NanoVNASaver.Hardware.Simulated import make_simulated_vna

        worker, recorder = _make_worker(_app())
        calibration = Calibration()
        calibration.load("tests/data/sol_27_30.cal")
        calibration.calc_corrections()
        worker.app.calibration = calibration
        worker.app.sweep.update(
            start=27_000_000, end=30_000_000, segments=1, points=101
        )
        worker.app.vna = make_simulated_vna("antenna")
        return worker, recorder

    @staticmethod
    def test_a_calibrated_sweep_reports_the_correction():
        worker, recorder = TestCorrectionProvenance._calibrated_worker()
        worker.run_once()
        snapshot = recorder.sample_ready[-1].snapshot
        assert snapshot.correction.applied is True
        assert snapshot.correction.identity
        assert snapshot.correction.source.value == "application"
        assert (
            snapshot.correction.identity
            == snapshot.provenance.calibration.identity
        )

    @staticmethod
    def test_an_uncalibrated_sweep_reports_no_correction():
        worker, recorder = _make_worker(_app())
        worker.run_once()
        snapshot = recorder.sample_ready[-1].snapshot
        assert snapshot.correction.applied is False
        assert snapshot.correction.identity == ""
        assert snapshot.correction.source.value == "none"

    @staticmethod
    def test_corrected_samples_differ_from_raw_when_calibrated():
        worker, recorder = TestCorrectionProvenance._calibrated_worker()
        worker.run_once()
        snapshot = recorder.sample_ready[-1].snapshot
        raw = snapshot.channel("S11", corrected=False)
        corrected = snapshot.channel("S11")
        assert raw != corrected, "calibration did not change the samples"
        assert snapshot.correction.applied is True
