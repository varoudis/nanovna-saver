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
"""SWEEP-05: each pass of a continuous sweep replaces the previous one.

Written after a real defect: the buffers were allocated once per *run*, so
every pass of a continuous sweep appended another copy of the grid.  The
trace list grew without bound, contained repeated frequencies, and the chart
drew a line back and forth between passes.
"""

import threading
from dataclasses import dataclass, field

from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Settings.Sweep import Sweep
from NanoVNASaver.SweepState import SweepState, SweepStateMachine
from NanoVNASaver.SweepWorker import SweepCapture, SweepWorker


@dataclass
class Recorder:
    sample_ready: list = field(default_factory=list)
    run_finished: list = field(default_factory=list)
    state_changed: list = field(default_factory=list)
    progress: list = field(default_factory=list)
    finished: int = 0
    sweep_error: int = 0


class Emitter:
    def __init__(self, recorder, name):
        self._recorder = recorder
        self._name = name

    def emit(self, payload=None):
        getattr(self._recorder, self._name).append(payload)


class Counter:
    def __init__(self, recorder, name):
        self._recorder = recorder
        self._name = name

    def emit(self, *_args):
        setattr(self, self._name, 1)


class Signals:
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
    finished = property(lambda self: Counter(self._recorder, "finished"))
    sweep_error = property(lambda self: Counter(self._recorder, "sweep_error"))
    updated = property(lambda self: Counter(self._recorder, "updated"))


@dataclass
class FakeApp:
    vna: object
    sweep: Sweep
    calibration: Calibration


def _worker(segments=1, points=101):
    app = FakeApp(
        vna=make_simulated_vna("antenna"),
        sweep=Sweep(
            start=13_000_000,
            end=15_000_000,
            points=points,
            segments=segments,
        ),
        calibration=Calibration(),
    )
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


def _run_pass(worker, recorder):
    recorder.sample_ready.clear()
    worker.run_once()
    return recorder.sample_ready[-1]


def _continuous_worker(passes=2, segments=1, points=101):
    """Drive the continuous loop directly and stop after ``passes`` passes.

    ``run_once`` performs a single pass, so the accumulation defect only
    appears when the continuous loop inside ``_run_loop`` iterates.
    """
    worker, recorder = _worker(segments=segments, points=points)

    remaining = {"passes": passes}
    segments_run = {"count": 0}
    original_update = worker.update_data

    def update_data(frequencies, values11, values21, index):
        original_update(frequencies, values11, values21, index)
        segments_run["count"] += 1
        if segments_run["count"] % segments == 0:
            remaining["passes"] -= 1
            if remaining["passes"] <= 0:
                worker._cancel.set()

    worker.update_data = update_data
    capture = SweepCapture(
        sweep_id=1,
        sweep=worker.app.sweep,
        calibration=worker.app.calibration,
        offset_delay=0.0,
        averages=1,
        truncates=0,
        continuous=True,
    )
    worker._capture = capture
    worker._sweep_counter = 1
    worker._state_machine.reset(1)
    worker._transition(SweepState.PREPARING)
    terminal = worker._run_loop()
    return worker, recorder, terminal


class TestContinuousSweepPasses:
    @staticmethod
    def test_second_pass_does_not_accumulate():
        worker, recorder, terminal = _continuous_worker(passes=2)
        assert terminal is SweepState.CANCELLED
        last = recorder.sample_ready[-1]
        assert len(last.s11) == 101, "second pass appended a second grid"
        assert len(worker.data11) == 101

    @staticmethod
    def test_second_pass_frequencies_do_not_repeat():
        worker, recorder, _terminal = _continuous_worker(passes=2)
        last = recorder.sample_ready[-1]
        freqs = [dp.freq for dp in last.s11]
        assert freqs == sorted(set(freqs))
        assert len(freqs) == 101

    @staticmethod
    def test_multi_segment_pass_is_not_doubled():
        worker, recorder, _terminal = _continuous_worker(passes=2, segments=2)
        last = recorder.sample_ready[-1]
        assert len(last.s11) == 202
        assert len(worker.data11) == 202


class TestSinglePassUnaffected:
    @staticmethod
    def test_single_pass_still_publishes_every_point():
        worker, recorder = _worker()
        sample = _run_pass(worker, recorder)
        assert len(sample.s11) == 101
        assert len(recorder.sample_ready) == 1
