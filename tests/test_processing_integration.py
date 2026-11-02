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
"""DSP-02/DSP-03 at the acquisition boundary.

`Processing.py` owns the transforms.  This module pins the *integration*:
the worker must apply exactly the configuration frozen at the start of a run,
must keep the raw acquisition beside the processed view, and must never
publish a half-processed segment.
"""

import threading
from dataclasses import dataclass

import pytest

from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Measurement import S11
from NanoVNASaver.Processing import (
    STAGE_SMOOTHING,
    ProcessingConfig,
    ProcessingError,
    ProcessingStage,
    RunningAverageState,
    apply_processing,
    average_stage,
    smoothing_aperture_for,
    smoothing_stage,
)
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Settings.Sweep import Sweep
from NanoVNASaver.SweepState import SweepState, SweepStateMachine
from NanoVNASaver.SweepWorker import SweepCapture, SweepWorker, _is_identity


@dataclass
class FakeApp:
    vna: object
    sweep: Sweep
    calibration: Calibration
    processing_config: object = None
    running_average: object = None
    reference_impedance: float = 50.0


class Recorder:
    def __init__(self):
        self.sample_ready = []
        self.state_changed = []
        self.run_finished = []
        self.run_request_finished = []
        self.progress = []
        self.finished = 0
        self.errors = 0
        self.updated = 0


class _Emitter:
    def __init__(self, recorder, name):
        self._recorder = recorder
        self._name = name

    def emit(self, *payload):
        getattr(self._recorder, self._name).append(
            payload[0] if len(payload) == 1 else payload
        )


class _Counter:
    def __init__(self, recorder, name):
        self._recorder = recorder
        self._name = name

    def emit(self, *_args):
        setattr(
            self._recorder,
            self._name,
            getattr(self._recorder, self._name) + 1,
        )


class _Signals:
    def __init__(self, recorder):
        self._recorder = recorder

    state_changed = property(lambda s: _Emitter(s._recorder, "state_changed"))
    progress = property(lambda s: _Emitter(s._recorder, "progress"))
    sample_ready = property(lambda s: _Emitter(s._recorder, "sample_ready"))
    run_finished = property(lambda s: _Emitter(s._recorder, "run_finished"))
    run_request_finished = property(
        lambda s: _Emitter(s._recorder, "run_request_finished")
    )
    finished = property(lambda s: _Counter(s._recorder, "finished"))
    sweep_error = property(lambda s: _Counter(s._recorder, "errors"))
    updated = property(lambda s: _Counter(s._recorder, "updated"))


def _worker(config=None, **sweep_kwargs):
    options = {
        "start": 13_000_000,
        "end": 15_000_000,
        "points": 101,
        "segments": 1,
    }
    options.update(sweep_kwargs)
    app = FakeApp(
        vna=make_simulated_vna("antenna"),
        sweep=Sweep(**options),
        calibration=Calibration(),
        processing_config=config,
        running_average=RunningAverageState(),
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
    worker.signals = _Signals(recorder)
    return worker, recorder


def _smooth_config(points=5):
    return ProcessingConfig.build(smoothing_stage(points))


class TestIdentity:
    @staticmethod
    def test_a_missing_configuration_is_identity():
        assert _is_identity(None) is True

    @staticmethod
    def test_an_empty_configuration_is_identity():
        assert _is_identity(ProcessingConfig()) is True

    @staticmethod
    def test_a_configured_smoothing_stage_is_not_identity():
        assert _is_identity(_smooth_config()) is False


class TestCaptureFreezesProcessing:
    @staticmethod
    def test_the_start_of_a_run_freezes_the_configuration():
        worker, _recorder = _worker(_smooth_config())
        worker.run_once()
        capture = worker._capture
        assert capture is not None
        assert capture.processing is not None
        assert capture.processing.find(STAGE_SMOOTHING) is not None

    @staticmethod
    def test_an_edit_after_the_capture_does_not_reach_the_running_run():
        """Replacing the app's configuration mid-run must not alter the run."""
        worker, _recorder = _worker(_smooth_config(5))
        worker.run_once()
        frozen = worker._capture.processing
        worker.app.processing_config = _smooth_config(15)
        assert worker._capture.processing is frozen
        assert frozen.find(STAGE_SMOOTHING).as_dict["aperture_points"] == 5


class TestProcessedPublication:
    @staticmethod
    def test_an_identity_run_publishes_the_calibrated_samples_unchanged():
        plain, plain_recorder = _worker(None)
        plain.run_once()
        identity, identity_recorder = _worker(ProcessingConfig())
        identity.run_once()
        assert [tuple(s.s11) for s in plain_recorder.sample_ready] == [
            tuple(s.s11) for s in identity_recorder.sample_ready
        ]

    @staticmethod
    def test_smoothing_changes_the_published_channels():
        plain, plain_recorder = _worker(None)
        plain.run_once()
        smoothed, smoothed_recorder = _worker(_smooth_config(11))
        smoothed.run_once()
        plain_s11 = plain_recorder.sample_ready[-1].s11
        smooth_s11 = smoothed_recorder.sample_ready[-1].s11
        assert len(plain_s11) == len(smooth_s11)
        assert [point.freq for point in plain_s11] == [
            point.freq for point in smooth_s11
        ]
        # A smoothed channel is a different measurement, not a relabelled one.
        assert any(
            abs(a.z - b.z) > 1e-12
            for a, b in zip(plain_s11, smooth_s11, strict=True)
        )

    @staticmethod
    def test_the_raw_acquisition_is_kept_beside_the_processed_view():
        worker, recorder = _worker(_smooth_config(11))
        worker.run_once()
        sample = recorder.sample_ready[-1]
        snapshot = sample.snapshot
        assert snapshot is not None
        # Raw and corrected are both present, and they are not the same data.
        raw = snapshot.channel(S11, corrected=False)
        corrected = snapshot.channel(S11)
        assert raw
        assert corrected
        assert raw != corrected

    @staticmethod
    def test_the_processing_that_was_applied_is_recorded_on_the_snapshot():
        worker, recorder = _worker(_smooth_config(11))
        worker.run_once()
        provenance = recorder.sample_ready[-1].snapshot.provenance
        assert provenance.processing is not None
        assert "smoothing" in provenance.processing.describe().lower()

    @staticmethod
    def test_an_identity_run_records_no_processing():
        worker, recorder = _worker(None)
        worker.run_once()
        assert recorder.sample_ready[-1].snapshot.provenance.processing is None

    @staticmethod
    def test_the_completion_state_is_unaffected_by_processing():
        worker, recorder = _worker(_smooth_config(11))
        worker.run_once()
        assert recorder.run_finished == [SweepState.COMPLETED]


class TestFailureContainment:
    @staticmethod
    def test_a_failing_stage_raises_instead_of_publishing_partial_output():
        """Unprocessed samples must never be published as processed ones."""
        worker, _recorder = _worker(_smooth_config(5))
        worker.app.processing_config = _smooth_config(5)
        capture = worker._capture_settings()
        # A smoothing stage whose aperture cannot be honoured is a broken
        # configuration; the worker must surface it, not fall back silently.
        stages = capture.processing.stages
        object.__setattr__(
            capture.processing,
            "stages",
            (ProcessingStage.build("smoothing", {"aperture_points": 0}),),
        )
        try:
            with pytest.raises(ProcessingError):
                worker._apply_processing([Datapoint(1, 0.1, 0.0)], [], capture)
        finally:
            object.__setattr__(capture.processing, "stages", stages)


class TestAverageContext:
    @staticmethod
    def test_a_new_grid_changes_the_context():
        worker, _recorder = _worker()
        first = SweepCapture(
            sweep_id=1,
            sweep=Sweep(start=13_000_000, end=14_000_000, points=11),
            calibration=Calibration(),
            offset_delay=0.0,
            averages=1,
            truncates=0,
            continuous=False,
        )
        second = SweepCapture(
            sweep_id=2,
            sweep=Sweep(start=13_000_000, end=15_000_000, points=11),
            calibration=Calibration(),
            offset_delay=0.0,
            averages=1,
            truncates=0,
            continuous=False,
        )
        assert worker._average_context(first) != worker._average_context(second)

    @staticmethod
    def test_the_same_capture_keeps_a_stable_context():
        worker, _recorder = _worker()
        capture = SweepCapture(
            sweep_id=7,
            sweep=Sweep(start=13_000_000, end=14_000_000, points=11),
            calibration=Calibration(),
            offset_delay=0.0,
            averages=1,
            truncates=0,
            continuous=True,
        )
        assert worker._average_context(capture) == worker._average_context(
            capture
        )

    @staticmethod
    def test_the_reference_comes_from_the_device_not_the_display():
        worker, _recorder = _worker()
        # What an opened 75 ohm file leaves behind on the application.
        worker.app.reference_impedance = 75.0
        assert worker._reference_impedance() == 50.0

    @staticmethod
    def test_a_missing_impedance_defaults_to_50():
        worker, _recorder = _worker()
        del worker.app.reference_impedance
        assert worker._reference_impedance() == 50.0


class TestSmoothingAperture:
    @staticmethod
    def test_a_percent_aperture_is_odd_and_bounded():
        aperture = smoothing_aperture_for(101, 10)
        assert aperture % 2 == 1
        assert 1 <= aperture <= 101

    @staticmethod
    def test_a_tiny_dataset_yields_a_one_point_aperture():
        assert smoothing_aperture_for(1, 50) == 1


class TestReferenceImpedance:
    """RF-02: a renormalized run must not publish a 50 ohm snapshot."""

    @staticmethod
    def test_an_identity_run_states_the_reference_it_measured_at(monkeypatch):
        worker, recorder = _worker()
        monkeypatch.setattr(
            type(worker.app.vna), "reference_impedance", 75.0, raising=True
        )
        worker.run_once()
        snapshot = recorder.sample_ready[-1].snapshot
        assert snapshot.reference_impedance == 75.0

    @staticmethod
    def test_a_file_opened_earlier_does_not_relabel_a_live_sweep():
        worker, recorder = _worker()
        worker.app.reference_impedance = 75.0
        worker.run_once()
        assert recorder.sample_ready[-1].snapshot.reference_impedance == 50.0

    @staticmethod
    def test_a_renormalizing_run_reports_the_new_reference():
        from NanoVNASaver.Processing import renormalize_stage

        worker, recorder = _worker(
            ProcessingConfig.build(renormalize_stage(75.0))
        )
        worker.run_once()
        snapshot = recorder.sample_ready[-1].snapshot
        assert snapshot.reference_impedance == 75.0
        assert (
            worker._apply_processing(
                [Datapoint(1, 0.1, 0.0)],
                [],
                worker._capture,
            )[2]
            == 75.0
        )

    @staticmethod
    def test_the_snapshot_and_the_returned_reference_cannot_disagree():
        """The number comes from the same run as the samples."""
        from NanoVNASaver.Processing import renormalize_stage

        worker, recorder = _worker(
            ProcessingConfig.build(renormalize_stage(75.0))
        )
        worker.run_once()
        processed11, _processed21, reference = worker._apply_processing(
            [Datapoint(1, 0.1, 0.0)], [], worker._capture
        )
        assert reference == 75.0
        assert len(processed11) == 1
        assert recorder.sample_ready[-1].snapshot.reference_impedance == 75.0


class TestReportLabels:
    @staticmethod
    def test_a_processed_trace_is_labelled_with_its_pipeline():
        from NanoVNASaver.Reports import ReportTrace, provenance_labels

        worker, recorder = _worker(_smooth_config(11))
        worker.run_once()
        snapshot = recorder.sample_ready[-1].snapshot
        labels = provenance_labels(ReportTrace("s1", "S1", "#fff", snapshot))
        assert any(label.startswith("PROCESSED (") for label in labels)
        assert any("smoothing" in label.lower() for label in labels)

    @staticmethod
    def test_an_unprocessed_trace_says_nothing_about_processing():
        from NanoVNASaver.Reports import ReportTrace, provenance_labels

        worker, recorder = _worker(None)
        worker.run_once()
        snapshot = recorder.sample_ready[-1].snapshot
        labels = provenance_labels(ReportTrace("s1", "S1", "#fff", snapshot))
        assert not any("PROCESSED" in label for label in labels)


class TestRunningAverageIsPerSweep:
    """DSP-03: the average belongs to a whole sweep, not to one segment."""

    @staticmethod
    def _config():
        return ProcessingConfig.build(average_stage(mode="cumulative"))

    @staticmethod
    def test_two_sweeps_accumulate_instead_of_restarting():
        worker, _recorder = _worker(TestRunningAverageIsPerSweep._config())
        worker.run_once()
        assert worker.app.running_average.count == 1
        worker.data11 = []
        worker.data21 = []
        worker.rawData11 = []
        worker.rawData21 = []
        worker._sweep_counter += 1
        worker.run_once()
        # The accumulator must continue, not start over: publishing
        # ``result.running`` back is what makes the second sweep count 2.
        assert worker.app.running_average.count == 2

    @staticmethod
    def test_a_segmented_sweep_is_one_pass_not_one_per_segment():
        """Each segment has its own grid; blending per segment resets."""
        worker, _recorder = _worker(
            TestRunningAverageIsPerSweep._config(), points=51, segments=3
        )
        worker.run_once()
        assert worker.sweep.segments == 3
        assert worker.app.running_average.count == 1

    @staticmethod
    def test_the_published_samples_are_the_averaged_ones_but_raw_is_not():
        worker, _recorder = _worker(TestRunningAverageIsPerSweep._config())
        worker.run_once()
        published = list(worker.data11)
        raw = list(worker.rawData11)
        assert published
        assert raw
        # Raw is the acquisition: it is never the blended view.
        assert published[0].freq == raw[0].freq
        assert worker.app.running_average.count == 1

    @staticmethod
    def test_editing_the_pipeline_starts_a_new_average():
        """A different pipeline must not blend into the old accumulator."""
        worker, _recorder = _worker(TestRunningAverageIsPerSweep._config())
        worker.run_once()
        first = worker.app.running_average.context
        worker.data11 = []
        worker.data21 = []
        worker.rawData11 = []
        worker.rawData21 = []
        worker.app.processing_config = ProcessingConfig.build(
            average_stage(mode="cumulative"),
            smoothing_stage(5),
        )
        worker._sweep_counter += 1
        worker.run_once()
        assert worker.app.running_average.context != first
        assert worker.app.running_average.count == 1


def _second_run(worker, scale=0.5):
    """Sweep again with the device reading ``scale`` times its first values."""
    original = worker.app.vna.readValues
    worker.app.vna.readValues = lambda name: [
        value * scale for value in original(name)
    ]
    worker.run_once()
    worker.app.vna.readValues = original


class TestRunningAverageIsPublished:
    """What the provenance claims must be what the sample holds."""

    @staticmethod
    def _config():
        return ProcessingConfig.build(average_stage(mode="cumulative"))

    def test_the_published_sample_holds_the_blended_average(self):
        worker, recorder = _worker(self._config())
        worker.run_once()
        first = recorder.sample_ready[-1].s11[0].z
        _second_run(worker)

        sample = recorder.sample_ready[-1]
        expected = (first + first * 0.5) / 2
        assert sample.s11[0].z == pytest.approx(expected)
        assert sample.snapshot.corrected[S11][0].z == pytest.approx(expected)
        assert sample.snapshot.provenance.processing is not None

    def test_raw_keeps_the_single_sweep_beside_the_average(self):
        worker, recorder = _worker(self._config())
        worker.run_once()
        first = recorder.sample_ready[-1].snapshot.raw[S11][0].z
        _second_run(worker)
        raw = recorder.sample_ready[-1].snapshot.raw[S11][0].z
        assert raw == pytest.approx(first * 0.5)

    def test_a_partial_sample_does_not_claim_the_average(self):
        worker, recorder = _worker(self._config(), segments=2)
        worker.run_once()
        partial, complete = recorder.sample_ready[0], recorder.sample_ready[-1]
        assert partial.snapshot.provenance.processing is None
        assert complete.snapshot.provenance.processing is not None
        assert worker.app.running_average.count == 1

    def test_smoothing_runs_after_the_average_as_the_pipeline_declares(self):
        plain, plain_recorder = _worker()
        plain.run_once()
        measured = plain_recorder.sample_ready[-1].s11

        config = ProcessingConfig.build(
            average_stage(mode="cumulative"), smoothing_stage(5)
        )
        worker, recorder = _worker(config)
        worker.run_once()

        expected = apply_processing(
            {S11: measured}, ProcessingConfig.build(smoothing_stage(5))
        ).channels[S11]
        published = recorder.sample_ready[-1].s11
        assert [p.z for p in published] == pytest.approx(
            [p.z for p in expected]
        )
        # The accumulator holds the unsmoothed average: smoothing the running
        # mean again on every pass would widen it without limit.
        assert dict(worker.app.running_average.values)
        stored = worker.app.running_average.values[0][1]
        assert stored == pytest.approx(measured[0].z)


def _calibration(load):
    calibration = Calibration()
    for name, value in (("short", -0.99), ("open", 0.98), ("load", load)):
        calibration.insert(
            name,
            [Datapoint(f, value, 0.01) for f in (13_000_000, 15_000_000)],
        )
    calibration.calc_corrections()
    return calibration


class TestAverageContextSeparatesDataStreams:
    @staticmethod
    def test_a_new_calibration_starts_a_new_average():
        worker, _recorder = _worker()
        worker.app.calibration = _calibration(0.01)
        first = worker._average_context(worker._capture_settings())
        worker.app.calibration = _calibration(0.02)
        assert worker._average_context(worker._capture_settings()) != first

    @staticmethod
    def test_the_same_calibration_keeps_the_average():
        worker, _recorder = _worker()
        worker.app.calibration = _calibration(0.01)
        first = worker._average_context(worker._capture_settings())
        worker.app.calibration = _calibration(0.01)
        assert worker._average_context(worker._capture_settings()) == first

    @staticmethod
    def test_a_different_device_starts_a_new_average():
        worker, _recorder = _worker()
        first = worker._average_context(worker._capture_settings())
        worker.app.vna = make_simulated_vna("filter")
        assert worker._average_context(worker._capture_settings()) != first

    @staticmethod
    def test_a_recalibrated_run_does_not_blend_with_the_old_average():
        worker, _recorder = _worker(
            ProcessingConfig.build(average_stage(mode="cumulative"))
        )
        worker.app.calibration = _calibration(0.01)
        worker.run_once()
        worker.app.calibration = _calibration(0.02)
        worker.run_once()
        assert worker.app.running_average.count == 1
