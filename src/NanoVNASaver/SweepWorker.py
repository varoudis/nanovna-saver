#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020  Rune B. Broberg
#  Copyright (C) 2020,2021 NanoVNA-Saver Authors
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
"""Sweep acquisition (SWEEP-01..04).

The worker owns the device conversation.  It never touches widgets: every
publication is a signal carrying an immutable value object, so the GUI
thread applies state changes.

Lifecycle rules
---------------
* Exactly one terminal outcome per run, whichever path is taken.
* Settings are captured at acquisition start; later edits apply to the next
  sweep only.
* Only measured samples are ever published, so an interrupted sweep shows a
  labelled partial result instead of preallocated zeros.
* Every wait is interruptible, so cancellation is bounded by one device read
  rather than by a fixed sleep.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from .Calibration import (
    arrays_to_datapoints,
    correct_delay_bulk,
    datapoints_to_arrays,
    validate_delay,
)
from .Hardware.VNA import VNA
from .Measurement import (
    S11,
    S21,
    CompletionState,
    CorrectionSource,
    CorrectionState,
    SweepInfo,
    compute_calibration_identity,
    snapshot_from_datapoints,
)
from .Processing import (
    STAGE_AVERAGE,
    ProcessingConfig,
    apply_processing,
)
from .ReferenceImpedance import DEFAULT_REFERENCE_IMPEDANCE
from .RFTools import Datapoint, corr_att_data
from .Settings.Sweep import Sweep, SweepMode
from .SweepState import (
    SweepProgress,
    SweepState,
    SweepStateMachine,
    estimate_remaining,
)


def _is_identity(config) -> bool:
    """True when ``config`` describes no work (including "no configuration").

    A missing configuration and an empty one mean the same thing, so callers
    never have to distinguish "never configured" from "configured to do
    nothing".
    """
    if config is None:
        return True
    if isinstance(config, ProcessingConfig):
        return config.is_identity
    return True


def _apply_offset_delay(
    data: list[Datapoint], offset_delay: float, *, reflect: bool
) -> list[Datapoint]:
    """Bulk variant of :func:`Calibration.correct_delay` for one channel."""
    if not data:
        return []
    freq, values = datapoints_to_arrays(data)
    return arrays_to_datapoints(
        freq, correct_delay_bulk(freq, values, offset_delay, reflect=reflect)
    )


def _freeze_calibration(calibration):
    """Return the calibration state as it is now.

    A run must keep one coherent calibration for all of its samples: the
    identity published with a snapshot has to describe the coefficients that
    actually produced the corrected values.  :meth:`Calibration.frozen_copy`
    gives that, so a later ``load()``/``calc_corrections()`` on the live
    object — in place or by replacement — cannot change a running sweep.
    Objects without that method (test doubles) are used as they are.
    """
    freeze = getattr(calibration, "frozen_copy", None)
    if freeze is None:
        return calibration
    return freeze()


if TYPE_CHECKING:
    from .NanoVNASaver.NanoVNASaver import NanoVNASaver as vna_app


logger = logging.getLogger(__name__)

VALUE_MAX: float = 9.5
RETRIES_RECONNECT: int = 5
RETRIES_MAX: int = 10

#: Longest the worker can be busy without checking the cancel flag, outside
#: of a device read.
POLL_INTERVAL = 0.05


def truncate(values: list[list[complex]], count: int) -> list[list[complex]]:
    """truncate drops extrema from data list if averaging is active"""
    keep = len(values) - count
    logger.debug("Truncating from %d values to %d", len(values), keep)
    if count < 1 or keep < 1:
        logger.info("Not doing illegal truncate")
        return values
    truncated = []
    for valueset in np.swapaxes(values, 0, 1).tolist():
        avg = complex(np.average(valueset))

        def cmp_fn(v: complex, avg=avg) -> float:
            return float(abs(avg - v))

        truncated.append(sorted(valueset, key=cmp_fn)[:keep])
    return np.swapaxes(truncated, 0, 1).tolist()


@dataclass(frozen=True)
class SweepCapture:
    """Immutable settings captured when a sweep starts (SWEEP-02.1).

    Calibration and offset delay are held by value for the duration of the
    run.  ``calibration`` is a frozen copy taken at the capture boundary
    (see :meth:`Calibration.frozen_copy`), so replacing ``app.calibration``
    mid-sweep *and* reloading or recalculating the calibration in place both
    leave this run's coefficients, validity and identity untouched.
    """

    sweep_id: int
    sweep: Sweep
    calibration: object
    offset_delay: float
    averages: int
    truncates: int
    continuous: bool
    #: Ordered processing configuration frozen at the run boundary.  Held by
    #: value so an edit made mid-sweep cannot change what the samples already
    #: published by this run mean (DSP-01/DSP-02).
    processing: object = None
    #: Inline S21 attenuator in dB, frozen with the rest of the measurement
    #: configuration.  It is a physical pad in front of the S21 input, so it
    #: is undone *before* the calibration runs: the isolation standard is
    #: stored un-padded, and correcting a padded reading would subtract an
    #: unattenuated leakage term from an attenuated measurement.
    s21_att: float = 0.0

    @property
    def segments(self) -> int:
        return self.sweep.segments

    @property
    def total_units(self) -> int:
        return self.segments * self.averages

    @property
    def sweep_info(self) -> SweepInfo:
        return SweepInfo.from_sweep(self.sweep)


@dataclass(frozen=True)
class SweepSample:
    """One published, immutable acquisition result (SWEEP-02.2).

    ``request_token`` is the token the caller bound to this run (``None``
    when it bound none).  It is captured here, in the acquisition thread,
    rather than looked up when the queued signal is delivered: by then a
    newer run may own the worker, and the old sample must not be attributed
    to it.
    """

    sweep_id: int
    segment: int
    segments: int
    s11: tuple[Datapoint, ...]
    s21: tuple[Datapoint, ...]
    completion: CompletionState
    capture: SweepCapture
    snapshot: object = None
    request_token: object = None


class WorkerSignals(QObject):
    updated = Signal()
    finished = Signal()
    sweep_error = Signal()
    #: Structured lifecycle notifications.
    state_changed = Signal(object)
    progress = Signal(object)
    #: Published measured data; carries a :class:`SweepSample`.
    sample_ready = Signal(object)
    #: Emitted once per run with the terminal ``SweepState``.
    run_finished = Signal(object)
    #: Emitted once per run with ``(terminal, request_token)``.
    #:
    #: ``run_finished`` alone cannot say *which* run finished: a terminal
    #: event already queued for an earlier run is delivered to whoever is
    #: listening next, so a caller that re-armed an identical sweep would
    #: adopt somebody else's result.  This signal repeats the same terminal
    #: state and adds the correlation token the caller supplied.
    run_request_finished = Signal(object, object)


class SweepWorker(QThread):
    """Runs exactly one acquisition at a time on its own thread."""

    #: Reason the last segment read was unusable ("" once data is valid).
    _segment_error = ""

    #: Class-level defaults keep the worker usable when it is built without
    #: ``__init__`` (headless harnesses construct it via ``__new__``);
    #: ``__init__`` rebinds them per instance.
    _request_token: object = None
    _active_token: object = None

    def __init__(self, app: "vna_app") -> None:
        super().__init__()
        logger.info("Initializing SweepWorker")
        self.signals: WorkerSignals = WorkerSignals()
        self._request_token = None
        self._active_token = None
        self.app = app
        self.sweep = Sweep()
        self.percentage: float = 0.0
        self.data11: list[Datapoint] = []
        self.data21: list[Datapoint] = []
        self.rawData11: list[Datapoint] = []
        self.rawData21: list[Datapoint] = []
        self.error_message: str = ""
        self.offsetDelay: float = 0.0
        self._cancel = Event()
        # Lifecycle state must exist before init_data() logs the sweep id.
        self._state_machine = SweepStateMachine()
        self._capture: SweepCapture | None = None
        self._sweep_counter = 0
        self._started_at = 0.0
        self._retry_count = 0
        self._retry_reason = ""
        self._last_terminal: SweepState | None = None
        self._last_progress = None
        self._segment_error = ""
        self.init_data()

    # -- lifecycle -----------------------------------------------------
    @property
    def state(self) -> SweepState:
        return self._state_machine.state

    @property
    def sweep_id(self) -> int:
        return self._state_machine.sweep_id

    @property
    def last_terminal_state(self) -> SweepState | None:
        return self._last_terminal

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def capture(self) -> SweepCapture | None:
        return self._capture

    @Slot()
    def quit(self) -> None:
        """Request cancellation; returns immediately."""
        logger.debug("Worker quit request")
        if self._state_machine.state.active:
            self._transition(SweepState.CANCELLING)
        self._cancel.set()

    def cancel(self) -> None:
        """Alias for :meth:`quit` used by the UI."""
        self.quit()

    def bind_next_run(self, token: object) -> None:
        """Correlate the *next* run with ``token``.

        Nothing is started here.  A caller that must recognise its own
        terminal event - rather than any event that happens to arrive next -
        passes a token it can compare by identity; :attr:`request_token`
        then reports which token the running (or last finished) run carried.
        """
        self._request_token = token

    @property
    def request_token(self) -> object:
        """Token of the run currently publishing (``None`` when idle)."""
        return self._active_token

    @Slot()
    def run(self) -> None:
        """QThread entry point."""
        self._cancel.clear()
        self.run_once()

    def run_once(self) -> None:
        """Synchronous entry point used by tests and headless runs."""
        self._sweep_counter += 1
        self._active_token = self._request_token
        self._request_token = None
        self._state_machine.reset(self._sweep_counter)
        self._last_terminal = None
        self._started_at = monotonic()
        self.error_message = ""
        self._segment_error = ""
        terminal = SweepState.FAILED
        try:
            terminal = self._run()
        except BaseException as exc:
            logger.exception("%s", exc)
            self.error_message = f"ERROR during sweep\n\nStopped\n\n{exc}"
            terminal = (
                SweepState.CANCELLED
                if self._cancel.is_set()
                else SweepState.FAILED
            )
            if terminal is SweepState.FAILED:
                self.signals.sweep_error.emit()
        finally:
            self._finish(terminal)

    def _finish(self, terminal: SweepState) -> None:
        """Exactly one terminal outcome and one cleanup path."""
        if (
            terminal is SweepState.CANCELLED
            and self._state_machine.state.active
            and self._state_machine.state is not SweepState.CANCELLING
        ):
            # Cancellation always passes through CANCELLING so the machine's
            # declared transitions hold even for an immediately cancelled run.
            self._transition(SweepState.CANCELLING)
        for _ in range(3):
            if self._state_machine.state.terminal:
                break
            try:
                self._transition(terminal)
            except Exception:  # fall back to FAILED
                logger.debug("Cannot reach %s, forcing", terminal)
                if terminal is SweepState.FAILED:
                    break
                terminal = SweepState.FAILED
        if not self._state_machine.state.terminal:
            # Directly record the outcome; the machine cannot reach it.
            self._state_machine.state = terminal
            self._state_machine.terminal_count += 1
            self._state_machine.history.append(terminal)
        self._last_terminal = self._state_machine.state
        token = self._active_token
        logger.debug(
            "Sweep %s finished as %s", self.sweep_id, self._last_terminal.value
        )
        if self._last_terminal is SweepState.COMPLETED:
            self.signals.finished.emit()
        self.signals.run_finished.emit(self._last_terminal)
        # The token is emitted *after* the generic signal and never cleared
        # before emitting: a listener that only knows the token can still
        # correlate this completion, and a listener that ignores it keeps
        # the established behaviour.
        self.signals.run_request_finished.emit(self._last_terminal, token)
        self._active_token = None

    def _transition(self, new_state: SweepState) -> None:
        if self._state_machine.transition(new_state):
            self.signals.state_changed.emit(new_state)

    def _run(self) -> SweepState:
        logger.info("Initializing SweepWorker")
        self._transition(SweepState.PREPARING)
        if not self.app.vna.connected():
            logger.debug(
                "Attempted to run without being connected to the NanoVNA"
            )
            return SweepState.FAILED

        preflight_problem = self._preflight_problem()
        if preflight_problem is not None:
            # A scenario that cannot answer this span is rejected before any
            # sample is read, so the display keeps the previous measurement
            # instead of showing an invented one.  The reason is reported
            # through the normal error path so the user can act on it.
            logger.error("Sweep preflight failed: %s", preflight_problem)
            self.error_message = preflight_problem
            self.signals.sweep_error.emit()
            return SweepState.FAILED

        self._capture = self._capture_settings()
        self.percentage = 0.0
        self._retry_count = 0
        self._retry_reason = ""
        self.init_data()

        if self._cancel.is_set():
            return SweepState.CANCELLED

        terminal = self._run_loop()
        if terminal is not None:
            return terminal

        self._restore_sweep_range()
        self.percentage = 100.0
        self._emit_progress(message="Complete")
        logger.debug('Sending "finished" signal')
        return SweepState.COMPLETED

    def _preflight_problem(self) -> str | None:
        """Ask the connected device whether the configured sweep is usable.

        Every segment's own request range is checked: rounding in
        ``Sweep.get_index_range`` can push a segment end past the configured
        stop, so a segmented sweep is rejected before any sample is read
        instead of failing halfway through.
        """
        check = getattr(self.app.vna, "preflight", None)
        if not callable(check):
            return None
        sweep = self.app.sweep
        for index in range(max(sweep.segments, 1)):
            start, stop = sweep.get_index_range(index)
            try:
                problem = check(start, stop)
            except Exception as exc:  # a broken preflight must not read
                logger.exception("Device preflight failed: %s", exc)
                return f"The device rejected the sweep configuration: {exc}"
            if problem is not None:
                return problem
        return None

    def _capture_settings(self) -> SweepCapture:
        sweep = self.app.sweep.copy()
        self.sweep = sweep
        averages = (
            sweep.properties.averages[0]
            if sweep.properties.mode == SweepMode.AVERAGE
            else 1
        )
        try:
            offset_delay = validate_delay(self.offsetDelay)
        except ValueError as exc:
            raise ValueError(f"Cannot start sweep: {exc}") from exc
        # Freeze at the run boundary: the coefficients used by every sample
        # of this run, and the identity published with them, must stay the
        # ones that were in effect when it started.  The processing
        # configuration is frozen for the same reason.
        return SweepCapture(
            sweep_id=self.sweep_id,
            sweep=sweep,
            calibration=_freeze_calibration(self.app.calibration),
            offset_delay=offset_delay,
            averages=averages,
            truncates=sweep.properties.averages[1],
            continuous=sweep.properties.mode == SweepMode.CONTINOUS,
            processing=getattr(self.app, "processing_config", None),
            s21_att=self._capture_attenuator(),
        )

    def _capture_attenuator(self) -> float:
        """The inline S21 attenuator this run will un-pad, in dB.

        Only a positive value is a pad: anything else means there is none.
        Reading it here, beside the calibration and offset delay, is what
        stops an edit made mid-sweep from changing what the samples already
        published mean.
        """
        att = float(getattr(self.app, "s21att", 0.0) or 0.0)
        return att if att > 0.0 else 0.0

    def _run_loop(self) -> SweepState | None:
        """Acquire segments; return a terminal state when stopping early."""
        capture = self._capture
        if capture is None:
            return SweepState.FAILED
        sweep = capture.sweep
        logger.info("%d averages", capture.averages)

        while True:
            # Each pass is a complete measurement: start from empty buffers so
            # a continuous sweep replaces the previous pass instead of
            # appending another copy of the grid (repeated frequencies,
            # unbounded growth).
            self.init_data()
            for index in range(sweep.segments):
                if self._cancel.is_set():
                    logger.debug("Stopping sweeping as signalled")
                    return SweepState.CANCELLED
                logger.debug("Sweep segment no %d", index)
                start, stop = sweep.get_index_range(index)

                self._transition(SweepState.MEASURING)
                result = self.read_averaged_segment(
                    start, stop, capture.averages, segment=index
                )
                if result is None:
                    return SweepState.CANCELLED
                freq, values11, values21 = result
                self.update_data(freq, values11, values21, index)
            if not capture.continuous:
                break
            if self._cancel.is_set():
                return SweepState.CANCELLED
        return None

    def _restore_sweep_range(self) -> None:
        """Restore the device range, without masking an earlier failure."""
        capture = self._capture
        if capture is None or capture.segments <= 1:
            return
        sweep = capture.sweep
        logger.debug(
            "Resetting NanoVNA sweep to full range: %d to %d",
            sweep.start,
            sweep.end,
        )
        try:
            self.app.vna.resetSweep(sweep.start, sweep.end)
        except Exception as exc:  # report, do not raise
            logger.warning("Unable to restore sweep range: %s", exc)
            self.error_message = (
                "The sweep completed but the device sweep range could not be"
                f" restored: {exc}"
            )

    # -- progress ------------------------------------------------------
    def _emit_progress(
        self,
        state: SweepState | None = None,
        *,
        segment: int | None = None,
        average: int | None = None,
        completed_units: int | None = None,
        message: str = "",
    ) -> SweepProgress:
        capture = self._capture
        segments = capture.segments if capture else 1
        averages = capture.averages if capture else 1
        elapsed = monotonic() - self._started_at
        previous = getattr(self, "_last_progress", None)
        completed_segments = (
            previous.completed_segments if previous is not None else 0
        )
        completed_averages = (
            previous.completed_averages if previous is not None else 0
        )
        if completed_units is not None and averages > 0:
            completed_segments = completed_units // averages
            completed_averages = completed_units % averages
        progress = SweepProgress(
            sweep_id=self.sweep_id,
            state=state or self._state_machine.state,
            segment=segment if segment is not None else completed_segments,
            segments=segments,
            average=average if average is not None else completed_averages,
            averages=averages,
            completed_segments=completed_segments,
            completed_averages=completed_averages,
            retry_count=self._retry_count,
            retry_reason=self._retry_reason,
            elapsed_seconds=elapsed,
            estimate_seconds=estimate_remaining(
                elapsed,
                completed_segments * averages + completed_averages,
                segments * averages,
            ),
            continuous=capture.continuous if capture else False,
            message=message,
        )
        self._last_progress = progress
        self.percentage = progress.percentage
        self.signals.progress.emit(progress)
        return progress

    # -- waits ---------------------------------------------------------
    def _sleep(self, seconds: float) -> bool:
        """Interruptible wait.  Returns False when cancellation was asked."""
        if self._cancel.wait(seconds):
            logger.debug("Wait cancelled after %.3fs of %.3fs", 0, seconds)
            return False
        return True

    # -- data ----------------------------------------------------------
    def init_data(self) -> None:
        """Start with no samples: only measured data is ever published."""
        self.data11 = []
        self.data21 = []
        self.rawData11 = []
        self.rawData21 = []
        logger.debug("Init data cleared for sweep %s", self.sweep_id)

    def _apply_processing(
        self,
        data11: list[Datapoint],
        data21: list[Datapoint],
        capture: "SweepCapture | None",
    ) -> tuple[list[Datapoint], list[Datapoint], float]:
        """Apply this run's frozen processing to one segment.

        The raw and calibrated buffers kept by the caller are the acquisition;
        this returns only the view that is published and snapshotted.  An
        identity configuration returns the inputs unchanged, so a run without
        processing behaves exactly as before.

        A stage that fails raises: publishing a half-processed segment would
        present unprocessed samples as if they had been processed.  The run
        reports the failure through its normal error path instead.

        The third element is the reference impedance the returned samples are
        expressed in.  A renormalization stage changes it, so it is returned
        rather than re-read from the application: the number and the samples
        must come from the same run.
        """
        config = capture.processing if capture is not None else None
        reference = self._reference_impedance()
        if _is_identity(config):
            return data11, data21, reference
        # A running average is a property of a whole sweep, not of one
        # segment: each segment has a different frequency grid, so blending
        # per segment would reset the accumulator every time.  It and every
        # stage after it run once per pass in ``_blend_whole_sweep`` instead.
        config = self._segment_config(config)
        channels: dict[str, list[Datapoint]] = {}
        if data11:
            channels[S11] = list(data11)
        if data21:
            channels[S21] = list(data21)
        if not channels:
            return data11, data21, reference
        result = apply_processing(
            channels,
            config,
            reference_impedance=reference,
        )
        return (
            list(result.channels.get(S11, ())),
            list(result.channels.get(S21, ())),
            float(getattr(result, "reference_impedance", reference)),
        )

    def _reference_impedance(self) -> float:
        """The reference the connected device's samples are expressed in.

        This comes from the device, never from the application's display
        reference: that one follows whatever was last shown, so a 75 ohm file
        that was opened earlier must not relabel a 50 ohm measurement.
        """
        value = getattr(self.app.vna, "reference_impedance", None)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
        return DEFAULT_REFERENCE_IMPEDANCE

    def _average_context(self, capture: "SweepCapture | None") -> str:
        """Identity of the data stream an average may blend across.

        A new source, a new calibration or a new grid must not be averaged
        with the previous one, so the context changes whenever any of those
        does; ``RunningAverageState`` resets on any context change.  Within
        one run every segment shares it, which is what lets a running average
        accumulate across passes of a continuous sweep.
        """
        if capture is None:
            return "no-capture"
        correction = self._correction_state(capture)
        device = self.app.vna
        port = getattr(getattr(device, "serial", None), "port", "")
        source = (
            getattr(device, "name", ""),
            getattr(device, "scenario_key", ""),
            getattr(device, "provenance_note", ""),
            port,
        )
        return "|".join(
            (
                f"source:{source}",
                f"cal:{correction.identity}",
                f"delay:{correction.offset_delay!r}",
                f"grid:{capture.sweep}",
                f"avg:{capture.averages}",
            )
        )

    @staticmethod
    def _split_at_average(config) -> tuple:
        """Stages before the running average, and the average onwards.

        The pipeline's declared order applies to whole sweeps too: a stage
        ordered after the average has to see the averaged samples, or the
        provenance would describe an order that never ran.
        """
        if config is None or config.find(STAGE_AVERAGE) is None:
            return config, None
        stages = tuple(config.stages)
        position = stages.index(config.find(STAGE_AVERAGE))
        return (
            ProcessingConfig(
                stages=stages[:position], format_version=config.format_version
            ),
            ProcessingConfig(
                stages=stages[position:], format_version=config.format_version
            ),
        )

    @staticmethod
    def _segment_config(config):
        """``config`` without the stages that need a whole sweep."""
        return SweepWorker._split_at_average(config)[0]

    def _blend_whole_sweep(
        self, capture: "SweepCapture", reference_impedance: float
    ) -> None:
        """Blend one complete pass into the running average (DSP-03).

        Called from :meth:`update_data` when the last segment arrives and
        before that sample is published, so the accumulator sees complete
        channels on a single grid and the snapshot holds what its provenance
        says.  The published buffers are replaced by the blended ones;
        ``rawData11``/``rawData21`` keep the unprocessed acquisition, so an
        averaged sample is never presented as a measurement.

        The accumulator is published back to the application only for a
        complete, non-cancelled pass: a partial pass must not advance the
        average that the next run continues from.
        """
        average = self._split_at_average(capture.processing)[1]
        if average is None:
            return
        channels: dict[str, list[Datapoint]] = {}
        if self.data11:
            channels[S11] = list(self.data11)
        if self.data21:
            channels[S21] = list(self.data21)
        if not channels:
            return
        result = apply_processing(
            channels,
            average,
            reference_impedance=reference_impedance,
            running=getattr(self.app, "running_average", None),
            average_context=self._whole_sweep_context(capture),
        )
        self.data11 = list(result.channels.get(S11, ()))
        self.data21 = list(result.channels.get(S21, ()))
        if result.running is not None and hasattr(self.app, "running_average"):
            # Immutable state: publishing it back is what lets the next run
            # continue this average rather than start a new one.
            self.app.running_average = result.running

    def _whole_sweep_context(self, capture: "SweepCapture") -> str:
        """Context for an average blended across whole sweeps.

        It extends the per-segment context with the pipeline identity, so
        editing any processing stage starts a new accumulation instead of
        blending samples that mean something different now.
        """
        identity = ""
        config = capture.processing
        if config is not None:
            identity = config.identity()
        return "|".join(
            (
                self._average_context(capture),
                f"proc:{identity}",
            )
        )

    def update_data(
        self,
        frequencies: list[int],
        values11: list[complex],
        values21: list[complex],
        index: int,
    ) -> None:
        logger.debug(
            "Calculating data and inserting in existing data at index %d", index
        )
        raw_data11 = [
            Datapoint(freq, values11[i].real, values11[i].imag)
            for i, freq in enumerate(frequencies)
        ]
        # A one-port source publishes no S21 at all; indexing an absent
        # channel would fabricate zero samples (REPLAY-01.3).
        raw_data21 = (
            [
                Datapoint(freq, values21[i].real, values21[i].imag)
                for i, freq in enumerate(frequencies)
            ]
            if values21
            else []
        )

        capture = self._capture
        data11, data21 = self.applyCalibration(
            raw_data11, raw_data21, capture=capture
        )
        # Processing is applied to the per-segment view that is published and
        # snapshotted; the raw and calibrated buffers above keep what was
        # acquired, so a processed sample never overwrites a measurement.
        applied = self._apply_processing(data11, data21, capture)
        applied11, applied21, reference_impedance = applied
        self.data11.extend(applied11)
        self.data21.extend(applied21)
        self.rawData11.extend(raw_data11)
        self.rawData21.extend(raw_data21)
        simulated = bool(getattr(self.app.vna, "simulated", False))
        scenario = (
            getattr(self.app.vna, "scenario_key", "") if simulated else ""
        )
        # A replay scenario describes its source file here, so snapshots,
        # exports, sessions and reports all carry the replay provenance
        # (REPLAY-01.4).
        provenance_note = (
            str(getattr(self.app.vna, "provenance_note", "") or "")
            if simulated
            else ""
        )
        source = "simulated" if simulated else ""
        complete = capture is not None and index + 1 >= capture.segments
        if complete:
            # Every segment is in, so the pass is a whole sweep: this is the
            # only correct place to blend a running average (DSP-03), and it
            # has to happen before the sample below is built.
            self._blend_whole_sweep(capture, reference_impedance)
        completion = (
            CompletionState.COMPLETE if complete else CompletionState.PARTIAL
        )
        snapshot = snapshot_from_datapoints(
            self.data11,
            self.data21,
            raw11=self.rawData11,
            raw21=self.rawData21,
            completion=completion,
            sweep=capture.sweep if capture else None,
            calibration=capture.calibration if capture else None,
            simulated=simulated,
            simulator_scenario=scenario,
            device=self.app.vna.name,
            source=source,
            notes=provenance_note,
            correction=self._correction_state(capture),
            # The grid was measured against this reference (RF-02), so a
            # snapshot must state it rather than default to 50 ohm.
            reference_impedance=reference_impedance,
        )
        # A partial sample has not been through the whole-sweep stages yet,
        # so it must not claim them.
        applied_config = (
            None
            if capture is None
            else capture.processing
            if complete
            else self._segment_config(capture.processing)
        )
        if not _is_identity(applied_config):
            # The channels above are the processed ones, so the snapshot's
            # corrected view already holds them; this records what was
            # applied.  ``raw`` and the calibration identity are untouched,
            # so the unprocessed acquisition stays available beside it.
            snapshot = snapshot.with_processing(
                applied_config, reference_impedance=reference_impedance
            )
        sample = SweepSample(
            sweep_id=self.sweep_id,
            segment=index,
            segments=capture.segments if capture else 1,
            s11=tuple(self.data11),
            s21=tuple(self.data21),
            completion=completion,
            capture=capture,
            snapshot=snapshot,
            request_token=self._active_token,
        )
        logger.debug(
            "Publishing %d/%d points for sweep %s (%s)",
            len(self.data11),
            len(self.data21),
            self.sweep_id,
            completion.value,
        )
        # Publication is a signal: the GUI thread owns all widget mutation
        # and the application's live data buffers (SWEEP-02.2).
        self.signals.sample_ready.emit(sample)

    def _correction_state(self, capture) -> CorrectionState:
        """Describe the correction that was actually applied.

        ``applyCalibration`` falls back to the raw samples when the
        calibration is unusable, and applies an offset delay even without a
        SOL calibration, so the snapshot must record what happened rather
        than assume that carrying a calibration means it took effect.  A
        delay-only correction is reported as applied with an empty identity:
        that keeps it distinct from a calibration correction, and from raw
        data.
        """
        calibration = (
            capture.calibration if capture is not None else self.app.calibration
        )
        offset_delay = (
            capture.offset_delay if capture is not None else self.offsetDelay
        )
        calibrated = bool(
            getattr(calibration, "isCalculated", False)
            and calibration.isValid1Port()
        )
        delay = float(offset_delay) if offset_delay else 0.0
        applied = calibrated or delay != 0.0
        return CorrectionState(
            applied=applied,
            identity=compute_calibration_identity(calibration)
            if calibrated
            else "",
            source=CorrectionSource.APPLICATION
            if applied
            else CorrectionSource.NONE,
            offset_delay=delay,
        )

    def applyCalibration(
        self,
        raw_data11: list[Datapoint],
        raw_data21: list[Datapoint],
        *,
        capture: SweepCapture | None = None,
    ) -> tuple[list[Datapoint], list[Datapoint]]:
        """Apply the captured calibration and offset delay to raw samples.

        Correction runs through :class:`Calibration`'s bulk (NumPy) path, so
        a whole sweep paragraph is corrected with a handful of array
        operations instead of one SciPy interpolation call per datapoint.
        The scalar ``correct11``/``correct21`` methods remain the reference
        implementation; results here are numerically equal, not necessarily
        bit-identical (NumPy and CPython complex division may differ in the
        last bit).

        Raw samples are returned unchanged whenever the calibration cannot be
        applied (not calculated, incomplete, or a channel unavailable); the
        correction state published with the sweep records which happened.
        The offset delay is independent of the SOL calibration: it is applied
        to the (corrected or raw) samples whenever it is non-zero.

        Anything that would produce a non-finite sample raises instead of
        publishing an invalid measurement: a non-finite offset delay, or a
        correction that is undefined at some frequency.

        The inline S21 attenuator is undone before the correction runs, so a
        padded reading is corrected against standards measured the same way
        and the pad never reaches the published samples.

        Without an explicit ``capture`` the *live* calibration, offset
        delay and attenuator are used, which is what re-correcting already
        acquired raw data from the calibration window needs.  During a sweep
        the worker always passes its frozen capture, so a run can never mix
        calibration states.
        """
        calibration = (
            capture.calibration if capture is not None else self.app.calibration
        )
        offset_delay = validate_delay(
            capture.offset_delay if capture is not None else self.offsetDelay
        )
        # The inline attenuator is a physical pad in front of the S21 input,
        # so it belongs to the measurement, not to the published result.  It
        # is therefore undone *here*, before the calibration, and the padding
        # never reaches ``saveData``.  Correcting the padded reading instead
        # subtracted an unattenuated isolation term from an attenuated
        # measurement, leaving ``|isolation| * (a - 1)`` in every corrected
        # S21 sample.
        att = (
            capture.s21_att
            if capture is not None
            else self._capture_attenuator()
        )
        padded21 = corr_att_data(raw_data21, att)

        if not calibration.isCalculated:
            data11 = raw_data11.copy()
            data21 = padded21
        elif not raw_data11 and not raw_data21:
            return [], []
        elif raw_data11 and raw_data21:
            data11, data21 = self._calibrate_bulk(
                calibration, raw_data11, padded21
            )
        elif raw_data21:
            if calibration.isValid2Port():
                raise ValueError(
                    "Cannot calibrate: S21 samples need the matching raw S11"
                    " samples, but none were supplied."
                )
            data11 = []
            data21 = padded21
        else:
            # Only S11 was measured (one-port device or an unread S21).
            data11 = self._calibrate_channel(calibration, raw_data11)
            data21 = []

        if offset_delay != 0.0:
            data11, data21 = self._apply_delay(data11, data21, offset_delay)

        return data11, data21

    @staticmethod
    def _apply_delay(
        data11: list[Datapoint],
        data21: list[Datapoint],
        offset_delay: float,
    ) -> tuple[list[Datapoint], list[Datapoint]]:
        """Apply the offset delay in bulk; S11 gets the doubled factor."""
        try:
            return (
                _apply_offset_delay(data11, offset_delay, reflect=True),
                _apply_offset_delay(data21, offset_delay, reflect=False),
            )
        except ValueError as exc:
            raise ValueError(f"Cannot apply offset delay: {exc}") from exc

    @staticmethod
    def _calibrate_bulk(
        calibration,
        raw_data11: list[Datapoint],
        raw_data21: list[Datapoint],
    ) -> tuple[list[Datapoint], list[Datapoint]]:
        """Correct both channels over the raw frequency grid, in bulk.

        Both channels must describe the same frequency grid: the S21
        correction is evaluated per frequency and consumes the matching raw
        S11 sample, so a shifted grid would silently pair the wrong values.
        The integer hertz grid is carried through unchanged, which keeps the
        published frequencies exactly as measured.
        """
        if len(raw_data11) != len(raw_data21):
            raise ValueError(
                "Cannot calibrate: S11 has"
                f" {len(raw_data11)} point(s) but S21 has"
                f" {len(raw_data21)}."
            )
        freq, raw11 = datapoints_to_arrays(raw_data11)
        freq21, raw21 = datapoints_to_arrays(raw_data21)
        if not np.array_equal(freq, freq21):
            raise ValueError(
                "Cannot calibrate: the S11 and S21 frequency grids differ."
            )

        corrected11 = (
            calibration.correct11_bulk(freq, raw11)
            if calibration.isValid1Port()
            else raw11
        )
        corrected21 = (
            calibration.correct21_bulk(freq, raw21, raw11)
            if calibration.isValid2Port()
            else raw21
        )
        return (
            arrays_to_datapoints(freq, corrected11),
            arrays_to_datapoints(freq, corrected21),
        )

    @staticmethod
    def _calibrate_channel(
        calibration, raw_data: list[Datapoint]
    ) -> list[Datapoint]:
        """Correct a single channel (S11) when no S21 samples were read."""
        if not calibration.isValid1Port():
            return raw_data.copy()
        freq, values = datapoints_to_arrays(raw_data)
        return arrays_to_datapoints(
            freq, calibration.correct11_bulk(freq, values)
        )

    def read_averaged_segment(
        self,
        start: int,
        stop: int,
        averages: int = 1,
        *,
        segment: int = 0,
    ) -> tuple[list[int], list[complex], list[complex]] | None:
        """Read one segment, averaging as configured.

        Returns ``None`` when the run was cancelled during averaging, so the
        caller can record a cancellation instead of partial zeros.
        """
        logger.info(
            "Reading from %d to %d. Averaging %d values", start, stop, averages
        )

        freq: list[int] = []
        values11: list[list[complex]] = []
        values21: list[list[complex]] = []

        capture = self._capture
        total_units = capture.total_units if capture else averages

        for i in range(averages):
            if self._cancel.is_set():
                logger.debug("Stopping averaging as signalled.")
                if averages == 1:
                    break
                logger.warning("Stop during average. Discarding sweep result.")
                return None
            if i:
                self._transition(SweepState.AVERAGING)
            self._emit_progress(
                segment=segment,
                average=i,
                message=f"Average {i + 1} of {averages}",
            )
            logger.debug("Reading average no %d / %d", i + 1, averages)
            retries = RETRIES_RECONNECT
            tmp_11: list[complex] = []
            tmp_21: list[complex] = []
            while retries and not tmp_11:
                if self._cancel.is_set():
                    return None
                if retries < RETRIES_RECONNECT:
                    attempt = RETRIES_RECONNECT - retries + 1
                    logger.warning("retry readSegment(%s,%s)", start, stop)
                    self._retry_count += 1
                    self._retry_reason = (
                        f"No valid data for segment {segment + 1}"
                    )
                    self._transition(SweepState.RETRYING)
                    self._emit_progress(
                        segment=segment,
                        average=i,
                        message=self._retry_reason,
                    )
                    if not self._sleep(0.5):
                        return None
                    self._transition(SweepState.MEASURING)
                    logger.debug("Retry attempt %d", attempt)
                retries -= 1
                freq, tmp_11, tmp_21 = self.read_segment(start, stop)

            if not tmp_11:
                reason = self._segment_error or "no samples were returned"
                raise OSError(
                    f"Invalid data during sweep (segment {segment + 1}):"
                    f" {reason}"
                )

            values11.append(tmp_11)
            values21.append(tmp_21)
            completed = (
                (segment * (capture.averages if capture else averages)) + i + 1
            )
            self._emit_progress(
                segment=segment,
                average=i + 1,
                completed_units=min(completed, total_units),
                message=f"Segment {segment + 1} of "
                f"{capture.segments if capture else 1}",
            )

        if not values11:
            raise OSError("Invalid data during sweep")

        truncates = capture.truncates if capture else 0
        if truncates > 0 and averages > 1:
            logger.debug("Truncating %d values by %d", len(values11), truncates)
            values11 = truncate(values11, truncates)
            if values21 and values21[0]:
                values21 = truncate(values21, truncates)

        logger.debug("Averaging %d values", len(values11[0]))
        averaged21 = (
            np.average(values21, axis=0).tolist()
            if values21 and values21[0]
            else []
        )
        return (
            freq,
            np.average(values11, axis=0).tolist(),
            averaged21,
        )

    def _supported_channels(self) -> tuple[str, ...]:
        """Channels the connected device declares that it can provide.

        A device that exposes no such declaration is assumed to provide
        both channels, which is what every analytic scenario and ordinary
        VNA does.  A one-port replay source declares S11 only, so no S21 is
        ever read and none is invented.
        """
        channels = getattr(self.app.vna, "supported_channels", None)
        if not channels:
            return ("data 0", "data 1")
        return tuple(channels)

    def read_segment(
        self, start: int, stop: int
    ) -> tuple[list[int], list[complex], list[complex]]:
        logger.debug("Setting sweep range to %d to %d", start, stop)
        self.app.vna.setSweep(start, stop)

        frequencies = self.app.vna.read_frequencies()
        logger.debug("Read %s frequencies", len(frequencies))
        expected = len(frequencies)
        supported = self._supported_channels()

        # Every channel the device declares must deliver real samples, so a
        # device that advertises S21 can never be published as a one-port
        # measurement just because the read came back empty.
        values11 = self.read_data("data 0")
        values21 = self.read_data("data 1") if "data 1" in supported else []

        problem = self._segment_data_problem(
            expected, values11, values21, supported
        )
        if problem is not None:
            logger.info("No valid data during this run: %s", problem)
            self._segment_error = problem
            frequencies = []
            values11 = values21 = []
        else:
            self._segment_error = ""
        return frequencies, values11, values21

    @staticmethod
    def _segment_data_problem(
        expected: int,
        values11: list[complex],
        values21: list[complex],
        supported: tuple[str, ...],
    ) -> str | None:
        """Describe why a segment is unusable, or ``None`` when it is fine.

        Every channel the device declares must deliver exactly one sample
        per requested frequency.  An absent channel is only acceptable when
        the device does not declare it at all (a genuine one-port source);
        a declared read that comes back empty or short is a device fault,
        never a one-port measurement.
        """
        if not expected:
            return "the device returned no sweep frequencies"
        if len(values11) != expected:
            return (
                f"the S11 channel returned {len(values11)} samples for"
                f" {expected} requested frequencies"
            )
        if "data 1" not in supported:
            return None
        if len(values21) != expected:
            return (
                f"the device declares an S21 channel but it returned"
                f" {len(values21)} samples for {expected} requested"
                " frequencies"
            )
        return None

    def read_data(self, data) -> list[complex]:
        logger.debug("Reading %s", data)

        vna: "VNA" = self.app.vna  # shortcut to device
        retries = RETRIES_MAX
        while retries:
            if self._cancel.is_set():
                raise OSError("Sweep cancelled during read")
            retries -= 1
            try:
                result = vna.readValues(data)
                logger.debug("Read %d values", len(result))
                if vna.validateInput and any(
                    abs(v) > VALUE_MAX for v in result
                ):
                    logger.error("Got a non plausible data: (%s)", data)
                else:
                    return result
            except ValueError as exc:
                logger.exception(
                    "An exception occurred reading %s: %s", data, exc
                )
            logger.error("Re-reading %s", data)
            self._retry_count += 1
            self._retry_reason = f"Re-reading {data}"
            self._transition(SweepState.RETRYING)
            if not self._sleep(0.2):
                raise OSError("Sweep cancelled during retry")
            self._transition(SweepState.MEASURING)
            vna.reconnect()

        logger.critical(
            "Tried and failed to read %s %s times. Giving up.",
            data,
            RETRIES_MAX,
        )
        raise OSError(
            f"Failed reading {data} {RETRIES_MAX} times.\n"
            f"Data outside expected valid ranges,"
            f" or in an unexpected format.\n\n"
            f"You can disable data validation on the"
            f"device settings screen."
        )

    def gui_error(self, message: str) -> None:
        self.error_message = message
        self.signals.sweep_error.emit()
