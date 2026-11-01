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

from .Calibration import correct_delay
from .Hardware.VNA import VNA
from .Measurement import (
    CompletionState,
    SweepInfo,
    snapshot_from_datapoints,
)
from .RFTools import Datapoint
from .Settings.Sweep import Sweep, SweepMode
from .SweepState import (
    SweepProgress,
    SweepState,
    SweepStateMachine,
    estimate_remaining,
)

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
    run: replacing ``app.calibration`` mid-sweep cannot change the meaning
    of samples that were already acquired.
    """

    sweep_id: int
    sweep: Sweep
    calibration: object
    offset_delay: float
    averages: int
    truncates: int
    continuous: bool

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
    """One published, immutable acquisition result (SWEEP-02.2)."""

    sweep_id: int
    segment: int
    segments: int
    s11: tuple[Datapoint, ...]
    s21: tuple[Datapoint, ...]
    completion: CompletionState
    capture: SweepCapture
    snapshot: object = None


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


class SweepWorker(QThread):
    def __init__(self, app: "vna_app") -> None:
        super().__init__()
        logger.info("Initializing SweepWorker")
        self.signals: WorkerSignals = WorkerSignals()
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

    @Slot()
    def run(self) -> None:
        """QThread entry point."""
        self._cancel.clear()
        self.run_once()

    def run_once(self) -> None:
        """Synchronous entry point used by tests and headless runs."""
        self._sweep_counter += 1
        self._state_machine.reset(self._sweep_counter)
        self._last_terminal = None
        self._started_at = monotonic()
        self.error_message = ""
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
        logger.debug(
            "Sweep %s finished as %s", self.sweep_id, self._last_terminal.value
        )
        if self._last_terminal is SweepState.COMPLETED:
            self.signals.finished.emit()
        self.signals.run_finished.emit(self._last_terminal)

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

    def _capture_settings(self) -> SweepCapture:
        sweep = self.app.sweep.copy()
        self.sweep = sweep
        averages = (
            sweep.properties.averages[0]
            if sweep.properties.mode == SweepMode.AVERAGE
            else 1
        )
        return SweepCapture(
            sweep_id=self.sweep_id,
            sweep=sweep,
            calibration=self.app.calibration,
            offset_delay=self.offsetDelay,
            averages=averages,
            truncates=sweep.properties.averages[1],
            continuous=sweep.properties.mode == SweepMode.CONTINOUS,
        )

    def _run_loop(self) -> SweepState | None:
        """Acquire segments; return a terminal state when stopping early."""
        capture = self._capture
        if capture is None:
            return SweepState.FAILED
        sweep = capture.sweep
        logger.info("%d averages", capture.averages)

        while True:
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
        raw_data21 = [
            Datapoint(freq, values21[i].real, values21[i].imag)
            for i, freq in enumerate(frequencies)
        ]

        capture = self._capture
        data11, data21 = self.applyCalibration(
            raw_data11, raw_data21, capture=capture
        )
        self.data11.extend(data11)
        self.data21.extend(data21)
        self.rawData11.extend(raw_data11)
        self.rawData21.extend(raw_data21)

        simulated = bool(getattr(self.app.vna, "simulated", False))
        scenario = (
            getattr(self.app.vna, "scenario_key", "") if simulated else ""
        )
        complete = capture is not None and index + 1 >= capture.segments
        completion = (
            CompletionState.COMPLETE if complete else CompletionState.PARTIAL
        )
        sample = SweepSample(
            sweep_id=self.sweep_id,
            segment=index,
            segments=capture.segments if capture else 1,
            s11=tuple(self.data11),
            s21=tuple(self.data21),
            completion=completion,
            capture=capture,
            snapshot=snapshot_from_datapoints(
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
                source="simulated" if simulated else "",
            ),
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

    def applyCalibration(
        self,
        raw_data11: list[Datapoint],
        raw_data21: list[Datapoint],
        *,
        capture: SweepCapture | None = None,
    ) -> tuple[list[Datapoint], list[Datapoint]]:
        capture = capture or self._capture
        calibration = (
            capture.calibration if capture is not None else self.app.calibration
        )
        offset_delay = (
            capture.offset_delay if capture is not None else self.offsetDelay
        )
        data11: list[Datapoint] = []
        data21: list[Datapoint] = []

        if not calibration.isCalculated:
            data11 = raw_data11.copy()
            data21 = raw_data21.copy()
        elif calibration.isValid1Port():
            data11.extend(calibration.correct11(dp) for dp in raw_data11)
        else:
            data11 = raw_data11.copy()

        if calibration.isValid2Port():
            for counter, dp in enumerate(raw_data21):
                dp11 = raw_data11[counter]
                data21.append(calibration.correct21(dp, dp11))
        else:
            data21 = raw_data21

        if offset_delay != 0.0:
            data11 = [
                correct_delay(dp, offset_delay, reflect=True) for dp in data11
            ]
            data21 = [correct_delay(dp, offset_delay) for dp in data21]

        return data11, data21

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
                raise OSError("Invalid data during sweep")

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
            values21 = truncate(values21, truncates)

        logger.debug("Averaging %d values", len(values11[0]))
        return (
            freq,
            np.average(values11, axis=0).tolist(),
            np.average(values21, axis=0).tolist(),
        )

    def read_segment(
        self, start: int, stop: int
    ) -> tuple[list[int], list[complex], list[complex]]:
        logger.debug("Setting sweep range to %d to %d", start, stop)
        self.app.vna.setSweep(start, stop)

        frequencies = self.app.vna.read_frequencies()
        logger.debug("Read %s frequencies", len(frequencies))
        values11 = self.read_data("data 0")
        values21 = self.read_data("data 1")
        if not len(frequencies) == len(values11) == len(values21):
            logger.info("No valid data during this run")
            frequencies = []
            values11 = values21 = []
        return frequencies, values11, values21

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
