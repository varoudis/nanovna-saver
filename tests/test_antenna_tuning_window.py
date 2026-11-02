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
"""ANT-02/ANT-03: the antenna tuning window over the real sweep path.

The sweeps run through ``SweepWorker.run_once`` - the same code the worker
thread executes - so data publication, snapshot provenance, completion and
the application's own terminal handling are all exercised. Only the thread
launch is replaced, which keeps the tests deterministic.
"""

import time

import pytest

from NanoVNASaver.AntennaTuning import AntennaExportError
from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.Hardware.Simulated import (
    FaultConfig,
    FaultMode,
    make_simulated_vna,
)
from NanoVNASaver.Measurement import CompletionState
from NanoVNASaver.SweepState import SweepState
from NanoVNASaver.Windows.AntennaTuning import AntennaTuningWindow, Phase

_APP = None


@pytest.fixture(scope="session")
def vna_app(qt_app):
    """One main window; building it is the expensive part."""
    global _APP  # noqa: PLW0603 - reused for the whole run
    if _APP is None:
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        _APP = NanoVNASaver()
    return _APP


@pytest.fixture
def app(vna_app, monkeypatch):
    """The main window with a fresh simulated device and clean workflow."""
    window = vna_app
    window.vna = make_simulated_vna("antenna")
    window.calibration = Calibration()
    window.saveData([], [])
    window._last_complete = None
    window._live_snapshot = None
    window.serial_control.connected.emit(True)
    tuning = window.windows["antenna_tuning"]
    tuning._invalidate_pending()
    tuning._minimum = None
    tuning._snapshot = None
    tuning._last_plan = None
    tuning._last_written = None
    tuning.label_minimum.setText("No measurement yet")
    tuning.refresh_device_limits()
    tuning.use_device_range()
    tuning.refresh()

    errors = []
    monkeypatch.setattr(window, "showError", errors.append)
    window.errors = errors
    return window


def run_once(app, *, cancel: bool = False):
    """Run the worker exactly as ``QThread.run`` would, minus the thread.

    ``run_once`` is the synchronous entry point the worker itself uses;
    ``run`` clears the cancellation event first, so a previous run's
    cancellation does not leak into this one.  ``cancel`` asks for the
    cancellation *after* that reset, i.e. for a genuinely interrupted sweep.
    """
    app.worker._cancel.clear()
    if cancel:
        app.worker.quit()
    app.worker.run_once()


def run_synchronously(app, monkeypatch, *, extra_runs: int = 0):
    """Replace only the thread launch with a direct run."""

    def start():
        app._sweep_control(start=True)
        run_once(app)
        for _ in range(extra_runs):
            run_once(app)

    monkeypatch.setattr(app, "sweep_start", start)


@pytest.fixture
def tuning(app):
    return app.windows["antenna_tuning"]


@pytest.fixture
def workspace(tmp_path, app):
    """A private output directory per test."""
    return tmp_path


def enter_overview(tuning, start="13MHz", stop="15MHz", points=101):
    tuning.overview_start.setText(start)
    tuning.overview_stop.setText(stop)
    index = tuning.overview_points.findData(points)
    assert index >= 0
    tuning.overview_points.setCurrentIndex(index)


class TestWindowIntegration:
    @staticmethod
    def test_the_window_is_reachable_from_the_application(app):
        assert isinstance(app.windows["antenna_tuning"], AntennaTuningWindow)

    @staticmethod
    def test_planning_never_starts_a_sweep(app, tuning, monkeypatch):
        started = []
        monkeypatch.setattr(app, "sweep_start", lambda: started.append(1))
        enter_overview(tuning)
        tuning_plan = tuning.overview_plan()
        assert (tuning_plan.start, tuning_plan.end) == (13_000_000, 15_000_000)
        assert started == []

    @staticmethod
    def test_device_range_fills_the_overview_from_the_device(app, tuning):
        app.vna.setSweep(1_000_000, 60_000_000)
        app.sweep_control.set_start(1_000_000)
        app.sweep_control.set_end(60_000_000)
        tuning.refresh_device_limits()
        tuning.use_device_range()
        assert tuning.overview_plan().start == 1_000_000
        assert tuning.overview_plan().end == 60_000_000


class TestOverviewWorkflow:
    @staticmethod
    def test_overview_sweep_finds_the_measured_minimum(
        app, tuning, monkeypatch
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        summary = tuning.label_minimum.text()
        assert "Min VSWR" in summary
        minimum = tuning.measured_minimum()
        # The simulated antenna resonates at 14.15 MHz; the sampled minimum
        # is the nearest measured grid point, not a hardcoded index.
        assert abs(minimum.freq - 14_150_000) <= 20_000
        assert minimum.index == app.data.s11.index(
            min(app.data.s11, key=lambda dp: dp.vswr)
        )
        assert "complete" in tuning.label_status.text()

    @staticmethod
    def test_the_sweep_is_applied_to_the_normal_sweep_settings(
        app, tuning, monkeypatch
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning, "12MHz", "16MHz", points=201)
        tuning.overview_segments.setValue(2)
        assert tuning.start_overview()
        assert app.sweep_control.get_start() == 12_000_000
        assert app.sweep_control.get_end() == 16_000_000
        assert app.sweep.points == 201
        assert app.sweep.segments == 2
        assert app.vna.datapoints == 201
        assert len(app.data.s11) == 201 * 2

    @staticmethod
    def test_a_marker_is_placed_on_the_measured_minimum(
        app, tuning, monkeypatch
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        marker = app.markers[0]
        minimum = tuning.measured_minimum()
        assert marker.freq == minimum.freq
        assert marker.location == minimum.index

    @staticmethod
    def test_marker_placement_can_be_switched_off(app, tuning, monkeypatch):
        run_synchronously(app, monkeypatch)
        tuning.chk_marker.setChecked(False)
        enter_overview(tuning)
        # A marker's indexed position is what the workflow would move; its
        # reported frequency is derived from that position on every redraw,
        # so the position is what must stay put.
        marker = app.markers[0]
        marker.location = 0
        before = marker.location
        assert tuning.start_overview()
        assert marker.location == before
        assert marker.location != tuning.measured_minimum().index

    @staticmethod
    def test_the_completed_sweep_is_exportable_and_complete(
        app, tuning, monkeypatch
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        snapshot = tuning.completed_snapshot()
        assert snapshot.provenance.completion is CompletionState.COMPLETE
        assert snapshot.provenance.sweep.start == 13_000_000
        assert snapshot.simulated
        assert tuning.btn_export.isEnabled()


class TestZoomAndRepeat:
    @staticmethod
    def test_zoom_centres_a_narrow_sweep_on_the_minimum(
        app, tuning, monkeypatch
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        overview_minimum = tuning.measured_minimum()
        tuning.zoom_span.setText("200kHz")
        tuning.zoom_points.setCurrentIndex(tuning.zoom_points.findData(201))
        plan = tuning.zoom_plan()
        assert plan.start <= overview_minimum.freq <= plan.end
        assert plan.points == 201
        assert tuning.zoom_to_minimum()
        assert app.sweep_control.get_start() == plan.start
        assert app.sweep_control.get_end() == plan.end
        assert len(app.data.s11) == 201
        zoomed = tuning.measured_minimum()
        assert abs(zoomed.freq - 14_150_000) <= 1_000

    @staticmethod
    def test_repeat_keeps_the_span_and_refines_the_minimum(
        app, tuning, monkeypatch
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        tuning.zoom_span.setText("200kHz")
        assert tuning.zoom_to_minimum()
        span = (app.sweep_control.get_start(), app.sweep_control.get_end())
        first = tuning.measured_minimum()
        monkeypatch.setattr(
            app.vna.device,
            "scenario",
            _DetunedAntenna(app.vna.device.scenario, offset=40_000.0),
        )
        assert tuning.repeat_sweep()
        assert (app.sweep_control.get_start(), app.sweep_control.get_end()) == (
            span
        )
        second = tuning.measured_minimum()
        assert abs(second.freq - (first.freq + 40_000)) <= 5_000

    @staticmethod
    def test_repeat_before_any_sweep_is_refused(app, tuning):
        assert not tuning.repeat_sweep()
        assert "Nothing to repeat" in tuning.label_status.text()

    @staticmethod
    def test_an_unsafe_zoom_is_rejected_without_measuring(
        app, tuning, monkeypatch
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        sweeps = app.worker.sweep_id
        tuning.zoom_span.setText("1Hz")
        assert not tuning.zoom_to_minimum()
        assert "Cannot zoom" in tuning.label_status.text()
        assert app.worker.sweep_id == sweeps
        # The previous result is untouched.
        assert tuning.measured_minimum() is not None
        assert tuning.btn_export.isEnabled()


class TestBusyState:
    @staticmethod
    def test_controls_are_disabled_while_a_step_is_pending(
        app, tuning, monkeypatch
    ):
        # A start that never finishes: the window must not offer actions
        # that would queue a second sweep behind it.
        monkeypatch.setattr(app, "sweep_start", lambda: None)
        enter_overview(tuning)
        assert tuning.start_overview()
        assert tuning.pending_phase() == Phase.OVERVIEW
        assert not tuning.btn_overview.isEnabled()
        assert not tuning.btn_zoom.isEnabled()
        assert not tuning.btn_repeat.isEnabled()
        assert not tuning.overview_segments.isEnabled()
        assert tuning.btn_stop.isEnabled()

    @staticmethod
    def test_disconnected_device_offers_no_sweep(app, tuning):
        app.vna.disconnect()
        app.serial_control.connected.emit(False)
        tuning.refresh()
        assert not tuning.btn_overview.isEnabled()
        assert "No device connected" in tuning.label_status.text()
        assert not tuning.start_overview()


class TestCancellation:
    @staticmethod
    def test_cancel_abandons_the_pending_step(app, tuning, monkeypatch):
        started = []
        monkeypatch.setattr(app, "sweep_start", lambda: started.append(1))
        enter_overview(tuning)
        assert tuning.start_overview()
        tuning.cancel()
        assert tuning.pending_phase() is None
        assert not tuning.btn_stop.isEnabled()
        assert "cancel" in tuning.label_status.text().lower()
        assert tuning.btn_overview.isEnabled()

    @staticmethod
    def test_a_late_completion_cannot_advance_a_cancelled_workflow(
        app, tuning, monkeypatch
    ):
        started = []
        monkeypatch.setattr(app, "sweep_start", lambda: started.append(1))
        enter_overview(tuning)
        assert tuning.start_overview()
        tuning.cancel()
        # A sweep finishing late (the earlier one, or another window's).
        app.worker.signals.run_finished.emit(SweepState.COMPLETED)
        assert tuning.measured_minimum() is None
        assert tuning.completed_snapshot() is None
        assert not tuning.btn_export.isEnabled()

    @staticmethod
    def test_a_completion_of_another_span_is_not_adopted(
        app, tuning, monkeypatch
    ):
        # The workflow asks for a sweep but another sweep finishes first,
        # e.g. the user pressed Sweep with a different span.
        monkeypatch.setattr(app, "sweep_start", lambda: None)
        enter_overview(tuning)
        assert tuning.start_overview()
        app.sweep.update(20_000_000, 25_000_000, 1, 101)
        run_once(app)
        assert tuning.measured_minimum() is None
        assert "does not match" in tuning.label_status.text()
        assert not tuning.btn_export.isEnabled()

    @staticmethod
    def test_an_interrupted_sweep_never_offers_partial_data(
        app, tuning, monkeypatch
    ):
        # Run the real worker, but ask it to cancel before it measures.
        def start():
            app._sweep_control(start=True)
            run_once(app, cancel=True)

        monkeypatch.setattr(app, "sweep_start", start)
        enter_overview(tuning)
        assert tuning.start_overview()
        assert not app.data.s11
        assert tuning.measured_minimum() is None
        assert tuning.completed_snapshot() is None
        assert "no complete measurement" in tuning.label_status.text()

    @staticmethod
    def test_a_cancelled_sweep_keeps_the_previous_complete_result(
        app, tuning, monkeypatch
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        previous = tuning.measured_minimum()

        def cancelled_start():
            app._sweep_control(start=True)
            run_once(app, cancel=True)

        monkeypatch.setattr(app, "sweep_start", cancelled_start)
        assert tuning.start_overview()
        # The application restored the last complete sweep, so the previous
        # minimum is still what is on screen and stays exportable.
        assert app.data.s11
        assert tuning.measured_minimum() == previous
        assert tuning.btn_export.isEnabled()
        assert "unchanged" in tuning.label_status.text()


class TestRunOwnership:
    """A queued terminal belongs to one run, not to whatever is pending."""

    @staticmethod
    def test_an_old_queued_completion_cannot_adopt_a_restarted_step(
        app, tuning, monkeypatch
    ):
        """Cancel, restart, then deliver the old terminal.

        Regression: the cancelled step's completion is already in the event
        queue when the *same* plan is started again.  A generation counter
        (or plan equality) lets that stale terminal adopt the new pending
        step and its snapshot; a per-run token does not.
        """
        launches = []

        def start():
            app._sweep_control(start=True)
            launches.append(app.worker.request_token)

        monkeypatch.setattr(app, "sweep_start", start)
        enter_overview(tuning)
        assert tuning.start_overview()
        stale_token = app.worker._request_token
        assert stale_token is not None
        assert tuning._pending.token is stale_token
        tuning.cancel()
        # Restart the *identical* plan before the old terminal is delivered.
        assert tuning.start_overview()
        assert tuning._pending.token is not stale_token
        # Now the cancelled run's terminal finally arrives.
        app.worker.signals.run_request_finished.emit(
            SweepState.COMPLETED, stale_token
        )
        assert tuning.pending_phase() == Phase.OVERVIEW
        assert tuning.measured_minimum() is None
        assert tuning.completed_snapshot() is None
        assert not tuning.btn_export.isEnabled()

    @staticmethod
    def test_a_terminal_without_our_token_is_ignored(app, tuning, monkeypatch):
        monkeypatch.setattr(app, "sweep_start", lambda: None)
        enter_overview(tuning)
        assert tuning.start_overview()
        app.worker.signals.run_request_finished.emit(SweepState.COMPLETED, None)
        app.worker.signals.run_request_finished.emit(
            SweepState.COMPLETED, object()
        )
        assert tuning.pending_phase() == Phase.OVERVIEW
        assert tuning.completed_snapshot() is None

    @staticmethod
    def test_the_generic_terminal_signal_never_adopts_anything(
        app, tuning, monkeypatch
    ):
        # The generic signal carries no run identity, so even a COMPLETED
        # terminal on it must not advance a pending step.
        monkeypatch.setattr(app, "sweep_start", lambda: None)
        enter_overview(tuning)
        assert tuning.start_overview()
        app.worker.signals.run_finished.emit(SweepState.COMPLETED)
        assert tuning.pending_phase() == Phase.OVERVIEW
        assert tuning.completed_snapshot() is None

    @staticmethod
    def test_a_real_thread_run_completes_the_workflow(app, tuning):
        """Actual ``QThread`` + queued signals, not ``run_once`` in place.

        The queued-signal path is where stale terminals actually reach the
        window, so one end-to-end run through the real thread proves the run
        token survives genuine cross-thread delivery.
        """
        from PySide6 import QtCore

        enter_overview(tuning)
        assert tuning.start_overview()
        worker = app.worker
        assert worker.wait(20_000), "the sweep thread did not finish"
        # Let the queued signals reach the GUI thread.
        for _ in range(4):
            QtCore.QCoreApplication.processEvents()
        assert tuning.pending_phase() is None
        assert tuning.measured_minimum() is not None
        assert tuning.btn_export.isEnabled()

    @staticmethod
    def test_a_real_thread_cancel_is_never_adopted_and_a_restart_works(
        app, tuning
    ):
        """Real thread: cancel, restart the same plan, old terminal arrives.

        This is the end-to-end version of the queued-terminal regression:
        the cancelled run's queued completion must not adopt the restarted
        step, and the restarted run must still complete normally.
        """
        from PySide6 import QtCore

        enter_overview(tuning)
        assert tuning.start_overview()
        first_token = tuning._pending.token
        tuning.cancel()
        assert tuning._pending is None
        assert app.worker.wait(20_000)
        for _ in range(4):
            QtCore.QCoreApplication.processEvents()
        # The cancelled run's terminal must not resurrect a pending step.
        assert tuning._pending is None
        # Restarting the identical plan must then complete on its own token.
        assert tuning.start_overview()
        assert tuning._pending.token is not first_token
        assert app.worker.wait(20_000)
        for _ in range(4):
            QtCore.QCoreApplication.processEvents()
        assert tuning.pending_phase() is None
        assert tuning.measured_minimum() is not None


class TestControlRecoveryAfterTheFirstRun:
    """Controls must survive a first run that never stored a result.

    The terminal signals are emitted from inside ``run()``, so when they
    reach the window ``worker.isRunning()`` is still true and Stop must stay
    offered.  Only the thread's own completion turns the controls back, and
    a first cancelled or failed sweep has no stored snapshot to refresh
    from - the case that used to leave Overview disabled and Stop enabled
    forever.
    """

    @staticmethod
    def _settle(app, tuning):
        """Pump the GUI loop until the real thread and its step are done.

        ``wait()`` would block this thread and starve both the timers and the
        queued terminals, so events are pumped while the thread runs - which
        is exactly what the application does during a sweep.
        """
        from PySide6 import QtCore

        deadline = time.monotonic() + 20
        try:
            while app.worker.isRunning() or tuning._pending is not None:
                assert time.monotonic() < deadline, "the sweep thread hung"
                QtCore.QCoreApplication.processEvents()
        finally:
            # Never leave a live thread behind for the next test.
            if app.worker.isRunning():
                app.worker.quit()
                app.worker.wait(20_000)
        # Let anything posted around the thread's exit be delivered.
        end = time.monotonic() + 0.08
        while time.monotonic() < end:
            QtCore.QCoreApplication.processEvents()

    @staticmethod
    def _hold_while_terminal_is_handled(app, tuning, seconds=0.15):
        """Keep the worker running while its terminal is being delivered.

        A loaded GUI thread handles a queued terminal while ``isRunning()``
        is still true; holding the worker in the emitting thread reproduces
        that deterministically instead of relying on machine speed.  The
        returning callable connects and later disconnects the holder.
        """
        from PySide6 import QtCore

        def hold(*_args):
            time.sleep(seconds)

        app.worker.signals.run_request_finished.connect(
            hold, QtCore.Qt.ConnectionType.DirectConnection
        )

        def release():
            app.worker.signals.run_request_finished.disconnect(hold)

        return release

    @staticmethod
    def _cancel_mid_run(tuning, delay_ms=20):
        """Ask to stop while the sweep is still running."""
        from PySide6 import QtCore

        timer = QtCore.QTimer(tuning)
        timer.setSingleShot(True)
        timer.timeout.connect(tuning.cancel)
        timer.start(delay_ms)
        return timer

    @staticmethod
    def test_a_cancelled_first_sweep_hands_the_controls_back(app, tuning):
        """Cancel the first sweep: no snapshot exists to recover from."""
        # A read that outlives the stop request makes the cancellation land
        # mid-run on every machine, not just a slow one.
        app.vna = make_simulated_vna(
            "antenna", fault=FaultConfig(mode=FaultMode.SLOW, slow_delay=0.02)
        )
        app.serial_control.connected.emit(True)
        tuning.refresh_device_limits()
        tuning.use_device_range()
        hold = TestControlRecoveryAfterTheFirstRun
        release = hold._hold_while_terminal_is_handled(app, tuning)
        try:
            enter_overview(tuning)
            assert tuning.start_overview()
            timer = hold._cancel_mid_run(tuning)
            hold._settle(app, tuning)
        finally:
            release()

        assert not timer.isActive()
        assert tuning._pending is None
        # The interrupted measurement is never offered.
        assert tuning.completed_snapshot() is None
        assert not app.worker.isRunning()
        assert tuning.btn_overview.isEnabled()
        assert not tuning.btn_stop.isEnabled()
        assert not tuning.btn_export.isEnabled()
        assert "Sweep stopped" in tuning.label_status.text()

    @staticmethod
    def test_a_failed_first_sweep_hands_the_controls_back(app, tuning):
        """A first run that fails must also release Overview and Stop."""
        hold = TestControlRecoveryAfterTheFirstRun
        release = hold._hold_while_terminal_is_handled(app, tuning)
        try:
            app.vna = make_simulated_vna(
                "antenna", fault=FaultConfig(mode=FaultMode.EMPTY_SAMPLES)
            )
            app.serial_control.connected.emit(True)
            enter_overview(tuning)
            assert tuning.start_overview()
            TestControlRecoveryAfterTheFirstRun._settle(app, tuning)
        finally:
            release()

        assert not app.worker.isRunning()
        assert tuning.pending_phase() is None
        assert tuning.completed_snapshot() is None
        assert tuning.btn_overview.isEnabled()
        assert not tuning.btn_stop.isEnabled()
        assert not tuning.btn_export.isEnabled()
        assert "Sweep failed" in tuning.label_status.text()

    @staticmethod
    def test_a_stop_after_the_run_completed_is_reported_truthfully(app, tuning):
        """The run may finish before the stop request lands.

        The completion is deliberately not adopted (the step was abandoned),
        but the status must not claim the sweep was stopped mid-run.
        """
        hold = TestControlRecoveryAfterTheFirstRun
        enter_overview(tuning)
        assert tuning.start_overview()
        token = tuning._pending.token
        # Abandon the step while the thread is still alive, then deliver the
        # completion it had already produced.
        tuning._abandoned_token = token
        tuning._invalidate_pending()
        app.worker.signals.run_request_finished.emit(
            SweepState.COMPLETED, token
        )
        hold._settle(app, tuning)

        assert tuning._abandoned_token is None
        assert tuning.pending_phase() is None
        # Not adopted, but never described as an interrupted sweep either.
        assert tuning.completed_snapshot() is None
        assert not tuning.btn_export.isEnabled()
        text = tuning.label_status.text()
        assert "had already completed" in text
        # Not described as an interrupted run: nothing was lost.
        assert "Sweep stopped" not in text
        assert "no complete measurement is displayed" not in text

    @staticmethod
    def test_a_newer_step_owns_the_status_over_an_abandoned_run(
        app, tuning, monkeypatch
    ):
        """A delayed old terminal must not rewrite the new step's status.

        Regression: an abandoned token kept its claim on the workflow, so an
        old COMPLETED/CANCELLED arriving after a restarted step replaced
        "Measuring ..." with a message about the previous sweep.
        """
        monkeypatch.setattr(app, "sweep_start", lambda: None)
        enter_overview(tuning)
        assert tuning.start_overview()
        stale_token = tuning._pending.token
        tuning._abandoned_token = stale_token
        tuning._invalidate_pending()
        # A new step takes over the workflow.
        assert tuning.start_overview()
        new_token = tuning._pending.token
        assert new_token is not stale_token
        status = tuning.label_status.text()
        assert status == "Measuring (overview) ..."

        for terminal in (
            SweepState.COMPLETED,
            SweepState.CANCELLED,
            SweepState.FAILED,
        ):
            app.worker.signals.run_request_finished.emit(terminal, stale_token)
            assert tuning.label_status.text() == status
            # The new step is untouched and still owns the workflow.
            assert tuning._pending is not None
            assert tuning._pending.token is new_token

    @staticmethod
    @pytest.mark.parametrize(
        "terminal",
        [SweepState.CANCELLED, SweepState.FAILED, SweepState.COMPLETED],
    )
    def test_an_older_terminal_cannot_clear_a_newer_abandoned_run(
        app, tuning, monkeypatch, terminal
    ):
        """A terminal owning neither state must leave both untouched.

        Regression: every unrelated token cleared ``_abandoned_token``, so an
        old queued terminal arriving after a *newer* run was cancelled made
        that run's own later outcome unrecognisable: its status stayed
        "Stopping ..." for good.
        """
        monkeypatch.setattr(app, "sweep_start", lambda: None)
        enter_overview(tuning)
        assert tuning.start_overview()
        older_token = tuning._pending.token
        tuning._invalidate_pending()
        # A run that was cancelled while its terminal was still owed.
        abandoned = object()
        tuning._abandoned_token = abandoned
        tuning._set_status("Stopping the sweep ...")

        app.worker.signals.run_request_finished.emit(terminal, older_token)
        # The older terminal owns neither state: no authority at all.
        assert tuning._abandoned_token is abandoned
        assert tuning.label_status.text() == "Stopping the sweep ..."
        assert tuning._pending is None

        # The abandoned run's own terminal is still handled exactly once.
        app.worker.signals.run_request_finished.emit(terminal, abandoned)
        assert tuning._abandoned_token is None
        if terminal is SweepState.COMPLETED:
            assert "had already completed" in tuning.label_status.text()
        else:
            expected = (
                "Sweep failed"
                if terminal is SweepState.FAILED
                else "Sweep stopped"
            )
            assert expected in tuning.label_status.text()
        # That token is spent: a duplicate delivery has no authority either.
        handled = tuning.label_status.text()
        app.worker.signals.run_request_finished.emit(terminal, abandoned)
        assert tuning.label_status.text() == handled
        assert tuning._abandoned_token is None
        assert tuning._pending is None


class TestEdgeCases:
    @staticmethod
    def test_an_endpoint_minimum_is_reported_as_edge_limited(
        app, tuning, monkeypatch
    ):
        # 12-13 MHz is far below the 14.15 MHz resonance, so the lowest
        # measured VSWR is the last sample of the span.
        run_synchronously(app, monkeypatch)
        enter_overview(tuning, "12MHz", "12.1MHz")
        assert tuning.start_overview()
        minimum = tuning.measured_minimum()
        assert minimum.at_edge
        assert minimum.index == len(app.data.s11) - 1
        assert "sweep edge" in tuning.label_minimum.text()

    @staticmethod
    def test_zoom_outside_the_device_range_is_clamped_not_invented(
        app, tuning, monkeypatch
    ):
        # A simulated antenna resonance pushed to the top of the device
        # range: the zoom must be clamped into range and still keep the
        # minimum inside the measured window.
        app.vna.sweep_max_freq_hz = 14_152_000
        tuning.refresh_device_limits()
        run_synchronously(app, monkeypatch)
        enter_overview(tuning, "14MHz", "14.5MHz")
        assert tuning.start_overview()
        minimum = tuning.measured_minimum()
        tuning.zoom_span.setText("1MHz")
        plan = tuning.zoom_plan()
        assert plan.end == 14_152_000
        assert plan.start <= minimum.freq <= plan.end
        assert any("device maximum" in note for note in plan.notes)
        assert tuning.zoom_to_minimum()
        assert app.sweep_control.get_end() == 14_152_000
        assert tuning.measured_minimum() is not None

    @staticmethod
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered:RuntimeWarning"
    )
    def test_non_finite_samples_are_skipped_but_the_sweep_succeeds(
        app, tuning, tmp_path, monkeypatch
    ):
        # The simulated fault makes the first sample nan/inf.  Input
        # validation is off so the samples reach the workflow, which must
        # ignore them instead of reporting a broken minimum.
        app.vna = make_simulated_vna(
            "antenna", fault=FaultConfig(mode=FaultMode.NON_FINITE)
        )
        app.vna.validateInput = False
        tuning.refresh_device_limits()
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        minimum = tuning.measured_minimum()
        assert minimum is not None
        assert minimum.index != 0
        assert "non-finite" in tuning.label_minimum.text()
        # Unusable samples cannot be written, so no export is offered.
        assert "cannot be exported" in tuning.label_status.text()
        assert not tuning.btn_export.isEnabled()
        with pytest.raises(AntennaExportError, match="not finite"):
            tuning.export_to(tmp_path / "broken.s1p")
        assert not (tmp_path / "broken.s1p").exists()

    @staticmethod
    def test_a_single_point_sweep_is_refused_as_a_plan(
        app, tuning, monkeypatch
    ):
        monkeypatch.setattr(app, "sweep_start", lambda: None)
        enter_overview(tuning)
        tuning.overview_start.setText("14MHz")
        tuning.overview_stop.setText("14MHz")
        assert not tuning.start_overview()
        assert "greater than start" in tuning.label_status.text()


class TestCalibrationPreflight:
    @staticmethod
    def test_an_uncalibrated_span_is_reported(app, tuning):
        enter_overview(tuning, "27.5MHz", "29.5MHz")
        tuning._refresh_plan_preview()
        assert "No calibration" in tuning.label_calibration.text()
        assert "uncorrected" in tuning.label_calibration.text()

    @staticmethod
    def test_coverage_of_the_loaded_calibration_is_stated(app, tuning):
        app.calibration.load("tests/data/sol_27_30.cal")
        app.calibration.calc_corrections()
        enter_overview(tuning, "27.5MHz", "29.5MHz")
        tuning._refresh_plan_preview()
        covered = tuning.label_calibration.text()
        assert "Calibrated" in covered and "in range" in covered

        enter_overview(tuning, "20MHz", "40MHz")
        tuning._refresh_plan_preview()
        partial = tuning.label_calibration.text()
        assert "not fully covered" in partial


class TestExport:
    @staticmethod
    def test_export_writes_the_completed_measurement(
        app, tuning, workspace, monkeypatch
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        target = tuning.export_to(workspace / "antenna.s1p")
        text = target.read_text(encoding="utf-8")
        assert "# HZ S RI R 50" in text
        assert "minimum VSWR" in text
        assert "completion: complete sweep" in text
        assert "SIMULATED DATA" in text
        assert len(text.splitlines()) > len(app.data.s11) - 1
        assert tuning.last_written() == target

    @staticmethod
    def test_export_refuses_to_replace_a_file(
        app, tuning, workspace, monkeypatch
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        target = workspace / "keep.s1p"
        target.write_text("! mine\n", encoding="utf-8")
        with pytest.raises(Exception, match="already exists"):
            tuning.export_to(target)
        assert target.read_text(encoding="utf-8") == "! mine\n"

    @staticmethod
    def test_a_wrong_suffix_selection_cannot_replace_another_file(
        app, tuning, workspace, monkeypatch
    ):
        """Choosing ``named.s2p`` must never destroy ``named.s1p``.

        Regression: the dialog answer was checked as given (``named.s2p``)
        while the writer rewrote the suffix to ``named.s1p`` and authorised
        replacement for it, silently destroying an unrelated existing file.
        """
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        s1p = workspace / "named.s1p"
        s2p = workspace / "named.s2p"
        s1p.write_text("KEEP S1P\n", encoding="utf-8")
        s2p.write_text("KEEP S2P\n", encoding="utf-8")
        asked = []
        monkeypatch.setattr(
            "NanoVNASaver.Windows.AntennaTuning.QtWidgets."
            "QFileDialog.getSaveFileName",
            lambda *args, **kwargs: (str(s2p), "Touchstone 1-port (*.s1p)"),
        )
        monkeypatch.setattr(
            "NanoVNASaver.Windows.AntennaTuning.confirm_overwrite",
            lambda parent, filename: asked.append(filename) or False,
        )
        assert tuning.choose_and_export() is None
        # The confirmation named the file that would actually be written.
        assert asked == [str(s1p)]
        assert s1p.read_text(encoding="utf-8") == "KEEP S1P\n"
        assert s2p.read_text(encoding="utf-8") == "KEEP S2P\n"

    @staticmethod
    def test_confirming_the_normalized_path_replaces_only_that_file(
        app, tuning, workspace, monkeypatch
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        s1p = workspace / "named.s1p"
        s2p = workspace / "named.s2p"
        s1p.write_text("OLD S1P\n", encoding="utf-8")
        s2p.write_text("KEEP S2P\n", encoding="utf-8")
        asked = []
        monkeypatch.setattr(
            "NanoVNASaver.Windows.AntennaTuning.QtWidgets."
            "QFileDialog.getSaveFileName",
            lambda *args, **kwargs: (str(s2p), "Touchstone 1-port (*.s1p)"),
        )
        monkeypatch.setattr(
            "NanoVNASaver.Windows.AntennaTuning.confirm_overwrite",
            lambda parent, filename: asked.append(filename) or True,
        )
        written = tuning.choose_and_export()
        assert written == s1p
        assert asked == [str(s1p)]
        assert "antenna tuning measurement" in s1p.read_text(encoding="utf-8")
        assert s2p.read_text(encoding="utf-8") == "KEEP S2P\n"

    @staticmethod
    def test_the_suggested_name_describes_the_measurement(
        app, tuning, monkeypatch
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        suggestion = tuning.suggested_path()
        assert suggestion.suffix == ".s1p"
        assert "MHz" in suggestion.name
        assert not suggestion.exists()

    @staticmethod
    def test_export_is_unavailable_before_a_sweep(app, tuning):
        assert not tuning.btn_export.isEnabled()
        assert tuning.completed_snapshot() is None


class _DetunedAntenna:
    """Shifts the simulated resonance, standing in for a manual adjustment."""

    def __init__(self, inner, offset: float):
        self._inner = inner
        self._offset = offset

    def __call__(self, freq: float):
        return self._inner(freq - self._offset)
