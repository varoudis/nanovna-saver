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
"""Cross-feature regressions: replay acquisition + limits + reports.

The individual features are covered by their own suites.  These tests cover
only where they meet, because that is where an integration regression can
hide behind two passing suites:

* a real one-port replay file is acquired by the normal worker while a valid
  two-port calibration and a finite offset delay are active, and the
  resulting snapshot - not a synthetic stand-in - is what a session and a
  report carry;
* limit evaluation reads the capture's own recorded grid, so a complete
  segmented or logarithmic replay passes and a capture with a dropped sample
  is ``UNAVAILABLE`` rather than ``PASS``;
* limit checking never invents the channel a one-port source does not have;
* a built report stays frozen when the active limit set or a later
  acquisition changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest

from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.Hardware.ReplayScenario import (
    ReplayScenario,
    load_replay_file,
)
from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Limits import (
    LimitRule,
    LimitSegment,
    LimitSet,
    Metric,
    Verdict,
    evaluate_rule,
)
from NanoVNASaver.Measurement import (
    CompletionState,
    CorrectionSource,
    CorrectionState,
    compute_calibration_identity,
    snapshot_from_datapoints,
)
from NanoVNASaver.Reports import (
    build_report,
    csv_summary,
    limit_plots,
    provenance_labels,
    render_html,
)
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Session import (
    document_snapshots,
    load_session,
    make_document,
    save_session,
)
from NanoVNASaver.Settings.Sweep import Properties, Sweep
from NanoVNASaver.SweepState import SweepState
from NanoVNASaver.SweepWorker import SweepWorker
from NanoVNASaver.Traces import TraceCollection

CALIBRATION = "tests/data/test_2port_long.cal"
START = 200_000_000
STOP = 290_000_000
OFFSET_DELAY = 33e-12

ONE_PORT = (
    "! recorded antenna sweep\n"
    "# HZ S RI R 50\n"
    f"{START} 0.30 -0.45\n"
    f"{START + 50_000_000} 0.20 0.30\n"
    f"{STOP} 0.45 -0.10\n"
)


@dataclass
class App:
    vna: object
    sweep: Sweep
    calibration: Calibration


@dataclass
class Collected:
    """Real Qt signals, collected with direct connections."""

    samples: list = field(default_factory=list)
    terminals: list = field(default_factory=list)
    errors: int = 0


def _collect(worker) -> Collected:
    collected = Collected()
    worker.signals.sample_ready.connect(collected.samples.append)
    worker.signals.run_finished.connect(collected.terminals.append)
    worker.signals.sweep_error.connect(
        lambda: setattr(collected, "errors", collected.errors + 1)
    )
    return collected


def _two_port_calibration() -> Calibration:
    calibration = Calibration()
    calibration.load(CALIBRATION)
    calibration.calc_corrections()
    assert calibration.isValid1Port() and calibration.isValid2Port()
    return calibration


def _s1p(tmp_path):
    path = tmp_path / "recorded.s1p"
    path.write_text(ONE_PORT, encoding="utf-8")
    return load_replay_file(str(path))


def _dense_s2p(tmp_path, name="dense.s2p"):
    path = tmp_path / name
    lines = ["# HZ S RI R 50"]
    for freq in range(START, STOP + 10_000_001, 100_000):
        lines.append(f"{freq} 0.20 -0.10 0.95 0.05 0.02 0.01 0 0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_replay_file(str(path))


def _acquire(source, sweep, calibration=None, offset_delay=0.0):
    vna = make_simulated_vna(ReplayScenario(source))
    vna.datapoints = sweep.points
    worker = SweepWorker(
        App(vna=vna, sweep=sweep, calibration=calibration or Calibration())
    )
    worker.offsetDelay = offset_delay
    collected = _collect(worker)
    worker.run_once()
    assert worker.last_terminal_state is SweepState.COMPLETED
    return worker, collected


def _loss_rule(maximum: float = 1.0, stop: int = STOP) -> LimitRule:
    return LimitRule(
        Metric.S21_INSERTION_LOSS_DB,
        (LimitSegment(START, stop, maximum=maximum),),
        id="through",
    )


def _match_rule(maximum: float = 2.0, stop: int = STOP) -> LimitRule:
    return LimitRule(
        Metric.S11_VSWR,
        (LimitSegment(START, stop, maximum=maximum),),
        id="match",
    )


# ----------------------------------------------------------------------
# One-port replay through the normal worker, with a real calibration
# ----------------------------------------------------------------------
class TestReplayAcquisitionWithCalibration:
    @staticmethod
    def test_a_one_port_replay_is_corrected_and_stays_s11_only(tmp_path):
        worker, collected = _acquire(
            _s1p(tmp_path),
            Sweep(start=START, end=STOP, points=3, segments=1),
            calibration=_two_port_calibration(),
            offset_delay=OFFSET_DELAY,
        )

        snapshot = collected.samples[-1].snapshot
        assert snapshot.available == ("S11",)
        assert not snapshot.has("S21")
        # The correction really happened: raw and corrected differ, and the
        # S21 half of a two-port calibration has nothing to fabricate.
        assert snapshot.raw["S11"] != snapshot.corrected["S11"]
        assert worker.data21 == []
        assert snapshot.correction.applied is True
        assert snapshot.correction.identity
        assert snapshot.correction.offset_delay == pytest.approx(OFFSET_DELAY)

        # Replay provenance survives the correction.
        assert snapshot.simulated is True
        assert snapshot.provenance.simulator_scenario == "replay"
        assert "recorded.s1p" in snapshot.provenance.notes
        assert "S11 only" in snapshot.provenance.notes

    @staticmethod
    def test_the_snapshot_carries_the_captured_calibration_identity(tmp_path):
        calibration = _two_port_calibration()
        _worker, collected = _acquire(
            _s1p(tmp_path),
            Sweep(start=START, end=STOP, points=3, segments=1),
            calibration=calibration,
        )

        snapshot = collected.samples[-1].snapshot
        assert snapshot.correction.identity == (
            snapshot.provenance.calibration.identity
        )
        assert snapshot.provenance.calibration.identity

    @staticmethod
    def test_a_replay_session_round_trip_keeps_correction_and_raw_data(
        tmp_path,
    ):
        _worker, collected = _acquire(
            _s1p(tmp_path),
            Sweep(start=START, end=STOP, points=3, segments=1),
            calibration=_two_port_calibration(),
            offset_delay=OFFSET_DELAY,
        )
        acquired = collected.samples[-1].snapshot

        filename = str(tmp_path / "replay.nvsession")
        save_session(make_document([acquired]), filename)
        restored = document_snapshots(load_session(filename))["trace-0"]

        assert restored.to_dict() == acquired.to_dict()
        assert restored.available == ("S11",)
        assert restored.raw["S11"] != restored.corrected["S11"]
        assert restored.correction.identity == acquired.correction.identity
        assert restored.correction.offset_delay == pytest.approx(OFFSET_DELAY)
        assert restored.simulated is True
        assert "recorded.s1p" in restored.provenance.notes

    @staticmethod
    def test_frozen_reports_keep_replay_provenance_and_s11_correction(
        tmp_path,
    ):
        _worker, collected = _acquire(
            _s1p(tmp_path),
            Sweep(start=START, end=STOP, points=3, segments=1),
            calibration=_two_port_calibration(),
            offset_delay=OFFSET_DELAY,
        )
        snapshot = collected.samples[-1].snapshot
        collection = TraceCollection()
        trace = collection.add(snapshot, name="Replayed")

        model = build_report("Replay", [trace], limits=None)
        labels = provenance_labels(model.traces[0])
        assert "SIMULATED (replay)" in labels
        assert "one-port data (no S21)" in labels
        assert any(label.startswith("corrected (") for label in labels)

        document = render_html(model)
        assert "SIMULATED (replay)" in document
        assert "recorded.s1p" in document
        # The applied delay is reported, not left as "identity unknown".
        assert "Correction: applied from application, calibration" in document
        assert "identity unknown" not in document


# ----------------------------------------------------------------------
class TestReplayLimits:
    @staticmethod
    def test_a_one_port_capture_never_satisfies_an_s21_limit(tmp_path):
        _worker, collected = _acquire(
            _s1p(tmp_path),
            Sweep(start=START, end=STOP, points=3, segments=1),
        )
        snapshot = collected.samples[-1].snapshot

        result = evaluate_rule(_loss_rule(maximum=1000.0), snapshot)
        assert result.verdict is Verdict.UNAVAILABLE
        assert "no S21 channel" in result.reason

        match = evaluate_rule(_match_rule(maximum=10.0), snapshot)
        assert match.verdict is Verdict.PASS, match.reason

    @staticmethod
    def test_a_report_over_a_one_port_replay_is_never_a_pass(tmp_path):
        _worker, collected = _acquire(
            _s1p(tmp_path),
            Sweep(start=START, end=STOP, points=3, segments=1),
        )
        collection = TraceCollection()
        trace = collection.add(collected.samples[-1].snapshot, name="Replayed")
        model = build_report(
            "Replay limits",
            [trace],
            limits=LimitSet(
                "bench",
                (_match_rule(maximum=10.0), _loss_rule(maximum=1000.0)),
            ),
        )

        assert model.verdict is Verdict.UNAVAILABLE
        document = render_html(model)
        assert "UNAVAILABLE" in document
        assert "no S21 channel in this trace" in document
        rows = [
            row
            for row in csv_summary(model).splitlines()
            if row.startswith("trace-1,")
        ]
        assert any("limit:through,unavailable" in row for row in rows)
        assert not any("limit:through,pass" in row for row in rows)

    @pytest.mark.parametrize("logarithmic", [False, True])
    def test_a_complete_segmented_replay_capture_satisfies_its_grid(
        self, tmp_path, logarithmic
    ):
        sweep = Sweep(START, STOP, 51, 3, Properties(logarithmic=logarithmic))
        _worker, collected = _acquire(_dense_s2p(tmp_path), sweep)
        snapshot = collected.samples[-1].snapshot

        declared = snapshot.provenance.sweep.declared_frequencies()
        captured = tuple(dp.freq for dp in snapshot.channel("S11"))
        assert len(declared) == 51 * 3
        # The captured grid is what the recorded settings declare; limit
        # completeness is judged against that, not an assumed uniform grid.
        assert len(captured) == len(declared)

        # The limit band must lie inside the captured grid: a segmented
        # linear sweep rounds its last step down, so the rule stops at the
        # final captured sample instead of the configured stop.
        last = max(dp.freq for dp in snapshot.channel("S11"))
        result = evaluate_rule(_loss_rule(maximum=1.0, stop=last), snapshot)
        assert result.verdict is Verdict.PASS, result.reason

    @staticmethod
    def test_a_dropped_replay_sample_is_detected_from_the_recorded_grid(
        tmp_path,
    ):
        sweep = Sweep(START, STOP, 51, 3)
        _worker, collected = _acquire(_dense_s2p(tmp_path), sweep)
        snapshot = collected.samples[-1].snapshot

        dropped = snapshot.channel("S11")[20].freq
        s11 = [dp for dp in snapshot.channel("S11") if dp.freq != dropped]
        s21 = [dp for dp in snapshot.channel("S21") if dp.freq != dropped]
        damaged = snapshot_from_datapoints(
            s11,
            s21,
            sweep=sweep,
            completion=snapshot.provenance.completion,
        )

        result = evaluate_rule(
            _loss_rule(maximum=1.0, stop=max(dp.freq for dp in s11)), damaged
        )
        assert result.verdict is Verdict.UNAVAILABLE
        assert f"{dropped} Hz" in result.reason
        assert "missing 1 of" in result.reason

    @staticmethod
    def test_limit_bounds_and_measurement_share_one_axis(tmp_path):
        _worker, collected = _acquire(
            _dense_s2p(tmp_path),
            Sweep(start=START, end=STOP, points=51, segments=1),
        )
        collection = TraceCollection()
        trace = collection.add(collected.samples[-1].snapshot, name="Replayed")
        model = build_report(
            "Overlay", [trace], limits=LimitSet("bench", (_loss_rule(1.0),))
        )

        plots = limit_plots(model)
        assert [title for title, _svg in plots] == [
            "S21 insertion loss (dB, positive for loss) with limits"
        ]
        svg = plots[0][1]
        assert "through maximum" in svg
        # One SVG therefore one axis pair, for bounds and measurement.
        assert svg.count("<svg") == 1


# ----------------------------------------------------------------------
# Freezing
# ----------------------------------------------------------------------
class TestReplayReportFreezing:
    @staticmethod
    def test_a_later_limit_change_cannot_alter_a_built_report(tmp_path):
        _worker, collected = _acquire(
            _dense_s2p(tmp_path),
            Sweep(start=START, end=STOP, points=51, segments=1),
        )
        collection = TraceCollection()
        trace = collection.add(collected.samples[-1].snapshot, name="Replayed")
        limits = LimitSet("bench", (_loss_rule(maximum=0.1),))
        model = build_report("Frozen", [trace], limits=limits)
        assert model.verdict is Verdict.FAIL

        # A later active set - the next one the user selects - is a new
        # object; the frozen report keeps the set it was built with.
        subsequent = LimitSet("bench", (_loss_rule(maximum=1000.0),))
        assert subsequent is not limits
        assert model.limit_set is limits
        assert model.verdict is Verdict.FAIL
        assert model.traces[0].limits[0].verdict is Verdict.FAIL
        assert "limit:through,fail" in csv_summary(model)

    @staticmethod
    def test_a_later_acquisition_cannot_alter_a_built_report(tmp_path):
        source = _dense_s2p(tmp_path)
        sweep = Sweep(start=START, end=STOP, points=51, segments=1)
        _worker, collected = _acquire(source, sweep)
        collection = TraceCollection()
        trace = collection.add(collected.samples[-1].snapshot, name="Replayed")
        model = build_report("Frozen", [trace], limits=None)
        before = render_html(model)

        _other, second = _acquire(
            source, Sweep(start=START, end=START + 10_000_000, points=11)
        )
        assert second.samples[-1].snapshot.channel("S11")[0].freq == START
        assert render_html(model) == before


# ----------------------------------------------------------------------
# Reporting a correction that is not a calibration
# ----------------------------------------------------------------------
class TestDelayOnlyCorrectionReporting:
    """An offset delay is a correction, not a calibration.

    A delay-only trace must keep the "no calibration" warning while still
    being described truthfully: it is phase-corrected, not raw.
    """

    @staticmethod
    def _snapshot():
        from NanoVNASaver.Measurement import CorrectionSource, CorrectionState

        return snapshot_from_datapoints(
            [
                Datapoint(freq, 0.3, -0.1)
                for freq in range(START, STOP, 10_000_000)
            ],
            None,
            source="simulated",
            simulated=True,
            correction=CorrectionState(
                applied=True,
                identity="",
                source=CorrectionSource.APPLICATION,
                offset_delay=33e-12,
            ),
        )

    @staticmethod
    def test_labels_and_html_name_the_delay_without_a_calibration():
        collection = TraceCollection()
        trace = collection.add(TestDelayOnlyCorrectionReporting._snapshot())
        model = build_report("Delay", [trace])

        assert "offset delay correction (no calibration)" in (
            provenance_labels(model.traces[0])
        )
        document = render_html(model)
        assert "offset delay only (no calibration)" in document
        assert "33 ps" in document
        assert "calibration identity unknown" not in document

    @staticmethod
    def test_the_report_window_warns_and_does_not_claim_raw_data(qt_app):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver
        from NanoVNASaver.Windows.ReportWindow import ReportWindow

        window = NanoVNASaver()
        report = None
        try:
            collection = window.trace_control.collection
            for trace_id in list(collection.ordered_ids()):
                collection.remove(trace_id)
            collection.add(
                TestDelayOnlyCorrectionReporting._snapshot(), name="Delayed"
            )
            report = ReportWindow(window)
            report.refresh()

            text = report.label_uncalibrated.text()
            assert report.label_uncalibrated.isVisibleTo(report)
            # Named, distinguished from a calibration, and never described
            # as raw: the samples carry an offset delay correction.
            assert "Delayed" in text
            assert "offset delay correction applied" in text
            assert "not a calibration" in text
            assert "Values are raw measurements" not in text
        finally:
            if report is not None:
                report.close()
                report.deleteLater()
            window.worker.quit()
            window.deleteLater()


# ----------------------------------------------------------------------
# The GUI restore path keeps the corrected data (KEEP_CORRECTED)
# ----------------------------------------------------------------------
class TestSessionRestoreKeepsCorrectedReplay:
    """A restored session must show what was measured, not the raw samples.

    The snapshot carries both halves of the correction; only the corrected
    channel may become the displayed data, and the raw one must stay
    reachable for the report's provenance.
    """

    @staticmethod
    def test_open_session_file_restores_the_corrected_channel(qt_app, tmp_path):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        _worker, collected = _acquire(
            _s1p(tmp_path),
            Sweep(start=START, end=STOP, points=3, segments=1),
            calibration=_two_port_calibration(),
            offset_delay=OFFSET_DELAY,
        )
        acquired = collected.samples[-1].snapshot
        filename = str(tmp_path / "restore.nvsession")
        save_session(make_document([acquired], title="replay"), filename)

        window = NanoVNASaver()
        try:
            assert window.open_session_file(filename)
            displayed = list(window.data.s11)
            assert displayed == list(acquired.corrected["S11"])
            assert displayed != list(acquired.raw["S11"])
            assert window.data.s21 == []

            live = window.current_snapshot()
            assert live is not None
            assert live.available == ("S11",)
            # The corrected channel is what the user sees and exports, and
            # the raw channel and correction identity survive with it.
            assert live.corrected["S11"] == acquired.corrected["S11"]
            assert live.raw["S11"] == acquired.raw["S11"]
            assert live.correction.identity == acquired.correction.identity
            assert live.simulated is True
            assert "recorded.s1p" in live.provenance.notes
        finally:
            window.worker.quit()
            window.worker.wait(2_000)
            window.deleteLater()


# ----------------------------------------------------------------------
# Capture isolation: an in-place calibration reload cannot rewrite history
# ----------------------------------------------------------------------
class TestCaptureIsolation:
    @staticmethod
    def test_reloading_the_live_calibration_cannot_change_a_captured_run(
        tmp_path,
    ):
        source = _s1p(tmp_path)
        sweep = Sweep(start=START, end=STOP, points=3, segments=1)
        calibration = _two_port_calibration()
        first, collected = _acquire(
            source, sweep, calibration=calibration, offset_delay=OFFSET_DELAY
        )
        acquired = collected.samples[-1].snapshot
        frozen = acquired.to_dict()

        # Reload and recalculate the *same* object in place: an acquisition
        # must never read the live object again.
        calibration.load(CALIBRATION)
        calibration.calc_corrections()
        reloaded_identity = compute_calibration_identity(calibration)
        assert reloaded_identity

        assert acquired.to_dict() == frozen
        assert acquired.correction.identity == reloaded_identity
        assert acquired.raw["S11"] != acquired.corrected["S11"]
        # The worker's captured state is a frozen copy, not the live object.
        assert first.capture is not None
        assert first.capture.calibration is not calibration

        # A report built from the captured snapshot is unchanged as well.
        collection = TraceCollection()
        trace = collection.add(acquired, name="Captured")
        model = build_report("Isolated", [trace])
        before = render_html(model)
        calibration.load(CALIBRATION)
        calibration.calc_corrections()
        assert render_html(model) == before


# ----------------------------------------------------------------------
# The antenna workflow on a replay device
# ----------------------------------------------------------------------
class TestAntennaWorkflowOnReplay:
    """A guided sweep on a replayed file must never invent samples.

    The replay device only answers inside the file it loaded; a wider or
    shifted span is refused by the worker's preflight before a single sample
    is read, so the workflow keeps the previous measurement instead of
    extrapolating.  Inside the recorded span the workflow measures and its
    export writes the replayed samples unchanged.
    """

    @staticmethod
    def _replay_window(tmp_path):
        path = tmp_path / "window.s1p"
        lines = ["# HZ S RI R 50"]
        for freq in range(10_000_000, 20_000_001, 100_000):
            lines.append(f"{freq} 0.30 -0.20")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return load_replay_file(str(path))

    @staticmethod
    def test_a_span_outside_the_recorded_file_publishes_nothing(tmp_path):
        source = TestAntennaWorkflowOnReplay._replay_window(tmp_path)
        vna = make_simulated_vna(ReplayScenario(source))

        wide = Sweep(start=1_000_000, end=30_000_000, points=101)
        worker = SweepWorker(
            App(vna=vna, sweep=wide, calibration=Calibration())
        )
        collected = _collect(worker)
        worker.run_once()

        assert worker.last_terminal_state is SweepState.FAILED
        assert collected.samples == []
        assert collected.errors == 1
        assert "outside the replay" in worker.error_message
        assert "10000000 Hz" in worker.error_message

    @staticmethod
    def test_an_overview_inside_the_file_measures_and_exports(tmp_path):
        source = TestAntennaWorkflowOnReplay._replay_window(tmp_path)
        sweep = Sweep(start=13_000_000, end=15_000_000, points=101)
        worker, collected = _acquire(source, sweep)
        snapshot = collected.samples[-1].snapshot

        assert snapshot.available == ("S11",)
        assert snapshot.provenance.completion is CompletionState.COMPLETE
        # Every exported sample is a recorded one, inside the file's span.
        frequencies = [dp.freq for dp in snapshot.channel("S11")]
        assert min(frequencies) >= source.start
        assert max(frequencies) <= source.stop
        assert len(frequencies) == 101

        from NanoVNASaver.AntennaTuning import find_vswr_minimum, write_s1p

        minimum = find_vswr_minimum(snapshot.channel("S11"))
        target = tmp_path / "exported.s1p"
        write_s1p(snapshot, target, minimum=minimum.minimum)
        written = target.read_text(encoding="utf-8")
        assert written.startswith("!")
        assert "SIMULATED" in written
        assert str(len(frequencies)) in written or "101" in written


class TestDelayOnlySessionRestore:
    """A delay-only measurement keeps its phase correction across a session.

    The restore path must show the corrected samples (an offset delay is a
    real correction) while still reporting honestly that no SOL calibration
    took part - through the session, the window and the report.
    """

    @staticmethod
    def test_a_delay_only_session_restores_the_corrected_samples(
        qt_app, tmp_path
    ):
        from NanoVNASaver.Calibration import correct_delay
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        raw = [
            Datapoint(freq, 0.4, -0.2)
            for freq in range(START, STOP + 1, 10_000_000)
        ]
        delayed = [
            correct_delay(point, OFFSET_DELAY, reflect=True) for point in raw
        ]
        captured = snapshot_from_datapoints(
            delayed,
            None,
            raw11=raw,
            source="device",
            correction=CorrectionState(
                applied=True,
                identity="",
                source=CorrectionSource.APPLICATION,
                offset_delay=OFFSET_DELAY,
            ),
        )
        assert captured.raw["S11"] != captured.corrected["S11"]
        assert not captured.correction.identity

        filename = str(tmp_path / "delay-only.nvsession")
        save_session(make_document([captured]), filename)
        restored = document_snapshots(load_session(filename))["trace-0"]
        # The session keeps both halves, so nothing is re-derived wrongly.
        assert restored.corrected["S11"] == captured.corrected["S11"]
        assert restored.raw["S11"] == captured.raw["S11"]
        assert restored.correction.offset_delay == pytest.approx(OFFSET_DELAY)

        window = NanoVNASaver()
        try:
            assert window.open_session_file(filename)
            displayed = list(window.data.s11)
            assert displayed == list(captured.corrected["S11"])
            assert displayed != list(captured.raw["S11"])

            collection = TraceCollection()
            trace = collection.add(window.current_snapshot(), name="Delay only")
            model = build_report("Delay only", [trace])
            labels = provenance_labels(model.traces[0])
            assert "offset delay correction (no calibration)" in labels
            assert "corrected (device)" not in labels
            assert "Correction: offset delay only (no calibration)" in (
                render_html(model)
            )
        finally:
            window.worker.quit()
            window.worker.wait(2_000)
            window.deleteLater()


class TestMeasuredDataAcrossFeatures:
    """Data measured by one feature is what the others must report on.

    A replayed sweep is corrected, kept as a trace and then judged by a limit
    set; the limits window and the report must describe the same samples the
    antenna workflow would export.
    """

    @staticmethod
    def test_a_limits_report_describes_the_antennas_measured_samples(
        tmp_path,
    ):
        from NanoVNASaver.AntennaTuning import find_vswr_minimum

        source = TestAntennaWorkflowOnReplay._replay_window(tmp_path)
        sweep = Sweep(start=12_000_000, end=16_000_000, points=101)
        _worker, collected = _acquire(source, sweep)
        snapshot = collected.samples[-1].snapshot

        found = find_vswr_minimum(snapshot.channel("S11"))
        assert found.minimum is not None
        minimum = found.minimum
        frequencies = [dp.freq for dp in snapshot.channel("S11")]
        collection = TraceCollection()
        trace = collection.add(snapshot, name="Antenna")
        rule = LimitRule(
            Metric.S11_VSWR,
            (
                LimitSegment(
                    min(frequencies), max(frequencies), maximum=minimum.vswr
                ),
            ),
            id="resonance",
        )
        model = build_report(
            "Antenna limits", [trace], limits=LimitSet("bench", (rule,))
        )

        # The measured minimum sits exactly on the bound, so the verdict is a
        # pass over the very samples the workflow reports and would export.
        assert model.verdict is Verdict.PASS, model.verdict
        evaluation = evaluate_rule(rule, snapshot)
        assert evaluation.worst_value == pytest.approx(minimum.vswr, rel=1e-12)
        assert evaluation.worst_frequency == minimum.freq
        assert "limit:resonance,pass" in csv_summary(model)
        assert "SIMULATED (replay)" in render_html(model)


class TestAntennaWorkflowOnReplayDevice:
    """The guided workflow driven end-to-end by real queued QThreads.

    The unit-level seams are covered above; this drives the actual window
    against a replayed file so the token machinery, the adopted minimum and
    the export all run through production signal delivery.
    """

    @staticmethod
    def _pump(app, tuning, timeout=20.0):
        from PySide6 import QtCore

        deadline = time.monotonic() + timeout
        while app.worker.isRunning() or tuning.pending_phase() is not None:
            assert time.monotonic() < deadline, "the sweep thread hung"
            QtCore.QCoreApplication.processEvents()
        end = time.monotonic() + 0.08
        while time.monotonic() < end:
            QtCore.QCoreApplication.processEvents()

    @staticmethod
    def _replay_resonator(tmp_path):
        """A recorded antenna dip at 14.1 MHz, inside the file's span."""
        path = tmp_path / "resonator.s1p"
        lines = ["# HZ S RI R 50"]
        for freq in range(13_000_000, 15_200_001, 20_000):
            # Deepest match at 14.1 MHz, reflection rising either side.
            distance = abs(freq - 14_100_000) / 1_100_000
            magnitude = 0.05 + 0.7 * min(distance, 1.0)
            lines.append(f"{freq} {magnitude:.6f} 0.0")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return load_replay_file(str(path))

    @staticmethod
    def test_a_real_threaded_overview_adopts_measures_and_exports(
        qt_app, tmp_path
    ):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        window = NanoVNASaver()
        try:
            window.vna = make_simulated_vna(
                ReplayScenario(
                    TestAntennaWorkflowOnReplayDevice._replay_resonator(
                        tmp_path
                    )
                )
            )
            window.serial_control.connected.emit(True)
            tuning = window.windows["antenna_tuning"]
            tuning._invalidate_pending()
            tuning._drop_result()
            tuning.refresh_device_limits()
            tuning.overview_start.setText("13MHz")
            tuning.overview_stop.setText("15MHz")
            index = tuning.overview_points.findData(101)
            assert index >= 0
            tuning.overview_points.setCurrentIndex(index)
            tuning.refresh()

            assert tuning.start_overview()
            token = tuning._pending.token
            assert token is not None
            TestAntennaWorkflowOnReplayDevice._pump(window, tuning)

            # The workflow adopted its own run, not a stale terminal.
            assert tuning.pending_phase() is None
            assert not window.worker.isRunning()
            snapshot = tuning.completed_snapshot()
            assert snapshot is not None
            assert snapshot.simulated is True

            found = tuning.measured_minimum()
            assert found is not None
            # The recorded dip is at 14.1 MHz; the adopted minimum must be the
            # measured sample nearest it, not an invented curve fit.
            assert abs(found.freq - 14_100_000) <= 40_000
            samples = snapshot.channel("S11")
            at_minimum = next(dp for dp in samples if dp.freq == found.freq)
            assert at_minimum.z == pytest.approx(found.s11, abs=1e-12)
            assert found.vswr == pytest.approx(
                (1 + abs(found.s11)) / (1 - abs(found.s11)), rel=1e-9
            )
            assert "Min VSWR" in tuning.label_minimum.text()

            target = tmp_path / "resonance.s1p"
            tuning.export_to(target)
            written = target.read_text(encoding="utf-8")
            assert "SIMULATED" in written
            # Standard positive return loss for a passive match, written as
            # the same number the workflow reported.
            comment = _return_loss_comment(written)
            assert comment, written
            reported = float(comment.rsplit("=", 1)[1].split()[0])
            assert reported > 0
            assert reported == pytest.approx(found.return_loss, abs=5e-3)
        finally:
            window.worker.quit()
            window.worker.wait(2_000)
            window.deleteLater()


def _return_loss_comment(text):
    for line in text.splitlines():
        if "return loss" in line.lower():
            return line
    return ""
