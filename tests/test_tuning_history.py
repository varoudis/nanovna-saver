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
"""TUNE-01: antenna tuning history, comparison, recall and persistence.

Every numeric expectation is derived by hand from the reflection
coefficient, never by re-running the comparison code:

* ``|S11| = 0.1``  -> VSWR ``1.1 / 0.9 = 1.2222...``, return loss ``20 dB``
* ``|S11| = 0.2``  -> VSWR ``1.2 / 0.8 = 1.5``, return loss
  ``-20 log10(0.2) = 13.979400086720377 dB``
* VSWR ``1.5 -> 1.2222...`` is ``-0.2777...``: lower is better, so the
  verdict is *improved* even though the number went down.
* Return loss ``13.9794... -> 20`` is ``+6.0206... dB``: higher is better,
  so the verdict is *improved* even though the sign is the opposite one.

The window tests run the real sweep worker (``run_once``, the body of
``QThread.run``) over the simulated antenna, so publication, snapshot
provenance, completion and the terminal signals are exercised for real;
only the thread launch is replaced.
"""

import dataclasses
import json
import math
import zipfile

import pytest

from NanoVNASaver.AntennaTuning import (
    MinimumPoint,
    SweepPlan,
    find_vswr_minimum,
)
from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.Hardware.Simulated import (
    FaultConfig,
    FaultMode,
    make_simulated_vna,
)
from NanoVNASaver.MarkerSearch import LocationKind
from NanoVNASaver.Measurement import CompletionState, snapshot_from_datapoints
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Session import (
    MANIFEST_NAME,
    SESSION_VERSION,
    SessionDocument,
    load_session,
    make_document,
    parse_session,
    save_session,
)
from NanoVNASaver.SessionWorkflow import (
    RestoreOutcome,
    SessionState,
    record_tuning_history,
    restore_tuning_history,
    summary_lines,
    write_recovery,
)
from NanoVNASaver.SweepState import SweepState
from NanoVNASaver.TuningHistory import (
    CSV_FIELDS,
    MAX_STEPS,
    TuningAppendError,
    TuningFormatError,
    TuningHistory,
    TuningStep,
    band_for,
)
from NanoVNASaver.Windows.AntennaTuning import AntennaTuningWindow

#: The model tests use a two-sample span, so the grid is exactly these two
#: frequencies: "measured" can only ever mean one of them.
START = 14_000_000
STOP = 15_000_000
#: -20 log10(0.2), to 15 places.
RL_WEAK = 13.979400086720377
#: 20 - 13.979400086720377
RL_GAIN = 6.020599913279623
#: 11/9 - 3/2
VSWR_GAIN = -0.2777777777777777


def _manifest_of(path) -> dict:
    """The manifest JSON stored in a session archive."""
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(MANIFEST_NAME))


def _rewrite_manifest(source, target, change) -> None:
    """Copy an archive, applying ``change`` to its manifest in place.

    Trace and calibration members live outside the manifest, so an "older"
    archive must be produced by editing the manifest rather than by
    rebuilding a document from it -- rebuilding would silently drop them.
    """
    with (
        zipfile.ZipFile(source) as reader,
        zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as writer,
    ):
        for item in reader.infolist():
            payload = reader.read(item)
            if item.filename == MANIFEST_NAME:
                manifest = json.loads(payload)
                change(manifest)
                payload = json.dumps(manifest).encode("utf-8")
            writer.writestr(item, payload)


def dp(freq: int, magnitude: float) -> Datapoint:
    """One sample whose reflection coefficient is ``magnitude`` (0 degrees)."""
    return Datapoint(freq, magnitude, 0.0)


def plan(start: int = START, stop: int = STOP, points: int = 2) -> SweepPlan:
    return SweepPlan(start=start, end=stop, points=points, segments=1)


def snapshot(magnitudes, *, completion=CompletionState.COMPLETE, freqs=None):
    """A snapshot over a real grid, one |S11| per sample."""
    freqs = list(freqs if freqs is not None else (START, STOP))
    return snapshot_from_datapoints(
        [
            dp(freq, magnitude)
            for freq, magnitude in zip(freqs, magnitudes, strict=True)
        ],
        None,
        name="unit",
        source="unit",
        completion=completion,
    )


def weak():
    """|S11| = 0.2 at 14 MHz -> VSWR 1.5, RL 13.9794 dB."""
    return snapshot([0.2, 0.25])


def strong():
    """|S11| = 0.1 at 15 MHz -> VSWR 1.2222..., RL 20 dB."""
    return snapshot([0.25, 0.1])


def own_minimum(snap) -> MinimumPoint:
    found = find_vswr_minimum(snap.channel("S11")).minimum
    assert found is not None
    return found


def record(history: TuningHistory, snap, *, run_id: str, **kwargs):
    """Append ``snap`` using its own minimum, i.e. the honest path."""
    kwargs.setdefault("plan", plan())
    kwargs.setdefault("minimum", None)
    minimum = kwargs.pop("minimum")
    return history.append(
        snapshot=snap,
        plan=kwargs.pop("plan"),
        minimum=minimum if minimum is not None else own_minimum(snap),
        run_id=run_id,
        **kwargs,
    )


# ----------------------------------------------------------------------
# Hand-computed anchors
# ----------------------------------------------------------------------
class TestAnchors:
    @staticmethod
    def test_the_reference_numbers_are_what_this_file_assumes():
        assert math.isclose(Datapoint(START, 0.2, 0.0).vswr, 1.5)
        assert math.isclose(-20.0 * math.log10(0.2), RL_WEAK)
        assert math.isclose(Datapoint(START, 0.1, 0.0).vswr, 11 / 9)
        assert math.isclose(-20.0 * math.log10(0.1), 20.0)
        assert math.isclose(11 / 9 - 1.5, VSWR_GAIN)
        assert math.isclose(20.0 - RL_WEAK, RL_GAIN)

    @staticmethod
    def test_band_lookup_takes_the_first_containing_inclusive_range():
        bands = [
            ("30 m", "10100000", "10150000"),
            ("20 m", 14000000, 14350000),
        ]
        assert band_for(14_000_000, bands) == "20 m"
        assert band_for(14_350_000, bands) == "20 m"
        assert band_for(14_350_001, bands) == ""
        assert band_for(14_150_000, None) == ""
        # A malformed row is skipped, never an invented name.
        assert (
            band_for(14_150_000, [("bad",), ("20 m", 14_000_000, 14_300_000)])
            == "20 m"
        )


# ----------------------------------------------------------------------
# Recording a step
# ----------------------------------------------------------------------
class TestAppend:
    @staticmethod
    def test_a_completed_step_records_the_measured_minimum():
        history = TuningHistory()
        snap = weak()
        result = record(
            history,
            snap,
            run_id="run-1",
            band="20 m",
            notes="shortened 5 mm",
            settings={"coil": "3.5 turns"},
            trace_id="trace-live",
        )
        step = result.step
        assert result.dropped == () and not result.duplicate
        assert step.frequency == START
        assert math.isclose(step.vswr, 1.5)
        assert math.isclose(step.return_loss, RL_WEAK)
        assert step.location is LocationKind.MEASURED and step.measured
        assert (step.start, step.end, step.points) == (START, STOP, 2)
        assert step.band == "20 m" and step.notes == "shortened 5 mm"
        assert step.settings == (("coil", "3.5 turns"),)
        assert step.settings_text == "coil=3.5 turns"
        assert step.snapshot_id == snap.id
        assert step.run_id == "run-1" and step.trace_id == "trace-live"
        assert step.completion is CompletionState.COMPLETE
        assert step.samples == 2

    @staticmethod
    def test_a_duplicated_completion_is_appended_once():
        history = TuningHistory()
        snap = weak()
        first = record(history, snap, run_id="run-dup")
        second = record(history, snap, run_id="run-dup")
        assert len(history) == 1
        assert second.duplicate and second.step is first.step
        # A different run is a different step, even with identical data.
        third = record(history, strong(), run_id="run-other")
        assert third.step is not first.step
        assert len(history) == 2

    @staticmethod
    @pytest.mark.parametrize(
        "completion", [CompletionState.PARTIAL, CompletionState.INTERRUPTED]
    )
    def test_an_incomplete_measurement_is_refused_with_a_reason(completion):
        history = TuningHistory()
        with pytest.raises(TuningAppendError) as excinfo:
            record(
                history,
                snapshot([0.2, 0.25], completion=completion),
                run_id="run-incomplete",
            )
        message = str(excinfo.value)
        assert "did not complete" in message and completion.value in message
        assert history.empty

    @staticmethod
    def test_an_unavailable_minimum_is_refused_not_recorded_as_zero():
        history = TuningHistory()
        reflecting = snapshot([1.0, 1.0])  # VSWR is infinite everywhere
        assert find_vswr_minimum(reflecting.channel("S11")).minimum is None
        with pytest.raises(TuningAppendError) as excinfo:
            history.append(
                snapshot=reflecting,
                plan=plan(),
                minimum=None,
                run_id="run-none",
            )
        assert "no minimum" in str(excinfo.value)
        assert history.empty

    @staticmethod
    def test_a_stale_minimum_is_refused_and_the_reason_is_stated():
        """The defect this guards: the previous run's dip filed under a new
        run whose own measurement no longer shows it."""
        history = TuningHistory()
        stale = own_minimum(weak())  # 14 MHz, VSWR 1.5
        fresh = strong()  # its own minimum is 15 MHz, VSWR 1.2222...
        with pytest.raises(TuningAppendError) as excinfo:
            history.append(
                snapshot=fresh, plan=plan(), minimum=stale, run_id="run-new"
            )
        message = str(excinfo.value)
        assert "does not belong to this measurement" in message
        assert "stale" in message
        assert history.empty
        # The measurement's own minimum is what is accepted.
        record(history, fresh, run_id="run-new")
        assert len(history) == 1

    @staticmethod
    def test_a_minimum_outside_the_planned_span_is_refused():
        history = TuningHistory()
        snap = snapshot([0.2, 0.1], freqs=(13_000_000, 16_000_000))
        with pytest.raises(TuningAppendError) as excinfo:
            history.append(
                # Plan says 14-15 MHz; the measurement is 13-16 MHz, so its
                # minimum at 16 MHz is outside what was asked for.
                snapshot=snap,
                plan=plan(),
                minimum=own_minimum(snap),
                run_id="run-outside",
            )
        assert "outside the planned span" in str(excinfo.value)
        assert history.empty

    @staticmethod
    def test_an_invented_frequency_cannot_pass_as_measured():
        history = TuningHistory()
        snap = snapshot([0.25, 0.1], freqs=(START, STOP))
        invented = MinimumPoint(
            index=0,
            freq=14_500_000,  # never measured: the grid is 14/15 MHz
            vswr=1.1,
            s11=complex(0.05, 0.0),
            return_loss=26.020599913279625,
            at_edge=False,
            span_start=START,
            span_end=STOP,
            samples=2,
        )
        with pytest.raises(TuningAppendError) as excinfo:
            history.append(
                snapshot=snap,
                plan=plan(),
                minimum=invented,
                run_id="run-invented",
            )
        assert "not one of the measured samples" in str(excinfo.value)
        assert history.empty

    @staticmethod
    def test_an_estimate_must_be_bracketed_and_named():
        history = TuningHistory()
        snap = snapshot([0.2, 0.1], freqs=(START, STOP))
        measured = own_minimum(snap)  # 15 MHz
        estimate = MinimumPoint(
            index=0,
            freq=14_500_000,
            vswr=1.15,
            s11=complex(0.0698, 0.0),
            return_loss=23.12,
            at_edge=False,
            span_start=START,
            span_end=STOP,
            samples=2,
        )
        with pytest.raises(TuningAppendError) as excinfo:
            history.append(
                snapshot=snap,
                plan=plan(),
                minimum=estimate,
                run_id="run-no-basis",
                location=LocationKind.ESTIMATED,
            )
        assert "name the measured minimum" in str(excinfo.value)

        outside = MinimumPoint(
            index=0,
            freq=13_500_000,  # below every measured sample
            vswr=1.15,
            s11=complex(0.0698, 0.0),
            return_loss=23.12,
            at_edge=False,
            span_start=13_000_000,
            span_end=STOP,
            samples=2,
        )
        with pytest.raises(TuningAppendError) as excinfo:
            history.append(
                snapshot=snap,
                plan=plan(start=13_000_000),
                minimum=outside,
                run_id="run-extrapolated",
                location=LocationKind.ESTIMATED,
                basis=measured,
            )
        assert "not between two measured samples" in str(excinfo.value)
        assert history.empty

        result = history.append(
            snapshot=snap,
            plan=plan(),
            minimum=estimate,
            run_id="run-estimate-ok",
            location=LocationKind.ESTIMATED,
            basis=measured,
        )
        assert result.step.location is LocationKind.ESTIMATED
        assert result.step.measured is False
        assert "estimated" in result.step.label

    @staticmethod
    def test_a_step_without_a_run_identity_is_refused():
        history = TuningHistory()
        with pytest.raises(TuningAppendError) as excinfo:
            record(history, weak(), run_id="")
        assert "run that measured it" in str(excinfo.value)
        assert history.empty

    @staticmethod
    def test_a_recorded_step_cannot_be_modified():
        history = TuningHistory()
        step = record(history, weak(), run_id="run-frozen").step
        with pytest.raises(dataclasses.FrozenInstanceError):
            step.vswr = 1.0


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------
class TestTrend:
    @staticmethod
    def test_each_metric_has_its_own_improvement_direction():
        history = TuningHistory()
        record(history, weak(), run_id="run-w")
        record(history, strong(), run_id="run-s")
        trend = history.trend()
        # VSWR 1.2222... - 1.5 = -0.2777...; lower is better.
        assert math.isclose(trend.vswr.delta, VSWR_GAIN)
        assert trend.vswr.improved is True
        assert trend.vswr.verdict == "improved"
        # Return loss 20 - 13.9794... = +6.0206...; higher is better.
        assert math.isclose(trend.return_loss.delta, RL_GAIN)
        assert trend.return_loss.improved is True
        assert trend.return_loss.verdict == "improved"
        assert trend.frequency_delta == STOP - START
        assert trend.is_best and trend.best.run_id == "run-s"
        text = "\n".join(trend.lines())
        assert "improved" in text and "VSWR" in text

    @staticmethod
    def test_going_backwards_is_worse_for_both_metrics():
        history = TuningHistory()
        record(history, strong(), run_id="run-first")
        record(history, weak(), run_id="run-second")
        trend = history.trend()
        # VSWR 1.5 - 1.2222... = +0.2777...; higher is worse.
        assert math.isclose(trend.vswr.delta, -VSWR_GAIN)
        assert trend.vswr.improved is False and trend.vswr.verdict == "worse"
        assert math.isclose(trend.return_loss.delta, -RL_GAIN)
        assert trend.return_loss.improved is False
        # The best step is still the first one.
        assert trend.best.run_id == "run-first"
        assert trend.is_best is False
        assert math.isclose(trend.vswr_to_best.delta, -VSWR_GAIN)
        assert "vs best so far" in "\n".join(trend.lines())

    @staticmethod
    def test_the_first_step_says_there_is_nothing_to_compare():
        history = TuningHistory()
        record(history, weak(), run_id="run-only")
        trend = history.trend()
        assert trend.previous is None
        assert trend.vswr.delta is None and trend.vswr.improved is None
        assert trend.vswr.verdict == "not comparable"
        assert trend.same_band is None
        assert "First recorded step" in trend.lines()[1]

    @staticmethod
    def test_a_band_change_is_named_not_hidden():
        history = TuningHistory()
        record(history, weak(), run_id="run-20m", band="20 m")
        record(history, strong(), run_id="run-17m", band="17 m")
        trend = history.trend()
        assert trend.same_band is False and trend.band_changed is True
        assert any("Band changed" in line for line in trend.lines())

    @staticmethod
    def test_an_infinite_metric_is_not_comparable_rather_than_best():
        history = TuningHistory()
        record(history, weak(), run_id="run-real")
        ideal = TuningStep(
            band="20 m",
            start=START,
            end=STOP,
            points=2,
            segments=1,
            logarithmic=False,
            frequency=START,
            vswr=1.0,
            return_loss=math.inf,
            location=LocationKind.MEASURED,
            at_edge=False,
            timestamp=weak().timestamp,
            snapshot_id="ideal",
            run_id="run-ideal",
        )
        history._steps.append(ideal)
        history._by_run[ideal.run_id] = ideal
        delta = history.trend().return_loss
        assert delta.value == math.inf
        assert delta.delta is None and delta.improved is None
        assert delta.verdict == "not comparable"


# ----------------------------------------------------------------------
# Retention, clearing, export, recall
# ----------------------------------------------------------------------
class TestRetentionAndExport:
    @staticmethod
    def test_retention_drops_the_oldest_and_reports_exactly_which():
        history = TuningHistory(max_steps=3)
        for index in range(5):
            record(history, weak(), run_id=f"run-{index}")
        assert len(history) == 3
        assert [step.run_id for step in history] == [
            "run-2",
            "run-3",
            "run-4",
        ]
        # An evicted run's identity is gone with it, so a late duplicate of
        # it cannot resurrect the step as the newest entry.
        assert history.step_for_run("run-0") is None
        result = record(history, strong(), run_id="run-5")
        assert [step.run_id for step in result.dropped] == ["run-2"]
        assert len(history) == 3
        assert history.step_for_run("run-5") is result.step

    @staticmethod
    def test_the_default_cap_is_bounded():
        assert MAX_STEPS == 200
        assert TuningHistory().max_steps == MAX_STEPS

    @staticmethod
    def test_clear_removes_every_step():
        history = TuningHistory()
        record(history, weak(), run_id="run-1")
        record(history, strong(), run_id="run-2")
        assert history.clear() == 2
        assert history.empty and history.steps == ()
        assert history.trend() is None and history.best() is None
        assert history.clear() == 0

    @staticmethod
    def test_json_round_trip_preserves_every_field(tmp_path):
        history = TuningHistory()
        record(
            history,
            weak(),
            run_id="run-a",
            band="20 m",
            notes="tap moved",
            settings={"tap": "12 mm", "coil": "3.5 turns"},
            trace_id="trace-live",
        )
        record(history, strong(), run_id="run-b", band="20 m")
        target = history.export_json(tmp_path / "history.json")
        restored = TuningHistory.import_json(target)
        assert restored == history
        assert restored.steps[0].notes == "tap moved"
        assert restored.steps[0].settings == (
            ("coil", "3.5 turns"),
            ("tap", "12 mm"),
        )
        assert math.isclose(restored.steps[1].return_loss, 20.0)

    @staticmethod
    def test_csv_round_trip_preserves_every_field(tmp_path):
        history = TuningHistory()
        record(history, weak(), run_id="run-csv", band="20 m", notes="a;b")
        target = history.export_csv(tmp_path / "history.csv")
        with open(target, encoding="utf-8") as stream:
            header = stream.readline().strip().split(",")
        assert header == list(CSV_FIELDS)
        restored = TuningHistory.import_csv(target)
        assert restored == history
        assert restored.steps[0].notes == "a;b"
        assert restored.steps[0].band == "20 m"

    @staticmethod
    def test_an_unknown_format_or_version_is_refused():
        payload = TuningHistory().to_dict()
        payload["version"] = SESSION_VERSION + 99
        with pytest.raises(TuningFormatError) as excinfo:
            TuningHistory.from_dict(payload)
        assert "newer than supported" in str(excinfo.value)
        with pytest.raises(TuningFormatError):
            TuningHistory.from_dict(
                {"format": "nonsense", "version": 1, "steps": []}
            )
        with pytest.raises(TuningFormatError):
            TuningHistory.from_dict(
                {
                    "format": "nanovna-saver-tuning-history",
                    "version": "1",
                    "steps": [],
                }
            )

    @staticmethod
    def test_an_absent_document_is_an_empty_history():
        assert TuningHistory.from_dict(None).empty
        assert TuningHistory.from_dict({}).empty

    @staticmethod
    def test_recall_returns_the_recorded_span_only():
        history = TuningHistory()
        snap = snapshot([0.2, 0.1], freqs=(14_100_000, 14_200_000))
        history.append(
            snapshot=snap,
            plan=plan(start=14_100_000, stop=14_200_000, points=2),
            minimum=own_minimum(snap),
            run_id="run-recall",
        )
        recall = history.recall(0)
        assert (recall.start, recall.end) == (14_100_000, 14_200_000)
        assert recall.centre == 14_150_000
        assert (recall.points, recall.segments) == (2, 1)
        assert recall.logarithmic is False
        assert recall.describe().startswith("14.1")
        # Recalling by the step object is the same lookup.
        assert history.recall(history[0]) == recall


# ----------------------------------------------------------------------
# Session persistence
# ----------------------------------------------------------------------
class TestSessionPersistence:
    @staticmethod
    def test_a_history_survives_save_and_reload(tmp_path):
        history = TuningHistory()
        record(history, weak(), run_id="run-1", band="20 m")
        record(history, strong(), run_id="run-2", band="20 m")
        document = make_document(
            [weak()],
            trace_ids=["trace-1"],
            title="tuned",
            tuning_history=history.to_dict(),
        )
        path = tmp_path / "s.nvsession"
        save_session(document, str(path))
        reloaded = load_session(str(path))
        assert reloaded.tuning_history == history.to_dict()
        assert TuningHistory.from_dict(reloaded.tuning_history) == history
        assert [trace["trace_id"] for trace in reloaded.traces] == ["trace-1"]
        assert reloaded.title == "tuned"

    @staticmethod
    def test_a_session_without_history_loads_empty_and_keeps_its_state(
        tmp_path,
    ):
        """A document written before the field existed carries no key."""
        document = make_document(
            [weak()],
            trace_ids=["trace-1"],
            title="old",
            notes="keep me",
            sweep={"start": START, "end": STOP},
        )
        manifest = document.to_manifest()
        # The key is always written (an empty history is ``{}``), so it is
        # removed from the archive's manifest to model a document from
        # before the field existed.  The trace members must be carried over
        # untouched: they live outside the manifest, and a document rebuilt
        # with ``from_manifest`` would lose them.
        assert "tuning_history" in manifest
        path = tmp_path / "old.nvsession"
        save_session(document, str(path))
        old_path = tmp_path / "nokey.nvsession"
        _rewrite_manifest(path, old_path, lambda m: m.pop("tuning_history"))

        loaded = load_session(str(old_path))
        assert "tuning_history" not in _manifest_of(old_path)
        assert loaded.tuning_history == {}
        assert TuningHistory.from_dict(loaded.tuning_history).empty
        # Nothing else was lost with the missing field.
        assert len(loaded.traces) == 1
        assert loaded.title == "old" and loaded.notes == "keep me"
        assert loaded.sweep == {"start": START, "end": STOP}

    @staticmethod
    def test_the_history_is_optional_so_the_session_version_is_unchanged():
        history = TuningHistory()
        record(history, weak(), run_id="run-m")
        manifest = make_document(
            [weak()], tuning_history=history.to_dict()
        ).to_manifest()
        assert manifest["version"] == SESSION_VERSION == 2
        assert manifest["tuning_history"]["format"] == (
            "nanovna-saver-tuning-history"
        )
        assert len(manifest["tuning_history"]["steps"]) == 1
        # A session without history writes an empty object, not a key that
        # a reader would have to guess about.
        assert make_document([weak()]).to_manifest()["tuning_history"] == {}

    @staticmethod
    def test_parsing_reports_the_step_count_and_keeps_the_history(tmp_path):
        history = TuningHistory()
        record(history, weak(), run_id="run-1")
        document = make_document(
            [weak()], trace_ids=["trace-1"], tuning_history=history.to_dict()
        )
        path = tmp_path / "s.nvsession"
        save_session(document, str(path))
        parsed = parse_session(path.read_bytes())
        assert TuningHistory.from_dict(parsed.tuning_history) == history
        lines = list(summary_lines(RestoreOutcome(document=parsed)))
        assert "Tuning steps: 1" in lines
        # Nothing recorded means no misleading line.
        empty = RestoreOutcome(document=make_document([]))
        assert not any(
            line.startswith("Tuning steps") for line in summary_lines(empty)
        )


# ----------------------------------------------------------------------
# Application plumbing (widget-free)
# ----------------------------------------------------------------------
class _FakeWindow:
    def __init__(self):
        self.history = TuningHistory()
        self.restored = None

    def source(self):
        return self.history.to_dict()

    def target(self, document):
        self.restored = document
        return len(document.get("steps") or ())


class TestApplicationPlumbing:
    @staticmethod
    def test_the_hooks_are_optional_and_absent_means_no_history():
        class Bare:
            pass

        assert record_tuning_history(Bare(), SessionDocument()) is False
        assert restore_tuning_history(Bare(), SessionDocument()) == 0

    @staticmethod
    def test_the_document_hook_records_and_restores():
        window = _FakeWindow()
        record(window.history, weak(), run_id="run-hook")
        app = type("App", (), {})()
        app.tuning_history_source = window.source
        app.tuning_history_target = window.target

        document = SessionDocument()
        assert record_tuning_history(app, document) is True
        assert document.tuning_history == window.history.to_dict()

        restored = SessionDocument(tuning_history=document.tuning_history)
        assert restore_tuning_history(app, restored) == 1
        assert window.restored == document.tuning_history

        # An older session has no history: the target still runs, with {}.
        assert restore_tuning_history(app, SessionDocument()) == 0
        assert window.restored == {}

    @staticmethod
    def test_a_recovery_snapshot_carries_the_history(tmp_path):
        history = TuningHistory()
        record(history, weak(), run_id="run-rec")
        document = make_document([weak()], tuning_history=history.to_dict())
        state = SessionState(filename=str(tmp_path / "s.nvsession"))
        target = write_recovery(document, state)
        assert (
            TuningHistory.from_dict(load_session(target).tuning_history)
            == history
        )


# ----------------------------------------------------------------------
# The window over the real sweep path
# ----------------------------------------------------------------------
_APP = None


@pytest.fixture(scope="session")
def vna_app(qt_app):
    global _APP  # noqa: PLW0603 - one main window for the whole module
    if _APP is None:
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        _APP = NanoVNASaver()
    return _APP


@pytest.fixture
def app(vna_app, monkeypatch):
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
    tuning.forget_history()
    tuning.history_notes.clear()
    tuning.history_settings.clear()
    tuning.refresh_device_limits()
    tuning.use_device_range()
    tuning.refresh()
    errors = []
    monkeypatch.setattr(window, "showError", errors.append)
    window.errors = errors
    return window


@pytest.fixture
def tuning(app):
    return app.windows["antenna_tuning"]


def run_once(app, *, cancel: bool = False):
    """Run the worker exactly as ``QThread.run`` would, minus the thread."""
    app.worker._cancel.clear()
    if cancel:
        app.worker.quit()
    app.worker.run_once()


def run_synchronously(app, monkeypatch):
    def start():
        app._sweep_control(start=True)
        run_once(app)

    monkeypatch.setattr(app, "sweep_start", start)


def enter_overview(tuning, start="13MHz", stop="15MHz", points=101):
    tuning.overview_start.setText(start)
    tuning.overview_stop.setText(stop)
    index = tuning.overview_points.findData(points)
    assert index >= 0
    tuning.overview_points.setCurrentIndex(index)


class TestWindowRecording:
    @staticmethod
    def test_a_completed_sweep_records_exactly_one_step(
        app, tuning, monkeypatch
    ):
        assert isinstance(tuning, AntennaTuningWindow)
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        assert len(tuning.history) == 1
        step = tuning.history[0]
        assert step.measured
        assert step.snapshot_id == tuning._snapshot.id
        assert (step.start, step.end, step.points) == (
            13_000_000,
            15_000_000,
            101,
        )
        assert step.trace_id == "trace-live"
        # The recorded minimum is the sample the window reports, and the
        # simulated antenna resonates near 14.15 MHz.
        minimum = tuning.measured_minimum()
        assert step.frequency == minimum.freq
        assert math.isclose(step.vswr, minimum.vswr)
        assert abs(step.frequency - 14_150_000) <= 20_000
        assert step.samples == 101
        # Recording a step is a meaningful experiment change.
        assert app.session.dirty
        assert "tuning_history" in app.session.dirty_reasons

    @staticmethod
    def test_a_cancelled_sweep_records_nothing(app, tuning, monkeypatch):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        assert len(tuning.history) == 1
        recorded = tuning.history.steps

        def cancelled_start():
            app._sweep_control(start=True)
            run_once(app, cancel=True)

        monkeypatch.setattr(app, "sweep_start", cancelled_start)
        assert tuning.start_overview()
        # The previous record is untouched: an interrupted sweep is not a step.
        assert tuning.history.steps == recorded
        assert "unchanged" in tuning.label_status.text()

    @staticmethod
    def test_a_failed_sweep_records_nothing(app, tuning, monkeypatch):
        app.vna.device.fault = FaultConfig(
            mode=FaultMode.EMPTY_SAMPLES, fail_after_samples=0
        )
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        assert tuning.history.empty
        assert app.data.s11 == []

    @staticmethod
    def test_a_cancelled_before_starting_step_records_nothing(
        app, tuning, monkeypatch
    ):
        started = []
        monkeypatch.setattr(app, "sweep_start", lambda: started.append(1))
        enter_overview(tuning)
        assert tuning.start_overview()
        tuning.cancel()
        assert tuning.pending_phase() is None
        # A queued completion of that abandoned run is not adopted.
        app.worker.signals.run_finished.emit(SweepState.COMPLETED)
        assert tuning.history.empty

    @staticmethod
    def test_a_run_from_another_window_records_nothing(
        app, tuning, monkeypatch
    ):
        """Ownership: a terminal without this step's token may not append."""
        monkeypatch.setattr(app, "sweep_start", lambda: None)
        enter_overview(tuning)
        assert tuning.start_overview()
        app.worker.signals.run_request_finished.emit(
            SweepState.COMPLETED, object()
        )
        assert tuning.pending_phase() == "overview"
        assert tuning.history.empty

    @staticmethod
    def test_duplicating_the_completion_appends_the_step_once(
        app, tuning, monkeypatch
    ):
        """A completion delivered twice must not record two steps.

        The window refuses the second delivery because the pending step is
        already adopted; the history's own run identity is the second line
        of defence and refuses it even if it is reached directly.
        """
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        assert len(tuning.history) == 1
        step = tuning.history[0]
        # Replaying the very same completion must not add a step: no pending
        # step owns the token any more.
        assert not tuning.record_step(
            snapshot=tuning._snapshot,
            plan=tuning._last_plan,
            minimum=tuning._minimum,
            run_id=step.run_id,
        )
        assert tuning.history.steps == (step,)

    @staticmethod
    def test_the_notes_and_settings_of_the_next_step_are_recorded(
        app, tuning, monkeypatch
    ):
        tuning.history_notes.setText("element shortened 5 mm")
        tuning.history_settings.setText("coil=3.5 turns; tap=12 mm")
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        step = tuning.history[0]
        assert step.notes == "element shortened 5 mm"
        assert step.settings == (("coil", "3.5 turns"), ("tap", "12 mm"))
        assert step.settings_text == "coil=3.5 turns; tap=12 mm"


class TestWindowRecall:
    @staticmethod
    def test_recall_fills_the_form_and_starts_nothing(app, tuning, monkeypatch):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        recorded = tuning.history[0]

        started = []
        monkeypatch.setattr(app, "sweep_start", lambda: started.append(1))
        # Change the form, so an accidental no-op recall would show.
        tuning.overview_start.setText("1MHz")
        tuning.overview_stop.setText("2MHz")
        tuning.overview_segments.setValue(1)
        tuning.list_history.setCurrentRow(0)
        assert tuning.recall_selected()

        recalled = tuning.overview_plan()
        assert (recalled.start, recalled.end) == (recorded.start, recorded.end)
        assert recalled.points == recorded.points
        assert recalled.segments == recorded.segments
        assert started == []
        assert not app.worker.isRunning()
        assert tuning.pending_phase() is None
        assert "Nothing was" in tuning.label_status.text()
        assert len(tuning.history) == 1

    @staticmethod
    def test_recall_without_a_selection_says_so(app, tuning):
        tuning.list_history.setCurrentRow(-1)
        assert not tuning.recall_selected()
        assert "Select a recorded step" in tuning.label_status.text()


class TestWindowHistoryControls:
    @staticmethod
    def test_clear_removes_the_steps_and_disables_the_buttons(
        app, tuning, monkeypatch
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        assert len(tuning.history) == 1
        assert tuning.btn_recall.isEnabled()
        assert tuning.clear_history() == 1
        assert tuning.history.empty
        assert tuning.list_history.count() == 0
        assert not tuning.btn_recall.isEnabled()
        assert not tuning.btn_export_history_json.isEnabled()
        assert not tuning.btn_export_history_csv.isEnabled()

    @staticmethod
    def test_export_writes_both_forms(app, tuning, monkeypatch, tmp_path):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        written = []
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            lambda *args, **kwargs: (str(tmp_path / "out"), ""),
        )
        written.append(tuning.export_history("json"))
        written.append(tuning.export_history("csv"))
        assert [path.suffix for path in written] == [".json", ".csv"]
        assert TuningHistory.import_json(written[0]) == tuning.history
        assert TuningHistory.import_csv(written[1]) == tuning.history
        # The suffix is enforced even when the dialog omits it.
        assert (tmp_path / "out.json").exists()
        assert (tmp_path / "out.csv").exists()

    @staticmethod
    def test_exporting_an_empty_history_is_refused_not_written(
        app, tuning, tmp_path
    ):
        monkeypatch_target = tmp_path / "never"
        assert tuning.history.empty
        assert tuning.export_history("json") is None
        assert not monkeypatch_target.exists()
        assert "no tuning history" in tuning.label_status.text()

    @staticmethod
    def test_the_window_installs_the_session_hooks(app, tuning):
        assert app.tuning_history_source is not None
        assert app.tuning_history_target is not None
        document = app.current_session_document("hooks")
        assert document.tuning_history == tuning.history.to_dict()
        tuning.forget_history()
        forgetting = app.current_session_document("hooks").tuning_history
        # An empty history is serialised as ``{format, version, steps: []}``,
        # not as a missing key: the key's presence is what tells a reader the
        # history was cleared rather than never recorded.
        assert TuningHistory.from_dict(forgetting).empty


class TestWindowSession:
    @staticmethod
    def test_a_saved_session_reloads_the_recorded_steps(
        app, tuning, monkeypatch, tmp_path
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        recorded = tuning.history.steps
        assert len(recorded) == 1
        path = tmp_path / "s.nvsession"
        assert app._write_session(str(path))

        tuning.forget_history()
        assert tuning.history.empty
        assert app.open_session_file(str(path))
        assert tuning.history.steps == recorded
        assert tuning.list_history.count() == 1
        assert tuning.btn_recall.isEnabled()

    @staticmethod
    def test_an_older_session_loads_with_empty_history_and_keeps_data(
        app, tuning, monkeypatch, tmp_path
    ):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        path = tmp_path / "s.nvsession"
        assert app._write_session(str(path))

        document = load_session(str(path))
        # Rewrite the manifest inside the existing archive: rebuilding a
        # document with ``from_manifest`` would drop ``trace_files`` and
        # ``traces`` (they live in separate members, not in the manifest),
        # so the archive would no longer be a valid *older* session.
        old_path = tmp_path / "old.nvsession"
        with (
            zipfile.ZipFile(path) as source,
            zipfile.ZipFile(old_path, "w", zipfile.ZIP_DEFLATED) as target,
        ):
            for item in source.infolist():
                payload = source.read(item)
                if item.filename == MANIFEST_NAME:
                    manifest = json.loads(payload)
                    manifest.pop("tuning_history")
                    assert "tuning_history" not in manifest
                    payload = json.dumps(manifest).encode("utf-8")
                target.writestr(item, payload)
        assert "tuning_history" not in json.loads(
            zipfile.ZipFile(old_path).read(MANIFEST_NAME)
        )

        assert app.open_session_file(str(old_path)), app.errors
        assert tuning.history.empty
        assert tuning.list_history.count() == 0
        # Nothing else was lost with the missing field.
        assert app.data.s11
        assert document.title == load_session(str(old_path)).title
        assert app.errors == []

    @staticmethod
    def test_a_new_session_forgets_the_history(app, tuning, monkeypatch):
        run_synchronously(app, monkeypatch)
        enter_overview(tuning)
        assert tuning.start_overview()
        assert len(tuning.history) == 1
        monkeypatch.setattr(app, "_confirm_discard", lambda: True)
        app.action_new_session()
        assert tuning.history.empty
        assert tuning.list_history.count() == 0
        # The record is plain data, so a session can always serialize it.
        assert json.dumps(tuning.history_document()) == json.dumps(
            TuningHistory().to_dict()
        )


class TestCsvSettingsRoundTrip:
    @staticmethod
    def test_multi_character_keys_and_values_survive(tmp_path):
        history = TuningHistory()
        record(history, weak(), run_id="run-settings")
        step = dataclasses.replace(
            history.steps[0], settings=(("c2", "7"), ("tuner", "3"))
        )
        history = TuningHistory(steps=[step])
        target = history.export_csv(tmp_path / "history.csv")
        restored = TuningHistory.import_csv(target)
        assert restored.steps[0].settings == (("c2", "7"), ("tuner", "3"))

    @staticmethod
    def test_a_one_character_key_survives(tmp_path):
        history = TuningHistory()
        record(history, weak(), run_id="run-settings")
        step = dataclasses.replace(history.steps[0], settings=(("c", "7"),))
        history = TuningHistory(steps=[step])
        target = history.export_csv(tmp_path / "history.csv")
        assert TuningHistory.import_csv(target).steps[0].settings == (
            ("c", "7"),
        )

    @staticmethod
    def test_a_malformed_settings_entry_is_a_format_error():
        history = TuningHistory()
        record(history, weak(), run_id="run-settings")
        entry = history.steps[0].to_dict()
        entry["settings"] = [["only-a-key"]]
        with pytest.raises(TuningFormatError):
            TuningStep.from_dict(entry)
