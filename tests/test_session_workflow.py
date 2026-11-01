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
"""SESSION-03/04: workflow, dirty state, recovery and offline restoration."""

from datetime import datetime, timedelta, timezone

import pytest

from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Measurement import snapshot_from_datapoints
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Session import (
    make_document,
    parse_session,
    save_session,
    snapshot_to_trace_dict,
)
from NanoVNASaver.SessionWorkflow import (
    INCIDENTAL_CHANGES,
    RECOVERY_INTERVAL_SECONDS,
    SessionState,
    UnsavedAction,
    decide_unsaved_action,
    discard_recovery,
    find_recovery_for,
    plan_restore,
    recovery_is_due,
    recovery_path_for,
    session_path_for,
    summary_lines,
    trace_names,
    write_recovery,
)


def _snapshot(scenario="antenna", name="trace"):
    vna = make_simulated_vna(scenario)
    vna.setSweep(13_000_000, 15_000_000)
    frequencies = vna.read_frequencies()
    convert = lambda values: [  # noqa: E731
        Datapoint(freq, value.real, value.imag)
        for freq, value in zip(frequencies, values, strict=True)
    ]
    s21 = None if scenario == "antenna" else convert(vna.readValues("data 1"))
    return snapshot_from_datapoints(
        convert(vna.readValues("data 0")),
        s21,
        name=name,
        simulated=True,
        simulator_scenario=vna.scenario_key,
        source="simulated",
    )


class TestDirtyState:
    @staticmethod
    def test_new_session_is_clean():
        state = SessionState()
        assert not state.dirty
        assert state.display_name == "Untitled"

    @staticmethod
    def test_incidental_changes_do_not_dirty():
        state = SessionState(title="T")
        for change in INCIDENTAL_CHANGES:
            assert state.mark_changed(change) is False
        assert not state.dirty

    @staticmethod
    def test_experiment_changes_dirty_and_are_recorded():
        state = SessionState(title="T")
        assert state.mark_changed("sweep") is True
        assert state.mark_changed("marker") is False  # already dirty
        assert state.dirty
        assert state.dirty_reasons == {"sweep", "marker"}
        assert state.display_name.endswith("*")

    @staticmethod
    def test_saving_clears_the_dirty_state():
        state = SessionState(title="T")
        state.mark_changed("sweep")
        state.mark_saved("/tmp/x.nvsession")
        assert not state.dirty
        assert not state.dirty_reasons
        assert state.filename == "/tmp/x.nvsession"
        assert not state.display_name.endswith("*")

    @staticmethod
    def test_display_name_falls_back_to_filename():
        state = SessionState(filename="/tmp/experiment.nvsession")
        assert state.display_name == "experiment.nvsession"

    @staticmethod
    def test_reset_returns_to_a_clean_untitled_state():
        state = SessionState(title="T", filename="/tmp/a.nvsession")
        state.mark_changed("sweep")
        state.reset(title="Untitled")
        assert not state.dirty
        assert state.filename == ""
        assert state.display_name == "Untitled"


class TestUnsavedPrompt:
    @staticmethod
    def test_clean_session_proceeds_silently():
        for answer in ("save", "discard", "cancel"):
            assert (
                decide_unsaved_action(dirty=False, can_save=True, answer=answer)
                == UnsavedAction.PROCEED.value
            )

    @staticmethod
    def test_discard_proceeds_without_saving():
        assert (
            decide_unsaved_action(dirty=True, can_save=True, answer="discard")
            == UnsavedAction.PROCEED.value
        )

    @staticmethod
    def test_save_requests_a_save_first():
        assert (
            decide_unsaved_action(dirty=True, can_save=True, answer="save")
            == UnsavedAction.SAVE_THEN_PROCEED.value
        )

    @staticmethod
    def test_cancel_stops_the_action():
        assert (
            decide_unsaved_action(dirty=True, can_save=True, answer="cancel")
            == UnsavedAction.CANCEL.value
        )

    @staticmethod
    def test_saving_without_a_document_cancels():
        assert (
            decide_unsaved_action(dirty=True, can_save=False, answer="save")
            == UnsavedAction.CANCEL.value
        )


class TestRecovery:
    @staticmethod
    def test_recovery_path_differs_from_the_saved_path():
        saved = "/tmp/experiment.nvsession"
        recovery = recovery_path_for(saved)
        assert recovery != saved
        assert recovery.endswith(".nvrecovery")
        assert "experiment" in recovery

    @staticmethod
    def test_recovery_path_without_a_saved_file_is_temporary():
        path = recovery_path_for("")
        assert path.endswith(".nvrecovery")
        assert "nanovna-unsaved" in path

    @staticmethod
    def test_recovery_is_only_due_for_dirty_sessions():
        clean = SessionState(title="T")
        assert recovery_is_due(clean) is False

    @staticmethod
    def test_recovery_waits_for_the_interval():
        state = SessionState(title="T")
        state.mark_changed("sweep")
        now = datetime.now(timezone.utc)
        state.last_recovery = now
        assert recovery_is_due(state, now) is False
        later = now + timedelta(seconds=RECOVERY_INTERVAL_SECONDS + 1)
        assert recovery_is_due(state, later) is True

    @staticmethod
    def test_recovery_never_overwrites_the_saved_session(tmp_path):
        saved = tmp_path / "experiment.nvsession"
        document = make_document([_snapshot()], title="T")
        save_session(document, str(saved))
        before = saved.read_bytes()

        state = SessionState(filename=str(saved), title="T")
        state.mark_changed("sweep")
        state.recovery_filename = str(saved)  # hostile configuration
        with pytest.raises(Exception, match="refusing"):
            write_recovery(document, state)
        assert saved.read_bytes() == before

    @staticmethod
    def test_recovery_writes_a_loadable_document(tmp_path):
        saved = str(tmp_path / "experiment.nvsession")
        document = make_document([_snapshot()], title="T")
        state = SessionState(filename=saved, title="T")
        state.mark_changed("sweep")
        target = write_recovery(document, state)
        assert target != saved
        restored = parse_session(open(target, "rb").read())
        assert restored.title == "T"

    @staticmethod
    def test_discarding_removes_the_recovery_file(tmp_path):
        saved = str(tmp_path / "experiment.nvsession")
        state = SessionState(filename=saved, title="T")
        state.mark_changed("sweep")
        target = write_recovery(make_document([_snapshot()]), state)
        assert find_recovery_for(saved) == target
        assert discard_recovery(state) is True
        assert not find_recovery_for(saved)

    @staticmethod
    def test_a_deliberate_save_survives_recovery_discard(tmp_path):
        saved = tmp_path / "experiment.nvsession"
        document = make_document([_snapshot()], title="T")
        save_session(document, str(saved))
        state = SessionState(filename=str(saved), title="T")
        state.mark_changed("sweep")
        write_recovery(document, state)
        discard_recovery(state)
        assert saved.exists()


class TestRestore:
    @staticmethod
    def test_restore_never_connects_or_sweeps():
        document = make_document([_snapshot()], title="T")
        outcome = plan_restore(document, "1.0")
        assert outcome.connected is False
        assert outcome.started_sweep is False

    @staticmethod
    def test_version_mismatch_is_reported_without_data_loss():
        snapshot = _snapshot()
        document = make_document([snapshot], title="T", app_version="0.1")
        outcome = plan_restore(document, "9.9")
        assert outcome.app_version_mismatch is True
        assert any("0.1" in line for line in outcome.warnings)
        assert len(outcome.document.traces) == 1

    @staticmethod
    def test_session_layout_takes_precedence():
        document = make_document([_snapshot()], title="T", ui={"zoom": "in"})
        outcome = plan_restore(document, "1.0")
        assert any("take precedence" in line for line in outcome.warnings)

    @staticmethod
    def test_empty_session_is_reported():
        outcome = plan_restore(make_document([], title="Empty"), "1.0")
        assert any("no traces" in line.lower() for line in outcome.warnings)

    @staticmethod
    def test_summary_lists_title_and_traces():
        outcome = plan_restore(
            make_document([_snapshot()], title="Mine"), "1.0"
        )
        lines = list(summary_lines(outcome))
        assert "Title: Mine" in lines
        assert "Traces: 1" in lines


class TestSessionRoundTrip:
    @staticmethod
    def test_four_trace_experiment_restores_identically(tmp_path):
        snapshots = [
            _snapshot("antenna", name="antenna"),
            _snapshot("filter", name="filter"),
            _snapshot("cable", name="cable"),
            _snapshot("attenuator", name="attenuator"),
        ]
        document = make_document(
            snapshots,
            title="Four traces",
            notes="Round trip test",
            app_version="0.7.0",
            sweep={"start": 13_000_000, "end": 15_000_000, "points": 101},
            charts={"s11": ["Smith Chart"], "s21": ["Gain"]},
        )
        path = tmp_path / "four.nvsession"
        save_session(document, str(path))

        restored = parse_session(path.read_bytes())
        assert restored.title == "Four traces"
        assert restored.notes == "Round trip test"
        assert trace_names(restored) == [
            "antenna",
            "filter",
            "cable",
            "attenuator",
        ]
        assert restored.sweep["points"] == 101
        assert restored.charts["s11"] == ["Smith Chart"]

        preserved = restored.traces[0]
        original = snapshot_to_trace_dict(snapshots[0], "trace-0")
        assert preserved == original

    @staticmethod
    def test_simulated_provenance_survives_a_round_trip(tmp_path):
        document = make_document([_snapshot("cable")], title="Sim")
        path = tmp_path / "sim.nvsession"
        save_session(document, str(path))
        restored = parse_session(path.read_bytes())
        provenance = restored.traces[0]["provenance"]
        assert provenance["simulated"] is True
        assert provenance["simulator_scenario"] == "cable"

    @staticmethod
    def test_one_port_and_two_port_survive_together(tmp_path):
        one_port = _snapshot("antenna")
        two_port = _snapshot("filter")
        document = make_document([one_port, two_port], title="Mixed")
        path = tmp_path / "mixed.nvsession"
        save_session(document, str(path))
        restored = parse_session(path.read_bytes())
        channels = [trace["corrected"] for trace in restored.traces]
        assert "S11" in channels[0]
        assert "S21" not in channels[0]
        assert "S21" in channels[1]


class TestFilenameSuggestion:
    @staticmethod
    def test_title_becomes_a_safe_filename():
        path = session_path_for("2m dipole / test", "/tmp")
        assert path.startswith("/tmp/")
        assert path.endswith(".nvsession")
        assert "/" not in path.split("/tmp/")[1]

    @staticmethod
    def test_empty_title_still_produces_a_name():
        assert session_path_for("", "/tmp").endswith("session.nvsession")
