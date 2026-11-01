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
"""SESSION-01/SESSION-02: versioned, transactional session persistence."""

import io
import json
import os
import zipfile

import pytest

from NanoVNASaver.Measurement import (
    CompletionState,
    CorrectionSource,
    snapshot_from_datapoints,
)
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Session import (
    MANIFEST_NAME,
    SESSION_FORMAT,
    SESSION_SUFFIX,
    SESSION_VERSION,
    SessionError,
    SessionFormatError,
    SessionVersionError,
    build_archive,
    document_snapshots,
    load_session,
    make_document,
    parse_session,
    save_session,
    snapshot_from_trace_dict,
    snapshot_to_trace_dict,
)


def _points(freqs=(100, 200, 300), value=(0.5, 0.25)):
    return [Datapoint(freq, value[0], value[1]) for freq in freqs]


def _session(tmp_path, **kwargs):
    snapshot = snapshot_from_datapoints(
        _points(), _points(), name="antenna", source="sim"
    )
    document = make_document([snapshot], **kwargs)
    path = tmp_path / f"experiment{SESSION_SUFFIX}"
    save_session(document, str(path))
    return path, snapshot


class TestArchiveStructure:
    @staticmethod
    def test_archive_is_a_zip_with_a_versioned_manifest(tmp_path):
        path, _ = _session(tmp_path, title="My experiment", notes="notes")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            assert MANIFEST_NAME in names
            assert any(name.startswith("traces/") for name in names)
            manifest = json.loads(archive.read(MANIFEST_NAME))
        assert manifest["format"] == SESSION_FORMAT
        assert manifest["version"] == SESSION_VERSION
        assert manifest["title"] == "My experiment"

    @staticmethod
    def test_no_pickle_payload_is_ever_written(tmp_path):
        path, _ = _session(tmp_path)
        payload = path.read_bytes()
        assert b"pickle" not in payload.lower()

    @staticmethod
    def test_archive_output_is_deterministic_for_the_same_document():
        snapshot = snapshot_from_datapoints(_points(), None, name="a")
        document = make_document([snapshot], title="t")
        reference = document.created
        assert build_archive(document) == build_archive(document)
        # Rebuilding an equal document with the same timestamp matches too.
        clone = make_document([snapshot], title="t")
        clone.created = reference
        assert build_archive(clone) == build_archive(document)

    @staticmethod
    def test_embedded_calibration_text_is_stored(tmp_path):
        snapshot = snapshot_from_datapoints(_points(), None)
        document = make_document(
            [snapshot], calibration_files={"mine.cal": "! cal\n# Hz S O L\n"}
        )
        path = tmp_path / "cal.nvsession"
        save_session(document, str(path))
        loaded = load_session(str(path))
        assert loaded.calibration_files["mine.cal"].startswith("! cal")


class TestRoundTrip:
    @staticmethod
    def test_numerical_arrays_round_trip_exactly(tmp_path):
        path, original = _session(tmp_path)
        loaded = load_session(str(path))
        restored = document_snapshots(loaded)["trace-0"]
        for parameter in original.available:
            assert restored.channel(parameter) == original.channel(parameter)

    @staticmethod
    def test_multiple_traces_round_trip(tmp_path):
        snapshots = [
            snapshot_from_datapoints(
                [Datapoint(100, 0.1 * index, 0.0)], None, name=f"t{index}"
            )
            for index in range(4)
        ]
        document = make_document(
            snapshots, trace_ids=["a", "b", "c", "d"], title="four"
        )
        path = tmp_path / "four.nvsession"
        save_session(document, str(path))
        loaded = load_session(str(path))
        assert len(loaded.traces) == 4
        restored = document_snapshots(loaded)
        for index, key in enumerate(["a", "b", "c", "d"]):
            assert restored[key].channel("S11")[0].re == pytest.approx(
                0.1 * index
            )

    @staticmethod
    def test_full_two_port_channels_survive(tmp_path):
        from NanoVNASaver.Touchstone import Touchstone

        touchstone = Touchstone("./tests/data/valid.s2p")
        touchstone.load()
        from NanoVNASaver.Measurement import snapshot_from_touchstone

        snapshot = snapshot_from_touchstone(touchstone)
        path = tmp_path / "full.nvsession"
        save_session(make_document([snapshot]), str(path))
        restored = document_snapshots(load_session(str(path)))["trace-0"]
        assert restored.available == snapshot.available
        assert restored.channel("S12") == snapshot.channel("S12")
        assert restored.channel("S22") == snapshot.channel("S22")

    @staticmethod
    def test_provenance_and_simulation_flag_survive(tmp_path):
        snapshot = snapshot_from_datapoints(
            _points(),
            None,
            simulated=True,
            simulator_scenario="antenna",
            device="Simulated VNA",
            completion=CompletionState.PARTIAL,
        )
        path = tmp_path / "sim.nvsession"
        save_session(make_document([snapshot]), str(path))
        restored = document_snapshots(load_session(str(path)))["trace-0"]
        assert restored.simulated
        assert restored.provenance.simulator_scenario == "antenna"
        assert restored.provenance.device == "Simulated VNA"
        assert restored.provenance.completion is CompletionState.PARTIAL

    @staticmethod
    def test_correction_state_survives_so_it_is_not_reapplied(tmp_path):
        snapshot = snapshot_from_datapoints(_points(), None)
        corrected = snapshot.mark_corrected(
            "cal-abc",
            {"S11": _points(value=(0.1, 0.1))},
            source=CorrectionSource.APPLICATION,
        )
        path = tmp_path / "corrected.nvsession"
        save_session(make_document([corrected]), str(path))
        restored = document_snapshots(load_session(str(path)))["trace-0"]
        assert restored.correction.applied
        assert restored.correction.identity == "cal-abc"
        assert restored.channel("S11")[0].re == pytest.approx(0.1)
        assert restored.channel("S11", corrected=False)[0].re == pytest.approx(
            0.5
        )

    @staticmethod
    def test_optional_settings_defaults(tmp_path):
        path, _ = _session(tmp_path)
        loaded = load_session(str(path))
        assert loaded.markers == []
        assert loaded.charts == {}
        assert loaded.sweep == {}
        assert loaded.notes == ""

    @staticmethod
    def test_notes_and_ui_state_round_trip(tmp_path):
        path, _ = _session(
            tmp_path,
            notes="long notes\nsecond line",
            charts={"chart_00": "S11 Smith Chart"},
            markers=[{"freq": 14_000_000, "name": "M1"}],
            sweep={"start": 1_000_000, "end": 30_000_000},
        )
        loaded = load_session(str(path))
        assert loaded.notes == "long notes\nsecond line"
        assert loaded.charts["chart_00"] == "S11 Smith Chart"
        assert loaded.markers == [{"freq": 14_000_000, "name": "M1"}]
        assert loaded.sweep["end"] == 30_000_000


class TestAtomicity:
    @staticmethod
    def test_invalid_session_does_not_replace_a_good_save(tmp_path):
        path = tmp_path / "good.nvsession"
        snapshot = snapshot_from_datapoints(_points(), None, name="good")
        save_session(make_document([snapshot]), str(path))
        original = path.read_bytes()

        broken = tmp_path / "broken.nvsession"
        broken.write_bytes(b"not a zip at all")
        with pytest.raises(SessionFormatError):
            load_session(str(broken))
        # The good file is untouched by the failed load.
        assert path.read_bytes() == original

    @staticmethod
    def test_failed_replace_leaves_no_partial_destination(
        tmp_path, monkeypatch
    ):
        """A failure mid-write must not create or corrupt the destination."""
        path = tmp_path / "target.nvsession"
        snapshot = snapshot_from_datapoints(_points(), None, name="one")
        save_session(make_document([snapshot]), str(path))
        original = path.read_bytes()

        import NanoVNASaver.Session as session_module

        def explode(_src, _dst):
            raise OSError("disk full")

        monkeypatch.setattr(session_module.os, "replace", explode)
        with pytest.raises(SessionError, match="could not write"):
            save_session(
                make_document(
                    [snapshot_from_datapoints(_points(), None, name="two")]
                ),
                str(path),
            )
        monkeypatch.undo()
        # The previous save survives unchanged and no temp file is left over.
        assert path.read_bytes() == original
        assert not [
            name for name in os.listdir(tmp_path) if name.endswith(".tmp")
        ]

    @staticmethod
    def test_overwrite_replaces_destination_atomically(tmp_path):
        path = tmp_path / "target.nvsession"
        save_session(
            make_document(
                [snapshot_from_datapoints(_points(), None, name="one")]
            ),
            str(path),
        )
        save_session(
            make_document(
                [snapshot_from_datapoints(_points(), None, name="two")]
            ),
            str(path),
        )
        loaded = load_session(str(path))
        assert loaded.traces[0]["name"] == "two"


class TestValidation:
    @staticmethod
    def test_newer_version_is_rejected_with_actionable_message():
        manifest = {
            "format": SESSION_FORMAT,
            "version": SESSION_VERSION + 1,
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(MANIFEST_NAME, json.dumps(manifest))
        with pytest.raises(SessionVersionError, match="upgrade"):
            parse_session(buffer.getvalue())

    @staticmethod
    def test_wrong_format_is_rejected():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                MANIFEST_NAME, json.dumps({"format": "other", "version": 1})
            )
        with pytest.raises(SessionFormatError, match="format"):
            parse_session(buffer.getvalue())

    @staticmethod
    def test_missing_manifest_is_rejected():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("traces/x.json", "{}")
        with pytest.raises(SessionFormatError, match="manifest"):
            parse_session(buffer.getvalue())

    @staticmethod
    def test_missing_trace_member_is_rejected():
        manifest = {
            "format": SESSION_FORMAT,
            "version": SESSION_VERSION,
            "trace_files": ["traces/missing.json"],
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(MANIFEST_NAME, json.dumps(manifest))
        with pytest.raises(SessionFormatError, match="missing trace"):
            parse_session(buffer.getvalue())

    @staticmethod
    def test_malformed_trace_json_is_rejected():
        manifest = {
            "format": SESSION_FORMAT,
            "version": SESSION_VERSION,
            "trace_files": ["traces/bad.json"],
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(MANIFEST_NAME, json.dumps(manifest))
            archive.writestr("traces/bad.json", "{not json")
        with pytest.raises(SessionFormatError, match="invalid trace"):
            parse_session(buffer.getvalue())

    @staticmethod
    def test_path_traversal_member_is_rejected():
        manifest = {"format": SESSION_FORMAT, "version": SESSION_VERSION}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(MANIFEST_NAME, json.dumps(manifest))
            archive.writestr("../../etc/passwd", "x")
        with pytest.raises(SessionFormatError, match="unsafe"):
            parse_session(buffer.getvalue())

    @staticmethod
    def test_unexpected_member_is_rejected():
        manifest = {"format": SESSION_FORMAT, "version": SESSION_VERSION}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(MANIFEST_NAME, json.dumps(manifest))
            archive.writestr("scripts/run.py", "print('no')")
        with pytest.raises(SessionFormatError, match="unexpected"):
            parse_session(buffer.getvalue())

    @staticmethod
    def test_missing_calibration_member_is_rejected():
        manifest = {
            "format": SESSION_FORMAT,
            "version": SESSION_VERSION,
            "calibration_files": ["gone.cal"],
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(MANIFEST_NAME, json.dumps(manifest))
        with pytest.raises(SessionFormatError, match="calibration"):
            parse_session(buffer.getvalue())

    @staticmethod
    def test_dangling_selected_trace_is_rejected():
        manifest = {
            "format": SESSION_FORMAT,
            "version": SESSION_VERSION,
            "selected_trace": "nope",
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(MANIFEST_NAME, json.dumps(manifest))
        with pytest.raises(SessionFormatError, match="selected trace"):
            parse_session(buffer.getvalue())

    @staticmethod
    def test_unreadable_file_raises_session_error(tmp_path):
        with pytest.raises(SessionError, match="could not read"):
            load_session(str(tmp_path / "absent.nvsession"))


class TestTracePayload:
    @staticmethod
    def test_trace_payload_round_trip():
        snapshot = snapshot_from_datapoints(_points(), None, name="t")
        payload = snapshot_to_trace_dict(snapshot, "trace-9")
        restored = snapshot_from_trace_dict(payload)
        assert restored.id == snapshot.id
        assert restored.channel("S11") == snapshot.channel("S11")

    @staticmethod
    def test_symbolic_parameters_are_absent_not_zero():
        from NanoVNASaver.Measurement import missing_parameters

        snapshot = snapshot_from_datapoints(_points(), None)
        payload = snapshot_to_trace_dict(snapshot, "t")
        restored = snapshot_from_trace_dict(payload)
        assert restored.available == ("S11",)
        assert "S21" in missing_parameters(restored)

    @staticmethod
    def test_future_trace_version_is_rejected():
        snapshot = snapshot_from_datapoints(_points(), None)
        payload = snapshot_to_trace_dict(snapshot, "t")
        payload["format_version"] = 999
        with pytest.raises(SessionVersionError):
            snapshot_from_trace_dict(payload)
