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
"""Versioned session format (SESSION-01, SESSION-02).

A session is a ZIP archive with the extension ``.nvsession``::

    manifest.json          versioned JSON document (see SESSION_FORMAT)
    traces/<trace-id>.json one file per trace, numeric data only
    calibration/<name>.cal  embedded calibration dataset text

Everything is plain JSON or UTF-8 text.  ``pickle`` and any other executable
object deserialization are deliberately never used, and archive members are
read by name instead of being extracted, so no path traversal is possible.

Format rule: numeric arrays are written with ``repr`` precision preserved by
``json``'s float serialization, so a saved/reloaded array round-trips exactly
for finite values.  Non-finite values are stored as the strings ``"nan"``,
``"inf"`` and ``"-inf"``.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .Measurement import (
    SNAPSHOT_FORMAT_VERSION,
    CompletionState,
    MeasurementSnapshot,
    Provenance,
    _decode_float,
    _encode_float,
    _encode_time,
)
from .RFTools import Datapoint

SESSION_FORMAT = "nanovna-saver-session"
SESSION_VERSION = 1
SESSION_SUFFIX = ".nvsession"

MANIFEST_NAME = "manifest.json"
TRACE_PREFIX = "traces/"
CALIBRATION_PREFIX = "calibration/"

MAX_MEMBER_SIZE = 64 * 1024 * 1024
MAX_TOTAL_SIZE = 512 * 1024 * 1024
MAX_TRACES = 256


class SessionError(Exception):
    """Base class for session problems the UI can report verbatim."""


class SessionVersionError(SessionError):
    """The archive declares a format version this build cannot read."""


class SessionFormatError(SessionError):
    """The archive is present but its content is invalid."""


class SessionWriteError(SessionError):
    """Writing the archive failed; the destination was left untouched."""


def _encode_channel(points) -> list[list]:
    return [
        [dp.freq, _encode_float(dp.re), _encode_float(dp.im)] for dp in points
    ]


def _decode_channel(rows) -> list[Datapoint]:
    return [
        Datapoint(int(row[0]), _decode_float(row[1]), _decode_float(row[2]))
        for row in rows
    ]


def snapshot_to_trace_dict(
    snapshot: MeasurementSnapshot, trace_id: str, *, trace_name: str = ""
) -> dict[str, Any]:
    """Serializable trace payload for one snapshot.

    ``trace_name`` is the operator-visible label and is kept separate from
    ``name``, which belongs to the measurement itself.  Without it a saved
    session reopens with generated names and the user's labels are lost.
    """
    return {
        "trace_id": trace_id,
        "measurement_id": snapshot.id,
        "name": snapshot.name,
        "trace_name": trace_name or snapshot.name,
        "timestamp": _encode_time(snapshot.timestamp),
        "reference_impedance": _encode_float(snapshot.reference_impedance),
        "frequencies": list(snapshot.frequencies),
        "raw": {
            parameter: _encode_channel(points)
            for parameter, points in snapshot.raw.items()
        },
        "corrected": {
            parameter: _encode_channel(points)
            for parameter, points in snapshot.corrected.items()
        },
        "provenance": snapshot.provenance.to_dict(),
        "correction": snapshot.correction.to_dict(),
        "format_version": snapshot.format_version,
    }


def snapshot_from_trace_dict(data: Mapping[str, Any]) -> MeasurementSnapshot:
    """Rebuild a snapshot from a trace payload.

    ``correction.applied`` is trusted from the file, so reloading a corrected
    trace never re-applies calibration.
    """
    version = int(data.get("format_version", SNAPSHOT_FORMAT_VERSION))
    if version > SNAPSHOT_FORMAT_VERSION:
        raise SessionVersionError(
            f"trace format version {version} is newer than supported"
            f" {SNAPSHOT_FORMAT_VERSION}"
        )
    raw = {
        str(param): tuple(_decode_channel(rows))
        for param, rows in data.get("raw", {}).items()
    }
    corrected = {
        str(param): tuple(_decode_channel(rows))
        for param, rows in data.get("corrected", {}).items()
    }
    from .Measurement import CorrectionState

    return MeasurementSnapshot(
        id=str(data.get("measurement_id") or data.get("trace_id")),
        name=str(data.get("name", "")),
        timestamp=datetime.fromisoformat(data["timestamp"])
        if data.get("timestamp")
        else datetime.now(timezone.utc),
        reference_impedance=_decode_float(
            data.get("reference_impedance", 50.0)
        ),
        raw=raw,
        corrected=corrected,
        provenance=Provenance.from_dict(data.get("provenance", {})),
        correction=CorrectionState.from_dict(data.get("correction", {})),
        frequencies=tuple(int(f) for f in data.get("frequencies", ())),
        format_version=version,
    )


@dataclass
class SessionDocument:
    """Everything a saved experiment contains, as plain data."""

    title: str = ""
    notes: str = ""
    created: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    app_version: str = ""
    traces: list[dict[str, Any]] = field(default_factory=list)
    selected_trace: str = ""
    markers: list[dict[str, Any]] = field(default_factory=list)
    charts: dict[str, Any] = field(default_factory=dict)
    sweep: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    preset: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    calibration_files: dict[str, str] = field(default_factory=dict)
    ui: dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "format": SESSION_FORMAT,
            "version": SESSION_VERSION,
            "title": self.title,
            "notes": self.notes,
            "created": _encode_time(self.created),
            "app_version": self.app_version,
            "trace_files": [
                f"{TRACE_PREFIX}{trace['trace_id']}.json"
                for trace in self.traces
            ],
            "selected_trace": self.selected_trace,
            "markers": self.markers,
            "charts": self.charts,
            "sweep": self.sweep,
            "analysis": self.analysis,
            "preset": self.preset,
            "calibration": self.calibration,
            "calibration_files": sorted(self.calibration_files),
            "ui": self.ui,
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "SessionDocument":
        return cls(
            title=str(manifest.get("title", "")),
            notes=str(manifest.get("notes", "")),
            created=datetime.fromisoformat(manifest["created"])
            if manifest.get("created")
            else datetime.now(timezone.utc),
            app_version=str(manifest.get("app_version", "")),
            selected_trace=str(manifest.get("selected_trace", "")),
            markers=list(manifest.get("markers") or []),
            charts=dict(manifest.get("charts") or {}),
            sweep=dict(manifest.get("sweep") or {}),
            analysis=dict(manifest.get("analysis") or {}),
            preset=dict(manifest.get("preset") or {}),
            calibration=dict(manifest.get("calibration") or {}),
            calibration_files={
                name: "" for name in manifest.get("calibration_files") or []
            },
            ui=dict(manifest.get("ui") or {}),
        )


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def _check_archive_size(members: Mapping[str, bytes]) -> None:
    if len(members) > MAX_TRACES + 8:
        raise SessionFormatError(
            f"session archive has too many members ({len(members)})"
        )
    total = 0
    for name, payload in members.items():
        if len(payload) > MAX_MEMBER_SIZE:
            raise SessionFormatError(f"session member {name!r} is too large")
        total += len(payload)
    if total > MAX_TOTAL_SIZE:
        raise SessionFormatError("session archive is too large")


def build_archive(document: SessionDocument) -> bytes:
    """Serialize ``document`` to archive bytes in memory."""
    members: dict[str, bytes] = {}
    for trace in document.traces:
        trace_id = trace["trace_id"]
        members[f"{TRACE_PREFIX}{trace_id}.json"] = json.dumps(
            trace, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    for name, text in document.calibration_files.items():
        members[f"{CALIBRATION_PREFIX}{name}"] = text.encode("utf-8")
    _check_archive_size(members)
    members[MANIFEST_NAME] = json.dumps(
        document.to_manifest(), sort_keys=True, indent=2
    ).encode("utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, members[name])
    return buffer.getvalue()


def save_session(document: SessionDocument, filename: str) -> None:
    """Write ``document`` atomically; the old file survives any failure."""
    payload = build_archive(document)
    directory = os.path.dirname(os.path.abspath(filename)) or "."
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=directory, prefix=".nvsession-", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, filename)
    except BaseException as exc:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise SessionWriteError(f"could not write {filename}: {exc}") from exc


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def _read_members(payload: bytes) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for info in archive.infolist():
                name = info.filename
                if name.endswith("/"):
                    continue
                if name.startswith("/") or ".." in name.split("/"):
                    raise SessionFormatError(f"unsafe archive member {name!r}")
                if name not in {MANIFEST_NAME} and not name.startswith(
                    (TRACE_PREFIX, CALIBRATION_PREFIX)
                ):
                    raise SessionFormatError(
                        f"unexpected archive member {name!r}"
                    )
                if info.file_size > MAX_MEMBER_SIZE:
                    raise SessionFormatError(
                        f"archive member {name!r} is too large"
                    )
                total += info.file_size
                if total > MAX_TOTAL_SIZE:
                    raise SessionFormatError("session archive is too large")
                members[name] = archive.read(info)
    except zipfile.BadZipFile as exc:
        raise SessionFormatError("not a valid session archive") from exc
    if MANIFEST_NAME not in members:
        raise SessionFormatError("session archive has no manifest")
    return members


def load_session(filename: str) -> SessionDocument:
    """Read and fully validate a session without touching application state."""
    try:
        with open(filename, "rb") as source:
            payload = source.read()
    except OSError as exc:
        raise SessionError(f"could not read {filename}: {exc}") from exc
    return parse_session(payload)


def parse_session(payload: bytes) -> SessionDocument:
    members = _read_members(payload)
    try:
        manifest = json.loads(members[MANIFEST_NAME].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SessionFormatError(f"invalid session manifest: {exc}") from exc

    if not isinstance(manifest, dict):
        raise SessionFormatError("session manifest is not an object")
    if manifest.get("format") != SESSION_FORMAT:
        raise SessionFormatError(
            f"unexpected session format {manifest.get('format')!r}"
        )
    version = int(manifest.get("version", 0))
    if version > SESSION_VERSION:
        raise SessionVersionError(
            f"session version {version} is newer than supported"
            f" version {SESSION_VERSION}; upgrade NanoVNASaver to open it"
        )

    document = SessionDocument.from_manifest(manifest)
    trace_files = list(manifest.get("trace_files") or [])
    if len(trace_files) > MAX_TRACES:
        raise SessionFormatError("session declares too many traces")

    for name in trace_files:
        if name not in members:
            raise SessionFormatError(f"session is missing trace {name!r}")
        try:
            trace = json.loads(members[name].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SessionFormatError(f"invalid trace {name!r}: {exc}") from exc
        # Validating here means a broken archive never replaces the open
        # experiment.
        try:
            snapshot_from_trace_dict(trace)
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionFormatError(
                f"invalid trace data in {name!r}: {exc}"
            ) from exc
        document.traces.append(trace)

    for name in manifest.get("calibration_files") or []:
        member = f"{CALIBRATION_PREFIX}{name}"
        if member not in members:
            raise SessionFormatError(
                f"session is missing calibration file {name!r}"
            )
        try:
            document.calibration_files[name] = members[member].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SessionFormatError(
                f"calibration file {name!r} is not UTF-8"
            ) from exc

    if document.selected_trace and not any(
        trace["trace_id"] == document.selected_trace
        for trace in document.traces
    ):
        raise SessionFormatError(
            f"selected trace {document.selected_trace!r} does not exist"
        )
    return document


def make_document(  # noqa: PLR0913
    snapshots: list[MeasurementSnapshot],
    *,
    trace_ids: list[str] | None = None,
    trace_names: dict[str, str] | None = None,
    title: str = "",
    notes: str = "",
    app_version: str = "",
    selected_trace: str = "",
    markers: list[dict] | None = None,
    charts: dict | None = None,
    sweep: dict | None = None,
    analysis: dict | None = None,
    preset: dict | None = None,
    calibration: dict | None = None,
    calibration_files: dict[str, str] | None = None,
    ui: dict | None = None,
) -> SessionDocument:
    """Assemble a document from snapshots (SESSION-01.2)."""
    ids = trace_ids or [f"trace-{index}" for index in range(len(snapshots))]
    if len(ids) != len(snapshots):
        raise ValueError("trace_ids must match the number of snapshots")
    document = SessionDocument(
        title=title,
        notes=notes,
        app_version=app_version,
        traces=[
            snapshot_to_trace_dict(
                snapshot,
                trace_id,
                trace_name=(trace_names or {}).get(trace_id, ""),
            )
            for snapshot, trace_id in zip(snapshots, ids, strict=True)
        ],
        selected_trace=selected_trace or (ids[0] if ids else ""),
        markers=list(markers or []),
        charts=dict(charts or {}),
        sweep=dict(sweep or {}),
        analysis=dict(analysis or {}),
        preset=dict(preset or {}),
        calibration=dict(calibration or {}),
        calibration_files=dict(calibration_files or {}),
        ui=dict(ui or {}),
    )
    return document


def document_snapshots(
    document: SessionDocument,
) -> dict[str, MeasurementSnapshot]:
    """Rebuild snapshots keyed by trace id."""
    return {
        trace["trace_id"]: snapshot_from_trace_dict(trace)
        for trace in document.traces
    }


def completion_of(trace: Mapping[str, Any]) -> CompletionState:
    """Completion state recorded for a trace payload."""
    provenance = trace.get("provenance") or {}
    return CompletionState.parse(
        provenance.get("completion", CompletionState.COMPLETE.value)
    )


def is_finite(value: float) -> bool:
    return math.isfinite(value)
