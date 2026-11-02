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
"""Session workflow and recovery (SESSION-03).

This module holds the *policy* of the session workflow, deliberately free of
widgets so it can be tested headlessly:

* which changes mark a session dirty,
* when a recovery snapshot may be written,
* how an unsaved-change prompt is decided,
* how a restored session is applied without touching hardware.

Deliberate saves and recovery snapshots live in different files, so recovery
can never overwrite work the user asked to keep (SESSION-03.3).
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable

from .Session import (
    SESSION_SUFFIX,
    SessionDocument,
    SessionError,
    save_session,
)

logger = logging.getLogger(__name__)

RECOVERY_SUFFIX = ".nvrecovery"
RECOVERY_INTERVAL_SECONDS = 120.0

#: Changes that alter the experiment and therefore the dirty state.
EXPERIMENT_CHANGES = frozenset(
    {
        "sweep",
        "trace_added",
        "trace_removed",
        "trace_renamed",
        "trace_reordered",
        "trace_visibility",
        "trace_color",
        "marker",
        "reference",
        "notes",
        "title",
        "calibration",
        "analysis",
        "chart_selection",
        "limits",
    }
)

#: Changes that are purely incidental UI state.
INCIDENTAL_CHANGES = frozenset(
    {
        "window_position",
        "window_size",
        "scroll_position",
        "cursor",
        "selected_tab",
        "zoom_level",
    }
)


class UnsavedAction(Enum):
    """What to do about unsaved changes before a destructive action."""

    PROCEED = "proceed"
    SAVE_THEN_PROCEED = "save_then_proceed"
    CANCEL = "cancel"


@dataclass
class SessionState:
    """Workflow state of one experiment, independent of any widget."""

    filename: str = ""
    title: str = ""
    dirty: bool = False
    recovery_filename: str = ""
    last_recovery: datetime | None = None
    _dirty_reasons: set[str] = field(default_factory=set)

    @property
    def has_file(self) -> bool:
        return bool(self.filename)

    @property
    def display_name(self) -> str:
        """Title for the window, with the unsaved marker."""
        base = self.title or (
            os.path.basename(self.filename) if self.filename else "Untitled"
        )
        return f"{base} *" if self.dirty else base

    @property
    def dirty_reasons(self) -> frozenset[str]:
        return frozenset(self._dirty_reasons)

    def mark_changed(self, change: str) -> bool:
        """Record a change.  Returns True when it dirtied the session."""
        if change in INCIDENTAL_CHANGES:
            return False
        if change not in EXPERIMENT_CHANGES:
            logger.debug("Unknown change %r treated as meaningful", change)
        was_dirty = self.dirty
        self._dirty_reasons.add(change)
        self.dirty = True
        return not was_dirty

    def mark_saved(self, filename: str | None = None) -> None:
        if filename:
            self.filename = filename
        self.dirty = False
        self._dirty_reasons.clear()

    def reset(self, *, title: str = "") -> None:
        self.filename = ""
        self.title = title
        self.dirty = False
        self._dirty_reasons.clear()
        self.last_recovery = None


def session_path_for(title: str, directory: str) -> str:
    """Suggest a filename for ``title`` inside ``directory``."""
    clean = "".join(
        character if character.isalnum() or character in " -_" else "_"
        for character in (title or "session")
    ).strip()
    return os.path.join(
        directory or ".", f"{clean or 'session'}{SESSION_SUFFIX}"
    )


def decide_unsaved_action(*, dirty: bool, can_save: bool, answer: str) -> str:
    """Map a user's answer to an action (SESSION-03.3).

    ``answer`` is ``"save"``, ``"discard"`` or ``"cancel"``.  Saving is only
    offered when a document exists to save.
    """
    if not dirty:
        return UnsavedAction.PROCEED.value
    if answer == "cancel":
        return UnsavedAction.CANCEL.value
    if answer == "save":
        if not can_save:
            return UnsavedAction.CANCEL.value
        return UnsavedAction.SAVE_THEN_PROCEED.value
    return UnsavedAction.PROCEED.value


# ----------------------------------------------------------------------
# Recovery (SESSION-03.3)
# ----------------------------------------------------------------------
def recovery_path_for(filename: str) -> str:
    """Recovery file beside the deliberate save, never the same path."""
    if not filename:
        directory = tempfile.gettempdir()
        return os.path.join(directory, f"nanovna-unsaved{RECOVERY_SUFFIX}")
    directory = os.path.dirname(os.path.abspath(filename)) or "."
    stem = os.path.basename(filename)
    if stem.endswith(SESSION_SUFFIX):
        stem = stem[: -len(SESSION_SUFFIX)]
    return os.path.join(directory, f"{stem}{RECOVERY_SUFFIX}")


def recovery_is_due(state: SessionState, now: datetime | None = None) -> bool:
    """Recovery snapshots are periodic and only for unsaved work."""
    if not state.dirty:
        return False
    if state.last_recovery is None:
        return True
    moment = now or datetime.now(timezone.utc)
    return (
        moment - state.last_recovery
    ).total_seconds() >= RECOVERY_INTERVAL_SECONDS


def write_recovery(document: SessionDocument, state: SessionState) -> str:
    """Write a recovery snapshot.  Never touches the deliberate save."""
    target = state.recovery_filename or recovery_path_for(state.filename)
    if state.filename and os.path.abspath(target) == os.path.abspath(
        state.filename
    ):
        raise SessionError(
            "refusing to write a recovery snapshot over the saved session"
        )
    save_session(document, target)
    state.recovery_filename = target
    state.last_recovery = datetime.now(timezone.utc)
    return target


def discard_recovery(state: SessionState) -> bool:
    """Remove the recovery file after a deliberate save or discard."""
    target = state.recovery_filename or recovery_path_for(state.filename)
    state.recovery_filename = ""
    state.last_recovery = None
    if os.path.exists(target):
        try:
            os.unlink(target)
        except OSError:
            logger.warning("Unable to remove recovery file %s", target)
            return False
        return True
    return False


def find_recovery_for(filename: str) -> str | None:
    """Return an existing recovery file for ``filename``, if any."""
    candidate = recovery_path_for(filename)
    return candidate if os.path.exists(candidate) else None


# ----------------------------------------------------------------------
# Restoration (SESSION-03.4)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RestoreOutcome:
    """What a restore did, so the UI can report it without guessing."""

    document: SessionDocument
    app_version_mismatch: bool = False
    missing_traces: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    #: A restore never connects to hardware.
    connected: bool = False
    started_sweep: bool = False


def plan_restore(document: SessionDocument, app_version: str) -> RestoreOutcome:
    """Decide what restoring ``document`` implies.

    Deliberately produces no device interaction: no connection, no sweep, no
    device commands.  Restored chart and layout choices override the global
    preferences for this session only (SESSION-03.5).
    """
    warnings: list[str] = []
    mismatch = bool(
        document.app_version
        and app_version
        and document.app_version != app_version
    )
    if mismatch:
        warnings.append(
            f"Session was created with version {document.app_version};"
            f" this is {app_version}. Data is preserved as recorded."
        )
    if not document.traces:
        warnings.append("Session contains no traces")
    for trace in document.traces:
        # Channels live under ``corrected``/``raw``; either proves the trace
        # carries samples, so check the real shape rather than guessing.
        present = trace.get("corrected") or trace.get("raw") or {}
        if not present:
            warnings.append(
                f"Trace {trace.get('trace_id', '?')} has no channel data"
            )
    if document.ui:
        warnings.append(
            "Chart and layout choices from the session take precedence over"
            " global preferences for this session"
        )
    return RestoreOutcome(
        document=document,
        app_version_mismatch=mismatch,
        warnings=tuple(warnings),
    )


def trace_names(document: SessionDocument) -> list[str]:
    return [
        str(trace.get("name") or trace.get("trace_id") or "")
        for trace in document.traces
    ]


def summary_lines(outcome: RestoreOutcome) -> Iterable[str]:
    """Human-readable restore summary lines."""
    document = outcome.document
    yield f"Title: {document.title or 'Untitled'}"
    if document.created:
        created = document.created.astimezone(timezone.utc)
        yield f"Created: {created:%Y-%m-%d %H:%M UTC}"
    yield f"Traces: {len(document.traces)}"
    for line in outcome.warnings:
        yield f"Note: {line}"
