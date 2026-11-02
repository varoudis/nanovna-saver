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
"""Antenna tuning workflow window (ANT-02/ANT-03).

A guided front end over the *existing* sweep machinery:

1. a broad overview sweep (linear or logarithmic) inside the device range;
2. the minimum VSWR is read from the measured S11 samples and a marker is
   placed on that exact sample;
3. a narrower sweep is centred on it, keeping device limits, point count and
   segment constraints intact;
4. the narrow sweep can be repeated while the antenna is adjusted;
5. the completed measurement is exported as an honest one-port Touchstone
   file whose name states the timestamp and the measured resonance.

The window never connects a device, never starts a sweep by itself and never
runs an unattended loop: every step is one user action routed through
``app.sweep_start()`` and the normal worker lifecycle.  Completion is
accepted only when it belongs to the sweep this window started, so a late
completion from an earlier - or somebody else's - sweep cannot advance the
workflow.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from ..AntennaTuning import (
    AntennaExportError,
    DeviceLimits,
    MinimumPoint,
    SweepPlan,
    SweepPlanError,
    auto_zoom_span,
    export_summary,
    find_vswr_minimum,
    normalize_s1p_path,
    plan_sweep,
    plan_zoom,
    sweep_grid,
    tuning_filename,
    unique_path,
    write_s1p,
)
from ..CalibrationCoverage import coverage_from_calibration
from ..Controls.SweepControl import FrequencyInputWidget
from ..Formatting import (
    format_frequency,
    format_frequency_inputs,
    format_frequency_sweep,
)
from ..Measurement import CompletionState
from ..SweepState import SweepState
from ..TuningHistory import (
    TuningAppendError,
    TuningFormatError,
    TuningHistory,
    band_for,
)
from .Defaults import make_scrollable
from .ui import get_window_icon

if TYPE_CHECKING:
    from ..NanoVNASaver.NanoVNASaver import NanoVNASaver as vna_app

logger = logging.getLogger(__name__)


class Phase:
    """What a pending sweep is for."""

    OVERVIEW = "overview"
    ZOOM = "zoom"
    REPEAT = "repeat"


#: Trace id of the measurement currently displayed (see
#: ``NanoVNASaver.NanoVNASaver.LIVE_TRACE_ID``).  A history step says which
#: trace it describes without importing the application.
LIVE_TRACE_ID = "trace-live"


def _parse_settings(text: str) -> dict[str, str]:
    """Parse ``"name=value; name=value"`` into a mapping.

    Free text is the whole point of the field, so a fragment without ``=``
    is kept under a numeric key rather than dropped.
    """
    settings: dict[str, str] = {}
    for chunk in (text or "").split(";"):
        entry = chunk.strip()
        if not entry:
            continue
        if "=" in entry:
            key, _, value = entry.partition("=")
            key, value = key.strip(), value.strip()
        else:
            key, value = f"item{len(settings) + 1}", entry
        if key:
            settings[key] = value
    return settings


@dataclass(frozen=True)
class PendingStep:
    """One scheduled sweep, bound to the run that carries it.

    ``token`` is a fresh object per step and is handed to the worker with
    :meth:`SweepWorker.bind_next_run`.  Comparing plan fields cannot identify
    a run - two identical plans are indistinguishable, and a terminal that
    was already queued for a cancelled run is delivered to the next step -
    so ownership is decided by object identity alone.
    """

    token: object
    phase: str
    plan: SweepPlan


class AntennaTuningWindow(QtWidgets.QWidget):
    """Guided overview -> minimum -> zoom -> repeat -> export workflow."""

    def __init__(self, app: "vna_app"):
        super().__init__()
        self.app = app

        self.setWindowTitle("Antenna tuning")
        self.setWindowIcon(get_window_icon())
        QtGui.QShortcut(QtCore.Qt.Key.Key_Escape, self, self.hide)

        self._limits = DeviceLimits.from_vna(app.vna)
        self._pending: PendingStep | None = None
        #: Last sweep span this workflow applied (the repeat target).
        self._last_plan: SweepPlan | None = None
        self._minimum: MinimumPoint | None = None
        #: Snapshot of the last completed measurement of this workflow.
        self._snapshot = None
        #: Samples of that measurement that cannot be written to a file.
        self._unusable = 0
        self._last_written: Path | None = None
        #: Token of a step abandoned by :meth:`cancel` while its run was
        #: still going.  Its terminal is still owed, so the outcome is
        #: reported when that run really ends rather than guessed here: the
        #: run may complete before the stop request lands.  Matching the
        #: token keeps somebody else's terminal out of this workflow.
        self._abandoned_token: object = None
        #: Completed tuning steps of this session (TUNE-01).  Append-only;
        #: the window owns the record and the session saves it.
        self.history = TuningHistory()
        #: The window installs both hooks on the application *instance* so
        #: session plumbing (which must stay widget-free) can reach the
        #: record without importing this module.
        app.tuning_history_source = self.history_document
        app.tuning_history_target = self.restore_history

        layout = QtWidgets.QVBoxLayout()
        make_scrollable(self, layout)
        layout.addWidget(self._build_overview_box())
        layout.addWidget(self._build_zoom_box())
        layout.addWidget(self._build_status_box())
        layout.addWidget(self._build_history_box())
        layout.addWidget(self._build_export_box())

        app.worker.signals.run_finished.connect(self._on_run_finished)
        app.worker.signals.run_request_finished.connect(
            self._on_run_request_finished
        )
        app.worker.signals.state_changed.connect(self._on_state_changed)
        app.serial_control.connected.connect(self._on_connection_changed)
        # A new session, a restored session or a cleared display changes what
        # may honestly be exported; refresh then, never on every redraw.
        app.communicate.data_available.connect(self._on_data_available)
        # Terminal states are published from inside ``run()``, while the
        # worker thread is still running: those handlers always see
        # ``worker.isRunning()`` true and must keep offering Stop.  Only the
        # thread's own ``finished`` signal reports the moment the controls
        # become usable again, so the refresh is bound to that real lifetime.
        # Without it a cancelled first sweep (no previous result stored) left
        # Overview disabled and Stop enabled.
        self.app.worker.finished.connect(self._on_thread_finished)

        self.refresh_device_limits()
        self._seed_from_sweep()
        self.refresh()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------
    def _build_overview_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("1. Broad overview sweep")
        form = QtWidgets.QFormLayout(box)

        self.overview_start = FrequencyInputWidget("")
        self.overview_stop = FrequencyInputWidget("")
        self.overview_start.setToolTip(
            "Start of the wide sweep used to find the antenna response."
        )
        self.overview_stop.setToolTip(
            "Stop of the wide sweep. Frequencies outside the device range are"
            " clamped to it and reported."
        )
        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(QtWidgets.QLabel("Start"))
        range_row.addWidget(self.overview_start)
        range_row.addWidget(QtWidgets.QLabel("Stop"))
        range_row.addWidget(self.overview_stop)
        form.addRow(range_row)

        self.overview_points = QtWidgets.QComboBox()
        self.overview_points.setToolTip(
            "Point count the connected device supports."
        )
        self.overview_segments = QtWidgets.QSpinBox()
        self.overview_segments.setRange(1, 64)
        self.overview_segments.setToolTip(
            "Segments split the span into several device sweeps, giving finer"
            " steps. Reduced automatically when the span cannot fit them."
        )
        self.overview_log = QtWidgets.QCheckBox("Logarithmic steps")
        self.overview_log.setToolTip(
            "Logarithmic spacing only changes step widths when the sweep has"
            " more than one segment; the device is told the real segment"
            " count either way."
        )
        form.addRow("Points", self.overview_points)
        form.addRow("Segments", self.overview_segments)
        form.addRow(self.overview_log)
        # Keep the preview and the calibration preflight in step with what
        # the user typed, without starting anything.
        self.overview_start.textChanged.connect(self._refresh_plan_preview)
        self.overview_stop.textChanged.connect(self._refresh_plan_preview)
        self.overview_segments.valueChanged.connect(self._refresh_plan_preview)
        self.overview_points.currentIndexChanged.connect(
            self._refresh_plan_preview
        )
        self.overview_log.toggled.connect(self._refresh_plan_preview)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_overview = QtWidgets.QPushButton("Overview sweep")
        self.btn_overview.setMinimumHeight(20)
        self.btn_overview.setToolTip(
            "Sweep the wide span once and report the lowest measured VSWR."
        )
        self.btn_overview.clicked.connect(self.start_overview)
        self.btn_device_range = QtWidgets.QPushButton("Use device range")
        self.btn_device_range.setMinimumHeight(20)
        self.btn_device_range.setToolTip(
            "Fill start and stop with the range the device reports."
        )
        self.btn_device_range.clicked.connect(self.use_device_range)
        buttons.addWidget(self.btn_overview)
        buttons.addWidget(self.btn_device_range)
        form.addRow(buttons)

        self.label_device = QtWidgets.QLabel("")
        self.label_device.setWordWrap(True)
        form.addRow(self.label_device)
        return box

    def _build_zoom_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("2. Zoom, tune and repeat")
        form = QtWidgets.QFormLayout(box)

        self.label_minimum = QtWidgets.QLabel("No measurement yet")
        self.label_minimum.setWordWrap(True)
        form.addRow("Measured minimum", self.label_minimum)

        self.zoom_span = FrequencyInputWidget("")
        self.zoom_span.setToolTip(
            "Width of the narrow sweep centred on the measured minimum."
        )
        form.addRow("Zoom span", self.zoom_span)

        self.zoom_points = QtWidgets.QComboBox()
        self.zoom_points.setToolTip(
            "Point count inside the narrow sweep; more points resolve the dip"
            " more finely."
        )
        form.addRow("Zoom points", self.zoom_points)

        self.chk_marker = QtWidgets.QCheckBox("Place a marker at the minimum")
        self.chk_marker.setChecked(True)
        self.chk_marker.setToolTip(
            "Move marker 1 onto the measured minimum sample after every"
            " completed sweep of this workflow."
        )
        form.addRow(self.chk_marker)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_zoom = QtWidgets.QPushButton("Zoom to minimum")
        self.btn_zoom.setMinimumHeight(20)
        self.btn_zoom.setToolTip(
            "Sweep a narrow span centred on the measured minimum."
        )
        self.btn_zoom.clicked.connect(self.zoom_to_minimum)
        self.btn_repeat = QtWidgets.QPushButton("Repeat sweep")
        self.btn_repeat.setMinimumHeight(20)
        self.btn_repeat.setToolTip(
            "Repeat the current narrow sweep, e.g. after adjusting the"
            " antenna. The span stays put so the dip is visible."
        )
        self.btn_repeat.clicked.connect(self.repeat_sweep)
        buttons.addWidget(self.btn_zoom)
        buttons.addWidget(self.btn_repeat)
        form.addRow(buttons)

        self.btn_stop = QtWidgets.QPushButton("Stop sweep")
        self.btn_stop.setMinimumHeight(20)
        self.btn_stop.setToolTip(
            "Cancel the running sweep. The last complete measurement is kept"
            " and this workflow stops advancing."
        )
        self.btn_stop.clicked.connect(self.cancel)
        form.addRow(self.btn_stop)
        return box

    def _build_status_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Status")
        form = QtWidgets.QFormLayout(box)
        self.label_plan = QtWidgets.QLabel("")
        self.label_plan.setWordWrap(True)
        form.addRow("Next sweep", self.label_plan)
        self.label_calibration = QtWidgets.QLabel("")
        self.label_calibration.setWordWrap(True)
        form.addRow("Calibration", self.label_calibration)
        self.label_status = QtWidgets.QLabel("")
        self.label_status.setWordWrap(True)
        self.label_status.setObjectName("antenna_tuning_status")
        form.addRow(self.label_status)
        return box

    def _build_export_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("4. Export the measurement")
        form = QtWidgets.QFormLayout(box)
        self.label_export = QtWidgets.QLabel("")
        self.label_export.setWordWrap(True)
        form.addRow("Contents", self.label_export)
        self.label_filename = QtWidgets.QLabel("")
        self.label_filename.setWordWrap(True)
        self.label_filename.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("Suggested file", self.label_filename)
        self.btn_export = QtWidgets.QPushButton("Export S1P ...")
        self.btn_export.setMinimumHeight(20)
        self.btn_export.setToolTip(
            "Save the completed measurement as a one-port Touchstone file."
        )
        self.btn_export.clicked.connect(self.choose_and_export)
        form.addRow(self.btn_export)
        return box

    def _build_history_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("3. Tuning history")
        form = QtWidgets.QFormLayout(box)
        self.label_history = QtWidgets.QLabel("")
        self.label_history.setWordWrap(True)
        self.label_history.setObjectName("antenna_tuning_history")
        form.addRow(self.label_history)

        self.list_history = QtWidgets.QListWidget()
        self.list_history.setObjectName("antenna_tuning_history_list")
        self.list_history.setMinimumHeight(110)
        self.list_history.setToolTip(
            "Every completed tuning step of this session, oldest first."
        )
        self.list_history.currentRowChanged.connect(
            self._refresh_history_details
        )
        form.addRow(self.list_history)

        self.label_trend = QtWidgets.QLabel("")
        self.label_trend.setWordWrap(True)
        self.label_trend.setObjectName("antenna_tuning_trend")
        form.addRow("Comparison", self.label_trend)

        self.history_notes = QtWidgets.QLineEdit()
        self.history_notes.setPlaceholderText(
            "Note for the next recorded step, e.g. 'element shortened 5 mm'"
        )
        form.addRow("Note", self.history_notes)
        self.history_settings = QtWidgets.QLineEdit()
        self.history_settings.setPlaceholderText(
            "Component/position, e.g. 'coil=3.5 turns; tap=12 mm'"
        )
        form.addRow("Settings", self.history_settings)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_recall = QtWidgets.QPushButton("Recall span")
        self.btn_recall.setToolTip(
            "Fill the sweep form with a recorded step's span. Nothing is"
            " measured: press a sweep button when you are ready."
        )
        self.btn_recall.clicked.connect(self.recall_selected)
        self.btn_clear_history = QtWidgets.QPushButton("Clear history")
        self.btn_clear_history.setToolTip(
            "Forget every recorded step of this session."
        )
        self.btn_clear_history.clicked.connect(self.clear_history)
        buttons.addWidget(self.btn_recall)
        buttons.addWidget(self.btn_clear_history)
        form.addRow(buttons)

        export_row = QtWidgets.QHBoxLayout()
        self.btn_export_history_json = QtWidgets.QPushButton(
            "Export history JSON ..."
        )
        self.btn_export_history_json.clicked.connect(
            lambda: self.export_history("json")
        )
        self.btn_export_history_csv = QtWidgets.QPushButton(
            "Export history CSV ..."
        )
        self.btn_export_history_csv.clicked.connect(
            lambda: self.export_history("csv")
        )
        export_row.addWidget(self.btn_export_history_json)
        export_row.addWidget(self.btn_export_history_csv)
        form.addRow(export_row)
        self._refresh_history()
        return box

    # ------------------------------------------------------------------
    # Tuning history (TUNE-01)
    # ------------------------------------------------------------------
    def history_document(self) -> dict:
        """The session-persistable form of the recorded steps."""
        return self.history.to_dict()

    def restore_history(self, document) -> int:
        """Replace the record with a session's history; returns step count.

        An empty or absent document is an empty history, which is what a
        session written before the history existed means.  A malformed one is
        reported instead of being silently treated as empty.
        """
        try:
            self.history = TuningHistory.from_dict(document)
        except TuningFormatError as exc:
            self.history = TuningHistory()
            self._set_status(
                f"The session's tuning history could not be read: {exc}",
                error=True,
            )
            self._refresh_history()
            return 0
        self._refresh_history()
        return len(self.history)

    def forget_history(self) -> int:
        """Drop every step: a new session starts with no history."""
        removed = self.history.clear()
        self._refresh_history()
        return removed

    def _bands(self) -> list:
        bands = getattr(self.app, "bands", None)
        return list(getattr(bands, "bands", ()) or ())

    def record_step(
        self,
        *,
        snapshot,
        plan: SweepPlan,
        minimum: MinimumPoint,
        run_id: str,
    ) -> bool:
        """Append one completed step; a refusal is reported, never hidden."""
        try:
            result = self.history.append(
                snapshot=snapshot,
                plan=plan,
                minimum=minimum,
                run_id=run_id,
                band=band_for(minimum.freq, self._bands()),
                trace_id=LIVE_TRACE_ID,
                notes=self.history_notes.text().strip(),
                settings=_parse_settings(self.history_settings.text()),
            )
        except TuningAppendError as exc:
            self._set_status(
                f"This measurement was not recorded in the history: {exc}",
                error=True,
            )
            return False
        if result.duplicate:
            return False
        if result.dropped:
            self._set_status(
                f"{len(result.dropped)} older tuning step(s) were dropped:"
                f" the history keeps its most recent {self.history.max_steps}."
            )
        self.app.mark_session_changed("tuning_history")
        self._refresh_history(select=len(self.history) - 1)
        return True

    def recall_selected(self) -> bool:
        """Fill the sweep form from a recorded step.  Starts nothing."""
        row = self.list_history.currentRow()
        if not 0 <= row < len(self.history):
            self._set_status(
                "Select a recorded step to recall its span.", error=True
            )
            return False
        recall = self.history.recall(row)
        self.overview_start.setText(format_frequency_sweep(recall.start))
        self.overview_stop.setText(format_frequency_sweep(recall.end))
        index = self.overview_points.findData(recall.points)
        if index >= 0:
            self.overview_points.setCurrentIndex(index)
        self.overview_segments.setValue(recall.segments)
        self.overview_log.setChecked(recall.logarithmic)
        self._refresh_plan_preview()
        self._set_status(
            f"Recalled step {row + 1}: {recall.describe()}. Nothing was"
            " swept; press Overview sweep to measure that span again."
        )
        return True

    def clear_history(self) -> int:
        removed = self.forget_history()
        if removed:
            self.app.mark_session_changed("tuning_history")
        self._set_status(
            f"Cleared {removed} tuning step(s) from the history."
            if removed
            else "The tuning history is already empty."
        )
        return removed

    def export_history(self, kind: str) -> Path | None:
        """Export the whole history as JSON or CSV."""
        if self.history.empty:
            self._set_status(
                "There is no tuning history to export yet.", error=True
            )
            return None
        filters = {
            "json": "JSON (*.json);;All files (*.*)",
            "csv": "CSV (*.csv);;All files (*.*)",
        }
        suffix = f".{kind}"
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            f"Export tuning history ({kind.upper()})",
            str(Path.home() / f"tuning-history{suffix}"),
            filters[kind],
        )
        if not filename:
            return None
        target = Path(filename)
        if target.suffix.lower() != suffix:
            target = target.with_suffix(suffix)
        try:
            if kind == "json":
                written = self.history.export_json(target)
            else:
                written = self.history.export_csv(target)
        except (OSError, TuningFormatError) as exc:
            self.app.showError(f"Unable to export the tuning history: {exc}")
            return None
        self._set_status(
            f"Exported {len(self.history)} tuning step(s) to {written}"
        )
        return Path(written)

    def _refresh_history(self, select: int | None = None) -> None:
        blocked = self.list_history.blockSignals(True)
        try:
            self.list_history.clear()
            for index, step in enumerate(self.history, start=1):
                item = QtWidgets.QListWidgetItem(f"{index}. {step.describe()}")
                item.setToolTip(step.describe())
                self.list_history.addItem(item)
        finally:
            self.list_history.blockSignals(blocked)
        if select is None:
            select = len(self.history) - 1
        if 0 <= select < self.list_history.count():
            self.list_history.setCurrentRow(select)
        else:
            self.list_history.setCurrentRow(-1)
        self.label_history.setText(self.history.summary())
        self._refresh_history_details()
        has_steps = not self.history.empty
        self.btn_recall.setEnabled(has_steps)
        self.btn_clear_history.setEnabled(has_steps)
        self.btn_export_history_json.setEnabled(has_steps)
        self.btn_export_history_csv.setEnabled(has_steps)

    def _refresh_history_details(self, *_args) -> None:
        row = self.list_history.currentRow()
        if not 0 <= row < len(self.history):
            self.label_trend.setText(
                "Completed steps appear here with their comparison."
            )
            return
        comparison = self.history.trend(row)
        self.label_trend.setText(
            "\n".join(comparison.lines()) if comparison else ""
        )

    # ------------------------------------------------------------------
    # Device facts and defaults
    # ------------------------------------------------------------------
    def device_limits(self) -> DeviceLimits:
        return self._limits

    def refresh_device_limits(self) -> None:
        """Re-read the device's declared limits and repopulate selectors."""
        self._limits = DeviceLimits.from_vna(self.app.vna)
        for combo, preferred in (
            (self.overview_points, self.app.vna.datapoints),
            (self.zoom_points, self._limits.default_points),
        ):
            blocked = combo.blockSignals(True)
            try:
                combo.clear()
                for count in self._limits.points:
                    combo.addItem(f"{count} points", count)
                index = combo.findData(preferred)
                combo.setCurrentIndex(index if index >= 0 else 0)
            finally:
                combo.blockSignals(blocked)
        self.overview_segments.setMaximum(
            max(1, min(self._limits.max_segments, 64))
        )
        self.label_device.setText(self._limits.describe())

    def use_device_range(self) -> None:
        """Fill the overview with the range the device reports."""
        start, stop = self._device_range()
        self.overview_start.setText(format_frequency_sweep(start))
        self.overview_stop.setText(format_frequency_sweep(stop))
        self._refresh_plan_preview()

    def _device_range(self) -> tuple[int, int]:
        limits = self._limits
        start = self.app.sweep_control.get_start()
        stop = self.app.sweep_control.get_end()
        if start <= 0 or stop <= start:
            start = limits.min_frequency
            stop = limits.max_frequency or 30_000_000
        if stop <= start:
            stop = start + 1_000_000
        return start, stop

    def _seed_from_sweep(self) -> None:
        start, stop = self._device_range()
        self.overview_start.setText(format_frequency_sweep(start))
        self.overview_stop.setText(format_frequency_sweep(stop))
        self.overview_segments.setValue(
            max(1, min(self.app.sweep_control.get_segments(), 64))
        )
        self._refresh_plan_preview()

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    @staticmethod
    def _combo_points(combo: QtWidgets.QComboBox) -> int | None:
        data = combo.currentData()
        return int(data) if data is not None else None

    def overview_plan(self) -> SweepPlan:
        """Validate the entered overview sweep; raises ``SweepPlanError``."""
        start = self.overview_start.get_freq()
        stop = self.overview_stop.get_freq()
        if start < 0 or stop < 0:
            raise SweepPlanError(
                "Enter whole start and stop frequencies, e.g. 7MHz or 30000000."
            )
        return plan_sweep(
            start,
            stop,
            limits=self._limits,
            points=self._combo_points(self.overview_points),
            segments=self.overview_segments.value(),
            logarithmic=self.overview_log.isChecked(),
        )

    def zoom_plan(self) -> SweepPlan:
        """Validate the narrow sweep for the current measured minimum."""
        minimum = self._minimum
        if minimum is None:
            raise SweepPlanError(
                "Run an overview sweep first: the zoom needs a measured"
                " minimum."
            )
        span = self.zoom_span.get_freq()
        if span <= 0:
            raise SweepPlanError(
                "Enter a zoom span (e.g. 200kHz) wider than one device step."
            )
        return plan_zoom(
            minimum.freq,
            span,
            limits=self._limits,
            points=self._combo_points(self.zoom_points),
        )

    def planned_frequencies(self, plan: SweepPlan) -> list[int]:
        """The exact grid the device measures for ``plan``."""
        return list(
            sweep_grid(
                plan.start,
                plan.end,
                plan.points,
                plan.segments,
                logarithmic=plan.logarithmic,
            )
        )

    def _coverage_text(self, plan: SweepPlan) -> str:
        """Calibration preflight for the planned grid."""
        try:
            frequencies = self.planned_frequencies(plan)
        except ValueError as exc:  # pragma: no cover - guarded by the plan
            return f"Calibration: unavailable ({exc})"
        coverage = coverage_from_calibration(
            self.app.calibration, frequencies, parameters=("S11",)
        )
        text = coverage.describe()
        if not coverage.has_calibration:
            return f"{text} - values will be uncorrected"
        if not coverage.covered:
            return (
                f"{text} - this span is not fully covered by the loaded"
                " calibration"
            )
        return f"{text} - in range"

    def _refresh_plan_preview(self, *_args) -> None:
        try:
            plan = self.overview_plan()
        except SweepPlanError as exc:
            self.label_plan.setText(f"Overview: {exc}")
            self.label_calibration.setText("")
            return
        text = f"Overview: {plan.describe()}"
        notes = list(plan.notes)
        if plan.logarithmic and plan.segments == 1:
            notes.append(
                "one segment always measures linearly; raise the segment"
                " count for logarithmic steps"
            )
        if notes:
            text += " (" + "; ".join(notes) + ")"
        self.label_plan.setText(text)
        self.label_calibration.setText(self._coverage_text(plan))
        if self._minimum is None and self.zoom_span.get_freq() <= 0:
            self.zoom_span.setText(
                format_frequency_sweep(auto_zoom_span(plan.span, plan.points))
            )

    # ------------------------------------------------------------------
    # Workflow actions
    # ------------------------------------------------------------------
    def start_overview(self) -> bool:
        """Run the broad sweep once through the normal sweep path."""
        if not self._can_start():
            return False
        try:
            plan = self.overview_plan()
        except SweepPlanError as exc:
            self._set_status(f"Cannot sweep that span: {exc}", error=True)
            return False
        return self._start_plan(plan, Phase.OVERVIEW)

    def zoom_to_minimum(self) -> bool:
        """Centre a narrower sweep on the measured minimum."""
        if not self._can_start():
            return False
        try:
            plan = self.zoom_plan()
        except SweepPlanError as exc:
            self._set_status(f"Cannot zoom: {exc}", error=True)
            return False
        return self._start_plan(plan, Phase.ZOOM)

    def repeat_sweep(self) -> bool:
        """Repeat the current narrow sweep without moving the window."""
        if not self._can_start():
            return False
        plan = self._last_plan
        if plan is None:
            self._set_status(
                "Nothing to repeat yet: run an overview sweep first.",
                error=True,
            )
            return False
        try:
            repeated = plan_sweep(
                plan.start,
                plan.end,
                limits=self._limits,
                points=plan.points,
                segments=plan.segments,
                logarithmic=plan.logarithmic,
            )
        except SweepPlanError as exc:
            self._set_status(f"Cannot repeat: {exc}", error=True)
            return False
        return self._start_plan(repeated, Phase.REPEAT)

    def cancel(self) -> None:
        """Stop the running sweep and abandon the pending step."""
        was_pending = self._pending is not None
        token = self._invalidate_pending()
        if was_pending and self.app.worker.isRunning():
            # The run keeps its token, so its real outcome can still be
            # recognised and reported once it ends; the step itself is gone,
            # which is what makes an already-queued completion unusable.
            self._abandoned_token = token
            self.app.worker.quit()
            self._set_status("Stopping the sweep ...")
        elif was_pending:
            self._set_status("Measurement cancelled before it started.")
        elif self.app.worker.isRunning():
            # Nothing of this workflow is waiting for that sweep: either its
            # run already reported and only the thread outlives it, or it was
            # started elsewhere.  Stop it, but never overwrite a result this
            # workflow has just adopted.
            self.app.worker.quit()
        else:
            self._set_status("No sweep is running.")
        self.refresh()

    def _can_start(self) -> bool:
        if not self.app.vna.connected():
            self._set_status(
                "Connect a device (or the simulated device) first.",
                error=True,
            )
            return False
        if self._pending is not None or self.app.worker.isRunning():
            self._set_status(
                "A sweep is already running; wait for it to finish.",
                error=True,
            )
            return False
        return True

    def _invalidate_pending(self) -> object:
        """Abandon the scheduled step; returns the token that owned it.

        Ownership is a token, not a counter and not a plan: dropping the
        reference is what makes a completion that is already queued for this
        step unrecognisable, even when the next step measures exactly the
        same span with the same settings.
        """
        token = self._pending.token if self._pending is not None else None
        self._pending = None
        return token

    def _apply_plan(self, plan: SweepPlan) -> bool:
        """Set every sweep setting the plan needs as one coherent change."""
        applied = self.app.apply_sweep_configuration(
            start=plan.start,
            end=plan.end,
            segments=plan.segments,
            points=plan.points,
            logarithmic=plan.logarithmic,
        )
        if not applied:
            sweep = self.app.sweep
            self._set_status(
                "The device did not accept the planned sweep"
                f" ({sweep.start} - {sweep.end}, {sweep.points} points);"
                " nothing was measured.",
                error=True,
            )
            return False
        self.app.mark_session_changed("sweep")
        return True

    def _start_plan(self, plan: SweepPlan, phase: str) -> bool:
        """Apply ``plan`` and start exactly one sweep."""
        if self.app.worker.isRunning():
            self._set_status(
                "A sweep is already running; wait for it to finish.",
                error=True,
            )
            return False
        if not self._apply_plan(plan):
            return False
        token = object()
        self._pending = PendingStep(token=token, phase=phase, plan=plan)
        # A newer step takes over the workflow, so any run abandoned before
        # it must no longer speak for the status: its outcome is history, and
        # a delayed terminal would otherwise replace "Measuring ..." with a
        # message about a sweep that is no longer the current one.
        self._abandoned_token = None
        # The token must be bound before the run can start, and it is bound
        # on the same thread that publishes the pending step.
        self.app.worker.bind_next_run(token)
        self.label_plan.setText(f"{phase.capitalize()}: {plan.describe()}")
        self.label_calibration.setText(self._coverage_text(plan))
        self._set_status(f"Measuring ({phase}) ...")
        self.refresh()
        # The only two reasons for ``sweep_start`` to refuse were checked
        # above, on this thread, so the sweep either starts or the device is
        # gone - in which case the connection signal refreshes the window.
        self.app.sweep_start()
        return True

    # ------------------------------------------------------------------
    # Sweep lifecycle
    # ------------------------------------------------------------------
    def _on_state_changed(self, _state) -> None:
        # Cheap: only enable/disable widgets from the current worker state.
        self.refresh()

    def _on_connection_changed(self, _connected: bool) -> None:
        self.refresh_device_limits()
        self.refresh()

    def _on_data_available(self) -> None:
        """Notice when the display no longer holds our measurement.

        Coalesced live redraws also announce session-level changes (new
        session, an imported file, a sweep started elsewhere), which is when
        the stored result stops describing what is on screen.
        """
        if self._snapshot is not None and not self.is_busy():
            self.refresh()

    def _on_run_finished(self, terminal) -> None:
        """Notice an unrelated finish; never adopt a terminal from it.

        The generic signal carries no run identity, so a terminal that was
        queued while an earlier run was cancelled is indistinguishable from
        this step's own completion.  It is therefore used only to refresh
        the controls; adoption happens in
        :meth:`_on_run_request_finished`, which can prove ownership.
        """
        if self._pending is None:
            self.refresh()

    def _on_thread_finished(self) -> None:
        """Re-derive the controls once the worker thread has really ended.

        The terminal signals are emitted from inside ``run()``, so at that
        moment ``worker.isRunning()`` is still true and the window must keep
        Stop offered.  A started thread always reaches this point, however
        its run ended (completion, cancellation, failure or an exception),
        so this is the only place where a step that never stored a result can
        hand the controls back.  The pending step itself is untouched here:
        ownership stays with :meth:`_on_run_request_finished`, and a stale
        queued terminal remains unrecognisable.
        """
        self.refresh()

    def _on_run_request_finished(self, terminal, token) -> None:
        """Terminal of a *specific* run; only our own step may advance."""
        pending = self._pending
        if pending is None or token is None or token is not pending.token:
            # Somebody else's run, or a step that :meth:`cancel` already
            # abandoned while its terminal was still queued.  An abandoned
            # step is never resumed - the completion is deliberately not
            # adopted - but the run may have finished before the stop landed,
            # so its real outcome is what the status reports.
            if (
                token is not None
                and token is self._abandoned_token
                and self._pending is None
            ):
                self._abandoned_token = None
                if terminal is SweepState.COMPLETED:
                    self._set_status(
                        "The sweep had already completed when the stop"
                        " arrived. Its measurement is on screen but this"
                        " workflow did not adopt it - run the overview again"
                        " to use it."
                    )
                else:
                    self._after_interrupted(terminal)
            # A terminal that owns neither the current pending step nor the
            # current abandoned one is somebody else's run - possibly an
            # older queued terminal, or a newer run whose own outcome is
            # still owed.  It has no authority over this workflow: it may
            # neither clear the abandoned token nor touch the status.
            self.refresh()
            return
        if terminal is not SweepState.COMPLETED:
            self._invalidate_pending()
            self._after_interrupted(terminal)
            self.refresh()
            return
        # The token identifies the run; the completed data must still prove
        # it measures the planned sweep, because the worker's settings are
        # shared mutable state.
        self._adopt_completion(pending)

    def _after_interrupted(self, terminal) -> None:
        """Keep the display and the export offer consistent after a stop.

        The application restores the last complete sweep, so the previous
        result usually stays valid; when it does not, nothing may be offered
        for zooming or export.  The interrupted sweep's partial samples are
        never adopted.
        """
        failed = terminal is SweepState.FAILED
        keep = self._snapshot is not None and _describes_samples(
            self._snapshot, self._displayed_samples()
        )
        if not keep:
            self._drop_result()
        self._set_status(
            ("Sweep failed" if failed else "Sweep stopped")
            + (
                " - the last complete measurement is unchanged."
                if keep
                else " - no complete measurement is displayed."
            ),
            error=failed,
        )

    def _adopt_completion(self, pending: PendingStep) -> None:
        self._pending = None
        snapshot = self.app.current_snapshot()
        if not self._describes_plan(snapshot, pending.plan):
            self._last_plan = None
            self._set_status(
                "The completed sweep does not match the planned span; no"
                " minimum is reported.",
                error=True,
            )
            self.refresh()
            return
        self._snapshot = snapshot
        result = find_vswr_minimum(snapshot.channel("S11") or ())
        self._minimum = result.minimum
        # Non-finite samples are skipped when searching for the minimum, and
        # they also make the measurement unwritable; say so now rather than
        # only when the user asks for the export.
        self._unusable = result.skipped
        self.label_minimum.setText(result.describe())
        if result.minimum is None:
            self._set_status(
                f"Sweep complete but no minimum is available:"
                f" {result.describe()}",
                error=True,
            )
            self.refresh()
            return
        if self.chk_marker.isChecked():
            self._place_marker(result.minimum)
        self._last_plan = pending.plan
        # Only a completed sweep whose samples match the plan reaches this
        # point, so this is the single place a step can be recorded: a
        # cancelled or failed run never arrives here, and a duplicated
        # completion is refused by the run's own token upstream.
        self.record_step(
            snapshot=snapshot,
            plan=pending.plan,
            minimum=result.minimum,
            run_id=_run_identity(snapshot, pending),
        )
        self.zoom_span.setText(
            format_frequency_sweep(
                auto_zoom_span(pending.plan.span, pending.plan.points)
            )
        )
        self.zoom_span.setToolTip(
            "Width of the narrow sweep centred on the measured minimum"
            f" ({format_frequency(result.minimum.freq)}); minimum VSWR"
            f" {result.minimum.vswr:.3f}."
        )
        status = (
            f"{pending.phase.capitalize()} sweep complete: {result.describe()}"
        )
        if self._unusable:
            status += (
                f"; {self._unusable} sample(s) are not finite, so this"
                " measurement cannot be exported"
            )
        self._set_status(status)
        self.refresh()

    def _displayed_samples(self) -> list:
        with self.app.dataLock:
            return list(self.app.data.s11)

    def _describes_plan(self, snapshot, plan: SweepPlan) -> bool:
        """Whether ``snapshot`` is the completed sweep that was planned.

        Every plan field that changes the measured grid is compared:
        span, point count, segment count and logarithmic stepping.  Two
        plans that differ only in segments or spacing measure different
        sample sets, so start/end/points alone are not enough.  When the
        snapshot carries no sweep description (a rebuilt snapshot), the
        measured frequency grid itself must equal the grid this plan asks
        for.
        """
        if snapshot is None:
            return False
        provenance = snapshot.provenance
        if provenance.completion is not CompletionState.COMPLETE:
            return False
        info = provenance.sweep
        if info is not None:
            return (
                info.start,
                info.end,
                info.points,
                info.segments,
                bool(info.logarithmic),
            ) == (
                plan.start,
                plan.end,
                plan.points,
                plan.segments,
                bool(plan.logarithmic),
            )
        samples = snapshot.channel("S11") or ()
        try:
            expected = sweep_grid(
                plan.start,
                plan.end,
                plan.points,
                plan.segments,
                logarithmic=plan.logarithmic,
            )
        except SweepPlanError:
            return False
        measured = tuple(int(point.freq) for point in samples)
        # A logarithmic multi-segment sweep repeats the shared segment
        # boundary, so the expected grid may hold one extra sample per
        # boundary; compare the distinct ordered frequencies instead.
        if len(measured) == len(expected):
            return measured == expected
        return measured == tuple(dict.fromkeys(expected))

    def _place_marker(self, minimum: MinimumPoint) -> None:
        """Move marker 1 onto the measured minimum sample."""
        if not self.app.markers:
            return
        marker = self.app.markers[0]
        marker.setFrequency(format_frequency_inputs(minimum.freq))
        for chart in self.app.subscribing_charts:
            chart.update()

    # ------------------------------------------------------------------
    # Status and enablement
    # ------------------------------------------------------------------
    def is_busy(self) -> bool:
        """Busy while this workflow waits for its sweep or one runs at all."""
        return self._pending is not None or self.app.worker.isRunning()

    def pending_phase(self) -> str | None:
        return self._pending.phase if self._pending is not None else None

    def measured_minimum(self) -> MinimumPoint | None:
        return self._minimum

    def completed_snapshot(self):
        """The completed measurement this window would export."""
        return self._snapshot

    def last_written(self) -> Path | None:
        return self._last_written

    def _drop_result(self) -> None:
        """Forget the stored measurement and everything derived from it."""
        self._snapshot = None
        self._minimum = None
        self._last_plan = None
        self._unusable = 0
        self.label_minimum.setText("No measurement yet")
        self.zoom_span.setToolTip(
            "Width of the narrow sweep centred on the measured minimum."
        )

    def refresh(self) -> None:
        """Enable exactly the actions that are valid right now."""
        connected = bool(self.app.vna.connected())
        busy = self.is_busy()
        if (
            self._snapshot is not None
            and not busy
            and not _describes_samples(
                self._snapshot, self._displayed_samples()
            )
        ):
            # A new session, an imported file or a sweep started elsewhere
            # means the stored measurement no longer describes the display,
            # so it must not be offered as this workflow's result.
            self._drop_result()
        has_minimum = self._minimum is not None
        snapshot = self._snapshot
        exportable = (
            snapshot is not None
            and snapshot.provenance.completion is CompletionState.COMPLETE
            and not self._unusable
        )
        self.btn_overview.setEnabled(connected and not busy)
        self.btn_device_range.setEnabled(not busy)
        self.btn_zoom.setEnabled(connected and not busy and has_minimum)
        self.btn_repeat.setEnabled(
            connected and not busy and self._last_plan is not None
        )
        self.btn_stop.setEnabled(busy)
        self.btn_export.setEnabled(exportable)
        for widget in (
            self.overview_points,
            self.overview_segments,
            self.overview_log,
            self.zoom_points,
            self.zoom_span,
            self.chk_marker,
        ):
            widget.setEnabled(not busy)
        disconnected = (
            "No device connected. Connect a VNA (or the simulated device)"
            " in the serial control to start measuring."
        )
        if not connected:
            self.label_status.setText(disconnected)
        elif self.label_status.text() == disconnected:
            self.label_status.setText("")
        self._refresh_export_labels()

    def _set_status(self, text: str, *, error: bool = False) -> None:
        self.label_status.setText(text)
        self.label_status.setStyleSheet(
            "QLabel { color: red; }" if error else ""
        )

    def _refresh_export_labels(self) -> None:
        self.label_export.setText(export_summary(self._snapshot, self._minimum))
        if self._snapshot is None:
            self.label_filename.setText("")
            return
        self.label_filename.setText(
            tuning_filename(
                self._minimum.freq if self._minimum else 0, datetime.now()
            )
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def suggested_path(self) -> Path:
        """Where the export dialog should start, never an existing file."""
        directory = self.app.settings.value("SaveDirectory", "", str)
        base = Path(directory) if directory else Path.home()
        return unique_path(
            base / tuning_filename(self._minimum.freq if self._minimum else 0)
        )

    def export_to(self, path: Path | str, *, overwrite: bool = False) -> Path:
        """Write the completed measurement; raises ``AntennaExportError``."""
        if self._snapshot is None:
            raise AntennaExportError(
                "No completed antenna measurement to export."
            )
        if self._unusable:
            raise AntennaExportError(
                f"{self._unusable} sample(s) are not finite; the measurement"
                " cannot be written as a Touchstone file"
            )
        target = write_s1p(
            self._snapshot,
            path,
            minimum=self._minimum,
            allow_partial=False,
            overwrite=overwrite,
        )
        self._last_written = target
        self._set_status(f"Exported {target}")
        logger.info("Antenna measurement exported to %s", target)
        return target

    def choose_and_export(self) -> Path | None:
        """Ask for a filename; never replace a file without asking.

        The dialog may return a name whose suffix is not ``.s1p``; the writer
        normalizes it, so the existence check, the confirmation and the
        granted replacement authority all use that *effective* path.  Asking
        about (and authorising) ``named.s2p`` while the writer destroys
        ``named.s1p`` would be a silent loss of a different file.

        The native dialog's own overwrite confirmation is disabled so the
        user is asked exactly once, about exactly the file that is written.
        """
        if self._snapshot is None:
            self._set_status("Run a sweep before exporting.", error=True)
            return None
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export antenna measurement",
            str(self.suggested_path()),
            "Touchstone 1-port (*.s1p);;All files (*.*)",
            options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite,
        )
        if not filename:
            return None
        target = normalize_s1p_path(filename)
        if target.suffix != Path(filename).suffix:
            self._set_status(
                f"{Path(filename).name} is not a Touchstone 1-port name;"
                f" the measurement is written as {target.name}."
            )
        replace = target.exists()
        if replace and not confirm_overwrite(self, str(target)):
            self._set_status(f"Export cancelled; {target.name} is unchanged.")
            return None
        try:
            return self.export_to(target, overwrite=replace)
        except (AntennaExportError, OSError) as exc:
            self.app.showError(f"Unable to export the measurement: {exc}")
            return None


def confirm_overwrite(parent, filename: str) -> bool:
    """Ask before replacing an existing file; the default keeps it."""
    answer = QtWidgets.QMessageBox.question(
        parent,
        "File exists",
        f"{filename} already exists. Replace it?",
        QtWidgets.QMessageBox.StandardButton.Yes
        | QtWidgets.QMessageBox.StandardButton.No,
        QtWidgets.QMessageBox.StandardButton.No,
    )
    return answer == QtWidgets.QMessageBox.StandardButton.Yes


def _run_identity(snapshot, pending: "PendingStep") -> str:
    """Identity of the run that produced ``snapshot``.

    The pending step's token is the only thing that can prove ownership, but
    it must never be serialised.  Its ``id`` is unique for the lifetime of
    the process and the snapshot id is unique per capture, so together they
    identify the run for the one purpose the history needs: recognising a
    duplicated completion inside this session.
    """
    token = pending.token
    return f"{token.__class__.__name__}-{id(token):x}-{snapshot.id}"


def _same_value(a: float, b: float) -> bool:
    """Equality that treats two NaNs as the same stored value."""
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _same_point(point, other) -> bool:
    return (
        point.freq == other.freq
        and _same_value(point.z.real, other.z.real)
        and _same_value(point.z.imag, other.z.imag)
    )


def _describes_samples(snapshot, samples) -> bool:
    """Whether ``snapshot``'s S11 channel is exactly ``samples``."""
    points = snapshot.channel("S11") or ()
    if len(points) != len(samples):
        return False
    return all(
        _same_point(point, other)
        for point, other in zip(points, samples, strict=True)
    )
