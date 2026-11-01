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
"""Trace controls (TRACE-02.1).

The list is the single place where traces are captured, renamed, hidden,
reordered, deleted and chosen as the comparison reference.  It only edits
the :class:`TraceCollection`; the main window reacts to ``changed``.
"""

import logging
import os
from typing import TYPE_CHECKING

from PySide6 import QtGui, QtWidgets

from ..Measurement import SPARAMETERS as PARAMETERS
from ..Traces import (
    ComparisonCache,
    ComparisonKind,
    TraceCollection,
    TraceKind,
    marker_delta,
)
from .Control import Control

if TYPE_CHECKING:
    from ..NanoVNASaver.NanoVNASaver import NanoVNASaver as vna_app

logger = logging.getLogger(__name__)

#: Chart types that can display several traces at once (TRACE-02.3).
MULTI_TRACE_CHARTS = (
    "S11 Smith Chart",
    "S11 Polar",
    "S11 VSWR",
    "S11 Return Loss",
    "S11 Impedance",
    "S11 Phase",
    "S21 Gain",
    "S21 Phase",
    "S21 Group Delay",
)

#: Charts that currently show a single trace only.
SINGLE_TRACE_CHARTS = ("TDR",)


class TraceControl(Control):
    def __init__(self, app: "vna_app"):
        super().__init__(app, "Trace control")
        self.collection = TraceCollection()

        self.list_traces = QtWidgets.QListWidget()
        self.list_traces.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.list_traces.currentItemChanged.connect(self._selection_changed)
        self.list_traces.itemChanged.connect(self._item_changed)
        self.layout.addRow(self.list_traces)

        self.btn_capture = QtWidgets.QPushButton("Capture current sweep")
        self.btn_capture.clicked.connect(self.capture_current)
        self.btn_import = QtWidgets.QPushButton("Import file ...")
        self.btn_import.clicked.connect(self.import_file)
        self.btn_rename = QtWidgets.QPushButton("Rename ...")
        self.btn_rename.clicked.connect(self.rename_selected)
        self.btn_delete = QtWidgets.QPushButton("Delete")
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_reference = QtWidgets.QPushButton("Use as reference")
        self.btn_reference.clicked.connect(self.set_reference)
        self.btn_up = QtWidgets.QPushButton("Move up")
        self.btn_up.clicked.connect(lambda: self.move_selected(-1))
        self.btn_down = QtWidgets.QPushButton("Move down")
        self.btn_down.clicked.connect(lambda: self.move_selected(1))

        buttons = QtWidgets.QGridLayout()
        buttons.addWidget(self.btn_capture, 0, 0, 1, 2)
        buttons.addWidget(self.btn_import, 4, 0, 1, 2)
        buttons.addWidget(self.btn_rename, 1, 0)
        buttons.addWidget(self.btn_delete, 1, 1)
        buttons.addWidget(self.btn_reference, 2, 0, 1, 2)
        buttons.addWidget(self.btn_up, 3, 0)
        buttons.addWidget(self.btn_down, 3, 1)
        widget = QtWidgets.QWidget()
        widget.setLayout(buttons)
        self.layout.addRow(widget)

        # -- comparison (TRACE-03/04) ----------------------------------
        self.cache = ComparisonCache()
        self.cb_comparison = QtWidgets.QComboBox()
        for kind in ComparisonKind:
            self.cb_comparison.addItem(kind.label, kind)
        self.cb_comparison.currentIndexChanged.connect(self._update_comparison)
        self.layout.addRow(QtWidgets.QLabel("Compare"), self.cb_comparison)

        self.label_comparison = QtWidgets.QLabel("")
        self.label_comparison.setWordWrap(True)
        self.layout.addRow(self.label_comparison)

        self.label_limit = QtWidgets.QLabel("")
        self.label_limit.setWordWrap(True)
        self.layout.addRow(self.label_limit)
        self._set_limitation_text()

        self.refresh()

    # -- model access --------------------------------------------------
    @property
    def traces(self) -> TraceCollection:
        return self.collection

    def selected_id(self) -> str:
        item = self.list_traces.currentItem()
        return (
            item.data(QtWidgets.QListWidgetItem.ItemType.UserType)
            if (item)
            else ""
        )

    def _set_limitation_text(self) -> None:
        self.label_limit.setText(
            "Simultaneous comparison is available on: "
            + ", ".join(MULTI_TRACE_CHARTS[:3])
            + ", ..."
            f" Not yet available on: {', '.join(SINGLE_TRACE_CHARTS)}"
            " (shows the selected trace only)."
        )

    # -- rendering -----------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the list from the collection, preserving selection."""
        current = self.selected_id()
        self.list_traces.blockSignals(True)
        self.list_traces.clear()
        for trace_id in self.collection.ordered_ids():
            trace = self.collection.get(trace_id)
            if trace is None:
                continue
            item = QtWidgets.QListWidgetItem(self._label_for(trace))
            item.setData(QtWidgets.QListWidgetItem.ItemType.UserType, trace.id)
            item.setFlags(item.flags() | QtGui.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtGui.Qt.CheckState.Checked
                if trace.visible
                else QtGui.Qt.CheckState.Unchecked
            )
            color = QtGui.QColor(trace.color)
            item.setForeground(color)
            if self.collection.reference is trace:
                font = item.font()
                font.setBold(True)
                font.setItalic(True)
                item.setFont(font)
            self.list_traces.addItem(item)
            if trace.id == current:
                self.list_traces.setCurrentItem(item)
        self.list_traces.blockSignals(False)

    def _label_for(self, trace) -> str:
        reference = self.collection.reference
        marker = (
            " [ref]"
            if reference is not None and reference.id == trace.id
            else ""
        )
        channel = " ".join(trace.snapshot.available) or "no data"
        return f"{trace.name}{marker} ({channel})"

    def current_comparison(self):
        """Compare the reference against the selected trace (TRACE-03)."""
        source_id = self.selected_id()
        reference = self.collection.reference
        if not source_id or reference is None:
            return None
        if source_id == reference.id:
            return None
        source = self.collection.get(source_id)
        if source is None:
            return None
        kind = self.cb_comparison.currentData() or ComparisonKind.MAGNITUDE_DB
        # Compare on a channel both traces actually carry (TRACE-02.4).
        shared = [
            parameter
            for parameter in ("S11", "S21")
            if source.has_channel(parameter)
            and reference.has_channel(parameter)
        ]
        if not shared:
            return None
        return self.cache.get(source, reference, shared[0], kind)

    def _update_comparison(self, *_args) -> None:
        result = self.current_comparison()
        if result is None:
            self.label_comparison.setText(
                "Select a trace and a reference to compare."
            )
            self.updated.emit(self)
            return
        self.label_comparison.setText(result.summary())
        self.updated.emit(self)

    def comparison_at(self, frequency: int):
        """Marker delta at ``frequency`` for the active comparison."""
        result = self.current_comparison()
        if result is None:
            return None
        return marker_delta(result, frequency)

    def trace_series(self, parameter: str):
        """``(name, color, points)`` for every visible trace."""
        return [
            (trace.name, trace.color, list(trace.channel(parameter)))
            for trace in self.collection.visible()
            if trace.has_channel(parameter)
        ]

    # -- actions -------------------------------------------------------
    def capture_current(self) -> None:
        snapshot = self.app.current_snapshot()
        if snapshot is None:
            self.app.showError(
                "There is no completed measurement to capture. Connect a"
                " device or load a file first."
            )
            return
        self.collection.add(snapshot)
        self.app.mark_session_changed("trace_added")
        self.refresh()
        self.updated.emit(self)

    def import_file(self) -> None:
        """Import a Touchstone file as a frozen trace (TRACE-02.1)."""
        filename, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import a Touchstone file as a trace",
            "",
            "Touchstone Files [s1p, s2p] (*.s1p *.s2p *.S1P *.S2P)",
        )
        if not filename:
            return
        self.import_path(filename)

    def import_path(self, filename: str) -> bool:
        """Import one Touchstone file as a frozen trace.

        Returns ``True`` when a trace was added.  ``Touchstone.load`` logs
        unreadable files instead of raising, so the loaded channels are
        checked rather than trusting the call to fail loudly.
        """
        from ..Measurement import snapshot_from_touchstone
        from ..Touchstone import Touchstone

        touchstone = Touchstone(filename)
        try:
            touchstone.load()
        except (OSError, ValueError, IndexError) as error:
            self.app.showError(f"Could not read {filename}:\n{error}")
            return False
        if not any(touchstone.channel(parameter) for parameter in PARAMETERS):
            self.app.showError(f"{filename} contained no S-parameter data.")
            return False
        snapshot = snapshot_from_touchstone(
            touchstone,
            name=os.path.basename(filename),
            source=filename,
            simulated=getattr(touchstone, "simulated", False),
            simulator_scenario=getattr(touchstone, "simulated_scenario", "")
            or None,
        )
        if not snapshot.available:
            self.app.showError(f"{filename} contained no S-parameter data.")
            return False
        self.collection.add(
            snapshot,
            name=os.path.basename(filename),
            kind=TraceKind.IMPORTED,
        )
        self.app.mark_session_changed("trace_imported")
        self.refresh()
        self.updated.emit(self)
        return True

    def rename_selected(self) -> None:
        trace_id = self.selected_id()
        if not trace_id:
            return
        trace = self.collection.get(trace_id)
        name, accepted = QtWidgets.QInputDialog.getText(
            self, "Rename trace", "Name:", text=trace.name
        )
        if not accepted or not name.strip():
            return
        self.collection.rename(trace_id, name.strip())
        self.app.mark_session_changed("trace_renamed")
        self.refresh()
        self.updated.emit(self)

    def delete_selected(self) -> None:
        trace_id = self.selected_id()
        if not trace_id:
            return
        self.collection.remove(trace_id)
        self.app.mark_session_changed("trace_removed")
        self.refresh()
        self.updated.emit(self)

    def set_reference(self) -> None:
        trace_id = self.selected_id()
        if not trace_id:
            return
        self.collection.set_reference(trace_id)
        self.app.mark_session_changed("reference")
        self.refresh()
        self.updated.emit(self)

    def move_selected(self, offset: int) -> None:
        trace_id = self.selected_id()
        if not trace_id:
            return
        order = self.collection.ordered_ids()
        index = order.index(trace_id) + offset
        if not 0 <= index < len(order):
            return
        self.collection.reorder(trace_id, index)
        self.app.mark_session_changed("trace_reordered")
        self.refresh()
        self.updated.emit(self)

    # -- list signals --------------------------------------------------
    def _selection_changed(self, *_args) -> None:
        self._update_comparison()
        trace_id = self.selected_id()
        self.btn_rename.setEnabled(bool(trace_id))
        self.btn_delete.setEnabled(bool(trace_id))
        self.btn_reference.setEnabled(bool(trace_id))
        self.btn_up.setEnabled(bool(trace_id))
        self.btn_down.setEnabled(bool(trace_id))

    def _item_changed(self, item) -> None:
        trace_id = item.data(QtWidgets.QListWidgetItem.ItemType.UserType)
        if not trace_id:
            return
        visible = item.checkState() == QtGui.Qt.CheckState.Checked
        self.collection.set_visible(trace_id, visible)
        self.app.mark_session_changed("trace_visibility")
        self.updated.emit(self)
