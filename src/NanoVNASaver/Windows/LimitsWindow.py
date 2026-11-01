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
"""Limit set window (LIMIT-03).

A thin view over :mod:`NanoVNASaver.Limits`.  The catalogue owns validation
and persistence, so this window only loads files, elects the active set and
shows the frozen result of checking the captured traces.  It never changes a
measurement and never starts a sweep.
"""

import logging
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from ..Limits import (
    LIMIT_SUFFIX,
    LimitCatalogue,
    LimitError,
    LimitEvaluation,
    LimitSet,
    Verdict,
)
from .Defaults import make_scrollable
from .ui import get_window_icon

if TYPE_CHECKING:
    from ..NanoVNASaver.NanoVNASaver import NanoVNASaver as vna_app

logger = logging.getLogger(__name__)

LIMIT_FILE_FILTER = (
    "NanoVNA-Saver limit sets (*" + LIMIT_SUFFIX + ");;All files (*)"
)

#: Marker appended to the list entry of the active limit set.
ACTIVE_SUFFIX = " (active)"


def describe_limit_set(limit_set: LimitSet) -> str:
    """Read-only summary of every rule in a limit set."""
    lines = [limit_set.name, ""]
    if limit_set.description:
        lines.extend([limit_set.description, ""])
    lines.append(f"Format version: {limit_set.version}")
    lines.append(f"Rules: {len(limit_set.rules)}")
    for rule in limit_set.rules:
        lines.append("")
        lines.append(f"{rule.id}: {rule.label or rule.metric.description}")
        for segment in rule.segments:
            bounds = []
            if segment.minimum is not None:
                bounds.append(f"min {segment.minimum:g}")
            if segment.maximum is not None:
                bounds.append(f"max {segment.maximum:g}")
            unit = f" {rule.metric.unit}" if rule.metric.unit else ""
            lines.append(
                f"  {segment.start} Hz to {segment.stop} Hz:"
                f" {', '.join(bounds)}{unit}"
            )
    return "\n".join(lines)


def describe_result(result: LimitEvaluation) -> str:
    return result.describe()


class LimitsWindow(QtWidgets.QWidget):
    """Load limit sets, elect one, and inspect the pass/fail results."""

    def __init__(
        self,
        app: "vna_app",
        catalogue: LimitCatalogue | None = None,
    ):
        super().__init__()

        self.app = app
        if catalogue is None:
            catalogue = getattr(app, "limits", None)
        # An empty catalogue is falsy, so it must be picked explicitly.
        self.catalogue = (
            catalogue if catalogue is not None else LimitCatalogue()
        )
        self.message = ""
        #: Last evaluated results, kept for tests and for the detail view.
        self.results: tuple[LimitEvaluation, ...] = ()

        self.setWindowTitle("Measurement limits")
        self.setWindowIcon(get_window_icon())
        self.setMinimumWidth(420)
        QtGui.QShortcut(QtCore.Qt.Key.Key_Escape, self, self.hide)

        layout = QtWidgets.QVBoxLayout()
        make_scrollable(self, layout)

        layout.addWidget(QtWidgets.QLabel("Loaded limit sets"))
        self.list_sets = QtWidgets.QListWidget()
        self.list_sets.currentItemChanged.connect(
            lambda *_: self._show_details()
        )
        layout.addWidget(self.list_sets)

        transfer = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load ...")
        self.btn_load.setToolTip(
            "Load a versioned JSON limit set; invalid files are rejected"
        )
        self.btn_load.clicked.connect(lambda: self.load_file())
        self.btn_remove = QtWidgets.QPushButton("Remove")
        self.btn_remove.clicked.connect(self.remove_selected)
        transfer.addWidget(self.btn_load)
        transfer.addWidget(self.btn_remove)
        layout.addLayout(transfer)

        use = QtWidgets.QHBoxLayout()
        self.btn_activate = QtWidgets.QPushButton("Use for reports")
        self.btn_activate.clicked.connect(self.activate_selected)
        self.btn_deactivate = QtWidgets.QPushButton("No limits")
        self.btn_deactivate.clicked.connect(self.deactivate)
        use.addWidget(self.btn_activate)
        use.addWidget(self.btn_deactivate)
        layout.addLayout(use)

        self.label_active = QtWidgets.QLabel("")
        self.label_active.setWordWrap(True)
        layout.addWidget(self.label_active)

        layout.addWidget(QtWidgets.QLabel("Selected limit set"))
        self.detail = QtWidgets.QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMinimumHeight(140)
        layout.addWidget(self.detail)

        self.btn_check = QtWidgets.QPushButton("Check captured traces")
        self.btn_check.clicked.connect(self.check_traces)
        layout.addWidget(self.btn_check)

        self.label_verdict = QtWidgets.QLabel("")
        self.label_verdict.setWordWrap(True)
        self.label_verdict.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.label_verdict)

        self.results_view = QtWidgets.QPlainTextEdit()
        self.results_view.setReadOnly(True)
        self.results_view.setMinimumHeight(140)
        layout.addWidget(self.results_view)

        self.label_message = QtWidgets.QLabel("")
        self.label_message.setObjectName("limits_message")
        self.label_message.setWordWrap(True)
        layout.addWidget(self.label_message)

        layout.addStretch(1)
        self.refresh()

    # ------------------------------------------------------------------
    # Catalogue handling
    # ------------------------------------------------------------------
    def refresh(self, select: str = "") -> None:
        """Rebuild the list while keeping a sensible selection."""
        current = select or self.selected_name() or self.catalogue.active_name
        self.list_sets.blockSignals(True)
        self.list_sets.clear()
        for limit_set in self.catalogue.all():
            suffix = ACTIVE_SUFFIX if self.is_active(limit_set.name) else ""
            item = QtWidgets.QListWidgetItem(limit_set.name + suffix)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, limit_set.name)
            self.list_sets.addItem(item)
        self.list_sets.blockSignals(False)

        names = self.catalogue.names()
        if current in names:
            self.select(current)
        elif names:
            self.list_sets.setCurrentRow(0)
        else:
            self._show_details()
        self._update_active_label()

    def selected_name(self) -> str:
        item = self.list_sets.currentItem()
        if item is None:
            return ""
        return str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")

    def selected_limit_set(self) -> LimitSet | None:
        return self.catalogue.find(self.selected_name())

    def select(self, name: str) -> bool:
        for row in range(self.list_sets.count()):
            item = self.list_sets.item(row)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == name:
                self.list_sets.setCurrentItem(item)
                return True
        return False

    def is_active(self, name: str) -> bool:
        return bool(name) and name == self.catalogue.active_name

    def _show_details(self) -> None:
        limit_set = self.selected_limit_set()
        self.detail.setPlainText(
            describe_limit_set(limit_set) if limit_set else ""
        )

    def _set_message(self, text: str) -> None:
        self.message = text
        self.label_message.setText(text)

    def _update_active_label(self) -> None:
        active = self.catalogue.active()
        if active is None:
            self.label_active.setText(
                "No limit set is active: reports are built without limit "
                "checks."
            )
            return
        self.label_active.setText(
            f"Active for reports: {active.name} ({len(active.rules)} rule(s))."
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_file(self, filename: str = "") -> LimitSet | None:
        """Ask for a file (unless given one), validate and store it.

        A rejected file leaves the catalogue untouched and reports the
        reason verbatim.
        """
        if not filename:
            filename, _selected = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load limit set", "", LIMIT_FILE_FILTER
            )
            if not filename:
                return None
        try:
            limit_set = self.catalogue.load(filename)
        except LimitError as exc:
            logger.warning("Rejected limit set %s: %s", filename, exc)
            self._set_message(str(exc))
            return None
        self.refresh(select=limit_set.name)
        self._set_message(f"Loaded limit set {limit_set.name!r}.")
        return limit_set

    def remove_selected(self) -> bool:
        name = self.selected_name()
        if not name:
            self._set_message("Select a limit set first.")
            return False
        removed = self.catalogue.remove(name)
        if removed:
            self.refresh()
            self._set_message(f"Removed limit set {name!r}.")
        return removed

    def activate_selected(self) -> LimitSet | None:
        name = self.selected_name()
        if not name:
            self._set_message("Select a limit set first.")
            return None
        limit_set = self.catalogue.activate(name)
        self.refresh(select=name)
        self._set_message(f"Reports now use limit set {name!r}.")
        return limit_set

    def deactivate(self) -> None:
        self.catalogue.deactivate()
        self.refresh()
        self._set_message("Reports are built without limit checks.")

    def active_limit_set(self) -> LimitSet | None:
        return self.catalogue.active()

    # ------------------------------------------------------------------
    # Checking
    # ------------------------------------------------------------------
    def check_traces(self) -> tuple[LimitEvaluation, ...]:
        """Evaluate the active set against every captured trace."""
        limit_set = self.catalogue.active()
        if limit_set is None:
            self.results = ()
            self.results_view.setPlainText("")
            self.label_verdict.setText("No limit set is active.")
            return ()
        results: list[LimitEvaluation] = []
        for trace in self.app.trace_control.collection:
            results.extend(
                limit_set.evaluate(
                    trace.snapshot,
                    trace_id=trace.id,
                    trace_name=trace.name,
                )
            )
        self.results = tuple(results)
        self._show_results(limit_set)
        return self.results

    def _show_results(self, limit_set: LimitSet) -> None:
        if not self.results:
            self.label_verdict.setText(
                f"Limit set {limit_set.name!r}: no traces to check."
            )
            self.results_view.setPlainText("")
            return
        lines = [
            f"{result.trace_name}: {describe_result(result)}"
            for result in self.results
        ]
        self.results_view.setPlainText("\n".join(lines))
        failed = sum(1 for item in self.results if item.verdict is Verdict.FAIL)
        unavailable = sum(
            1 for item in self.results if item.verdict is Verdict.UNAVAILABLE
        )
        if failed:
            verdict = Verdict.FAIL
        elif unavailable:
            verdict = Verdict.UNAVAILABLE
        else:
            verdict = Verdict.PASS
        self.label_verdict.setText(
            f"{verdict.label}: {len(self.results) - failed - unavailable}"
            f" pass, {failed} fail, {unavailable} unavailable"
        )
