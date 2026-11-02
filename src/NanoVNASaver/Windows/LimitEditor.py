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
"""Limit set editor (NEXT-04).

The editor is a *transaction* over an immutable :class:`LimitSet`.  Editing
builds candidate documents from the current one; nothing reaches the catalogue
until the operator applies, and a candidate that the engine rejects is never
stored.  That is what keeps a frozen report safe: it holds its own limit set,
and the catalogue only ever holds validated, complete documents.

Units and signs are shown next to every bound rather than being implied,
because the engine's conventions are the ones that decide a PASS:

* VSWR is a ratio with no unit; a lower maximum is a tighter match.
* Return loss and insertion loss are **positive** for a match / a passive
  path, so a gain is a negative insertion loss.
* Phase is wrapped degrees in ``[-180, +180]``.

A slope is entered per gigahertz (which is how a specification is written) and
stored per hertz (which is display independent).
"""

from __future__ import annotations

import logging

from PySide6 import QtWidgets

from ..Limits import (
    LIMIT_SET_VERSION,
    LimitRule,
    LimitSegment,
    LimitSet,
    LimitValidationError,
    Metric,
    save,
)
from .ui import get_window_icon

logger = logging.getLogger(__name__)

#: Hz in one GHz; used to convert a displayed slope to the stored one.
HZ_PER_GHZ = 1e9

#: Human readable unit and sign note per metric, shown beside the inputs.
UNIT_NOTES = {
    Metric.S11_VSWR: (
        "VSWR",
        "Ratio, no unit. Lower is a better match; a maximum is usual.",
    ),
    Metric.S11_RETURN_LOSS_DB: (
        "dB",
        "Positive for a match (-20log|S11|). A gain reads negative, so "
        "rejecting gain needs a minimum.",
    ),
    Metric.S21_INSERTION_LOSS_DB: (
        "dB",
        "Positive for loss (-20log|S21|). A gain reads negative, so "
        "rejecting gain needs a minimum.",
    ),
    Metric.S11_PHASE_DEG: (
        "deg",
        "Wrapped into -180..+180. Meaningful only where the measurement "
        "does not cross the branch.",
    ),
}


def _parse_frequency(text: str) -> int:
    """Parse a frequency in Hz, allowing ``12.5M``-style suffixes."""
    raw = str(text).strip().replace(" ", "")
    if not raw:
        raise ValueError("frequency is required")
    multipliers = {
        "g": 1_000_000_000,
        "m": 1_000_000,
        "k": 1_000,
        "hz": 1,
    }
    lowered = raw.lower()
    factor = 1
    for suffix, scale in multipliers.items():
        if lowered.endswith(suffix) and len(lowered) > len(suffix):
            lowered = lowered[: -len(suffix)]
            factor = scale
            break
    value = float(lowered) * factor
    if value < 0:
        raise ValueError("frequency must not be negative")
    if abs(value - round(value)) > 1e-6:
        raise ValueError("frequency must be a whole number of hertz")
    return int(round(value))


def _parse_optional(text: str) -> float | None:
    raw = str(text).strip()
    if not raw:
        return None
    return float(raw)


def _parse_slope(text: str) -> float:
    """A slope typed per gigahertz, returned per hertz."""
    raw = str(text).strip()
    if not raw:
        return 0.0
    return float(raw) / HZ_PER_GHZ


def _format_bound(value: float) -> str:
    """The shortest text that parses back to exactly ``value``.

    Editing re-parses this text, and a bound is a specification: rounding it
    for display would change a limit nobody meant to touch.
    """
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _format_slope(slope_per_hz: float) -> str:
    if not slope_per_hz:
        return ""
    return _format_bound(slope_per_hz * HZ_PER_GHZ)


def _read_slope(text: str, original: float) -> float:
    """``text`` as a slope, or ``original`` when it was left as shown.

    Scaling to per-GHz and back can move the last bit even though the text
    round-trips, so an untouched field keeps the stored value itself.
    """
    if text.strip() == _format_slope(original):
        return original
    return _parse_slope(text)


class SegmentRow:
    """One editable segment row; reads back into a candidate segment."""

    def __init__(self, segment: LimitSegment):
        self.segment = segment
        self.edit_start = QtWidgets.QLineEdit(str(segment.start))
        self.edit_stop = QtWidgets.QLineEdit(str(segment.stop))
        self.edit_min = QtWidgets.QLineEdit(
            "" if segment.minimum is None else _format_bound(segment.minimum)
        )
        self.edit_max = QtWidgets.QLineEdit(
            "" if segment.maximum is None else _format_bound(segment.maximum)
        )
        self.edit_min_slope = QtWidgets.QLineEdit(
            _format_slope(segment.minimum_slope)
        )
        self.edit_max_slope = QtWidgets.QLineEdit(
            _format_slope(segment.maximum_slope)
        )
        for edit in (
            self.edit_min_slope,
            self.edit_max_slope,
        ):
            edit.setToolTip("Change per GHz; empty means flat")
        self.widget = QtWidgets.QWidget()
        row = QtWidgets.QGridLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for column, (label, edit) in enumerate(
            (
                ("Start (Hz)", self.edit_start),
                ("Stop (Hz)", self.edit_stop),
                ("Min", self.edit_min),
                ("Max", self.edit_max),
                ("Min slope /GHz", self.edit_min_slope),
                ("Max slope /GHz", self.edit_max_slope),
            )
        ):
            row.addWidget(QtWidgets.QLabel(label), 0, column)
            row.addWidget(edit, 1, column)
        self.widget.setLayout(row)
        self.remove_button = QtWidgets.QPushButton("Remove segment")

    def candidate(self) -> LimitSegment:
        return LimitSegment(
            start=_parse_frequency(self.edit_start.text()),
            stop=_parse_frequency(self.edit_stop.text()),
            minimum=_parse_optional(self.edit_min.text()),
            maximum=_parse_optional(self.edit_max.text()),
            minimum_slope=_read_slope(
                self.edit_min_slope.text(), self.segment.minimum_slope
            ),
            maximum_slope=_read_slope(
                self.edit_max_slope.text(), self.segment.maximum_slope
            ),
        )


class RuleBox(QtWidgets.QGroupBox):
    """One editable rule: a metric plus its piecewise segments."""

    def __init__(self, rule: LimitRule, parent=None):
        super().__init__(rule.label or rule.id, parent)
        self.rule = rule
        self.rows: list[SegmentRow] = []

        outer = QtWidgets.QVBoxLayout()

        header = QtWidgets.QFormLayout()
        self.edit_id = QtWidgets.QLineEdit(rule.id)
        self.edit_label = QtWidgets.QLineEdit(rule.label)
        self.combo_metric = QtWidgets.QComboBox()
        for metric in Metric:
            self.combo_metric.addItem(metric.description, metric.value)
        index = self.combo_metric.findData(rule.metric.value)
        self.combo_metric.setCurrentIndex(max(index, 0))
        header.addRow("Rule id", self.edit_id)
        header.addRow("Label", self.edit_label)
        header.addRow("Metric", self.combo_metric)
        unit, note = UNIT_NOTES.get(rule.metric, ("", ""))
        self.label_unit = QtWidgets.QLabel(f"Unit: {unit or '(none)'}")
        self.label_note = QtWidgets.QLabel(note)
        self.label_note.setWordWrap(True)
        header.addRow(self.label_unit)
        header.addRow(self.label_note)
        outer.addLayout(header)

        self.segments_box = QtWidgets.QVBoxLayout()
        outer.addLayout(self.segments_box)
        for segment in rule.segments:
            self.add_row(segment)
        add_segment = QtWidgets.QPushButton("Add segment")
        add_segment.clicked.connect(lambda: self.add_row(None))
        outer.addWidget(add_segment)
        self.setLayout(outer)
        self.combo_metric.currentIndexChanged.connect(self._metric_changed)

    def _metric_changed(self) -> None:
        metric = Metric.parse(self.combo_metric.currentData())
        unit, note = UNIT_NOTES.get(metric, ("", ""))
        self.label_unit.setText(f"Unit: {unit or '(none)'}")
        self.label_note.setText(note)

    def metric(self) -> Metric:
        return Metric.parse(self.combo_metric.currentData())

    def add_row(self, segment: LimitSegment | None) -> SegmentRow:
        if segment is None:
            # A new segment starts where the last one ended, so the default is
            # a tiling document rather than one that must be repaired first.
            last = self.rows[-1].segment if self.rows else None
            start = last.stop if last else 0
            segment = LimitSegment(
                start=start, stop=start + 1_000_000, maximum=1.0
            )
        row = SegmentRow(segment)
        row.remove_button.clicked.connect(
            lambda _checked=False, target=row: self.remove_row(target)
        )
        row.widget.layout().addWidget(row.remove_button, 2, 0, 1, 6)
        self.rows.append(row)
        self.segments_box.addWidget(row.widget)
        return row

    def remove_row(self, row: SegmentRow) -> None:
        if row in self.rows:
            self.rows.remove(row)
            row.widget.setParent(None)
            row.widget.deleteLater()

    def candidate(self) -> LimitRule:
        if not self.rows:
            raise LimitValidationError("a rule needs at least one segment")
        return LimitRule(
            metric=self.metric(),
            segments=tuple(row.candidate() for row in self.rows),
            label=self.edit_label.text().strip(),
            id=self.edit_id.text().strip() or "rule",
        )


class LimitEditor(QtWidgets.QDialog):
    """Edit one limit set as a transaction; returns the applied candidate."""

    def __init__(self, limit_set: LimitSet, parent=None):
        super().__init__(parent)
        self.original = limit_set
        #: The applied result, or ``None`` when the dialog was cancelled.
        self.result_set: LimitSet | None = None
        self.message = ""

        self.setWindowTitle(f"Edit limit set: {limit_set.name}")
        self.setWindowIcon(get_window_icon())
        self.setMinimumWidth(640)

        layout = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        self.edit_name = QtWidgets.QLineEdit(limit_set.name)
        self.edit_description = QtWidgets.QPlainTextEdit(limit_set.description)
        self.edit_description.setFixedHeight(56)
        form.addRow("Name", self.edit_name)
        form.addRow("Description", self.edit_description)
        layout.addLayout(form)

        layout.addWidget(QtWidgets.QLabel("Rules"))
        self.rules_area = QtWidgets.QVBoxLayout()
        self.rule_boxes: list[RuleBox] = []
        for rule in limit_set.rules:
            self._add_rule_box(rule)
        layout.addLayout(self.rules_area)
        add_rule = QtWidgets.QPushButton("Add rule")
        add_rule.clicked.connect(self.add_rule)
        layout.addWidget(add_rule)

        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMinimumHeight(120)
        layout.addWidget(QtWidgets.QLabel("Preview (validated)"))
        layout.addWidget(self.preview)

        self.label_error = QtWidgets.QLabel("")
        self.label_error.setWordWrap(True)
        layout.addWidget(self.label_error)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_preview = QtWidgets.QPushButton("Preview")
        self.btn_preview.clicked.connect(self.preview_candidate)
        self.btn_save = QtWidgets.QPushButton("Save to file ...")
        self.btn_save.clicked.connect(self.save_candidate)
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.clicked.connect(self.apply)
        self.btn_apply.setDefault(True)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        for widget in (
            self.btn_preview,
            self.btn_save,
            self.btn_apply,
            self.btn_cancel,
        ):
            buttons.addWidget(widget)
        layout.addLayout(buttons)
        self.setLayout(layout)

    # -- rules ---------------------------------------------------------
    def _add_rule_box(self, rule: LimitRule) -> RuleBox:
        box = RuleBox(rule)
        self.rule_boxes.append(box)
        self.rules_area.addWidget(box)
        return box

    def add_rule(self) -> RuleBox:
        start = self.original.rules[0].start if self.original.rules else 0
        stop = self.original.rules[0].stop if self.original.rules else 1_000_000
        rule = LimitRule(
            metric=Metric.S11_VSWR,
            segments=(LimitSegment(start=start, stop=stop, maximum=2.0),),
            id=f"rule-{len(self.rule_boxes) + 1}",
        )
        return self._add_rule_box(rule)

    # -- transactions --------------------------------------------------
    def candidate(self) -> LimitSet:
        """Build and validate a candidate document from the inputs.

        Raises the engine's own validation errors, so the message an operator
        sees is the message a rejected file would produce.
        """
        rules = tuple(box.candidate() for box in self.rule_boxes)
        return LimitSet(
            name=self.edit_name.text().strip(),
            rules=rules,
            description=self.edit_description.toPlainText().strip(),
            version=LIMIT_SET_VERSION,
        )

    def preview_candidate(self) -> LimitSet | None:
        """Validate without touching anything; show the document."""
        try:
            candidate = self.candidate()
        except (LimitValidationError, ValueError) as exc:
            self._show_error(str(exc))
            self.preview.setPlainText("")
            return None
        from ..Windows.LimitsWindow import describe_limit_set

        self._show_error("")
        self.preview.setPlainText(describe_limit_set(candidate))
        return candidate

    def save_candidate(self) -> bool:
        """Preview, then write the candidate to a chosen file."""
        candidate = self.preview_candidate()
        if candidate is None:
            return False
        filename, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save limit set", f"{candidate.name}.nvlimits", "*.nvlimits"
        )
        if not filename:
            return False
        try:
            save(candidate, filename)
        except OSError as exc:
            self._show_error(f"could not write {filename}: {exc}")
            return False
        self._show_error("")
        return True

    def apply(self) -> None:
        candidate = self.preview_candidate()
        if candidate is None:
            return
        self.result_set = candidate
        self.accept()

    def _show_error(self, text: str) -> None:
        self.message = text
        self.label_error.setText(text)


def export_limit_set(limit_set: LimitSet, filename: str) -> None:
    """Write a frozen copy to ``filename`` (used by the window's Export)."""
    save(limit_set, filename)


def duplicate_name(limit_set: LimitSet, taken: set[str]) -> str:
    """A non-colliding name for a duplicated set."""
    base = f"{limit_set.name} copy"
    if base not in taken:
        return base
    index = 2
    while f"{base} {index}" in taken:
        index += 1
    return f"{base} {index}"
