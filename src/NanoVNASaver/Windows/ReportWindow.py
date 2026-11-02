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
"""Report preview and export (REPORT-02.1, REPORT-02.5).

The window freezes the selected traces, their analyses and their calibration
coverage into one immutable :class:`ReportModel` and reuses that model for
the preview and for every export taken from it.  A later sweep therefore
cannot change a report that is already on screen; the user has to press
"Refresh" to rebuild a model from the current traces.

The preview deliberately renders the HTML without the plots: ``QTextBrowser``
drops inline SVG silently, so a preview that claimed to show them would be
broken.  The plots are part of the exported file and the window says so.
"""

import logging
from typing import TYPE_CHECKING, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from ..Analysis.Results import AnalysisKind, analyse_snapshot
from ..CalibrationCoverage import coverage_from_calibration
from ..Limits import LimitSet, Verdict
from ..Reports import (
    CorrectionKind,
    ReportModel,
    ReportTrace,
    build_report,
    correction_kind,
    render_html,
    report_plots,
    write_csv,
    write_html,
)
from .Defaults import make_scrollable
from .ui import get_window_icon

if TYPE_CHECKING:
    from ..NanoVNASaver.NanoVNASaver import NanoVNASaver as vna_app
    from ..Traces import Trace

logger = logging.getLogger(__name__)

#: Analyses that describe a reflection (S11) measurement.
S11_ANALYSES = (
    AnalysisKind.RESONANCE,
    AnalysisKind.VSWR,
    AnalysisKind.CABLE,
)

#: Analyses that describe a transmission (S21) measurement.
S21_ANALYSES = (
    AnalysisKind.INSERTION_LOSS,
    AnalysisKind.CUTOFF,
    AnalysisKind.GROUP_DELAY,
)


def analysis_kinds(trace: "Trace") -> list[AnalysisKind]:
    """Analyses that apply to the channels ``trace`` actually carries."""
    available = trace.snapshot.available
    kinds: list[AnalysisKind] = []
    if "S11" in available:
        kinds.extend(S11_ANALYSES)
    if "S21" in available:
        kinds.extend(S21_ANALYSES)
    return kinds


class ReportWindow(QtWidgets.QWidget):
    def __init__(self, app: "vna_app"):
        super().__init__()
        self.app = app
        #: Frozen report; ``None`` until the first freeze.
        self.model: ReportModel | None = None
        self._checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        #: True once the operator picked a limit set explicitly; until then
        #: the selector follows the catalogue's active set.
        self._limits_explicit = False

        self.setWindowTitle("Report")
        self.setWindowIcon(get_window_icon())
        self.setMinimumWidth(360)
        QtGui.QShortcut(QtCore.Qt.Key.Key_Escape, self, self.hide)

        layout = QtWidgets.QVBoxLayout()
        make_scrollable(self, layout)

        form = QtWidgets.QFormLayout()
        self.input_title = QtWidgets.QLineEdit()
        self.input_title.setPlaceholderText("Measurement report")
        self.input_dut = QtWidgets.QLineEdit()
        self.input_dut.setPlaceholderText("Device under test")
        self.input_notes = QtWidgets.QPlainTextEdit()
        self.input_notes.setPlaceholderText("Notes for the report")
        self.input_notes.setFixedHeight(64)
        form.addRow("Title", self.input_title)
        form.addRow("DUT", self.input_dut)
        form.addRow("Notes", self.input_notes)
        layout.addLayout(form)

        for widget in (
            self.input_title,
            self.input_dut,
            self.input_notes,
        ):
            widget.textChanged.connect(self._mark_stale)

        layout.addWidget(QtWidgets.QLabel("Limit set"))
        self.combo_limits = QtWidgets.QComboBox()
        self.combo_limits.setToolTip(
            "Limits frozen into this report. Changing the selection does not"
            ' alter a report that was already built; press "Refresh".'
        )
        self.combo_limits.currentIndexChanged.connect(self._limits_changed)
        layout.addWidget(self.combo_limits)
        self.btn_limits = QtWidgets.QPushButton("Manage limits ...")
        self.btn_limits.clicked.connect(
            lambda: self.app.display_window("limits")
        )
        layout.addWidget(self.btn_limits)

        layout.addWidget(QtWidgets.QLabel("Traces included"))
        self.trace_box = QtWidgets.QVBoxLayout()
        container = QtWidgets.QWidget()
        container.setLayout(self.trace_box)
        layout.addWidget(container)

        self.label_stale = QtWidgets.QLabel(
            "Settings changed since the preview was built. "
            'Press "Refresh" to rebuild the report.'
        )
        self.label_stale.setWordWrap(True)
        self.label_stale.setStyleSheet("color: #b06a00;")
        layout.addWidget(self.label_stale)

        self.label_simulated = QtWidgets.QLabel("")
        self.label_simulated.setWordWrap(True)
        self.label_simulated.setStyleSheet("color: #b00020; font-weight: bold;")
        layout.addWidget(self.label_simulated)

        self.label_uncalibrated = QtWidgets.QLabel("")
        self.label_uncalibrated.setWordWrap(True)
        self.label_uncalibrated.setStyleSheet("color: #b06a00;")
        layout.addWidget(self.label_uncalibrated)

        self.label_limits = QtWidgets.QLabel("")
        self.label_limits.setWordWrap(True)
        self.label_limits.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.label_limits)

        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        layout.addWidget(self.btn_refresh)

        self.preview = QtWidgets.QTextBrowser()
        self.preview.setMinimumHeight(320)
        layout.addWidget(self.preview)

        self.label_plots = QtWidgets.QLabel("")
        self.label_plots.setWordWrap(True)
        layout.addWidget(self.label_plots)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_save_html = QtWidgets.QPushButton("Save HTML ...")
        self.btn_save_html.clicked.connect(self.save_html)
        self.btn_save_csv = QtWidgets.QPushButton("Save CSV ...")
        self.btn_save_csv.clicked.connect(self.save_csv)
        buttons.addWidget(self.btn_save_html)
        buttons.addWidget(self.btn_save_csv)
        layout.addLayout(buttons)

        self.label_status = QtWidgets.QLabel("")
        self.label_status.setWordWrap(True)
        layout.addWidget(self.label_status)

        layout.addStretch(1)
        self.refresh()

    # ------------------------------------------------------------------
    # Trace selection
    # ------------------------------------------------------------------
    def checkboxes(self) -> dict[str, QtWidgets.QCheckBox]:
        return dict(self._checkboxes)

    def limits_choices(self) -> list[LimitSet]:
        """Limit sets offered in the selector."""
        catalogue = getattr(self.app, "limits", None)
        return list(catalogue.all()) if catalogue is not None else []

    def _limits_changed(self, *_args) -> None:
        self._limits_explicit = True
        self._mark_stale()

    def _rebuild_limit_list(self) -> None:
        """Recreate the limit selector, keeping a sensible selection.

        The active catalogue entry is used until the operator picks
        something explicitly, so loading a set becomes visible here without
        a second action, while an explicit "No limits" stays selected.
        """
        if self._limits_explicit:
            previous = self.selected_limit_name()
        else:
            catalogue = getattr(self.app, "limits", None)
            previous = catalogue.active_name if catalogue is not None else ""
        self.combo_limits.blockSignals(True)
        self.combo_limits.clear()
        self.combo_limits.addItem("No limits", "")
        for limit_set in self.limits_choices():
            self.combo_limits.addItem(limit_set.name, limit_set.name)
        index = self.combo_limits.findData(previous)
        self.combo_limits.setCurrentIndex(max(index, 0))
        self.combo_limits.blockSignals(False)

    def limits_changed(self) -> None:
        """The limit catalogue changed in the limits window (NEXT-04).

        The selector is rebuilt so a newly loaded set becomes choosable, but
        an explicit "No limits" choice is preserved: this is a notification,
        not a selection.  An already frozen report is flagged stale rather
        than rebuilt, because a frozen report is evidence of what was shown
        at freeze time and must not change underneath the operator.
        """
        self._rebuild_limit_list()
        self._mark_stale()

    def selected_limit_name(self) -> str:
        return str(self.combo_limits.currentData() or "")

    def selected_limit_set(self) -> LimitSet | None:
        """The set the next freeze will use, or ``None`` for no limits."""
        name = self.selected_limit_name()
        if not name:
            return None
        catalogue = getattr(self.app, "limits", None)
        return catalogue.find(name) if catalogue is not None else None

    def select_limit_set(self, name: str) -> bool:
        index = self.combo_limits.findData(name)
        if index < 0:
            # A set may have been loaded since the selector was built.
            self._rebuild_limit_list()
            index = self.combo_limits.findData(name)
        if index < 0:
            return False
        self._limits_explicit = True
        self.combo_limits.setCurrentIndex(index)
        self._mark_stale()
        return True

    def _rebuild_trace_list(self) -> None:
        """Recreate one checkbox per trace, keeping earlier choices.

        A trace that was not listed before defaults to included.
        """
        previous = {
            trace_id: box.isChecked()
            for trace_id, box in self._checkboxes.items()
        }
        while self.trace_box.count():
            item = self.trace_box.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._checkboxes = {}
        for trace in self.app.trace_control.collection:
            box = QtWidgets.QCheckBox(trace.name)
            box.setChecked(previous.get(trace.id, True))
            box.toggled.connect(self._mark_stale)
            self.trace_box.addWidget(box)
            self._checkboxes[trace.id] = box
        if not self._checkboxes:
            self.trace_box.addWidget(
                QtWidgets.QLabel("No traces captured yet.")
            )

    def selected_traces(self) -> list["Trace"]:
        return [
            trace
            for trace in self.app.trace_control.collection
            if (box := self._checkboxes.get(trace.id)) is not None
            and box.isChecked()
        ]

    def _mark_stale(self, *_args) -> None:
        self.label_stale.setVisible(self.model is not None)

    # ------------------------------------------------------------------
    # Freezing (REPORT-02.5)
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-read the traces and freeze a new report model."""
        self._rebuild_limit_list()
        self._rebuild_trace_list()
        self.freeze()

    def freeze(self) -> ReportModel:
        """Build one model from the current traces and reuse it."""
        model = self.build_model()
        self.model = model
        self.preview.setHtml(render_html(model, include_plots=False))
        self._update_warnings(model)
        plots = report_plots(model)
        if plots:
            self.label_plots.setText(
                f"{len(plots)} plot(s) are included in the exported HTML "
                "file. They are not shown here because the preview cannot "
                "display vector plots."
            )
        else:
            self.label_plots.setText(
                "No plots are available for the selected traces."
            )
        self.label_stale.setVisible(False)
        return model

    def frozen_model(self) -> ReportModel:
        """The model shared by the preview and the exports."""
        if self.model is None:
            return self.freeze()
        return self.model

    def build_model(self) -> ReportModel:
        traces = self.selected_traces()
        analyses = {}
        coverage = {}
        parameters = dict(getattr(self.app, "_preset_parameters", {}))
        if "vswr_limit" in parameters:
            parameters["limit"] = parameters["vswr_limit"]
        for trace in traces:
            analyses[trace.id] = analyse_snapshot(
                trace.snapshot,
                analysis_kinds(trace),
                parameters=parameters,
                trace_id=trace.id,
                trace_name=trace.name,
            )
            coverage[trace.id] = self._coverage(trace)
        return build_report(
            self.input_title.text().strip(),
            traces,
            dut=self.input_dut.text().strip(),
            notes=self.input_notes.toPlainText().strip(),
            analyses=analyses,
            coverage=coverage,
            app_version=self._app_version(),
            limits=self.selected_limit_set(),
        )

    def _app_version(self) -> str:
        from ..NanoVNASaver import NanoVNASaver

        return NanoVNASaver.version

    def _coverage(self, trace: "Trace"):
        """Coverage of the application calibration over this trace's grid."""
        calibration = getattr(self.app, "calibration", None)
        parameters = trace.snapshot.available
        for parameter in parameters or ("S11",):
            frequencies = [dp.freq for dp in trace.channel(parameter)]
            if frequencies:
                return coverage_from_calibration(
                    calibration, frequencies, parameters=tuple(parameters)
                )
        return None

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------
    def _uncorrected(self, trace: ReportTrace) -> bool:
        """True when the reported values are not calibration-corrected.

        Both facts are required: a correction must actually have been
        *applied*, and it must be backed by an identified calibration whose
        coverage says it applies here.  A recorded identity alone is not
        evidence that anything took effect, and an applied correction with no
        calibration identity (an offset-delay-only run) is not calibrated.
        """
        correction = trace.snapshot.correction
        if not correction.applied or not correction.identity:
            return True
        coverage = trace.coverage
        return coverage is None or not coverage.has_calibration

    def _warning_text(self, traces: Sequence[ReportTrace]) -> str:
        """Describe calibration state without calling corrected data raw."""
        uncorrected = [trace for trace in traces if self._uncorrected(trace)]
        if not uncorrected:
            return ""
        raw: list[str] = []
        delay_only: list[str] = []
        unverified: list[str] = []
        for trace in uncorrected:
            kind = correction_kind(trace.snapshot)
            if kind is CorrectionKind.OFFSET_DELAY_ONLY:
                delay_only.append(trace.name)
            elif kind is CorrectionKind.NONE:
                raw.append(trace.name)
            else:
                # A correction ran, but nothing ties it to a calibration that
                # covers this sweep: neither "raw" nor "calibrated" is true.
                unverified.append(trace.name)
        sentences = []
        if raw:
            sentences.append(
                "No calibration applied to: "
                + ", ".join(raw)
                + ". Values are raw measurements."
            )
        if delay_only:
            sentences.append(
                "No SOL calibration applied to: "
                + ", ".join(delay_only)
                + ". Values have an offset delay correction applied, not a"
                " calibration."
            )
        if unverified:
            sentences.append(
                "Correction provenance not verified for: "
                + ", ".join(unverified)
                + ". Values are corrected, but no calibration is verified to"
                " cover them."
            )
        return " ".join(sentences)

    def _update_warnings(self, model: ReportModel) -> None:
        if model.simulated:
            names = [t.name for t in model.traces if t.simulated]
            self.label_simulated.setText(
                "Simulated data: this report contains no measurement from a "
                "device (" + ", ".join(names) + ")."
            )
        else:
            self.label_simulated.setText("")
        self.label_simulated.setVisible(bool(model.simulated))

        text = self._warning_text(model.traces)
        self.label_uncalibrated.setText(text)
        self.label_uncalibrated.setVisible(bool(text))

        self._update_limit_label(model)

    def _update_limit_label(self, model: ReportModel) -> None:
        """Show the frozen verdict, never the live limit selection."""
        verdict = model.verdict
        if model.limit_set is None:
            self.label_limits.setText("No limit checking in this report.")
            self.label_limits.setStyleSheet("color: #555555;")
            return
        if verdict is None:
            self.label_limits.setText(
                f"Limits ({model.limit_set.name}): no traces to check."
            )
            self.label_limits.setStyleSheet(
                "color: #b06a00; font-weight: bold;"
            )
            return
        counts = {value: 0 for value in Verdict}
        for trace in model.traces:
            for result in trace.limits:
                counts[result.verdict] += 1
        self.label_limits.setText(
            f"Limits ({model.limit_set.name}): {verdict.label}"
            f" - {counts[Verdict.PASS]} pass, {counts[Verdict.FAIL]} fail,"
            f" {counts[Verdict.UNAVAILABLE]} unavailable"
        )
        if verdict is Verdict.PASS:
            self.label_limits.setStyleSheet(
                "color: #1b6e2f; font-weight: bold;"
            )
        elif verdict is Verdict.FAIL:
            self.label_limits.setStyleSheet(
                "color: #b00020; font-weight: bold;"
            )
        else:
            self.label_limits.setStyleSheet(
                "color: #b06a00; font-weight: bold;"
            )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _suggested_name(self) -> str:
        title = self.input_title.text().strip() or "report"
        directory = ""
        settings = getattr(self.app, "settings", None)
        if settings is not None:
            directory = settings.value("SaveDirectory", "", str)
        return str(QtCore.QDir(directory).filePath(title + ".html"))

    def _ask_filename(self, extension: str, filter_text: str) -> str:
        suggestion = self._suggested_name()
        if suggestion.endswith(".html"):
            suggestion = suggestion[: -len(".html")] + extension
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save report", suggestion, filter_text
        )
        return filename

    def save_html(self) -> bool:
        """Ask for a path and write the frozen model as HTML."""
        filename = self._ask_filename(".html", "HTML report (*.html)")
        if not filename:
            return False
        return self.save_html_at(filename)

    def save_html_at(self, filename: str) -> bool:
        """Write the frozen model to ``filename``; never rebuilds it."""
        try:
            write_html(self.frozen_model(), filename)
        except OSError as exc:
            logger.warning("Unable to write the report: %s", exc)
            self.label_status.setText(f"Unable to write the report: {exc}")
            return False
        self.label_status.setText(f"Report saved to {filename}")
        return True

    def save_csv(self) -> bool:
        """Ask for a path and write the frozen model's summary as CSV."""
        filename = self._ask_filename(".csv", "CSV summary (*.csv)")
        if not filename:
            return False
        return self.save_csv_at(filename)

    def save_csv_at(self, filename: str) -> bool:
        """Write the frozen model to ``filename``; never rebuilds it."""
        try:
            write_csv(self.frozen_model(), filename)
        except OSError as exc:
            logger.warning("Unable to write the CSV summary: %s", exc)
            self.label_status.setText(f"Unable to write the CSV summary: {exc}")
            return False
        self.label_status.setText(f"CSV summary saved to {filename}")
        return True
