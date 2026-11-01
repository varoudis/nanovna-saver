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
"""REPORT-02.1 / REPORT-02.5: report preview, export and freeze."""

import csv
import io
import os

import pytest

from NanoVNASaver.Hardware.SimulatedScenarios import build_scenario
from NanoVNASaver.Measurement import snapshot_from_datapoints
from NanoVNASaver.RFTools import Datapoint

_QT_APP = None


@pytest.fixture(scope="module")
def qt_app():
    global _QT_APP  # noqa: PLW0603 - owned by the interpreter for the run
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    if _QT_APP is None:
        _QT_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )
    yield _QT_APP


@pytest.fixture(scope="module")
def app(qt_app):
    from NanoVNASaver.NanoVNASaver import NanoVNASaver

    window = NanoVNASaver()
    yield window
    window.close()


def _clear_traces(app):
    collection = app.trace_control.collection
    for trace_id in list(collection.ordered_ids()):
        collection.remove(trace_id)


@pytest.fixture
def report_window(app):
    from NanoVNASaver.Windows.ReportWindow import ReportWindow

    _clear_traces(app)
    window = ReportWindow(app)
    yield window
    window.close()
    _clear_traces(app)


def _scenario_snapshot(
    name, scenario_name, points=41, start=1_000_000, step=500_000
):
    scenario = build_scenario(scenario_name)
    s11 = []
    s21 = []
    for index in range(points):
        freq = int(start + index * step)
        result = scenario(freq)
        if result.s11 is not None:
            s11.append(Datapoint(freq, result.s11.real, result.s11.imag))
        if result.s21 is not None:
            s21.append(Datapoint(freq, result.s21.real, result.s21.imag))
    return snapshot_from_datapoints(
        s11 or None,
        s21 or None,
        name=name,
        source="simulated",
        simulated=True,
        simulator_scenario=scenario_name,
    )


def _add_traces(app, count=2):
    traces = []
    for index in range(count):
        snapshot = _scenario_snapshot(
            f"Sim {index}", "filter" if index % 2 else "antenna"
        )
        traces.append(
            app.trace_control.collection.add(snapshot, name=f"Sim {index}")
        )
    return traces


class TestReportWindow:
    @staticmethod
    def test_preview_lists_the_selected_traces(app, report_window):
        _add_traces(app, 2)
        report_window.refresh()
        text = report_window.preview.toPlainText()
        assert "Sim 0" in text
        assert "Sim 1" in text
        assert len(report_window.frozen_model().traces) == 2

    @staticmethod
    def test_all_traces_are_included_by_default(app, report_window):
        _add_traces(app, 2)
        report_window.refresh()
        boxes = report_window.checkboxes()
        assert len(boxes) == 2
        assert all(box.isChecked() for box in boxes.values())

    @staticmethod
    def test_unchecked_traces_are_excluded_after_a_refresh(app, report_window):
        _add_traces(app, 2)
        report_window.refresh()
        first = next(iter(report_window.checkboxes().values()))
        first.setChecked(False)
        report_window.refresh()
        names = [trace.name for trace in report_window.frozen_model().traces]
        assert names == ["Sim 1"]

    @staticmethod
    def test_saving_html_writes_plots_and_names(app, report_window, tmp_path):
        _add_traces(app, 2)
        report_window.refresh()
        target = tmp_path / "report.html"
        assert report_window.save_html_at(str(target))
        content = target.read_text(encoding="utf-8")
        assert "<svg" in content
        assert "Sim 0" in content
        assert "Sim 1" in content

    @staticmethod
    def test_the_preview_itself_has_no_plots(app, report_window):
        _add_traces(app, 2)
        report_window.refresh()
        # QTextBrowser drops inline SVG, so the preview is built without it
        # and a visible label says where the plots went.
        assert "<svg" not in report_window.preview.toHtml()
        assert "exported HTML" in report_window.label_plots.text()

    @staticmethod
    def test_saving_csv_writes_the_header(app, report_window, tmp_path):
        _add_traces(app, 2)
        report_window.refresh()
        target = tmp_path / "report.csv"
        assert report_window.save_csv_at(str(target))
        rows = list(csv.reader(io.StringIO(target.read_text(encoding="utf-8"))))
        assert rows[0][:4] == ["trace_id", "trace", "channel", "analysis"]
        assert {row[1] for row in rows[1:]} == {"Sim 0", "Sim 1"}

    @staticmethod
    def test_the_model_is_frozen_until_an_explicit_refresh(app, report_window):
        _add_traces(app, 2)
        report_window.refresh()
        frozen = report_window.frozen_model()
        _add_traces(app, 1)
        assert len(frozen.traces) == 2
        assert report_window.frozen_model() is frozen
        report_window.refresh()
        assert len(report_window.frozen_model().traces) == 3

    @staticmethod
    def test_an_export_reuses_the_previewed_model(app, report_window, tmp_path):
        _add_traces(app, 2)
        report_window.refresh()
        frozen = report_window.frozen_model()
        _add_traces(app, 1)
        target = tmp_path / "report.html"
        assert report_window.save_html_at(str(target))
        assert report_window.frozen_model() is frozen
        content = target.read_text(encoding="utf-8")
        assert "Sim 2" not in content

    @staticmethod
    def test_simulated_traces_are_announced(app, report_window):
        _add_traces(app, 2)
        report_window.refresh()
        assert report_window.frozen_model().simulated
        assert "Simulated" in report_window.label_simulated.text()
        assert report_window.label_simulated.isVisibleTo(report_window)

    @staticmethod
    def test_uncalibrated_traces_are_announced(app, report_window):
        _add_traces(app, 2)
        report_window.refresh()
        assert "No calibration applied to" in (
            report_window.label_uncalibrated.text()
        )
        assert report_window.label_uncalibrated.isVisibleTo(report_window)

    @staticmethod
    def test_editing_the_title_marks_the_preview_stale(app, report_window):
        _add_traces(app, 2)
        report_window.refresh()
        assert not report_window.label_stale.isVisibleTo(report_window)
        report_window.input_title.setText("Bench check")
        assert report_window.label_stale.isVisibleTo(report_window)
        report_window.refresh()
        assert not report_window.label_stale.isVisibleTo(report_window)
        assert report_window.frozen_model().title == "Bench check"

    @staticmethod
    def test_the_window_is_registered_with_the_application(app):
        assert "report" in app.windows
