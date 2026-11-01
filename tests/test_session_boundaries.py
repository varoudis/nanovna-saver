"""Regression checks for experiment state across save, load and presets."""

from dataclasses import replace

import pytest

from NanoVNASaver.Analysis.Results import AnalysisKind
from NanoVNASaver.Calibration import Calibration
from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Measurement import CompletionState
from NanoVNASaver.NanoVNASaver import NanoVNASaver
from NanoVNASaver.Presets import CABLE_PRESET, apply_preset
from NanoVNASaver.RFTools import Datapoint


@pytest.fixture
def windows(qt_app):
    created = []

    def make():
        window = NanoVNASaver()
        created.append(window)
        return window

    yield make
    for window in created:
        window.worker.quit()
        window.deleteLater()


def fill(window, start):
    window.saveData(
        [Datapoint(start, 0.2, 0.1), Datapoint(start + 100, 0.3, 0.1)],
        [],
        str(start),
    )


@pytest.mark.parametrize("completion", list(CompletionState))
def test_reopened_capture_keeps_acquisition_metadata(
    windows, tmp_path, completion
):
    window = windows()
    window.vna = make_simulated_vna("antenna")
    window.sweep.update(27_000_000, 30_000_000, 1, 101)
    window.calibration = Calibration()
    window.calibration.load("tests/data/sol_27_30.cal")
    window.calibration.calc_corrections()
    window.worker.run_once()
    acquired = window.current_snapshot()
    assert acquired.correction.applied
    assert acquired.raw != acquired.corrected
    acquired = replace(
        acquired, provenance=replace(acquired.provenance, completion=completion)
    )
    window._live_snapshot = acquired
    filename = str(tmp_path / "corrected.nvsession")
    assert window._write_session(filename)

    reopened = windows()
    assert reopened.open_session_file(filename)
    reopened.trace_control.capture_current()
    captured = list(reopened.trace_control.collection)[-1].snapshot
    assert captured.to_dict() == acquired.to_dict()
    assert reopened._displayed_snapshot().to_dict() == acquired.to_dict()


def test_save_reopen_keeps_an_earlier_captured_trace_displayed(
    windows, tmp_path
):
    window = windows()
    fill(window, 100)
    window.trace_control.capture_current()
    first_id = window.trace_control.collection.ordered_ids()[0]
    fill(window, 500)
    window.trace_control.capture_current()
    fill(window, 100)
    filename = str(tmp_path / "earlier.nvsession")
    assert window.current_session_document().ui["displayed_trace"] == first_id
    assert window._write_session(filename)

    reopened = windows()
    assert reopened.open_session_file(filename)
    assert reopened.data.s11[0].freq == 100


def test_restored_layout_has_only_the_saved_chart(windows, tmp_path):
    window = windows()
    window.apply_preset_charts(["S21 Gain"])
    expected = window.chart_state_dict()
    filename = str(tmp_path / "one-chart.nvsession")
    assert window._write_session(filename)

    reopened = windows()
    assert reopened.open_session_file(filename)
    assert reopened.chart_state_dict() == expected
    assert reopened.charts_layout.count() == 1


def test_repeated_chart_names_keep_separate_cells(windows):
    window = windows()
    state = {
        "cells": [
            {"row": 0, "column": 0, "name": "S21 Gain"},
            {"row": 0, "column": 1, "name": "S21 Gain"},
        ]
    }
    assert window.apply_chart_state(state)
    assert window.chart_state_dict() == state
    assert window.charts_layout.itemAt(0).widget() is not (
        window.charts_layout.itemAt(1).widget()
    )


def test_explicitly_empty_layout_is_restored(windows):
    window = windows()
    assert window.charts_layout.count()
    assert window.apply_chart_state({"cells": []})
    assert window.charts_layout.count() == 0
    assert window.apply_chart_state({"s11": ["log_mag"]}) is False


def test_cable_preset_updates_the_actual_tdr_velocity(windows, tmp_path):
    window = windows()
    window.vna = make_simulated_vna("cable")
    tdr = window.windows["tdr"]
    tdr.tdr_velocity_dropdown.setCurrentIndex(
        tdr.tdr_velocity_dropdown.findData(-1)
    )
    tdr.tdr_velocity_input.setText("0.8")
    window.apply_preset_application(apply_preset(CABLE_PRESET))
    assert float(tdr.tdr_velocity_input.text()) == 0.66
    tdr.updateTDR()
    assert float(tdr.tdr_velocity_input.text()) == 0.66

    # Restoring parameters must apply them to controls as well as metadata.
    custom = replace(CABLE_PRESET, parameters={"velocity_factor": 0.72})
    window.apply_preset_application(apply_preset(custom))
    filename = str(tmp_path / "velocity.nvsession")
    assert window._write_session(filename)
    reopened = windows()
    assert reopened.open_session_file(filename)
    assert float(reopened.windows["tdr"].tdr_velocity_input.text()) == 0.72
    assert reopened._preset_parameters["velocity_factor"] == 0.72


def test_report_analysis_uses_the_applied_workflow_parameters(windows):
    window = windows()
    window.vna = make_simulated_vna("cable")
    preset = replace(
        CABLE_PRESET, parameters={"velocity_factor": 0.72, "vswr_limit": 2.0}
    )
    window.apply_preset_application(apply_preset(preset))
    window.worker.run_once()
    window.trace_control.capture_current()
    report = window.windows["report"]
    report.refresh()
    results = {
        result.kind: result for result in report.model.traces[0].analyses
    }
    assert results[AnalysisKind.CABLE].parameters["velocity_factor"] == 0.72
    assert results[AnalysisKind.VSWR].parameters["limit"] == 2.0
