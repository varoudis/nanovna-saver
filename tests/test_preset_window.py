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
"""PRESET-02.5 / PRESET-03.1: applying and managing presets from the UI."""

import os

import pytest

from NanoVNASaver.Hardware.Simulated import make_simulated_vna
from NanoVNASaver.Presets import (
    BUILTIN_NAMES,
    Preset,
    PresetStore,
    PresetValidationError,
    SweepPreset,
    apply_preset,
)
from NanoVNASaver.Windows.PresetWindow import BUILTIN_SUFFIX, PresetWindow

_QT_APP = None
_APP = None


@pytest.fixture(scope="session")
def qt_app():
    global _QT_APP  # noqa: PLW0603 - owned by the interpreter for the run
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    if _QT_APP is None:
        _QT_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )
    yield _QT_APP


@pytest.fixture(scope="session")
def vna_app(qt_app):
    """One real main window; building it is the expensive part."""
    global _APP  # noqa: PLW0603 - reused for the whole run
    if _APP is None:
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        _APP = NanoVNASaver()
    # A simulated device that accepts the built-in point counts and stays
    # disconnected unless a test connects it.
    _APP.vna = make_simulated_vna("antenna")
    return _APP


@pytest.fixture
def store(tmp_path):
    return PresetStore(filename=str(tmp_path / "presets.nvpreset"))


@pytest.fixture
def preset_window(vna_app, store):
    window = PresetWindow(vna_app, store=store, filename=store.filename)
    yield window
    window.deleteLater()


class TestPresetListing:
    @staticmethod
    def test_built_in_presets_are_listed(preset_window):
        assert preset_window.preset_names()[: len(BUILTIN_NAMES)] == list(
            BUILTIN_NAMES
        )

    @staticmethod
    def test_built_ins_are_marked_in_the_list(preset_window):
        labels = [
            preset_window.list_presets.item(row).text()
            for row in range(preset_window.list_presets.count())
        ]
        assert labels[: len(BUILTIN_NAMES)] == [
            name + BUILTIN_SUFFIX for name in BUILTIN_NAMES
        ]

    @staticmethod
    def test_selection_describes_the_preset(preset_window):
        assert preset_window.select("Cable testing")
        text = preset_window.detail.toPlainText()
        assert "Cable testing" in text
        assert "velocity_factor=0.66" in text
        assert "Analyses: cable" in text

    @staticmethod
    def test_an_unreadable_store_falls_back_to_built_ins(vna_app, tmp_path):
        broken = tmp_path / "broken.nvpreset"
        broken.write_text("{not json", encoding="utf-8")
        window = PresetWindow(vna_app, filename=str(broken))
        try:
            assert window.preset_names() == list(BUILTIN_NAMES)
            assert "could not read" in window.label_message.text()
        finally:
            window.deleteLater()


class TestApplying:
    @staticmethod
    def test_applying_the_antenna_preset_updates_the_sweep(
        vna_app, preset_window
    ):
        assert preset_window.select("Antenna tuning")
        application = preset_window.apply_selected()
        assert application is not None
        assert vna_app.sweep_control.get_start() == 7_000_000
        assert vna_app.sweep_control.get_end() == 30_000_000
        assert vna_app.sweep_control.get_segments() == 1

    @staticmethod
    def test_applying_never_starts_a_sweep(vna_app, preset_window, monkeypatch):
        started = []
        monkeypatch.setattr(vna_app, "sweep_start", lambda: started.append(1))
        assert preset_window.select("Filter testing")
        assert preset_window.apply_selected() is not None
        assert started == []

    @staticmethod
    def test_applying_leaves_the_device_untouched(
        vna_app, preset_window, monkeypatch
    ):
        device = vna_app.vna
        device.disconnect()
        before = device.connected()
        assert before is False
        calls = []
        for name in ("connect", "disconnect", "setSweep", "readValues"):
            monkeypatch.setattr(
                device, name, lambda *a, _n=name, **kw: calls.append(_n)
            )
        assert preset_window.select("Antenna tuning")
        assert preset_window.apply_selected() is not None
        assert calls == []
        assert vna_app.vna is device
        assert device.connected() is before

    @staticmethod
    def test_a_rejected_preset_does_not_touch_the_sweep(
        vna_app, preset_window, store
    ):
        store.add(
            Preset(
                name="Too wide",
                sweep=SweepPreset(
                    start=1_000_000, end=9_000_000_000, points=101
                ),
            )
        )
        preset_window.refresh()
        assert preset_window.select("Too wide")
        before = (
            vna_app.sweep_control.get_start(),
            vna_app.sweep_control.get_end(),
        )
        assert preset_window.apply_selected() is None
        assert "cannot be applied" in preset_window.label_message.text()
        assert (
            vna_app.sweep_control.get_start(),
            vna_app.sweep_control.get_end(),
        ) == before

    @staticmethod
    def test_unsupported_point_count_is_reported(vna_app, preset_window):
        device = vna_app.vna
        device.valid_datapoints = (101, 51, 11)
        try:
            assert preset_window.select("Antenna tuning")
            with pytest.raises(PresetValidationError):
                apply_preset(
                    preset_window.selected_preset(),
                    valid_points=device.valid_datapoints,
                )
            assert preset_window.apply_selected() is None
            assert "not supported" in preset_window.label_message.text()
        finally:
            device.valid_datapoints = (101, 51, 11, 201, 401)

    @staticmethod
    def test_analysis_kinds_are_recorded_on_the_main_window(
        vna_app, preset_window
    ):
        assert preset_window.select("Antenna tuning")
        application = preset_window.apply_selected()
        assert application is not None
        assert vna_app._preset_analyses == application.analysis_kinds

    @staticmethod
    def test_applying_marks_the_session_changed(vna_app, preset_window):
        vna_app._restoring = False
        vna_app.session.dirty = False
        vna_app.session._dirty_reasons.clear()
        assert preset_window.select("Cable testing")
        assert preset_window.apply_selected() is not None
        assert vna_app.session.dirty is True
        assert "sweep" in vna_app.session.dirty_reasons


class TestUserPresets:
    @staticmethod
    def test_a_built_in_cannot_be_deleted(preset_window):
        names_before = preset_window.preset_names()
        assert preset_window.delete_preset("Antenna tuning") is False
        assert "built-in" in preset_window.label_message.text()
        assert preset_window.preset_names() == names_before

    @staticmethod
    def test_a_built_in_cannot_be_renamed(preset_window):
        assert preset_window.rename_selected("My antenna") is None
        assert "built-in" in preset_window.label_message.text()
        assert "Antenna tuning" in preset_window.preset_names()

    @staticmethod
    def test_saved_preset_survives_a_reload(vna_app, preset_window, store):
        saved = preset_window.save_current_preset("Custom span")
        assert saved is not None
        assert "Custom span" in preset_window.preset_names()

        reloaded = PresetStore.load(store.filename)
        assert [preset.name for preset in reloaded.presets] == ["Custom span"]
        window = PresetWindow(vna_app, store=reloaded, filename=store.filename)
        try:
            assert "Custom span" in window.preset_names()
            assert window.select("Custom span")
            assert "Analyses: none" in window.detail.toPlainText()
        finally:
            window.deleteLater()

    @staticmethod
    def test_duplicate_and_delete_work_on_user_presets(preset_window):
        preset_window.save_current_preset("Base preset")
        assert preset_window.select("Base preset")
        copy = preset_window.duplicate_selected("Base preset copy")
        assert copy is not None
        assert "Base preset copy" in preset_window.preset_names()
        assert preset_window.delete_preset("Base preset copy") is True
        assert "Base preset copy" not in preset_window.preset_names()

    @staticmethod
    def test_export_and_import_round_trip(vna_app, preset_window, tmp_path):
        preset_window.save_current_preset("Transferable")
        target = tmp_path / "transferable.nvpreset"
        assert preset_window.export_preset(
            filename=str(target), name="Transferable"
        )
        assert target.exists()

        empty = PresetStore(filename=str(tmp_path / "other.nvpreset"))
        window = PresetWindow(vna_app, store=empty, filename=empty.filename)
        try:
            assert "Transferable" not in window.preset_names()
            imported = window.import_file(str(target))
            assert imported is not None
            assert "Transferable" in window.preset_names()
            assert PresetStore.load(empty.filename).presets[0].name == (
                "Transferable"
            )
        finally:
            window.deleteLater()

    @staticmethod
    def test_importing_a_broken_file_reports_the_error(
        vna_app, preset_window, tmp_path
    ):
        broken = tmp_path / "broken.nvpreset"
        broken.write_text("{not json", encoding="utf-8")
        assert preset_window.import_file(str(broken)) is None
        assert "could not read" in preset_window.label_message.text()


class TestPresetApplicationIsComplete:
    """PRESET-02.1: a preset applies its whole sweep setup.

    Written after a real defect: only start/stop/segments were pushed into the
    UI, so a preset asking for 401 points and 3 averages left the window at
    the device default of 101 points and 3 averages - and because
    ``SweepControl.update_sweep`` derives the point count from
    ``vna.datapoints``, the point count was reverted even when set.
    """

    @staticmethod
    def test_every_builtin_preset_sets_the_point_count(vna_app):
        from NanoVNASaver import Presets

        for name in Presets.BUILTIN_NAMES:
            preset = Presets.find_builtin(name)
            application = apply_preset(
                preset,
                max_frequency=6_000_000_000,
                valid_points=vna_app.vna.valid_datapoints,
            )
            before = vna_app.sweep.points
            vna_app.apply_preset_application(application)
            assert vna_app.sweep.points == preset.sweep.points, name
            assert vna_app.vna.datapoints == preset.sweep.points, name
            assert vna_app.sweep.start == preset.sweep.start, name
            assert vna_app.sweep.end == preset.sweep.end, name
            assert vna_app.sweep.segments == preset.sweep.segments, name
            assert vna_app.sweep.properties.averages[0] == (
                preset.sweep.averages
            ), name
            assert vna_app.sweep.points != before or before == (
                preset.sweep.points
            )

    @staticmethod
    def test_an_unsupported_point_count_is_refused_not_silently_sent(
        vna_app,
    ):
        preset = Preset(
            name="odd points",
            sweep=SweepPreset(
                start=1_000_000,
                end=2_000_000,
                points=12345,
                segments=1,
                averages=1,
            ),
        )
        with pytest.raises(PresetValidationError):
            apply_preset(
                preset,
                max_frequency=6_000_000_000,
                valid_points=vna_app.vna.valid_datapoints,
            )

    @staticmethod
    def test_the_point_count_applier_refuses_unsupported_counts(vna_app):
        """Defence in depth: the applier is reachable without validation."""
        vna_app.vna.datapoints = 101
        vna_app.sweep.set_points(101)
        assert vna_app._apply_point_count(12345) is False
        assert vna_app.vna.datapoints == 101
        assert vna_app.sweep.points == 101

    @staticmethod
    def test_applying_a_preset_never_connects(vna_app):
        from NanoVNASaver import Presets

        preset = Presets.find_builtin("Antenna tuning")
        application = apply_preset(
            preset,
            max_frequency=6_000_000_000,
            valid_points=vna_app.vna.valid_datapoints,
        )
        connected_before = vna_app.vna.connected()
        vna_app.apply_preset_application(application)
        assert vna_app.vna.connected() == connected_before


class TestPresetAppliesChartsAndParameters:
    """PRESET-02.1: a preset is a workflow, not just a frequency range.

    Written after a real defect: applying a preset changed the sweep but left
    the chart layout and the workflow parameters untouched, so applying the
    cable preset produced a screen indistinguishable from before.
    """

    @staticmethod
    def _layout(app):
        cells = []
        for index in range(app.charts_layout.count()):
            item = app.charts_layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is None:
                continue
            row, column, _rs, _cs = app.charts_layout.getItemPosition(index)
            cells.append((row, column, widget.name))
        return sorted(cells)

    @staticmethod
    def test_the_preset_charts_become_the_main_layout(vna_app):
        from NanoVNASaver import Presets

        preset = Presets.find_builtin("Cable testing")
        application = apply_preset(
            preset,
            max_frequency=6_000_000_000,
            valid_points=vna_app.vna.valid_datapoints,
        )
        vna_app.apply_preset_application(application)
        assert [
            cell[2]
            for cell in TestPresetAppliesChartsAndParameters._layout(vna_app)
        ] == list(preset.charts)

    @staticmethod
    def test_a_different_preset_replaces_the_layout(vna_app):
        from NanoVNASaver import Presets

        antenna = Presets.find_builtin("Antenna tuning")
        cable = Presets.find_builtin("Cable testing")
        for preset in (antenna, cable, antenna):
            vna_app.apply_preset_application(
                apply_preset(
                    preset,
                    max_frequency=6_000_000_000,
                    valid_points=vna_app.vna.valid_datapoints,
                )
            )
            names = [
                cell[2]
                for cell in TestPresetAppliesChartsAndParameters._layout(
                    vna_app
                )
            ]
            assert names == list(preset.charts), preset.name
            assert len(names) == len(set(names)), "a chart was duplicated"

    @staticmethod
    def test_workflow_parameters_are_recorded(vna_app):
        from NanoVNASaver import Presets

        preset = Presets.find_builtin("Cable testing")
        vna_app.apply_preset_application(
            apply_preset(
                preset,
                max_frequency=6_000_000_000,
                valid_points=vna_app.vna.valid_datapoints,
            )
        )
        assert vna_app._preset_parameters["velocity_factor"] == 0.66
        assert vna_app._preset_parameters["cable_length"] == 10.0
        assert vna_app._preset_parameters["marker_count"] == 2

    @staticmethod
    def test_the_vswr_limit_reaches_the_analysis(vna_app):
        from NanoVNASaver import Presets
        from NanoVNASaver.Analysis.VSWRAnalysis import VSWRAnalysis

        preset = Presets.find_builtin("Antenna tuning")
        vna_app.apply_preset_application(
            apply_preset(
                preset,
                max_frequency=6_000_000_000,
                valid_points=vna_app.vna.valid_datapoints,
            )
        )
        assert VSWRAnalysis.vswr_limit_value == pytest.approx(2.0)

    @staticmethod
    def test_the_analysis_kinds_are_recorded_as_enums(vna_app):
        from NanoVNASaver import Presets
        from NanoVNASaver.Analysis.Results import AnalysisKind

        preset = Presets.find_builtin("Filter testing")
        vna_app.apply_preset_application(
            apply_preset(
                preset,
                max_frequency=6_000_000_000,
                valid_points=vna_app.vna.valid_datapoints,
            )
        )
        assert all(
            isinstance(kind, AnalysisKind) for kind in vna_app._preset_analyses
        )
