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
"""DSP-02/DSP-03 controls in the sweep settings window.

The window is built against a stub application so these tests exercise the
controls and the configuration they produce, not the whole main window.
"""

import pytest

from NanoVNASaver.Processing import (
    STAGE_AVERAGE,
    STAGE_SMOOTHING,
    ProcessingConfig,
    RunningAverageState,
)
from NanoVNASaver.Settings.Sweep import Sweep


@pytest.fixture
def qt_app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _bands_model():
    """A one-band item model: the window reads its row as start/stop."""
    from PySide6 import QtGui

    model = QtGui.QStandardItemModel()
    model.setColumnCount(3)
    model.appendRow(
        [
            QtGui.QStandardItem("Test band"),
            QtGui.QStandardItem(str(10_000_000)),
            QtGui.QStandardItem(str(20_000_000)),
        ]
    )
    return model


class _VNA:
    name = "stub"
    txPowerRanges = ()  # noqa: N815 - mirrors the interface the window reads

    def setTXPower(self, *_args):
        pass


class _Inputs(dict):
    """The sweep control's frequency inputs, keyed by label."""

    def __init__(self):
        from PySide6 import QtWidgets

        super().__init__()
        for key in ("Start", "Stop"):
            self[key] = QtWidgets.QLineEdit("")


class StubApp:
    """The minimum a sweep-settings window touches."""

    def __init__(self, points=101):
        self.sweep = Sweep(
            start=10_000_000, end=20_000_000, points=points, segments=1
        )
        self.processing_config = ProcessingConfig()
        self.running_average = RunningAverageState()
        self.changes = []
        self.s21att = 0
        self.bands = _bands_model()
        self.vna = _VNA()
        self.sweep_control = type("_SweepControl", (), {"inputs": _Inputs()})()

    def mark_session_changed(self, change):
        self.changes.append(change)

    def update_sweep_title(self):
        pass

    def update_band(self):
        pass

    def update_display_bands(self):
        pass


def _window(app):
    from NanoVNASaver.Windows.SweepSettings import SweepSettingsWindow

    return SweepSettingsWindow(app)


class TestProcessingDefaults:
    @staticmethod
    def test_processing_starts_off(qt_app):
        app = StubApp()
        window = _window(app)
        assert app.processing_config.is_identity
        assert window.check_smoothing.isChecked() is False
        assert window.check_average.isChecked() is False
        window.deleteLater()

    @staticmethod
    def test_the_active_pipeline_is_described(qt_app):
        app = StubApp()
        window = _window(app)
        assert "no processing" in window.label_processing.text().lower()
        window.deleteLater()


class TestSmoothingControls:
    @staticmethod
    def test_enabling_smoothing_configures_a_stage(qt_app):
        app = StubApp()
        window = _window(app)
        window.spin_aperture_percent.setValue(10)
        window.check_smoothing.setChecked(True)
        stage = app.processing_config.find(STAGE_SMOOTHING)
        assert stage is not None
        aperture = stage.as_dict["aperture_points"]
        assert aperture % 2 == 1
        assert aperture > 1
        window.deleteLater()

    @staticmethod
    def test_the_effective_point_count_is_shown(qt_app):
        app = StubApp(points=101)
        window = _window(app)
        window.spin_aperture_percent.setValue(10)
        window.check_smoothing.setChecked(True)
        text = window.label_aperture.text()
        assert "point" in text
        assert "101" in text
        window.deleteLater()

    @staticmethod
    def test_a_zero_aperture_removes_the_stage(qt_app):
        app = StubApp()
        window = _window(app)
        window.spin_aperture_percent.setValue(10)
        window.check_smoothing.setChecked(True)
        assert app.processing_config.find(STAGE_SMOOTHING) is not None
        window.spin_aperture_percent.setValue(0)
        assert app.processing_config.find(STAGE_SMOOTHING) is None
        window.deleteLater()

    @staticmethod
    def test_ticking_the_box_with_no_width_is_reported_not_guessed(qt_app):
        """A ticked box must never mean "nothing happened" and nothing said."""
        app = StubApp()
        window = _window(app)
        window.spin_aperture_percent.setValue(0)
        window.check_smoothing.setChecked(True)
        assert app.processing_config.find(STAGE_SMOOTHING) is None
        assert window.check_smoothing.isChecked() is True
        assert "applies nothing" in window.label_processing_message.text()
        # Widening the aperture without touching the box applies it.
        window.spin_aperture_percent.setValue(10)
        assert app.processing_config.find(STAGE_SMOOTHING) is not None
        assert window.label_processing_message.text() == ""
        window.deleteLater()

    @staticmethod
    def test_disabling_smoothing_removes_it(qt_app):
        app = StubApp()
        window = _window(app)
        window.spin_aperture_percent.setValue(10)
        window.check_smoothing.setChecked(True)
        window.check_smoothing.setChecked(False)
        assert app.processing_config.find(STAGE_SMOOTHING) is None
        assert app.processing_config.is_identity
        window.deleteLater()

    @staticmethod
    def test_a_tiny_sweep_still_yields_a_usable_aperture(qt_app):
        app = StubApp(points=3)
        window = _window(app)
        window.spin_aperture_percent.setValue(40)
        window.check_smoothing.setChecked(True)
        stage = app.processing_config.find(STAGE_SMOOTHING)
        assert stage is not None
        assert 1 <= stage.as_dict["aperture_points"] <= 3
        window.deleteLater()


class TestAverageControls:
    @staticmethod
    def test_enabling_the_average_defaults_to_cumulative(qt_app):
        app = StubApp()
        window = _window(app)
        window.check_average.setChecked(True)
        stage = app.processing_config.find(STAGE_AVERAGE)
        assert stage is not None
        assert stage.as_dict["mode"] == "cumulative"
        assert "alpha" not in stage.as_dict
        window.deleteLater()

    @staticmethod
    def test_exponential_mode_carries_its_alpha(qt_app):
        app = StubApp()
        window = _window(app)
        window.check_average.setChecked(True)
        index = window.combo_average_mode.findData("exponential")
        window.combo_average_mode.setCurrentIndex(index)
        window.spin_average_alpha.setValue(0.4)
        stage = app.processing_config.find(STAGE_AVERAGE)
        assert stage.as_dict["mode"] == "exponential"
        assert stage.as_dict["alpha"] == pytest.approx(0.4)
        window.deleteLater()

    @staticmethod
    def test_disabling_the_average_removes_it(qt_app):
        app = StubApp()
        window = _window(app)
        window.check_average.setChecked(True)
        window.check_average.setChecked(False)
        assert app.processing_config.find(STAGE_AVERAGE) is None
        window.deleteLater()


class TestIntegrationWithTheApp:
    @staticmethod
    def test_each_change_marks_the_session_dirty(qt_app):
        app = StubApp()
        window = _window(app)
        window.check_smoothing.setChecked(True)
        assert "sweep" in app.changes
        window.deleteLater()

    @staticmethod
    def test_a_new_pipeline_resets_the_running_average(qt_app):
        """An old average belongs to the old pipeline and must not blend."""
        app = StubApp()
        app.running_average = RunningAverageState(
            context="old", count=5, values=((1, 0.5 + 0j),)
        )
        window = _window(app)
        window.check_smoothing.setChecked(True)
        assert app.running_average.count == 0
        window.deleteLater()

    @staticmethod
    def test_the_stage_order_is_the_engine_order(qt_app):
        app = StubApp()
        window = _window(app)
        window.check_average.setChecked(True)
        window.spin_aperture_percent.setValue(10)
        window.check_smoothing.setChecked(True)
        kinds = [stage.kind for stage in app.processing_config.stages]
        assert kinds == [STAGE_AVERAGE, STAGE_SMOOTHING]
        window.deleteLater()
