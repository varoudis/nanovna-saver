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
"""RF-03 in the user interface: choosing, previewing and installing a fixture.

The fixture files used here are written by the test from textbook formulas
for a matched attenuator of ``L`` dB, whose two-port is perfectly known::

    S11 = S22 = 0,  S21 = S12 = 10**(-L/20)

Nothing is compared against the application's own de-embedding: the expected
numbers come from the closed-form cascade of two ideal attenuators, so a
mistake in the module cannot hide behind a matching mistake in the test.
"""

import json
import math

import pytest

from NanoVNASaver.Fixture import (
    Fixture,
    FixturePoint,
    fixture_from_stage,
    fixture_stage,
)
from NanoVNASaver.Measurement import snapshot_from_datapoints
from NanoVNASaver.Processing import (
    STAGE_FIXTURE,
    ProcessingConfig,
    ProcessingStage,
    apply_processing,
)
from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Session import (
    SESSION_SUFFIX,
    build_archive,
    document_snapshots,
    make_document,
    parse_session,
    save_session,
)
from NanoVNASaver.Settings.Sweep import Sweep

#: The sweep every GUI test is built on: 10 points from 10 to 19 MHz, which
#: is exactly the fixture grid below.
START = 10_000_000
STOP = 19_000_000
GRID = tuple(range(START, STOP + 1, 1_000_000))
ATTENUATION_DB = 6.0


def _through_attenuator(freqs=GRID, db=ATTENUATION_DB):
    """A matched attenuator's S-parameters, from the definition of dB."""
    gain = 10 ** (-db / 20)
    return {
        "S11": [Datapoint(freq, 0.0, 0.0) for freq in freqs],
        "S21": [Datapoint(freq, gain, 0.0) for freq in freqs],
        "S12": [Datapoint(freq, gain, 0.0) for freq in freqs],
        "S22": [Datapoint(freq, 0.0, 0.0) for freq in freqs],
    }


def _write_s2p(path, freqs=GRID, db=ATTENUATION_DB):
    """Write a two-port touchstone file: ``# HZ S RI R 50``, 9 columns."""
    gain = 10 ** (-db / 20)
    lines = ["! synthetic matched attenuator", "# HZ S RI R 50"]
    for freq in freqs:
        lines.append(
            f"{freq} {0.0!r} {0.0!r} {gain!r} {0.0!r}"
            f" {gain!r} {0.0!r} {0.0!r} {0.0!r}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _one_port_s1p(path, freqs=GRID):
    lines = ["# HZ S RI R 50"]
    lines.extend(f"{freq} 0.5 0.0" for freq in freqs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def qt_app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Bands:
    """The one-band item model the sweep-settings window reads."""

    @staticmethod
    def build():
        from PySide6 import QtGui

        model = QtGui.QStandardItemModel()
        model.setColumnCount(3)
        model.appendRow(
            [
                QtGui.QStandardItem("Test band"),
                QtGui.QStandardItem(str(START)),
                QtGui.QStandardItem(str(STOP)),
            ]
        )
        return model


class _Data:
    """The displayed measurement: only what the window may look at."""

    def __init__(self, channels=None):
        channels = channels if channels is not None else _through_attenuator()
        self.s11 = channels.get("S11", [])
        self.s21 = channels.get("S21", [])
        self.s12 = channels.get("S12", [])
        self.s22 = channels.get("S22", [])


class _VNA:
    name = "stub"
    txPowerRanges = ()  # noqa: N815 - mirrors the interface the window reads

    def setTXPower(self, *_args):
        pass


class _Inputs(dict):
    def __init__(self):
        from PySide6 import QtWidgets

        super().__init__()
        for key in ("Start", "Stop"):
            self[key] = QtWidgets.QLineEdit("")


class StubApp:
    """The minimum a sweep-settings window touches."""

    def __init__(self, channels=None, points=None):
        points = len(GRID) if points is None else points
        from NanoVNASaver.Processing import RunningAverageState

        self.sweep = Sweep(start=START, end=STOP, points=points, segments=1)
        self.processing_config = ProcessingConfig()
        self.running_average = RunningAverageState()
        self.changes = []
        self.s21att = 0
        self.data = _Data(channels)
        self.bands = _Bands.build()
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


@pytest.fixture
def fixture_file(tmp_path):
    return _write_s2p(tmp_path / "attenuator.s2p")


class TestFixtureSummary:
    @staticmethod
    def test_the_summary_reports_the_span_point_count_and_identity(
        qt_app, fixture_file
    ):
        app = StubApp()
        window = _window(app)
        try:
            assert window.load_fixture_file(str(fixture_file)) is True
            summary = window.fixture_summary()
            assert "10.00MHz" in summary
            assert "19.00MHz" in summary
            assert f"{len(GRID)} points" in summary
            assert "Z0 50" in summary

            fixture = Fixture(
                name="attenuator.s2p",
                reference_impedance=50.0,
                points=tuple(
                    FixturePoint(
                        freq=freq,
                        s21=10 ** (-ATTENUATION_DB / 20),
                        s12=10 ** (-ATTENUATION_DB / 20),
                    )
                    for freq in GRID
                ),
            )
            # The digest is computed here from the analytic network, so the
            # summary cannot be quoting a digest of something else.
            assert fixture.identity()[:8] in summary
        finally:
            window.deleteLater()

    @staticmethod
    def test_without_a_file_the_summary_says_so(qt_app):
        app = StubApp()
        window = _window(app)
        try:
            assert "no fixture loaded" in window.fixture_summary()
        finally:
            window.deleteLater()


class TestInstall:
    @staticmethod
    def test_installing_creates_a_stage_carrying_the_identity(
        qt_app, fixture_file
    ):
        app = StubApp()
        window = _window(app)
        try:
            assert window.load_fixture_file(str(fixture_file)) is True
            assert window.install_fixture() is True

            stage = app.processing_config.find(STAGE_FIXTURE)
            assert stage is not None
            values = stage.as_dict
            expected = Fixture(
                name="attenuator.s2p",
                reference_impedance=50.0,
                points=tuple(
                    FixturePoint(
                        freq=freq,
                        s21=10 ** (-ATTENUATION_DB / 20),
                        s12=10 ** (-ATTENUATION_DB / 20),
                    )
                    for freq in GRID
                ),
            )
            assert values["identity"] == expected.identity()
            assert values["name"] == "attenuator.s2p"
            # The stage is self-contained: the samples travel with it.
            assert len(values["points"]) == len(GRID)
            assert "fixture" in app.processing_config.describe().lower()
        finally:
            window.deleteLater()

    @staticmethod
    def test_clearing_removes_the_stage_through_the_pipeline_path(
        qt_app, fixture_file
    ):
        app = StubApp()
        window = _window(app)
        try:
            window.load_fixture_file(str(fixture_file))
            window.install_fixture()
            assert app.processing_config.find(STAGE_FIXTURE) is not None
            window.clear_fixture()
            assert app.processing_config.find(STAGE_FIXTURE) is None
            assert app.processing_config.is_identity
            assert "cleared" in window.label_processing_message.text().lower()
        finally:
            window.deleteLater()

    @staticmethod
    def test_an_installed_fixture_is_adopted_back_into_the_panel(fixture_file):
        """Reopening the window must still show which fixture is in force."""
        fixture = fixture_from_touchstone_named(fixture_file)
        app = StubApp()
        app.processing_config = ProcessingConfig.build(fixture_stage(fixture))
        window = _window(app)
        try:
            assert fixture.identity()[:8] in window.fixture_summary()
            assert "installed as a stage" in (
                window.label_fixture_preview.text()
            )
            # Installing the same file again is a no-op the button refuses.
            assert window.btn_install_fixture.isEnabled() is False
        finally:
            window.deleteLater()


class TestPreviews:
    @staticmethod
    def test_a_one_port_measurement_is_refused_with_the_four_channels(
        qt_app, fixture_file
    ):
        """A two-port cannot be removed from S11 alone; say why, install no."""
        app = StubApp(
            channels={
                "S11": [Datapoint(freq, 0.5, 0.0) for freq in GRID],
                "S21": [Datapoint(freq, 0.5, 0.0) for freq in GRID],
            }
        )
        window = _window(app)
        try:
            window.load_fixture_file(str(fixture_file))
            assert window.install_fixture() is False
            reason = window.label_processing_message.text()
            assert "all four S-parameters" in reason
            assert "S12" in reason
            assert "S22" in reason
            assert app.processing_config.find(STAGE_FIXTURE) is None
        finally:
            window.deleteLater()

    @staticmethod
    def test_a_stage_installed_elsewhere_is_adopted_by_the_panel(
        qt_app, fixture_file
    ):
        """A restored session may install a fixture the panel has not seen."""
        fixture = fixture_from_touchstone_named(fixture_file)
        app = StubApp()
        window = _window(app)
        try:
            assert window.fixture_summary() == "no fixture loaded"
            app.processing_config = ProcessingConfig.build(
                fixture_stage(fixture)
            )
            window._refresh_processing_labels()
            assert fixture.identity()[:8] in window.fixture_summary()
            assert "installed as a stage" in window.fixture_status()
        finally:
            window.deleteLater()

    @staticmethod
    def test_a_corrupt_installed_stage_is_reported_not_hidden(qt_app):
        """A stage that cannot be rebuilt must be named, never dropped."""
        stage = ProcessingStage.build(
            STAGE_FIXTURE, {"name": "broken", "reference_impedance": 50.0}
        )
        app = StubApp()
        app.processing_config = ProcessingConfig.build(stage)
        window = _window(app)
        try:
            assert "cannot be read" in window.fixture_summary()
            assert app.processing_config.find(STAGE_FIXTURE) is not None
            # Nothing is loaded, so nothing can be installed over it.
            assert window.btn_install_fixture.isEnabled() is False
        finally:
            window.deleteLater()

    @staticmethod
    def test_clearing_leaves_the_loaded_file_available(qt_app, fixture_file):
        app = StubApp()
        window = _window(app)
        try:
            window.load_fixture_file(str(fixture_file))
            window.install_fixture()
            window.clear_fixture()
            assert app.processing_config.find(STAGE_FIXTURE) is None
            # The file is still described, and can be reinstalled.
            assert "attenuator.s2p" in window.fixture_summary()
            assert window.btn_install_fixture.isEnabled() is True
            assert window.install_fixture() is True
        finally:
            window.deleteLater()

    @staticmethod
    def test_a_fixture_that_misses_the_sweep_is_refused_naming_the_span(
        qt_app, tmp_path
    ):
        short = _write_s2p(tmp_path / "short.s2p", freqs=GRID[:4])
        app = StubApp()
        window = _window(app)
        try:
            # It loads: the operator has to see what the file contains.
            assert window.load_fixture_file(str(short)) is True
            assert window.install_fixture() is False
            reason = window.label_processing_message.text()
            assert "13.00MHz" in reason
            assert "19.00MHz" in reason
            assert app.processing_config.find(STAGE_FIXTURE) is None
        finally:
            window.deleteLater()

    @staticmethod
    def test_a_one_port_touchstone_file_is_refused_with_a_reason(
        qt_app, tmp_path
    ):
        one_port = _one_port_s1p(tmp_path / "one_port.s1p")
        app = StubApp()
        window = _window(app)
        try:
            assert window.load_fixture_file(str(one_port)) is False
            assert "S21" in window.label_processing_message.text()
            assert window.install_fixture() is False
            assert app.processing_config.find(STAGE_FIXTURE) is None
        finally:
            window.deleteLater()

    @staticmethod
    def test_a_complete_two_port_on_the_fixture_grid_is_ready(
        qt_app, fixture_file
    ):
        app = StubApp()
        window = _window(app)
        try:
            window.load_fixture_file(str(fixture_file))
            preview = window.current_preview()
            assert preview.ok is True
            assert preview.sweep_span == (START, STOP)
            assert preview.fixture_span == (GRID[0], GRID[-1])
            assert "S12" in preview.describe()
        finally:
            window.deleteLater()


class TestProvenance:
    @staticmethod
    def test_apply_processing_records_the_fixture_it_removed(fixture_file):
        """The result states which fixture was removed, and its identity."""
        touchstone = _touchstone(fixture_file)
        from NanoVNASaver.Fixture import fixture_from_touchstone

        fixture = fixture_from_touchstone(touchstone, name="attenuator.s2p")
        stage = fixture_stage(fixture)
        config = ProcessingConfig.build(stage)

        measured = _through_attenuator(db=ATTENUATION_DB + 3.0)
        result = apply_processing(measured, config, reference_impedance=50.0)

        # Two cascaded 6 dB and 9 dB pads: the survivor is the 3 dB one.
        expected = 10 ** (-3.0 / 20)
        for name in ("S21", "S12"):
            assert result.channels[name][0].re == pytest.approx(
                expected, rel=1e-12
            )
        recorded = result.config.find(STAGE_FIXTURE)
        assert recorded is not None
        assert recorded.as_dict["identity"] == fixture.identity()
        assert "attenuator.s2p" in result.config.describe()
        assert fixture.identity()[:8] in result.config.describe()

    @staticmethod
    def test_the_stage_survives_a_configuration_round_trip(fixture_file):
        fixture = fixture_from_touchstone_named(fixture_file)
        config = ProcessingConfig.build(fixture_stage(fixture))
        restored = ProcessingConfig.from_dict(
            json.loads(json.dumps(config.to_dict()))
        )
        stage = restored.find(STAGE_FIXTURE)
        assert stage is not None
        assert stage.as_dict["identity"] == fixture.identity()
        # Rebuilding the network from the restored stage is exact.
        assert fixture_from_stage(stage).identity() == fixture.identity()

    @staticmethod
    def test_a_session_document_restores_the_same_fixture_identity(
        fixture_file, tmp_path
    ):
        fixture = fixture_from_touchstone_named(fixture_file)
        config = ProcessingConfig.build(fixture_stage(fixture))
        channels = _through_attenuator()
        snapshot = snapshot_from_datapoints(
            channels["S11"], channels["S21"], name="attenuator"
        ).with_processing(config)
        document = make_document([snapshot], title="fixture session")
        path = tmp_path / f"fixture{SESSION_SUFFIX}"
        save_session(document, str(path))

        restored = parse_session(path.read_bytes())
        snapshots = document_snapshots(restored)
        processing = snapshots["trace-0"].provenance.processing
        assert processing is not None
        stage = processing.find(STAGE_FIXTURE)
        assert stage is not None
        assert stage.as_dict["identity"] == fixture.identity()
        assert stage.as_dict["name"] == "attenuator.s2p"

    @staticmethod
    def test_the_archive_carries_the_fixture_samples(fixture_file):
        """The content, not a path, is stored: a moved file changes nothing."""
        fixture = fixture_from_touchstone_named(fixture_file)
        config = ProcessingConfig.build(fixture_stage(fixture))
        snapshot = snapshot_from_datapoints(
            _through_attenuator()["S11"],
            _through_attenuator()["S21"],
            name="attenuator",
        ).with_processing(config)
        payload = build_archive(make_document([snapshot]))
        restored = parse_session(payload)
        trace = restored.traces[0]
        stage = ProcessingStage.from_dict(
            trace["provenance"]["processing"]["stages"][0]
        )
        assert fixture_from_stage(stage).identity() == fixture.identity()


def _touchstone(path):
    from NanoVNASaver.Touchstone import Touchstone

    touchstone = Touchstone(str(path))
    touchstone.loads_strict(path.read_text(encoding="utf-8"))
    return touchstone


def fixture_from_touchstone_named(path) -> Fixture:
    from NanoVNASaver.Fixture import fixture_from_touchstone

    return fixture_from_touchstone(_touchstone(path), name="attenuator.s2p")


class TestAnalyticReference:
    """The fixture written by the test is the network the test claims."""

    @staticmethod
    def test_the_file_holds_a_matched_attenuator(fixture_file):
        fixture = fixture_from_touchstone_named(fixture_file)
        gain = 10 ** (-ATTENUATION_DB / 20)
        assert len(fixture.points) == len(GRID)
        for point in fixture.points:
            assert point.s11 == 0
            assert point.s22 == 0
            assert point.s21 == pytest.approx(gain, rel=1e-15)
            assert point.s12 == pytest.approx(gain, rel=1e-15)
        # |S21| in dB is the attenuation stated above.
        assert 20 * math.log10(abs(fixture.points[0].s21)) == pytest.approx(
            -ATTENUATION_DB, abs=1e-12
        )


class TestLiveWindow:
    """The group must exist in the real window and drive the real pipeline."""

    @staticmethod
    def test_the_live_window_shows_and_installs_a_fixture(qt_app, fixture_file):
        from NanoVNASaver.NanoVNASaver import NanoVNASaver

        window = NanoVNASaver()
        window.show()
        try:
            settings = window.windows["sweep_settings"]
            assert settings.label_fixture.text() == "no fixture loaded"
            window.sweep_control.set_start(GRID[0])
            window.sweep_control.set_end(GRID[-1])
            window.sweep.set_points(len(GRID))

            channels = _through_attenuator()
            window.data.s11 = channels["S11"]
            window.data.s21 = channels["S21"]
            window.data.s12 = channels["S12"]
            window.data.s22 = channels["S22"]

            assert settings.load_fixture_file(str(fixture_file)) is True
            assert settings.install_fixture() is True
            stage = window.processing_config.find(STAGE_FIXTURE)
            assert stage is not None
            expected = fixture_from_touchstone_named(fixture_file)
            assert stage.as_dict["identity"] == expected.identity()
            assert "Fixture de-embedding" in window.processing_config.describe()
        finally:
            # Every control here marks the session dirty by design.
            window.session.dirty = False
            window.close()
