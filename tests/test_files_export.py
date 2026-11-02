#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020  Rune B. Broberg
#  Copyright (C) 2020,2021 NanoVNA-Saver Authors
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
"""FIX-02: the export UI must state what it will write and keep channels."""

import math
import os

import pytest

from NanoVNASaver.RFTools import Datapoint
from NanoVNASaver.Touchstone import Touchstone

VALID_S2P = "./tests/data/valid.s2p"


@pytest.fixture(scope="module")
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def communicate(qt_app):
    from PySide6 import QtCore

    class Communicate(QtCore.QObject):
        data_available = QtCore.Signal()

    return Communicate()


class FakeApp:
    def __init__(self, communicate):
        self.data = Touchstone()
        self.sweepSource = ""
        self.communicate = communicate
        self.updated = 0

    def display_window(self, name):
        self.name = name

    def dataUpdated(self):
        self.updated += 1

    def saveData(self, data, data21, source=None):
        self.data.s11 = data
        self.data.s21 = data21
        self.sweepSource = source or ""


@pytest.fixture
def files_window(communicate):
    from NanoVNASaver.Windows.Files import FilesWindow

    window = FilesWindow(FakeApp(communicate))
    yield window
    window.close()


def test_s2p_label_names_placeholders_for_forward_only_data(files_window):
    app = files_window.app
    app.data.s11 = [Datapoint(1, 0.1, 0.0)]
    app.data.s21 = [Datapoint(1, 0.5, 0.0)]
    files_window.refresh_export_actions()
    assert "S12/S22 = 0" in files_window.btn_export_s2p.text()
    assert "not a complete two-port characterization" in (
        files_window.btn_export_s2p.toolTip()
    )


def test_s2p_label_is_plain_when_all_channels_are_measured(files_window):
    app = files_window.app
    app.data.s11 = [Datapoint(1, 0.1, 0.0)]
    app.data.s21 = [Datapoint(1, 0.5, 0.0)]
    app.data.s12 = [Datapoint(1, 0.0, 0.0)]
    app.data.s22 = [Datapoint(1, 0.0, 0.0)]
    files_window.refresh_export_actions()
    assert "= 0" not in files_window.btn_export_s2p.text()
    assert "zero" not in files_window.btn_export_s2p.toolTip()


def test_s2p_label_updates_on_new_data(files_window):
    app = files_window.app
    app.data.s11 = [Datapoint(1, 0.1, 0.0)]
    app.data.s21 = [Datapoint(1, 0.5, 0.0)]
    files_window.refresh_export_actions()
    assert "= 0" in files_window.btn_export_s2p.text()

    app.data.s12 = [Datapoint(1, 0.0, 0.0)]
    app.data.s22 = [Datapoint(1, 0.0, 0.0)]
    app.communicate.data_available.emit()
    assert "= 0" not in files_window.btn_export_s2p.text()


def test_imported_full_s2p_survives_reexport(files_window, tmp_path):
    app = files_window.app
    imported = Touchstone(VALID_S2P)
    imported.load()
    app.saveData(imported.s11, imported.s21, VALID_S2P)
    files_window._imported = imported
    files_window._imported_source = VALID_S2P
    files_window.refresh_export_actions()
    assert "= 0" not in files_window.btn_export_s2p.text()

    target = tmp_path / "roundtrip.s2p"
    measurement = files_window.current_touchstone()
    measurement.filename = str(target)
    measurement.save(4)

    reloaded = Touchstone(str(target))
    reloaded.load()
    for parameter in ("s11", "s21", "s12", "s22"):
        original = getattr(imported, parameter)
        exported = getattr(reloaded, parameter)
        assert len(exported) == len(original)
        for first, second in zip(original, exported, strict=True):
            assert first.freq == second.freq
            assert first.re == pytest.approx(second.re)
            assert first.im == pytest.approx(second.im)

    assert reloaded.s12[0].re == pytest.approx(6.75290e-001)
    assert reloaded.s22[0].im == pytest.approx(3.08078e-004)


def test_inconsistent_data_is_reported_and_nothing_written(
    files_window, tmp_path, monkeypatch
):
    from PySide6 import QtWidgets

    app = files_window.app
    app.data.s11 = [Datapoint(1, 0.1, 0.0), Datapoint(2, 0.2, 0.0)]
    app.data.s21 = [Datapoint(1, 0.5, 0.0), Datapoint(2, 0.0, math.nan)]
    target = tmp_path / "rejected.s2p"

    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "exec",
        lambda self: QtWidgets.QFileDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "selectedFiles", lambda self: [str(target)]
    )

    files_window.exportFile(4)

    assert warnings, "the failure must be reported to the user"
    assert "not finite" in warnings[0][2]
    assert not target.exists()


def test_non_ascending_data_is_reported_and_nothing_written(
    files_window, tmp_path, monkeypatch
):
    """Bug 5: descending samples are refused, not sorted or truncated."""
    from PySide6 import QtWidgets

    app = files_window.app
    app.data.s11 = [Datapoint(3, 0.1, 0.0), Datapoint(1, 0.2, 0.0)]
    target = tmp_path / "descending.s1p"
    target.write_text("sentinel\n", encoding="utf-8")

    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "exec",
        lambda self: QtWidgets.QFileDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "selectedFiles", lambda self: [str(target)]
    )

    files_window.exportFile(1)

    assert warnings, "the failure must be reported to the user"
    assert "never descending" in warnings[0][2]
    assert target.read_text(encoding="utf-8") == "sentinel\n"


@pytest.fixture(scope="module")
def _log_sweep_frequencies():
    """Frequencies of a real two-segment logarithmic sweep.

    ``Sweep.get_index_range`` makes each segment end on the next one's
    start, so the grid holds its boundary frequency twice -- the shape the
    export used to refuse.
    """
    from NanoVNASaver.Settings.Sweep import Properties, Sweep

    sweep = Sweep(14_000_000, 14_300_000, 101, 2, Properties(logarithmic=True))
    return list(sweep.get_frequencies())


def test_a_log_sweep_repeat_is_exported_not_refused(
    files_window, tmp_path, monkeypatch, _log_sweep_frequencies
):
    """Bug 5 regression: a repeated boundary must not block the export."""
    from PySide6 import QtWidgets

    app = files_window.app
    app.data.s11 = [Datapoint(f, 0.5, 0.1) for f in _log_sweep_frequencies]
    target = tmp_path / "log.s1p"

    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "exec",
        lambda self: QtWidgets.QFileDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "selectedFiles", lambda self: [str(target)]
    )

    files_window.exportFile(1)

    assert warnings == []
    written = [
        line
        for line in target.read_text(encoding="utf-8").splitlines()
        if line and line[0].isdigit()
    ]
    assert len(written) == len(_log_sweep_frequencies)
    assert written[100].split()[0] == written[101].split()[0]


def test_s1p_export_keeps_only_s11(files_window, tmp_path, monkeypatch):
    from PySide6 import QtWidgets

    app = files_window.app
    app.data.s11 = [Datapoint(1, 0.1, 0.0)]
    app.data.s21 = [Datapoint(1, 0.5, 0.0)]
    target = tmp_path / "only_s11.s1p"

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "exec",
        lambda self: QtWidgets.QFileDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "selectedFiles", lambda self: [str(target)]
    )

    files_window.exportFile(1)

    rows = [
        line
        for line in target.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(("!", "#"))
    ]
    assert rows == ["1 0.1 0.0"]


def test_exported_copy_keeps_the_simulated_marker(files_window):
    """SIM-02.2: the copy handed to Touchstone must carry the flag.

    Without it the exported file is indistinguishable from a physical
    measurement even though the window says [SIMULATED].
    """
    app = files_window.app
    app.data.s11 = [Datapoint(1, 0.1, 0.0)]
    app.data.simulated = True
    app.data.simulated_scenario = "antenna"

    touchstone = files_window.current_touchstone()
    assert touchstone.simulated is True
    assert touchstone.simulated_scenario == "antenna"
    assert "SIMULATED DATA" in touchstone.saves(1)
    assert "scenario antenna" in touchstone.saves(1)


def test_physical_data_stays_marked_as_physical(files_window):
    app = files_window.app
    app.data.s11 = [Datapoint(1, 0.1, 0.0)]
    app.data.simulated = False

    touchstone = files_window.current_touchstone()
    assert touchstone.simulated is False
    assert "SIMULATED" not in touchstone.saves(1)
