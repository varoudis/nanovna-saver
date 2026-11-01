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
import logging
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from ..Touchstone import PARAMETERS, Touchstone, TouchstoneExportError
from .Defaults import make_scrollable
from .ui import get_window_icon

if TYPE_CHECKING:
    from ..NanoVNASaver.NanoVNASaver import NanoVNASaver as vna_app
logger = logging.getLogger(__name__)


class FilesWindow(QtWidgets.QWidget):
    def __init__(self, app: "vna_app"):
        super().__init__()
        #: Dataset imported with "Load as sweep", kept so genuine S12/S22
        #: (and its reference impedance and comments) survive a re-export.
        #: Only used while it is still the displayed measurement.
        self._imported: Touchstone | None = None
        self._imported_source = ""

        self.app = app

        self.setWindowTitle("Files")
        self.setWindowIcon(get_window_icon())
        self.setMinimumWidth(200)
        QtGui.QShortcut(QtCore.Qt.Key.Key_Escape, self, self.hide)

        file_window_layout = QtWidgets.QVBoxLayout()
        make_scrollable(self, file_window_layout)

        load_file_control_box = QtWidgets.QGroupBox("Import file")
        load_file_control_box.setMaximumWidth(300)
        load_file_control_layout = QtWidgets.QFormLayout(load_file_control_box)

        btn_load_sweep = QtWidgets.QPushButton("Load as sweep")
        btn_load_sweep.clicked.connect(self.loadSweepFile)
        btn_load_reference = QtWidgets.QPushButton("Load reference")
        btn_load_reference.clicked.connect(self.loadReferenceFile)
        load_file_control_layout.addRow(btn_load_sweep)
        load_file_control_layout.addRow(btn_load_reference)

        file_window_layout.addWidget(load_file_control_box)

        save_file_control_box = QtWidgets.QGroupBox("Export file")
        save_file_control_box.setMaximumWidth(300)
        save_file_control_layout = QtWidgets.QFormLayout(save_file_control_box)

        btn_export_file = QtWidgets.QPushButton("Save 1-Port file (S1P)")
        btn_export_file.clicked.connect(lambda: self.exportFile(1))
        save_file_control_layout.addRow(btn_export_file)

        self.btn_export_s2p = QtWidgets.QPushButton("Save 2-Port file (S2P)")
        self.btn_export_s2p.clicked.connect(lambda: self.exportFile(4))
        save_file_control_layout.addRow(self.btn_export_s2p)

        file_window_layout.addWidget(save_file_control_box)

        self.refresh_export_actions()
        # Keep the S2P label honest when a sweep or an import replaces the
        # displayed data.
        self.app.communicate.data_available.connect(self.refresh_export_actions)

        btn_open_file_window = QtWidgets.QPushButton("Files ...")
        btn_open_file_window.clicked.connect(
            lambda: self.app.display_window("file")
        )

    def displayed_import(self) -> Touchstone | None:
        """The imported dataset, or ``None`` when it is no longer shown."""
        if self._imported is None:
            return None
        if self.app.sweepSource != self._imported_source:
            return None
        return self._imported

    def current_touchstone(self) -> Touchstone:
        """Return the channels that belong to the displayed measurement.

        The measurement itself lives in ``app.data`` (its S-parameters, the
        reference impedance and its comments).  When the displayed data
        came from an imported file, that file is authoritative for the
        channels ``app.data`` does not keep, so genuine S12/S22 are not
        degraded to synthesized zeros on re-export.  Any channel left
        empty here is written as an explicit zero placeholder by
        ``Touchstone.saves``.
        """
        data = self.app.data
        imported = self.displayed_import()
        source = imported if imported is not None else data
        touchstone = Touchstone()
        touchstone.opts.resistance = source.opts.resistance
        touchstone.comments = list(source.comments)
        # Provenance must survive the copy: without these the exported file
        # would silently lose the SIMULATED marker (SIM-02.2).
        touchstone.simulated = bool(
            getattr(source, "simulated", False)
            or getattr(data, "simulated", False)
        )
        touchstone.simulated_scenario = str(
            getattr(source, "simulated_scenario", "")
            or getattr(data, "simulated_scenario", "")
        )
        for parameter in PARAMETERS:
            samples = data.channel(parameter)
            if not samples and imported is not None:
                samples = imported.channel(parameter)
            setattr(touchstone, parameter.lower(), samples)
        return touchstone

    def refresh_export_actions(self) -> None:
        """Label the S2P action with what will actually be written."""
        placeholders = self.current_touchstone().placeholder_parameters(4)
        if placeholders:
            names = "/".join(placeholders)
            self.btn_export_s2p.setText(f"Save 2-Port file (S2P, {names} = 0)")
            self.btn_export_s2p.setToolTip(
                f"{names} are not measured: they are written as zero"
                " placeholders and named as such in the file comments."
                " This is not a complete two-port characterization."
            )
        else:
            self.btn_export_s2p.setText("Save 2-Port file (S2P)")
            self.btn_export_s2p.setToolTip(
                "All four S-parameters are measured and are written as"
                " recorded."
            )

    def exportFile(self, parameter_count: int = 1):
        """Export the displayed measurement as a Touchstone file.

        ``parameter_count`` is the number of S-parameter columns: 1 for an
        ``.s1p`` file (S11 only) or 4 for an ``.s2p`` file with the full
        S11, S21, S12, S22 column set.  S12/S22 that are not measured are
        written as explicit zero placeholders.
        """
        if len(self.app.data.s11) == 0:
            QtWidgets.QMessageBox.warning(
                self, "No data to save", "There is no data to save."
            )
            return
        if parameter_count > 2 and len(self.app.data.s21) == 0:
            QtWidgets.QMessageBox.warning(
                self, "No S21 data to save", "There is no S21 data to save."
            )
            return

        filedialog = QtWidgets.QFileDialog(self)
        if parameter_count == 1:
            filedialog.setDefaultSuffix("s1p")
            filedialog.setNameFilter(
                "Touchstone 1-Port Files (*.s1p);;All files (*.*)"
            )
        else:
            filedialog.setDefaultSuffix("s2p")
            filedialog.setNameFilter(
                "Touchstone 2-Port Files (*.s2p);;All files (*.*)"
            )
        filedialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        selected = filedialog.exec()
        if not selected:
            return
        filename = filedialog.selectedFiles()[0]
        if filename == "":
            logger.debug("No file name selected.")
            return

        ts = self.current_touchstone()
        ts.filename = filename
        placeholders = ts.placeholder_parameters(parameter_count)
        if placeholders:
            logger.info(
                "Exporting zero placeholders for %s",
                ", ".join(placeholders),
            )
        try:
            ts.save(parameter_count)
        except TouchstoneExportError as e:
            logger.warning("Refusing to write inconsistent data: %s", e)
            QtWidgets.QMessageBox.warning(
                self,
                "Export failed",
                f"The measurement cannot be exported as requested:\n\n{e}",
            )
            return
        except IOError as e:
            logger.exception("Error during file export: %s", e)
            return

    def loadReferenceFile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            filter="Touchstone Files (*.s1p *.s2p);;All files (*.*)"
        )
        if filename != "":
            self.app.resetReference()
            t = Touchstone(filename)
            t.load()
            self.app.setReference(t.s11, t.s21, filename)

    def loadSweepFile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            filter="Touchstone Files (*.s1p *.s2p);;All files (*.*)"
        )
        if filename != "":
            self.app.data.s11 = []
            self.app.data.s21 = []
            t = Touchstone(filename)
            t.load()
            self.app.saveData(t.s11, t.s21, filename)
            self._imported = t
            self._imported_source = filename
            self.refresh_export_actions()
            self.app.dataUpdated()
