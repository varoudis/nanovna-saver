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
import os
from dataclasses import replace
from functools import partial

from PySide6 import QtCore, QtGui, QtWidgets

from ..Calibration import Calibration
from ..CalibrationHealth import (
    DEFAULT_REMINDER_HOURS,
    DEFAULT_TOLERANCE,
    VERIFIABLE_STANDARDS,
    VerificationError,
    verify_standard,
)
from ..CalibrationKits import (
    BUILTIN_NAMES,
    COEFFICIENT_FIELDS,
    MODEL_FILE,
    MODEL_IDEAL,
    CalibrationKit,
    CalibrationKitError,
    KitLibrary,
    kit_from_legacy_values,
    legacy_values,
)
from ..RFTools import Datapoint, corr_att_data
from ..Settings.Sweep import SweepMode
from ..SweepState import SweepState
from ..Touchstone import Touchstone
from .Defaults import make_scrollable
from .ui import get_window_icon

logger = logging.getLogger(__name__)


def _format_cal_label(size: int, prefix: str = "Set") -> str:
    return f"{prefix} ({size} points)"


def getFloatValue(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def formatCoefficient(value: float) -> str:
    """Format a coefficient for a text field without float noise."""
    return f"{value:.15g}"


class CalibrationWindow(QtWidgets.QWidget):
    next_step = -1

    def __init__(self, app: QtWidgets.QWidget):
        super().__init__()
        self.app = app

        self.setMinimumWidth(450)
        self.setWindowTitle("Calibration")
        self.setWindowIcon(get_window_icon())
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
        )

        QtGui.QShortcut(QtCore.Qt.Key.Key_Escape, self, self.hide)

        top_layout = QtWidgets.QHBoxLayout()
        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        top_layout.addLayout(left_layout)
        top_layout.addLayout(right_layout)

        make_scrollable(self, top_layout)

        calibration_status_group = QtWidgets.QGroupBox("Active calibration")
        calibration_status_layout = QtWidgets.QFormLayout()
        self.calibration_status_label = QtWidgets.QLabel("Device calibration")
        self.calibration_source_label = QtWidgets.QLabel("NanoVNA")
        calibration_status_layout.addRow(
            "Calibration:", self.calibration_status_label
        )
        calibration_status_layout.addRow(
            "Source:", self.calibration_source_label
        )
        calibration_status_group.setLayout(calibration_status_layout)
        left_layout.addWidget(calibration_status_group)

        calibration_control_group = QtWidgets.QGroupBox("Calibrate")
        calibration_control_layout = QtWidgets.QFormLayout(
            calibration_control_group
        )
        cal_btn = {}
        self.cal_label = {}
        for label_name in (
            "short",
            "open",
            "load",
            "through",
            "thrurefl",
            "isolation",
        ):
            self.cal_label[label_name] = QtWidgets.QLabel("Uncalibrated")
            cal_btn[label_name] = QtWidgets.QPushButton(label_name.capitalize())
            cal_btn[label_name].setMinimumHeight(20)
            cal_btn[label_name].clicked.connect(
                partial(self.manual_save, label_name)
            )
            calibration_control_layout.addRow(
                cal_btn[label_name], self.cal_label[label_name]
            )

        self.input_offset_delay = QtWidgets.QDoubleSpinBox()
        self.input_offset_delay.setMinimumHeight(20)
        self.input_offset_delay.setValue(0)
        self.input_offset_delay.setSuffix(" ps")
        self.input_offset_delay.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.input_offset_delay.valueChanged.connect(self.setOffsetDelay)
        self.input_offset_delay.setRange(-10e6, 10e6)

        calibration_control_layout.addRow(QtWidgets.QLabel(""))
        calibration_control_layout.addRow(
            "Offset delay", self.input_offset_delay
        )

        self.btn_automatic = QtWidgets.QPushButton("Calibration assistant")
        self.btn_automatic.setMinimumHeight(20)
        calibration_control_layout.addRow(self.btn_automatic)
        self.btn_automatic.clicked.connect(self.automaticCalibration)

        apply_reset_layout = QtWidgets.QHBoxLayout()

        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setMinimumHeight(20)
        btn_apply.clicked.connect(self.calculate)

        btn_reset = QtWidgets.QPushButton("Reset")
        btn_reset.setMinimumHeight(20)
        btn_reset.clicked.connect(self.reset)

        apply_reset_layout.addWidget(btn_apply)
        apply_reset_layout.addWidget(btn_reset)

        calibration_control_layout.addRow(apply_reset_layout)

        left_layout.addWidget(calibration_control_group)

        calibration_notes_group = QtWidgets.QGroupBox("Notes")
        calibration_notes_layout = QtWidgets.QVBoxLayout(
            calibration_notes_group
        )
        self.notes_textedit = QtWidgets.QPlainTextEdit()
        calibration_notes_layout.addWidget(self.notes_textedit)

        left_layout.addWidget(calibration_notes_group)
        left_layout.addWidget(self._buildHealthGroup())

        file_box = QtWidgets.QGroupBox("Files")
        file_layout = QtWidgets.QFormLayout(file_box)
        btn_save_file = QtWidgets.QPushButton("Save calibration")
        btn_save_file.setMinimumHeight(20)
        btn_save_file.clicked.connect(lambda: self.saveCalibration())
        btn_load_file = QtWidgets.QPushButton("Load calibration")
        btn_load_file.setMinimumHeight(20)
        btn_load_file.clicked.connect(lambda: self.loadCalibration())

        save_load_layout = QtWidgets.QHBoxLayout()
        save_load_layout.addWidget(btn_save_file)
        save_load_layout.addWidget(btn_load_file)

        file_layout.addRow(save_load_layout)

        left_layout.addWidget(file_box)

        cal_standard_box = QtWidgets.QGroupBox("Calibration standards")
        cal_standard_layout = QtWidgets.QFormLayout(cal_standard_box)
        self.use_ideal_values = QtWidgets.QRadioButton("Use ideal values")
        self.use_s1p_files = QtWidgets.QRadioButton("Use s1p files")
        self.use_coefficients = QtWidgets.QRadioButton("Use coefficients")

        self.use_ideal_values.setChecked(True)
        self.radio_group = QtWidgets.QButtonGroup(self)
        self.radio_group.addButton(self.use_ideal_values)
        self.radio_group.addButton(self.use_s1p_files)
        self.radio_group.addButton(self.use_coefficients)
        self.radio_group.buttonClicked.connect(self.calStandardChanged)

        self.radio_layout = QtWidgets.QHBoxLayout()
        self.radio_layout.addWidget(self.use_ideal_values)
        self.radio_layout.addWidget(self.use_s1p_files)
        self.radio_layout.addWidget(self.use_coefficients)
        cal_standard_layout.addRow(self.radio_layout)

        self.file_button_short = QtWidgets.QPushButton("Short S1P file")
        self.file_button_open = QtWidgets.QPushButton("Open S1P file")
        self.file_button_load = QtWidgets.QPushButton("Load S1P file")
        self.file_button_short.setEnabled(False)
        self.file_button_open.setEnabled(False)
        self.file_button_load.setEnabled(False)

        cal_standard_layout.addRow(self.file_button_short)
        cal_standard_layout.addRow(self.file_button_open)
        cal_standard_layout.addRow(self.file_button_load)

        self.file_button_open.clicked.connect(self.select_file_open)
        self.file_button_short.clicked.connect(self.select_file_short)
        self.file_button_load.clicked.connect(self.select_file_load)

        self.cal_short_box = QtWidgets.QGroupBox("Short")
        cal_short_form = QtWidgets.QFormLayout(self.cal_short_box)
        self.cal_short_box.setDisabled(True)
        self.short_l0_input = QtWidgets.QLineEdit("0")
        self.short_l0_input.setMinimumHeight(20)
        self.short_l1_input = QtWidgets.QLineEdit("0")
        self.short_l1_input.setMinimumHeight(20)
        self.short_l2_input = QtWidgets.QLineEdit("0")
        self.short_l2_input.setMinimumHeight(20)
        self.short_l3_input = QtWidgets.QLineEdit("0")
        self.short_l3_input.setMinimumHeight(20)
        self.short_length = QtWidgets.QLineEdit("0")
        self.short_length.setMinimumHeight(20)
        cal_short_form.addRow("L0 (H(e-12))", self.short_l0_input)
        cal_short_form.addRow("L1 (H(e-24))", self.short_l1_input)
        cal_short_form.addRow("L2 (H(e-33))", self.short_l2_input)
        cal_short_form.addRow("L3 (H(e-42))", self.short_l3_input)
        cal_short_form.addRow("Offset Delay (ps)", self.short_length)

        self.cal_open_box = QtWidgets.QGroupBox("Open")
        cal_open_form = QtWidgets.QFormLayout(self.cal_open_box)
        self.cal_open_box.setDisabled(True)
        self.open_c0_input = QtWidgets.QLineEdit("50")
        self.open_c0_input.setMinimumHeight(20)
        self.open_c1_input = QtWidgets.QLineEdit("0")
        self.open_c1_input.setMinimumHeight(20)
        self.open_c2_input = QtWidgets.QLineEdit("0")
        self.open_c2_input.setMinimumHeight(20)
        self.open_c3_input = QtWidgets.QLineEdit("0")
        self.open_c3_input.setMinimumHeight(20)
        self.open_length = QtWidgets.QLineEdit("0")
        self.open_length.setMinimumHeight(20)
        cal_open_form.addRow("C0 (F(e-15))", self.open_c0_input)
        cal_open_form.addRow("C1 (F(e-27))", self.open_c1_input)
        cal_open_form.addRow("C2 (F(e-36))", self.open_c2_input)
        cal_open_form.addRow("C3 (F(e-45))", self.open_c3_input)
        cal_open_form.addRow("Offset Delay (ps)", self.open_length)

        self.cal_load_box = QtWidgets.QGroupBox("Load")
        cal_load_form = QtWidgets.QFormLayout(self.cal_load_box)
        self.cal_load_box.setDisabled(True)
        self.load_resistance = QtWidgets.QLineEdit("50")
        self.load_resistance.setMinimumHeight(20)
        self.load_inductance = QtWidgets.QLineEdit("0")
        self.load_inductance.setMinimumHeight(20)
        self.load_capacitance = QtWidgets.QLineEdit("0")
        self.load_capacitance.setMinimumHeight(20)
        # self.load_capacitance.setDisabled(True)  # Not yet implemented
        self.load_length = QtWidgets.QLineEdit("0")
        self.load_length.setMinimumHeight(20)
        cal_load_form.addRow("Resistance (\N{OHM SIGN})", self.load_resistance)
        cal_load_form.addRow("Inductance (H(e-12))", self.load_inductance)
        cal_load_form.addRow("Capacitance (F(e-15))", self.load_capacitance)
        cal_load_form.addRow("Offset Delay (ps)", self.load_length)

        self.cal_through_box = QtWidgets.QGroupBox("Through")
        cal_through_form = QtWidgets.QFormLayout(self.cal_through_box)
        self.cal_through_box.setDisabled(True)
        self.through_length = QtWidgets.QLineEdit("0")
        self.through_length.setMinimumHeight(20)
        cal_through_form.addRow("Offset Delay (ps)", self.through_length)

        cal_standard_layout.addWidget(self.cal_short_box)
        cal_standard_layout.addWidget(self.cal_open_box)
        cal_standard_layout.addWidget(self.cal_load_box)
        cal_standard_layout.addWidget(self.cal_through_box)

        self.cal_kit_box = QtWidgets.QGroupBox("Calibration kit")
        cal_kit_layout = QtWidgets.QVBoxLayout(self.cal_kit_box)
        self.cal_kit_selector = QtWidgets.QComboBox()
        self.cal_kit_selector.setMinimumHeight(20)
        self.cal_kit_selector.activated.connect(self.kitSelected)
        cal_kit_layout.addWidget(self.cal_kit_selector)
        self.cal_kit_detail = QtWidgets.QLabel("")
        self.cal_kit_detail.setWordWrap(True)
        cal_kit_layout.addWidget(self.cal_kit_detail)
        kit_button_layout = QtWidgets.QHBoxLayout()
        self.kitApplyButton = QtWidgets.QPushButton("Apply")
        self.kitSaveButton = QtWidgets.QPushButton("Save")
        self.kitDuplicateButton = QtWidgets.QPushButton("Duplicate")
        self.kitRenameButton = QtWidgets.QPushButton("Rename")
        self.kitDeleteButton = QtWidgets.QPushButton("Delete")
        for button, slot in (
            (self.kitApplyButton, self.applyCalibrationKit),
            (self.kitSaveButton, self.saveCalibrationKit),
            (self.kitDuplicateButton, self.duplicateCalibrationKit),
            (self.kitRenameButton, self.renameCalibrationKit),
            (self.kitDeleteButton, self.deleteCalibrationKit),
        ):
            button.setMinimumHeight(20)
            button.clicked.connect(slot)
            kit_button_layout.addWidget(button)
        cal_kit_layout.addLayout(kit_button_layout)
        kit_transfer_layout = QtWidgets.QHBoxLayout()
        btn_import_kit = QtWidgets.QPushButton("Import kit")
        btn_import_kit.setMinimumHeight(20)
        btn_import_kit.clicked.connect(self.importCalibrationKit)
        btn_export_kit = QtWidgets.QPushButton("Export kit")
        btn_export_kit.setMinimumHeight(20)
        btn_export_kit.clicked.connect(self.exportCalibrationKit)
        kit_transfer_layout.addWidget(btn_import_kit)
        kit_transfer_layout.addWidget(btn_export_kit)
        cal_kit_layout.addLayout(kit_transfer_layout)
        cal_standard_layout.addWidget(self.cal_kit_box)
        self.cal_standard_save_box = QtWidgets.QGroupBox("Saved settings")
        cal_standard_save_layout = QtWidgets.QVBoxLayout(
            self.cal_standard_save_box
        )
        self.cal_standard_save_box.setDisabled(True)

        self.cal_standard_save_selector = QtWidgets.QComboBox()
        self.cal_standard_save_selector.setMinimumHeight(20)
        self.listCalibrationStandards()
        cal_standard_save_layout.addWidget(self.cal_standard_save_selector)
        cal_standard_save_button_layout = QtWidgets.QHBoxLayout()
        btn_save_standard = QtWidgets.QPushButton("Save")
        btn_save_standard.setMinimumHeight(20)
        btn_save_standard.clicked.connect(self.saveCalibrationStandard)
        btn_load_standard = QtWidgets.QPushButton("Load")
        btn_load_standard.setMinimumHeight(20)
        btn_load_standard.clicked.connect(self.loadCalibrationStandard)
        btn_delete_standard = QtWidgets.QPushButton("Delete")
        btn_delete_standard.setMinimumHeight(20)
        btn_delete_standard.clicked.connect(self.deleteCalibrationStandard)
        cal_standard_save_button_layout.addWidget(btn_load_standard)
        cal_standard_save_button_layout.addWidget(btn_save_standard)
        cal_standard_save_button_layout.addWidget(btn_delete_standard)
        cal_standard_save_layout.addLayout(cal_standard_save_button_layout)

        cal_standard_layout.addWidget(self.cal_standard_save_box)
        right_layout.addWidget(cal_standard_box)
        self.open_touchstone = None
        self.short_touchstone = None
        self.load_touchstone = None

        # The legacy "Calibration standards" keys, in the order they are
        # shown; kits are written into and read back from these fields.
        self.coefficientFields = {
            "ShortL0": self.short_l0_input,
            "ShortL1": self.short_l1_input,
            "ShortL2": self.short_l2_input,
            "ShortL3": self.short_l3_input,
            "ShortDelay": self.short_length,
            "OpenC0": self.open_c0_input,
            "OpenC1": self.open_c1_input,
            "OpenC2": self.open_c2_input,
            "OpenC3": self.open_c3_input,
            "OpenDelay": self.open_length,
            "LoadR": self.load_resistance,
            "LoadL": self.load_inductance,
            "LoadC": self.load_capacitance,
            "LoadDelay": self.load_length,
            "ThroughDelay": self.through_length,
        }

        self._pending_element: (
            tuple[object, dict[str, str], object, str, str] | None
        ) = None
        self._kit_load_error = ""
        self.kit_library = KitLibrary()
        try:
            self.kit_library.load()
        except (CalibrationKitError, OSError) as exc:
            # A broken library file must not stop the calibration window
            # from opening; it is reported by the selector detail label.
            logger.warning(
                "Could not read the calibration kit library: %s", exc
            )
            self._kit_load_error = str(exc)
        self.migrateLegacyStandards()
        self.refreshKitSelector()

    # -- calibration health (CAL-02) -----------------------------------
    def _buildHealthGroup(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Calibration age and verification")
        layout = QtWidgets.QFormLayout(group)
        self.health_age_label = QtWidgets.QLabel("No calibration applied")
        self.health_age_label.setWordWrap(True)
        layout.addRow("Age:", self.health_age_label)
        self.health_metadata_label = QtWidgets.QLabel("")
        self.health_metadata_label.setWordWrap(True)
        layout.addRow("Made with:", self.health_metadata_label)

        self.operator_input = QtWidgets.QLineEdit()
        self.operator_input.setPlaceholderText("Optional")
        self.operator_input.editingFinished.connect(self.operatorEdited)
        layout.addRow("Operator:", self.operator_input)

        self.reminder_input = QtWidgets.QDoubleSpinBox()
        self.reminder_input.setRange(0.0, 8760.0)
        self.reminder_input.setDecimals(1)
        self.reminder_input.setSuffix(" h")
        self.reminder_input.setSpecialValueText("Off")
        self.reminder_input.setToolTip(
            "Remind me after this long. There is no fixed age at which a"
            " calibration expires, so this is off until you choose one."
        )
        self.reminder_input.setValue(
            float(
                self.app.settings.value(
                    "CalibrationAgeReminderHours",
                    DEFAULT_REMINDER_HOURS,
                    type=float,
                )
            )
        )
        self.reminder_input.valueChanged.connect(self.reminderChanged)
        layout.addRow("Remind after:", self.reminder_input)

        self.verify_standard_input = QtWidgets.QComboBox()
        for name in VERIFIABLE_STANDARDS:
            self.verify_standard_input.addItem(name.capitalize(), name)
        layout.addRow("Connected standard:", self.verify_standard_input)

        self.verify_tolerance_input = QtWidgets.QDoubleSpinBox()
        self.verify_tolerance_input.setRange(0.001, 2.0)
        self.verify_tolerance_input.setDecimals(3)
        self.verify_tolerance_input.setSingleStep(0.005)
        self.verify_tolerance_input.setToolTip(
            "Largest accepted |ΔΓ| between the corrected standard and the"
            " model this calibration assumes. Your pass/fail threshold, not"
            " an uncertainty."
        )
        self.verify_tolerance_input.setValue(
            float(
                self.app.settings.value(
                    "CalibrationVerifyTolerance", DEFAULT_TOLERANCE, type=float
                )
            )
        )
        layout.addRow("Tolerance:", self.verify_tolerance_input)

        self.verify_button = QtWidgets.QPushButton("Sweep and verify")
        self.verify_button.clicked.connect(self.verifyConnectedStandard)
        layout.addRow(self.verify_button)
        self.health_verification_label = QtWidgets.QLabel("Not verified")
        self.health_verification_label.setWordWrap(True)
        layout.addRow("Result:", self.health_verification_label)
        self._verification_pending: (
            tuple[str, object, float, Calibration, int] | None
        ) = None
        return group

    def refreshHealth(self) -> None:
        """Show age, metadata and verification, each in its own row."""
        calibration = self.app.calibration
        age = self.app.calibration_age_now()
        self.health_age_label.setText(
            "No calibration applied" if age is None else age.describe()
        )
        metadata = calibration.metadata
        self.health_metadata_label.setText(
            metadata.describe() or "Not recorded"
        )
        if self.operator_input.text() != metadata.operator:
            self.operator_input.setText(metadata.operator)
        results = [
            calibration.verifications[name].describe()
            for name in VERIFIABLE_STANDARDS
            if name in calibration.verifications
        ]
        self.health_verification_label.setText(
            "\n".join(results) if results else "Not verified"
        )

    def recordMetadata(self) -> None:
        """Note what a calibration measured here was measured with.

        A loaded file keeps what it recorded: the device connected now is not
        evidence of which one measured those standards.
        """
        calibration = self.app.calibration
        if not calibration.measured_here:
            return
        device = self.app.vna
        calibration.metadata = replace(
            calibration.metadata,
            device=device.name if device.connected() else "",
            operator=self.operator_input.text().strip(),
        )

    def operatorEdited(self) -> None:
        calibration = self.app.calibration
        calibration.metadata = replace(
            calibration.metadata, operator=self.operator_input.text().strip()
        )
        self.refreshHealth()

    def reminderChanged(self, hours: float) -> None:
        self.app.settings.setValue("CalibrationAgeReminderHours", float(hours))
        self.refreshHealth()
        self.app.refreshCoverage()

    def verifyConnectedStandard(self) -> None:
        """Sweep the connected standard, then check it (see ``_verified``)."""
        if self._verification_pending is not None:
            return
        problem = ""
        if not self.app.calibration.isCalculated:
            problem = "Apply a calibration before verifying it."
        elif not self.app.vna.connected():
            problem = "Connect a device to re-measure the standard."
        elif self.app.worker.isRunning():
            problem = "Stop the running sweep before verifying."
        elif self.app.sweep.properties.mode == SweepMode.CONTINOUS:
            problem = "Verification needs a single sweep, not a continuous one."
        if problem:
            self.health_verification_label.setText(problem)
            return
        tolerance = float(self.verify_tolerance_input.value())
        self.app.settings.setValue("CalibrationVerifyTolerance", tolerance)
        token = object()
        self._verification_pending = (
            str(self.verify_standard_input.currentData()),
            token,
            tolerance,
            self.app.calibration,
            self.app.calibration.revision,
        )
        self.app.worker.signals.run_request_finished.connect(self._verified)
        self.app.worker.bind_next_run(token)
        self.health_verification_label.setText("Measuring...")
        self.app.sweep_start()
        if not self.app.worker.isRunning():
            # Refused, for example because the automation service holds the
            # device.  An unused token must not attach to some later run.
            self.app.worker.bind_next_run(None)
            self._endVerification()
            self.refreshHealth()

    def _endVerification(self) -> None:
        self._verification_pending = None
        self.app.worker.signals.run_request_finished.disconnect(self._verified)

    def _verified(self, terminal, token) -> None:
        pending = self._verification_pending
        if pending is None or token is not pending[1]:
            return
        standard, _token, tolerance, calibration, revision = pending
        self._endVerification()
        if terminal is not SweepState.COMPLETED:
            self.health_verification_label.setText(
                f"The {standard} sweep did not complete, so nothing was"
                " verified."
            )
            return
        if (
            self.app.calibration is not calibration
            or calibration.revision != revision
        ):
            self.health_verification_label.setText(
                "The calibration changed during the sweep, so this result"
                " does not describe it. Verify again."
            )
            return
        capture = self.app.worker.capture
        snapshot = self.app._live_snapshot
        try:
            result = verify_standard(
                capture.calibration,
                standard,
                snapshot.raw.get("S11", ()),
                tolerance=tolerance,
                simulated=snapshot.simulated,
            )
        except VerificationError as exc:
            self.health_verification_label.setText(str(exc))
            return
        calibration.verifications[standard] = result
        self.refreshHealth()

    # -- calibration kit library ---------------------------------------
    def migrateLegacyStandards(self):
        """Show the saved "Calibration standards" entries as kits.

        The legacy ``QSettings`` array is read only: entries become
        in-memory kits so they can be applied and exported with the same
        code as any other kit.  The settings themselves are never written,
        moved or deleted, so an older build keeps seeing its standards.
        """

        self.kit_library.clear_transient()
        settings = self.app.settings
        count = settings.beginReadArray("CalibrationStandards")
        try:
            entries = []
            for index in range(count):
                settings.setArrayIndex(index)
                entries.append(
                    (
                        settings.value("Name", ""),
                        {
                            key: settings.value(key)
                            for key in (
                                "ShortL0",
                                "ShortL1",
                                "ShortL2",
                                "ShortL3",
                                "ShortDelay",
                                "OpenC0",
                                "OpenC1",
                                "OpenC2",
                                "OpenC3",
                                "OpenDelay",
                                "LoadR",
                                "LoadL",
                                "LoadC",
                                "LoadDelay",
                                "ThroughDelay",
                            )
                        },
                    )
                )
        finally:
            settings.endArray()

        for index, (name, values) in enumerate(entries):
            if not str(name).strip():
                logger.warning(
                    "Skipping saved calibration standard %d: it has no name",
                    index,
                )
                continue
            try:
                kit = kit_from_legacy_values(str(name), values)
            except CalibrationKitError as exc:
                logger.warning(
                    "Skipping saved calibration standard %r: %s", name, exc
                )
                continue
            self.kit_library.register_transient(kit)

    def reloadLegacyKits(self):
        """Re-read the saved standards after the legacy array changed."""

        self.migrateLegacyStandards()
        self.refreshKitSelector()

    def refreshKitSelector(self):
        """Rebuild the kit list, keeping the current choice when possible."""

        previous = self.cal_kit_selector.currentText()
        self.cal_kit_selector.blockSignals(True)
        try:
            self.cal_kit_selector.clear()
            for kit in self.kit_library.all_kits():
                self.cal_kit_selector.addItem(kit.name)
            index = self.cal_kit_selector.findText(previous)
            if index < 0:
                index = self.cal_kit_selector.findText(BUILTIN_NAMES[0])
            if index >= 0:
                self.cal_kit_selector.setCurrentIndex(index)
        finally:
            self.cal_kit_selector.blockSignals(False)
        self.kitComboSync()

    def kitComboSync(self):
        """Keep the detail label and the button states in step."""

        kit = self.currentKit()
        if self._kit_load_error:
            self.cal_kit_detail.setText(
                f"Calibration kit library unreadable: {self._kit_load_error}"
            )
        elif kit is None:
            self.cal_kit_detail.setText("No calibration kit selected")
        else:
            suffix = (
                "" if kit.source != "legacy-settings" else " (saved settings)"
            )
            self.cal_kit_detail.setText(f"{kit.describe()}{suffix}")
        builtin = kit is not None and kit.name in BUILTIN_NAMES
        editable = kit is not None and not builtin
        self.kitSaveButton.setEnabled(kit is not None)
        self.kitRenameButton.setEnabled(editable)
        self.kitDeleteButton.setEnabled(editable)
        self.kitApplyButton.setEnabled(kit is not None)

    def currentKit(self) -> CalibrationKit | None:
        name = self.cal_kit_selector.currentText()
        if not name:
            return None
        return self.kit_library.get(name)

    def kitSelected(self, _index: int):
        self.kitComboSync()

    def _kitButtonFor(self, kit: CalibrationKit):
        """The standard-source radio button that matches ``kit``."""
        models = {
            kit.standards[name].model for name in ("short", "open", "load")
        }
        if models == {MODEL_IDEAL}:
            return self.use_ideal_values
        if models == {MODEL_FILE}:
            return self.use_s1p_files
        return self.use_coefficients

    def applyCalibrationKit(self):
        """Install the selected kit and recalculate through ``calculate()``.

        The element built from the kit is handed to :meth:`calculate`, which
        keeps its exact stored coefficients instead of re-reading them from
        the decimal text fields.  Editing any coefficient field drops the
        pending element, so a hand-edited field always wins from then on.
        """

        kit = self.currentKit()
        if kit is None:
            return
        try:
            element = kit.apply_to(
                self.app.calibration,
                base_dir=os.path.dirname(self.kit_library.filename),
            )
        except CalibrationKitError as exc:
            self.app.showError(f"Cannot apply the calibration kit: {exc}")
            return
        self.app.calibration.metadata = replace(
            self.app.calibration.metadata,
            kit=kit.name,
            connector=" ".join(
                part
                for part in (kit.connector.type, kit.connector.sex)
                if part and part != "unknown"
            ),
        )
        self.populateStandardsFromKit(kit, element)
        button = self._kitButtonFor(kit)
        button.setChecked(True)
        self.calStandardChanged(button)
        self._pending_element = (
            element,
            {
                key: field.text()
                for key, field in self.coefficientFields.items()
            },
            button,
            kit.name,
            kit.calibration_identity(),
        )
        self.calculate()

    def populateStandardsFromKit(
        self, kit: CalibrationKit, element=None
    ) -> None:
        """Show the kit's coefficients in the standard entry fields."""

        if element is None:
            element = kit.to_cal_element(
                base_dir=os.path.dirname(self.kit_library.filename)
            )
        for legacy_key, value in legacy_values(kit).items():
            self.coefficientFields[legacy_key].setText(formatCoefficient(value))
        for standard_name in ("short", "open", "load"):
            setattr(
                self,
                f"{standard_name}_touchstone",
                getattr(element, f"{standard_name}_touchstone", None)
                if kit.standards[standard_name].model == MODEL_FILE
                else None,
            )

    def pendingCalElement(self):
        """The exact element an applied kit produced, if still valid.

        Returns ``None`` once any coefficient field or the standard-source
        radio button no longer matches what the kit produced, so a manual
        edit is never silently overridden.
        """

        pending = self._pending_element
        if pending is None:
            return None
        element, texts, button, _name, _identity = pending
        current = {
            key: field.text() for key, field in self.coefficientFields.items()
        }
        if current != texts or self.radio_group.checkedButton() is not button:
            self._pending_element = None
            return None
        return element

    def saveCalibrationKit(self):
        kit = self.currentKit()
        if kit is None:
            return
        default_name = (
            kit.name if kit.name not in BUILTIN_NAMES else f"{kit.name} copy"
        )
        name, accepted = QtWidgets.QInputDialog.getText(
            self, "Save calibration kit", "Kit name", text=default_name
        )
        if not accepted or not name.strip():
            return
        name = name.strip()
        if name in BUILTIN_NAMES:
            self.app.showError(
                f"{name!r} is a built-in kit name. Choose another name."
            )
            return
        replace = False
        if self.kit_library.get(name) is not None:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Replace calibration kit?",
                f"A calibration kit named {name!r} already exists. Replace it?",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            replace = True
        candidate = CalibrationKit.from_cal_element(
            self.app.calibration.cal_element,
            name=name,
            description=kit.description,
            connector=kit.connector,
            reference_impedance=kit.reference_impedance,
            valid_span=kit.valid_span,
            source="user",
            base_dir=os.path.dirname(self.kit_library.filename),
        )
        try:
            self.kit_library.add(candidate, replace=replace)
            self.kit_library.save()
        except (CalibrationKitError, OSError) as exc:
            self.app.showError(f"Could not save the calibration kit: {exc}")
            return
        self.refreshKitSelector()
        self.cal_kit_selector.setCurrentText(name)
        self.kitComboSync()

    def duplicateCalibrationKit(self):
        kit = self.currentKit()
        if kit is None:
            return
        name, accepted = QtWidgets.QInputDialog.getText(
            self,
            "Duplicate calibration kit",
            "New kit name",
            text=f"{kit.name} copy",
        )
        if not accepted or not name.strip():
            return
        try:
            copy = self.kit_library.duplicate(kit.name, name.strip())
            self.kit_library.save()
        except (CalibrationKitError, OSError) as exc:
            self.app.showError(
                f"Could not duplicate the calibration kit: {exc}"
            )
            return
        self.refreshKitSelector()
        self.cal_kit_selector.setCurrentText(copy.name)
        self.kitComboSync()

    def renameCalibrationKit(self):
        kit = self.currentKit()
        if kit is None:
            return
        name, accepted = QtWidgets.QInputDialog.getText(
            self, "Rename calibration kit", "New kit name", text=kit.name
        )
        if not accepted or not name.strip() or name.strip() == kit.name:
            return
        try:
            renamed = self.kit_library.rename(kit.name, name.strip())
            self.kit_library.save()
        except (CalibrationKitError, OSError) as exc:
            self.app.showError(f"Could not rename the calibration kit: {exc}")
            return
        if self.app.calibration.kit_name == kit.name:
            self.app.calibration.kit_name = renamed.name
        self.refreshKitSelector()
        self.cal_kit_selector.setCurrentText(renamed.name)
        self.kitComboSync()

    def deleteCalibrationKit(self):
        kit = self.currentKit()
        if kit is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete calibration kit?",
            f"Delete the calibration kit {kit.name!r}?",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            removed = self.kit_library.delete(kit.name)
            if removed:
                self.kit_library.save()
        except (CalibrationKitError, OSError) as exc:
            self.app.showError(f"Could not delete the calibration kit: {exc}")
            return
        self.refreshKitSelector()

    def importCalibrationKit(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import calibration kit",
            "",
            "Calibration kit files (*.json *.calkit);;All files (*.*)",
        )
        if not filename:
            return
        try:
            kits = self.kit_library.import_file(filename)
            self.kit_library.save()
        except (CalibrationKitError, OSError) as exc:
            self.app.showError(f"Could not import the calibration kit: {exc}")
            return
        self.refreshKitSelector()
        if kits:
            self.cal_kit_selector.setCurrentText(kits[0].name)
            self.kitComboSync()

    def exportCalibrationKit(self):
        kit = self.currentKit()
        if kit is None:
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export calibration kit",
            f"{kit.name}.json",
            "Calibration kit files (*.json *.calkit);;All files (*.*)",
        )
        if not filename:
            return
        try:
            self.kit_library.export_kit(kit.name, filename)
        except (CalibrationKitError, OSError) as exc:
            self.app.showError(f"Could not export the calibration kit: {exc}")

    def checkExpertUser(self):
        if not self.app.settings.value("ExpertCalibrationUser", False, bool):
            response = QtWidgets.QMessageBox.question(
                self,
                "Are you sure?",
                (
                    "Use of the manual calibration buttons is non-intuitive"
                    " and primarily suited for users with very specialized"
                    " needs. The buttons do not sweep for you nor do"
                    " they interact with the NanoVNA calibration.\n\n"
                    "If you are trying to do a calibration of the NanoVNA, do"
                    " so on the device itself instead. If you are trying to do"
                    ' a calibration with NanoVNA-Saver, use the "Calibration'
                    ' assistant" if possible.\n\n'
                    "If you are certain you know what you are doing, click"
                    " Yes."
                ),
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )

            if response == QtWidgets.QMessageBox.StandardButton.Yes:
                self.app.settings.setValue("ExpertCalibrationUser", True)
                return True
            return False
        return True

    def _standard_samples(self, channel: str) -> list[Datapoint]:
        """The connected standard as the device measured it.

        ``app.data`` holds the *published* view: it has been corrected and
        processed (offset delay, smoothing, running average, fixture,
        reference impedance), so storing it would make every later
        measurement inherit that processing and be corrected against data
        that was already transformed.  The acquisition keeps the unprocessed
        samples beside the published ones, so its channel is used whenever
        it is what produced the trace on screen.

        One correction does belong in the stored value.  The inline S21
        attenuator is part of the measurement configuration, not of the
        processing pipeline, and the acquisition undoes it before the
        calibration runs.  A transmission standard is therefore stored with
        that same factor applied to its raw samples, on the unattenuated
        footing the correction is computed on.

        Data that did not come from an acquisition -- a loaded file, or a
        trace whose published samples no longer match this acquisition --
        has no raw channel to recover, so it is stored as it is.
        """
        with self.app.dataLock:
            published = list(getattr(self.app.data, channel.lower()))
        att = self.app.s21att if channel == "S21" else 0.0
        snapshot = getattr(self.app, "_live_snapshot", None)
        if snapshot is None:
            return published
        raw = snapshot.channel(channel, corrected=False)
        displayed = snapshot.channel(channel)
        if not raw or not displayed:
            return published
        # ``saveData`` publishes the acquisition's corrected channel as it
        # stands; anything else means the trace is no longer this
        # acquisition.
        if len(displayed) != len(published) or any(
            now.freq != published_point.freq or now.z != published_point.z
            for now, published_point in zip(displayed, published, strict=True)
        ):
            return published
        return corr_att_data(list(raw), att)

    def cal_save(self, name: str):
        """Store the standard that is connected, as it was measured."""
        channel = "S21" if name in {"through", "isolation"} else "S11"
        data = self._standard_samples(channel)
        self.app.calibration.insert(name, data)
        self.cal_label[name].setText(_format_cal_label(len(data)))

    def manual_save(self, name: str):
        if self.checkExpertUser():
            self.cal_save(name)

    def listCalibrationStandards(self):
        self.cal_standard_save_selector.clear()
        num_standards = self.app.settings.beginReadArray("CalibrationStandards")
        for i in range(num_standards):
            self.app.settings.setArrayIndex(i)
            name = self.app.settings.value("Name", defaultValue="INVALID NAME")
            self.cal_standard_save_selector.addItem(name, userData=i)
        self.app.settings.endArray()
        self.cal_standard_save_selector.addItem("New", userData=-1)
        self.cal_standard_save_selector.setCurrentText("New")

    def saveCalibrationStandard(self):
        num_standards = self.app.settings.beginReadArray("CalibrationStandards")
        self.app.settings.endArray()

        if self.cal_standard_save_selector.currentData() == -1:
            # New cal standard
            # Get a name
            name, selected = QtWidgets.QInputDialog.getText(
                self, "Calibration standard name", "Enter name to save as"
            )
            if not selected or not name:
                return
            write_num = num_standards
            num_standards += 1
        else:
            write_num = self.cal_standard_save_selector.currentData()
            name = self.cal_standard_save_selector.currentText()

        self.app.settings.beginWriteArray("CalibrationStandards", num_standards)
        self.app.settings.setArrayIndex(write_num)
        self.app.settings.setValue("Name", name)

        self.app.settings.setValue("ShortL0", self.short_l0_input.text())
        self.app.settings.setValue("ShortL1", self.short_l1_input.text())
        self.app.settings.setValue("ShortL2", self.short_l2_input.text())
        self.app.settings.setValue("ShortL3", self.short_l3_input.text())
        self.app.settings.setValue("ShortDelay", self.short_length.text())

        self.app.settings.setValue("OpenC0", self.open_c0_input.text())
        self.app.settings.setValue("OpenC1", self.open_c1_input.text())
        self.app.settings.setValue("OpenC2", self.open_c2_input.text())
        self.app.settings.setValue("OpenC3", self.open_c3_input.text())
        self.app.settings.setValue("OpenDelay", self.open_length.text())

        self.app.settings.setValue("LoadR", self.load_resistance.text())
        self.app.settings.setValue("LoadL", self.load_inductance.text())
        self.app.settings.setValue("LoadC", self.load_capacitance.text())
        self.app.settings.setValue("LoadDelay", self.load_length.text())

        self.app.settings.setValue("ThroughDelay", self.through_length.text())

        self.app.settings.endArray()
        self.app.settings.sync()
        self.listCalibrationStandards()
        self.cal_standard_save_selector.setCurrentText(name)
        self.reloadLegacyKits()

    def loadCalibrationStandard(self):
        if self.cal_standard_save_selector.currentData() == -1:
            return
        read_num = self.cal_standard_save_selector.currentData()
        logger.debug("Loading calibration no %d", read_num)
        self.app.settings.beginReadArray("CalibrationStandards")
        self.app.settings.setArrayIndex(read_num)

        name = self.app.settings.value("Name")
        logger.info("Loading: %s", name)

        self.short_l0_input.setText(str(self.app.settings.value("ShortL0", 0)))
        self.short_l1_input.setText(str(self.app.settings.value("ShortL1", 0)))
        self.short_l2_input.setText(str(self.app.settings.value("ShortL2", 0)))
        self.short_l3_input.setText(str(self.app.settings.value("ShortL3", 0)))
        self.short_length.setText(str(self.app.settings.value("ShortDelay", 0)))

        self.open_c0_input.setText(str(self.app.settings.value("OpenC0", 50)))
        self.open_c1_input.setText(str(self.app.settings.value("OpenC1", 0)))
        self.open_c2_input.setText(str(self.app.settings.value("OpenC2", 0)))
        self.open_c3_input.setText(str(self.app.settings.value("OpenC3", 0)))
        self.open_length.setText(str(self.app.settings.value("OpenDelay", 0)))

        self.load_resistance.setText(str(self.app.settings.value("LoadR", 50)))
        self.load_inductance.setText(str(self.app.settings.value("LoadL", 0)))
        self.load_capacitance.setText(str(self.app.settings.value("LoadC", 0)))
        self.load_length.setText(str(self.app.settings.value("LoadDelay", 0)))

        self.through_length.setText(
            str(self.app.settings.value("ThroughDelay", 0))
        )

        self.app.settings.endArray()
        self.reloadLegacyKits()
        self.cal_kit_selector.setCurrentText(str(name))
        self.kitComboSync()

    def deleteCalibrationStandard(self):
        if self.cal_standard_save_selector.currentData() == -1:
            return
        delete_num = self.cal_standard_save_selector.currentData()
        logger.debug("Deleting calibration no %d", delete_num)
        num_standards = self.app.settings.beginReadArray("CalibrationStandards")
        self.app.settings.endArray()

        logger.debug("Number of standards known: %d", num_standards)

        if num_standards == 1:
            logger.debug("Only one standard known")
            self.app.settings.beginWriteArray("CalibrationStandards", 0)
            self.app.settings.endArray()
        else:
            names = []

            shortL0 = []
            shortL1 = []
            shortL2 = []
            shortL3 = []
            shortDelay = []

            openC0 = []
            openC1 = []
            openC2 = []
            openC3 = []
            openDelay = []

            loadR = []
            loadL = []
            loadC = []
            loadDelay = []

            throughDelay = []

            self.app.settings.beginReadArray("CalibrationStandards")
            for i in range(num_standards):
                if i == delete_num:
                    continue
                self.app.settings.setArrayIndex(i)
                names.append(self.app.settings.value("Name"))

                shortL0.append(self.app.settings.value("ShortL0"))
                shortL1.append(self.app.settings.value("ShortL1"))
                shortL2.append(self.app.settings.value("ShortL2"))
                shortL3.append(self.app.settings.value("ShortL3"))
                shortDelay.append(self.app.settings.value("ShortDelay"))

                openC0.append(self.app.settings.value("OpenC0"))
                openC1.append(self.app.settings.value("OpenC1"))
                openC2.append(self.app.settings.value("OpenC2"))
                openC3.append(self.app.settings.value("OpenC3"))
                openDelay.append(self.app.settings.value("OpenDelay"))

                loadR.append(self.app.settings.value("LoadR"))
                loadL.append(self.app.settings.value("LoadL"))
                loadC.append(self.app.settings.value("LoadC"))
                loadDelay.append(self.app.settings.value("LoadDelay"))

                throughDelay.append(self.app.settings.value("ThroughDelay"))
            self.app.settings.endArray()

            self.app.settings.beginWriteArray("CalibrationStandards")
            self.app.settings.remove("")
            self.app.settings.endArray()

            self.app.settings.beginWriteArray(
                "CalibrationStandards", len(names)
            )
            for i, name in enumerate(names):
                self.app.settings.setArrayIndex(i)
                self.app.settings.setValue("Name", name)

                self.app.settings.setValue("ShortL0", shortL0[i])
                self.app.settings.setValue("ShortL1", shortL1[i])
                self.app.settings.setValue("ShortL2", shortL2[i])
                self.app.settings.setValue("ShortL3", shortL3[i])
                self.app.settings.setValue("ShortDelay", shortDelay[i])

                self.app.settings.setValue("OpenC0", openC0[i])
                self.app.settings.setValue("OpenC1", openC1[i])
                self.app.settings.setValue("OpenC2", openC2[i])
                self.app.settings.setValue("OpenC3", openC3[i])
                self.app.settings.setValue("OpenDelay", openDelay[i])

                self.app.settings.setValue("LoadR", loadR[i])
                self.app.settings.setValue("LoadL", loadL[i])
                self.app.settings.setValue("LoadC", loadC[i])
                self.app.settings.setValue("LoadDelay", loadDelay[i])

                self.app.settings.setValue("ThroughDelay", throughDelay[i])
            self.app.settings.endArray()

        self.app.settings.sync()
        self.listCalibrationStandards()
        self.reloadLegacyKits()

    def reset(self):
        self.app.calibration = Calibration()
        for label in self.cal_label.values():
            label.setText("Uncalibrated")
        self.calibration_status_label.setText("Device calibration")
        self.calibration_source_label.setText("Device")
        self.notes_textedit.clear()
        self.refreshHealth()
        self.short_touchstone = None
        self.open_touchstone = None
        self.load_touchstone = None
        self._pending_element = None

        if len(self.app.worker.rawData11) > 0:
            # Republish what an uncalibrated sweep of the same raw data
            # would: the attenuator and offset delay are applied by
            # ``applyCalibration``, not by ``saveData``.
            logger.debug("Saving and displaying uncalibrated data.")
            (
                self.app.worker.data11,
                self.app.worker.data21,
            ) = self.app.worker.applyCalibration(
                self.app.worker.rawData11, self.app.worker.rawData21
            )
            self.app.saveData(
                self.app.worker.data11,
                self.app.worker.data21,
                self.app.sweepSource,
            )
            self.app.worker.signals.updated.emit()
        self.app.sweep_control.update_text()

    def setOffsetDelay(self, value: float):
        logger.debug("New offset delay value: %f ps", value)
        self.app.worker.offsetDelay = value / 1e12
        if len(self.app.worker.rawData11) > 0:
            # There's raw data, so we can get corrected data
            logger.debug("Applying new offset to existing sweep data.")
            (
                self.app.worker.data11,
                self.app.worker.data21,
            ) = self.app.worker.applyCalibration(
                self.app.worker.rawData11, self.app.worker.rawData21
            )
            logger.debug("Saving and displaying corrected data.")
            self.app.saveData(
                self.app.worker.data11,
                self.app.worker.data21,
                self.app.sweepSource,
            )
            self.app.worker.signals.updated.emit()

    def fillIdealStandards(self, cal_element) -> None:
        """Describe all four standards as ideal."""

        cal_element.short_state = "IDEAL"
        cal_element.open_state = "IDEAL"
        cal_element.load_state = "IDEAL"
        cal_element.through_is_ideal = True

    def fillCoefficientStandards(self, cal_element) -> None:
        """Read the coefficient entry fields into ``cal_element``.

        The divisors are the ones the fields are labelled with (e-12 H,
        e-15 F, ps) and are the only place where the displayed unit is
        applied.
        """

        cal_element.short_state = "COEFF"
        cal_element.open_state = "COEFF"
        cal_element.load_state = "COEFF"
        cal_element.through_is_ideal = False
        for standard, key, legacy_key, divisor, _default in COEFFICIENT_FIELDS:
            setattr(
                cal_element,
                f"{standard}_{key}",
                getFloatValue(self.coefficientFields[legacy_key].text())
                / divisor,
            )

    def fillFileStandards(self, cal_element) -> None:
        """Use the selected Touchstone files for the measured standards."""

        cal_element.short_state = "IDEAL"
        cal_element.open_state = "IDEAL"
        cal_element.load_state = "IDEAL"
        if self.short_touchstone is not None:
            cal_element.short_state = "FILE"
            cal_element.short_touchstone = self.short_touchstone
        if self.open_touchstone is not None:
            cal_element.open_state = "FILE"
            cal_element.open_touchstone = self.open_touchstone
        if self.load_touchstone is not None:
            cal_element.load_state = "FILE"
            cal_element.load_touchstone = self.load_touchstone
        cal_element.through_is_ideal = False
        cal_element.through_length = (
            getFloatValue(self.through_length.text()) / 1.0e12
        )

    def calculate(self):
        cal_element = self.app.calibration.cal_element
        if self.app.sweep_control.btn_stop.isEnabled():
            self.app.showError(
                "Unable to apply calibration while a sweep is running."
                " Please stop the sweep and try again."
            )
            return

        if not self.app.calibration.isValid1Port():
            self.app.showError(
                "Not enough data to apply calibration."
                " Please complete SOL calibration and try again."
            )
            return

        pending = self.pendingCalElement()
        if pending is not None:
            # An applied kit already describes the standards exactly; keep
            # its element rather than re-deriving the coefficients from the
            # text fields, which cannot represent every stored value.
            cal_element = pending
            self.app.calibration.cal_element = cal_element
        elif self.radio_group.checkedButton() == self.use_coefficients:
            self.fillCoefficientStandards(cal_element)
        elif self.radio_group.checkedButton() == self.use_s1p_files:
            self.fillFileStandards(cal_element)
        else:
            self.fillIdealStandards(cal_element)

        self.recordMetadata()
        logger.debug("Attempting calibration calculation.")
        try:
            self.app.calibration.calc_corrections()
            self.calibration_status_label.setText(
                _format_cal_label(
                    self.app.calibration.size(), "Application calibration"
                )
            )
            if self.use_ideal_values.isChecked():
                self.calibration_source_label.setText(
                    self.app.calibration.source
                )
            else:
                self.calibration_source_label.setText(
                    f"{self.app.calibration.source} (Standards: Custom)"
                )

            if self.app.worker.rawData11:
                # There's raw data, so we can get corrected data
                logger.debug("Applying calibration to existing sweep data.")
                (
                    self.app.worker.data11,
                    self.app.worker.data21,
                ) = self.app.worker.applyCalibration(
                    self.app.worker.rawData11, self.app.worker.rawData21
                )
                logger.debug("Saving and displaying corrected data.")
                self.app.saveData(
                    self.app.worker.data11,
                    self.app.worker.data21,
                    self.app.sweepSource,
                )
                self.app.worker.signals.updated.emit()

        except ValueError as e:
            # showError here hides the calibration window,
            # so we need to pop up our own
            self.calibration_status_label.setText(
                "Applying calibration failed."
            )
            self.calibration_source_label.setText(self.app.calibration.source)
            self.app.showError(
                f"{e} Please complete SOL calibration and try again."
            )
            self.reset()
            return
        self.refreshHealth()
        self.app.sweep_control.update_text()

    def loadCalibration(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            filter="Calibration Files (*.cal);;All files (*.*)"
        )
        if filename:
            self.app.calibration.load(filename)
        if not self.app.calibration.isValid1Port():
            return
        for i, name in enumerate(
            ("short", "open", "load", "through", "isolation", "thrurefl")
        ):
            self.cal_label[name].setText(
                _format_cal_label(
                    self.app.calibration.data_size(name), "Loaded"
                )
            )
            if i == 2 and not self.app.calibration.isValid2Port():
                break
        self.calculate()
        self.notes_textedit.clear()
        for note in self.app.calibration.notes:
            self.notes_textedit.appendPlainText(note)
        self.app.settings.setValue("CalibrationFile", filename)
        self.app.sweep_control.update_text()

    def saveCalibration(self):
        if not self.app.calibration.isCalculated:
            logger.debug("Attempted to save an uncalculated calibration.")
            self.app.showError("Cannot save an unapplied calibration state.")
            return
        filedialog = QtWidgets.QFileDialog(self)
        filedialog.setDefaultSuffix("cal")
        filedialog.setNameFilter("Calibration Files (*.cal);;All files (*.*)")
        filedialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        if filedialog.exec():
            filename = filedialog.selectedFiles()[0]
        else:
            return
        if not filename:
            logger.debug("No file name selected.")
            return
        self.app.calibration.notes = (
            self.notes_textedit.toPlainText().splitlines()
        )
        try:
            self.app.calibration.save(filename)
            self.app.settings.setValue("CalibrationFile", filename)
        except IOError:
            logger.error("Calibration save failed!")
            self.app.showError("Calibration save failed.")

    def calStandardChanged(self, button):
        if button == self.use_ideal_values:
            self.cal_short_box.setDisabled(True)
            self.cal_open_box.setDisabled(True)
            self.cal_load_box.setDisabled(True)
            self.cal_through_box.setDisabled(True)
            self.cal_standard_save_box.setDisabled(True)
            self.file_button_short.setDisabled(True)
            self.file_button_open.setDisabled(True)
            self.file_button_load.setDisabled(True)
        elif button == self.use_s1p_files:
            self.cal_short_box.setDisabled(True)
            self.cal_open_box.setDisabled(True)
            self.cal_load_box.setDisabled(True)
            self.cal_through_box.setDisabled(False)
            self.cal_standard_save_box.setDisabled(True)
            self.file_button_short.setDisabled(False)
            self.file_button_open.setDisabled(False)
            self.file_button_load.setDisabled(False)
        elif button == self.use_coefficients:
            self.cal_short_box.setDisabled(False)
            self.cal_open_box.setDisabled(False)
            self.cal_load_box.setDisabled(False)
            self.cal_through_box.setDisabled(False)
            self.cal_standard_save_box.setDisabled(False)
            self.file_button_short.setDisabled(True)
            self.file_button_open.setDisabled(True)
            self.file_button_load.setDisabled(True)

    def automaticCalibration(self):
        self.btn_automatic.setDisabled(True)
        introduction = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Icon.Information,
            "Calibration assistant",
            (
                "This calibration assistant will help you create a calibration"
                " in the NanoVNASaver application. It will sweep the"
                " standards for you and guide you through the process.<br><br>"
                "Before starting, ensure you have Open, Short and Load"
                " standards available and the cables you wish to have"
                " calibrated connected to the device.<br><br>"
                'If you want a 2-port calibration, also have a "through"'
                " connector on hand.<br><br>"
                "<b>The best results are achieved by having the NanoVNA"
                " calibrated on-device for the full span of interest and stored"
                " in save slot 0 before starting.</b><br><br>"
                "Once you are ready to proceed, press Ok."
            ),
            QtWidgets.QMessageBox.StandardButton.Ok
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        response = introduction.exec()
        if response != QtWidgets.QMessageBox.StandardButton.Ok:
            self.btn_automatic.setDisabled(False)
            return
        logger.info("Starting automatic calibration assistant.")
        if not self.app.vna.connected():
            QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Icon.Information,
                "NanoVNA not connected",
                (
                    "Please ensure the NanoVNA is connected before attempting"
                    " calibration."
                ),
            ).exec()
            self.btn_automatic.setDisabled(False)
            return

        if self.app.sweep.properties.mode == SweepMode.CONTINOUS:
            QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Icon.Information,
                "Continuous sweep enabled",
                (
                    "Please disable continuous sweeping before attempting"
                    " calibration."
                ),
            ).exec()
            self.btn_automatic.setDisabled(False)
            return

        short_step = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Icon.Information,
            "Calibrate short",
            (
                'Please connect the "short" standard to port 0 of the'
                " NanoVNA.\n\n"
                "Press Ok when you are ready to continue."
            ),
            QtWidgets.QMessageBox.StandardButton.Ok
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )

        response = short_step.exec()
        if response != QtWidgets.QMessageBox.StandardButton.Ok:
            self.btn_automatic.setDisabled(False)
            return
        self.reset()
        self.app.calibration.source = "Calibration assistant"
        self.next_step = 0
        self.app.worker.signals.finished.connect(self.automaticCalibrationStep)
        self.app.sweep_start()
        return

    def automaticCalibrationStep(self):
        if self.next_step == -1:
            self.app.worker.signals.finished.disconnect(
                self.automaticCalibrationStep
            )
            return

        if self.next_step == 0:
            # Short
            self.cal_save("short")
            self.next_step = 1

            open_step = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Icon.Information,
                "Calibrate open",
                (
                    'Please connect the "open" standard to port 0 of the'
                    " NanoVNA.\n\n"
                    "Either use a supplied open, or leave the end of the"
                    " cable unconnected if desired.\n\n"
                    "Press Ok when you are ready to continue."
                ),
                QtWidgets.QMessageBox.StandardButton.Ok
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )

            response = open_step.exec()
            if response != QtWidgets.QMessageBox.StandardButton.Ok:
                self.next_step = -1
                self.btn_automatic.setDisabled(False)
                self.app.worker.signals.finished.disconnect(
                    self.automaticCalibrationStep
                )
                return
            self.app.sweep_start()
            return

        if self.next_step == 1:
            # Open
            self.cal_save("open")
            self.next_step = 2
            load_step = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Icon.Information,
                "Calibrate load",
                (
                    'Please connect the "load" standard to port 0 of the'
                    " NanoVNA.\n\n"
                    "Press Ok when you are ready to continue."
                ),
                QtWidgets.QMessageBox.StandardButton.Ok
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )

            response = load_step.exec()
            if response != QtWidgets.QMessageBox.StandardButton.Ok:
                self.btn_automatic.setDisabled(False)
                self.next_step = -1
                self.app.worker.signals.finished.disconnect(
                    self.automaticCalibrationStep
                )
                return
            self.app.sweep_start()
            return

        if self.next_step == 2:
            # Load
            self.cal_save("load")
            self.next_step = 3
            continue_step = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Icon.Information,
                "1-port calibration complete",
                (
                    "The required steps for a 1-port calibration are now"
                    " complete.\n\n"
                    "If you wish to continue and perform a 2-port calibration,"
                    ' press "Yes". To apply the 1-port calibration and stop,'
                    ' press "Apply"'
                ),
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Apply
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )

            response = continue_step.exec()
            if response == QtWidgets.QMessageBox.StandardButton.Apply:
                self.calculate()
                self.next_step = -1
                self.app.worker.signals.finished.disconnect(
                    self.automaticCalibrationStep
                )
                self.btn_automatic.setDisabled(False)
                return
            if response != QtWidgets.QMessageBox.StandardButton.Yes:
                self.btn_automatic.setDisabled(False)
                self.next_step = -1
                self.app.worker.signals.finished.disconnect(
                    self.automaticCalibrationStep
                )
                return

            isolation_step = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Icon.Information,
                "Calibrate isolation",
                (
                    'Please connect the "load" standard to port 1 of the'
                    " NanoVNA.\n\n"
                    "If available, also connect a load standard to"
                    " port 0.\n\n"
                    "Press Ok when you are ready to continue."
                ),
                QtWidgets.QMessageBox.StandardButton.Ok
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )

            response = isolation_step.exec()
            if response != QtWidgets.QMessageBox.StandardButton.Ok:
                self.btn_automatic.setDisabled(False)
                self.next_step = -1
                self.app.worker.signals.finished.disconnect(
                    self.automaticCalibrationStep
                )
                return
            self.app.sweep_start()
            return

        if self.next_step == 3:
            # Isolation
            self.cal_save("isolation")
            self.next_step = 4
            through_step = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Icon.Information,
                "Calibrate through",
                (
                    'Please connect the "through" standard between'
                    " port 0 and port 1 of the NanoVNA.\n\n"
                    "Press Ok when you are ready to continue."
                ),
                QtWidgets.QMessageBox.StandardButton.Ok
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )

            response = through_step.exec()
            if response != QtWidgets.QMessageBox.StandardButton.Ok:
                self.btn_automatic.setDisabled(False)
                self.next_step = -1
                self.app.worker.signals.finished.disconnect(
                    self.automaticCalibrationStep
                )
                return
            self.app.sweep_start()
            return

        if self.next_step == 4:
            # Done
            self.cal_save("thrurefl")
            self.cal_save("through")
            apply_step = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Icon.Information,
                "Calibrate complete",
                (
                    "The calibration process is now complete. Press"
                    ' "Apply" to apply the calibration parameters.'
                ),
                QtWidgets.QMessageBox.StandardButton.Apply
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )

            response = apply_step.exec()
            if response != QtWidgets.QMessageBox.StandardButton.Apply:
                self.btn_automatic.setDisabled(False)
                self.next_step = -1
                self.app.worker.signals.finished.disconnect(
                    self.automaticCalibrationStep
                )
                return

            self.calculate()
            self.btn_automatic.setDisabled(False)
            self.next_step = -1
            self.app.worker.signals.finished.disconnect(
                self.automaticCalibrationStep
            )
            return

    def select_file_open(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Open S1P", "", "Touchstone Files (*.s1p)"
        )
        if filename != "":
            self.open_touchstone = Touchstone(filename)
            self.open_touchstone.load()

    def select_file_short(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Short S1P", "", "Touchstone Files (*.s1p)"
        )
        if filename != "":
            self.short_touchstone = Touchstone(filename)
            self.short_touchstone.load()

    def select_file_load(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Load S1P", "", "Touchstone Files (*.s1p)"
        )
        if filename != "":
            self.load_touchstone = Touchstone(filename)
            self.load_touchstone.load()
