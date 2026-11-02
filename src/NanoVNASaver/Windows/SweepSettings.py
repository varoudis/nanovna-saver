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
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from ..Fixture import (
    DEFAULT_FIXTURE_Z0,
    Fixture,
    fixture_from_stage,
    fixture_from_touchstone,
    fixture_stage,
)
from ..Formatting import (
    format_frequency_short,
    format_frequency_sweep,
)
from ..PortExtension import StandardKind, fit_port_extension
from ..Processing import (
    STAGE_AVERAGE,
    STAGE_FIXTURE,
    STAGE_PORT_EXTENSION,
    STAGE_SMOOTHING,
    ProcessingConfig,
    ProcessingError,
    average_stage,
    port_extension_stage,
    smoothing_aperture_for,
    smoothing_stage,
)
from ..Settings.Sweep import SweepMode
from ..Touchstone import Touchstone
from .Defaults import make_scrollable
from .ui import get_window_icon

if TYPE_CHECKING:
    from ..NanoVNASaver.NanoVNASaver import NanoVNASaver as vna_app

logger = logging.getLogger(__name__)

#: Channels a general de-embedding needs.  Named once so the preview, the
#: refusal and the report cannot drift apart about how many ports are
#: required: removing a two-port is a matrix operation, not two reflections.
FIXTURE_CHANNELS: tuple[str, ...] = ("S11", "S21", "S12", "S22")


class FixturePreviewStatus(Enum):
    READY = "ready"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class FixturePreview:
    """What a loaded fixture would do to the sweep, before it is installed.

    ``UNAVAILABLE`` means no stage must be installed: a stage that fails on
    every sweep is worse than no stage at all, because it records a
    correction that never ran.
    """

    status: FixturePreviewStatus
    fixture_span: tuple[int, int] | None = None
    sweep_span: tuple[int, int] | None = None
    missing_channels: tuple[str, ...] = ()
    #: First swept frequency the fixture has no sample at, if any.
    absent_frequency: int | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is FixturePreviewStatus.READY

    def describe(self) -> str:
        if not self.ok:
            return self.reason or "unavailable"
        start, stop = self.sweep_span or (0, 0)
        return (
            f"covers {format_frequency_short(start)} to"
            f" {format_frequency_short(stop)};"
            f" {', '.join(FIXTURE_CHANNELS)} complete on the fixture grid"
        )


def preview_fixture(  # noqa: PLR0913 - one flat call, no config object to thread
    fixture: "Fixture | None",
    *,
    sweep_start: int,
    sweep_stop: int,
    sweep_frequencies: Iterable[int] | None = None,
    channels: Mapping[str, Sequence] | None = None,
    reference_impedance: float = DEFAULT_FIXTURE_Z0,
) -> FixturePreview:
    """Report whether ``fixture`` can be removed from the current sweep.

    The questions asked here are exactly the ones
    :func:`~NanoVNASaver.Fixture.remove_fixture` asks at run time — matching
    reference impedance, a complete two-port, one shared grid, a fixture
    sample at every swept frequency, and coverage of the span — so the panel
    cannot call a fixture installable and then fail on every sweep.  A "no"
    is returned with its reason; nothing is interpolated and nothing is
    assumed, because de-embedding on a grid the fixture was not measured on
    is a different operation with its own error.
    """
    span = (int(sweep_start), int(sweep_stop))
    if fixture is None:
        return FixturePreview(
            status=FixturePreviewStatus.UNAVAILABLE,
            sweep_span=span,
            reason="no fixture is loaded",
        )
    fixture_span = (fixture.start, fixture.stop)
    if abs(reference_impedance - fixture.reference_impedance) > 1e-9:
        return FixturePreview(
            status=FixturePreviewStatus.UNAVAILABLE,
            fixture_span=fixture_span,
            sweep_span=span,
            reason=(
                "the fixture reference impedance"
                f" ({fixture.reference_impedance:g} ohm) does not match the"
                f" measurement ({reference_impedance:g} ohm)"
            ),
        )
    if fixture_span[0] > span[0] or fixture_span[1] < span[1]:
        return FixturePreview(
            status=FixturePreviewStatus.UNAVAILABLE,
            fixture_span=fixture_span,
            sweep_span=span,
            reason=(
                f"the fixture covers"
                f" {format_frequency_short(fixture_span[0])} to"
                f" {format_frequency_short(fixture_span[1])} but the sweep"
                f" runs {format_frequency_short(span[0])} to"
                f" {format_frequency_short(span[1])}"
            ),
        )
    measured = channels or {}
    missing = tuple(name for name in FIXTURE_CHANNELS if not measured.get(name))
    if missing:
        present = [name for name in FIXTURE_CHANNELS if measured.get(name)]
        return FixturePreview(
            status=FixturePreviewStatus.UNAVAILABLE,
            fixture_span=fixture_span,
            sweep_span=span,
            missing_channels=missing,
            reason=(
                "de-embedding needs all four S-parameters"
                f" ({', '.join(FIXTURE_CHANNELS)}); this measurement has"
                f" {'/'.join(present) if present else 'no samples'}"
            ),
        )
    grids = {
        name: tuple(point.freq for point in measured[name])
        for name in FIXTURE_CHANNELS
    }
    first = grids["S11"]
    for name, grid in grids.items():
        if grid != first:
            return FixturePreview(
                status=FixturePreviewStatus.UNAVAILABLE,
                fixture_span=fixture_span,
                sweep_span=span,
                reason=f"the {name} channel is not on the same grid as S11",
            )
    # A sweep that is asked about its own grid must be measured on it: a
    # stage installed over a stale copy of some other sweep would de-embed
    # the wrong samples.
    frequencies = tuple(int(freq) for freq in (sweep_frequencies or ()))
    if frequencies and first and first != frequencies:
        offline = [freq for freq in frequencies if freq not in set(first)]
        where = (
            format_frequency_short(offline[0])
            if offline
            else format_frequency_short(first[0])
        )
        return FixturePreview(
            status=FixturePreviewStatus.UNAVAILABLE,
            fixture_span=fixture_span,
            sweep_span=span,
            reason=(
                "the measurement is not on this sweep's grid"
                f" (first difference at {where}); sweep again before"
                " de-embedding"
            ),
        )
    fixture_frequencies = {point.freq for point in fixture.points}
    absent = [freq for freq in first if freq not in fixture_frequencies]
    if absent:
        return FixturePreview(
            status=FixturePreviewStatus.UNAVAILABLE,
            fixture_span=fixture_span,
            sweep_span=span,
            absent_frequency=absent[0],
            reason=(
                f"the fixture has no sample at"
                f" {format_frequency_short(absent[0])}: it was not measured"
                " on this sweep's grid"
            ),
        )
    return FixturePreview(
        status=FixturePreviewStatus.READY,
        fixture_span=fixture_span,
        sweep_span=span,
    )


class SweepSettingsWindow(QtWidgets.QWidget):
    def __init__(self, app: "vna_app"):
        super().__init__()
        self.app = app
        self.padding = 0

        self.setWindowTitle("Sweep settings")
        self.setWindowIcon(get_window_icon())

        QtGui.QShortcut(QtCore.Qt.Key.Key_Escape, self, self.hide)

        layout = QtWidgets.QVBoxLayout()
        make_scrollable(self, layout)

        layout.addWidget(self.title_box())
        layout.addWidget(self.settings_box())
        # We can only populate this box after the VNA has been connected.
        self._power_box = QtWidgets.QGroupBox("Power")
        self._power_layout = QtWidgets.QFormLayout(self._power_box)
        layout.addWidget(self._power_box)
        layout.addWidget(self.sweep_box())
        layout.addWidget(self.processing_box())
        self.update_band()

    def showEvent(self, event: "QtGui.QShowEvent") -> None:
        """Re-read the live state: sweep and pipeline can both change."""
        super().showEvent(event)
        self._refresh_processing_labels()

    def title_box(self):
        box = QtWidgets.QGroupBox("Sweep name")
        layout = QtWidgets.QFormLayout(box)

        input_title = QtWidgets.QLineEdit(self.app.sweep.properties.name)
        input_title.setMinimumHeight(20)
        input_title.editingFinished.connect(
            lambda: self.update_title(input_title.text())
        )
        layout.addRow(input_title)
        return box

    def settings_box(self) -> "QtWidgets.QWidget":
        box = QtWidgets.QGroupBox("Settings")
        layout = QtWidgets.QFormLayout(box)

        # Sweep Mode
        sweep_btn_layout = QtWidgets.QHBoxLayout()

        radio_button = QtWidgets.QRadioButton("Single sweep")
        radio_button.setMinimumHeight(20)
        radio_button.setChecked(
            self.app.sweep.properties.mode == SweepMode.SINGLE
        )
        radio_button.clicked.connect(lambda: self.update_mode(SweepMode.SINGLE))
        sweep_btn_layout.addWidget(radio_button)

        radio_button = QtWidgets.QRadioButton("Continous sweep")
        radio_button.setMinimumHeight(20)
        radio_button.setChecked(
            self.app.sweep.properties.mode == SweepMode.CONTINOUS
        )
        radio_button.clicked.connect(
            lambda: self.update_mode(SweepMode.CONTINOUS)
        )
        sweep_btn_layout.addWidget(radio_button)

        radio_button = QtWidgets.QRadioButton("Averaged sweep")
        radio_button.setMinimumHeight(20)
        radio_button.setChecked(
            self.app.sweep.properties.mode == SweepMode.AVERAGE
        )
        radio_button.clicked.connect(
            lambda: self.update_mode(SweepMode.AVERAGE)
        )
        sweep_btn_layout.addWidget(radio_button)

        layout.addRow(sweep_btn_layout)

        # Log sweep
        label = QtWidgets.QLabel(
            "Logarithmic sweeping changes the step width in each segment"
            " in logarithmical manner. Useful in conjunction with small"
            " amount of datapoints and many segments. Step display in"
            " SweepControl cannot reflect this currently."
        )
        label.setWordWrap(True)
        label.setMinimumSize(600, 70)
        layout.addRow(label)
        checkbox = QtWidgets.QCheckBox("Logarithmic sweep")
        checkbox.setMinimumHeight(20)
        checkbox.setCheckState(
            Qt.CheckState.Checked
            if self.app.sweep.properties.logarithmic
            else Qt.CheckState.Unchecked
        )
        checkbox.toggled.connect(
            lambda: self.update_logarithmic(checkbox.isChecked())
        )
        layout.addRow(checkbox)

        # Averaging
        label = QtWidgets.QLabel(
            "Averaging allows discarding outlying samples to get better"
            " averages. Common values are 3/0, 5/2, 9/4 and 25/6."
        )
        label.setWordWrap(True)
        label.setMinimumHeight(50)
        layout.addRow(label)
        averages = QtWidgets.QLineEdit(
            str(self.app.sweep.properties.averages[0])
        )
        averages.setMinimumHeight(20)
        truncates = QtWidgets.QLineEdit(
            str(self.app.sweep.properties.averages[1])
        )
        truncates.setMinimumHeight(20)
        averages.editingFinished.connect(
            lambda: self.update_averaging(averages, truncates)
        )
        truncates.editingFinished.connect(
            lambda: self.update_averaging(averages, truncates)
        )
        layout.addRow("Number of measurements to average", averages)
        layout.addRow("Number to discard", truncates)

        # TODO: is this more a device than a sweep property?
        label = QtWidgets.QLabel(
            "Some times when you measure amplifiers you need to use an"
            " attenuator in line with  the S21 input (CH1) here you can"
            " specify it."
        )
        label.setWordWrap(True)
        label.setMinimumHeight(50)
        layout.addRow(label)

        input_att = QtWidgets.QLineEdit(str(self.app.s21att))
        input_att.setMinimumHeight(20)
        input_att.editingFinished.connect(
            lambda: self.update_attenuator(input_att)
        )
        layout.addRow("Attenuator in port CH1 (s21) in dB", input_att)
        return box

    def sweep_box(self) -> "QtWidgets.QWidget":
        box = QtWidgets.QGroupBox("Sweep band")
        layout = QtWidgets.QFormLayout(box)
        sweep_pad_layout = QtWidgets.QHBoxLayout()

        self.band_list = QtWidgets.QComboBox()
        self.band_list.setMinimumHeight(20)
        self.band_list.setModel(self.app.bands)
        # pylint: disable=unnecessary-lambda
        self.band_list.currentIndexChanged.connect(lambda: self.update_band())
        layout.addRow("Select band", self.band_list)

        sweep_pad_layout.addWidget(QtWidgets.QLabel("Pad band limits:"))
        for btn_label, value in (
            ("None", 0),
            ("10%", 10),
            ("25%", 25),
            ("100%", 100),
        ):
            radio_button = QtWidgets.QRadioButton(btn_label)
            radio_button.setMinimumHeight(20)
            radio_button.setChecked(self.padding == value)
            radio_button.clicked.connect(partial(self.update_padding, value))
            sweep_pad_layout.addWidget(radio_button)

        layout.addRow(sweep_pad_layout)
        self.band_label = QtWidgets.QLabel()
        layout.addRow(self.band_label)

        btn_set_band_sweep = QtWidgets.QPushButton("Set band sweep")
        btn_set_band_sweep.setMinimumHeight(20)
        btn_set_band_sweep.clicked.connect(lambda: self.update_band(True))
        layout.addRow(btn_set_band_sweep)
        return box

    def vna_connected(self):
        while self._power_layout.rowCount():
            self._power_layout.removeRow(0)
        for freq_range, power_descs in self.app.vna.txPowerRanges:
            power_sel = QtWidgets.QComboBox()
            power_sel.addItems(power_descs)
            power_sel.currentTextChanged.connect(
                partial(self.update_tx_power, freq_range)
            )
            self._power_layout.addRow(
                f"TX power {format_frequency_short}..{freq_range}",
                power_sel,
            )

    def update_band(self, apply: bool = False):
        logger.debug("update_band(%s)", apply)
        index_start = self.band_list.model().index(
            self.band_list.currentIndex(), 1
        )
        index_stop = self.band_list.model().index(
            self.band_list.currentIndex(), 2
        )
        start = int(
            self.band_list.model().data(index_start, Qt.ItemDataRole.EditRole)
        )
        stop = int(
            self.band_list.model().data(index_stop, Qt.ItemDataRole.EditRole)
        )

        if self.padding > 0:
            span = stop - start
            start -= round(span * self.padding / 100)
            start = max(1, start)
            stop += round(span * self.padding / 100)

        self.band_label.setText(
            f"Sweep span: {format_frequency_short(start)}"
            f" to {format_frequency_short(stop)}"
        )

        if not apply:
            return

        self.app.sweep_control.inputs["Start"].setText(
            format_frequency_sweep(start)
        )
        self.app.sweep_control.inputs["Stop"].setText(
            format_frequency_sweep(stop)
        )
        self.app.sweep_control.inputs["Stop"].textEdited.emit(
            self.app.sweep_control.inputs["Stop"].text()
        )

    def update_attenuator(self, value: "QtWidgets.QLineEdit"):
        try:
            att = float(value.text())
            assert att >= 0
        except (ValueError, AssertionError):
            logger.warning(
                "Values for attenuator are absolute and with no"
                " minus sign, resetting."
            )
            att = 0
        logger.debug("Attenuator %sdB inline with S21 input", att)
        value.setText(str(att))
        self.app.s21att = att

    def update_averaging(
        self, averages: "QtWidgets.QLineEdit", truncs: "QtWidgets.QLineEdit"
    ):
        try:
            amount = int(averages.text())
            truncates = int(truncs.text())
            assert amount > 0
            assert truncates >= 0
            assert amount > truncates
        except (AssertionError, ValueError):
            logger.warning("Illegal averaging values, set default")
            amount = 3
            truncates = 0
        logger.debug("update_averaging(%s, %s)", amount, truncates)
        averages.setText(str(amount))
        truncs.setText(str(truncates))
        self.app.sweep.set_averages(amount, truncates)

    def update_logarithmic(self, logarithmic: bool):
        logger.debug("update_logarithmic(%s)", logarithmic)
        self.app.sweep.set_logarithmic(logarithmic)

    def update_mode(self, mode: "SweepMode"):
        logger.debug("update_mode(%s)", mode)
        self.app.sweep.set_mode(mode)

    def update_padding(self, padding: int):
        logger.debug("update_padding(%s)", padding)
        self.padding = padding
        self.update_band()

    def update_title(self, title: str = ""):
        logger.debug("update_title(%s)", title)
        self.app.sweep.set_name(title)
        self.app.update_sweep_title()

    def update_tx_power(self, freq_range, power_desc):
        logger.debug("update_tx_power(%r)", power_desc)
        self.app.vna.setTXPower(freq_range, power_desc)

    # ------------------------------------------------------------------
    # Processing (DSP-02/DSP-03): opt-in, shown with its effective settings
    # ------------------------------------------------------------------
    def processing_box(self) -> "QtWidgets.QWidget":
        """Controls for the ordered post-calibration processing stages.

        Both stages are off by default and stay off until the operator asks
        for them, so an ordinary sweep is unchanged.  The aperture and the
        average mode are described in the same words the reports use, because
        a processed trace must never look like a raw measurement.
        """
        box = QtWidgets.QGroupBox("Processing (applied after calibration)")
        layout = QtWidgets.QFormLayout(box)

        self.check_smoothing = QtWidgets.QCheckBox("Smooth frequency data")
        self.check_smoothing.setToolTip(
            "Averages neighbouring complex samples. This changes narrow"
            " features; it does not improve accuracy."
        )
        config = self.app.processing_config
        smoothing = config.find(STAGE_SMOOTHING)
        self.check_smoothing.setChecked(smoothing is not None)
        self.check_smoothing.toggled.connect(self.update_smoothing)
        layout.addRow(self.check_smoothing)

        self.spin_aperture_percent = QtWidgets.QSpinBox()
        self.spin_aperture_percent.setRange(0, 50)
        self.spin_aperture_percent.setSuffix(" % of span")
        self.spin_aperture_percent.setToolTip(
            "Aperture as a fraction of the sweep. The effective point count"
            " is always odd and never exceeds the number of points."
        )
        if smoothing is not None:
            points = int(smoothing.as_dict.get("aperture_points", 1) or 1)
            total = max(1, self.app.sweep.points)
            self.spin_aperture_percent.setValue(
                round(100.0 * (points - 1) / max(total - 1, 1))
            )
        self.spin_aperture_percent.valueChanged.connect(self.update_aperture)
        layout.addRow("Aperture", self.spin_aperture_percent)

        self.label_aperture = QtWidgets.QLabel("")
        self.label_aperture.setWordWrap(True)
        layout.addRow("Effective", self.label_aperture)

        self.check_average = QtWidgets.QCheckBox("Continuous running average")
        self.check_average.setToolTip(
            "Blends successive complete sweeps. Resets when the grid,"
            " calibration or source changes."
        )
        self.check_average.setChecked(config.find(STAGE_AVERAGE) is not None)
        self.check_average.toggled.connect(self.update_average)
        layout.addRow(self.check_average)

        self.combo_average_mode = QtWidgets.QComboBox()
        self.combo_average_mode.addItem(
            "Cumulative (equal weight)", "cumulative"
        )
        self.combo_average_mode.addItem("Exponential", "exponential")
        stage = config.find(STAGE_AVERAGE)
        if stage is not None:
            mode = str(stage.as_dict.get("mode", "cumulative"))
            index = self.combo_average_mode.findData(mode)
            self.combo_average_mode.setCurrentIndex(max(index, 0))
        self.combo_average_mode.currentIndexChanged.connect(self.update_average)
        layout.addRow("Average mode", self.combo_average_mode)

        self.spin_average_alpha = QtWidgets.QDoubleSpinBox()
        self.spin_average_alpha.setRange(0.01, 1.0)
        self.spin_average_alpha.setSingleStep(0.05)
        self.spin_average_alpha.setDecimals(2)
        if stage is not None and stage.as_dict.get("alpha"):
            self.spin_average_alpha.setValue(float(stage.as_dict["alpha"]))
        else:
            self.spin_average_alpha.setValue(0.25)
        self.spin_average_alpha.valueChanged.connect(self.update_average)
        layout.addRow("Exponential alpha", self.spin_average_alpha)

        # -- Port extension (RF-01) ----------------------------------
        extension = QtWidgets.QGroupBox("Port extension (port 1 fixture)")
        extension_layout = QtWidgets.QFormLayout(extension)
        self.spin_port1_delay = QtWidgets.QDoubleSpinBox()
        self.spin_port1_delay.setRange(0.0, 100_000.0)
        self.spin_port1_delay.setDecimals(3)
        self.spin_port1_delay.setSuffix(" ps")
        self.spin_port1_delay.setToolTip(
            "One-way delay between the reference plane and the device."
        )
        self.spin_port1_loss = QtWidgets.QDoubleSpinBox()
        self.spin_port1_loss.setRange(0.0, 100.0)
        self.spin_port1_loss.setDecimals(3)
        self.spin_port1_loss.setSuffix(" dB/GHz")
        self.spin_port1_loss.setToolTip(
            "One-way loss of the extension per gigahertz."
        )
        current = config.find(STAGE_PORT_EXTENSION)
        if current is not None:
            values = current.as_dict
            self.spin_port1_delay.setValue(
                float(values.get("port1_delay_ps", 0.0) or 0.0)
            )
            self.spin_port1_loss.setValue(
                float(values.get("port1_loss_db_per_ghz", 0.0) or 0.0)
            )
        self.spin_port1_delay.valueChanged.connect(self.update_port_extension)
        self.spin_port1_loss.valueChanged.connect(self.update_port_extension)
        extension_layout.addRow("Delay", self.spin_port1_delay)
        extension_layout.addRow("Loss", self.spin_port1_loss)
        fit_row = QtWidgets.QHBoxLayout()
        self.combo_fit_standard = QtWidgets.QComboBox()
        self.combo_fit_standard.addItem("Open", "open")
        self.combo_fit_standard.addItem("Short", "short")
        self.combo_fit_standard.setToolTip(
            "Measure this standard through the fixture, then press Fit."
        )
        btn_fit = QtWidgets.QPushButton("Fit from sweep")
        btn_fit.clicked.connect(self.fit_port_extension_from_standard)
        fit_row.addWidget(self.combo_fit_standard)
        fit_row.addWidget(btn_fit)
        extension_layout.addRow(fit_row)
        self.label_extension = QtWidgets.QLabel("")
        self.label_extension.setWordWrap(True)
        extension_layout.addRow("Fit", self.label_extension)
        layout.addRow(extension)

        # -- Two-port fixture de-embedding (RF-03) -------------------
        layout.addRow(self._fixture_box())

        self.label_processing = QtWidgets.QLabel("")
        self.label_processing.setWordWrap(True)
        layout.addRow("Active", self.label_processing)

        self.label_processing_message = QtWidgets.QLabel("")
        self.label_processing_message.setWordWrap(True)
        layout.addRow(self.label_processing_message)
        self._refresh_processing_labels()
        return box

    # -- Two-port fixture (RF-03) --------------------------------------
    def _fixture_box(self) -> "QtWidgets.QWidget":
        """Load, describe and clear a two-port fixture (RF-03).

        The group only *shows* a fixture and asks for one to be installed;
        the stage itself is created by :func:`fixture_stage` and stored
        through :meth:`_set_processing`, so it cannot bypass the ordered
        pipeline that every sweep reads.
        """
        box = QtWidgets.QGroupBox("Fixture (two-port de-embedding)")
        fixture_layout = QtWidgets.QFormLayout(box)
        self._fixture: Fixture | None = None
        #: Why the last file could not be read as a fixture, verbatim.  It is
        #: about the *file*, so clearing a stage never erases it.
        self._fixture_error = ""
        #: Why the stage in the pipeline cannot be rebuilt, verbatim.
        self._stage_error = ""
        #: Identity of the stage this panel last saw in the pipeline, so a
        #: change made elsewhere is noticed and adopted exactly once.
        self._installed_identity: str | None = None

        buttons = QtWidgets.QHBoxLayout()
        self.btn_load_fixture = QtWidgets.QPushButton("Load touchstone file")
        self.btn_load_fixture.setToolTip(
            "A 2-port .s2p file of the fixture alone, on the sweep's grid."
        )
        self.btn_load_fixture.clicked.connect(self.load_fixture)
        self.btn_preview_fixture = QtWidgets.QPushButton("Preview")
        self.btn_preview_fixture.setToolTip(
            "Check the span and the channel set before installing anything."
        )
        self.btn_preview_fixture.clicked.connect(self.preview_current_fixture)
        self.btn_clear_fixture = QtWidgets.QPushButton("Clear fixture")
        self.btn_clear_fixture.clicked.connect(self.clear_fixture)
        buttons.addWidget(self.btn_load_fixture)
        buttons.addWidget(self.btn_preview_fixture)
        buttons.addWidget(self.btn_clear_fixture)
        fixture_layout.addRow(buttons)

        self.label_fixture = QtWidgets.QLabel("")
        self.label_fixture.setWordWrap(True)
        #: Read-only, selectable: the summary is what a report quotes, so it
        #: must not be editable into a state the stage does not carry.
        self.label_fixture.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        fixture_layout.addRow("Loaded", self.label_fixture)

        self.label_fixture_preview = QtWidgets.QLabel("")
        self.label_fixture_preview.setWordWrap(True)
        fixture_layout.addRow("Preview", self.label_fixture_preview)

        self.btn_install_fixture = QtWidgets.QPushButton("Install as stage")
        self.btn_install_fixture.setToolTip(
            "Add the fixture to the processing pipeline. It is applied"
            " after port extension and before smoothing."
        )
        self.btn_install_fixture.clicked.connect(self.install_fixture)
        fixture_layout.addRow(self.btn_install_fixture)

        self._load_installed_fixture()
        return box

    def _load_installed_fixture(self) -> None:
        """Adopt the fixture already in the pipeline, if there is one."""
        self._sync_fixture_with_config()
        self._refresh_fixture_labels()

    def _sync_fixture_with_config(self) -> None:
        """Adopt a fixture that reached the pipeline from another source.

        A session, a preset or another window can install or remove a
        fixture stage while this panel is open.  Adopting it keeps the
        summary truthful.  A stage that was *removed* elsewhere leaves the
        loaded file alone: the operator may still want to install it.
        """
        installed = self.app.processing_config.find(STAGE_FIXTURE)
        identity = (
            ""
            if installed is None
            else str(installed.as_dict.get("identity", "") or "")
        )
        if identity == self._installed_identity:
            return
        self._installed_identity = identity
        if installed is None:
            # A removal elsewhere says nothing about the loaded file, but it
            # does retire any complaint about the stage that is now gone.
            self._stage_error = ""
            return
        try:
            self._fixture = fixture_from_stage(installed)
        except ProcessingError as exc:
            # A stage that cannot be read is reported, not silently dropped:
            # the pipeline still holds it and will refuse every sweep.
            self._fixture = None
            self._stage_error = str(exc)
            return
        # The adopted fixture supersedes any complaint about an earlier file.
        self._stage_error = ""
        self._fixture_error = ""

    def fixture_summary(self) -> str:
        """The loaded fixture in one line, or why there is nothing to show."""
        if self._fixture_error:
            return f"unavailable ({self._fixture_error})"
        fixture = self._fixture
        if fixture is None:
            if self._stage_error:
                return (
                    "unavailable (the installed fixture cannot be read:"
                    f" {self._stage_error})"
                )
            return "no fixture loaded"
        return (
            f"{fixture.name or 'fixture'}"
            f" ({format_frequency_short(fixture.start)} to"
            f" {format_frequency_short(fixture.stop)},"
            f" {len(fixture.points)} points,"
            f" Z0 {fixture.reference_impedance:g} ohm,"
            f" identity {fixture.identity()[:8]})"
        )

    def _fixture_identity(self) -> str:
        """Identity recorded by the stage the pipeline actually holds."""
        installed = self.app.processing_config.find(STAGE_FIXTURE)
        if installed is None:
            return ""
        return str(installed.as_dict.get("identity", "") or "")

    def fixture_status(self, preview: FixturePreview | None = None) -> str:
        """State of the pipeline, in one sentence, with its reason.

        What the pipeline holds is reported before what the loaded file
        *would* do, so an operator reading only this line is never told a
        fixture is in force when it is not.  A fixture that is in force but
        cannot apply to the current sweep is reported as exactly that: the
        stage stays installed and every sweep will refuse.
        """
        preview = self.current_preview() if preview is None else preview
        identity = self._fixture_identity()
        loaded = "" if self._fixture is None else self._fixture.identity()
        if identity and identity == loaded:
            if preview.ok:
                return f"installed as a stage ({identity[:8]})"
            return (
                f"installed as a stage ({identity[:8]}), but it cannot apply"
                f" to this sweep: {preview.reason}"
            )
        if identity:
            # Installed from elsewhere, and not the network on screen.
            return (
                "the pipeline holds a different fixture, installed as a stage"
                f" ({identity[:8]})"
            )
        if self._fixture is not None:
            # Computed now, never remembered: a stored preview would describe
            # the sweep as it was when the file was loaded.
            return f"not installed; {preview.describe()}"
        return "not installed"

    def _refresh_fixture_labels(self) -> None:
        """Show the loaded fixture and what the pipeline currently holds.

        One place decides every widget state, so the summary, the status
        line and the buttons cannot disagree about the fixture in force.
        """
        preview = self.current_preview()
        identity = self._fixture_identity()
        loaded = "" if self._fixture is None else self._fixture.identity()
        # A stage whose identity cannot be read is never assumed equal:
        # installing the loaded file is the way to repair it.
        in_force = bool(identity) and identity == loaded
        self.label_fixture.setText(self.fixture_summary())
        self.label_fixture_preview.setText(self.fixture_status(preview))
        self.btn_clear_fixture.setEnabled(bool(identity))
        self.btn_install_fixture.setEnabled(
            self._fixture is not None and not in_force and preview.ok
        )

    # -- fixture loading ------------------------------------------------
    def load_fixture_file(self, filename: str) -> bool:
        """Load a fixture from ``filename`` and report what it would do.

        The file is parsed with :meth:`Touchstone.loads_strict` so a broken
        or one-port file is refused with its own error instead of leaving a
        partial dataset behind.  A fixture that cannot cover this sweep is
        still *shown* — the operator needs to see what was loaded — but it is
        never installed.
        """
        touchstone = Touchstone(filename)
        try:
            with open(filename, encoding="utf-8") as infile:
                touchstone.loads_strict(infile.read())
        except (OSError, TypeError, ValueError, IndexError) as exc:
            self._fixture = None
            self._fixture_error = f"could not read {filename}: {exc}"
            self._refresh_fixture_labels()
            self._set_message(f"Fixture: {self._fixture_error}")
            return False
        try:
            fixture = fixture_from_touchstone(
                touchstone,
                name=os.path.basename(filename),
                source=filename,
            )
        except (ValueError, IndexError) as exc:
            self._fixture = None
            self._fixture_error = str(exc)
            self._refresh_fixture_labels()
            self._set_message(f"Fixture: {exc}")
            return False
        self._fixture = fixture
        self._fixture_error = ""
        self._refresh_fixture_labels()
        self.preview_current_fixture()
        return True

    def load_fixture(self) -> bool:
        """Ask for a touchstone file, then load it."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load fixture",
            "",
            "Touchstone 2-Port Files (*.s2p);;All files (*.*)",
        )
        if not filename:
            return False
        return self.load_fixture_file(filename)

    def _sweep_frequencies(self) -> tuple[int, ...]:
        """Frequencies of the sweep that will actually be processed."""
        get_frequencies = getattr(self.app.sweep, "get_frequencies", None)
        if get_frequencies is None:
            return ()
        try:
            return tuple(int(freq) for freq in get_frequencies())
        except (TypeError, ValueError):
            return ()

    def _measured_channels(self) -> dict[str, Sequence]:
        """The channels the displayed measurement actually holds."""
        data = getattr(self.app, "data", None)
        channels: dict[str, Sequence] = {}
        for name in FIXTURE_CHANNELS:
            channel = getattr(data, name.lower(), None) if data else None
            if channel:
                channels[name] = channel
        return channels

    def _measurement_impedance(self) -> float:
        """Reference impedance of the measurement, as the app states it."""
        impedance = getattr(self.app, "reference_impedance", None)
        if impedance is None:
            return DEFAULT_FIXTURE_Z0
        return float(impedance)

    def current_preview(self) -> FixturePreview:
        """Preview the loaded fixture against the current sweep."""
        sweep = self.app.sweep
        return preview_fixture(
            self._fixture,
            sweep_start=int(sweep.start),
            sweep_stop=int(sweep.end),
            sweep_frequencies=self._sweep_frequencies(),
            channels=self._measured_channels(),
            reference_impedance=self._measurement_impedance(),
        )

    def preview_current_fixture(self) -> bool:
        """Show the preview; return whether installing it would work."""
        preview = self.current_preview()
        self._refresh_fixture_labels()
        if not preview.ok:
            self._set_message(f"Fixture not installable: {preview.reason}")
        return preview.ok

    def install_fixture(self) -> bool:
        """Install the loaded fixture as a stage, if it would work.

        The stage is built by :func:`fixture_stage`, which embeds the sampled
        S-parameters and the content identity, and stored through
        :meth:`_set_processing`.  Nothing is installed on a preview that
        failed: a stage that refuses every sweep would record a correction
        that never ran.
        """
        fixture = self._fixture
        if fixture is None or not self.preview_current_fixture():
            return False
        stage = fixture_stage(fixture)
        config = self.app.processing_config
        # Installing is the repair for a pipeline holding a different
        # fixture, so the sync in _refresh_processing_labels must not treat
        # this identity as a foreign change.
        self._installed_identity = fixture.identity()
        self._set_processing(config.with_stage(stage, kind=STAGE_FIXTURE))
        self._set_message("")
        return True

    def clear_fixture(self) -> None:
        """Remove the fixture stage; keep the loaded file for reinstalling."""
        config = self.app.processing_config
        self._set_processing(config.with_stage(None, kind=STAGE_FIXTURE))
        self._refresh_fixture_labels()
        self._set_message("Fixture cleared: sweeps are no longer de-embedded.")

    def _refresh_processing_labels(self) -> None:
        config = self.app.processing_config
        self.label_processing.setText(config.describe())
        if hasattr(self, "label_fixture"):
            self._sync_fixture_with_config()
            self._refresh_fixture_labels()
        stage = config.find(STAGE_SMOOTHING)
        if stage is None:
            self.label_aperture.setText("not smoothing")
            return
        points = int(stage.as_dict.get("aperture_points", 1) or 1)
        self.label_aperture.setText(
            f"{points} point(s) across {max(1, self.app.sweep.points)}"
            " measured samples"
        )

    def _set_processing(self, config: ProcessingConfig) -> None:
        """Store a configuration and describe it; never silently drops it."""
        self.app.processing_config = config
        # A new pipeline is a new data stream, so an accumulated average may
        # not be blended with it.
        reset = getattr(self.app.running_average, "reset", None)
        if reset is not None:
            self.app.running_average = reset()
        self.app.mark_session_changed("sweep")
        self._refresh_processing_labels()

    def _current_smoothing(self) -> "object | None":
        percent = int(self.spin_aperture_percent.value())
        if percent <= 0:
            # Zero percent means "no smoothing"; the checkbox alone does not
            # imply a width, so this is reported rather than defaulted.
            return None
        points = smoothing_aperture_for(max(1, self.app.sweep.points), percent)
        try:
            return smoothing_stage(points)
        except ProcessingError:
            return None

    def update_smoothing(self, enabled: bool) -> None:
        """Enable or disable smoothing from the checkbox and the width.

        The checkbox records intent and is never flipped behind the
        operator's back.  A ticked box with a zero width applies no stage and
        says so in the message line, so "ticked but doing nothing" is always
        visible rather than silent.
        """
        config = self.app.processing_config
        stage = self._current_smoothing() if enabled else None
        self._set_processing(config.with_stage(stage, kind=STAGE_SMOOTHING))
        if enabled and stage is None:
            self._set_message(
                "Smoothing is selected but applies nothing: set an aperture"
                " wider than zero."
            )
        elif enabled:
            self._set_message("")

    def _set_message(self, text: str) -> None:
        """Show a short explanation next to the processing settings."""
        label = getattr(self, "label_processing_message", None)
        if label is not None:
            label.setText(text)

    def update_port_extension(self, *_args) -> None:
        """Store the hand-entered extension; zero removes the stage."""
        config = self.app.processing_config
        try:
            stage = port_extension_stage(
                port1_delay_ps=float(self.spin_port1_delay.value()),
                port1_loss_db_per_ghz=float(self.spin_port1_loss.value()),
            )
        except ProcessingError as error:
            self._set_message(f"Port extension: {error}")
            return
        self._set_processing(
            config.with_stage(stage, kind=STAGE_PORT_EXTENSION)
        )
        self._set_message("")
        self.label_extension.setText("")

    def fit_port_extension_from_standard(self) -> bool:
        """Fit delay and loss from the current sweep of an open or a short.

        The operator measures the standard through the fixture and presses
        Fit.  The fit refuses rather than guessing, and its reason is shown
        verbatim so a failed fit is never mistaken for a zero extension.
        """
        points = getattr(self.app.data, "s11", None) or []
        kind = (
            StandardKind.OPEN
            if self.combo_fit_standard.currentData() == "open"
            else StandardKind.SHORT
        )
        fit = fit_port_extension(points, kind)
        if not fit.ok:
            self.label_extension.setText(f"no fit: {fit.reason}")
            return False
        self.spin_port1_delay.setValue(fit.delay_ps)
        self.spin_port1_loss.setValue(fit.loss_db_per_ghz)
        # The spin boxes fire the value handlers, but a fit that happened to
        # match them exactly would not; store the stage explicitly.
        self.update_port_extension()
        self.label_extension.setText(fit.describe())
        return True

    def update_aperture(self, _value: int) -> None:
        if not self.check_smoothing.isChecked():
            self._refresh_processing_labels()
            return
        self.update_smoothing(True)

    def update_average(self, *_args) -> None:
        config = self.app.processing_config
        if not self.check_average.isChecked():
            self._set_processing(config.with_stage(None, kind=STAGE_AVERAGE))
            return
        mode = str(self.combo_average_mode.currentData() or "cumulative")
        alpha = (
            float(self.spin_average_alpha.value())
            if mode == "exponential"
            else None
        )
        try:
            stage = average_stage(mode=mode, alpha=alpha)
        except ProcessingError as exc:
            logger.warning("Rejected average settings: %s", exc)
            return
        self._set_processing(config.with_stage(stage))
