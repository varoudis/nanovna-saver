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
from threading import RLock
from time import sleep
from typing import TYPE_CHECKING

import serial
from PySide6 import QtWidgets
from PySide6.QtCore import Signal

from ..Hardware.Hardware import Interface, get_comment, get_interfaces, get_VNA
from ..Hardware.Simulated import make_simulated_vna
from .Control import Control

if TYPE_CHECKING:
    from ..NanoVNASaver.NanoVNASaver import NanoVNASaver as vna_app


logger = logging.getLogger(__name__)

#: Scenario names offered in demonstration mode (SIM-02.1).
SCENARIO_NAMES = ("antenna", "filter", "cable", "attenuator")

SCENARIO_LABELS = {
    "antenna": "Antenna resonance",
    "filter": "Butterworth low-pass filter",
    "cable": "Coaxial cable",
    "attenuator": "Fixed attenuator",
}


class DeviceConnectionError(Exception):
    """A connection attempt failed; the message is meant for the user."""


def _close_quietly(interface) -> None:
    if interface is None:
        return
    try:
        interface.close()
    except Exception:
        logger.debug("Unable to close %s", interface, exc_info=True)


def open_tcp_interface(address: str) -> Interface:
    """Open a TCP interface for address, closing it again on any failure."""
    url = f"socket://{address}"
    logger.info("Connection TCP %s", url)
    interface = None
    try:
        interface = serial.serial_for_url(url, do_not_open=True)
        interface.type = "network"
        interface.lock = RLock()
        interface.timeout = 0.05
        interface.open()
        interface.comment = get_comment(interface)
        interface.close()
        interface.open()
    except Exception as exc:
        _close_quietly(interface)
        raise DeviceConnectionError(
            f"Unable to connect to {address}: {exc}"
        ) from exc
    return interface


def open_serial_interface(interface) -> Interface:
    """Open the selected serial interface, closing it on any failure."""
    logger.info("Connection %s", interface)
    try:
        interface.open()
        is_open = interface.is_open
    except Exception as exc:
        _close_quietly(interface)
        raise DeviceConnectionError(
            f"Unable to open {interface}: {exc}"
        ) from exc
    if not is_open:
        _close_quietly(interface)
        raise DeviceConnectionError(f"Unable to open port {interface}")
    interface.timeout = 0.05
    return interface


def validate_frequencies(frequencies) -> list[int]:
    """Return the frequencies as integers, rejecting unusable responses."""
    if not frequencies:
        raise DeviceConnectionError(
            "The device reported no sweep frequencies. Check that the "
            "correct port and connection type are selected, then try again."
        )
    normalized = []
    for frequency in frequencies:
        if isinstance(frequency, bool) or not isinstance(
            frequency, (int, float)
        ):
            raise DeviceConnectionError(
                "The device reported an invalid sweep frequency "
                f"({frequency!r}). Check the device firmware and connection."
            )
        if frequency < 0:
            raise DeviceConnectionError(
                "The device reported a negative sweep frequency "
                f"({frequency!r}). Check the device firmware and connection."
            )
        normalized.append(int(frequency))
    return normalized


def initialize_vna(app: "vna_app", interface) -> tuple[object, list[int]]:
    """Initialize a candidate device on an already opened interface.

    The device is only returned once initialization and frequency
    validation succeeded. On any failure whatever was opened is closed
    again and a DeviceConnectionError carrying a user facing message is
    raised; the caller is responsible for publishing the connected state.
    """
    stage = "identify the connected device"
    try:
        vna = get_VNA(interface)
        vna.validateInput = app.settings.value(
            "SerialInputValidation", False, bool
        )
        stage = "read the sweep frequencies from the connected device"
        frequencies = validate_frequencies(list(vna.read_frequencies()))
    except DeviceConnectionError:
        _close_quietly(interface)
        raise
    except Exception as exc:
        _close_quietly(interface)
        raise DeviceConnectionError(f"Unable to {stage}: {exc}") from exc
    return vna, frequencies


def apply_frequency_range(app: "vna_app", vna, frequencies: list[int]) -> None:
    """Seed the sweep control from the validated device frequencies."""
    start, stop = frequencies[0], frequencies[-1]
    logger.info("Read starting frequency %s and end frequency %s", start, stop)
    app.sweep_control.set_start(start)
    if start < stop:
        app.sweep_control.set_end(stop)
    else:
        app.sweep_control.set_end(
            start + vna.datapoints * app.sweep_control.get_segments()
        )

    app.sweep_control.set_segments(1)  # speed up things
    app.sweep_control.update_center_span()
    app.sweep_control.update_step_size()


class SerialControl(Control):
    # true when serial port was connected and false when it was disconnected
    connected = Signal(bool)

    def __init__(self, app: "vna_app"):
        super().__init__(app, "Serial port control")

        self.interface = Interface("serial", "none")

        self.cb_tcp = QtWidgets.QCheckBox("TCP/IP")
        self.cb_tcp.stateChanged.connect(self.update_connect_btn_state)

        # Move TCP/IP checkbox to the title
        # self.setTitle("")

        # Create a widget for the header
        header_widget = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # The label
        lbl_title = QtWidgets.QLabel("Network connection")
        # Make it look like a groupbox title (usually bold)
        font = lbl_title.font()
        font.setBold(True)
        font.setPointSize(10)
        lbl_title.setFont(font)

        header_layout.addWidget(lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(self.cb_tcp)

        self.layout.addRow(header_widget)

        # SIM-02.1: a clearly labelled simulated device, using the same
        # connection workflow as a physical one.  Test-only fault injection
        # stays out of this control (SIM-03.3).
        self.cb_simulated = QtWidgets.QCheckBox("Simulated device (demo)")
        self.cb_simulated.setToolTip(
            "Explore the application without hardware. Simulated data is"
            " labelled everywhere it is saved or exported."
        )
        self.cb_simulated.stateChanged.connect(self.update_simulation_state)
        self.layout.addRow(self.cb_simulated)

        self.cb_scenario = QtWidgets.QComboBox()
        for name in SCENARIO_NAMES:
            self.cb_scenario.addItem(SCENARIO_LABELS[name], name)
        self.cb_scenario.setToolTip(
            "Analytic scenario used by the simulated device."
        )
        scenario_row = QtWidgets.QHBoxLayout()
        scenario_row.addWidget(QtWidgets.QLabel("Scenario"), stretch=0)
        scenario_row.addWidget(self.cb_scenario, stretch=1)
        self.row_scenario = QtWidgets.QWidget()
        self.row_scenario.setLayout(scenario_row)
        self.layout.addRow(self.row_scenario)

        self.inp_port = QtWidgets.QComboBox()
        self.inp_port.setMinimumHeight(20)
        self.rescanSerialPort()
        self.inp_port.setEditable(True)
        self.inp_port.currentIndexChanged.connect(self.update_connect_btn_state)
        self.inp_port.editTextChanged.connect(self.update_connect_btn_state)

        self.btn_rescan = QtWidgets.QPushButton("Rescan")
        self.btn_rescan.setMinimumHeight(20)
        self.btn_rescan.setFixedWidth(60)
        self.btn_rescan.clicked.connect(self.rescanSerialPort)
        intput_layout = QtWidgets.QHBoxLayout()
        intput_layout.addWidget(QtWidgets.QLabel("Port"), stretch=0)
        intput_layout.addWidget(self.inp_port, stretch=1)
        intput_layout.addWidget(self.btn_rescan, stretch=0)
        self.layout.addRow(intput_layout)

        button_layout = QtWidgets.QHBoxLayout()

        self.btn_toggle = QtWidgets.QPushButton("Connect to device")
        self.btn_toggle.setMinimumHeight(20)
        self.btn_toggle.clicked.connect(self.serialButtonClick)
        button_layout.addWidget(self.btn_toggle, stretch=1)

        self.btn_settings = QtWidgets.QPushButton("Manage")
        self.btn_settings.setMinimumHeight(20)
        self.btn_settings.setFixedWidth(60)
        self.btn_settings.clicked.connect(
            lambda: self.app.display_window("device_settings")
        )

        button_layout.addWidget(self.btn_settings, stretch=0)
        self.layout.addRow(button_layout)

        # Assuming serial is disconnected
        self.update_settings_state(False)
        self.update_simulation_state()

        self.connected.connect(self.update_settings_state)

    def rescanSerialPort(self):
        self.inp_port.clear()
        for iface in get_interfaces():
            self.inp_port.insertItem(1, f"{iface}", iface)
        self.inp_port.repaint()

    def serialButtonClick(self):
        if not self.app.vna.connected():
            self.connect_device()
        else:
            self.disconnect_device()

    def connect_device(self):
        """Connect, publishing the connected state only on full success."""
        previous_vna = self.app.vna
        interface = None
        try:
            if self.cb_simulated.isChecked():
                return self._connect_simulated()
            interface = self._open_interface()
            sleep(0.1)
            vna, frequencies = initialize_vna(self.app, interface)
            self.app.vna = vna
            apply_frequency_range(self.app, vna, frequencies)
        except DeviceConnectionError as exc:
            self.app.vna = previous_vna
            self._connection_failed(exc)
            return
        except Exception as exc:
            self.app.vna = previous_vna
            _close_quietly(interface)
            self._connection_failed(
                DeviceConnectionError(f"Unable to prepare the sweep: {exc}")
            )
            return

        self.interface = interface

        # connected
        self.btn_toggle.setText("Disconnect")
        self.btn_toggle.repaint()
        self.connected.emit(True)

        self.app.windows["sweep_settings"].vna_connected()

        logger.debug("Starting initial sweep")
        self.app.sweep_start()

    def _open_interface(self):
        """Open the serial or TCP interface selected in the UI."""
        if self.cb_simulated.isChecked():
            # A simulated device needs no transport at all.
            return None
        if self.cb_tcp.isChecked():
            address = self.inp_port.currentText()
            if not address:
                raise DeviceConnectionError(
                    "Enter a host and port to connect to over TCP/IP."
                )
            return open_tcp_interface(address)

        selected = self.inp_port.currentData()
        if not selected:
            raise DeviceConnectionError("Select a serial port to connect to.")
        return open_serial_interface(selected)

    def selected_scenario(self) -> str:
        return self.cb_scenario.currentData() or SCENARIO_NAMES[0]

    def _connect_simulated(self) -> None:
        """Connect the simulated device through the normal workflow.

        A simulated device owns its own in-process transport, so no serial
        port is opened and the control's interface is left as the inert
        placeholder it started with.
        """
        scenario = self.selected_scenario()
        vna = make_simulated_vna(scenario)
        self.app.vna = vna
        logger.info("Connected simulated device (%s)", scenario)

        self.btn_toggle.setText("Disconnect")
        self.btn_toggle.repaint()
        self.connected.emit(True)
        self.app.windows["sweep_settings"].vna_connected()
        self.app.sweep_start()

    def update_simulation_state(self, *_args) -> None:
        """Keep the transport controls consistent with the mode selector."""
        simulated = self.cb_simulated.isChecked()
        self.row_scenario.setVisible(simulated)
        self.cb_scenario.setEnabled(simulated)
        self.inp_port.setEnabled(not simulated)
        self.btn_rescan.setEnabled(
            not simulated and not self.is_vna_connected()
        )
        self.cb_tcp.setEnabled(not simulated)
        self.update_connect_btn_state()

    def _connection_failed(self, failure: DeviceConnectionError) -> None:
        """Keep the control and the application in a disconnected state."""
        logger.error("Connection attempt failed: %s", failure)
        self.btn_toggle.setText("Connect to device")
        self.btn_toggle.repaint()
        self.update_settings_state(False)
        self.app.showError(str(failure))

    def disconnect_device(self):
        if getattr(self.app.vna, "simulated", False):
            self._disconnect_simulated()
            return
        with self.interface.lock:
            logger.info("Closing connection to %s", self.interface)
            try:
                self.app.vna.disconnect()
            except IOError as exc:
                logger.error("Unable to disconnect from VNA: %s", exc)

            self.interface.close()
            self.btn_toggle.setText("Connect to device")
            self.btn_toggle.repaint()

            self.connected.emit(False)

    def _disconnect_simulated(self) -> None:
        """End a demonstration session without touching a serial port."""
        logger.info("Closing simulated device connection")
        try:
            self.app.vna.disconnect()
        except OSError as exc:
            logger.error("Unable to disconnect simulated device: %s", exc)
        self.btn_toggle.setText("Connect to device")
        self.btn_toggle.repaint()
        self.connected.emit(False)

    def update_connect_btn_state(self) -> None:
        # Eanble Connect/Dicsonnect button only if:
        # a) serial was already connected
        # b) serial is disconnected AND inp_port was selected
        # c) TCP is checked and text is present
        port_selected = self.inp_port.currentData() is not None
        tcp_selected = self.cb_tcp.isChecked() and bool(
            self.inp_port.currentText()
        )
        self.btn_toggle.setEnabled(
            self.is_vna_connected()
            or self.cb_simulated.isChecked()
            or port_selected
            or tcp_selected
        )

    def is_vna_connected(self) -> bool:
        return self.app.vna and self.app.vna.connected()

    def update_settings_state(self, was_connected: bool) -> None:
        self.btn_rescan.setEnabled(not was_connected)
        self.btn_settings.setEnabled(was_connected)
        self.update_connect_btn_state()
