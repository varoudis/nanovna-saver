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
from functools import partial
from threading import RLock
from typing import TYPE_CHECKING

from PySide6 import QtWidgets
from PySide6.QtCore import Signal, Slot

from ..Connection import (
    CANCELLED,
    CONNECTED,
    ConnectionBusyError,
    ConnectionOutcome,
    ConnectionRequest,
    ConnectionService,
    DeviceConnectionError,
    PortScanService,
    ScanOutcome,
    close_quietly,
    identify_device,
    open_serial_interface,
    open_tcp_interface,
)
from ..Connection import validate_frequencies as _validate_frequencies
from ..Hardware.Hardware import (
    Interface,
    get_comment,
    get_interfaces,
    get_VNA,
    list_serial_ports,
)
from ..Hardware.ReplayScenario import (
    ReplayData,
    ReplayError,
    ReplayScenario,
    load_replay_file,
)
from ..Hardware.Simulated import make_simulated_vna
from ..Settings.ReplayRange import (
    ReplayRangeError,
    ReplayRangePlan,
    plan_replay_range,
)
from .Control import Control

if TYPE_CHECKING:
    from ..NanoVNASaver.NanoVNASaver import NanoVNASaver as vna_app


logger = logging.getLogger(__name__)

#: Scenario name used for a Touchstone replay source.
REPLAY_SCENARIO = "replay"

#: Scenario names offered in demonstration mode (SIM-02.1).
SCENARIO_NAMES = ("antenna", "filter", "cable", "attenuator", REPLAY_SCENARIO)

SCENARIO_LABELS = {
    "antenna": "Antenna resonance",
    "filter": "Butterworth low-pass filter",
    "cable": "Coaxial cable",
    "attenuator": "Fixed attenuator",
    REPLAY_SCENARIO: "Touchstone replay (file)",
}

#: Button caption while an attempt owns the device conversation.
CONNECTING_LABEL = "Connecting ..."


# Kept importable from here: the helpers now live next to the worker that
# uses them, but the control (and its tests) refer to them by this name.
_close_quietly = close_quietly
validate_frequencies = _validate_frequencies


def initialize_vna(app: "vna_app", interface) -> tuple[object, list[int]]:
    """Initialize a candidate device on an already opened interface.

    The device is only returned once initialization and frequency
    validation succeeded. On any failure whatever was opened is closed
    again and a DeviceConnectionError carrying a user facing message is
    raised; the caller is responsible for publishing the connected state.
    """
    try:
        return identify_device(
            interface,
            get_VNA,
            validate_input=app.settings.value(
                "SerialInputValidation", False, bool
            ),
            comment_probe=get_comment,
        )
    except DeviceConnectionError:
        _close_quietly(interface)
        raise
    except Exception as exc:
        _close_quietly(interface)
        raise DeviceConnectionError(
            f"Unable to identify the connected device: {exc}"
        ) from exc


def open_selected_interface(tcp: bool, address: str, selected) -> object:
    """Open the transport the user selected, described by plain values.

    Runs in the worker thread and deliberately takes only plain values, so
    it never touches a widget; the caller captured them on the GUI thread.
    ``get_comment`` is looked up at call time to stay patchable.
    """
    if tcp:
        if not address:
            raise DeviceConnectionError(
                "Enter a host and port to connect to over TCP/IP."
            )
        return open_tcp_interface(address, comment_probe=get_comment)
    if selected is None:
        raise DeviceConnectionError("Select a serial port to connect to.")
    return open_serial_interface(selected)


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
    #: Emitted once per hardware connection attempt with a status string
    #: (``"connected"``, ``"failed"``, ``"cancelled"`` or ``"timed out"``).
    connection_finished = Signal(str)

    def __init__(self, app: "vna_app"):
        super().__init__(app, "Serial port control")

        self.interface = Interface("serial", "none")
        #: Single owner of the (blocking) device conversation.
        self._service = ConnectionService(self)
        self._service.signals.attempt_finished.connect(
            self._on_attempt_finished
        )
        #: Owns the (blocking) port enumeration when it runs in a thread.
        self._scanner = PortScanService(self)
        self._scanner.signals.results.connect(self._on_scan_finished)
        # The result is emitted from inside the scan; the thread may still be
        # leaving its run loop when it arrives, so the button state is
        # restored once more when the worker is really gone.
        self._scanner.signals.finished.connect(self.update_scan_state)
        #: Token of the attempt the control is waiting for (``None`` when
        #: no hardware attempt is pending).
        self._pending_token: object = None
        #: Token of the scan whose result will repopulate the port list.
        self._scan_token: object = None
        #: The interface a live device was connected on.
        self._live_interface = None
        #: True between starting the worker and applying its result.
        self._connecting = False
        #: Serialises port-list replacement against the scan result.
        self._port_lock = RLock()

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
            # Replay is a scenario, not a separate import path: its
            # Touchstone file runs through the same sweep pipeline.
            self.cb_scenario.addItem(SCENARIO_LABELS[name], name)
        self.cb_scenario.setToolTip(
            "Scenario used by the simulated device. Touchstone replay sweeps"
            " a recorded S1P/S2P file instead of an analytic model."
        )
        self.cb_scenario.currentIndexChanged.connect(
            self.update_simulation_state
        )
        scenario_row = QtWidgets.QHBoxLayout()
        scenario_row.addWidget(QtWidgets.QLabel("Scenario"), stretch=0)
        scenario_row.addWidget(self.cb_scenario, stretch=1)
        self.row_scenario = QtWidgets.QWidget()
        self.row_scenario.setLayout(scenario_row)
        self.layout.addRow(self.row_scenario)

        # -- Touchstone replay source -------------------------------
        self.btn_replay_file = QtWidgets.QPushButton("Choose replay file ...")
        self.btn_replay_file.setMinimumHeight(20)
        self.btn_replay_file.setToolTip(
            "Choose a Touchstone .s1p or .s2p file to replay. Replay never"
            " extrapolates beyond the file's own frequency range."
        )
        self.btn_replay_file.clicked.connect(self.choose_replay_file)
        self.row_replay_file = QtWidgets.QWidget()
        replay_file_layout = QtWidgets.QVBoxLayout(self.row_replay_file)
        replay_file_layout.setContentsMargins(0, 0, 0, 0)
        replay_file_layout.setSpacing(1)
        replay_file_layout.addWidget(self.btn_replay_file)
        self.label_replay_source = QtWidgets.QLabel("No replay file loaded")
        self.label_replay_source.setObjectName("replay_source")
        self.label_replay_source.setWordWrap(True)
        replay_file_layout.addWidget(self.label_replay_source)
        self.layout.addRow(self.row_replay_file)

        #: Validated replay source; ``None`` until a file is chosen.
        self._replay: ReplayData | None = None

        self.inp_port = QtWidgets.QComboBox()
        self.inp_port.setMinimumHeight(20)
        # Startup must not touch a port: only the OS metadata is read here
        # (microseconds), and the firmware variant is filled in by a
        # background rescan or by the connection attempt itself.
        self._fill_ports(list_serial_ports())
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

    # -- port enumeration ----------------------------------------------
    def is_scanning(self) -> bool:
        """True while a port scan owns the enumeration conversation."""
        return self._scanner.busy

    def rescanSerialPort(self) -> bool:
        """Refresh the port list without blocking the GUI thread.

        Enumeration asks every candidate port what firmware it runs, which
        is a blocking serial conversation; it therefore runs in a
        :class:`ScanWorker` and its result is applied by
        :meth:`_on_scan_finished`.  Re-entering while a scan is running is a
        no-op, so a double click cannot start two scans over the same ports.
        """
        if self.is_connecting():
            logger.info("Not scanning ports while a connection is pending")
            return False
        if self._scanner.busy:
            logger.info("Ignoring a rescan: a port scan is already running")
            return False
        try:
            token = self._scanner.begin(self._scan_ports)
        except ConnectionBusyError:
            logger.info("Ignoring a rescan: a port scan is already running")
            return False
        self._scan_token = token
        self.update_scan_state()
        return True

    def _scan_ports(self):
        """Enumerate and probe the ports; runs in the scan worker thread.

        ``get_interfaces`` is looked up here so it stays patchable on the
        module; the call already bounds itself with ``SCAN_BUDGET``.
        """
        return get_interfaces()

    def cancel_scan(self) -> None:
        """Discard a running scan's result; the thread finishes on its own."""
        if not self._scanner.busy:
            return
        logger.info("Cancelling the pending port scan")
        self._scan_token = None
        self._scanner.cancel()
        self.update_scan_state()

    @Slot(object)
    def _on_scan_finished(self, outcome: ScanOutcome) -> None:
        """Apply a scan result on the GUI thread - only if it is current."""
        if outcome.token is not self._scan_token:
            logger.info("Discarding a stale serial port scan result")
            return
        self._scan_token = None
        if not outcome.ok:
            logger.info("Port scan did not complete: %s", outcome.message)
            self.update_scan_state()
            return
        self._fill_ports(outcome.interfaces)
        self.update_scan_state()

    def _fill_ports(self, interfaces) -> None:
        """Replace the port list, keeping the user's selection when possible.

        A port that disappears is not silently swapped for another one: the
        previous text stays selected so the user sees what they chose.
        """
        with self._port_lock:
            previous = self.inp_port.currentData()
            previous_text = self.inp_port.currentText()
            self.inp_port.clear()
            for iface in interfaces:
                self.inp_port.insertItem(1, f"{iface}", iface)
            if previous is not None:
                index = self.inp_port.findData(previous)
                if index >= 0:
                    self.inp_port.setCurrentIndex(index)
            elif previous_text and self.inp_port.isEditable():
                self.inp_port.setEditText(previous_text)
        self.inp_port.repaint()

    def update_scan_state(self) -> None:
        """Show the scan in progress and disable conflicting controls."""
        scanning = self._scanner.busy
        self.btn_rescan.setEnabled(
            not scanning
            and not self.is_connecting()
            and not self.is_vna_connected()
            and not self.cb_simulated.isChecked()
        )
        self.btn_rescan.setText("Scanning" if scanning else "Rescan")

    def serialButtonClick(self):
        if self.is_connecting():
            # A second click must not start a second attempt.
            return
        if not self.app.vna.connected():
            self.connect_device()
        else:
            self.disconnect_device()

    # -- connection lifecycle -----------------------------------------
    def is_connecting(self) -> bool:
        """True while a hardware attempt owns the device conversation."""
        return self._connecting

    def connect_device(self):
        """Start connecting without blocking the GUI thread.

        A simulated or replay source is local and fast, so it is published
        synchronously.  A hardware device needs a whole serial conversation
        (port open, firmware identification, frequency read); that runs in a
        :class:`ConnectionWorker` and its result is applied by
        :meth:`_on_attempt_finished` on the GUI thread.
        """
        if self.sweep_active():
            # Replacing the device mid-sweep would change what the running
            # worker reads between segments.
            self.app.showError(
                "A sweep is running. Stop it before changing the device or"
                " the replay source."
            )
            return
        if self.is_connecting():
            logger.info("Ignoring a second connect request while connecting")
            return
        previous_vna = self.app.vna
        try:
            if self.cb_simulated.isChecked():
                return self._connect_simulated()
        except DeviceConnectionError as exc:
            self.app.vna = previous_vna
            self._connection_failed(exc)
            return
        except Exception as exc:
            self.app.vna = previous_vna
            self._connection_failed(
                DeviceConnectionError(f"Unable to prepare the sweep: {exc}")
            )
            return
        self._begin_hardware_attempt()

    def _begin_hardware_attempt(self) -> None:
        """Resolve the request on the GUI thread and start the worker.

        Everything the worker needs is captured here, as a plain value or a
        callable closing over plain values: reading a widget from the worker
        thread would be a cross-thread access to a Qt object.
        """
        tcp = self.cb_tcp.isChecked()
        address = self.inp_port.currentText()
        selected = self.inp_port.currentData()
        if tcp and not address:
            self._connection_failed(
                DeviceConnectionError(
                    "Enter a host and port to connect to over TCP/IP."
                )
            )
            return
        if not tcp and selected is None:
            self._connection_failed(
                DeviceConnectionError("Select a serial port to connect to.")
            )
            return
        request = ConnectionRequest(
            open=partial(
                open_selected_interface,
                tcp=tcp,
                address=address,
                selected=selected,
            ),
            identify=partial(self._identify, self._input_validation()),
            note=str(address if tcp else selected or ""),
        )
        try:
            token = self._service.begin(request)
        except ConnectionBusyError:
            logger.info("A connection attempt is already running")
            return
        self._pending_token = token
        self._previous_vna_snapshot = self.app.vna
        self._connecting = True
        self._entering_connecting_state()

    def _identify(self, validate_input: bool, interface):
        """Identify the device on ``interface``; runs in the worker thread.

        ``get_VNA``/``get_comment`` are looked up here rather than captured
        so they stay patchable on the module; ``validate_input`` was read
        from the settings on the GUI thread.
        """
        return identify_device(
            interface,
            get_VNA,
            validate_input=validate_input,
            comment_probe=get_comment,
        )

    def _input_validation(self) -> bool:
        try:
            return bool(
                self.app.settings.value("SerialInputValidation", False, bool)
            )
        except Exception:  # a stub app may not have settings at all
            logger.debug("No settings available for input validation")
            return False

    def _entering_connecting_state(self) -> None:
        """Keep the button busy and the conflicting controls disabled."""
        self.btn_toggle.setText(CONNECTING_LABEL)
        self.btn_toggle.setEnabled(False)
        self.inp_port.setEnabled(False)
        self.cb_tcp.setEnabled(False)
        self.btn_rescan.setEnabled(False)
        self.btn_settings.setEnabled(False)
        self.cb_simulated.setEnabled(False)
        self.cb_scenario.setEnabled(False)
        self.btn_replay_file.setEnabled(False)
        self.btn_toggle.repaint()

    def _leaving_connecting_state(self) -> None:
        self._connecting = False
        self._pending_token = None
        self.btn_toggle.setText("Connect to device")
        self.cb_simulated.setEnabled(True)
        self.update_simulation_state()
        self.update_settings_state(self.is_vna_connected())
        self.btn_toggle.repaint()

    def cancel_connection(self) -> None:
        """Cancel a pending hardware attempt and discard its result."""
        if not self.is_connecting():
            return
        logger.info("Cancelling the pending connection attempt")
        self._pending_token = None
        self._service.cancel()
        self._leaving_connecting_state()

    def shutdown(self, timeout_ms: int | None = None) -> bool:
        """Stop any pending work and release the device (window close).

        Returns ``True`` when no worker is left running.  Called from
        ``closeEvent``; an attempt or scan that is still in flight is
        cancelled and waited for here, so the process never exits with a
        live thread reading a port that is being torn down.
        """
        self._pending_token = None
        self._scan_token = None
        self._connecting = False
        if timeout_ms is None:
            stopped = self._service.shutdown()
            scanned = self._scanner.shutdown()
        else:
            stopped = self._service.shutdown(timeout_ms)
            scanned = self._scanner.shutdown(timeout_ms)
        self.disconnect_device()
        return stopped and scanned

    def owns_attempt(self, outcome: ConnectionOutcome) -> bool:
        """True when ``outcome`` belongs to the attempt being waited for."""
        token = self._pending_token
        return token is not None and outcome.token is token

    @Slot(object)
    def _on_attempt_finished(self, outcome: ConnectionOutcome) -> None:
        """Apply a worker result on the GUI thread - only if it is current."""
        if not self.is_connecting():
            logger.info("Discarding a connection result: no attempt is pending")
            close_quietly(outcome.interface)
            return
        if not self.owns_attempt(outcome):
            # A cancelled or superseded attempt: never publish its device.
            logger.info("Discarding a stale connection result")
            close_quietly(outcome.interface)
            return
        self._pending_token = None
        if outcome.status != CONNECTED:
            previous = self._previous_vna_snapshot
            self._leaving_connecting_state()
            self._previous_vna_snapshot = None
            self.app.vna = previous
            if outcome.status == CANCELLED:
                logger.info("Connection attempt cancelled")
                self.connection_finished.emit(outcome.status)
                return
            self._connection_failed(
                DeviceConnectionError(outcome.message), status=outcome.status
            )
            return
        previous = self._previous_vna_snapshot
        try:
            self._publish_connected(outcome)
        except Exception as exc:
            logger.exception("Unable to prepare the sweep after connecting")
            close_quietly(outcome.interface)
            self.app.vna = previous
            self._leaving_connecting_state()
            self._connection_failed(
                DeviceConnectionError(f"Unable to prepare the sweep: {exc}")
            )
        finally:
            self._previous_vna_snapshot = None

    #: The device that was live when the current attempt started, so a
    #: failed attempt can restore it instead of leaving a half-applied one.
    _previous_vna_snapshot = None

    def _publish_connected(self, outcome: ConnectionOutcome) -> None:
        """Adopt a successfully identified device and start the first sweep."""
        vna = outcome.vna
        frequencies = list(outcome.frequencies)
        if not frequencies:
            raise DeviceConnectionError("The device reported no frequencies.")
        self.app.vna = vna

        self.interface = outcome.interface
        self._live_interface = outcome.interface
        # Seed the sweep control before publishing the connected state: a
        # range that cannot be applied keeps the control disconnected.
        apply_frequency_range(self.app, vna, frequencies)
        self._leaving_connecting_state()

        # connected
        self.btn_toggle.setText("Disconnect")
        self.btn_toggle.repaint()
        self.connected.emit(True)
        self.connection_finished.emit(CONNECTED)

        self.app.windows["sweep_settings"].vna_connected()

        logger.debug("Starting initial sweep")
        self.app.sweep_start()

    def selected_scenario(self) -> str:
        return self.cb_scenario.currentData() or SCENARIO_NAMES[0]

    # -- replay source -------------------------------------------------
    def replay_data(self) -> ReplayData | None:
        """The loaded replay source, or ``None``."""
        return self._replay

    def sweep_active(self) -> bool:
        """True while the worker is acquiring a sweep."""
        worker = getattr(self.app, "worker", None)
        running = getattr(worker, "isRunning", None)
        return bool(running()) if callable(running) else False

    def choose_replay_file(self, filename: str | None = None) -> bool:
        """Load a Touchstone replay source, keeping the previous one on error.

        Returns ``True`` when a valid source is loaded.  Any failure leaves
        the previously working source untouched and reports the reason, so
        a bad pick never silently changes what the next sweep replays.
        """
        if self.sweep_active():
            # The running worker owns the device conversation; swapping the
            # source would change the meaning of samples mid-acquisition.
            self.app.showError(
                "A sweep is running. Stop it before changing the replay source."
            )
            return False
        if self.is_connecting():
            self.app.showError(
                "A connection attempt is in progress. Wait for it to finish"
                " before changing the replay source."
            )
            return False
        connected_vna = getattr(self.app, "vna", None)
        if getattr(connected_vna, "scenario_key", "") == REPLAY_SCENARIO and (
            getattr(connected_vna, "connected", lambda: False)()
        ):
            # The connected device already captured its source; swapping the
            # UI state would make the next sweep replay something other than
            # what was announced.
            self.app.showError(
                "Disconnect the replay device before choosing another file."
            )
            return False
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Choose a Touchstone file to replay",
                "",
                "Touchstone Files (*.s1p *.s2p);;All files (*.*)",
            )
        if not filename:
            return False
        try:
            source = load_replay_file(filename)
        except ReplayError as exc:
            logger.error("Replay source rejected: %s", exc)
            self.app.showError(str(exc))
            return False
        except Exception as exc:  # never let a pick break the control
            logger.exception("Unexpected replay load failure: %s", exc)
            self.app.showError(f"The replay file could not be loaded: {exc}")
            return False
        self.set_replay_data(source)
        return True

    def set_replay_data(self, source: ReplayData | None) -> None:
        """Publish a validated replay source and select the replay scenario."""
        self._replay = source
        if source is None:
            self.label_replay_source.setText("No replay file loaded")
            return
        self.label_replay_source.setText(f"{source.name}\n{source.summary()}")
        index = self.cb_scenario.findData(REPLAY_SCENARIO)
        if index >= 0:
            self.cb_scenario.setCurrentIndex(index)

    def replay_vna(self):
        """Build the replay device, or raise a user-facing failure."""
        if self._replay is None:
            raise DeviceConnectionError(
                "No replay file is loaded. Choose a Touchstone .s1p or"
                " .s2p file before connecting the replay scenario."
            )
        return make_simulated_vna(
            ReplayScenario(self._replay),
            sweep_start=self._replay.start,
            sweep_stop=self._replay.stop,
        )

    def _connect_simulated(self) -> None:
        """Connect the simulated device through the normal workflow.

        A simulated device owns its own in-process transport, so no serial
        port is opened and the control's interface is left as the inert
        placeholder it started with.  For replay the sweep default is
        planned *before* the device is published, so a file whose span
        cannot support the configured points leaves the previous device
        and sweep untouched.  This path is local and fast, so it stays
        synchronous: a pending hardware attempt never blocks it (the port
        selection is disabled while connecting, but the demo path is not).
        """
        scenario = self.selected_scenario()
        replay = scenario == REPLAY_SCENARIO
        if not replay:
            self.app.vna = make_simulated_vna(scenario)
            logger.info("Connected simulated device (%s)", scenario)
            self._simulated_connected()
            return

        vna = self.replay_vna()
        plan = self.replay_range_plan(vna)
        self.app.vna = vna
        logger.info("Connected simulated device (%s)", scenario)
        self.apply_replay_range(plan)
        self._simulated_connected()

    def _simulated_connected(self) -> None:
        """Publish the connected state and start the first sweep."""
        self.btn_toggle.setText("Disconnect")
        self.btn_toggle.repaint()
        self.connected.emit(True)
        self.app.windows["sweep_settings"].vna_connected()
        self.app.sweep_start()

    def replay_range_plan(self, vna=None) -> ReplayRangePlan:
        """Plan the default replay sweep for the connected device.

        Uses the point count the sweep control will actually request (the
        device's ``datapoints``) and the configured segment count.  Raises
        :class:`DeviceConnectionError` when the file span cannot hold that
        many distinct frequencies, so the caller can keep the previous
        state instead of replaying an illegal or degenerate grid.
        """
        if self._replay is None:
            raise DeviceConnectionError(
                "No replay file is loaded. Choose a Touchstone .s1p or"
                " .s2p file before connecting the replay scenario."
            )
        sweep = self.app.sweep
        points = int(getattr(vna, "datapoints", sweep.points))
        segments = self.segment_count()
        try:
            return plan_replay_range(
                self._replay.start,
                self._replay.stop,
                points=points,
                segments=segments,
                logarithmic=sweep.properties.logarithmic,
            )
        except ReplayRangeError as exc:
            raise DeviceConnectionError(
                f"{exc} The replay source {self._replay.name} is still"
                " loaded; adjust the sweep settings or choose another file."
            ) from exc

    def segment_count(self) -> int:
        """Segment count the sweep control will use."""
        getter = getattr(self.app.sweep_control, "get_segments", None)
        segments = getter() if callable(getter) else self.app.sweep.segments
        return max(int(segments), 1)

    def apply_replay_range(self, plan: ReplayRangePlan) -> None:
        """Apply a planned replay sweep to the sweep control.

        The plan is validated against the real :class:`Sweep` grid, so the
        stop it carries never requests a frequency outside the file and a
        positive-span file never collapses into repeated samples.  A span
        the user sets outside the file afterwards is still rejected by the
        worker's preflight with an explicit message.
        """
        if plan.reduced:
            self.app.sweep_control.set_segments(plan.segments)
        self.app.sweep_control.set_start(plan.start)
        self.app.sweep_control.set_end(plan.stop)
        self.app.sweep_control.update_center_span()
        note = plan.describe()
        if note:
            logger.info("Replay default sweep adjusted: %s", note)
        self._report_plan(plan)

    def _report_plan(self, plan: ReplayRangePlan) -> None:
        """Show the planned default next to the loaded replay source."""
        if self._replay is None:
            return
        text = f"{self._replay.name}\n{self._replay.summary()}"
        note = plan.describe()
        if note:
            text += f"\nDefault sweep: {note}"
        self.label_replay_source.setText(text)

    def update_simulation_state(self, *_args) -> None:
        """Keep the transport controls consistent with the mode selector."""
        connecting = self.is_connecting()
        simulated = self.cb_simulated.isChecked()
        replay_selected = simulated and (
            self.selected_scenario() == REPLAY_SCENARIO
        )
        self.row_scenario.setVisible(simulated)
        self.cb_scenario.setEnabled(simulated and not connecting)
        # The replay file picker is only meaningful for the replay scenario,
        # but a loaded source stays visible when switching scenarios so it
        # is never lost silently.
        self.row_replay_file.setVisible(simulated)
        self.btn_replay_file.setEnabled(replay_selected and not connecting)
        self.inp_port.setEnabled(
            not simulated and not connecting and not self.is_scanning()
        )
        self.update_scan_state()
        self.cb_tcp.setEnabled(not simulated and not connecting)
        self.update_connect_btn_state()

    def _connection_failed(
        self, failure: DeviceConnectionError, status: str = "failed"
    ) -> None:
        """Keep the control and the application in a disconnected state."""
        logger.error("Connection attempt failed: %s", failure)
        self.btn_toggle.setText("Connect to device")
        self.btn_toggle.repaint()
        self.update_settings_state(False)
        self.app.showError(str(failure))
        self.connection_finished.emit(status)

    def disconnect_device(self):
        if self.is_connecting():
            # A pending attempt owns the port; cancel it first so the device
            # it may have opened is closed instead of leaked.
            self.cancel_connection()
        if not self.is_vna_connected() and self._live_interface is None:
            # Nothing to release: a spurious ``connected(False)`` would look
            # like a dropped session to every listener.
            logger.debug("Ignoring a disconnect: no device is connected")
            return
        if getattr(self.app.vna, "simulated", False):
            self._disconnect_simulated()
            return
        interface = self._live_interface or self.interface
        with interface.lock:
            logger.info("Closing connection to %s", interface)
            try:
                self.app.vna.disconnect()
            except IOError as exc:
                logger.error("Unable to disconnect from VNA: %s", exc)

            close_quietly(interface)
            self._live_interface = None
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
        # Enable Connect/Disconnect button only if:
        # a) serial was already connected
        # b) serial is disconnected AND inp_port was selected
        # c) TCP is checked and text is present
        # d) a simulated device is selected
        # A pending attempt keeps the button disabled: a second click must
        # not start a second conversation with the device.
        if self.is_connecting():
            self.btn_toggle.setEnabled(False)
            return
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
        return bool(self.app.vna and self.app.vna.connected())

    def update_settings_state(self, was_connected: bool) -> None:
        connecting = self.is_connecting()
        self.update_scan_state()
        self.btn_settings.setEnabled(was_connected and not connecting)
        self.update_connect_btn_state()
