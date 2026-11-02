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
import contextlib
import logging
import os
import threading
from dataclasses import replace
from time import localtime, strftime

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QWidget

from .About import VERSION
from .Analysis.Results import AnalysisKind
from .Calibration import Calibration
from .CalibrationCoverage import coverage_from_calibration
from .Charts import (
    CapacitanceChart,
    CombinedLogMagChart,
    GroupDelayChart,
    InductanceChart,
    LogMagChart,
    MagnitudeChart,
    MagnitudeZChart,
    MagnitudeZSeriesChart,
    MagnitudeZShuntChart,
    PermeabilityChart,
    PhaseChart,
    PolarChart,
    QualityFactorChart,
    RealImaginaryMuChart,
    RealImaginaryZChart,
    RealImaginaryZSeriesChart,
    RealImaginaryZShuntChart,
    SmithChart,
    SParameterChart,
    TDRChart,
    VSWRChart,
)
from .Charts.Chart import Chart
from .Controls.MarkerControl import MarkerControl
from .Controls.SerialControl import SerialControl
from .Controls.SweepControl import SweepControl
from .Controls.TraceControl import TraceControl
from .Defaults import APP_SETTINGS, AppSettings, get_app_config
from .Formatting import format_frequency, format_gain, format_vswr
from .Hardware.Hardware import Interface
from .Hardware.VNA import VNA
from .Limits import LimitCatalogue
from .Marker.Delta import DeltaMarker
from .Marker.Widget import Marker
from .Measurement import (
    S11,
    S21,
    CompletionState,
    compute_calibration_identity,
    snapshot_from_datapoints,
)
from .ReferenceImpedance import DEFAULT_REFERENCE_IMPEDANCE
from .RFTools import corr_att_data
from .Session import (
    SessionError,
    document_snapshots,
    load_session,
    make_document,
    save_session,
)
from .SessionWorkflow import (
    SessionState,
    UnsavedAction,
    decide_unsaved_action,
    discard_recovery,
    find_recovery_for,
    plan_restore,
    recovery_is_due,
    session_path_for,
    summary_lines,
    write_recovery,
)
from .Settings.Bands import BandsModel
from .Settings.Sweep import Sweep, SweepMode
from .SweepState import SweepState
from .SweepWorker import SweepWorker
from .Touchstone import Touchstone
from .Traces import TraceKind
from .Windows import (
    AboutWindow,
    AnalysisWindow,
    AntennaTuningWindow,
    CalibrationWindow,
    DeviceSettingsWindow,
    DisplaySettingsWindow,
    FilesWindow,
    LimitsWindow,
    PresetWindow,
    ReportWindow,
    SessionWindow,
    SweepSettingsWindow,
    TDRWindow,
)
from .Windows.ui import get_window_icon

logger = logging.getLogger(__name__)

WORKING_KILL_TIME_MS = 10 * 1000
#: Minimum gap between live chart repaints (TRACE-04.3).  Roughly one frame
#: on a 60 Hz display; samples arriving faster are merged into the next paint.
LIVE_REDRAW_INTERVAL_MS = 16

#: Trace id used for the measurement currently displayed, when it is not one
#: of the captured traces.
LIVE_TRACE_ID = "trace-live"

#: Prefix of the generated provenance comment on the displayed data.  It is
#: rebuilt from the acquisition snapshot, never accumulated.
_SIMULATED_COMMENT_PREFIX = "SIMULATED DATA"


class Communicate(QObject):
    data_available = QtCore.Signal()


def _displayed_trace_id(document, snapshots) -> str:
    """Which archived trace supplied the displayed measurement.

    Recorded in ``ui`` at save time; older sessions have no entry, in which
    case the first trace is the best available answer.
    """
    recorded = (document.ui or {}).get("displayed_trace", "")
    if recorded and recorded in snapshots:
        return recorded
    return next(iter(snapshots), "")


def _displayed_snapshot_of(document, snapshots):
    return snapshots.get(_displayed_trace_id(document, snapshots))


def _unique_trace_id(base: str, taken) -> str:
    """Return ``base`` or ``base-N``, whichever is free."""
    used = set(taken)
    if base not in used:
        return base
    index = 2
    while f"{base}-{index}" in used:
        index += 1
    return f"{base}-{index}"


def _matches_display(snapshot, s11, s21) -> bool:
    """Whether ``snapshot`` still describes the samples on screen.

    The acquisition's snapshot is authoritative only while the display it
    was taken from is unchanged; capturing twice must never hand back the
    same frozen object.
    """
    if not snapshot.available:
        return False
    for parameter, displayed in ((S11, s11), (S21, s21)):
        points = snapshot.channel(parameter)
        if not displayed:
            if points:
                return False
            continue
        if points is None or len(points) != len(displayed):
            return False
        if any(
            point.freq != other.freq or point.z != other.z
            for point, other in zip(points, displayed, strict=True)
        ):
            return False
    return True


def _analysis_kinds(stored) -> tuple[AnalysisKind, ...]:
    """Normalize stored analysis kinds to enum members.

    Sessions store the kind names; restoring them verbatim left strings that
    later crashed on ``.name`` when the session was saved again.
    """
    kinds = []
    for entry in stored or ():
        if isinstance(entry, AnalysisKind):
            kinds.append(entry)
            continue
        text = str(entry)
        try:
            kinds.append(AnalysisKind(text))
            continue
        except ValueError:
            pass
        # Sessions store the member *name* ("RESONANCE"), not its value.
        try:
            kinds.append(AnalysisKind[text.upper()])
        except KeyError:
            logger.warning("Ignoring unknown analysis kind %r", entry)
    return tuple(kinds)


def _snapshot_identity(snapshot) -> tuple:
    """Content identity of a measurement, used to avoid saving it twice.

    ``MeasurementSnapshot.id`` is freshly generated for every snapshot, so it
    cannot tell whether the displayed measurement is already one of the
    captured traces.  The samples themselves can.
    """
    return tuple(
        (parameter, tuple((dp.freq, dp.re, dp.im) for dp in points))
        for parameter, points in sorted(snapshot.corrected.items())
    )


def _sweep_mode(name: str) -> SweepMode:
    """Map a preset's stored mode name onto :class:`SweepMode`.

    An unknown name falls back to a single sweep rather than raising: a
    preset is data from a file, and a stale label must not break applying it.
    """
    try:
        return SweepMode[str(name).upper()]
    except KeyError:
        logger.warning("Unknown sweep mode %r; using SINGLE", name)
        return SweepMode.SINGLE


class NanoVNASaver(QWidget):
    version = VERSION
    scale_factor = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.communicate = Communicate()
        self.s21att = 0.0
        self.setWindowIcon(get_window_icon())
        # TODO APP_SETTINGS should be used instead app.setting\
        self.settings: AppSettings = APP_SETTINGS
        app_config = self.settings.restore_config()
        self.threadpool = QtCore.QThreadPool()
        self.sweep = Sweep()
        self.worker = SweepWorker(self)

        self.worker.signals.sample_ready.connect(self.applySample)
        self.worker.signals.updated.connect(self.dataUpdated)
        self.worker.signals.finished.connect(self.sweepFinished)
        self.worker.signals.sweep_error.connect(self.showSweepError)
        self.worker.signals.progress.connect(self.sweepProgress)
        self.worker.signals.state_changed.connect(self.sweepStateChanged)
        self.worker.signals.run_finished.connect(self.sweepRunFinished)
        #: Set once a sweep produced at least one measurement.
        self._sweep_had_data = False
        self._last_progress = None
        self._coverage = None
        #: Last fully completed sweep, retained across a cancelled run.
        self._last_complete: tuple[list, list, str] | None = None
        #: The acquisition's own snapshot, which carries calibration identity,
        #: correction state and completion.  Capturing reconstructs nothing
        #: from displayed samples when this is available.
        self._live_snapshot = None
        self._redraw_timer: QtCore.QTimer | None = None
        #: Session workflow state (SESSION-03.1).
        self.session = SessionState(title="Untitled")
        self._restoring = False
        #: One-shot continuation run when the next sweep completes
        #: successfully (SWEEP-04.3).
        self._after_sweep = None
        #: Analysis kinds requested by the last applied preset.
        self._preset_analyses: tuple = ()
        #: Workflow parameters requested by the last applied preset.
        self._preset_parameters: dict = {}
        #: Cached device calibration state; never queried from a redraw.
        self._device_calibration = "unknown"
        #: Reference impedance the calibration and derived displays assume.
        self.reference_impedance = DEFAULT_REFERENCE_IMPEDANCE
        #: Limit sets loaded in this session (LIMIT-03).  Reports freeze the
        #: active set together with its results, so changing the selection
        #: never alters an already built report.
        self.limits: LimitCatalogue = LimitCatalogue()

        self.markers: list[Marker] = []
        self.marker_ref = False

        self.marker_column = QtWidgets.QVBoxLayout()
        self.marker_frame = QtWidgets.QFrame()
        self.marker_column.setContentsMargins(0, 0, 0, 0)
        self.marker_frame.setLayout(self.marker_column)

        self.interface = Interface("serial", "None")
        self.vna: VNA = VNA(self.interface)

        self.calibration: Calibration = Calibration()
        self.sweep_control = SweepControl(self)
        self.marker_control = MarkerControl(self)
        self.serial_control = SerialControl(self)
        self.trace_control = TraceControl(self)
        # Capturing, importing, hiding or reordering a trace must repaint the
        # overlays immediately; without this the charts only caught up on the
        # next sweep (TRACE-02.1).
        self.trace_control.updated.connect(self._trace_control_changed)
        self.serial_control.connected.connect(
            self.sweep_control.update_sweep_btn
        )
        self.serial_control.connected.connect(self._on_connection_changed)

        self.bands: BandsModel = BandsModel()

        self.dataLock = threading.Lock()
        self.data: Touchstone = Touchstone()
        self.ref_data: Touchstone = Touchstone()

        self.sweepSource = ""
        self.referenceSource = ""

        logger.debug("Building user interface")

        self.baseTitle = f"NanoVNA Saver {NanoVNASaver.version}"
        self.updateTitle()
        layout = QtWidgets.QBoxLayout(
            QtWidgets.QBoxLayout.Direction.LeftToRight
        )

        scrollarea = QtWidgets.QScrollArea()
        outer = QtWidgets.QVBoxLayout()
        outer.addWidget(scrollarea)
        self.setLayout(outer)
        scrollarea.setWidgetResizable(True)
        self.resize(app_config.gui.window_width, app_config.gui.window_height)
        scrollarea.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
        )
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
        )
        widget = QWidget()
        widget.setLayout(layout)
        scrollarea.setWidget(widget)

        self.charts = {
            "s11": {
                "capacitance": CapacitanceChart("S11 Serial C"),
                "group_delay": GroupDelayChart("S11 Group Delay"),
                "inductance": InductanceChart("S11 Serial L"),
                "log_mag": LogMagChart("S11 Return Loss"),
                "magnitude": MagnitudeChart("|S11|"),
                "magnitude_z": MagnitudeZChart("S11 |Z|"),
                "permeability": PermeabilityChart(
                    "S11 R/\N{GREEK SMALL LETTER OMEGA} &"
                    " X/\N{GREEK SMALL LETTER OMEGA}"
                ),
                "phase": PhaseChart("S11 Phase"),
                "q_factor": QualityFactorChart("S11 Quality Factor"),
                "real_imag": RealImaginaryZChart("S11 R+jX"),
                "real_imag_mu": RealImaginaryMuChart(
                    "S11 \N{GREEK SMALL LETTER MU}"
                ),
                "smith": SmithChart("S11 Smith Chart"),
                "s_parameter": SParameterChart("S11 Real/Imaginary"),
                "vswr": VSWRChart("S11 VSWR"),
                "sa_dbm": LogMagChart("Signal Analyser dBm"),
            },
            "s21": {
                "group_delay": GroupDelayChart(
                    "S21 Group Delay", reflective=False
                ),
                "log_mag": LogMagChart("S21 Gain"),
                "magnitude": MagnitudeChart("|S21|"),
                "magnitude_z_shunt": MagnitudeZShuntChart("S21 |Z| shunt"),
                "magnitude_z_series": MagnitudeZSeriesChart("S21 |Z| series"),
                "real_imag_shunt": RealImaginaryZShuntChart("S21 R+jX shunt"),
                "real_imag_series": RealImaginaryZSeriesChart(
                    "S21 R+jX series"
                ),
                "phase": PhaseChart("S21 Phase"),
                "polar": PolarChart("S21 Polar Plot"),
                "s_parameter": SParameterChart("S21 Real/Imaginary"),
            },
            "combined": {
                "log_mag": CombinedLogMagChart("S11 & S21 LogMag"),
            },
        }
        self.tdr_chart: TDRChart = TDRChart("TDR")
        self.tdr_mainwindow_chart = TDRChart("TDR")

        # List of all the S11 charts, for selecting
        self.s11charts = list(self.charts["s11"].values())

        # List of all the S21 charts, for selecting
        self.s21charts = list(self.charts["s21"].values())

        # List of all charts that use both S11 and S21
        self.combinedCharts = list(self.charts["combined"].values())

        # List of all charts that can be selected for display
        self.selectable_charts = (
            self.s11charts
            + self.s21charts
            + self.combinedCharts
            + [
                self.tdr_mainwindow_chart,
            ]
        )

        # List of all charts that subscribe to updates (including duplicates!)
        self.subscribing_charts = []
        self.subscribing_charts.extend(self.selectable_charts)
        self.subscribing_charts.append(self.tdr_chart)

        for c in self.subscribing_charts:
            c.popout_requested.connect(self.popoutChart)

        self.charts_layout = QtWidgets.QGridLayout()

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self, self.close)

        ###############################################################
        #  Create main layout
        ###############################################################

        left_column = QtWidgets.QVBoxLayout()
        right_column = QtWidgets.QVBoxLayout()
        right_column.addLayout(self.charts_layout)
        self.marker_frame.setHidden(app_config.gui.markers_hidden)
        chart_widget = QWidget()
        chart_widget.setLayout(right_column)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.marker_frame)
        self.splitter.addWidget(chart_widget)

        self.splitter.restoreState(app_config.gui.splitter_sizes)

        layout.addLayout(left_column)
        layout.addWidget(self.splitter, 2)

        ###############################################################
        #  Windows
        ###############################################################

        self.windows: dict[str, QtWidgets.QDialog] = {
            "about": AboutWindow(self),
            "analysis": AnalysisWindow(self),
            "antenna_tuning": AntennaTuningWindow(self),
            "calibration": CalibrationWindow(self),
            "device_settings": DeviceSettingsWindow(self),
            "file": FilesWindow(self),
            "preset": PresetWindow(self),
            "sweep_settings": SweepSettingsWindow(self),
            "report": ReportWindow(self),
            "limits": LimitsWindow(self),
            "session": SessionWindow(self),
            "setup": DisplaySettingsWindow(self),
            "tdr": TDRWindow(self),
        }

        ###############################################################
        #  Sweep control
        ###############################################################

        left_column.addWidget(self.sweep_control)
        left_column.addWidget(self.trace_control)

        # ###############################################################
        #  Marker control
        ###############################################################

        left_column.addWidget(self.marker_control)

        for c in self.subscribing_charts:
            c.setMarkers(self.markers)
            c.setBands(self.bands)

        self.marker_data_layout = QtWidgets.QVBoxLayout()
        self.marker_data_layout.setContentsMargins(0, 0, 0, 0)

        for m in self.markers:
            self.marker_data_layout.addWidget(m.get_data_layout())

        scroll2 = QtWidgets.QScrollArea()
        scroll2.setWidgetResizable(True)
        scroll2.setVisible(True)

        widget2 = QWidget()
        widget2.setLayout(self.marker_data_layout)
        scroll2.setWidget(widget2)
        self.marker_column.addWidget(scroll2)

        self.marker_coverage_label = QtWidgets.QLabel("")
        self.marker_coverage_label.setWordWrap(True)
        self.marker_data_layout.addWidget(self.marker_coverage_label)

        self.marker_compare_label = QtWidgets.QLabel("")
        self.marker_compare_label.setWordWrap(True)
        self.marker_data_layout.addWidget(self.marker_compare_label)

        # init delta marker (but assume only one marker exists)
        self.delta_marker = DeltaMarker("Delta Marker 2 - Marker 1")
        self.delta_marker_layout = self.delta_marker.get_data_layout()
        self.delta_marker_layout.hide()
        self.marker_column.addWidget(self.delta_marker_layout)

        ###############################################################
        #  Statistics/analysis
        ###############################################################

        s11_control_box = QtWidgets.QGroupBox()
        s11_control_box.setTitle("S11")
        s11_control_layout = QtWidgets.QFormLayout()
        s11_control_layout.setVerticalSpacing(0)
        s11_control_box.setLayout(s11_control_layout)

        self.s11_min_swr_label = QtWidgets.QLabel()
        s11_control_layout.addRow("Min VSWR:", self.s11_min_swr_label)
        self.s11_min_rl_label = QtWidgets.QLabel()
        s11_control_layout.addRow("Return loss:", self.s11_min_rl_label)

        self.marker_column.addWidget(s11_control_box)

        s21_control_box = QtWidgets.QGroupBox()
        s21_control_box.setTitle("S21")
        s21_control_layout = QtWidgets.QFormLayout()
        s21_control_layout.setVerticalSpacing(0)
        s21_control_box.setLayout(s21_control_layout)

        self.s21_min_gain_label = QtWidgets.QLabel()
        s21_control_layout.addRow("Min gain:", self.s21_min_gain_label)

        self.s21_max_gain_label = QtWidgets.QLabel()
        s21_control_layout.addRow("Max gain:", self.s21_max_gain_label)

        self.marker_column.addWidget(s21_control_box)

        # self.marker_column.addStretch(1)

        btn_show_analysis = QtWidgets.QPushButton("Analysis ...")
        btn_show_analysis.setMinimumHeight(20)
        btn_show_analysis.clicked.connect(
            lambda: self.display_window("analysis")
        )
        self.marker_column.addWidget(btn_show_analysis)

        ###############################################################
        # TDR
        ###############################################################

        self.tdr_chart.tdrWindow = self.windows["tdr"]
        self.tdr_mainwindow_chart.tdrWindow = self.windows["tdr"]
        self.windows["tdr"].updated.connect(self.tdr_chart.update)
        self.windows["tdr"].updated.connect(self.tdr_mainwindow_chart.update)

        tdr_control_box = QtWidgets.QGroupBox()
        tdr_control_box.setTitle("TDR")
        tdr_control_layout = QtWidgets.QFormLayout()
        tdr_control_box.setLayout(tdr_control_layout)

        self.tdr_result_label = QtWidgets.QLabel()
        self.tdr_result_label.setMinimumHeight(20)
        tdr_control_layout.addRow(
            "Estimated cable length:", self.tdr_result_label
        )

        self.tdr_button = QtWidgets.QPushButton("Time Domain Reflectometry ...")
        self.tdr_button.setMinimumHeight(20)
        self.tdr_button.clicked.connect(lambda: self.display_window("tdr"))

        tdr_control_layout.addRow(self.tdr_button)

        left_column.addWidget(tdr_control_box)

        ###############################################################
        #  Spacer
        ###############################################################

        left_column.addSpacerItem(
            QtWidgets.QSpacerItem(
                1,
                1,
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
        )

        ###############################################################
        #  Reference control
        ###############################################################

        reference_control_box = QtWidgets.QGroupBox()
        reference_control_box.setTitle("Reference sweep")
        reference_control_layout = QtWidgets.QFormLayout(reference_control_box)

        btn_set_reference = QtWidgets.QPushButton("Set current as reference")
        btn_set_reference.setMinimumHeight(20)
        btn_set_reference.clicked.connect(self.setReference)
        self.btnResetReference = QtWidgets.QPushButton("Reset reference")
        self.btnResetReference.setMinimumHeight(20)
        self.btnResetReference.clicked.connect(self.resetReference)
        self.btnResetReference.setDisabled(True)

        reference_control_layout.addRow(btn_set_reference)
        reference_control_layout.addRow(self.btnResetReference)

        left_column.addWidget(reference_control_box)

        ###############################################################
        #  Serial control
        ###############################################################

        left_column.addWidget(self.serial_control)

        ###############################################################
        #  Calibration
        ###############################################################

        btnOpenCalibrationWindow = QtWidgets.QPushButton("Calibration ...")
        btnOpenCalibrationWindow.setMinimumHeight(20)
        self.calibrationWindow = CalibrationWindow(self)
        btnOpenCalibrationWindow.clicked.connect(
            lambda: self.display_window("calibration")
        )

        ###############################################################
        #  Display setup
        ###############################################################

        btn_display_setup = QtWidgets.QPushButton("Display setup ...")
        btn_display_setup.setMinimumHeight(20)
        btn_display_setup.clicked.connect(lambda: self.display_window("setup"))

        btn_about = QtWidgets.QPushButton("About ...")
        btn_about.setMinimumHeight(20)

        btn_about.clicked.connect(lambda: self.display_window("about"))

        btn_open_file_window = QtWidgets.QPushButton("Files ...")
        btn_open_file_window.setMinimumHeight(20)

        btn_open_file_window.clicked.connect(
            lambda: self.display_window("file")
        )

        btn_session = QtWidgets.QPushButton("Session ...")
        btn_session.setMinimumHeight(20)
        btn_session.setToolTip(
            "New, open, save and save-as for measurement sessions"
        )
        btn_session.clicked.connect(lambda: self.display_window("session"))

        btn_presets = QtWidgets.QPushButton("Presets ...")
        btn_presets.setMinimumHeight(20)
        btn_presets.setToolTip(
            "Apply a built-in workflow preset or manage your own"
        )
        btn_presets.clicked.connect(lambda: self.display_window("preset"))

        btn_limits = QtWidgets.QPushButton("Limits ...")
        btn_limits.setMinimumHeight(20)
        btn_limits.setToolTip(
            "Load measurement limit sets and check pass/fail against the"
            " captured traces"
        )
        btn_limits.clicked.connect(lambda: self.display_window("limits"))

        btn_antenna = QtWidgets.QPushButton("Antenna tuning ...")
        btn_antenna.setMinimumHeight(20)
        btn_antenna.setToolTip(
            "Guided overview sweep, zoom to the measured VSWR minimum,"
            " repeat while tuning, and export the result as an S1P file"
        )
        btn_antenna.clicked.connect(
            lambda: self.display_window("antenna_tuning")
        )

        btn_report = QtWidgets.QPushButton("Report ...")
        btn_report.setMinimumHeight(20)
        btn_report.setToolTip(
            "Preview a measurement report and export it as HTML or CSV"
        )
        btn_report.clicked.connect(lambda: self.display_window("report"))

        button_grid = QtWidgets.QGridLayout()
        button_grid.addWidget(btn_open_file_window, 0, 0)
        button_grid.addWidget(btnOpenCalibrationWindow, 0, 1)
        button_grid.addWidget(btn_session, 1, 0)
        button_grid.addWidget(btn_presets, 1, 1)
        button_grid.addWidget(btn_report, 2, 0)
        button_grid.addWidget(btn_limits, 2, 1)
        # The guided antenna workflow gets its own full-width row: every
        # control must stay visible in a distinct cell (limits already owns
        # the cell antenna was drafted into).
        button_grid.addWidget(btn_antenna, 3, 0, 1, 2)
        button_grid.addWidget(btn_display_setup, 4, 0)
        button_grid.addWidget(btn_about, 4, 1)
        left_column.addLayout(button_grid)

        logger.debug("Finished building interface")

    def auto_connect(
        self,
    ):  # connect if there is exactly one detected serial device
        if self.serial_control.inp_port.count() == 1:
            self.serial_control.connect_device()

    def _sweep_control(self, start: bool = True) -> None:
        self.sweep_control.progress_bar.setValue(0 if start else 100)
        self.sweep_control.btn_start.setDisabled(start)
        self.sweep_control.btn_stop.setDisabled(not start)
        self.sweep_control.toggle_settings(start)

    def sweep_start(self):
        # Run the device data update
        if not self.vna.connected():
            return
        if self.worker.isRunning():
            logger.debug("Ignoring sweep start while a sweep is active")
            return
        self._sweep_had_data = False
        self._sweep_control(start=True)

        for m in self.markers:
            m.resetLabels()
        self.s11_min_rl_label.setText("")
        self.s11_min_swr_label.setText("")
        self.s21_min_gain_label.setText("")
        self.s21_max_gain_label.setText("")
        self.tdr_result_label.setText("")

        logger.debug("Starting worker thread")
        self.worker.start()
        # TODO: Rewrite to make worker a qrunnable with worker signals
        # https://www.pythonguis.com/tutorials/multithreading-pyqt6-applications-qthreadpool/
        # self.threadpool.start(self.worker)

    def saveData(self, data, data21, source=None):
        with self.dataLock:
            self.data.s11 = data
            self.data.s21 = data21
            if self.s21att > 0:
                self.data.s21 = corr_att_data(self.data.s21, self.s21att)
        if source is not None:
            self.sweepSource = source
        else:
            time = strftime("%Y-%m-%d %H:%M:%S", localtime())
            name = self.sweep.properties.name or "nanovna"
            self.sweepSource = f"{name}_{time}"

    def adopt_reference_impedance(self, value) -> None:
        """Adopt the reference impedance a loaded file was measured at.

        Charts, markers and analysis all derive impedance from a reflection
        coefficient, so they must be told the reference those coefficients
        belong to.  A value that cannot be used leaves the current one alone
        rather than silently reverting to 50 ohm mid-session.
        """
        from .ReferenceImpedance import (
            ReferenceImpedanceError,
            require_reference_impedance,
        )

        try:
            self.reference_impedance = require_reference_impedance(value)
        except ReferenceImpedanceError:
            logger.warning(
                "Ignoring unusable reference impedance %r; keeping %s ohm",
                value,
                self.reference_impedance,
            )

    def markerUpdated(self, marker: Marker):
        with self.dataLock:
            marker.findLocation(self.data.s11)
            marker.resetLabels()
            marker.updateLabels(self.data.s11, self.data.s21)
            for c in self.subscribing_charts:
                c.update()
        if not self.delta_marker_layout.isHidden():
            m1 = self.markers[0]
            m2 = None
            if self.marker_ref:
                if self.ref_data:
                    m2 = Marker("Reference")
                    m2.location = self.markers[0].location
                    m2.set_reference_impedance(self.reference_impedance)
                    m2.resetLabels()
                    m2.updateLabels(self.ref_data.s11, self.ref_data.s21)
                else:
                    logger.warning("No reference data for marker")

            elif Marker.count() >= 2:
                m2 = self.markers[1]

            if m2 is None:
                logger.error("No data for delta, missing marker or reference")
            else:
                self.delta_marker.set_reference_impedance(
                    self.reference_impedance
                )
                self.delta_marker.set_markers(m1, m2)
                self.delta_marker.resetLabels()
                with contextlib.suppress(IndexError):
                    self.delta_marker.updateLabels()
        self.refresh_marker_comparison()
        self.refresh_marker_coverage()

    def refresh_marker_comparison(self) -> None:
        """Report the trace comparison at the active marker (TRACE-02.4).

        Marker values always come from the live sweep, so the comparison is
        labelled with the marker frequency and the compared span instead of
        leaving the reader to guess which curve a delta belongs to.
        """
        marker = self.markers[0] if self.markers else None
        if marker is None or marker.location < 0 or not marker.freq:
            self.marker_compare_label.setText("")
            return
        result = self.trace_control.current_comparison()
        delta = self.trace_control.comparison_at(marker.freq)
        if result is None or delta is None:
            self.marker_compare_label.setText(
                "Comparison: pick a trace and a reference in Trace control"
            )
            return
        self.marker_compare_label.setText(
            f"Comparison @ {delta.freq / 1e6:.6g} MHz"
            f" ({result.span_label()}):"
            f" {result.source_trace} - {result.reference_trace}"
            f" = {delta.scalar:.4g} {result.unit}"
        )

    def dataUpdated(self):
        with self.dataLock:
            s11 = self.data.s11[:]
            s21 = self.data.s21[:]

        for m in self.markers:
            m.set_reference_impedance(self.reference_impedance)
            m.resetLabels()
            m.updateLabels(s11, s21)

        # Every chart inherits the reference impedance the data was measured
        # at (RF-02): a 75 ohm file must not be drawn as if it were 50 ohm.
        for c in self.s11charts:
            c.setData(s11, self.reference_impedance)

        for c in self.s21charts:
            c.setData(s21, self.reference_impedance)

        for c in self.combinedCharts:
            c.setCombinedData(s11, s21)

        self.sweep_control.progress_bar.setValue(int(self.worker.percentage))
        self.windows["tdr"].updateTDR()

        if s11:
            min_vswr = min(s11, key=lambda data: data.vswr)
            self.s11_min_swr_label.setText(
                f"{format_vswr(min_vswr.vswr)} @"
                f" {format_frequency(min_vswr.freq)}"
            )
            self.s11_min_rl_label.setText(format_gain(min_vswr.gain))
        else:
            self.s11_min_swr_label.setText("")
            self.s11_min_rl_label.setText("")

        if s21:
            min_gain = min(s21, key=lambda data: data.gain)
            max_gain = max(s21, key=lambda data: data.gain)
            self.s21_min_gain_label.setText(
                f"{format_gain(min_gain.gain)}"
                f" @ {format_frequency(min_gain.freq)}"
            )
            self.s21_max_gain_label.setText(
                f"{format_gain(max_gain.gain)}"
                f" @ {format_frequency(max_gain.freq)}"
            )
        else:
            self.s21_min_gain_label.setText("")
            self.s21_max_gain_label.setText("")

        self.updateTitle()
        self.refreshCoverage()
        self.refresh_trace_charts()
        self.communicate.data_available.emit()

    def sweepFinished(self):
        self._sweep_control(start=False)

        for marker in self.markers:
            marker.frequencyInput.textEdited.emit(marker.frequencyInput.text())

    def applySample(self, sample) -> None:
        """Adopt one published sample on the GUI thread (SWEEP-02.2).

        Data is adopted immediately so nothing is lost; the redraw is
        coalesced because a segmented sweep can publish samples far faster
        than a chart can repaint (TRACE-04.3).
        """
        if sample.sweep_id != self.worker.sweep_id:
            logger.debug("Ignoring sample from stale sweep %s", sample.sweep_id)
            return
        snapshot = sample.snapshot
        if snapshot is not None:
            # The display follows the data on screen: a live sweep replaces
            # whatever reference an opened file had set (RF-02).
            self.adopt_reference_impedance(snapshot.reference_impedance)
        self.saveData(list(sample.s11), list(sample.s21))
        self._live_snapshot = snapshot
        self.mark_session_changed("sweep")
        if snapshot is not None:
            self.data.simulated = snapshot.simulated
            self.data.simulated_scenario = (
                snapshot.provenance.simulator_scenario or ""
            )
            # Exactly one generated provenance comment: each published sample
            # refreshes it instead of stacking copies.
            self.data.comments = [
                comment
                for comment in self.data.comments
                if not comment.startswith(_SIMULATED_COMMENT_PREFIX)
            ]
            if snapshot.simulated:
                scenario = snapshot.provenance.simulator_scenario or "unknown"
                comment = f"{_SIMULATED_COMMENT_PREFIX} (scenario {scenario})"
                # A replay names its source file so a saved export never
                # looks like an independent measurement (REPLAY-01.4).
                note = " ".join(snapshot.provenance.notes.split())
                if note:
                    comment += f": {note}"
                self.data.comments.append(comment)
        self._sweep_had_data = True
        # Provenance is only known now, so refresh the title after it is set:
        # the [SIMULATED: ...] marker must never lag behind the data.
        self.updateTitle()
        self._schedule_redraw()

    def _schedule_redraw(self) -> None:
        """Coalesce live redraws into one repaint (TRACE-04.3).

        Full-resolution data is kept for analysis and export; only the
        repaints are merged, never the samples.
        """
        if self._redraw_timer is None:
            self._redraw_timer = QtCore.QTimer(self)
            self._redraw_timer.setSingleShot(True)
            self._redraw_timer.setInterval(LIVE_REDRAW_INTERVAL_MS)
            self._redraw_timer.timeout.connect(self._flush_redraw)
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    def _flush_redraw(self) -> None:
        self.dataUpdated()

    def sweepProgress(self, progress) -> None:
        """Render structured progress for the active sweep (SWEEP-04.1)."""
        if progress.sweep_id != self.worker.sweep_id:
            logger.debug(
                "Ignoring progress for stale sweep %s", progress.sweep_id
            )
            return
        self._last_progress = progress
        self.sweep_control.show_progress(progress)

    def sweepStateChanged(self, state) -> None:
        logger.debug("Sweep %s state %s", self.worker.sweep_id, state.value)

    def sweepRunFinished(self, terminal) -> None:
        """Handle the single terminal outcome of a sweep run (SWEEP-01.4)."""
        self._sweep_control(start=False)
        if terminal is SweepState.COMPLETED:
            self._remember_complete_sweep()
            self.sweep_control.show_idle("Complete")
            self._continue_after_sweep()
        elif terminal is SweepState.CANCELLED:
            self._after_sweep = None
            # A cancelled sweep is never presented as a measurement
            # (SWEEP-03.3): either the previous complete sweep is restored or
            # the display is emptied and labelled.
            restored = self._restore_last_complete()
            self.sweep_control.show_idle(
                "Cancelled - showing the last complete sweep"
                if restored
                else "Cancelled - no complete sweep to show"
            )
        else:
            self.sweep_control.show_idle("Failed")
            self._after_sweep = None  # a failure ends the chain
        self._sweep_had_data = False

    def run_after_sweep(self, callback) -> None:
        """Run ``callback`` once after the next successful sweep.

        Replaces any pending continuation, so a repeated analysis cannot
        queue up multiple follow-up sweeps (SWEEP-04.4).  A cancelled or
        failed sweep clears the continuation instead of running it.
        """
        self._after_sweep = callback

    def cancel_after_sweep(self) -> None:
        self._after_sweep = None

    def _continue_after_sweep(self) -> None:
        callback = self._after_sweep
        self._after_sweep = None
        if callback is None:
            return
        logger.debug("Continuing analysis sequence")
        callback()

    def _remember_complete_sweep(self) -> None:
        if self._live_snapshot is not None:
            # Completion lives on the provenance, and a sweep that reached
            # its final segment is complete even when the last sample was
            # reported mid-grid.
            self._live_snapshot = replace(
                self._live_snapshot,
                provenance=replace(
                    self._live_snapshot.provenance,
                    completion=CompletionState.COMPLETE,
                ),
            )
        with self.dataLock:
            self._last_complete = (
                list(self.data.s11),
                list(self.data.s21),
                self.sweepSource,
            )

    def _restore_last_complete(self) -> bool:
        """Replace any partial sweep with the last complete one."""
        with self.dataLock:
            if self._last_complete is None:
                self.data.s11 = []
                self.data.s21 = []
                empty = True
            else:
                s11, s21, source = self._last_complete
                self.data.s11 = list(s11)
                self.data.s21 = list(s21)
                self.sweepSource = source
                empty = False
        self.dataUpdated()
        if empty:
            for marker in self.markers:
                marker.resetLabels()
        return not empty

    def refreshCoverage(self) -> None:
        """Recompute and publish calibration coverage (CAL-02.3).

        Coverage is derived from the calibration samples and the frequencies
        actually displayed, so it refreshes whenever the span, the calibration
        or the displayed data changes.
        """
        with self.dataLock:
            frequencies = [dp.freq for dp in self.data.s11] or [
                dp.freq for dp in self.data.s21
            ]
        parameters = []
        with self.dataLock:
            if self.data.s11:
                parameters.append("S11")
            if self.data.s21:
                parameters.append("S21")
        if not frequencies:
            coverage = None
        else:
            coverage = coverage_from_calibration(
                self.calibration,
                frequencies,
                parameters=tuple(parameters) or ("S11",),
            )
        self._coverage = coverage
        for chart in self.subscribing_charts:
            chart.setCoverage(coverage)
        self.sweep_control.show_calibration_status(
            coverage, self._correction_source_label()
        )
        self.refresh_marker_coverage()

    def refresh_marker_coverage(self) -> None:
        """State coverage at the active marker (CAL-02.3).

        The span-level annotation on a chart says nothing about the exact
        frequency under a marker, which is where a user reads a value.
        """
        coverage = getattr(self, "_coverage", None)
        marker = self.markers[0] if self.markers else None
        label = getattr(self, "marker_coverage_label", None)
        if label is None:
            return
        if coverage is None or marker is None or marker.location < 0:
            label.setText("")
            return
        state = coverage.state_at(marker.freq)
        label.setText(
            f"Calibration @ {marker.freq / 1e6:.6g} MHz:"
            f" {state.value.replace('_', ' ')}"
        )

    def _on_connection_changed(self, _connected: bool) -> None:
        """Refresh device facts once per connection change, not per redraw."""
        self.refresh_device_calibration_state()
        self.refreshCoverage()

    def refresh_device_calibration_state(self) -> str:
        """Ask the device once where correction happens, and cache it.

        ``getCalibration`` is a serial command: issuing it from
        ``refreshCoverage`` - which runs on every redraw - blocked the GUI
        thread for the round trip.  The answer is refreshed only when the
        device connection changes or on an explicit action.
        """
        vna = getattr(self, "vna", None)
        if vna is None or not getattr(vna, "connected", lambda: False)():
            self._device_calibration = "unknown"
            return self._device_calibration
        try:
            reported = vna.getCalibration()
        except Exception:  # device refused to answer
            logger.debug("Device calibration state unavailable", exc_info=True)
            self._device_calibration = "unknown"
            return self._device_calibration
        if not reported:
            self._device_calibration = "unknown"
            return self._device_calibration
        text = str(reported).lower()
        if any(word in text for word in ("none", "off", "disabled", "empty")):
            self._device_calibration = "application"
        else:
            self._device_calibration = "device"
        return self._device_calibration

    def _correction_source_label(self) -> str:
        """Where correction happens; "unknown" without device metadata.

        CAL-02.4: the application can only state what it knows.  A device
        that reports its own calibration state is distinguished from one
        that says nothing.  Never performs I/O: the cached answer is set when
        the connection opens or the user asks for a refresh.
        """
        return getattr(self, "_device_calibration", "unknown")

    def setReference(self, s11=None, s21=None, source=None):
        if not s11:
            with self.dataLock:
                s11 = self.data.s11[:]
                s21 = self.data.s21[:]

        self.ref_data.s11 = s11
        for c in self.s11charts:
            c.setReference(s11)

        self.ref_data.s21 = s21
        for c in self.s21charts:
            c.setReference(s21)

        for c in self.combinedCharts:
            c.setCombinedReference(s11, s21)

        self.btnResetReference.setDisabled(False)

        self.referenceSource = source or self.sweepSource
        self.updateTitle()

    def updateTitle(self):
        insert = "("
        if self.sweepSource != "":
            insert += (
                f"Sweep: {self.sweepSource} @ {len(self.data.s11)} points"
                f"{', ' if self.referenceSource else ''}"
            )
        if self.referenceSource != "":
            insert += (
                f"Reference: {self.referenceSource} @"
                f" {len(self.ref_data.s11)} points"
            )
        insert += ")"
        if getattr(self.data, "simulated", False):
            scenario = self.data.simulated_scenario or "unknown"
            insert = f"[SIMULATED: {scenario}] " + insert
        title = f"{self._title_prefix()} {insert or ''}"
        self.setWindowTitle(title.strip())

    def resetReference(self):
        self.mark_session_changed("reference")
        self.ref_data = Touchstone()
        self.referenceSource = ""
        self.updateTitle()
        for c in self.subscribing_charts:
            c.resetReference()
        self.btnResetReference.setDisabled(True)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(1100, 950)

    def display_window(self, name):
        self.windows[name].show()
        QtWidgets.QApplication.setActiveWindow(self.windows[name])

    def showError(self, text):
        QtWidgets.QMessageBox.warning(self, "Error", text)

    def showSweepError(self):
        self.showError(self.worker.error_message)
        with contextlib.suppress(IOError):
            self.vna.flushSerialBuffers()  # Remove any left-over data
            self.vna.reconnect()  # try reconnection
        self.sweepFinished()

    def popoutChart(self, chart: Chart):
        logger.debug("Requested popout for chart: %s", chart.name)
        new_chart = self.copyChart(chart)
        new_chart.isPopout = True
        new_chart.show()
        new_chart.setWindowTitle(new_chart.name)
        new_chart.setWindowIcon(get_window_icon())

    def copyChart(self, chart: Chart):
        new_chart = chart.copy()
        self.subscribing_charts.append(new_chart)
        if chart in self.s11charts:
            self.s11charts.append(new_chart)
        if chart in self.s21charts:
            self.s21charts.append(new_chart)
        if chart in self.combinedCharts:
            self.combinedCharts.append(new_chart)
        new_chart.popout_requested.connect(self.popoutChart)
        return new_chart

    # ------------------------------------------------------------------
    # Presets (PRESET-02.5)
    # ------------------------------------------------------------------
    def _apply_point_count(self, points: int) -> bool:
        """Select the acquisition point count for the current device.

        ``SweepControl.update_sweep`` derives the sweep's point count from
        ``vna.datapoints``, so a preset or session asking for 401 points was
        silently reverted to the device default (101) unless the device
        selection itself changed.  The device is told first; a count the
        device does not offer is logged and ignored rather than sent.
        """
        if points <= 0:
            return False
        supported = tuple(getattr(self.vna, "valid_datapoints", ()) or ())
        if supported and points not in supported:
            logger.warning(
                "Device does not support %d points; keeping %d",
                points,
                self.vna.datapoints,
            )
            return False
        self.vna.datapoints = int(points)
        self.sweep.set_points(int(points))
        self.sweep_control.update_sweep()
        return True

    def apply_preset_application(self, application) -> None:
        """Apply a validated preset as one settings change.

        The whole sweep configuration is applied - span, segmentation, point
        count, averaging and mode - not just the span, so the applied setup is
        the one the preset describes.  The preset never connects,
        disconnects or starts a sweep: the user still has to ask for one.
        """
        sweep = application.sweep
        self._restoring = True
        try:
            self.sweep_control.set_start(sweep.start)
            self.sweep_control.set_end(sweep.end)
            self.sweep_control.set_segments(sweep.segments)
            mode = _sweep_mode(sweep.mode)
            self._apply_point_count(sweep.points)
            self.sweep.set_mode(mode)
            self.sweep.set_averages(
                max(int(sweep.averages), 1), sweep.truncates
            )
            self.sweep.set_logarithmic(sweep.logarithmic)
            sweep_window = self.windows.get("sweep")
            if sweep_window is not None and hasattr(sweep_window, "refresh"):
                sweep_window.refresh()
        finally:
            self._restoring = False
        # The analysis window drives the legacy per-algorithm analyses, whose
        # selectors do not map onto AnalysisKind, so there is no clean way to
        # preselect them.  The requested kinds are recorded and reported
        # instead of being silently dropped.
        self._preset_analyses = tuple(application.analysis_kinds)
        self._preset_parameters = dict(application.parameters)
        self.apply_preset_charts(application.preset.charts)
        self.apply_preset_parameters(application.parameters)
        self.sweep_control.update_center_span()
        self.mark_session_changed("sweep")

    def apply_sweep_configuration(
        self,
        *,
        start: int,
        end: int,
        segments: int,
        points: int,
        logarithmic: bool = False,
    ) -> bool:
        """Apply a complete sweep configuration as one change.

        Used by guided workflows that must not leave the sweep half-configured
        (a new span with the old point count is a different measurement).
        The device point count is applied first, because
        ``SweepControl.update_sweep`` derives the sweep's points from it.
        Nothing is started, connected or swept here.

        Returns whether the resulting sweep matches the request.
        """
        self._restoring = True
        try:
            self.sweep_control.set_start(start)
            self.sweep_control.set_end(end)
            self.sweep_control.set_segments(max(int(segments), 1))
            self._apply_point_count(int(points))
            # A guided workflow measures one span at a time; a continuous or
            # averaging mode left over from elsewhere would change what the
            # reported minimum means.
            self.sweep.set_mode(SweepMode.SINGLE)
            self.sweep.set_logarithmic(bool(logarithmic))
            self.sweep_control.update_center_span()
            self.sweep_control.update_step_size()
        finally:
            self._restoring = False
        sweep = self.sweep
        applied = (sweep.start, sweep.end, sweep.points, sweep.segments)
        return applied == (int(start), int(end), int(points), int(segments))

    def apply_preset_charts(self, names) -> bool:
        """Show the preset's charts in the main grid.

        A preset describes a workflow, so the plots that workflow needs are
        part of it; leaving the layout untouched made an applied preset look
        like it had done nothing.
        """
        wanted = [str(name) for name in names or () if str(name)]
        if not wanted:
            return False
        by_name = {chart.name: chart for chart in self.selectable_charts}
        cells = [
            (index, by_name[name])
            for index, name in enumerate(wanted)
            if name in by_name
        ]
        if not cells:
            return False
        # Removing while walking the layout shifts the indices and skips
        # every other item, so collect first.
        existing = [
            item.widget()
            for item in (
                self.charts_layout.itemAt(index)
                for index in range(self.charts_layout.count())
            )
            if item is not None and item.widget() is not None
        ]
        for widget in existing:
            self.charts_layout.removeWidget(widget)
            widget.hide()
        for index, chart in cells:
            row, column = divmod(index, 2)
            target = chart
            if chart.parent() is not None:
                target = self.copyChart(chart)
            self.charts_layout.addWidget(target, row, column)
            target.show()
        return True

    def apply_preset_parameters(self, parameters) -> dict:
        """Apply the workflow parameters a preset asks for.

        Recorded in the session so the value that shaped a report can be
        recovered; the values that drive analysis are applied where a live
        control exists.
        """
        applied = {}
        for key, value in dict(parameters or {}).items():
            if key == "marker_count":
                applied[key] = self._apply_marker_count(int(value))
            elif key == "vswr_limit":
                self._apply_vswr_limit(float(value))
                applied[key] = float(value)
            elif key == "velocity_factor":
                self.windows["tdr"].set_velocity_factor(float(value))
                applied[key] = float(value)
            else:
                applied[key] = value
        return applied

    def _apply_marker_count(self, count: int) -> int:
        """Record how many markers the workflow wants.

        Marker widgets are created once, from ``chart.marker_count`` at start
        up, and have no per-marker visibility switch; showing a different
        number therefore cannot be done without rebuilding the marker panel.
        The figure is applied to the configuration for the next start and
        reported, rather than silently ignored.
        """
        config = get_app_config().chart
        target = max(1, min(int(count), config.MAX_MARKER_COUNT))
        config.marker_count = target
        return target

    def _apply_vswr_limit(self, limit: float) -> None:
        from .Analysis.VSWRAnalysis import VSWRAnalysis

        VSWRAnalysis.vswr_limit_value = limit
        analyses = self.windows["analysis"].analysis_list
        for index in range(analyses.count()):
            analysis = analyses.itemData(index)
            if isinstance(analysis, VSWRAnalysis):
                analysis.input_vswr_limit.setValue(limit)

    # ------------------------------------------------------------------
    # Session workflow (SESSION-03)
    # ------------------------------------------------------------------
    def mark_session_changed(self, change: str) -> None:
        """Record a meaningful experiment change."""
        if self._restoring:
            return
        if self.session.mark_changed(change):
            self.update_session_title()
            self._refresh_session_labels()

    def update_session_title(self) -> None:
        """Refresh the title from the session state.

        The simulation marker and sweep description are part of the title,
        so the session name is a prefix rather than a replacement: losing
        ``[SIMULATED: ...]`` here would make simulated data look physical.
        """
        session_window = self.windows.get("session")
        if session_window is not None:
            session_window.refresh()
        self.updateTitle()

    def _title_prefix(self) -> str:
        name = self.session.display_name
        return f"{self.baseTitle} - {name}" if name else self.baseTitle

    def _refresh_session_labels(self) -> None:
        return None

    def current_snapshot(self):
        """Freeze the *displayed* measurement, or ``None`` when there is none.

        The acquisition's own snapshot is authoritative because it carries the
        calibration identity, the correction state and the completion state;
        rebuilding one from displayed samples would silently label a
        calibrated, partial sweep as complete and uncorrected.  Only when no
        acquisition or restored snapshot describes the displayed data is
        a snapshot derived from the displayed samples.
        """
        with self.dataLock:
            s11 = list(self.data.s11)
            s21 = list(self.data.s21)
        live = getattr(self, "_live_snapshot", None)
        if live is not None and _matches_display(live, s11, s21):
            return live
        if not s11 and not s21:
            return None
        return snapshot_from_datapoints(
            s11 or None,
            s21 or None,
            name=self.sweepSource or "measurement",
            simulated=getattr(self.data, "simulated", False),
            simulator_scenario=getattr(self.data, "simulated_scenario", ""),
            app_version=NanoVNASaver.version,
            source=self.sweepSource,
        )

    def _trace_control_changed(self, _control=None) -> None:
        """Repaint trace overlays after a trace action (TRACE-02.1)."""
        self.refresh_trace_charts()
        for chart in self.subscribing_charts:
            chart.update()

    def refresh_trace_charts(self) -> None:
        """Push the visible traces into every multi-trace chart."""
        for parameter, charts in (
            ("S11", self.s11charts),
            ("S21", self.s21charts),
        ):
            series = self.trace_control.trace_series(parameter)
            for chart in charts:
                if hasattr(chart, "setTraces"):
                    chart.setTraces(series)

    def current_session_document(self, title: str = ""):
        """Freeze the current experiment into a session document."""
        snapshots, trace_ids = self._current_snapshots()
        collection = self.trace_control.collection
        return make_document(
            snapshots,
            trace_ids=trace_ids,
            trace_names={
                trace_id: trace.name
                for trace_id in trace_ids
                if (trace := collection.get(trace_id)) is not None
            },
            title=title or self.session.title or "Untitled",
            app_version=NanoVNASaver.version,
            selected_trace=(
                collection.reference.id
                if collection.reference is not None
                else (trace_ids[0] if trace_ids else "")
            ),
            markers=self.marker_state(),
            sweep=self.sweep_info_dict(),
            charts=self.chart_state_dict(),
            analysis={
                "kinds": [kind.name for kind in self._preset_analyses],
                "parameters": dict(self._preset_parameters),
            },
            calibration=self.calibration_state(),
            calibration_files=self.calibration_file_state(),
            ui=self.ui_state(trace_ids),
        )

    def ui_state(self, trace_ids=None) -> dict:
        """View state that is part of the experiment, not of the settings.

        Records which trace supplied the displayed measurement so reopening
        shows it instead of the oldest trace in the archive.
        """
        ids = list(trace_ids if trace_ids is not None else [])
        if not ids:
            return {"displayed_trace": ""}
        return {"displayed_trace": self._last_displayed_id}

    def marker_state(self) -> list[dict]:
        """Serializable marker positions (SESSION-01.2)."""
        return [
            {
                "index": index,
                "frequency": int(marker.freq),
                "color": marker.color.name(),
            }
            for index, marker in enumerate(self.markers)
            if marker.freq
        ]

    def calibration_state(self) -> dict:
        """Identity and provenance of the loaded calibration."""
        calibration = self.calibration
        try:
            identity = compute_calibration_identity(calibration)
        except Exception:  # an incomplete calibration has no identity
            logger.debug("No calibration identity available", exc_info=True)
            identity = ""
        dataset = getattr(calibration, "dataset", None)
        frequencies = dataset.frequencies() if dataset is not None else []
        return {
            "identity": identity,
            "source": getattr(calibration, "source", "") or "",
            "notes": "\n".join(getattr(calibration, "notes", []) or []),
            "valid_one_port": bool(
                getattr(calibration, "isValid1Port", lambda: False)()
            ),
            "valid_two_port": bool(
                getattr(calibration, "isValid2Port", lambda: False)()
            ),
            "start": frequencies[0] if frequencies else 0,
            "end": frequencies[-1] if frequencies else 0,
            "points": len(frequencies),
        }

    def calibration_file_state(self) -> dict:
        """Verbatim calibration text, so a session can restore corrections."""
        calibration = self.calibration
        if not getattr(calibration, "isValid1Port", lambda: False)():
            return {}
        dataset = getattr(calibration, "dataset", None)
        if dataset is None:
            return {}
        try:
            return {"loaded.cal": str(dataset)}
        except Exception:  # unrepresentable calibration stays out of the file
            logger.warning("Unable to serialize the calibration", exc_info=True)
            return {}

    def _displayed_snapshot(self):
        """Snapshot of what is currently on screen, or ``None``.

        The acquisition's own snapshot is preferred: it carries calibration
        identity, correction state and completion, which a reconstruction
        from displayed samples cannot recover.
        """
        with self.dataLock:
            s11 = list(self.data.s11)
            s21 = list(self.data.s21)
        live = getattr(self, "_live_snapshot", None)
        if live is not None and _matches_display(live, s11, s21):
            return live
        if not s11 and not s21:
            return None
        return snapshot_from_datapoints(
            s11 or None,
            s21 or None,
            name=self.sweepSource or "measurement",
            simulated=getattr(self.data, "simulated", False),
            simulator_scenario=getattr(self.data, "simulated_scenario", ""),
            app_version=NanoVNASaver.version,
            source=self.sweepSource,
        )

    def _current_snapshots(self):
        """Every snapshot the session must preserve, never a partial sweep.

        Captured traces come first, in display order (SESSION-01.2), followed
        by the measurement currently on screen.  The displayed measurement is
        included even when captured traces exist: a sweep taken after the last
        capture is part of the experiment, and dropping it silently lost work.
        It is not duplicated when it is already one of the captured traces.
        """
        collection = self.trace_control.collection
        snapshots = []
        trace_ids = []
        seen = {}
        for trace_id in collection.ordered_ids():
            trace = collection.get(trace_id)
            if trace is None:
                continue
            snapshots.append(trace.snapshot)
            trace_ids.append(trace_id)
            seen.setdefault(_snapshot_identity(trace.snapshot), trace_id)

        self._last_displayed_id = ""
        displayed = self._displayed_snapshot()
        if displayed is not None:
            identity = _snapshot_identity(displayed)
            if identity in seen:
                self._last_displayed_id = seen[identity]
            else:
                snapshots.append(displayed)
                # The archive is a mapping keyed by trace id, so a reopened
                # session that already holds a restored ``trace-live`` would
                # have its newest measurement overwrite that entry - or be
                # overwritten by it.  Use a fresh id instead.
                trace_id = _unique_trace_id(LIVE_TRACE_ID, trace_ids)
                trace_ids.append(trace_id)
                self._last_displayed_id = trace_id
        return snapshots, trace_ids

    def sweep_info_dict(self) -> dict:
        sweep = self.sweep
        return {
            "start": sweep.start,
            "end": sweep.end,
            "points": sweep.points,
            "segments": sweep.segments,
            "mode": sweep.properties.mode.name,
            "averages": sweep.properties.averages[0],
            "truncates": sweep.properties.averages[1],
            "logarithmic": bool(sweep.properties.logarithmic),
        }

    def chart_state_dict(self) -> dict:
        """The charts actually on screen, by name, in grid order.

        Recording the catalogue instead described what *could* be shown, so
        reopening a session never restored the layout.
        """
        cells = []
        for index in range(self.charts_layout.count()):
            item = self.charts_layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is None or widget.isHidden():
                continue
            row, column, _row_span, _column_span = (
                self.charts_layout.getItemPosition(index)
            )
            cells.append({"row": row, "column": column, "name": widget.name})
        cells.sort(key=lambda cell: (cell["row"], cell["column"]))
        return {"cells": cells}

    def apply_chart_state(self, state) -> bool:
        """Show the saved chart layout; return whether anything was applied."""
        if "cells" not in (state or {}):
            return False
        cells = state["cells"] or []
        by_name = {chart.name: chart for chart in self.selectable_charts}
        # The saved layout replaces the current layout, including cells that
        # no longer exist. Keep the widgets alive for future chart selection.
        while self.charts_layout.count():
            item = self.charts_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
        for cell in cells:
            name = str(cell.get("name", ""))
            chart = by_name.get(name)
            if chart is None:
                logger.warning("Session chart %r is not available", name)
                continue
            row = int(cell.get("row", 0))
            column = int(cell.get("column", 0))
            # Repeated chart names need independent widgets. Charts owned by
            # another window (notably TDR) must remain in that window as well.
            parent = chart.parentWidget()
            if self.charts_layout.indexOf(chart) > -1 or (
                parent is not None
                and parent is not self.charts_layout.parentWidget()
            ):
                chart = self.copyChart(chart)
            self.charts_layout.addWidget(chart, row, column)
            chart.show()
        return True

    def action_new_session(self) -> None:
        """Start an empty experiment (SESSION-03).

        Everything that belongs to the previous experiment is cleared: the
        displayed data, the reference, the captured traces, the
        calibration-derived state and the remembered complete sweep.  Leaving
        any of it behind both leaked the old experiment into the new session
        and immediately marked the fresh session dirty.
        """
        if not self._confirm_discard():
            return
        discard_recovery(self.session)
        self.session.reset(title="Untitled")
        self._restoring = True
        try:
            self.data = Touchstone()
            self.ref_data = Touchstone()
            self.resetReference()
            self.trace_control.collection.clear()
            self.trace_control.refresh()
            self._last_complete = None
            self._live_snapshot = None
            self.reference_impedance = DEFAULT_REFERENCE_IMPEDANCE
            self._coverage = None
            self._after_sweep = None
            self._preset_analyses = ()
            self._preset_parameters = {}
            self.sweepSource = ""
        finally:
            self._restoring = False
        self.refreshCoverage()
        self.dataUpdated()
        self.update_session_title()

    def _confirm_discard(self) -> bool:
        """Ask about unsaved changes; True means proceed."""
        if not self.session.dirty:
            return True
        answer = QtWidgets.QMessageBox.question(
            self,
            "Unsaved session",
            "This session has unsaved changes. Save before continuing?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        mapping = {
            QtWidgets.QMessageBox.StandardButton.Save: "save",
            QtWidgets.QMessageBox.StandardButton.Discard: "discard",
            QtWidgets.QMessageBox.StandardButton.Cancel: "cancel",
        }
        decision = decide_unsaved_action(
            dirty=True,
            can_save=True,
            answer=mapping.get(answer, "cancel"),
        )
        if decision == UnsavedAction.CANCEL.value:
            return False
        if decision == UnsavedAction.SAVE_THEN_PROCEED.value:
            return self.action_save_session()
        return True

    def action_save_session(self) -> bool:
        filename = self.session.filename or self._ask_session_filename()
        if not filename:
            return False
        return self._write_session(filename)

    def action_save_session_as(self) -> bool:
        filename = self._ask_session_filename()
        if not filename:
            return False
        return self._write_session(filename)

    def _ask_session_filename(self) -> str:
        suggestion = session_path_for(
            self.session.title, self.settings.value("SaveDirectory", "", str)
        )
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save session",
            suggestion,
            "NanoVNA session (*.nvsession);;All files (*.*)",
        )
        return filename

    def _write_session(self, filename: str) -> bool:
        document = self.current_session_document(self.session.title)
        try:
            save_session(document, filename)
        except SessionError as exc:
            self.showError(f"Unable to save the session: {exc}")
            return False
        self.session.mark_saved(filename)
        discard_recovery(self.session)
        window = self.windows.get("session")
        if window is not None:
            window.remember(filename)
        self.update_session_title()
        return True

    def action_open_session(self) -> bool:
        if not self._confirm_discard():
            return False
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open session",
            "",
            "NanoVNA session (*.nvsession);;All files (*.*)",
        )
        if not filename:
            return False
        return self.open_session_file(filename)

    def open_session_file(self, filename: str) -> bool:
        """Restore a session without touching hardware (SESSION-03.4)."""
        try:
            document = load_session(filename)
        except SessionError as exc:
            self.showError(f"Unable to open the session: {exc}")
            return False
        outcome = plan_restore(document, NanoVNASaver.version)
        self._apply_session(document)
        self.session.filename = filename
        self.session.title = document.title or os.path.basename(filename)
        self.session.mark_saved(filename)
        self.update_session_title()
        if outcome.warnings:
            logger.info("Restored with notes: %s", list(summary_lines(outcome)))
        return True

    def _apply_session(self, document) -> None:
        """Apply a restored document; never connects or sweeps."""
        self._restoring = True
        try:
            snapshots = document_snapshots(document)
            # The session records which trace was displayed; reopening must
            # show that measurement, not whichever trace happens to be first
            # in the archive.
            first = _displayed_snapshot_of(document, snapshots)
            self._live_snapshot = first
            self.reference_impedance = (
                first.reference_impedance
                if first is not None
                else DEFAULT_REFERENCE_IMPEDANCE
            )
            with self.dataLock:
                if first is None:
                    self.data.s11 = []
                    self.data.s21 = []
                else:
                    self.data.s11 = list(first.channel("S11") or ())
                    self.data.s21 = list(first.channel("S21") or ())
                    self.data.simulated = first.simulated
                    self.data.simulated_scenario = (
                        first.provenance.simulator_scenario or ""
                    )
            self._restore_traces(document, snapshots)
            self._restore_markers(document)
            self._restore_sweep(document)
            self._restore_calibration(document)
            self.apply_chart_state(document.charts)
            self._preset_analyses = _analysis_kinds(
                document.analysis.get("kinds")
            )
            self._preset_parameters = dict(
                document.analysis.get("parameters") or {}
            )
            self.apply_preset_parameters(self._preset_parameters)
            self.sweepSource = document.title or ""
            recovery = find_recovery_for(self.session.filename)
            if recovery:
                logger.info("A recovery snapshot also exists: %s", recovery)
        finally:
            self._restoring = False
        self.dataUpdated()

    def _restore_markers(self, document) -> None:
        """Restore marker frequencies (SESSION-01.2)."""
        for marker in self.markers:
            marker.freq = 0
        for entry in document.markers:
            index = int(entry.get("index", -1))
            if not 0 <= index < len(self.markers):
                continue
            frequency = int(entry.get("frequency", 0))
            if frequency <= 0:
                continue
            marker = self.markers[index]
            marker.frequencyInput.setText(str(frequency))
            marker.freq = frequency
            color = entry.get("color")
            if color:
                with contextlib.suppress(ValueError):
                    marker.setColor(QtGui.QColor(color))
        for marker in self.markers:
            marker.findLocation(self.data.s11)

    def _restore_sweep(self, document) -> None:
        """Restore the saved sweep settings without touching a device."""
        sweep = document.sweep
        if not sweep:
            return
        start = int(sweep.get("start", 0))
        end = int(sweep.get("end", 0))
        if start <= 0 or end <= start:
            return
        self.sweep_control.set_start(start)
        self.sweep_control.set_end(end)
        self.sweep_control.set_segments(max(int(sweep.get("segments", 1)), 1))
        self._apply_point_count(int(sweep.get("points", 0)))
        self.sweep.set_mode(_sweep_mode(str(sweep.get("mode", "SINGLE"))))
        self.sweep.set_averages(
            max(int(sweep.get("averages", 1)), 1),
            int(sweep.get("truncates", 0)),
        )
        self.sweep.set_logarithmic(bool(sweep.get("logarithmic", False)))
        self.sweep_control.update_center_span()

    def _restore_calibration(self, document) -> None:
        """Reload the session's own calibration text, if it carried one."""
        text = (document.calibration_files or {}).get("loaded.cal", "")
        if not text:
            return
        calibration = Calibration()
        try:
            calibration.dataset = calibration.dataset.from_str(text)
            calibration.notes = calibration.dataset.notes.splitlines()
            calibration.calc_corrections()
        except (ValueError, KeyError, IndexError):
            logger.warning(
                "Session calibration could not be rebuilt", exc_info=True
            )
            return
        self.calibration = calibration
        expected = document.calibration.get("identity", "")
        actual = compute_calibration_identity(calibration)
        if expected and actual and expected != actual:
            logger.warning(
                "Session calibration identity changed: %s -> %s",
                expected,
                actual,
            )

    def _restore_traces(self, document, snapshots) -> None:
        """Rebuild the trace list from a document (SESSION-03.4).

        The trace that supplied the displayed data is restored as the live
        trace; the rest become frozen traces so nothing is duplicated on the
        next capture.
        """
        collection = self.trace_control.collection
        collection.clear()
        displayed = _displayed_trace_id(document, snapshots)
        for entry in document.traces:
            trace_id = entry.get("trace_id", "")
            snapshot = snapshots.get(trace_id)
            if snapshot is None:
                continue
            name = entry.get("trace_name") or entry.get("name") or trace_id
            kind = TraceKind.LIVE if trace_id == displayed else TraceKind.FROZEN
            collection.add(snapshot, name=name, kind=kind, trace_id=trace_id)
            color = entry.get("color")
            if color:
                collection.set_color(trace_id, color)
        selected = document.selected_trace
        if selected:
            collection.set_reference(selected)
        self.trace_control.refresh()

    def maybe_write_recovery(self) -> None:
        """Write a periodic recovery snapshot for unsaved work."""
        if not recovery_is_due(self.session):
            return
        try:
            target = write_recovery(
                self.current_session_document(self.session.title), self.session
            )
            logger.debug("Wrote recovery snapshot %s", target)
        except SessionError as exc:
            logger.warning("Unable to write recovery snapshot: %s", exc)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if not self._confirm_discard():
            a0.ignore()
            return
        self.worker.quit()
        self.worker.wait(WORKING_KILL_TIME_MS)
        for marker in self.markers:
            marker.update_settings()
        self.settings.sync()
        self.bands.saveSettings()
        self.threadpool.waitForDone(2500)

        app_config = get_app_config()
        # Persist how many markers this window owns.  ``Marker.count()`` is a
        # process-wide instance counter, so using it here made the saved
        # count grow with every window ever constructed and, once written,
        # balloon on the next startup.
        app_config.chart.marker_count = len(self.markers)
        app_config.gui.window_width = self.width()
        app_config.gui.window_height = self.height()
        app_config.gui.splitter_sizes = self.splitter.saveState()

        self.sweep_control.store_settings()

        self.settings.store_config()

        # Dosconnect connected devices and release serial port
        self.serial_control.disconnect_device()

        a0.accept()

    def changeFont(self, font: QtGui.QFont) -> None:
        qf_new = QtGui.QFontMetricsF(font)
        normal_font = QtGui.QFont(font)
        normal_font.setPointSize(8)
        qf_normal = QtGui.QFontMetricsF(normal_font)
        # Characters we would normally display
        standard_string = "0.123456789 0.123456789 MHz \N{OHM SIGN}"
        new_width = qf_new.horizontalAdvance(standard_string)
        old_width = qf_normal.horizontalAdvance(standard_string)
        self.scale_factor = new_width / old_width
        logger.debug(
            "New font width: %f, normal font: %f, factor: %f",
            new_width,
            old_width,
            self.scale_factor,
        )
        # TODO: Update all the fixed widths to account for the scaling
        for m in self.markers:
            m.get_data_layout().setFont(font)
            m.setScale(self.scale_factor)

    def update_sweep_title(self):
        for c in self.subscribing_charts:
            c.setSweepTitle(self.sweep.properties.name)
