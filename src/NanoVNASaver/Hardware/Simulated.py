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
"""Deterministic simulated device (SIM-01).

``SimulatedDevice`` implements the same command surface that
:class:`NanoVNASaver.Hardware.VNA.VNA` and
:class:`NanoVNASaver.SweepWorker.SweepWorker` use on real hardware, so the
normal sweep path runs unchanged with no serial port or discovery.

The simulated *interface* is a small transport that speaks the device command
language (``sweep``, ``frequencies``, ``data 0``, ``data 1``, ``version``,
``help``, ``bandwidth``, ``sn``, ``info``, ``cal``, ``reset``) exactly like
the ASCII protocol devices, without any bytes crossing a pipe.  That keeps
command parsing exercised and makes fault injection possible at the same
seam (SIM-03).
"""

from __future__ import annotations

import logging
import math
import random
import threading
from dataclasses import dataclass, field
from enum import Enum
from time import sleep
from typing import Sequence

from ..utils import Version
from .ReplayScenario import ReplayChannelError
from .SimulatedScenarios import (
    DEFAULT_SCENARIOS,
    REFERENCE_IMPEDANCE,
    SCENARIO_TYPES,
    Scenario,
    ScenarioResult,
    build_scenario,
)

logger = logging.getLogger(__name__)

SIMULATED_COMMENT = "SIMULATED"


class FaultMode(Enum):
    """Deterministic faults injected at the transport seam (SIM-03.1)."""

    NONE = "none"
    TIMEOUT = "timeout"
    """Every read blocks until the configured timeout elapses."""

    DISCONNECT = "disconnect"
    """The transport reports itself closed on the next access."""

    EMPTY_SAMPLES = "empty_samples"
    """``data`` commands return no sample lines."""

    MALFORMED_SAMPLES = "malformed_samples"
    """``data`` commands return non-numeric tokens."""

    MISMATCHED_LENGTHS = "mismatched_lengths"
    """``data 1`` returns fewer values than ``frequencies``."""

    NON_FINITE = "non_finite"
    """``data`` commands return nan/inf values."""

    FAIL_DURING_AVERAGE = "fail_during_average"
    """Succeeds for N data reads, then behaves as ``EMPTY_SAMPLES``."""

    SLOW = "slow"
    """Adds ``slow_delay`` seconds to every data read."""


@dataclass
class FaultConfig:
    """Fault injection configuration; kept separate from demo settings."""

    mode: FaultMode = FaultMode.NONE
    fail_after_samples: int = 1
    slow_delay: float = 0.5
    nonzero_channel: str = "data 1"


@dataclass
class SimulatedTransport:
    """In-process transport used instead of a serial port.

    Only the attributes that ``VNA``/``SweepWorker`` touch are implemented:
    ``is_open``, ``lock``, ``timeout``, ``open``, ``close``, ``write``,
    ``readline``, ``read``, ``reset_input_buffer``,
    ``reset_output_buffer``.
    """

    scenario: Scenario = field(
        default_factory=lambda: DEFAULT_SCENARIOS["antenna"]
    )
    fault: FaultConfig = field(default_factory=FaultConfig)
    seed: int | None = None
    noise_floor: float = 0.0
    latency: float = 0.0
    sweep_max_freq_hz: float = 6.3e9

    is_open: bool = False
    timeout: float = 0.05
    lock: threading.RLock = field(default_factory=threading.RLock)
    comment: str = SIMULATED_COMMENT
    type: str = "simulated"

    start: int = 3_600_000
    stop: int = 30_000_000
    points: int = 101
    bandwidth: int = 1000
    datapoints: int = 101

    _reads_served: int = 0
    _last_frequency_grid: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Transport API
    # ------------------------------------------------------------------
    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def reset_input_buffer(self) -> None:
        return None

    def reset_output_buffer(self) -> None:
        return None

    def read(self, _size: int = 1) -> bytes:
        return b""

    def write(self, data: bytes) -> int:
        return len(data)

    def readline(self) -> bytes:
        return b""

    def __str__(self) -> str:
        return f"<simulated {self.comment}>"


class SimulatedDevice:
    """Command-level simulation of an ASCII-protocol NanoVNA."""

    name = "Simulated VNA"
    SANE_VERSION = "sim-1.0.0"

    def __init__(  # noqa: PLR0913
        self,
        scenario: Scenario | str = "antenna",
        *,
        sweep_max_freq_hz: float = 6.3e9,
        seed: int | None = None,
        noise_floor: float = 0.0,
        latency: float = 0.0,
        fault: FaultConfig | None = None,
        scenario_kwargs: dict | None = None,
        sweep_start: int | None = None,
        sweep_stop: int | None = None,
    ) -> None:
        if isinstance(scenario, str):
            scenario = build_scenario(scenario, **(scenario_kwargs or {}))
        self.scenario: Scenario = scenario
        self.fault = fault or FaultConfig()
        self.seed = seed
        self.noise_floor = noise_floor
        self.latency = latency
        self.sweep_max_freq_hz = sweep_max_freq_hz
        self._random = random.Random(seed)
        self._reads_served = 0
        self._closed = False
        self.start = 3_600_000 if sweep_start is None else int(sweep_start)
        default_stop = max(30_000_000, self.start)
        self.stop = default_stop if sweep_stop is None else int(sweep_stop)
        self.points = 101
        self.bandwidth = 1000

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def scenario_key(self) -> str:
        """Canonical scenario name as offered by the demonstration mode."""
        key = getattr(self.scenario, "scenario_key", "")
        if isinstance(key, str) and key:
            return key
        for key, factory in SCENARIO_TYPES.items():
            if isinstance(self.scenario, factory):
                return key
        return type(self.scenario).__name__

    @property
    def scenario_name(self) -> str:
        """Human-readable scenario identity for display and provenance."""
        name = getattr(self.scenario, "display_name", "")
        if isinstance(name, str) and name:
            return name
        return type(self.scenario).__name__

    @property
    def supported_channels(self) -> tuple[str, ...]:
        """Channels this scenario can honestly produce.

        Analytic scenarios always describe a complete S11/S21 response.  A
        replay scenario exposes only the channels its file provides, so a
        one-port file never turns into a fabricated S21.
        """
        channels = getattr(self.scenario, "supported_channels", None)
        if channels:
            return tuple(channels)
        return ("data 0", "data 1")

    @property
    def provenance_note(self) -> str:
        """Extra provenance text for snapshots (replay source, if any)."""
        note = getattr(self.scenario, "provenance_text", None)
        return note() if callable(note) else ""

    def preflight(self, start: int, stop: int) -> str | None:
        """Scenario-specific sweep preflight; ``None`` means valid."""
        check = getattr(self.scenario, "preflight", None)
        return check(start, stop) if callable(check) else None

    @property
    def simulated(self) -> bool:
        return True

    @property
    def reference_impedance(self) -> float:
        """Reference impedance of everything this device emits.

        A replay scenario is only accepted when the file matches this
        system impedance (see ``ReplayScenario``), so no data is ever
        relabelled as if it had been measured at a different impedance.
        """
        return REFERENCE_IMPEDANCE

    def version(self) -> Version:
        return Version.parse(self.SANE_VERSION)

    def features(self) -> set[str]:
        return {"Customizable data points", "Bandwidth"}

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    def set_sweep(
        self, start: int, stop: int, points: int | None = None
    ) -> None:
        if self._closed:
            raise IOError("simulated device is closed")
        if points is None:
            points = self.points
        if points < 1:
            raise ValueError(f"illegal point count {points}")
        self.start, self.stop, self.points = int(start), int(stop), int(points)

    def frequency_grid(self) -> list[int]:
        if self.points == 1:
            return [self.start]
        step = (self.stop - self.start) / (self.points - 1)
        return [round(self.start + i * step) for i in range(self.points)]

    def sample(self, freq: int) -> ScenarioResult:
        if self.fault.mode is FaultMode.TIMEOUT:
            sleep(self.timeout)
            raise TimeoutError("simulated transport timeout")
        if self.fault.mode is FaultMode.SLOW:
            sleep(self.fault.slow_delay)

        result = self.scenario(float(freq))
        if self.noise_floor:
            result = ScenarioResult(
                self._add_noise(result.s11), self._add_noise(result.s21)
            )
        return result

    def _add_noise(self, value: complex) -> complex:
        sigma = self.noise_floor
        return complex(
            value.real + self._random.gauss(0.0, sigma),
            value.imag + self._random.gauss(0.0, sigma),
        )

    def read_channel(self, channel: str) -> list[complex]:
        """Return one channel of the current sweep, as ``readValues`` does."""
        if self._closed or self.fault.mode is FaultMode.DISCONNECT:
            raise IOError("simulated transport disconnected")

        if channel not in self.supported_channels:
            # No sample is invented for a channel this scenario cannot
            # produce: a one-port replay has no S21 at all (REPLAY-01.3).
            raise ReplayChannelError(
                f"the {self.scenario_name} scenario provides no"
                f" {'S11' if channel == 'data 0' else 'S21'} channel"
            )

        mode = self.fault.mode
        if (
            mode is FaultMode.FAIL_DURING_AVERAGE
            and self._reads_served >= self.fault.fail_after_samples
        ):
            raise IOError("simulated failure during averaging")
        self._reads_served += 1

        grid = self.frequency_grid()
        samples = [self.sample(freq) for freq in grid]
        values = [s.s11 if channel == "data 0" else s.s21 for s in samples]

        if mode is FaultMode.EMPTY_SAMPLES:
            return []
        if mode is FaultMode.MALFORMED_SAMPLES:
            raise ValueError("malformed sample data from simulated device")
        if (
            mode is FaultMode.MISMATCHED_LENGTHS
            and channel == self.fault.nonzero_channel
        ):
            return values[:-1]
        if mode is FaultMode.NON_FINITE:
            changed = list(values)
            changed[0] = complex(math.nan, math.inf)
            return changed
        return values

    def read_frequencies(self) -> list[int]:
        return self.frequency_grid()

    def read_calibration(self) -> str:
        return "simulated calibration state: none"

    def read_info(self) -> str:
        return (
            f"Simulated device {self.SANE_VERSION}\n"
            f"scenario: {self.scenario_name}\n"
            f"reference impedance: {REFERENCE_IMPEDANCE:g} ohm"
        )

    def read_serial_number(self) -> str:
        return "SIM-0000-0000"

    def read_bandwidths(self) -> list[int]:
        return [10, 33, 50, 100, 200, 250, 333, 500, 1000, 2000, 4000]

    def set_bandwidth(self, bandwidth: int) -> None:
        self.bandwidth = int(bandwidth)

    def close(self) -> None:
        self._closed = True

    # ------------------------------------------------------------------
    # Data acquisition with sweep-range restoration
    # ------------------------------------------------------------------
    def acquire(
        self,
        start: int,
        stop: int,
        points: int,
        channels: Sequence[str] = ("data 0", "data 1"),
    ) -> tuple[list[int], list[list[complex]]]:
        """Read a segment; mirrors real drivers' ``setSweep``+reads."""
        previous = (self.start, self.stop, self.points)
        self.set_sweep(start, stop, points)
        try:
            freqs = self.read_frequencies()
            return freqs, [self.read_channel(name) for name in channels]
        finally:
            self.start, self.stop, self.points = previous


class SimulatedVNA:
    """Drop-in replacement for ``VNA`` that wraps ``SimulatedDevice``.

    Deliberately a thin adapter: it exposes the exact members that the rest
    of the application uses, and delegates all physics to the device.
    """

    def __init__(  # noqa: PLR0913
        self,
        device: SimulatedDevice | None = None,
        *,
        scenario: Scenario | str = "antenna",
        seed: int | None = None,
        noise_floor: float = 0.0,
        latency: float = 0.0,
        fault: FaultConfig | None = None,
        scenario_kwargs: dict | None = None,
        sweep_start: int | None = None,
        sweep_stop: int | None = None,
    ) -> None:
        self.device = device or SimulatedDevice(
            scenario,
            seed=seed,
            noise_floor=noise_floor,
            latency=latency,
            fault=fault,
            scenario_kwargs=scenario_kwargs,
            sweep_start=sweep_start,
            sweep_stop=sweep_stop,
        )
        self.name = "Simulated VNA"
        self.valid_datapoints = (101, 51, 11, 201, 401)
        self.sweep_points_max = 1024
        self.sweep_points_min = 11
        self.sweep_max_freq_hz = self.device.sweep_max_freq_hz
        self.validateInput = True
        self.datapoints = 101
        self.bandwidth = 1000
        self.SN = "SIM-0000-0000"
        self.hardware_revision: Version | str = self.device.version()
        self.txPowerRanges: list = []
        self._connected = True

    # -- lifecycle -----------------------------------------------------
    def connected(self) -> bool:
        return self._connected and not self.device._closed

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def reconnect(self) -> None:
        self._connected = True

    def flushSerialBuffers(self) -> None:
        return None

    # -- device surface ------------------------------------------------
    @property
    def version(self) -> Version:
        return self.device.version()

    def get_features(self) -> set[str]:
        return self.device.features()

    def get_bandwidths(self) -> list[int]:
        return self.device.read_bandwidths()

    def set_bandwidth(self, bandwidth: int) -> None:
        self.device.set_bandwidth(bandwidth)
        self.bandwidth = bandwidth

    def getCalibration(self) -> str:
        return self.device.read_calibration()

    def getScreenshot(self):
        from PySide6 import QtGui

        return QtGui.QPixmap()

    def readFirmware(self) -> str:
        return self.device.read_info()

    def read_frequencies(self) -> list[int]:
        return self.device.read_frequencies()

    def readValues(self, value: str) -> list[complex]:
        return self.device.read_channel(value)

    def setSweep(self, start, stop) -> None:
        self.device.set_sweep(start, stop, self.datapoints)

    def resetSweep(self, start, stop) -> None:
        self.device.set_sweep(start, stop, self.datapoints)

    def setTXPower(self, freq_range, power_desc) -> None:
        return None

    # -- provenance ----------------------------------------------------
    @property
    def simulated(self) -> bool:
        return True

    @property
    def scenario_key(self) -> str:
        return self.device.scenario_key

    @property
    def scenario_name(self) -> str:
        return self.device.scenario_name

    @property
    def reference_impedance(self) -> float:
        return self.device.reference_impedance

    @property
    def supported_channels(self) -> tuple[str, ...]:
        return self.device.supported_channels

    @property
    def provenance_note(self) -> str:
        return self.device.provenance_note

    def preflight(self, start: int, stop: int) -> str | None:
        return self.device.preflight(start, stop)


def make_simulated_vna(  # noqa: PLR0913
    scenario: Scenario | str = "antenna",
    *,
    seed: int | None = None,
    noise_floor: float = 0.0,
    latency: float = 0.0,
    fault: FaultConfig | None = None,
    scenario_kwargs: dict | None = None,
    sweep_start: int | None = None,
    sweep_stop: int | None = None,
) -> SimulatedVNA:
    """Factory used by the connection control for demonstration mode."""
    return SimulatedVNA(
        scenario=scenario,
        seed=seed,
        noise_floor=noise_floor,
        latency=latency,
        fault=fault,
        scenario_kwargs=scenario_kwargs,
        sweep_start=sweep_start,
        sweep_stop=sweep_stop,
    )
