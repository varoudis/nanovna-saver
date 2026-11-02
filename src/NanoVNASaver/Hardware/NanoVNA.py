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

import serial
from PySide6.QtGui import QPixmap

from ..utils import Version
from .CombinedScan import (
    PATH_TRUSTED,
    CombinedSweep,
    one_shot_read,
    parse_combined_reply,
)
from .Convert import get_argb32_pixmap
from .ScanCapability import ScanPreferences
from .Serial import Interface, drain_serial
from .VNA import VNA

logger = logging.getLogger(__name__)


class NanoVNA(VNA):
    name = "NanoVNA"
    screenwidth = 320
    screenheight = 240

    def __init__(self, iface: Interface):
        super().__init__(iface)
        self.sweep_method = "sweep"
        self.init_features()
        logger.debug("Setting initial start,stop")
        self.start, self.stop = self._get_running_frequencies()
        self.sweep_max_freq_hz = 300e6
        self._sweepdata: list[tuple[complex, complex]] = []
        #: Channels produced by an accepted optimised read (PERF-02).
        self._optimised_cache: dict[str, tuple[complex, ...]] = {}
        self._optimised_path = PATH_TRUSTED
        #: ``(start, stop, points)`` the cache was filled for.  Subclasses
        #: (SV4401A, SV6301A) override ``setSweep``, so the cache cannot rely
        #: on that call alone to notice a changed grid.
        self._optimised_sweep: tuple[int, int, int] | None = None

    def _get_running_frequencies(self):
        logger.debug("Reading values: frequencies")
        try:
            frequencies = super().readValues("frequencies")
            return int(frequencies[0].real), int(frequencies[-1].real)
        except serial.SerialException as e:
            logger.warning("%s reading frequencies", e)
            logger.info("falling back to generic")

        return VNA._get_running_frequencies(self)

    def _capture_data(self) -> bytes:
        timeout = self.serial.timeout
        with self.serial.lock:
            drain_serial(self.serial)
            self.serial.write("capture\r".encode("ascii"))
            self.serial.readline()
            self.serial.timeout = 4
            image_data = self.serial.read(
                self.screenwidth * self.screenheight * 2
            )
            self.serial.timeout = timeout
        self.serial.timeout = timeout
        return image_data

    def getScreenshot(self) -> QPixmap:
        logger.debug("Capturing screenshot...")
        if not self.connected():
            return QPixmap()
        try:
            logger.debug("Captured screenshot")
            return get_argb32_pixmap(
                self._capture_data(), self.screenwidth, self.screenheight
            )
        except serial.SerialException as exc:
            logger.exception("Exception while capturing screenshot: %s", exc)
        return QPixmap()

    def resetSweep(self, start: int, stop: int):
        list(self.exec_command(f"sweep {start} {stop} {self.datapoints}"))
        list(self.exec_command("resume"))

    def setSweep(self, start, stop):
        self.start = start
        self.stop = stop
        # A new sweep invalidates any samples cached by an optimised read.
        self._optimised_cache = {}
        if self.sweep_method == "sweep":
            list(self.exec_command(f"sweep {start} {stop} {self.datapoints}"))
        elif self.sweep_method == "scan":
            list(self.exec_command(f"scan {start} {stop} {self.datapoints}"))

    def init_features(self) -> None:
        super().init_features()
        if self.version >= Version.parse("0.7.1"):
            logger.debug("Using scan mask command.")
            self.features.add("Scan mask command")
            self.sweep_method = "scan_mask"
        elif self.version >= Version.parse("0.2.0"):
            logger.debug("Using new scan command.")
            self.features.add("Scan command")
            self.sweep_method = "scan"

    def read_frequencies(self) -> list[int]:
        logger.debug("readFrequencies: %s", self.sweep_method)
        if self.sweep_method != "scan_mask":
            return super().read_frequencies()
        return [
            int(line)
            for line in self.exec_command(
                f"scan {self.start} {self.stop} {self.datapoints} 0b001"
            )
        ]

    def readValues(self, value) -> list[complex]:
        if self.sweep_method != "scan_mask":
            return super().readValues(value)
        logger.debug("readValue with scan mask (%s)", value)
        # Actually grab the data only when requesting channel 0.
        # The hardware will return all channels which we will store.
        if value == "data 0":
            self._sweepdata = []
            for line in self.exec_command(
                f"scan {self.start} {self.stop} {self.datapoints} 0b110"
            ):
                d = list(map(float, line.split()))
                self._sweepdata.append(
                    (complex(d[0], d[1]), complex(d[2], d[3]))
                )
        # A ``scan_mask`` read is this device's trusted path, not an
        # optimisation: it has always returned both channels per sweep.
        self.last_read_path = PATH_TRUSTED
        if value == "data 1":
            return [x[1] for x in self._sweepdata]
        # default to data 0
        return [x[0] for x in self._sweepdata]

    # -- optimised reads (PERF-02) -------------------------------------
    def read_values_optimised(
        self, value: str, preferences: ScanPreferences | None = None
    ) -> list[complex]:
        """Read a channel, trying one masked text scan when opted in.

        Only a driver whose firmware was *not* proven to answer
        ``scan ... 0b110`` ever needs this: a ``scan_mask`` device already
        reads one command for both channels in :meth:`readValues`, and is
        left exactly as it is.  The masked attempt is validated whole by
        :func:`parse_combined_reply`; on any doubt the sweep is re-read
        channel by channel and the optimised reply is discarded.
        """
        self.last_read_path = PATH_TRUSTED
        requested = self.scan_capability.requested(
            preferences if preferences is not None else ScanPreferences()
        )
        if requested != "combined-text":
            # Nothing was opted in for this driver: the trusted read runs
            # untouched, exactly as it did before PERF-02.
            self._optimised_path = PATH_TRUSTED
            self._optimised_cache = {}
            return self.readValues(value)
        # A subclass may override ``setSweep`` without calling this class,
        # so the cache is keyed on the sweep it was read for rather than
        # cleared by a call that might never happen.
        identity = (self.start, self.stop, self.datapoints)
        if identity != self._optimised_sweep:
            self._optimised_sweep = identity
            self._optimised_cache = {}
            self._optimised_path = PATH_TRUSTED
        if value != "data 0":
            cached = self._optimised_cache.get(value)
            if cached is not None:
                # A combined read already produced this channel; it is
                # never re-read, and never replaced by zeros.
                self.last_read_path = self._optimised_path
                return list(cached)
            return self.readValues(value)
        command = f"scan {self.start} {self.stop} {self.datapoints} 0b110"
        channels = ("data 0", "data 1")

        def attempt() -> CombinedSweep:
            return parse_combined_reply(
                self.exec_command(command),
                points=self.datapoints,
                channels=channels,
                frequencies=self.read_frequencies(),
                start=self.start,
                stop=self.stop,
            )

        outcome = one_shot_read(attempt, self._trusted_sweep)
        self.last_read_path = outcome.path
        self._optimised_path = outcome.path
        self._optimised_cache = {}
        if outcome.sweep is not None:
            # Either the validated optimised sweep, or the trusted re-read
            # that replaced it; both are complete, so the second channel
            # never triggers another transaction.
            for name in outcome.sweep.channels:
                self._optimised_cache[name] = outcome.sweep.channel(name)
            if value in self._optimised_cache:
                return list(self._optimised_cache[value])
        return self.readValues(value)

    def _trusted_sweep(self) -> CombinedSweep:
        """Re-read both channels the ordinary way, for ``one_shot_read``."""
        grid = tuple(int(value) for value in VNA.read_frequencies(self))
        return CombinedSweep(
            frequencies=grid,
            s11=tuple(VNA.readValues(self, "data 0")),
            s21=tuple(VNA.readValues(self, "data 1")),
        )
