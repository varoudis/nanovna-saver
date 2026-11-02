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
"""Per-driver acquisition-protocol matrix (PERF-02).

The matrix records what each driver in :mod:`NanoVNASaver.Hardware`
*actually does* when it reads a sweep, so an optimisation can never be
applied to a device that does not implement it:

* ``path`` - the command family the driver uses: ``sweep`` (old text
  firmware), ``scan`` (text firmware with the ``scan`` command),
  ``scan_mask`` (text firmware that accepts a channel mask and answers a
  whole sweep with one command) or ``fifo`` (the register/FIFO binary
  transport of the V2 family).
* ``combines_channels`` - one command already returns every channel.  Such
  a driver needs no optimisation: its reads are combined by construction.
* ``binary_transport`` - the driver already speaks a binary protocol (the
  V2/LiteVNA64 FIFO records), as opposed to ASCII lines.
* ``combined_text_opt_in`` - a masked ``scan`` reply is *possible* on this
  driver's command set but has not been observed on its firmware, so it
  may only be used after an explicit opt-in.
* ``binary_scan`` - whether the unverified ``scan_bin`` binary read of
  :mod:`NanoVNASaver.Hardware.CombinedScan` may be attempted.  ``verified``
  would mean hardware evidence exists; nothing in this repository has any,
  so only ``opt-in`` entries exist.

Everything here is immutable and Qt-free: it is data, not behaviour.  The
live driver state that can differ from the class default (a firmware that
does or does not answer ``scan ... 0b110``) is folded in by
:func:`scan_capability_for`, which is what devices expose through
``capability().scan``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

#: ``sweep start stop points``: text firmware without the ``scan`` command.
PATH_SWEEP = "sweep"
#: ``scan start stop points``: text firmware, one command per channel.
PATH_SCAN = "scan"
#: ``scan start stop points 0b110``: one command returns every channel.
PATH_SCAN_MASK = "scan_mask"
#: The V2 family register/FIFO binary protocol.
PATH_FIFO = "fifo"

#: Every acquisition path this matrix knows.
PATHS = frozenset({PATH_SWEEP, PATH_SCAN, PATH_SCAN_MASK, PATH_FIFO})

#: The driver cannot answer ``scan_bin`` at all.
BINARY_UNSUPPORTED = "unsupported"
#: The driver's command set may answer ``scan_bin``; unverified on hardware,
#: so it stays behind an explicit opt-in that defaults OFF.
BINARY_OPT_IN = "opt-in"
#: The driver's ``scan_bin`` framing was confirmed on a physical device.
BINARY_VERIFIED = "verified"

#: Every ``scan_bin`` state.
BINARY_STATES = frozenset({BINARY_UNSUPPORTED, BINARY_OPT_IN, BINARY_VERIFIED})


@dataclass(frozen=True)
class ScanPreferences:
    """Explicit opt-in for the unverified read paths (PERF-02).

    Both switches default to ``False``, which is the trusted configuration:
    one command per channel, exactly as before PERF-02.  Enabling either
    switch is a deliberate act, because the corresponding reply format has
    never been observed on the affected firmware.
    """

    #: Try ``scan ... 0b110`` on a driver that only ever issued ``scan``.
    combined_text: bool = False
    #: Try the unverified ``scan_bin`` binary frame.
    binary_scan: bool = False

    @property
    def enabled(self) -> bool:
        return self.combined_text or self.binary_scan

    def describe(self) -> str:
        active = [
            name
            for name, value in (
                ("combined text scan", self.combined_text),
                ("binary scan", self.binary_scan),
            )
            if value
        ]
        if not active:
            return "trusted per-channel reads"
        return "; ".join(active)


@dataclass(frozen=True)
class ScanCapability:
    """How one driver acquires a sweep, and what it may be trusted with."""

    driver: str
    path: str = PATH_SWEEP
    combines_channels: bool = False
    binary_transport: bool = False
    combined_text_opt_in: bool = False
    binary_scan: str = BINARY_UNSUPPORTED
    evidence: str = ""
    #: ``True`` only when the reported transport was observed on hardware
    #: shipped in this repository (the V2 FIFO path).
    hardware_verified: bool = False

    def __post_init__(self) -> None:
        if self.path not in PATHS:
            raise ValueError(f"unknown acquisition path {self.path!r}")
        if self.binary_scan not in BINARY_STATES:
            raise ValueError(f"unknown binary scan state {self.binary_scan!r}")

    # -- derived questions --------------------------------------------
    @property
    def scan_mask(self) -> bool:
        """True when the driver already sends a channel mask."""
        return self.path == PATH_SCAN_MASK

    @property
    def combined_text_available(self) -> bool:
        """True when a masked text read may be attempted on this driver."""
        return self.combines_channels or self.combined_text_opt_in

    @property
    def binary_available(self) -> bool:
        """True when the ``scan_bin`` frame may be attempted here."""
        return self.binary_scan in {BINARY_OPT_IN, BINARY_VERIFIED}

    @property
    def hardware_unverified(self) -> bool:
        """True when some described transport has no hardware evidence.

        A verified driver still counts as unverified when it carries an
        opt-in read, because the *opt-in* is the part nobody has observed:
        the V2 FIFO transport is proven, its validated ``scan_bin`` decode
        is not.
        """
        return (
            not self.hardware_verified
            or self.combined_text_opt_in
            or self.binary_scan == BINARY_OPT_IN
        )

    def requested(self, preferences: ScanPreferences) -> str:
        """The read path this driver would use under ``preferences``.

        A driver that already combines its channels (``scan_mask``,
        ``fifo``) is never reported as an opt-in: it needs no optimisation,
        and its existing reads must stay exactly as they are.  The trusted
        default therefore always resolves to ``path`` itself.
        """
        if preferences.binary_scan and self.binary_available:
            return "scan_bin"
        if preferences.combined_text:
            if self.path in {PATH_SCAN_MASK, PATH_FIFO}:
                return self.path
            if self.combined_text_available:
                return "combined-text"
        return self.path

    def describe(self) -> str:
        parts = [f"reads: {self.path}"]
        if self.combines_channels:
            parts.append("one command per sweep for all channels")
        if self.binary_transport:
            parts.append("binary transport")
        if self.binary_scan == BINARY_OPT_IN:
            parts.append("scan_bin opt-in only, unverified")
        elif self.binary_scan == BINARY_VERIFIED:
            parts.append("scan_bin verified")
        if self.combined_text_opt_in:
            parts.append("combined text opt-in only, unverified")
        return "; ".join(parts)

    def with_runtime(self, sweep_method: object) -> ScanCapability:
        """This capability as the live driver state makes it.

        ``sweep_method`` is the attribute the drivers already use to pick
        their command family, so the matrix can never claim a mask read for
        a firmware that fell back to plain ``sweep``/``scan``.
        """
        if not isinstance(sweep_method, str) or sweep_method not in PATHS:
            return self
        if sweep_method == PATH_SWEEP:
            # Old firmware: neither a masked scan nor a binary scan exists.
            return replace(
                self,
                path=PATH_SWEEP,
                combines_channels=False,
                combined_text_opt_in=False,
                binary_scan=BINARY_UNSUPPORTED,
            )
        return replace(
            self,
            path=sweep_method,
            combines_channels=sweep_method in {PATH_SCAN_MASK, PATH_FIFO},
        )


#: Generic entry used for a device type the matrix does not name.
GENERIC = ScanCapability(
    driver="",
    path=PATH_SWEEP,
    evidence="no declared protocol; assume one command per channel",
)

#: The audited driver matrix (PERF-02).  Every entry states where the claim
#: comes from, so it can be re-checked against the code cited.
DRIVERS: Mapping[str, ScanCapability] = MappingProxyType(
    {
        "VNA": ScanCapability(
            driver="VNA",
            path=PATH_SWEEP,
            evidence=(
                "VNA.readValues: one exec_command per channel"
                " (Hardware/VNA.py:206)"
            ),
            hardware_verified=True,
        ),
        "NanoVNA": ScanCapability(
            driver="NanoVNA",
            path=PATH_SCAN,
            combined_text_opt_in=True,
            evidence=(
                "NanoVNA.readValues sends `scan ... 0b110` and caches both"
                " channels for fw >= 0.7.1, plain `scan`/`sweep` below"
                " (Hardware/NanoVNA.py:88-136)"
            ),
            hardware_verified=True,
        ),
        "NanoVNA_H": ScanCapability(
            driver="NanoVNA_H",
            path=PATH_SCAN,
            combined_text_opt_in=True,
            evidence="inherits NanoVNA's text scan commands",
            hardware_verified=True,
        ),
        "NanoVNA_H4": ScanCapability(
            driver="NanoVNA_H4",
            path=PATH_SCAN,
            combined_text_opt_in=True,
            evidence=(
                "NanoVNA_H4.__init__ selects scan_mask only when the"
                " firmware announced it (Hardware/NanoVNA_H4.py:41-43)"
            ),
            hardware_verified=True,
        ),
        "NanoVNA_F": ScanCapability(
            driver="NanoVNA_F",
            path=PATH_SCAN,
            combined_text_opt_in=True,
            evidence="inherits NanoVNA's text scan commands",
            hardware_verified=True,
        ),
        "NanoVNA_F_V2": ScanCapability(
            driver="NanoVNA_F_V2",
            path=PATH_SCAN,
            combined_text_opt_in=True,
            evidence="inherits NanoVNA's text scan commands",
            hardware_verified=True,
        ),
        "NanoVNA_F_V3": ScanCapability(
            driver="NanoVNA_F_V3",
            path=PATH_SCAN,
            combined_text_opt_in=True,
            evidence="inherits NanoVNA's text scan commands",
            hardware_verified=True,
        ),
        "JNCRadio_VNA_3G": ScanCapability(
            driver="JNCRadio_VNA_3G",
            path=PATH_SCAN,
            combined_text_opt_in=True,
            evidence=(
                "setSweep sends plain `scan`"
                " (Hardware/JNCRadio_VNA_3G.py:62-65); channels read"
                " separately"
            ),
            hardware_verified=True,
        ),
        "SV4401A": ScanCapability(
            driver="SV4401A",
            path=PATH_SCAN,
            combined_text_opt_in=True,
            evidence=(
                "init_features drops the mask feature and setSweep sends"
                " plain `scan` (Hardware/SV4401A.py:47-65)"
            ),
            hardware_verified=True,
        ),
        "SV6301A": ScanCapability(
            driver="SV6301A",
            path=PATH_SCAN,
            combined_text_opt_in=True,
            evidence=(
                "init_features drops the mask feature and setSweep sends"
                " plain `scan` (Hardware/SV6301A.py:47-65)"
            ),
            hardware_verified=True,
        ),
        "NanoVNA_V2": ScanCapability(
            driver="NanoVNA_V2",
            path=PATH_FIFO,
            combines_channels=True,
            binary_transport=True,
            # The validated decode is an opt-in: the framing has never been
            # confirmed against a firmware dump in this repository.
            binary_scan=BINARY_OPT_IN,
            evidence=(
                "one READFIFO reply holds fwd/rev0/rev1 per point"
                " (Hardware/NanoVNA_V2.py:155-241)"
            ),
            hardware_verified=True,
        ),
        "LiteVNA64": ScanCapability(
            driver="LiteVNA64",
            path=PATH_FIFO,
            combines_channels=True,
            binary_transport=True,
            # The validated decode is an opt-in: the framing has never been
            # confirmed against a firmware dump in this repository.
            binary_scan=BINARY_OPT_IN,
            evidence=(
                "inherits NanoVNA_V2's FIFO read"
                " (Hardware/LiteVNA64.py:216-219)"
            ),
            hardware_verified=True,
        ),
        "TinySA": ScanCapability(
            driver="TinySA",
            path=PATH_SWEEP,
            evidence=(
                "`data 0` only, ASCII dB converted to a real complex value"
                " (Hardware/TinySA.py:98-111)"
            ),
            hardware_verified=True,
        ),
        "TinySA_Ultra": ScanCapability(
            driver="TinySA_Ultra",
            path=PATH_SWEEP,
            evidence="inherits TinySA's single-channel ASCII read",
            hardware_verified=True,
        ),
        "AVNA": ScanCapability(
            driver="AVNA",
            path=PATH_SWEEP,
            evidence=(
                "text `sweep` command, channels read one by one"
                " (Hardware/AVNA.py:42-46)"
            ),
            hardware_verified=True,
        ),
        "SimulatedDevice": ScanCapability(
            driver="SimulatedDevice",
            path=PATH_SWEEP,
            evidence=(
                "in-process analytic device; one read_channel call per"
                " channel (Hardware/Simulated.py:407-450)"
            ),
            hardware_verified=True,
        ),
        "SimulatedVNA": ScanCapability(
            driver="SimulatedVNA",
            path=PATH_SWEEP,
            evidence="adapter over SimulatedDevice",
            hardware_verified=True,
        ),
    }
)


def for_driver(name: str) -> ScanCapability:
    """The matrix entry for ``name``, or the generic entry."""
    return DRIVERS.get(name, GENERIC)


def scan_capability_for(vna: object) -> ScanCapability:
    """The capability of ``vna`` as its class and live state declare it.

    Looked up along the MRO so a subclass without its own row (for example
    a driver added later) inherits its parent's honest claim instead of
    silently acquiring optimised reads.
    """
    capability = GENERIC
    for cls in type(vna).__mro__:
        entry = DRIVERS.get(cls.__name__)
        if entry is not None:
            capability = entry
            break
    return replace(
        capability,
        driver=capability.driver or type(vna).__name__,
    ).with_runtime(getattr(vna, "sweep_method", None))


def hardware_unverified_drivers() -> tuple[str, ...]:
    """Drivers whose optimised reads have no hardware evidence yet.

    These are exactly the drivers an operator must exercise in VERIFY-02
    before enabling :class:`ScanPreferences` for them.
    """
    return tuple(
        sorted(
            name
            for name, entry in DRIVERS.items()
            if entry.hardware_unverified
            or entry.binary_scan == BINARY_OPT_IN
            or entry.combined_text_opt_in
        )
    )
