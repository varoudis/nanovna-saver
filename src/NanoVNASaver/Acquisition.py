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
"""A Qt-free acquisition API for scripts, tests and services (AUTO-01).

Nothing in this module imports Qt.  A script can open a device, configure a
sweep, take an immutable snapshot, evaluate limits and write Touchstone
without a display, and the results carry the same provenance the GUI would
have recorded.

Design rules
------------
* **A handle owns a device.**  Opening the same device twice is refused
  rather than silently sharing a serial port: two readers on one port produce
  interleaved garbage, and the second caller gets an :class:`DeviceBusyError`
  instead.
* **Cancellation is cooperative and bounded.**  A run checks a token between
  samples, so a cancel returns promptly and the partial sweep is discarded
  rather than published.
* **Every failure is a structured exception.**  Callers distinguish "the
  device is busy" from "the sweep was cancelled" from "the device said no"
  without parsing strings.
* **A snapshot is the result.**  Nothing is yielded as a bare list of
  complex numbers: a caller that needs the data gets the immutable snapshot
  the rest of the application uses, so an automated result cannot be
  richer or poorer than an interactive one.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence

from .Hardware.CombinedScan import PATH_TRUSTED
from .Hardware.ScanCapability import ScanPreferences
from .Measurement import (
    S11,
    S21,
    CompletionState,
    MeasurementSnapshot,
    snapshot_from_datapoints,
)
from .RFTools import Datapoint

#: The device protocol names its two data streams ``data 0`` and ``data 1``.
#: The rest of the application calls them S11 and S21, so the translation
#: lives here, in one place, rather than at every call site.
_READ_COMMAND = {S11: "data 0", S21: "data 1"}

__all__ = [
    "AcquisitionError",
    "CancelToken",
    "CancelledError",
    "DeviceBusyError",
    "DeviceError",
    "DeviceSession",
    "ScanPreferences",
    "SweepRequest",
    "SweepResult",
    "UnsupportedError",
    "open_device",
]


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
class AcquisitionError(Exception):
    """Base class for every failure this API reports."""


class DeviceError(AcquisitionError):
    """The device rejected a command or reported a fault."""


class DeviceBusyError(DeviceError):
    """Another handle already owns this device."""


class CancelledError(AcquisitionError):
    """The run was cancelled before it completed."""


class UnsupportedError(AcquisitionError):
    """The device cannot do what was asked; the message says why."""


# ----------------------------------------------------------------------
# Cancellation
# ----------------------------------------------------------------------
class CancelToken:
    """A cooperative cancellation flag shared with a running sweep.

    Thread-safe because a UI thread, a signal handler and a service client
    may all cancel the same run.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError("the sweep was cancelled")


# ----------------------------------------------------------------------
# Requests and results
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SweepRequest:
    """What to measure, in the units the device actually uses."""

    start_hz: int
    stop_hz: int
    points: int
    channels: tuple[str, ...] = (S11, S21)
    average: int = 1
    name: str = ""

    def __post_init__(self) -> None:
        if self.start_hz <= 0 or self.stop_hz <= 0:
            raise ValueError("sweep frequencies must be positive")
        if self.start_hz >= self.stop_hz:
            raise ValueError("the sweep start must be below the stop")
        if self.points < 2:
            raise ValueError("a sweep needs at least two points")
        if self.average < 1:
            raise ValueError("the average count must be at least one")
        unknown = [name for name in self.channels if name not in (S11, S21)]
        if unknown:
            raise ValueError(f"unsupported channel {unknown[0]!r}")

    def describe(self) -> str:
        span = f"{self.start_hz / 1e6:.6g}-{self.stop_hz / 1e6:.6g} MHz"
        return (
            f"{span}, {self.points} points,"
            f" {'+'.join(self.channels) or 'no channels'}"
        )


@dataclass(frozen=True)
class SweepResult:
    """One completed acquisition, with its snapshot and timing.

    ``read_path`` names the transport the samples actually came from:
    ``trusted`` is the per-channel read every driver has always used, and
    anything else is an opted-in optimisation that was validated before it
    was accepted (PERF-02).
    """

    snapshot: MeasurementSnapshot
    request: SweepRequest
    averages: int = 1
    read_path: str = PATH_TRUSTED

    @property
    def channels(self) -> Mapping[str, Sequence[Datapoint]]:
        return self.snapshot.corrected

    def describe(self) -> str:
        return (
            f"{self.snapshot.name or 'sweep'}: {self.request.describe()},"
            f" {self.averages} average(s),"
            f" channels {', '.join(sorted(self.channels)) or 'none'},"
            f" reads {self.read_path}"
        )


class SupportsRead(Protocol):
    """The device surface this module needs; ``VNA`` and the simulator fit."""

    def read_frequencies(self) -> list[int]: ...

    def readValues(self, value: str) -> list[complex]: ...

    def setSweep(self, start: int, stop: int) -> None: ...

    def connected(self) -> bool: ...

    def get_features(self) -> set[str]: ...


# ----------------------------------------------------------------------
# Device ownership
# ----------------------------------------------------------------------
_OWNERS: dict[int, str] = {}
_OWNERS_LOCK = threading.Lock()


def _claim(device: object, owner: str) -> None:
    key = id(device)
    with _OWNERS_LOCK:
        if key in _OWNERS:
            raise DeviceBusyError(
                f"this device is already owned by {_OWNERS[key]!r}"
            )
        _OWNERS[key] = owner


def _release(device: object) -> None:
    with _OWNERS_LOCK:
        _OWNERS.pop(id(device), None)


class BorrowedDevice:
    """A connected device lent to a session for the length of one run.

    A session owns what it opens and disconnects it on close.  That is wrong
    for a device someone else keeps connected, so this forwards everything
    except ``disconnect``, which instead returns the device: it restores the
    point count the session changed and tells the lender exactly once.
    """

    _OWN = ("_device", "_on_release", "_datapoints", "_released")

    def __init__(self, device: object, on_release=None) -> None:
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_on_release", on_release)
        object.__setattr__(
            self, "_datapoints", getattr(device, "datapoints", None)
        )
        object.__setattr__(self, "_released", False)

    def __getattr__(self, name: str):
        return getattr(self._device, name)

    def __setattr__(self, name: str, value) -> None:
        if name in BorrowedDevice._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self._device, name, value)

    def disconnect(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            if self._datapoints is not None:
                self._device.datapoints = self._datapoints
        finally:
            if self._on_release is not None:
                self._on_release()


@dataclass
class DeviceSession:
    """An owned device handle with a Qt-free sweep loop.

    Construct through :func:`open_device`; the constructor does not claim
    anything, so a half-built session never leaves a device locked.
    """

    device: SupportsRead
    name: str = ""
    simulated: bool = False
    reference_impedance: float = 50.0
    #: Opt-in read paths (PERF-02); both switches default OFF.
    preferences: ScanPreferences = field(default_factory=ScanPreferences)
    #: Called with the fraction complete, between 0 and 1.
    on_progress: Callable[[float], None] | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    # -- lifecycle -----------------------------------------------------
    def close(self) -> None:
        """Release the device; idempotent, so a ``finally`` is always safe."""
        if self._closed:
            return
        self._closed = True
        disable = getattr(self.device, "disconnect", None)
        if callable(disable):
            try:
                disable()
            except Exception as error:
                # Release the claim first: a device that failed to shut down
                # cleanly must not stay locked for the rest of the process,
                # or a retry (the usual response to a stuck port) becomes
                # impossible.
                _release(self.device)
                raise DeviceError(f"closing the device failed: {error}") from (
                    error
                )
        _release(self.device)

    def __enter__(self) -> "DeviceSession":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def _check_open(self) -> None:
        if self._closed:
            raise DeviceError("this session is closed")

    # -- capabilities --------------------------------------------------
    def features(self) -> set[str]:
        self._check_open()
        try:
            return set(self.device.get_features())
        except Exception as error:
            raise DeviceError(f"reading capabilities failed: {error}") from (
                error
            )

    @property
    def device_channels(self) -> tuple[str, ...]:
        """Device read commands this device can deliver.

        A device that declares nothing is assumed to provide both streams,
        which is what an ordinary VNA does; a one-port replay source
        declares only ``data 0`` and so never yields an invented S21.
        """
        channels = getattr(self.device, "supported_channels", None)
        if channels:
            return tuple(channels)
        return tuple(_READ_COMMAND.values())

    @property
    def supported_channels(self) -> tuple[str, ...]:
        """Semantic channels (``S11``/``S21``) this device can provide."""
        return tuple(
            name
            for name, command in _READ_COMMAND.items()
            if command in self.device_channels
        )

    def check(self, request: SweepRequest) -> None:
        """Raise :class:`UnsupportedError` unless the device can do this.

        Checked before any device state is changed, so a rejected request
        cannot leave the hardware configured for a sweep nobody will run.
        """
        self._check_open()
        offered = self.device_channels
        for channel in request.channels:
            if _READ_COMMAND[channel] in offered:
                continue
            readable = [
                name
                for name, command in _READ_COMMAND.items()
                if command in offered
            ]
            raise UnsupportedError(
                f"this device does not provide {channel}; it offers"
                f" {', '.join(readable) or 'nothing'}"
            )
        preflight = getattr(self.device, "preflight", None)
        if callable(preflight):
            reason = preflight(request.start_hz, request.stop_hz)
            if reason:
                raise UnsupportedError(reason)

    # -- the sweep -----------------------------------------------------
    def sweep(
        self, request: SweepRequest, cancel: CancelToken | None = None
    ) -> SweepResult:
        """Run ``request`` and return its snapshot.

        The sweep is repeated ``request.average`` times and averaged per
        sample.  A cancellation is checked between samples and between
        averages, so it is honoured within one sample's time and the partial
        result is discarded — nothing is published for a cancelled run.
        """
        self._check_open()
        self.check(request)
        # Not ``cancel or CancelToken()``: a live token is falsy until
        # someone cancels it, so that idiom silently discards the
        # caller's token and the run becomes uncancellable.
        token = cancel if cancel is not None else CancelToken()
        token.raise_if_cancelled()

        set_points = getattr(self.device, "datapoints", None)
        if set_points is not None:
            # The point count is a plain attribute on every device, but it
            # is not part of the read protocol this module depends on.
            self.device.datapoints = request.points
        try:
            self.device.setSweep(request.start_hz, request.stop_hz)
        except Exception as error:
            raise DeviceError(f"configuring the sweep failed: {error}") from (
                error
            )
        try:
            frequencies = list(self.device.read_frequencies())
        except Exception as error:
            raise DeviceError(f"reading frequencies failed: {error}") from error
        if len(frequencies) != request.points:
            raise DeviceError(
                f"the device returned {len(frequencies)} frequencies for a"
                f" {request.points}-point sweep"
            )

        sums: dict[str, list[complex]] = {name: [] for name in request.channels}
        for name in request.channels:
            sums[name] = [0j] * len(frequencies)

        # A device may offer an optimised read (PERF-02); it is only ever
        # used when this session was explicitly asked for it, and a device
        # without the entry point is read exactly as before.
        optimised = getattr(self.device, "read_values_optimised", None)
        # ``data 0`` is what triggers an acquisition on the V2 and scan-mask
        # drivers; ``data 1`` only returns what the last one cached.  So every
        # pass reads S11 first, even when it was not requested and is thrown
        # away: an S21-only or S21-first request would otherwise publish the
        # previous sweep's samples as a new measurement.
        offered = self.device_channels
        reads = tuple(
            name
            for name, command in _READ_COMMAND.items()
            if name in request.channels or (name == S11 and command in offered)
        )
        paths: list[str] = []
        for index in range(request.average):
            token.raise_if_cancelled()
            for name in reads:
                token.raise_if_cancelled()
                command = _READ_COMMAND[name]
                try:
                    if callable(optimised):
                        values = list(optimised(command, self.preferences))
                    else:
                        values = list(self.device.readValues(command))
                except Exception as error:
                    raise DeviceError(
                        f"reading {name} failed: {error}"
                    ) from error
                paths.append(
                    str(getattr(self.device, "last_read_path", PATH_TRUSTED))
                )
                if len(values) != len(frequencies):
                    raise DeviceError(
                        f"{name} returned {len(values)} samples for"
                        f" {len(frequencies)} frequencies"
                    )
                target = sums.get(name)
                if target is None:
                    continue
                for position, value in enumerate(values):
                    target[position] += value
            if self.on_progress is not None:
                self.on_progress((index + 1) / request.average)
        token.raise_if_cancelled()

        corrected: dict[str, list[Datapoint]] = {}
        for name, values in sums.items():
            count = float(request.average)
            corrected[name] = [
                Datapoint(freq, value.real / count, value.imag / count)
                for freq, value in zip(frequencies, values, strict=True)
            ]
        snapshot = snapshot_from_datapoints(
            corrected.get(S11),
            corrected.get(S21),
            name=request.name or "automated sweep",
            source="automation",
            simulated=self.simulated,
            reference_impedance=self.reference_impedance,
            completion=CompletionState.COMPLETE,
        )
        # The run is labelled with the path its samples came from; a run
        # that mixed transports is impossible because a rejected optimised
        # read is re-read whole through the trusted path.
        read_path = (
            paths[0]
            if paths and len(set(paths)) == 1
            else (PATH_TRUSTED if not paths else "mixed")
        )
        return SweepResult(
            snapshot=snapshot,
            request=request,
            averages=request.average,
            read_path=read_path,
        )


def open_device(
    device: SupportsRead,
    *,
    name: str = "",
    simulated: bool | None = None,
    reference_impedance: float | None = None,
    preferences: ScanPreferences | None = None,
) -> DeviceSession:
    """Claim ``device`` and return a session that owns it.

    ``simulated`` is taken from the device when it can say so, because a
    simulated capture labelled as physical is the one mistake this API must
    never make.
    """
    if simulated is None:
        simulated = bool(getattr(device, "simulated", False))
    if reference_impedance is None:
        reference_impedance = float(
            getattr(device, "reference_impedance", 50.0) or 50.0
        )
    label = name or str(getattr(device, "name", "device"))
    _claim(device, label)
    return DeviceSession(
        device=device,
        name=label,
        simulated=simulated,
        reference_impedance=reference_impedance,
        preferences=preferences
        if preferences is not None
        else ScanPreferences(),
    )
