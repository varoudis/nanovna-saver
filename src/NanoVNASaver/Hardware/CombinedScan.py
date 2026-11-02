#  NanoVNASaver
#
#  A python program to view and export Touchstone data from your NanoVNA
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
"""Validated decoders for the optimised read paths (PERF-02).

The optimised paths exist only to save round trips; they are not allowed to
change what a measurement *is*.  Everything here therefore runs in two
steps: decode a reply into a fully validated :class:`CombinedSweep`, or
raise.  A reply that is partial, short, malformed, non-finite or
implausible is never repaired, padded or interpolated - it is rejected as a
whole, and :func:`one_shot_read` then re-reads the sweep through the
trusted per-channel path.

The decoders are deliberately independent of the drivers: they take plain
data (reply lines, or a byte buffer) and the expected grid, so they can be
tested against hand-computed reference frames and reused by any driver that
gains a combined transport later.  Nothing here imports Qt or the
acquisition API, so both the GUI worker and the Qt-free session can use it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from struct import unpack_from
from typing import Callable, Iterable, Sequence

from .ScanCapability import ScanPreferences

#: Upper bound on a single Cartesian component of an S parameter, matching
#: the plausibility test ``SweepWorker`` already applies to trusted reads
#: (``VALUE_MAX``).  A magnitude above this is a transmission-desync
#: artefact, not a measurement.
VALUE_MAX = 9.5

#: One FIFO record of the V2 family: fwd, rev0, rev1 I/Q in little-endian
#: int32 plus the point index, padded to 32 bytes.
BINARY_RECORD = "<iiiiiihxxxxxx"
#: Size in bytes of one :data:`BINARY_RECORD`.
BINARY_RECORD_SIZE = 32

#: One command per channel: the trusted path, unchanged by PERF-02.
PATH_TRUSTED = "trusted"
#: One masked ``scan`` reply for every channel.
PATH_COMBINED = "combined"
#: One binary frame for every point and channel.
PATH_BINARY = "scan_bin"


class OptimisedReadError(Exception):
    """An optimised read produced something unusable.

    Always caught by :func:`one_shot_read`, which falls back to the trusted
    path.  Never allowed to escape into a measurement.
    """


class ShortDataError(OptimisedReadError):
    """The device answered, but with too few points to be a sweep."""


class MalformedDataError(OptimisedReadError):
    """The reply could not be decoded, or decoded to impossible values."""


@dataclass(frozen=True)
class CombinedSweep:
    """A whole sweep read in one go, already validated.

    ``s21`` is empty for a one-port device: an absent channel is absent,
    never a series of zeros.
    """

    frequencies: tuple[int, ...]
    s11: tuple[complex, ...]
    s21: tuple[complex, ...] = ()
    path: str = PATH_COMBINED

    def __post_init__(self) -> None:
        if len(self.frequencies) != len(self.s11):
            raise MalformedDataError(
                f"{len(self.frequencies)} frequencies for {len(self.s11)}"
                " S11 points"
            )
        if self.s21 and len(self.s21) != len(self.s11):
            raise MalformedDataError(
                f"{len(self.s21)} S21 points for {len(self.s11)} S11 points"
            )

    @property
    def channels(self) -> tuple[str, ...]:
        return ("data 0", "data 1") if self.s21 else ("data 0",)

    def channel(self, value: str) -> tuple[complex, ...]:
        if value == "data 0":
            return self.s11
        if value == "data 1" and self.s21:
            return self.s21
        raise MalformedDataError(f"channel {value!r} was not read")


@dataclass(frozen=True)
class ReadFailure:
    """Why an optimised read was rejected."""

    kind: str
    detail: str

    def describe(self) -> str:
        return f"{self.kind}: {self.detail}"


@dataclass(frozen=True)
class ReadOutcome:
    """What a read attempt actually produced, and by which path."""

    path: str
    sweep: CombinedSweep | None = None
    failure: ReadFailure | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def optimised(self) -> bool:
        return self.path != PATH_TRUSTED

    @property
    def fell_back(self) -> bool:
        return self.failure is not None and self.path == PATH_TRUSTED


def _check_component(value: float, where: str) -> float:
    if not math.isfinite(value):
        raise MalformedDataError(f"{where}: non-finite value {value!r}")
    if abs(value) > VALUE_MAX:
        raise MalformedDataError(f"{where}: |{value:g}| exceeds {VALUE_MAX:g}")
    return value


def _check_grid(
    frequencies: Sequence[int], *, start: int, stop: int
) -> tuple[int, ...]:
    """Reject a grid that is not the one the sweep was asked for."""
    if frequencies[0] != start or frequencies[-1] != stop:
        raise MalformedDataError(
            f"grid {frequencies[0]}..{frequencies[-1]} Hz, requested"
            f" {start}..{stop} Hz"
        )
    if len(frequencies) > 2 and frequencies[0] >= frequencies[-1]:
        raise MalformedDataError(
            f"grid is not ascending ({frequencies[0]}..{frequencies[-1]})"
        )
    return tuple(int(value) for value in frequencies)


def parse_combined_reply(  # noqa: PLR0913 - a reply decoder needs its grid
    lines: Iterable[str],
    *,
    points: int,
    channels: Sequence[str] = ("data 0", "data 1"),
    frequencies: Sequence[int] | None = None,
    start: int | None = None,
    stop: int | None = None,
) -> CombinedSweep:
    """Decode a masked ``scan ... 0b110`` reply into a validated sweep.

    Each line holds the Cartesian components of every requested channel for
    one point (``re0 im0`` for one channel, ``re0 im0 re1 im1`` for two).
    Extra lines, missing lines, non-numeric fields, non-finite or
    implausible values all raise: the reply is accepted whole or not at all.
    """
    if points <= 0:
        raise ShortDataError(f"requested {points} points")
    wanted = [name for name in channels if name in {"data 0", "data 1"}]
    if not wanted:
        raise MalformedDataError(f"no readable channel in {tuple(channels)}")
    width = 2 * len(wanted)

    rows: list[list[float]] = []
    for number, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != width:
            raise MalformedDataError(
                f"line {number}: {len(fields)} values, expected {width}"
            )
        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            raise MalformedDataError(f"line {number}: {exc}") from exc
        rows.append(
            [
                _check_component(value, f"line {number} value {index + 1}")
                for index, value in enumerate(values)
            ]
        )

    if len(rows) != points:
        raise ShortDataError(f"{len(rows)} points, expected {points}")

    if frequencies is not None:
        if len(frequencies) != points:
            raise ShortDataError(
                f"{len(frequencies)} frequencies for {points} points"
            )
        if start is not None and stop is not None:
            _check_grid(frequencies, start=start, stop=stop)
        grid = tuple(int(value) for value in frequencies)
    else:
        grid = tuple(range(points))

    s11 = tuple(complex(row[0], row[1]) for row in rows)
    s21 = tuple(complex(row[2], row[3]) for row in rows) if width == 4 else ()
    return CombinedSweep(frequencies=grid, s11=s11, s21=s21)


def decode_scan_bin(
    payload: bytes,
    *,
    points: int,
    channels: Sequence[str] = ("data 0", "data 1"),
) -> CombinedSweep:
    """Decode a ``scan_bin`` frame into a validated sweep.

    ``payload`` must hold exactly ``points`` complete records: the V2 family
    encodes 32-byte little-endian records in the same layout as the FIFO
    read, so a short, long or misaligned buffer means the stream is out of
    step and nothing in it can be trusted.  Each record's point index must
    be unique and inside the sweep, and every point must be present,
    because a missing index would otherwise become an interpolated sample.
    The returned grid is the point index itself, which is what the firmware
    numbered; the caller checks it against the requested frequencies.
    """
    if points <= 0:
        raise ShortDataError(f"requested {points} points")
    expected = points * BINARY_RECORD_SIZE
    if len(payload) != expected:
        raise ShortDataError(f"{len(payload)} bytes, expected {expected}")

    wanted = [name for name in channels if name in {"data 0", "data 1"}]
    if not wanted:
        raise MalformedDataError(f"no readable channel in {tuple(channels)}")
    two_port = len(wanted) == 2

    records: dict[int, tuple[complex, complex]] = {}
    for number in range(points):
        (
            fwd_real,
            fwd_imag,
            rev0_real,
            rev0_imag,
            rev1_real,
            rev1_imag,
            index,
        ) = unpack_from(BINARY_RECORD, payload, number * BINARY_RECORD_SIZE)
        if not 0 <= index < points:
            raise MalformedDataError(
                f"record {number}: point index {index} outside 0..{points - 1}"
            )
        if index in records:
            raise MalformedDataError(
                f"record {number}: duplicate index {index}"
            )
        forward = complex(fwd_real, fwd_imag)
        if forward == 0:
            raise MalformedDataError(f"record {number}: zero forward sample")
        reflected = complex(rev0_real, rev0_imag) / forward
        transmitted = complex(rev1_real, rev1_imag) / forward
        for value in (reflected, transmitted):
            _check_component(value.real, f"record {number}")
            _check_component(value.imag, f"record {number}")
        records[index] = (reflected, transmitted)

    missing = [index for index in range(points) if index not in records]
    if missing:
        listed = ", ".join(str(index) for index in missing[:4])
        more = "..." if len(missing) > 4 else ""
        raise ShortDataError(f"missing point indices {listed}{more}")
    return CombinedSweep(
        frequencies=tuple(range(points)),
        s11=tuple(records[index][0] for index in range(points)),
        s21=(
            tuple(records[index][1] for index in range(points))
            if two_port
            else ()
        ),
        path=PATH_BINARY,
    )


def one_shot_read(
    attempt: Callable[[], CombinedSweep],
    trusted: Callable[[], CombinedSweep],
) -> ReadOutcome:
    """Run an optimised read, falling back to the trusted path on any doubt.

    ``attempt`` performs the optimised read and raises if the device's reply
    cannot be validated.  ``trusted`` performs the ordinary per-channel read
    and is only called when the optimised read was rejected - an accepted
    optimised read is never mixed with, or partially replaced by, trusted
    data.  Errors from ``trusted`` itself propagate: a broken device must
    fail loudly rather than report a mystery sweep.
    """
    try:
        sweep = attempt()
    except OptimisedReadError as exc:
        return ReadOutcome(
            path=PATH_TRUSTED,
            sweep=trusted(),
            failure=ReadFailure(kind=_failure_kind(exc), detail=str(exc)),
        )
    except Exception as exc:  # device replies are arbitrary
        return ReadOutcome(
            path=PATH_TRUSTED,
            sweep=trusted(),
            failure=ReadFailure(
                kind="error", detail=f"{type(exc).__name__}: {exc}"
            ),
        )
    return ReadOutcome(path=sweep.path, sweep=sweep)


def _failure_kind(error: Exception) -> str:
    if isinstance(error, ShortDataError):
        return "short"
    if isinstance(error, MalformedDataError):
        return "malformed"
    return "rejected"


def preferences_from(
    *, combined_text: bool = False, binary_scan: bool = False
) -> ScanPreferences:
    """Explicit opt-in flags; both default OFF (the trusted path)."""
    return ScanPreferences(
        combined_text=bool(combined_text), binary_scan=bool(binary_scan)
    )
