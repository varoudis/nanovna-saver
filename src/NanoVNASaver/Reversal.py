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
"""Full two-port measurement by DUT reversal (RF-04).

A two-port instrument measures four S-parameters at once.  A one-port
instrument (or a two-port used one port at a time) measures two: the port it
drives sees ``S11``, and the port it receives sees ``S21``.  Turning the DUT
around gives the other two:

    forward   DUT in its normal orientation   measures S11, S21
    reverse   DUT rotated 180 degrees         measures S22, S12

Because reversing the DUT swaps ports 1 and 2, the *reverse* measurement's
S11 *is* the DUT's S22, and its S21 *is* the DUT's S12.  No transform is
applied: the numbers are moved into their correct channels, which is why
this is called a *combination* rather than a correction.

What this module refuses
------------------------
Two sweeps describe the same DUT only if they agree on everything that would
change the meaning of a sample.  ``check_compatibility`` therefore compares
the frequency grid exactly, the reference impedance, the calibration
identity, the source and the simulated flag, and refuses with a reason
naming the difference.  Combining a 50 ohm forward sweep with a 75 ohm
reverse sweep, or two sweeps taken with different calibrations, would
produce a matrix whose entries are not even in the same units.

What this module does not claim
-------------------------------
A combined matrix is only as good as the calibration behind both sweeps.
It carries a ``ReversalProvenance`` naming both source snapshots so an
export can say where each entry came from, and both originals are retained:
the combination never destroys the forward and reverse measurements it was
built from.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence

from .Measurement import (
    S11,
    S12,
    S21,
    S22,
    MeasurementSnapshot,
    _freeze_channel,
    new_snapshot_id,
)
from .RFTools import Datapoint

__all__ = [
    "ReversalCompatibility",
    "ReversalError",
    "ReversalProvenance",
    "check_compatibility",
    "combine_reversal",
    "describe_provenance",
    "own_channels",
    "reversed_channels",
]

#: The channel a forward sweep measures at each port.
FORWARD_MEASURED: tuple[str, ...] = (S11, S21)
#: The channel a reverse sweep measures, in the DUT's own numbering.
REVERSE_MEASURED: tuple[str, ...] = (S22, S12)


class ReversalError(ValueError):
    """Two sweeps cannot be combined into one S-matrix."""


class ReversalCompatibility(Enum):
    """Outcome of comparing the two sweeps."""

    SAME_DUT = "same DUT"
    DIFFERENT_GRID = "different frequency grid"
    DIFFERENT_REFERENCE = "different reference impedance"
    DIFFERENT_CALIBRATION = "different calibration"
    DIFFERENT_SOURCE = "different source"
    MIXED_SIMULATION = "one sweep is simulated and one is not"
    SAME_DIRECTION = "both sweeps describe the same orientation"


@dataclass(frozen=True)
class ReversalProvenance:
    """Which sweep each entry of a combined matrix came from."""

    forward_id: str
    reverse_id: str
    forward_direction: str = "forward"
    reverse_direction: str = "reverse"
    note: str = ""

    def describe(self) -> str:
        text = (
            f"S11/S21 from {self.forward_direction} sweep"
            f" {self.forward_id[:12]}; S22/S12 from"
            f" {self.reverse_direction} sweep {self.reverse_id[:12]}"
        )
        return f"{text} ({self.note})" if self.note else text

    def to_dict(self) -> dict:
        return {
            "forward_id": self.forward_id,
            "reverse_id": self.reverse_id,
            "forward_direction": self.forward_direction,
            "reverse_direction": self.reverse_direction,
            "note": self.note,
        }


def own_channels(snapshot: MeasurementSnapshot) -> tuple[str, ...]:
    """The channels a snapshot actually provides, never invented ones."""
    return tuple(
        name
        for name in (S11, S21, S12, S22)
        if name in snapshot.corrected or name in snapshot.raw
    )


def reversed_channels(
    snapshot: MeasurementSnapshot,
) -> dict[str, tuple[Datapoint, ...]]:
    """Re-label a reverse sweep into the DUT's own S-parameters.

    A reverse sweep drives port 2, so its ``S11`` is the DUT's ``S22`` and
    its ``S21`` is the DUT's ``S12``.  This is a re-labelling: the complex
    samples are moved unchanged, with no rotation, conjugation or scaling.
    """
    channels: dict[str, tuple[Datapoint, ...]] = {}
    source = snapshot.corrected or snapshot.raw
    if S11 in source:
        channels[S22] = tuple(source[S11])
    if S21 in source:
        channels[S12] = tuple(source[S21])
    # A snapshot that already carries S12/S22 (a genuine two-port capture)
    # keeps them as they are: they are measured, not inferred.
    if S12 in source and S12 not in channels:
        channels[S12] = tuple(source[S12])
    if S22 in source and S22 not in channels:
        channels[S22] = tuple(source[S22])
    return channels


def _frequencies(snapshot: MeasurementSnapshot) -> tuple[int, ...]:
    if snapshot.frequencies:
        return tuple(snapshot.frequencies)
    source = snapshot.corrected or snapshot.raw
    for name in (S11, S21, S12, S22):
        points = source.get(name)
        if points:
            return tuple(point.freq for point in points)
    return ()


def check_compatibility(
    forward: MeasurementSnapshot, reverse: MeasurementSnapshot
) -> ReversalCompatibility:
    """Compare the two sweeps and report the *first* disagreement.

    Returning a value rather than raising lets a caller explain every
    problem at once if it wants to; :func:`combine_reversal` raises.
    """
    forward_grid = _frequencies(forward)
    reverse_grid = _frequencies(reverse)
    if not forward_grid or not reverse_grid:
        raise ReversalError("both sweeps need frequencies to compare")
    if forward_grid != reverse_grid:
        raise ReversalError(
            "the sweeps are on different frequency grids:"
            f" {len(forward_grid)} points from {forward_grid[0]} Hz and"
            f" {len(reverse_grid)} points from {reverse_grid[0]} Hz."
            " A combined S-matrix needs one grid; resweep both"
            " orientations over the same span and point count."
        )
    if forward.reference_impedance != reverse.reference_impedance:
        return ReversalCompatibility.DIFFERENT_REFERENCE
    forward_cal = forward.provenance.calibration
    reverse_cal = reverse.provenance.calibration
    forward_identity = forward_cal.identity if forward_cal else ""
    reverse_identity = reverse_cal.identity if reverse_cal else ""
    if forward_identity != reverse_identity:
        return ReversalCompatibility.DIFFERENT_CALIBRATION
    if forward.provenance.source != reverse.provenance.source:
        return ReversalCompatibility.DIFFERENT_SOURCE
    if forward.simulated != reverse.simulated:
        return ReversalCompatibility.MIXED_SIMULATION
    return ReversalCompatibility.SAME_DUT


def _require_same_dut(
    forward: MeasurementSnapshot, reverse: MeasurementSnapshot
) -> None:
    verdict = check_compatibility(forward, reverse)
    if verdict is ReversalCompatibility.SAME_DUT:
        return
    detail = {
        ReversalCompatibility.DIFFERENT_REFERENCE: (
            f"the forward sweep is at {forward.reference_impedance:g} ohm and"
            f" the reverse sweep at {reverse.reference_impedance:g} ohm"
        ),
        ReversalCompatibility.DIFFERENT_CALIBRATION: (
            "the two sweeps were corrected with different calibrations"
            f" ({_cal_identity(forward)} vs {_cal_identity(reverse)})"
        ),
        ReversalCompatibility.DIFFERENT_SOURCE: (
            f"the sweeps came from different sources"
            f" ({forward.provenance.source!r} and"
            f" {reverse.provenance.source!r})"
        ),
        ReversalCompatibility.MIXED_SIMULATION: (
            "one sweep is simulated and the other is not, so they cannot"
            " describe the same physical measurement"
        ),
    }[verdict]
    raise ReversalError(
        f"these sweeps cannot be combined: {detail}. Both orientations must"
        " be measured through the same set-up."
    )


def _cal_identity(snapshot: MeasurementSnapshot) -> str:
    calibration = snapshot.provenance.calibration
    identity = calibration.identity if calibration else ""
    return identity[:12] or "uncalibrated"


def combine_reversal(
    forward: MeasurementSnapshot,
    reverse: MeasurementSnapshot,
    *,
    name: str = "S-matrix",
) -> MeasurementSnapshot:
    """Combine a forward and a reverse sweep into one four-channel snapshot.

    Both inputs must be complete measurements of the *same* DUT in
    opposite orientations.  The result names both sources, keeps every
    sample unchanged, and reports only the channels it actually has: a
    forward sweep alone never gains an S12 it did not measure.
    """
    _require_same_dut(forward, reverse)
    for label, snapshot in (("forward", forward), ("reverse", reverse)):
        if snapshot.provenance.completion.value != "complete":
            raise ReversalError(
                f"the {label} sweep is"
                f" {snapshot.provenance.completion.value}, not complete;"
                " a partial sweep cannot be combined"
            )

    forward_source = forward.corrected or forward.raw
    if S11 not in forward_source:
        raise ReversalError(
            "the forward sweep has no S11, so it cannot be the forward"
            " orientation of a two-port measurement"
        )

    channels: dict[str, tuple[Datapoint, ...]] = {}
    channels[S11] = tuple(forward_source[S11])
    if S21 in forward_source:
        channels[S21] = tuple(forward_source[S21])
    channels.update(reversed_channels(reverse))

    missing = [name for name in (S11, S21, S12, S22) if name not in channels]
    provenance = ReversalProvenance(
        forward_id=forward.id,
        reverse_id=reverse.id,
        note=("reverse sweep re-labelled: its S11 is S22 and its S21 is S12"),
    )
    if missing:
        note = f"absent channels: {', '.join(missing)}"
        provenance = replace(provenance, note=f"{provenance.note}; {note}")

    notes = "\n".join(
        filter(
            None,
            (
                forward.provenance.notes,
                f"combined by DUT reversal: {provenance.describe()}",
            ),
        )
    )
    return MeasurementSnapshot(
        id=new_snapshot_id(),
        name=name,
        timestamp=datetime.now(timezone.utc),
        reference_impedance=forward.reference_impedance,
        raw={},
        corrected={
            key: _freeze_channel(value)
            for key, value in channels.items()
            if value
        },
        provenance=replace(
            forward.provenance,
            derived_from=f"{forward.id}+{reverse.id}",
            notes=notes,
        ),
    )


def describe_provenance(snapshot: MeasurementSnapshot) -> Sequence[str]:
    """Lines a report can print to explain where the entries came from."""
    derived = snapshot.provenance.derived_from
    if not derived:
        return []
    if "+" not in derived:
        return [f"derived from {derived}"]
    forward, _, reverse = derived.partition("+")
    return [
        f"S11/S21 from forward sweep {forward}",
        f"S22/S12 from reverse sweep {reverse}",
    ]


# ----------------------------------------------------------------------
# Capture workflow
# ----------------------------------------------------------------------
class CaptureStage(Enum):
    """Where the forward/reverse capture sequence is."""

    IDLE = "idle"
    AWAITING_FORWARD = "awaiting forward sweep"
    AWAITING_REVERSE = "awaiting reverse sweep"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ReversalCapture:
    """Collects one forward and one reverse sweep, then combines them.

    The sequence is deliberately explicit: the operator is asked to turn
    the DUT around between the two runs, so the controller never starts the
    second sweep itself.  A caller binds each completed sweep to the stage
    that is waiting for it; a sweep that arrives out of order is refused
    rather than stored, because storing it would attribute a reverse sweep
    to a forward slot.

    ``forward`` and ``reverse`` are retained after combining, so an export
    can show both original measurements beside the matrix.
    """

    forward: MeasurementSnapshot | None = None
    reverse: MeasurementSnapshot | None = None
    stage: CaptureStage = CaptureStage.AWAITING_FORWARD
    reason: str = ""
    combined: MeasurementSnapshot | None = None

    @property
    def complete(self) -> bool:
        return self.stage is CaptureStage.COMPLETE

    def accept(
        self, snapshot: MeasurementSnapshot, *, direction: str
    ) -> MeasurementSnapshot | None:
        """Take one sweep for ``direction`` and combine when both exist."""
        if self.stage in (CaptureStage.COMPLETE, CaptureStage.FAILED):
            raise ReversalError(
                f"this capture is {self.stage.value}; start a new one"
                " rather than adding another sweep to it"
            )
        if direction == "forward":
            if self.forward is not None:
                raise ReversalError(
                    "a forward sweep has already been captured; the"
                    " forward step is not repeated"
                )
            self.forward = snapshot
            if self.reverse is None:
                self.stage = CaptureStage.AWAITING_REVERSE
                return None
        elif direction == "reverse":
            if self.forward is None:
                raise ReversalError(
                    "the reverse sweep cannot be captured before the"
                    " forward sweep; the DUT is still in its first"
                    " orientation"
                )
            if self.reverse is not None:
                raise ReversalError("a reverse sweep has already been captured")
            self.reverse = snapshot
        else:
            raise ReversalError(
                f"unknown direction {direction!r}; use 'forward' or 'reverse'"
            )
        return self._combine()

    def _combine(self) -> MeasurementSnapshot:
        try:
            combined = combine_reversal(self.forward, self.reverse)
        except ReversalError as error:
            # The reason is kept so the operator learns *why* the two
            # sweeps did not belong together; both are retained for export.
            self.stage = CaptureStage.FAILED
            self.reason = str(error)
            raise
        self.combined = combined
        self.stage = CaptureStage.COMPLETE
        return combined

    def cancel(self) -> None:
        """Abandon the sequence; captured sweeps are kept for inspection."""
        if self.stage is CaptureStage.COMPLETE:
            return
        self.stage = CaptureStage.FAILED
        self.reason = "cancelled"

    def describe(self) -> str:
        if self.stage is CaptureStage.COMPLETE and self.combined is not None:
            return (
                "complete: "
                + ReversalProvenance(
                    forward_id=self.forward.id,
                    reverse_id=self.reverse.id,
                ).describe()
            )
        if self.stage is CaptureStage.FAILED:
            return f"failed: {self.reason}"
        return self.stage.value
