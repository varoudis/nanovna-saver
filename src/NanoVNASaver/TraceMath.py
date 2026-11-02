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
"""Complex memory-trace mathematics (ANALYSIS-02).

This module combines the *complex S-parameters* of two immutable
:class:`~NanoVNASaver.Measurement.MeasurementSnapshot` objects into a new,
derived snapshot.  It is deliberately not
:func:`NanoVNASaver.Traces.compare_traces`: that function subtracts displayed
quantities (dB, wrapped phase, impedance) for the chart delta readout and
interpolates onto a shared grid.  The operations here are arithmetic on the
measured complex samples themselves -- addition, subtraction, multiplication
and division -- and the result is a trace of its own rather than a display
difference.  Neither function implements the other.

Alignment policy
----------------
Only exactly coincident frequencies are combined.  Two sweeps measured on
different grids cannot be combined honestly, so nothing is interpolated and
nothing is extrapolated: a frequency present in one operand only is dropped
and counted (``dropped_first`` / ``dropped_second``, counted over distinct
frequencies).  A duplicated frequency inside one channel resolves to its last
occurrence, matching :func:`compare_traces`.  The combined points are always
returned in ascending frequency order, whatever order the inputs used.  When
not a single frequency coincides, the result is
:data:`TraceMathStatus.UNAVAILABLE` and carries that reason.

Reference impedance
-------------------
Every operation requires both traces to state the same
``reference_impedance``.  S-parameters are normalized quantities: a reflection
coefficient measured against 50 ohm and one measured against 75 ohm are not
the same kind of number, so neither a difference nor a quotient of them means
anything until one has been renormalized.  Addition and multiplication are
refused for the same reason rather than being singled out, and the refusal
names both impedances so the operator knows what to fix.

Undefined points
----------------
Division by an exact zero (or by a non-finite sample) does not produce
``inf``/``nan``.  Such a frequency is reported as an explicitly undefined
:class:`TraceMathPoint` with a reason, is counted, and is left out of the
resulting snapshot channel, so nothing singular can reach a chart.

Provenance
----------
The result is built with :meth:`MeasurementSnapshot.derived`, so ``raw`` is
empty, ``derived_from`` records both source ids (joined by
:data:`DERIVED_SOURCE_SEPARATOR`), and the operation, channel and every
alignment count are written into the provenance notes.  A derived trace can
never masquerade as a new acquisition.

Source replacement and deletion
-------------------------------
A derived trace is only meaningful while the two captures behind it still
exist.  :func:`check_sources` reports whether both recorded ids are live,
:func:`plottable_points` withholds points when they are not, and
:func:`recompute` replays the recorded operation against a live pool.  A
derived trace is never silently rebound to a different capture that happens
to carry the same channel: it becomes unavailable and names what is missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Sequence

from .Measurement import S11, SPARAMETERS, MeasurementSnapshot
from .RFTools import Datapoint

#: Joins the two source snapshot ids recorded in ``derived_from``.  Snapshot
#: ids are uuid4 hex, so ``+`` never occurs inside one.
DERIVED_SOURCE_SEPARATOR = "+"


class TraceOperation(Enum):
    """Arithmetic applied to two complex S-parameter channels."""

    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"

    @property
    def symbol(self) -> str:
        return {
            TraceOperation.ADD: "+",
            TraceOperation.SUBTRACT: "-",
            TraceOperation.MULTIPLY: "*",
            TraceOperation.DIVIDE: "/",
        }[self]

    @property
    def label(self) -> str:
        return {
            TraceOperation.ADD: "sum",
            TraceOperation.SUBTRACT: "difference",
            TraceOperation.MULTIPLY: "product",
            TraceOperation.DIVIDE: "quotient",
        }[self]


class TraceMathStatus(Enum):
    """Whether the requested operation produced usable samples."""

    OK = "ok"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class TraceMathPoint:
    """One combined frequency.

    ``value`` is ``None`` exactly when ``defined`` is false: an undefined
    point never carries ``inf`` or ``nan``, it carries a reason.
    """

    freq: int
    value: complex | None = None
    defined: bool = False
    reason: str = ""


@dataclass(frozen=True)
class TraceMathResult:
    """Outcome of one complex operation, with its provenance made explicit.

    ``points`` holds every coincident frequency, including the ones an
    operation could not define, so the caller can report both "12 of 13
    points used" and why the thirteenth is missing.
    """

    status: TraceMathStatus
    operation: TraceOperation
    channel: str
    first_id: str
    second_id: str
    points: tuple[TraceMathPoint, ...] = ()
    first_name: str = ""
    second_name: str = ""
    dropped_first: int = 0
    dropped_second: int = 0
    reason: str = ""
    #: Which data layer of the sources was combined (``corrected`` or
    #: ``raw``); recorded so the operation can be replayed exactly.
    corrected: bool = True
    #: The derived snapshot; ``None`` when nothing could be computed.
    snapshot: MeasurementSnapshot | None = None

    @property
    def ok(self) -> bool:
        return self.status is TraceMathStatus.OK

    @property
    def overlap(self) -> int:
        """Number of coincident frequencies the two traces shared."""
        return len(self.points)

    @property
    def used(self) -> int:
        """Number of points actually written to the derived snapshot."""
        return sum(1 for point in self.points if point.defined)

    @property
    def dropped(self) -> int:
        """Points discarded because the other trace has no sample there."""
        return self.dropped_first + self.dropped_second

    @property
    def defined_points(self) -> tuple[TraceMathPoint, ...]:
        return tuple(point for point in self.points if point.defined)

    @property
    def undefined_points(self) -> tuple[TraceMathPoint, ...]:
        return tuple(point for point in self.points if not point.defined)

    @property
    def frequencies(self) -> tuple[int, ...]:
        return tuple(point.freq for point in self.points)

    def span(self) -> tuple[int, int]:
        """Combined frequency span as ``(start, stop)`` in Hz."""
        if not self.points:
            return (0, 0)
        return (self.points[0].freq, self.points[-1].freq)

    def span_label(self) -> str:
        start, stop = self.span()
        if not start and not stop:
            return "no span"
        return f"{start / 1e6:.4g}-{stop / 1e6:.4g} MHz"

    def describe(self) -> str:
        """Human-readable one-line statement of what was computed."""
        if not self.ok:
            detail = self.reason or "no result"
            return (
                f"{self.channel} {self.operation.value} unavailable ({detail})"
            )
        undefined = len(self.undefined_points)
        detail = (
            f", {undefined} undefined ({self.undefined_points[0].reason})"
            if undefined
            else ""
        )
        return (
            f"{self.channel} {self.operation.label} of"
            f" {self.first_name or self.first_id}"
            f" {self.operation.symbol}"
            f" {self.second_name or self.second_id}:"
            f" {self.used} of {self.overlap} coincident points"
            f" ({self.dropped} dropped as non-coincident){detail},"
            f" {self.span_label()}"
        )


@dataclass(frozen=True)
class SourceAvailability:
    """Whether a derived result can still be reproduced from live traces.

    "Reproducible" is about the *sources*: replaying the recorded operation
    against them yields the same outcome.  It deliberately does not depend on
    that outcome being usable -- an operation over disjoint grids is
    perfectly reproducible and perfectly unplottable.  Use
    :func:`plottable_points` (or :attr:`TraceMathResult.ok`) to gate
    rendering.
    """

    reproducible: bool
    present: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        if self.reproducible:
            return ""
        return "source snapshot(s) no longer live: " + ", ".join(self.missing)

    def describe(self) -> str:
        if self.reproducible:
            return "reproducible: all source snapshots are live"
        return self.reason


def _result(  # noqa: PLR0913 - one argument per result field, no hidden state
    first: MeasurementSnapshot,
    second: MeasurementSnapshot,
    operation: TraceOperation,
    channel: str,
    *,
    corrected: bool = True,
    status: TraceMathStatus = TraceMathStatus.UNAVAILABLE,
    points: tuple[TraceMathPoint, ...] = (),
    dropped_first: int = 0,
    dropped_second: int = 0,
    reason: str = "",
    snapshot: MeasurementSnapshot | None = None,
) -> TraceMathResult:
    """Build the result for both outcomes, so no field can be forgotten."""
    return TraceMathResult(
        status=status,
        operation=operation,
        channel=channel,
        first_id=first.id,
        second_id=second.id,
        points=points,
        first_name=first.name,
        second_name=second.name,
        dropped_first=dropped_first,
        dropped_second=dropped_second,
        reason=reason,
        corrected=corrected,
        snapshot=snapshot,
    )


# ----------------------------------------------------------------------
# Alignment
# ----------------------------------------------------------------------
def _samples(points: Sequence[Datapoint]) -> dict[int, complex]:
    """Frequency lookup; a duplicated frequency resolves to the last one."""
    return {dp.freq: dp.z for dp in points}


def _apply(
    operation: TraceOperation, left: complex, right: complex
) -> tuple[complex | None, str]:
    """Return ``(value, reason)``; ``value`` is ``None`` when undefined."""
    if operation is TraceOperation.ADD:
        value = left + right
    elif operation is TraceOperation.SUBTRACT:
        value = left - right
    elif operation is TraceOperation.MULTIPLY:
        value = left * right
    else:
        if right == 0:
            return None, "divisor is zero"
        value = left / right
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        return None, "result is not finite"
    return value, ""


def _combine(
    operation: TraceOperation,
    freqs: Iterable[int],
    left: dict[int, complex],
    right: dict[int, complex],
) -> tuple[TraceMathPoint, ...]:
    points: list[TraceMathPoint] = []
    for freq in freqs:
        left_value = left[freq]
        right_value = right[freq]
        if not (
            math.isfinite(left_value.real)
            and math.isfinite(left_value.imag)
            and math.isfinite(right_value.real)
            and math.isfinite(right_value.imag)
        ):
            points.append(
                TraceMathPoint(freq, None, False, "source sample is not finite")
            )
            continue
        value, reason = _apply(operation, left_value, right_value)
        points.append(TraceMathPoint(freq, value, value is not None, reason))
    return tuple(points)


def _notes(  # noqa: PLR0913 - mirrors the provenance fields it records
    first: MeasurementSnapshot,
    second: MeasurementSnapshot,
    operation: TraceOperation,
    channel: str,
    points: Sequence[TraceMathPoint],
    dropped_first: int,
    dropped_second: int,
    corrected: bool,
) -> str:
    """Provenance text stored on the derived snapshot."""
    undefined = [point for point in points if not point.defined]
    reasons = sorted({point.reason for point in undefined if point.reason})
    lines = [
        f"ANALYSIS-02 {operation.value} of {channel}",
        f"sources: {first.id} {operation.symbol} {second.id}",
        f"data layer: {'corrected' if corrected else 'raw'}",
        f"reference_impedance: {first.reference_impedance:.6g} ohm",
        f"coincident points: {len(points)}"
        f" (used {len(points) - len(undefined)})",
        "dropped as non-coincident:"
        f" {dropped_first} from first, {dropped_second} from second",
        f"undefined points: {len(undefined)}",
        "alignment: exact frequencies only, no interpolation"
        " and no extrapolation",
    ]
    if reasons:
        lines.append("undefined because: " + "; ".join(reasons))
    return "\n".join(lines)


# ----------------------------------------------------------------------
# The operation
# ----------------------------------------------------------------------
def compute_trace_math(
    first: MeasurementSnapshot,
    second: MeasurementSnapshot,
    operation: TraceOperation,
    *,
    channel: str = S11,
    corrected: bool = True,
) -> TraceMathResult:
    """Combine one channel of two snapshots into a derived snapshot.

    ``first`` is the left operand and supplies the reference impedance and
    the provenance the result is derived from.  Only complex samples are
    used; no dB, phase or impedance conversion happens anywhere here.
    """
    if not isinstance(operation, TraceOperation):
        raise TypeError(f"unsupported operation {operation!r}")
    if channel not in SPARAMETERS:
        raise ValueError(f"unsupported parameter {channel!r}")

    for snapshot in (first, second):
        if not snapshot.id:
            raise ValueError("snapshots must carry an id to be combined")
        if DERIVED_SOURCE_SEPARATOR in snapshot.id:
            # ``derived_from`` stores both ids in one string; an id that
            # contains the separator would make that record ambiguous.
            raise ValueError(
                f"snapshot id {snapshot.id!r} contains the reserved"
                f" separator {DERIVED_SOURCE_SEPARATOR!r}"
            )

    left_points = first.channel(channel, corrected)
    right_points = second.channel(channel, corrected)
    missing = [
        snapshot.name or snapshot.id
        for snapshot, points in ((first, left_points), (second, right_points))
        if not points
    ]
    if missing:
        where = "corrected" if corrected else "raw"
        return _result(
            first,
            second,
            operation,
            channel,
            corrected=corrected,
            reason=(
                f"{channel} unavailable in the {where} data of"
                f" {', '.join(missing)}"
            ),
        )

    if first.reference_impedance != second.reference_impedance:
        return _result(
            first,
            second,
            operation,
            channel,
            corrected=corrected,
            reason=(
                "reference impedances differ"
                f" ({first.reference_impedance:.6g} ohm vs"
                f" {second.reference_impedance:.6g} ohm);"
                " renormalize before combining"
            ),
        )

    left = _samples(left_points)
    right = _samples(right_points)
    common = sorted(set(left) & set(right))
    dropped_first = len(left) - len(common)
    dropped_second = len(right) - len(common)
    if not common:
        return _result(
            first,
            second,
            operation,
            channel,
            corrected=corrected,
            dropped_first=dropped_first,
            dropped_second=dropped_second,
            reason=(
                "no coincident frequencies"
                f" ({dropped_first} and {dropped_second} points dropped;"
                " grids are never interpolated)"
            ),
        )

    points = _combine(operation, common, left, right)
    defined = [
        Datapoint(point.freq, point.value.real, point.value.imag)
        for point in points
        if point.defined and point.value is not None
    ]
    if not defined:
        return _result(
            first,
            second,
            operation,
            channel,
            corrected=corrected,
            points=points,
            dropped_first=dropped_first,
            dropped_second=dropped_second,
            reason=(
                f"all {len(points)} coincident points are undefined"
                f" ({points[0].reason})"
            ),
        )

    snapshot = first.derived(
        f"{channel} {operation.label} ({first.name or 'trace 1'}"
        f" {operation.symbol} {second.name or 'trace 2'})",
        {channel: defined},
        source_id=f"{first.id}{DERIVED_SOURCE_SEPARATOR}{second.id}",
        notes=_notes(
            first,
            second,
            operation,
            channel,
            points,
            dropped_first,
            dropped_second,
            corrected,
        ),
    )
    return _result(
        first,
        second,
        operation,
        channel,
        corrected=corrected,
        status=TraceMathStatus.OK,
        points=points,
        dropped_first=dropped_first,
        dropped_second=dropped_second,
        snapshot=snapshot,
    )


# ----------------------------------------------------------------------
# Source replacement / deletion
# ----------------------------------------------------------------------
def derived_source_ids(snapshot: MeasurementSnapshot) -> tuple[str, ...]:
    """Snapshot ids recorded in ``derived_from``.

    This module records the two operand ids joined by
    :data:`DERIVED_SOURCE_SEPARATOR`.
    """
    return tuple(
        part
        for part in snapshot.provenance.derived_from.split(
            DERIVED_SOURCE_SEPARATOR
        )
        if part
    )


def check_sources(
    result: TraceMathResult, live_ids: Iterable[str]
) -> SourceAvailability:
    """Report whether ``result``'s two source snapshots are still live.

    A derived trace whose sources were deleted must not be rendered as if it
    were a live combination.  This answers only that question: a result that
    was unavailable while both sources existed is still *reproducible* (it
    will always produce the same refusal), so rendering additionally needs
    :attr:`TraceMathResult.ok`.  :func:`plottable_points` applies both
    tests.
    """
    live = set(live_ids)
    sources = (result.first_id, result.second_id)
    present = tuple(source for source in sources if source in live)
    missing = tuple(
        dict.fromkeys(source for source in sources if source not in live)
    )
    return SourceAvailability(
        reproducible=not missing
        and bool(result.first_id)
        and bool(result.second_id),
        present=present,
        missing=missing,
    )


def recompute(
    result: TraceMathResult,
    snapshots: Iterable[MeasurementSnapshot],
) -> TraceMathResult:
    """Re-run ``result``'s operation against the live snapshots.

    ``snapshots`` is the current pool of live snapshots (for example every
    capture a :class:`~NanoVNASaver.Traces.TraceCollection` still holds).
    When both original sources are present the operation is replayed with
    exactly the recorded parameters and a *new* derived snapshot is
    produced, which is what "reproducible" means for a memory operation.
    When one is gone the result stays unavailable with the missing ids
    named: never a silent fallback to a different snapshot that happens to
    carry the same channel.
    """
    pool = {snapshot.id: snapshot for snapshot in snapshots}
    availability = check_sources(result, pool)
    if not availability.reproducible:
        # Never fall back to a different snapshot that happens to carry the
        # same channel: the derived trace would silently change meaning.
        return replace(
            result,
            status=TraceMathStatus.UNAVAILABLE,
            snapshot=None,
            reason=availability.reason,
        )
    return compute_trace_math(
        pool[result.first_id],
        pool[result.second_id],
        result.operation,
        channel=result.channel,
        corrected=result.corrected,
    )


def plottable_points(
    result: TraceMathResult, live_ids: Iterable[str]
) -> tuple[TraceMathPoint, ...]:
    """Defined points of ``result``, or ``()`` when its sources are gone."""
    if not result.ok or not check_sources(result, live_ids).reproducible:
        return ()
    return result.defined_points
