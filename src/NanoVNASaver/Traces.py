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
"""Trace collection (TRACE-01) and comparison mathematics (TRACE-03).

A :class:`TraceCollection` owns live and frozen traces.  Each trace carries
its own immutable :class:`MeasurementSnapshot`, so capturing or importing a
trace never mutates another one and later sweeps cannot change a frozen
trace.

Comparison is deliberately explicit about what is being subtracted:

* ``magnitude_db`` subtracts dB values (a ratio of ratios), which is what a
  user expects from two return-loss plots.
* ``complex`` subtracts the complex S-parameters (vector difference).
* ``impedance`` subtracts complex impedances derived from each trace at the
  snapshot's reference impedance.
* ``phase`` subtracts wrapped phase in degrees.

Only overlapping frequencies are compared.  Unequal grids are interpolated
per complex component onto the reference trace's overlapping grid *before*
deriving displayed quantities, and nothing is ever extrapolated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from .Measurement import (
    S11,
    SPARAMETERS,
    MeasurementSnapshot,
)
from .RFTools import Datapoint, magnitude

DEFAULT_COLORS = (
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#9a6324",
)


class TraceKind(Enum):
    """Where a trace's data came from."""

    LIVE = "live"
    """Follows the current acquisition; re-captured on each sweep."""

    FROZEN = "frozen"
    """Captured from a sweep at a point in time and never updated again."""

    IMPORTED = "imported"
    """Loaded from a Touchstone file or restored from a session."""


@dataclass
class Trace:
    """One named, displayable measurement."""

    id: str
    name: str
    snapshot: MeasurementSnapshot
    kind: TraceKind = TraceKind.FROZEN
    color: str = DEFAULT_COLORS[0]
    visible: bool = True
    order: int = 0

    #: Incremented whenever the underlying data changes; used to invalidate
    #: caches (TRACE-04).
    revision: int = 0

    def channel(
        self, parameter: str = S11, corrected: bool = True
    ) -> tuple[Datapoint, ...]:
        return self.snapshot.channel(parameter, corrected) or ()

    def has_channel(self, parameter: str = S11) -> bool:
        return self.snapshot.has(parameter)

    @property
    def simulated(self) -> bool:
        return self.snapshot.simulated

    def frequencies(self, parameter: str = S11) -> tuple[int, ...]:
        return tuple(dp.freq for dp in self.channel(parameter))

    def replace_snapshot(
        self, snapshot: MeasurementSnapshot, *, keep_name: bool = True
    ) -> None:
        self.snapshot = snapshot
        if not keep_name:
            self.name = snapshot.name
        self.revision += 1


class TraceCollection:
    """Ordered collection of traces with a single comparison reference."""

    def __init__(self) -> None:
        self._traces: dict[str, Trace] = {}
        self._order: list[str] = []
        self._reference_id: str = ""
        self._live_id: str = ""
        self._next_index = 1
        self.revision = 0

    # -- iteration -----------------------------------------------------
    def __iter__(self):
        for trace_id in self._order:
            yield self._traces[trace_id]

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, trace_id: object) -> bool:
        return trace_id in self._traces

    def get(self, trace_id: str) -> Trace | None:
        return self._traces.get(trace_id)

    def by_name(self, name: str) -> Trace | None:
        for trace in self:
            if trace.name == name:
                return trace
        return None

    def visible(self) -> list[Trace]:
        return [trace for trace in self if trace.visible]

    @property
    def live_trace(self) -> Trace | None:
        return self._traces.get(self._live_id) if self._live_id else None

    @property
    def reference(self) -> Trace | None:
        return (
            self._traces.get(self._reference_id) if self._reference_id else None
        )

    # -- mutation ------------------------------------------------------
    def next_name(self, base: str = "Trace") -> str:
        existing = {trace.name for trace in self}
        index = self._next_index
        while f"{base} {index}" in existing:
            index += 1
        self._next_index = index + 1
        return f"{base} {index}"

    def add(
        self,
        snapshot: MeasurementSnapshot,
        *,
        name: str | None = None,
        kind: TraceKind = TraceKind.FROZEN,
        trace_id: str | None = None,
    ) -> Trace:
        """Add a new trace.  Never overwrites an existing trace."""
        if len(self._traces) >= 64:
            raise ValueError("too many traces; delete one first")
        trace_id = trace_id or f"trace-{self._next_index}"
        while trace_id in self._traces:
            self._next_index += 1
            trace_id = f"trace-{self._next_index}"
        trace = Trace(
            id=trace_id,
            name=name or self.next_name(),
            snapshot=snapshot,
            kind=kind,
            color=DEFAULT_COLORS[len(self._traces) % len(DEFAULT_COLORS)],
            order=len(self._order),
        )
        self._traces[trace_id] = trace
        self._order.append(trace_id)
        if kind is TraceKind.LIVE:
            self._live_id = trace_id
        if not self._reference_id:
            self._reference_id = trace_id
        self.revision += 1
        return trace

    def replace(self, trace_id: str, snapshot: MeasurementSnapshot) -> Trace:
        """Explicitly replace a trace's data (the only overwrite path)."""
        trace = self._traces[trace_id]
        trace.replace_snapshot(snapshot)
        self.revision += 1
        return trace

    def update_live(self, snapshot: MeasurementSnapshot) -> Trace:
        """Publish a new acquisition into the live trace, creating it.

        Frozen and imported traces are never touched.
        """
        live = self.live_trace
        if live is None:
            return self.add(
                snapshot, name="Live", kind=TraceKind.LIVE, trace_id="live"
            )
        live.replace_snapshot(snapshot)
        self.revision += 1
        return live

    def clear(self) -> None:
        """Remove every trace (session restore, SESSION-03.4)."""
        self._traces.clear()
        self._order.clear()
        self._reference_id = ""
        self._live_id = ""
        self._next_index = 0
        self.revision += 1

    def remove(self, trace_id: str) -> bool:
        if trace_id not in self._traces:
            return False
        self._traces.pop(trace_id)
        self._order.remove(trace_id)
        if self._reference_id == trace_id:
            self._reference_id = self._order[0] if self._order else ""
        if self._live_id == trace_id:
            self._live_id = ""
        for position, remaining in enumerate(self._order):
            self._traces[remaining].order = position
        self.revision += 1
        return True

    def rename(self, trace_id: str, name: str) -> None:
        name = name.strip()
        if not name:
            raise ValueError("trace name must not be empty")
        self._traces[trace_id].name = name
        self.revision += 1

    def set_visible(self, trace_id: str, visible: bool) -> None:
        self._traces[trace_id].visible = visible
        self.revision += 1

    def set_color(self, trace_id: str, color: str) -> None:
        self._traces[trace_id].color = color
        self.revision += 1

    def set_reference(self, trace_id: str) -> None:
        if trace_id and trace_id not in self._traces:
            raise KeyError(trace_id)
        self._reference_id = trace_id
        self.revision += 1

    def reorder(self, trace_id: str, new_index: int) -> None:
        """Move a trace to ``new_index`` (clamped into range)."""
        if trace_id not in self._traces:
            raise KeyError(trace_id)
        self._order.remove(trace_id)
        new_index = max(0, min(new_index, len(self._order)))
        self._order.insert(new_index, trace_id)
        for position, remaining in enumerate(self._order):
            self._traces[remaining].order = position
        self.revision += 1

    def ordered_ids(self) -> list[str]:
        return list(self._order)

    def comparison_candidates(self, parameter: str = S11) -> list[Trace]:
        """Traces that actually carry ``parameter``.

        Used so charts can show a clear limitation instead of an error when a
        trace has different channels (TRACE-02.4).
        """
        return [trace for trace in self if trace.has_channel(parameter)]


# ----------------------------------------------------------------------
# Comparison mathematics (TRACE-03)
# ----------------------------------------------------------------------
class ComparisonKind(Enum):
    MAGNITUDE_DB = "magnitude_db"
    PHASE = "phase"
    IMPEDANCE = "impedance"
    COMPLEX = "complex"

    @property
    def unit(self) -> str:
        return {
            ComparisonKind.MAGNITUDE_DB: "dB",
            ComparisonKind.PHASE: "deg",
            ComparisonKind.IMPEDANCE: "ohm",
            ComparisonKind.COMPLEX: "",
        }[self]

    @property
    def label(self) -> str:
        return {
            ComparisonKind.MAGNITUDE_DB: "Magnitude (dB)",
            ComparisonKind.PHASE: "Phase (deg)",
            ComparisonKind.IMPEDANCE: "Impedance (ohm)",
            ComparisonKind.COMPLEX: "Complex S",
        }[self]


@dataclass(frozen=True)
class ComparisonPoint:
    """One compared frequency."""

    freq: int
    value: complex
    """Difference; the imaginary part is zero for scalar kinds."""

    @property
    def scalar(self) -> float:
        return self.value.real


@dataclass(frozen=True)
class ComparisonResult:
    kind: ComparisonKind
    parameter: str
    source_trace: str
    reference_trace: str
    unit: str
    points: tuple[ComparisonPoint, ...]
    interpolated: bool = False
    unavailable_reason: str = ""

    @property
    def available(self) -> bool:
        return not self.unavailable_reason and bool(self.points)

    def values(self) -> list[float]:
        return [point.scalar for point in self.points]

    def frequencies(self) -> list[int]:
        return [point.freq for point in self.points]

    def span(self) -> tuple[int, int]:
        """Compared frequency span as ``(start, stop)`` in Hz."""
        frequencies = self.frequencies()
        if not frequencies:
            return (0, 0)
        return (frequencies[0], frequencies[-1])

    def span_label(self) -> str:
        """Human span description, so two curves are never confused.

        Comparisons are restricted to the overlap of the two traces, so a
        reader must be told which span the numbers cover (TRACE-02.4).
        """
        start, stop = self.span()
        if not start and not stop:
            return "no span"
        return f"{start / 1e6:.4g}-{stop / 1e6:.4g} MHz"

    def summary(self) -> str:
        """Compact labelled summary (TRACE-03.4)."""
        if not self.available:
            detail = (
                f" ({self.unavailable_reason})"
                if self.unavailable_reason
                else ""
            )
            return (
                f"{self.parameter}: {self.source_trace} vs"
                f" {self.reference_trace} unavailable{detail}"
            )
        values = self.values()
        # Undefined samples define no range: ``min``/``max`` would answer NaN
        # whenever one sits at the start of the sweep, so the summary would
        # depend on sweep order.  Infinities stay -- a zero magnitude really
        # is an infinite dB difference.
        measured = [value for value in values if not math.isnan(value)]
        interpolation = " (interpolated)" if self.interpolated else ""
        prefix = (
            f"{self.parameter} {self.kind.label}:"
            f" {self.source_trace} - {self.reference_trace}"
            f"{interpolation}"
        )
        if not measured:
            return (
                f"{prefix}: no measured values,"
                f" {len(values)} points, {self.span_label()}"
            )
        return (
            f"{prefix}: min {min(measured):.4g} {self.unit},"
            f" max {max(measured):.4g} {self.unit},"
            f" {len(values)} points, {self.span_label()}"
        )


def _unavailable(
    kind: ComparisonKind,
    parameter: str,
    source: str,
    reference: str,
    reason: str,
) -> ComparisonResult:
    return ComparisonResult(
        kind=kind,
        parameter=parameter,
        source_trace=source,
        reference_trace=reference,
        unit=kind.unit,
        points=(),
        unavailable_reason=reason,
    )


def _interleave(
    grid: Sequence[int],
    first: Sequence[Datapoint],
    second: Sequence[Datapoint],
) -> tuple[list[complex], list[complex]]:
    """Sample both channels at ``grid`` using only interpolation.

    Both interpolators are evaluated over the whole grid at once: a scalar
    call per point made a 10,000-point comparison take over a second
    (TRACE-04.2).
    """
    from scipy.interpolate import interp1d

    targets = list(grid)

    def build(points: Sequence[Datapoint]) -> list[complex]:
        freqs = [dp.freq for dp in points]
        reals = interp1d(freqs, [dp.re for dp in points], kind="linear")
        imags = interp1d(freqs, [dp.im for dp in points], kind="linear")
        real_values = reals(targets)
        imag_values = imags(targets)
        return [
            complex(float(re_value), float(im_value))
            for re_value, im_value in zip(real_values, imag_values, strict=True)
        ]

    return build(first), build(second)


def _sample(points: Sequence[Datapoint], grid: Sequence[int]) -> list[complex]:
    lookup = {dp.freq: dp.z for dp in points}
    return [lookup[freq] for freq in grid]


def _wrap_degrees(degrees: float) -> float:
    """Wrap to (-180, 180]."""
    wrapped = math.fmod(degrees + 180.0, 360.0)
    if wrapped <= 0:
        wrapped += 360.0
    return wrapped - 180.0


def compare_traces(
    source: Trace,
    reference: Trace,
    *,
    parameter: str = S11,
    kind: ComparisonKind = ComparisonKind.MAGNITUDE_DB,
    corrected: bool = True,
) -> ComparisonResult:
    """Compare two traces over their overlapping frequencies.

    Duplicate frequencies: the *last* occurrence wins consistently for both
    traces because a frequency lookup dictionary is used.  Disjoint spans or
    missing channels produce an unavailable result instead of invented data.
    """
    if parameter not in SPARAMETERS:
        raise ValueError(f"unsupported parameter {parameter!r}")
    if not source.has_channel(parameter):
        return _unavailable(
            kind,
            parameter,
            source.name,
            reference.name,
            f"{source.name} has no {parameter}",
        )
    if not reference.has_channel(parameter):
        return _unavailable(
            kind,
            parameter,
            source.name,
            reference.name,
            f"{reference.name} has no {parameter}",
        )

    source_points = source.channel(parameter, corrected)
    reference_points = reference.channel(parameter, corrected)
    source_freqs = {dp.freq for dp in source_points}
    reference_freqs = {dp.freq for dp in reference_points}

    # Hoist the bounds: recomputing min/max per frequency is O(n^2) and made
    # a 10,000-point comparison take over a second (TRACE-04.2).
    source_min = min(source_freqs)
    source_max = max(source_freqs)
    reference_min = min(reference_freqs)
    reference_max = max(reference_freqs)

    # Overlap is about *spans*, not shared grid points.  Two traces sampled
    # on different grids (say 100/200/300 and 150/250/350 Hz) overlap over
    # 150-300 Hz and can be compared by interpolation; requiring an exact
    # frequency match wrongly reported "no overlapping frequencies".
    if source_max < reference_min or reference_max < source_min:
        return _unavailable(
            kind,
            parameter,
            source.name,
            reference.name,
            "no overlapping frequencies",
        )

    # Compare on the reference trace's grid, restricted to the span where
    # the source also has data.  Nothing outside that intersection is used,
    # so no value is ever extrapolated.
    low = max(source_min, reference_min)
    high = min(source_max, reference_max)
    grid = sorted(freq for freq in reference_freqs if low <= freq <= high)
    if not grid:
        return _unavailable(
            kind,
            parameter,
            source.name,
            reference.name,
            "no overlapping frequencies",
        )

    interpolated = any(freq not in source_freqs for freq in grid)
    if interpolated:
        left, right = _interleave(grid, source_points, reference_points)
    else:
        left = _sample(source_points, grid)
        right = _sample(reference_points, grid)

    if (
        source.snapshot.reference_impedance
        != reference.snapshot.reference_impedance
    ):
        return _unavailable(
            kind,
            parameter,
            source.name,
            reference.name,
            "different reference impedances"
            f" ({source.snapshot.reference_impedance:g} vs"
            f" {reference.snapshot.reference_impedance:g} ohm)",
        )

    impedance = source.snapshot.reference_impedance
    points: list[ComparisonPoint] = []
    for freq, left_value, right_value in zip(grid, left, right, strict=True):
        points.append(
            ComparisonPoint(
                freq,
                _difference(kind, left_value, right_value, freq, impedance),
            )
        )

    return ComparisonResult(
        kind=kind,
        parameter=parameter,
        source_trace=source.name,
        reference_trace=reference.name,
        unit=kind.unit,
        points=tuple(points),
        interpolated=interpolated,
    )


def _difference(
    kind: ComparisonKind,
    left: complex,
    right: complex,
    freq: int,
    impedance: float,
) -> complex:
    if kind is ComparisonKind.COMPLEX:
        return left - right
    if kind is ComparisonKind.PHASE:
        left_deg = math.degrees(math.atan2(left.imag, left.real))
        right_deg = math.degrees(math.atan2(right.imag, right.real))
        return complex(_wrap_degrees(left_deg - right_deg), 0.0)
    if kind is ComparisonKind.IMPEDANCE:
        return _to_impedance(left, impedance) - _to_impedance(right, impedance)
    # Magnitude in dB.  Subtraction of dB values is a ratio of magnitudes;
    # a zero magnitude yields -inf and therefore an infinite difference.
    left_mag = magnitude(left)
    right_mag = magnitude(right)
    if left_mag == 0.0 or right_mag == 0.0:
        if left_mag == right_mag:
            return complex(0.0, 0.0)
        return complex(math.inf, 0.0)
    return complex(20.0 * math.log10(left_mag / right_mag), 0.0)


def _to_impedance(gamma: complex, reference: float) -> complex:
    denominator = 1.0 - gamma
    if denominator == 0:
        return complex(math.inf, 0.0)
    return reference * (1.0 + gamma) / denominator


def marker_delta(result: ComparisonResult, freq: int) -> ComparisonPoint | None:
    """Nearest compared frequency to ``freq`` (marker deltas, TRACE-03.4)."""
    if not result.points:
        return None
    return min(result.points, key=lambda point: abs(point.freq - freq))


@dataclass
class ComparisonCache:
    """Caches comparison results by trace revision and settings (TRACE-04.2)."""

    _entries: dict[tuple, ComparisonResult] = field(default_factory=dict)

    def get(
        self,
        source: Trace,
        reference: Trace,
        parameter: str,
        kind: ComparisonKind,
    ) -> ComparisonResult:
        key = (
            source.id,
            source.revision,
            reference.id,
            reference.revision,
            parameter,
            kind,
        )
        cached = self._entries.get(key)
        if cached is None:
            cached = compare_traces(
                source, reference, parameter=parameter, kind=kind
            )
            self._entries[key] = cached
        return cached

    def invalidate(self, trace_id: str | None = None) -> None:
        if trace_id is None:
            self._entries.clear()
            return
        for key in list(self._entries):
            if trace_id in key:
                del self._entries[key]
