#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020  Rune B. Broberg
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
"""Antenna tuning workflow primitives (ANT-01).

Pure, GUI-free helpers shared by the antenna tuning window and its tests:

* :class:`DeviceLimits` - what the *connected* device can actually sweep;
* :func:`plan_sweep` / :func:`plan_zoom` - validate a broad overview or a
  narrower zoom and coerce it into a legal sweep for this device;
* :func:`find_vswr_minimum` - the minimum VSWR **of the measured samples**,
  including every degenerate case (empty, one point, endpoint, non-finite);
* filename and Touchstone export helpers.

Nothing here measures, talks to hardware, or invents samples: a plan is a
request that the normal :class:`~NanoVNASaver.SweepWorker.SweepWorker` then
executes, and an export only ever writes the samples that were measured.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Iterable

from .Formatting import format_frequency, format_gain, format_vswr
from .Hardware.Capability import SweepCapability
from .Measurement import (
    CompletionState,
    MeasurementSnapshot,
    snapshot_to_touchstone,
)
from .RFTools import Datapoint
from .Settings.Sweep import Properties, Sweep, SweepMode
from .Touchstone import Touchstone, TouchstoneExportError

#: Point counts below this cannot describe a frequency span, so a sweep
#: that must contain a minimum needs at least this many samples.
MIN_POINTS = 2

#: Hard ceiling on segments, matching the preset validator.  Large segment
#: counts are slow and are never what a tuning zoom needs.
MAX_SEGMENTS = 1024

#: Width, in Hz, of the narrowest zoom that the planner will accept on top
#: of the point-count requirement.  A step smaller than this is not a
#: meaningful measurement on any supported device.
MIN_ZOOM_SPAN_HZ = 10


class SweepPlanError(ValueError):
    """A requested span cannot be measured with this device."""


class AntennaExportError(Exception):
    """An antenna measurement cannot be exported as requested."""


@dataclass(frozen=True)
class DeviceLimits:
    """Sweep constraints read from the connected device.

    ``max_frequency`` is ``None`` when the device does not advertise a
    ceiling; in that case only the lower bound and the step constraint are
    enforced rather than inventing a limit.
    """

    min_frequency: int = 1
    max_frequency: int | None = None
    points: tuple[int, ...] = ()
    default_points: int = 101
    max_segments: int = MAX_SEGMENTS

    @classmethod
    def from_vna(cls, vna) -> "DeviceLimits":
        """Read the limits the device itself declares.

        Delegates to :meth:`SweepCapability.from_vna`, so the tuning window
        and the preset/sweep planners read the very same interpretation of
        ``sweep_max_freq_hz`` (``0.0`` means "no ceiling advertised", never
        "no frequencies allowed") and of ``valid_datapoints``.
        """
        capability = SweepCapability.from_vna(vna)
        valid = tuple(
            count for count in capability.valid_points if count >= MIN_POINTS
        )
        default_points = capability.default_points
        if not valid:
            valid = (default_points,)
        return cls(
            min_frequency=capability.min_frequency,
            max_frequency=capability.max_frequency,
            points=valid,
            default_points=default_points,
            max_segments=MAX_SEGMENTS,
        )

    def capability(self) -> SweepCapability:
        """These limits in the shared sweep-capability form.

        Derived from the fields themselves, so limits built by hand (as the
        tests do) and limits read from a device validate identically.
        """
        return SweepCapability(
            min_frequency=self.min_frequency,
            max_frequency=self.max_frequency,
            valid_points=tuple(self.points),
            default_points=self.default_points,
            min_points=MIN_POINTS,
            max_segments=self.max_segments,
        )

    def describe(self) -> str:
        if self.max_frequency is None:
            upper = "unknown upper limit"
        else:
            upper = format_frequency(self.max_frequency)
        counts = (
            ", ".join(str(count) for count in self.points)
            if self.points
            else str(self.default_points)
        )
        return (
            f"Device range: {format_frequency(self.min_frequency)} - {upper};"
            f" supported points: {counts}"
        )


@dataclass(frozen=True)
class SweepPlan:
    """A validated sweep request for the normal acquisition path."""

    start: int
    end: int
    points: int
    segments: int = 1
    logarithmic: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def span(self) -> int:
        return self.end - self.start

    @property
    def divisor(self) -> int:
        """Number of steps between the first and last sample of the grid."""
        return self.points * self.segments - 1

    @property
    def step(self) -> int:
        """Step the device really uses, or ``0`` when the plan is degenerate."""
        divisor = self.divisor
        return round(self.span / divisor) if divisor > 0 else 0

    def describe(self) -> str:
        text = (
            f"{format_frequency(self.start)} - {format_frequency(self.end)},"
            f" {self.points} points, {self.segments} segment(s)"
        )
        if self.logarithmic:
            text += ", logarithmic"
        return text


def choose_points(limits: DeviceLimits, requested: int | None) -> int:
    """Pick a point count the device really offers.

    An unsupported request is replaced with the nearest supported count
    (preferring the larger one on a tie) so a zoom never changes the
    device's point configuration behind the user's back.
    """
    if not limits.points:
        if requested is None:
            return max(limits.default_points, MIN_POINTS)
        return max(int(requested), MIN_POINTS)
    if requested is not None and int(requested) in limits.points:
        return int(requested)
    if requested is None:
        if limits.default_points in limits.points:
            return limits.default_points
        return limits.points[0]
    target = int(requested)
    return min(limits.points, key=lambda count: (abs(count - target), -count))


def sweep_grid(
    start: int,
    end: int,
    points: int,
    segments: int = 1,
    *,
    logarithmic: bool = False,
) -> tuple[int, ...]:
    """The exact frequency grid the normal :class:`Sweep` measures.

    The acquisition contract is not ``(points - 1) * segments`` evenly
    spaced samples: :class:`Sweep` rounds a *single* step for the whole
    sweep and lays the segments out end to end.  Planning is validated
    against that arithmetic, because a plan accepted on the nominal span
    alone can measure frequencies outside the device range.
    """
    try:
        sweep = Sweep(
            int(start),
            int(end),
            int(points),
            int(segments),
            Properties(
                name="", mode=SweepMode.SINGLE, logarithmic=bool(logarithmic)
            ),
        )
        return tuple(sweep.get_frequencies())
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise SweepPlanError(f"illegal sweep settings: {exc}") from exc


def _checked_grid(  # noqa: PLR0913 - keyword-only, internal
    start: int,
    end: int,
    points: int,
    segments: int,
    limits: DeviceLimits,
    *,
    logarithmic: bool,
) -> tuple[int, ...]:
    """The measured grid of a plan, or ``SweepPlanError`` when it is unusable.

    A legal plan must start exactly at its start, stop at or before its stop,
    stay inside the device range, and produce an ordered, non-degenerate
    grid.  The last condition is what a nominal ``(points - 1) * segments``
    span check misses: :class:`Sweep` rounds one step for the whole sweep,
    so a span barely wider than the step count collapses samples onto one
    frequency.  Logarithmic multi-segment sweeps legitimately repeat the
    shared segment boundary, so the grid only has to be non-decreasing with
    at least :data:`MIN_POINTS` distinct frequencies.
    """
    grid = sweep_grid(start, end, points, segments, logarithmic=logarithmic)
    if len(grid) < MIN_POINTS:
        raise SweepPlanError(
            f"{points} points in {segments} segment(s) give fewer than"
            f" {MIN_POINTS} samples"
        )
    lower = limits.min_frequency
    upper = limits.max_frequency
    if grid[0] < lower or (upper is not None and grid[-1] > upper):
        raise SweepPlanError(
            f"the planned grid {grid[0]} - {grid[-1]} Hz leaves the device"
            f" range ({limits.describe()})"
        )
    if grid[0] != start or grid[-1] > end:
        raise SweepPlanError(
            f"the planned grid {grid[0]} - {grid[-1]} Hz is not inside the"
            f" requested {start} - {end} Hz"
        )
    if any(b < a for a, b in pairwise(grid)):
        raise SweepPlanError(
            f"{points} points in {segments} segment(s) do not give an"
            " ordered grid; widen the span or lower the segment count"
        )
    if len(set(grid)) < MIN_POINTS:
        raise SweepPlanError(
            f"{points} points in {segments} segment(s) collapse onto"
            f" {len(set(grid))} distinct frequency(ies); the span is too"
            " narrow for that configuration"
        )
    # Cross-check the accepted grid against the shared capability model, so
    # the tuning planner and every other planner agree on what is legal.
    validation = limits.capability().validate_grid(
        start, end, points, segments, logarithmic=logarithmic
    )
    if not validation.ok:
        raise SweepPlanError(validation.problem.message)
    return grid


def _attempt_grid(  # noqa: PLR0913 - keyword-only, internal
    start: int,
    end: int,
    points: int,
    segments: int,
    limits: DeviceLimits,
    *,
    logarithmic: bool,
    may_shorten: bool,
) -> tuple[int, tuple[int, ...]]:
    """The grid the device measures for ``(start, end)``, or a refusal.

    :class:`Sweep` rounds one absolute step for the whole sweep, so the
    measured grid reaches the requested stop within one step - but a rounded
    step can also carry it past the request and past the device maximum.

    * A grid that stays inside the request, and reaches within one step of
      it, is accepted unchanged: the requested stop is kept because the
      device reproduces exactly that grid.
    * Otherwise, with ``may_shorten``, the longest grid inside the request
      and the device maximum is used and the plan reports its stop.  A
      shortening beyond one step is refused: that would be a different
      measurement rather than a rounding detail.
    """
    grid = sweep_grid(start, end, points, segments, logarithmic=logarithmic)
    if grid[0] != start:
        raise SweepPlanError(
            f"the planned grid starts at {grid[0]} Hz instead of {start} Hz"
        )
    divisor = points * segments - 1
    upper = limits.max_frequency
    measured_end = grid[-1]
    step = (measured_end - grid[0]) // divisor if divisor > 0 else 0
    inside = (upper is None or measured_end <= upper) and measured_end <= end
    if inside and end - measured_end <= max(step, 1):
        return end, _checked_grid(
            start,
            end,
            points,
            segments,
            limits,
            logarithmic=logarithmic,
        )
    if not may_shorten or logarithmic or divisor <= 0:
        raise SweepPlanError(
            f"a {start} - {end} Hz sweep with {points} points in"
            f" {segments} segment(s) would sample up to {measured_end} Hz,"
            " outside the requested or device range"
        )
    ceiling = end if upper is None else min(end, upper)
    for candidate_step in range(round((end - start) / divisor), 0, -1):
        candidate_end = start + divisor * candidate_step
        if candidate_end > ceiling:
            continue
        if end - candidate_end > candidate_step:
            break
        return candidate_end, _checked_grid(
            start,
            candidate_end,
            points,
            segments,
            limits,
            logarithmic=False,
        )
    raise SweepPlanError(
        f"a {start} - {end} Hz sweep with {points} points in {segments}"
        " segment(s) cannot be measured on the device's step grid without"
        " leaving the requested or device range; use another point count,"
        " span or segment count"
    )


def _fit_plan(  # noqa: PLR0913 - keyword-only, internal
    start: int,
    end: int,
    points: int,
    segments: int,
    limits: DeviceLimits,
    *,
    logarithmic: bool,
) -> tuple[int, int, tuple[int, ...]]:
    """Find the largest requested segment count whose real grid is legal.

    Two passes: first a segment count whose grid lands inside the request
    without moving the stop, then - only if none does - a count whose stop
    has to be quantized onto the measured grid.  Only the segment count is
    ever reduced, and only down to one: a plan that cannot be measured at
    all is refused rather than rescaled into a different measurement.
    """
    structural = max(1, (end - start) // max(points - 1, 1))
    candidate = max(1, min(int(segments), structural, limits.max_segments))
    reason = (
        f"a {format_frequency(end - start)} span is too narrow for"
        f" {points} points"
    )
    for may_shorten in (False, True):
        for count in range(candidate, 0, -1):
            try:
                measured_end, grid = _attempt_grid(
                    start,
                    end,
                    points,
                    count,
                    limits,
                    logarithmic=logarithmic,
                    may_shorten=may_shorten,
                )
            except SweepPlanError as exc:
                reason = str(exc)
                continue
            return measured_end, count, grid
    raise SweepPlanError(reason)


def plan_sweep(  # noqa: PLR0913 - keyword-only, single call site
    start: int,
    end: int,
    *,
    limits: DeviceLimits,
    points: int | None = None,
    segments: int = 1,
    logarithmic: bool = False,
) -> SweepPlan:
    """Validate a broad overview sweep against the device limits.

    Out-of-range frequencies are clamped to the device range and reported
    in ``notes``; a span that cannot be measured is refused instead of being
    silently turned into a different measurement.  The accepted plan is
    checked against the grid the device really measures, so the plan's stop
    is the measured stop and can never lie beyond the device maximum.
    """
    try:
        start = int(start)
        end = int(end)
        segments = int(segments)
    except (TypeError, ValueError) as exc:
        raise SweepPlanError("sweep bounds must be whole numbers") from exc
    if end <= start:
        raise SweepPlanError(
            "stop frequency must be greater than start frequency"
        )
    notes: list[str] = []
    lower = limits.min_frequency
    upper = limits.max_frequency
    requested_end = end
    if start < lower:
        notes.append(
            f"start clamped to the device minimum {format_frequency(lower)}"
        )
        start = lower
    if upper is not None and end > upper:
        notes.append(
            f"stop clamped to the device maximum {format_frequency(upper)}"
        )
        end = upper
    if end <= start:
        raise SweepPlanError(
            "the requested span lies outside the device frequency range"
            f" ({(limits.describe())})"
        )
    selected = choose_points(limits, points)
    if selected != (points if points is None else int(points)):
        notes.append(
            f"{points} points are not supported by the device; using {selected}"
        )
    if segments < 1:
        raise SweepPlanError("segment count must be at least 1")
    if logarithmic and start <= 0:
        raise SweepPlanError("a logarithmic sweep needs a positive start")
    measured_end, fitted, _grid = _fit_plan(
        start,
        end,
        selected,
        segments,
        limits,
        logarithmic=bool(logarithmic),
    )
    if fitted != segments:
        notes.append(
            f"{segments} segments do not fit this span on the device's step"
            f" grid; using {fitted}"
        )
    if measured_end != end:
        notes.append(
            f"stop quantized to the measured grid:"
            f" {format_frequency(measured_end)} (requested"
            f" {format_frequency(requested_end)})"
        )
    return SweepPlan(
        start=start,
        end=measured_end,
        points=selected,
        segments=fitted,
        logarithmic=bool(logarithmic),
        notes=tuple(notes),
    )


def plan_zoom(
    center: int,
    span: int,
    *,
    limits: DeviceLimits,
    points: int | None = None,
    segments: int = 1,
) -> SweepPlan:
    """Centre a narrow sweep on ``center`` and clamp it into device range.

    Rejects spans that are non-positive, non-finite, wider than the whole
    device range, or too narrow for the requested point count.  Clamping
    only moves the requested window; it never adds measurements or shifts
    the minimum itself, and the accepted window is quantized onto the grid
    the device really measures.
    """
    try:
        center = int(center)
        span = int(span)
        segments = int(segments)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SweepPlanError(
            "zoom centre and span must be whole numbers"
        ) from exc
    if span <= 0:
        raise SweepPlanError("zoom span must be greater than zero")
    upper = limits.max_frequency
    lower = limits.min_frequency
    if upper is not None and not lower <= center <= upper:
        raise SweepPlanError(
            f"the minimum at {format_frequency(center)} lies outside the"
            f" device range ({limits.describe()})"
        )
    if center < lower:
        raise SweepPlanError(
            f"the minimum at {format_frequency(center)} lies below the"
            f" device minimum {format_frequency(lower)}"
        )
    notes: list[str] = []
    device_span = None if upper is None else upper - lower
    if device_span is not None and span > device_span:
        raise SweepPlanError(
            f"zoom span {format_frequency(span)} is wider than the device"
            f" range {format_frequency(device_span)}"
        )
    selected = choose_points(limits, points)
    required = max(selected - 1, 1)
    if span < required:
        raise SweepPlanError(
            f"zoom span {format_frequency(span)} is too narrow for"
            f" {selected} points; at least {format_frequency(required)}"
            " is needed"
        )
    if segments < 1:
        raise SweepPlanError("segment count must be at least 1")
    structural = max(1, span // required)
    fitted = max(1, min(int(segments), structural, limits.max_segments))
    if fitted != segments:
        raise SweepPlanError(
            f"{segments} segments leave less than one step per point for"
            f" {format_frequency(span)}"
        )
    divisor = selected * fitted - 1
    step = span // divisor
    if step < 1:
        raise SweepPlanError(
            f"zoom span {format_frequency(span)} is too narrow for"
            f" {selected} points; at least {format_frequency(required)}"
            " is needed"
        )
    measured_span = divisor * step
    if measured_span != span:
        notes.append(
            f"zoom span quantized to the measured grid:"
            f" {format_frequency(measured_span)} (requested"
            f" {format_frequency(span)})"
        )
    start = center - measured_span // 2
    end = start + measured_span
    if start < lower:
        notes.append(
            f"zoom clamped to the device minimum {format_frequency(lower)}"
        )
        start = lower
        end = start + measured_span
    if upper is not None and end > upper:
        notes.append(
            f"zoom clamped to the device maximum {format_frequency(upper)}"
        )
        end = upper
        start = max(lower, end - measured_span)
    if not start <= center <= end:
        # Only reachable if the device range itself cannot hold the window,
        # which the device_span check above already rejected.
        raise SweepPlanError("the zoom window does not contain the minimum")
    _checked_grid(start, end, selected, fitted, limits, logarithmic=False)
    return SweepPlan(
        start=start,
        end=end,
        points=selected,
        segments=fitted,
        logarithmic=False,
        notes=tuple(notes),
    )


def sweep_vswr(point: Datapoint) -> float | None:
    """VSWR of one sample, or ``None`` when the sample is unusable.

    A reflection coefficient of magnitude >= 1 is a legal (very poor)
    measurement and yields an infinite VSWR rather than being discarded.
    """
    real = float(point.re)
    imag = float(point.im)
    if not (math.isfinite(real) and math.isfinite(imag)):
        return None
    magnitude = math.hypot(real, imag)
    if magnitude >= 1.0:
        return math.inf
    return (1.0 + magnitude) / (1.0 - magnitude)


def return_loss(point: Datapoint) -> float | None:
    """Standard return loss of one sample: ``-20*log10(|S11|)`` in dB.

    Positive for a passive match (``|S11| = 0.1`` -> ``+20 dB``), ``0 dB``
    at total reflection and ``+inf`` for an ideal match, matching the
    instrument convention in Keysight application note 5952-3706 section
    3.4.3.3.  A ``|S11| > 1`` sample belongs to an active device and yields
    a negative return loss, which is reported as measured rather than
    clamped.  ``None`` means the sample is not finite.
    """
    real = float(point.re)
    imag = float(point.im)
    if not (math.isfinite(real) and math.isfinite(imag)):
        return None
    magnitude = math.hypot(real, imag)
    if magnitude == 0.0:
        return math.inf  # an ideal match: no reflected power
    # ``+ 0.0`` turns the ``-0.0`` of ``|S11| == 1`` into ``0.0``: total
    # reflection is exactly 0 dB, not "-0.000 dB".
    return -20.0 * math.log10(magnitude) + 0.0


@dataclass(frozen=True)
class MinimumPoint:
    """The measured sample with the lowest VSWR."""

    index: int
    freq: int
    vswr: float
    s11: complex
    return_loss: float | None
    at_edge: bool
    span_start: int
    span_end: int
    samples: int

    def describe(self) -> str:
        text = (
            f"Min VSWR {format_vswr(self.vswr)} @ {format_frequency(self.freq)}"
        )
        if self.return_loss is None:
            pass
        elif math.isfinite(self.return_loss):
            text += f" ({format_gain(self.return_loss)} return loss)"
        else:
            text += " (return loss above the measurable range: ideal match)"
        if self.at_edge:
            text += " - at the sweep edge, the dip may lie outside"
        return text


@dataclass(frozen=True)
class MinimumSearch:
    """Outcome of looking for the lowest measured VSWR."""

    minimum: MinimumPoint | None
    samples: int
    skipped: int
    reason: str = ""

    @property
    def found(self) -> bool:
        return self.minimum is not None

    def describe(self) -> str:
        if self.minimum is None:
            return self.reason or "no usable S11 samples"
        text = self.minimum.describe()
        if self.skipped:
            text += f" ({self.skipped} non-finite sample(s) ignored)"
        return text


def find_vswr_minimum(
    samples: Iterable[Datapoint] | None,
) -> MinimumSearch:
    """Lowest VSWR among the *measured* samples.

    The winning sample is selected by value, never by a fixed index, and its
    index is reported so the caller can place a marker on that sample.  A
    minimum sitting on the first or last sample is reported as an edge
    minimum: it means the true dip may be outside the measured span.
    """
    points = list(samples or ())
    total = len(points)
    if total == 0:
        return MinimumSearch(None, 0, 0, "no S11 samples measured")
    best_index = -1
    best_vswr = math.inf
    skipped = 0
    for index, point in enumerate(points):
        value = sweep_vswr(point)
        if value is None:
            skipped += 1
            continue
        if value < best_vswr:
            best_index = index
            best_vswr = value
    if best_index < 0:
        reason = (
            "every measured S11 sample is non-finite"
            if skipped == total
            else "every measured S11 sample has |S11| >= 1, so VSWR is infinite"
        )
        return MinimumSearch(None, total, skipped, reason)
    best = points[best_index]
    return MinimumSearch(
        minimum=MinimumPoint(
            index=best_index,
            freq=int(best.freq),
            vswr=best_vswr,
            s11=best.z,
            return_loss=return_loss(best),
            at_edge=best_index in (0, total - 1),
            span_start=int(points[0].freq),
            span_end=int(points[-1].freq),
            samples=total,
        ),
        samples=total,
        skipped=skipped,
    )


def auto_zoom_span(measured_span: int, points: int) -> int:
    """A sane default zoom span: one twentieth of what was just measured."""
    narrowest = max(int(points) - 1, MIN_ZOOM_SPAN_HZ)
    if measured_span <= 0:
        return narrowest
    return max(narrowest, min(int(measured_span), int(measured_span) // 20))


def tuning_filename(
    resonance_freq: int | None,
    when: datetime | None = None,
    prefix: str = "antenna",
) -> str:
    """Descriptive ``.s1p`` name carrying the timestamp and resonance.

    ``antenna_14p150MHz_20261101-123015.s1p`` - no character that is
    awkward in a filename, and the resonance is readable without opening
    the file.
    """
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    parts = [prefix]
    if resonance_freq and resonance_freq > 0:
        megahertz = f"{resonance_freq / 1e6:.3f}".replace(".", "p")
        parts.append(f"{megahertz}MHz")
    parts.append(stamp)
    return "_".join(parts) + ".s1p"


def unique_path(path: Path | str, *, suffix: str = ".s1p") -> Path:
    """``path`` or ``path_2.s1p``, ``path_3.s1p`` ... never an existing file."""
    target = Path(path)
    if target.suffix != suffix:
        target = target.with_suffix(suffix)
    if not target.exists():
        return target
    stem = target.stem
    index = 2
    while True:
        candidate = target.with_name(f"{stem}_{index}{target.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _provenance_comments(
    snapshot: MeasurementSnapshot,
    minimum: MinimumPoint | None,
) -> list[str]:
    """Honest, human-readable notes for an exported antenna measurement."""
    provenance = snapshot.provenance
    sweep = provenance.sweep
    lines = [
        "! NanoVNASaver antenna tuning measurement",
    ]
    if snapshot.simulated:
        lines.append(
            "! simulated data (scenario:"
            f" {provenance.simulator_scenario or 'n/a'}) - not a physical"
            " measurement"
        )
    channel = snapshot.channel("S11") or ()
    if sweep is not None:
        mode = sweep.mode
        if sweep.logarithmic:
            mode += ", logarithmic"
        lines.append(
            f"! sweep: {sweep.start} - {sweep.end} Hz, {sweep.points} points,"
            f" {sweep.segments} segment(s), mode {mode},"
            f" averages {sweep.averages}"
        )
    if channel:
        lines.append(
            f"! measured S11: {len(channel)} samples,"
            f" {channel[0].freq} - {channel[-1].freq} Hz"
        )
    if minimum is not None:
        edge = (
            " (at the sweep edge: the true minimum may lie outside this span)"
            if minimum.at_edge
            else ""
        )
        line = (
            f"! minimum VSWR {minimum.vswr:.6g} at {minimum.freq} Hz on the"
            f" measured sample #{minimum.index}{edge}"
        )
        # Standard return loss: positive for a passive match, 0 dB at total
        # reflection, unbounded for an ideal match (Keysight 5952-3706
        # 3.4.3.3).  Stated in the comment so no reader has to guess a sign
        # convention from the number alone.
        if minimum.return_loss is not None and math.isfinite(
            minimum.return_loss
        ):
            line += (
                f", return loss -20*log10(|S11|) = {minimum.return_loss:.3f} dB"
            )
        elif minimum.return_loss is not None:
            line += ", return loss -20*log10(|S11|) is unbounded (ideal match)"
        lines.append(line)
    completion = provenance.completion
    if completion is CompletionState.COMPLETE:
        lines.append("! completion: complete sweep")
    else:
        lines.append(
            f"! INCOMPLETE: {completion.value} sweep - this file is not a"
            " completed measurement"
        )
    correction = snapshot.correction
    if correction.applied:
        lines.append(
            "! correction: applied by the application"
            f" (identity {correction.identity or 'unknown'})"
            + (
                f", offset delay {correction.offset_delay * 1e12:g} ps"
                if correction.offset_delay
                else ""
            )
        )
    else:
        lines.append("! correction: none applied - values are uncorrected")
    if provenance.device:
        lines.append(f"! device: {provenance.device}")
    if provenance.source and not snapshot.simulated:
        lines.append(f"! source: {provenance.source}")
    return lines


def measurement_touchstone(
    snapshot: MeasurementSnapshot,
    minimum: MinimumPoint | None = None,
) -> Touchstone:
    """Adapt a snapshot to a one-port Touchstone carrying its provenance."""
    touchstone = snapshot_to_touchstone(snapshot, corrected=True)
    touchstone.filename = ""
    touchstone.simulated = bool(snapshot.simulated)
    touchstone.simulated_scenario = str(
        snapshot.provenance.simulator_scenario or ""
    )
    # ``snapshot_to_touchstone`` states the simulated/source fact once;
    # ``_provenance_comments`` states it again with the antenna context, so
    # the generic line is dropped rather than repeated twice in one file.
    touchstone.comments = [
        line
        for line in touchstone.comments
        if line.startswith("!")
        and line.strip() != f"! source: {snapshot.provenance.source}"
        and not line.startswith("! simulated measurement (")
    ]
    touchstone.comments.extend(_provenance_comments(snapshot, minimum))
    return touchstone


def normalize_s1p_path(path: Path | str) -> Path:
    """The file ``write_s1p`` really writes for ``path``.

    Callers must check and confirm *this* path: a dialog accepting
    ``named.s2p`` while the writer quietly produces ``named.s1p`` would
    otherwise authorise replacing a different existing file than the one
    that gets destroyed.
    """
    target = Path(path)
    if target.suffix.lower() == ".s1p":
        return target
    return target.with_suffix(".s1p")


def _publish_without_clobber(source: str, target: Path) -> None:
    """Move ``source`` onto ``target``; refuse when ``target`` exists.

    ``os.replace`` overwrites unconditionally, so an unrelated creator that
    wins the race between the existence check and the rename would be
    silently destroyed.  A hard link is atomic and fails with ``EEXIST``
    instead; where the filesystem cannot link (Windows, some network and
    FAT mounts) the payload is copied into an exclusively created file,
    which is atomic in exactly the same way.
    """
    try:
        os.link(source, target)
    except FileExistsError as exc:
        raise AntennaExportError(
            f"{target} already exists; choose another name or confirm"
            " replacement"
        ) from exc
    except OSError:
        _copy_without_clobber(Path(source), target)
    # The published name is the target; the temporary name is only ours.
    Path(source).unlink(missing_ok=True)


def _copy_without_clobber(source: Path, target: Path) -> None:
    """Copy ``source`` to ``target`` only if ``target`` does not exist."""
    try:
        # "x" is O_CREAT|O_EXCL: creation and the existence check are one
        # atomic operation, so a concurrent creator always wins the race
        # intact instead of being overwritten.
        with source.open("rb") as reader, target.open("xb") as writer:
            try:
                shutil.copyfileobj(reader, writer)
                writer.flush()
                os.fsync(writer.fileno())
            except BaseException:
                writer.close()
                target.unlink(missing_ok=True)
                raise
    except FileExistsError as exc:
        raise AntennaExportError(
            f"{target} already exists; choose another name or confirm"
            " replacement"
        ) from exc


def write_s1p(
    snapshot: MeasurementSnapshot,
    path: Path | str,
    *,
    minimum: MinimumPoint | None = None,
    allow_partial: bool = False,
    overwrite: bool = False,
) -> Path:
    """Write the measured S11 as an honest ``.s1p`` file.

    The data is fully serialized and validated before any file is touched.
    With ``overwrite=False`` an existing file is never replaced - including
    one created concurrently just before publication - because the last step
    is an atomic no-clobber link rather than ``os.replace``.  ``overwrite``
    authorises replacing *this* target only, and only because the caller
    confirmed that exact file.
    """
    target = normalize_s1p_path(path)
    completion = snapshot.provenance.completion
    if completion is not CompletionState.COMPLETE and not allow_partial:
        raise AntennaExportError(
            "the measurement is not complete"
            f" ({completion.value}); a partial sweep is not saved as a"
            " finished result"
        )
    if not snapshot.channel("S11"):
        raise AntennaExportError("the measurement has no S11 samples")
    if target.exists() and not overwrite:
        raise AntennaExportError(
            f"{target} already exists; choose another name or confirm"
            " replacement"
        )
    touchstone = measurement_touchstone(snapshot, minimum)
    try:
        payload = touchstone.saves(1)
    except TouchstoneExportError as exc:
        # ``write_s1p`` promises ``AntennaExportError``, and its GUI caller
        # catches only that and ``OSError``; letting an export refusal
        # escape would show it as an unhandled exception instead of an
        # "Unable to export" message.  Nothing has been written yet.
        raise AntennaExportError(str(exc)) from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=f".{target.stem}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(payload)
        if overwrite:
            os.replace(handle.name, target)
        else:
            _publish_without_clobber(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target


def export_summary(
    snapshot: MeasurementSnapshot | None, minimum: MinimumPoint | None
) -> str:
    """One-line description of what an export would contain."""
    if snapshot is None:
        return "No measurement to export"
    completion = snapshot.provenance.completion
    state = (
        "complete"
        if completion is CompletionState.COMPLETE
        else (f"incomplete ({completion.value})")
    )
    channel = snapshot.channel("S11") or ()
    parts = [f"S11, {len(channel)} samples", state]
    if snapshot.simulated:
        parts.append("simulated")
    parts.append("corrected" if snapshot.correction.applied else "uncorrected")
    if minimum is not None:
        parts.append(f"min VSWR {minimum.vswr:.3g} @ {minimum.freq} Hz")
    return "; ".join(parts)
