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
"""Default sweep configuration for a Touchstone replay source.

A replay source can only answer frequencies inside its own span, and the
grid is produced by :class:`~NanoVNASaver.Settings.Sweep.Sweep`, whose
segment boundaries are rounded to whole hertz.  This module derives a
*legal* default from the grid contract rather than from a guessed stop:

* every requested frequency lies inside the file span,
* the grid keeps one distinct sample per point, so a positive-span file is
  never collapsed into repeated samples of a single frequency, and
* the candidate stop is computed in closed form for linear grids and
  validated against the real grid for logarithmic ones.

The requested point count and grid mode are always preserved.  The segment
count is reduced when the requested one cannot fit, and reported; a span
that cannot hold the points even in one segment raises
:class:`ReplayRangeError` with the settings that would fit.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..Hardware.Capability import SweepCapability
from .Sweep import Properties, Sweep

#: Smallest step the sweep model can express: frequencies are whole hertz.
MIN_STEP_HZ = 1


class ReplayRangeError(Exception):
    """The file span cannot support the requested sweep configuration."""


@dataclass(frozen=True)
class ReplayRangePlan:
    """A sweep configuration that is proven to fit a replay file's span."""

    start: int
    stop: int
    points: int
    segments: int
    logarithmic: bool
    requested_segments: int
    cw: bool = False

    def grid(self) -> list[int]:
        """The frequencies this configuration actually requests."""
        if self.cw:
            return [self.start] * (self.points * self.segments)
        return list(
            Sweep(
                self.start,
                self.stop,
                self.points,
                self.segments,
                Properties(logarithmic=self.logarithmic),
            ).get_frequencies()
        )

    @property
    def reduced(self) -> bool:
        """True when the default had to use fewer segments than requested."""
        return self.segments != self.requested_segments

    def describe(self) -> str:
        """User-facing note about a changed default (empty when unchanged)."""
        if self.cw:
            return (
                f"the file holds one frequency ({self.start} Hz), so the"
                " default sweep is a start == stop sweep of that frequency"
            )
        if self.reduced:
            return (
                f"the file span fits {self.segments} segment(s) at"
                f" {self.points} points, not the requested"
                f" {self.requested_segments}; the default sweep uses"
                f" {self.segments}"
            )
        return ""


def replay_capability(file_start: int, file_stop: int) -> SweepCapability:
    """The capability of a source whose samples span the file exactly.

    Only the span is declared: a file carries no Nyquist grid and no point
    pick list, so nothing else is invented.  Planners use this instead of
    re-deriving what a file can serve.
    """
    return SweepCapability(
        source_min_frequency=file_start,
        source_max_frequency=file_stop,
        cw=file_start == file_stop,
    )


def _fits(plan: ReplayRangePlan, file_start: int, file_stop: int) -> bool:
    """True when the real grid stays in span and repeats nothing per segment.

    The check *is* the shared capability validation, so the default range
    here and every other planner agree on which grids a file can serve:
    adjacent segments may share their boundary frequency (that is how
    ``Sweep`` rounds), but no segment may collapse to repeated samples of
    one frequency, and no segment may ask outside the file.
    """
    validation = replay_capability(file_start, file_stop).validate_grid(
        plan.start,
        plan.stop,
        plan.points,
        plan.segments,
        logarithmic=plan.logarithmic,
    )
    return validation.ok


def _fit(
    file_start: int,
    file_stop: int,
    points: int,
    segments: int,
    logarithmic: bool,
) -> ReplayRangePlan | None:
    """The plan for these points/segments, or ``None`` when it cannot fit."""
    if logarithmic:
        # Logarithmic segment boundaries are exponential fractions of the
        # span, so the last frequency is the file stop itself; only the
        # rounded interior boundaries need validation.
        stop = file_stop
    else:
        # A linear grid has one integer step: the last frequency is
        # start + (points * segments - 1) * step.  Deriving the stop from
        # the largest legal step removes the rounding overshoot instead of
        # shrinking the stop until it happens to fit.
        units = points * segments - 1
        step = (file_stop - file_start) // units
        if step < MIN_STEP_HZ:
            return None
        stop = file_start + step * units

    plan = ReplayRangePlan(
        start=file_start,
        stop=stop,
        points=points,
        segments=segments,
        logarithmic=logarithmic,
        requested_segments=segments,
    )
    if not _fits(plan, file_start, file_stop):
        return None
    return plan


def _segment_bound(
    file_start: int, file_stop: int, points: int, logarithmic: bool
) -> int:
    """Cheapest upper bound on segments this span can hold (may be 0).

    Only a bound: every candidate is verified against the real grid, so a
    wrong bound can cost time but never produce an illegal plan.
    """
    span = file_stop - file_start
    if logarithmic:
        return span // max(points - 1, 1) + 1
    return (span + 1) // points


def plan_replay_range(
    file_start: int,
    file_stop: int,
    *,
    points: int,
    segments: int,
    logarithmic: bool = False,
) -> ReplayRangePlan:
    """Return a legal replay sweep for the file span.

    Keeps ``points`` and the grid mode, keeps ``segments`` when it fits, and
    otherwise reduces the segment count (reported via
    ``ReplayRangePlan.reduced``/``describe``).  Raises
    :class:`ReplayRangeError` when even one segment cannot place ``points``
    distinct frequencies inside the file.
    """
    if points < 1 or segments < 1:
        raise ReplayRangeError(
            "A replay sweep needs at least one point and one segment"
            f" (got {points} points and {segments} segments)."
        )
    if file_start < 1 or file_stop < file_start:
        raise ReplayRangeError(
            f"The replay range {file_start} Hz to {file_stop} Hz is not a"
            " valid frequency span."
        )
    if file_stop == file_start:
        # The file itself holds a single frequency: a start == stop sweep
        # is the honest grid, and no positive span is collapsed.
        return ReplayRangePlan(
            start=file_start,
            stop=file_stop,
            points=points,
            segments=segments,
            logarithmic=logarithmic,
            requested_segments=segments,
            cw=True,
        )
    if points * segments < 2:
        raise ReplayRangeError(
            f"The replay file spans {file_start} Hz to {file_stop} Hz, but a"
            " single point in a single segment requests one frequency."
            " Choose more sweep points or more segments."
        )

    bound = min(
        segments, _segment_bound(file_start, file_stop, points, logarithmic)
    )
    for candidate in range(bound, 0, -1):
        plan = _fit(file_start, file_stop, points, candidate, logarithmic)
        if plan is not None:
            if candidate == segments:
                return plan
            return ReplayRangePlan(
                start=plan.start,
                stop=plan.stop,
                points=plan.points,
                segments=plan.segments,
                logarithmic=plan.logarithmic,
                requested_segments=segments,
            )

    span = file_stop - file_start
    mode = "logarithmic" if logarithmic else "linear"
    raise ReplayRangeError(
        f"The replay file spans only {span} Hz ({file_start} Hz to"
        f" {file_stop} Hz) and cannot hold {points} distinct frequencies in"
        f" {mode} mode. Frequencies are whole hertz, so {points} points need"
        f" at least {points - 1} Hz of span; choose a file with a wider span"
        " or fewer sweep points."
    )
