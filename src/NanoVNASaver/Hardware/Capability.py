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
"""Common usable frequency span and sweep preflight (NEXT-01).

One immutable description of what a sweep source can honestly measure,
kept separate from *how* a request is made:

* **hardware limits** - what the device declares (``sweep_max_freq_hz``,
  ``valid_datapoints``, ``sweep_points_min``/``sweep_points_max``),
* **source limits** - the span of a loaded source such as a replayed
  Touchstone file, which is never extrapolated,
* **the effective span** - the intersection of the two, where ``None``
  means "unknown" and is never turned into an invented ceiling.

The model is pure logic: no Qt, no device access, no I/O.  Devices expose
it through ``capability()``; planning code consumes it instead of
re-deriving bounds.

Validation has three levels, because they answer different questions:

* :meth:`SweepCapability.preflight` decides whether a requested *span* is
  usable.  It never raises, because a rejection is a normal outcome the UI
  must display, and its verdict for a span always agrees with the device's
  own preflight (``VNA.preflight``/``ReplayScenario.preflight``), which
  stays authoritative for acquisition.
* :meth:`SweepCapability.request_problem` decides whether the point,
  segment and grid-mode combination is one the device offers.  The
  declared ``valid_datapoints`` are a pick list, not the only legal
  counts (the device settings clamp to ``sweep_points_min``/``_max``), so
  only the declared minimum and maximum are enforced here.
* :meth:`SweepCapability.validate_grid` decides whether the *grid the
  device really requests* is usable.  ``Sweep`` rounds one integer step for
  the whole sweep and lays segments end to end, so a span that looks fine
  can still sample past the stop, past the device maximum, or collapse a
  segment onto a single repeated frequency.  The grid is computed by
  ``Sweep.get_index_range``/``get_frequencies``, never re-derived, so
  planners and acquisitions cannot disagree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from itertools import pairwise

from ..Formatting import format_frequency
from ..Settings.Sweep import Properties, Sweep
from .ScanCapability import (
    BINARY_OPT_IN,
    ScanCapability,
    scan_capability_for,
)

#: Lowest frequency the sweep model itself can express (whole hertz, > 0).
DEFAULT_MIN_FREQUENCY = 1

#: Default ceiling on segments, matching the preset/antenna planners.
DEFAULT_MAX_SEGMENTS = 1024

#: Fewest samples that can describe a span rather than a single frequency.
MIN_GRID_SAMPLES = 2

# -- problem codes -----------------------------------------------------
#: The requested bounds are not whole numbers of hertz.
INVALID_BOUNDS = "invalid-bounds"
#: ``stop`` is below ``start``.
INVERTED_SPAN = "inverted-span"
#: The request leaves the loaded source's own span.
OUTSIDE_SOURCE = "outside-source"
#: The request starts below the lowest supported frequency.
BELOW_MINIMUM = "start-below-minimum"
#: The request stops above the highest supported frequency.
ABOVE_MAXIMUM = "stop-above-maximum"
#: The point count is below the declared minimum.
TOO_FEW_POINTS = "too-few-points"
#: The point count exceeds the declared maximum.
TOO_MANY_POINTS = "too-many-points"
#: The segment count is outside the supported range.
INVALID_SEGMENTS = "invalid-segments"
#: A logarithmic sweep was requested from a source that cannot do one.
LOGARITHMIC_UNSUPPORTED = "logarithmic-unsupported"
#: ``Sweep`` itself rejected the settings.
INVALID_SETTINGS = "invalid-sweep-settings"
#: The grid holds fewer than two samples.
GRID_TOO_SMALL = "grid-too-small"
#: The grid does not begin at the requested start.
GRID_START_MISMATCH = "grid-start-mismatch"
#: The grid reaches past the requested stop.
GRID_BEYOND_STOP = "grid-beyond-stop"
#: The grid starts below the lowest supported frequency.
GRID_BELOW_MINIMUM = "grid-below-minimum"
#: The grid reaches past the highest supported frequency.
GRID_ABOVE_MAXIMUM = "grid-above-maximum"
#: The grid is not non-decreasing.
GRID_NOT_ORDERED = "grid-not-ordered"
#: A segment repeats one frequency instead of measuring a span.
GRID_COLLAPSED = "grid-collapsed"

#: Every problem code this module can report.
PROBLEM_CODES = frozenset(
    {
        INVALID_BOUNDS,
        INVERTED_SPAN,
        OUTSIDE_SOURCE,
        BELOW_MINIMUM,
        ABOVE_MAXIMUM,
        TOO_FEW_POINTS,
        TOO_MANY_POINTS,
        INVALID_SEGMENTS,
        LOGARITHMIC_UNSUPPORTED,
        INVALID_SETTINGS,
        GRID_TOO_SMALL,
        GRID_START_MISMATCH,
        GRID_BEYOND_STOP,
        GRID_BELOW_MINIMUM,
        GRID_ABOVE_MAXIMUM,
        GRID_NOT_ORDERED,
        GRID_COLLAPSED,
    }
)


class SweepCapabilityError(ValueError):
    """A requested sweep cannot be measured; the message is for the user."""


def _coerce_int(value, what: str) -> int:
    """``value`` as a whole number, or ``ValueError`` naming the field."""
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"{what} must be a whole number, not {value!r}")
    return int(number)


def _points_text(counts: tuple[int, ...]) -> str:
    return ", ".join(str(count) for count in counts)


def _span_text(start: int, stop: int | None) -> str:
    if stop is None:
        return f"{format_frequency(start)} - unknown upper limit"
    return f"{format_frequency(start)} - {format_frequency(stop)}"


@dataclass(frozen=True)
class Span:
    """One frequency range; ``stop is None`` means "upper limit unknown"."""

    start: int
    stop: int | None = None

    def contains(self, start: int, stop: int) -> bool:
        """True when ``start..stop`` lies inside this span."""
        if stop < start or start < self.start:
            return False
        return self.stop is None or stop <= self.stop

    def describe(self) -> str:
        return _span_text(self.start, self.stop)


@dataclass(frozen=True)
class SourceSpan:
    """The span of a loaded source of samples (for example a replay file).

    ``label`` and ``name`` are used verbatim in rejection messages
    ("outside the replay range 1000000 Hz to 4000000 Hz of recorded.s1p"),
    so a message names the thing the user has to change.
    """

    start: int
    stop: int
    name: str = ""
    label: str = "the source range"

    @property
    def cw(self) -> bool:
        """True when the source holds a single frequency."""
        return self.start == self.stop

    def contains(self, start: int, stop: int) -> bool:
        """True when ``start..stop`` lies inside this source's own span."""
        if stop < start:
            return False
        return self.start <= start and stop <= self.stop

    def describe(self) -> str:
        text = f"{self.label} {self.start} Hz to {self.stop} Hz"
        if self.name:
            text += f" of {self.name}"
        return text


@dataclass(frozen=True)
class SweepProblem:
    """A non-raising, user-facing rejection of a sweep request."""

    code: str
    message: str
    start: int | None = None
    stop: int | None = None
    points: int | None = None
    segments: int | None = None
    logarithmic: bool | None = None

    def __str__(self) -> str:
        return self.message

    def describe(self) -> str:
        return self.message


@dataclass(frozen=True)
class GridValidation:
    """The grid a request really produces, plus why it is unusable (if so)."""

    frequencies: tuple[int, ...] = ()
    per_segment: tuple[tuple[int, ...], ...] = ()
    problem: SweepProblem | None = None

    @property
    def ok(self) -> bool:
        return self.problem is None

    def raise_for_problem(self) -> tuple[int, ...]:
        """The grid, or :class:`SweepCapabilityError` with the reason."""
        if self.problem is not None:
            raise SweepCapabilityError(self.problem.message)
        return self.frequencies


@dataclass(frozen=True)
class SweepCapability:
    """What a sweep source can honestly measure.

    ``max_frequency``/``source_max_frequency`` are ``None`` when the limit
    is unknown: an unknown ceiling is never replaced by a guess, it simply
    stops being enforced.  ``valid_points`` empty means "no declared pick
    list", not "no counts": the point range is ``min_points`` to
    ``max_points``.
    """

    min_frequency: int = DEFAULT_MIN_FREQUENCY
    max_frequency: int | None = None
    source_min_frequency: int | None = None
    source_max_frequency: int | None = None
    valid_points: tuple[int, ...] = ()
    default_points: int = 101
    min_points: int = 1
    max_points: int | None = None
    min_segments: int = 1
    max_segments: int = DEFAULT_MAX_SEGMENTS
    cw: bool = False
    logarithmic: bool = True
    supported_channels: tuple[str, ...] = ()
    reference_impedance: float | None = None
    source: str = ""
    #: How this source is read (PERF-02); see ``Hardware/ScanCapability.py``.
    scan: ScanCapability = field(
        default_factory=lambda: ScanCapability(driver="")
    )

    def __post_init__(self) -> None:
        if self.min_frequency < DEFAULT_MIN_FREQUENCY:
            object.__setattr__(self, "min_frequency", DEFAULT_MIN_FREQUENCY)
        if self.min_frequency < 1:
            object.__setattr__(self, "min_frequency", DEFAULT_MIN_FREQUENCY)
        if self.min_points < 1:
            object.__setattr__(self, "min_points", 1)
        if self.min_segments < 1:
            object.__setattr__(self, "min_segments", 1)
        if self.max_segments < self.min_segments:
            object.__setattr__(self, "max_segments", self.min_segments)
        if self.source_min_frequency is not None:
            if self.source_min_frequency < 1:
                object.__setattr__(self, "source_min_frequency", 1)
            if (
                self.source_max_frequency is not None
                and self.source_max_frequency < self.source_min_frequency
            ):
                object.__setattr__(
                    self,
                    "source_max_frequency",
                    self.source_min_frequency,
                )

    # -- spans ---------------------------------------------------------
    @property
    def hardware_span(self) -> Span:
        return Span(self.min_frequency, self.max_frequency)

    @property
    def source_span(self) -> Span | None:
        if self.source_min_frequency is None:
            return None
        return Span(self.source_min_frequency, self.source_max_frequency)

    @property
    def effective_span(self) -> Span:
        """The span both the hardware and the loaded source can serve."""
        lower = self.min_frequency
        if self.source_min_frequency is not None:
            lower = max(lower, self.source_min_frequency)
        uppers = [
            limit
            for limit in (self.max_frequency, self.source_max_frequency)
            if limit is not None
        ]
        return Span(lower, min(uppers) if uppers else None)

    @property
    def is_cw(self) -> bool:
        """True when only a single frequency can be measured."""
        return self.cw

    @property
    def has_source_limits(self) -> bool:
        return self.source_min_frequency is not None

    def supports_channel(self, channel: str) -> bool:
        """True unless the source declares fewer channels than ``channel``."""
        if not self.supported_channels:
            return True
        return channel in self.supported_channels

    def supports_points(self, points: int) -> bool:
        """True when ``points`` is one of the device's declared pick list."""
        if not self.valid_points:
            return True
        return points in self.valid_points

    def points_in_range(self, points: int) -> bool:
        """True when ``points`` is inside the declared minimum/maximum."""
        if points < self.min_points:
            return False
        if self.max_points is None:
            return True
        return points <= self.max_points

    def with_source(self, span: SourceSpan) -> SweepCapability:
        """This capability restricted to a loaded source's own span."""
        return replace(
            self,
            source_min_frequency=span.start,
            source_max_frequency=span.stop,
            cw=span.cw,
            source=span.describe(),
        )

    # -- construction --------------------------------------------------
    @classmethod
    def from_vna(
        cls,
        vna,
        *,
        source: SourceSpan | None = None,
    ) -> SweepCapability:
        """Read the limits a device object itself declares.

        Only declared values are used: ``sweep_max_freq_hz`` of ``0.0`` (or
        a missing member) means the ceiling is unknown, never "no
        frequencies allowed" and never an invented maximum.
        """
        try:
            raw_max = float(getattr(vna, "sweep_max_freq_hz", 0.0) or 0.0)
        except (TypeError, ValueError):
            raw_max = 0.0
        max_frequency = (
            int(raw_max)
            if math.isfinite(raw_max) and raw_max >= DEFAULT_MIN_FREQUENCY
            else None
        )

        try:
            declared = tuple(
                sorted(
                    {
                        int(count)
                        for count in (
                            getattr(vna, "valid_datapoints", ()) or ()
                        )
                        if int(count) >= 1
                    }
                )
            )
        except (TypeError, ValueError):
            declared = ()

        try:
            default_points = int(getattr(vna, "datapoints", 101))
        except (TypeError, ValueError):
            default_points = 101
        if default_points < 1:
            default_points = declared[0] if declared else 101

        try:
            min_points = max(int(getattr(vna, "sweep_points_min", 1)), 1)
        except (TypeError, ValueError):
            min_points = 1
        try:
            raw_max_points = int(getattr(vna, "sweep_points_max", 0) or 0)
        except (TypeError, ValueError):
            raw_max_points = 0
        max_points = raw_max_points if raw_max_points >= 1 else None

        try:
            supported_channels = tuple(
                str(name)
                for name in (getattr(vna, "supported_channels", ()) or ())
            )
        except TypeError:
            supported_channels = ()

        try:
            impedance = float(getattr(vna, "reference_impedance", None))
        except (TypeError, ValueError):
            impedance = None
        if impedance is not None and not (
            math.isfinite(impedance) and impedance > 0
        ):
            impedance = None

        capability = cls(
            min_frequency=DEFAULT_MIN_FREQUENCY,
            max_frequency=max_frequency,
            valid_points=declared,
            default_points=default_points,
            min_points=min_points,
            max_points=max_points,
            supported_channels=supported_channels,
            reference_impedance=impedance,
            scan=scan_capability_for(vna),
        )
        if source is None:
            return capability
        return capability.with_source(source)

    # -- validation ----------------------------------------------------
    def preflight(self, start, stop) -> SweepProblem | None:
        """Why ``start`` to ``stop`` cannot be swept, or ``None``.

        Never raises: a rejection is a normal, displayable outcome.  The
        verdict always agrees with the device's own preflight, which stays
        authoritative for acquisition: identical bounds are accepted when
        the source holds that frequency, an inverted span is rejected, and
        the source span is inclusive at both ends.
        """
        try:
            start = _coerce_int(start, "start frequency")
            stop = _coerce_int(stop, "stop frequency")
        except (TypeError, ValueError) as exc:
            return SweepProblem(
                INVALID_BOUNDS,
                f"The sweep bounds must be whole numbers of hertz ({exc}).",
            )
        if stop < start:
            return SweepProblem(
                INVERTED_SPAN,
                f"The sweep span is invalid: {start} Hz to {stop} Hz."
                " The stop frequency must not be below the start frequency.",
                start,
                stop,
            )

        source = self.source_span
        if source is not None and not source.contains(start, stop):
            where = self.source or source.describe()
            return SweepProblem(
                OUTSIDE_SOURCE,
                f"The sweep {start} Hz to {stop} Hz is outside {where}."
                " A loaded source is never extrapolated; keep the sweep"
                " inside its own span.",
                start,
                stop,
            )

        effective = self.effective_span
        if start < effective.start:
            return SweepProblem(
                BELOW_MINIMUM,
                f"The sweep starts at {format_frequency(start)}, below the"
                f" minimum {format_frequency(effective.start)}.",
                start,
                stop,
            )
        if effective.stop is not None and stop > effective.stop:
            return SweepProblem(
                ABOVE_MAXIMUM,
                f"The sweep stops at {format_frequency(stop)}, above the"
                f" maximum {format_frequency(effective.stop)}.",
                start,
                stop,
            )
        return None

    def request_problem(
        self,
        points,
        segments: int = 1,
        *,
        logarithmic: bool = False,
    ) -> SweepProblem | None:
        """Why this point/segment/mode combination is unusable, or ``None``."""
        try:
            points = _coerce_int(points, "point count")
            segments = _coerce_int(segments, "segment count")
        except (TypeError, ValueError) as exc:
            return SweepProblem(
                INVALID_BOUNDS,
                f"The sweep settings must be whole numbers ({exc}).",
                points=points,
                segments=segments,
            )
        if points < self.min_points:
            return SweepProblem(
                TOO_FEW_POINTS,
                f"{points} sweep points are below the declared minimum of"
                f" {self.min_points}.",
                points=points,
                segments=segments,
                logarithmic=logarithmic,
            )
        if self.max_points is not None and points > self.max_points:
            return SweepProblem(
                TOO_MANY_POINTS,
                f"{points} sweep points exceed the declared maximum of"
                f" {self.max_points}.",
                points=points,
                segments=segments,
                logarithmic=logarithmic,
            )
        if segments < self.min_segments or segments > self.max_segments:
            return SweepProblem(
                INVALID_SEGMENTS,
                f"{segments} segments is outside the supported range"
                f" {self.min_segments} to {self.max_segments}.",
                points=points,
                segments=segments,
                logarithmic=logarithmic,
            )
        if logarithmic and not self.logarithmic:
            return SweepProblem(
                LOGARITHMIC_UNSUPPORTED,
                "This source cannot sweep logarithmically.",
                points=points,
                segments=segments,
                logarithmic=logarithmic,
            )
        return None

    def validate_grid(
        self,
        start,
        stop,
        points,
        segments: int = 1,
        *,
        logarithmic: bool = False,
    ) -> GridValidation:
        """The real grid of a request, plus why it is unusable (if so).

        The span is checked first (:meth:`preflight`), then the point,
        segment and mode combination, then the grid exactly as the device
        will request it: it must start at ``start``, stay inside the span,
        stay ordered, and never repeat a frequency inside one segment
        (which a rounded step does when the span is narrower than the point
        count).
        """
        span_problem = self.preflight(start, stop)
        if span_problem is not None:
            return GridValidation(problem=span_problem)
        request_problem = self.request_problem(
            points, segments, logarithmic=logarithmic
        )
        if request_problem is not None:
            return GridValidation(problem=request_problem)

        start = int(start)
        stop = int(stop)
        points = int(points)
        segments = int(segments)

        def problem(code: str, message: str, **grid) -> GridValidation:
            return GridValidation(
                **grid,
                problem=SweepProblem(
                    code,
                    message,
                    start,
                    stop,
                    points,
                    segments,
                    logarithmic,
                ),
            )

        try:
            frequencies, per_segment = self._grid(
                start, stop, points, segments, logarithmic
            )
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            return problem(INVALID_SETTINGS, f"illegal sweep settings: {exc}")

        if len(frequencies) < MIN_GRID_SAMPLES:
            return problem(
                GRID_TOO_SMALL,
                f"{points} points in {segments} segment(s) give fewer than"
                f" {MIN_GRID_SAMPLES} samples.",
                frequencies=frequencies,
                per_segment=per_segment,
            )
        effective = self.effective_span
        if frequencies[0] < effective.start:
            return problem(
                GRID_BELOW_MINIMUM,
                f"The measured grid starts at {frequencies[0]} Hz, below the"
                f" minimum {format_frequency(effective.start)}.",
                frequencies=frequencies,
                per_segment=per_segment,
            )
        if effective.stop is not None and frequencies[-1] > effective.stop:
            return problem(
                GRID_ABOVE_MAXIMUM,
                f"The measured grid reaches {frequencies[-1]} Hz, above the"
                f" maximum {format_frequency(effective.stop)}.",
                frequencies=frequencies,
                per_segment=per_segment,
            )
        if frequencies[0] != start:
            return problem(
                GRID_START_MISMATCH,
                f"The measured grid starts at {frequencies[0]} Hz instead of"
                f" {start} Hz.",
                frequencies=frequencies,
                per_segment=per_segment,
            )
        if frequencies[-1] > stop:
            return problem(
                GRID_BEYOND_STOP,
                f"The measured grid reaches {frequencies[-1]} Hz, beyond the"
                f" requested stop {stop} Hz.",
                frequencies=frequencies,
                per_segment=per_segment,
            )
        if any(later < earlier for earlier, later in pairwise(frequencies)):
            return problem(
                GRID_NOT_ORDERED,
                f"{points} points in {segments} segment(s) do not give an"
                " ordered grid; widen the span or lower the segment count.",
                frequencies=frequencies,
                per_segment=per_segment,
            )
        if stop > start:
            collapsed = next(
                (
                    len(set(segment))
                    for segment in per_segment
                    if len(set(segment)) != len(segment)
                ),
                None,
            )
            if collapsed is not None:
                return problem(
                    GRID_COLLAPSED,
                    f"{points} points in {segments} segment(s) collapse onto"
                    f" {collapsed} distinct frequency(ies) per segment; the"
                    " span is too narrow for that configuration.",
                    frequencies=frequencies,
                    per_segment=per_segment,
                )
        return GridValidation(frequencies, per_segment)

    def _grid(
        self,
        start: int,
        stop: int,
        points: int,
        segments: int,
        logarithmic: bool,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        """The frequencies requested, exactly as ``Sweep`` produces them."""
        if points * segments == 1:
            # ``Sweep`` divides by ``points * segments - 1``, so a single
            # sample cannot be described by a sweep object at all: it is
            # exactly the one frequency requested.
            return (start,), ((start,),)
        sweep = Sweep(
            start,
            stop,
            points,
            segments,
            Properties(logarithmic=logarithmic),
        )
        if points <= 1:
            # ``Sweep.get_frequencies`` divides by ``points - 1``; a
            # one-point segment samples exactly its own segment start.
            frequencies = tuple(
                sweep.get_index_range(index)[0] for index in range(segments)
            )
        else:
            frequencies = tuple(sweep.get_frequencies())
        per_segment = tuple(
            frequencies[index : index + points]
            for index in range(0, len(frequencies), points)
        )
        return frequencies, per_segment

    # -- description ---------------------------------------------------
    def describe(self) -> str:
        """One-line, human-readable description of the usable span."""
        span = _span_text(self.min_frequency, self.max_frequency)
        text = f"Device range: {span}"
        if self.source_min_frequency is not None:
            source = self.source or self.source_span.describe()
            text += f"; usable span: {source}"
            if self.effective_span.stop is not None:
                text += f" ({self.effective_span.describe()})"
        if self.valid_points:
            counts = _points_text(self.valid_points)
        elif self.max_points is None:
            counts = "any count"
        else:
            counts = f"{self.min_points} to {self.max_points}"
        text += f"; supported points: {counts}"
        if self.cw:
            text += "; single-frequency source"
        if self.supported_channels:
            text += f"; channels: {', '.join(self.supported_channels)}"
        if self.reference_impedance is not None:
            text += f"; Z0 {self.reference_impedance:g} ohm"
        text += f"; reads: {self.scan.path}"
        if self.scan.binary_scan == BINARY_OPT_IN:
            text += " (scan_bin opt-in, unverified)"
        elif self.scan.combined_text_opt_in:
            text += " (combined text opt-in, unverified)"
        return text
