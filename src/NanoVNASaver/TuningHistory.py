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
"""Antenna tuning history and comparison (TUNE-01).

An append-only record of *completed* tuning steps, so an operator can see
whether an adjustment helped and can return to a previous span later.

Design rules
------------
* **One immutable step per completed run.**  A step is frozen at the end of
  the run that measured it: band, span, measured minimum (with an explicit
  measured/estimated flag), timestamp, trace identity and the operator's
  notes/settings.  Nothing is recomputed later from live UI state.
* **A step is tied to the run and the measurement that produced it.**  A
  cancelled or failed run has no step; a minimum that does not belong to the
  snapshot being recorded is refused with a stated reason, so a stale result
  can never be attributed to a newer step.  A duplicated completion is
  recognised by its run identity and appended once.
* **Measured is not estimated.**  A recorded minimum is either an existing
  measured sample or an explicitly supplied bracketed estimate - never an
  interpolated or extrapolated frequency that was never measured.
* **Recall touches nothing but a form.**  :meth:`TuningHistory.recall`
  returns the recorded span; it cannot start a sweep or move hardware.
* **Retention is bounded and stated.**  At most :data:`MAX_STEPS` steps are
  kept; appending one more drops the oldest and reports exactly which steps
  were dropped.  Clearing is explicit and exporting is explicit.
* **Comparison has a direction per metric.**  VSWR improves downwards and
  return loss improves upwards; a trend says which, it never guesses.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from typing import Any, Iterable, Mapping, Sequence

from .AntennaTuning import MinimumPoint, SweepPlan, find_vswr_minimum
from .Formatting import format_frequency
from .MarkerSearch import LocationKind
from .Measurement import (
    CompletionState,
    MeasurementSnapshot,
    _decode_float,
    _decode_time,
    _encode_float,
    _encode_time,
)

__all__ = [
    "CSV_FIELDS",
    "MAX_STEPS",
    "TUNING_HISTORY_FORMAT",
    "TUNING_HISTORY_VERSION",
    "AppendResult",
    "MetricDelta",
    "SpanRecall",
    "TrendComparison",
    "TuningAppendError",
    "TuningFormatError",
    "TuningHistory",
    "TuningHistoryError",
    "TuningStep",
    "band_for",
]

TUNING_HISTORY_FORMAT = "nanovna-saver-tuning-history"
#: Version 1 is the first published shape.  It is stored inside the session
#: manifest, so an older session (no ``tuning_history`` key at all) still
#: loads with an empty history and loses nothing else.
TUNING_HISTORY_VERSION = 1

#: Bounded retention: the oldest steps are dropped first.  A tuning session
#: legitimately accumulates hundreds of repeats; a thousand is an archive,
#: not a history, and would bloat every session save.
MAX_STEPS = 200

#: Sign of an improvement for each metric: a *negative* VSWR change is
#: better, a *positive* return-loss change is better.
VSWR_DIRECTION = -1
RETURN_LOSS_DIRECTION = 1

DOCUMENT_KEYS = ("format", "version", "steps")
STEP_KEYS = (
    "band",
    "start_hz",
    "end_hz",
    "points",
    "segments",
    "logarithmic",
    "frequency_hz",
    "vswr",
    "return_loss",
    "location",
    "at_edge",
    "samples",
    "timestamp",
    "snapshot_id",
    "run_id",
    "trace_id",
    "trace_name",
    "notes",
    "settings",
)

#: Columns of the CSV form, in order.  Frozen so an old file stays readable
#: and an exported history re-imports unchanged.
CSV_FIELDS = (
    "index",
    "timestamp",
    "band",
    "start_hz",
    "end_hz",
    "frequency_hz",
    "vswr",
    "return_loss",
    "location",
    "at_edge",
    "samples",
    "points",
    "segments",
    "logarithmic",
    "snapshot_id",
    "run_id",
    "trace_id",
    "trace_name",
    "notes",
    "settings",
)


class TuningHistoryError(Exception):
    """Base class for tuning history problems the UI can report verbatim."""


class TuningAppendError(TuningHistoryError):
    """A step cannot be recorded; ``reason`` says why, in operator terms."""


class TuningFormatError(TuningHistoryError):
    """A history document is unreadable, structurally wrong or too new."""


# ----------------------------------------------------------------------
# Band naming
# ----------------------------------------------------------------------
def band_for(frequency: int, bands: Iterable[Sequence[Any]] | None) -> str:
    """Name of the first band containing ``frequency``, else ``""``.

    ``bands`` rows are ``(name, start, end)`` with inclusive bounds, exactly
    as the band model stores them; the numbers may be strings.  An unknown
    band is an empty name, never an invented one.
    """
    for row in bands or ():
        try:
            name, start, stop = row[0], row[1], row[2]
            low = int(float(start))
            high = int(float(stop))
        except (IndexError, TypeError, ValueError):
            continue
        if low <= int(frequency) <= high:
            return str(name)
    return ""


# ----------------------------------------------------------------------
# Step record
# ----------------------------------------------------------------------
def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _settings_tuple(
    settings: Mapping[str, Any] | Sequence[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    if settings is None:
        return ()
    items = settings.items() if isinstance(settings, Mapping) else settings
    return tuple(
        sorted((str(key), str(value)) for key, value in items if str(key))
    )


@dataclass(frozen=True)
class TuningStep:
    """One immutable, completed tuning step."""

    band: str
    start: int
    end: int
    points: int
    segments: int
    logarithmic: bool
    frequency: int
    vswr: float
    return_loss: float | None
    location: LocationKind
    at_edge: bool
    timestamp: datetime
    snapshot_id: str
    run_id: str
    trace_id: str = ""
    trace_name: str = ""
    notes: str = ""
    #: Free-text component/position settings, stored as sorted pairs so the
    #: record stays hashable and compares by content.
    settings: tuple[tuple[str, str], ...] = ()
    samples: int = 0
    completion: CompletionState = CompletionState.COMPLETE

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _utc(self.timestamp))
        if isinstance(self.settings, Mapping) or not isinstance(
            self.settings, tuple
        ):
            object.__setattr__(self, "settings", _settings_tuple(self.settings))
        if self.start < 0 or self.end < self.start:
            raise TuningAppendError(
                f"impossible span {self.start} - {self.end} Hz"
            )
        if not self.start <= self.frequency <= self.end:
            raise TuningAppendError(
                f"{self.frequency} Hz is outside the recorded span"
                f" {self.start} - {self.end} Hz"
            )
        if self.points < 1 or self.segments < 1:
            raise TuningAppendError("a step needs at least one measured sample")
        if not math.isfinite(self.vswr) or self.vswr < 1.0:
            raise TuningAppendError(
                f"VSWR {self.vswr!r} is not a possible measurement"
            )
        if self.return_loss is not None and math.isnan(self.return_loss):
            raise TuningAppendError("return loss is not a number")
        if not self.run_id:
            raise TuningAppendError(
                "a tuning step must be tied to the run that measured it"
            )

    # -- presentation --------------------------------------------------
    @property
    def measured(self) -> bool:
        """True only for a minimum that is an existing measured sample."""
        return self.location is LocationKind.MEASURED

    @property
    def span(self) -> int:
        return self.end - self.start

    @property
    def label(self) -> str:
        where = "measured" if self.measured else "estimated"
        return (
            f"{format_frequency(self.frequency)} VSWR {self.vswr:.3f} ({where})"
        )

    @property
    def settings_text(self) -> str:
        return "; ".join(f"{key}={value}" for key, value in self.settings)

    def describe(self) -> str:
        text = (
            f"{self.band or 'no band'}: {self.label} over"
            f" {format_frequency(self.start)} - {format_frequency(self.end)}"
            f" ({self.points} points)"
        )
        if self.at_edge:
            text += " - at the sweep edge, the dip may lie outside"
        if self.notes:
            text += f"; note: {self.notes}"
        if self.settings:
            text += f"; {self.settings_text}"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "band": self.band,
            "start_hz": self.start,
            "end_hz": self.end,
            "points": self.points,
            "segments": self.segments,
            "logarithmic": self.logarithmic,
            "frequency_hz": self.frequency,
            "vswr": _encode_float(self.vswr),
            "return_loss": (
                None
                if self.return_loss is None
                else _encode_float(self.return_loss)
            ),
            "location": self.location.value,
            "at_edge": self.at_edge,
            "samples": self.samples,
            "timestamp": _encode_time(self.timestamp),
            "snapshot_id": self.snapshot_id,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "trace_name": self.trace_name,
            "notes": self.notes,
            "settings": [[key, value] for key, value in self.settings],
            "completion": self.completion.value,
        }

    @classmethod
    def from_dict(cls, data: object, where: str = "step") -> "TuningStep":
        entry = _require_object(data, where)
        try:
            return_loss = entry.get("return_loss")
            return cls(
                band=str(entry.get("band", "")),
                start=int(entry["start_hz"]),
                end=int(entry["end_hz"]),
                points=int(entry["points"]),
                segments=int(entry.get("segments", 1)),
                logarithmic=bool(entry.get("logarithmic", False)),
                frequency=int(entry["frequency_hz"]),
                vswr=_decode_float(entry["vswr"]),
                return_loss=(
                    None if return_loss is None else _decode_float(return_loss)
                ),
                location=LocationKind(str(entry.get("location", "measured"))),
                at_edge=bool(entry.get("at_edge", False)),
                samples=int(entry.get("samples", 0)),
                timestamp=_decode_time(str(entry["timestamp"])),
                snapshot_id=str(entry.get("snapshot_id", "")),
                run_id=str(entry.get("run_id", "")),
                trace_id=str(entry.get("trace_id", "")),
                trace_name=str(entry.get("trace_name", "")),
                notes=str(entry.get("notes", "")),
                settings=_settings_tuple(entry.get("settings")),
                completion=CompletionState.parse(
                    entry.get("completion", CompletionState.COMPLETE.value)
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TuningFormatError(f"{where}: unreadable step ({exc})") from (
                None
            )


def _require_object(data: object, where: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise TuningFormatError(f"{where} must be an object")
    return data


@dataclass(frozen=True)
class SpanRecall:
    """A recorded span, ready to be typed back into the sweep form."""

    start: int
    end: int
    points: int
    segments: int
    logarithmic: bool

    @property
    def centre(self) -> int:
        return (self.start + self.end) // 2

    def describe(self) -> str:
        text = (
            f"{format_frequency(self.start)} - {format_frequency(self.end)},"
            f" {self.points} points, {self.segments} segment(s)"
        )
        return text + (", logarithmic" if self.logarithmic else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_hz": self.start,
            "end_hz": self.end,
            "points": self.points,
            "segments": self.segments,
            "logarithmic": self.logarithmic,
        }


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MetricDelta:
    """One metric compared with one reference, with its own direction."""

    metric: str
    value: float | None
    reference: float | None
    direction: int

    @property
    def delta(self) -> float | None:
        if not _comparable(self.value) or not _comparable(self.reference):
            return None
        return float(self.value) - float(self.reference)

    @property
    def improved(self) -> bool | None:
        delta = self.delta
        if delta is None:
            return None
        return delta * self.direction > 0

    @property
    def verdict(self) -> str:
        improved = self.improved
        if improved is None:
            return "not comparable"
        if math.isclose(self.delta or 0.0, 0.0, abs_tol=1e-12):
            return "unchanged"
        return "improved" if improved else "worse"

    def describe(self, unit: str = "") -> str:
        delta = self.delta
        if delta is None:
            return f"{self.metric}: not comparable"
        sign = "+" if delta >= 0 else "-"
        return f"{self.metric} {sign}{abs(delta):.4f}{unit} {self.verdict}"


def _comparable(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


@dataclass(frozen=True)
class TrendComparison:
    """One step compared with its predecessor and with the best so far."""

    step: TuningStep
    previous: TuningStep | None
    best: TuningStep | None
    vswr: MetricDelta
    return_loss: MetricDelta
    vswr_to_best: MetricDelta
    return_loss_to_best: MetricDelta

    @property
    def is_best(self) -> bool:
        return self.best is not None and self.best.run_id == self.step.run_id

    @property
    def frequency_delta(self) -> int | None:
        if self.previous is None:
            return None
        return self.step.frequency - self.previous.frequency

    @property
    def same_band(self) -> bool | None:
        """Whether the predecessor measured the same band, if named."""
        if self.previous is None:
            return None
        return self.step.band == self.previous.band

    @property
    def band_changed(self) -> bool:
        return self.same_band is False

    def lines(self) -> list[str]:
        text = [self.step.describe()]
        if self.previous is None:
            text.append("First recorded step; nothing to compare it with yet.")
            return text
        text.append(
            "vs previous: "
            + self.vswr.describe()
            + "; "
            + self.return_loss.describe(" dB")
        )
        if self.band_changed:
            text.append(
                f"Band changed from {self.previous.band or 'unnamed'} to"
                f" {self.step.band or 'unnamed'}; compare like with like."
            )
        if self.frequency_delta:
            text.append(
                f"Resonance moved {self.frequency_delta:+d} Hz"
                " (lower is not necessarily better: it is a position)."
            )
        if not self.is_best and self.best is not None:
            text.append(
                "vs best so far: "
                + self.vswr_to_best.describe()
                + "; "
                + self.return_loss_to_best.describe(" dB")
            )
        return text


# ----------------------------------------------------------------------
# Append outcome
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AppendResult:
    """What recording a step did."""

    step: TuningStep
    dropped: tuple[TuningStep, ...] = ()
    duplicate: bool = False


# ----------------------------------------------------------------------
# The history itself
# ----------------------------------------------------------------------
class TuningHistory:
    """Append-only, bounded record of completed tuning steps."""

    def __init__(
        self,
        steps: Iterable[TuningStep] = (),
        *,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.max_steps = max(1, int(max_steps))
        self._steps: list[TuningStep] = list(steps)
        self._by_run: dict[str, TuningStep] = {}
        for step in self._steps:
            self._by_run.setdefault(step.run_id, step)

    # -- reading -------------------------------------------------------
    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self):
        return iter(tuple(self._steps))

    def __getitem__(self, index: int) -> TuningStep:
        return self._steps[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TuningHistory):
            return NotImplemented
        return tuple(self._steps) == tuple(other._steps)

    def __repr__(self) -> str:
        return f"TuningHistory({len(self._steps)} step(s))"

    @property
    def steps(self) -> tuple[TuningStep, ...]:
        return tuple(self._steps)

    @property
    def empty(self) -> bool:
        return not self._steps

    def step_for_run(self, run_id: str) -> TuningStep | None:
        return self._by_run.get(run_id)

    def best(self, index: int | None = None) -> TuningStep | None:
        """Lowest measured VSWR up to ``index`` (earliest wins a tie)."""
        upto = len(self._steps) if index is None else index + 1
        best: TuningStep | None = None
        for step in self._steps[: max(0, upto)]:
            if best is None or step.vswr < best.vswr:
                best = step
        return best

    # -- writing -------------------------------------------------------
    def append(  # noqa: PLR0913
        self,
        *,
        snapshot: MeasurementSnapshot,
        plan: SweepPlan,
        minimum: MinimumPoint | None,
        run_id: str,
        band: str = "",
        trace_id: str = "",
        trace_name: str = "",
        timestamp: datetime | None = None,
        notes: str = "",
        settings: Mapping[str, Any] | None = None,
        location: LocationKind = LocationKind.MEASURED,
        basis: MinimumPoint | None = None,
    ) -> AppendResult:
        """Record one completed step, or refuse it with a reason.

        The step is verified against ``snapshot`` *and* ``plan`` before it is
        recorded; a minimum that is not this measurement's own cannot be
        filed under this run.  A second call with the same ``run_id`` is a
        no-op returning the step that is already stored.
        """
        if not run_id:
            raise TuningAppendError(
                "a tuning step must be tied to the run that measured it"
            )
        existing = self._by_run.get(run_id)
        if existing is not None:
            return AppendResult(step=existing, duplicate=True)
        step = self._build_step(
            snapshot=snapshot,
            plan=plan,
            minimum=minimum,
            run_id=run_id,
            band=band,
            trace_id=trace_id,
            trace_name=trace_name,
            timestamp=timestamp,
            notes=notes,
            settings=settings,
            location=location,
            basis=basis,
        )
        self._steps.append(step)
        self._by_run[run_id] = step
        dropped = self._enforce_retention()
        return AppendResult(step=step, dropped=dropped)

    def _enforce_retention(self) -> tuple[TuningStep, ...]:
        """Drop the oldest steps beyond :attr:`max_steps`, and report them."""
        dropped: list[TuningStep] = []
        while len(self._steps) > self.max_steps:
            oldest = self._steps.pop(0)
            dropped.append(oldest)
            # The step being appended is never the one dropped, so its run
            # identity stays resolvable for a duplicated completion.
            if self._by_run.get(oldest.run_id) is oldest:
                del self._by_run[oldest.run_id]
        return tuple(dropped)

    def _build_step(  # noqa: PLR0913
        self,
        *,
        snapshot: MeasurementSnapshot,
        plan: SweepPlan,
        minimum: MinimumPoint | None,
        run_id: str,
        band: str,
        trace_id: str,
        trace_name: str,
        timestamp: datetime | None,
        notes: str,
        settings: Mapping[str, Any] | None,
        location: LocationKind,
        basis: MinimumPoint | None,
    ) -> TuningStep:
        if snapshot is None:
            raise TuningAppendError("no measurement was supplied")
        completion = snapshot.provenance.completion
        if completion is not CompletionState.COMPLETE:
            raise TuningAppendError(
                f"the sweep did not complete ({completion.value}); an"
                " interrupted measurement is never a tuning step"
            )
        if not snapshot.id:
            raise TuningAppendError(
                "the measurement has no identity, so a later session could"
                " not prove which capture a step describes"
            )
        if minimum is None:
            raise TuningAppendError(
                "no minimum is available, so there is nothing to record"
            )
        if not plan.start < plan.end:
            raise TuningAppendError(
                f"the planned span {plan.start} - {plan.end} Hz is empty"
            )
        if not plan.start <= minimum.freq <= plan.end:
            raise TuningAppendError(
                f"the minimum ({minimum.freq} Hz) is outside the planned span"
                f" {plan.start} - {plan.end} Hz"
            )
        points = snapshot.channel("S11") or ()
        if not points:
            raise TuningAppendError(
                "the measurement has no S11 samples to record"
            )
        measured = find_vswr_minimum(points)
        measured_minimum = measured.minimum
        if measured_minimum is None:
            raise TuningAppendError(
                "the measurement has no finite S11 sample:"
                f" {measured.reason or 'nothing measurable'}"
            )
        grid = tuple(int(point.freq) for point in points)
        if location is LocationKind.MEASURED:
            if minimum.freq not in grid:
                raise TuningAppendError(
                    f"{minimum.freq} Hz is not one of the measured samples;"
                    " a measured minimum must be an existing sample"
                )
            if minimum.freq != measured_minimum.freq or not _same_value(
                minimum.vswr, measured_minimum.vswr
            ):
                raise TuningAppendError(
                    "the supplied minimum"
                    f" ({minimum.freq} Hz VSWR {minimum.vswr:.4f}) does not"
                    " belong to this measurement; it is stale and cannot be"
                    f" recorded as a step of this run (this measurement's own"
                    f" minimum is {measured_minimum.freq} Hz VSWR"
                    f" {measured_minimum.vswr:.4f})"
                )
        else:
            self._verify_estimate(minimum, measured_minimum, grid, basis)
        return TuningStep(
            band=band,
            start=int(plan.start),
            end=int(plan.end),
            points=int(plan.points),
            segments=int(plan.segments),
            logarithmic=bool(plan.logarithmic),
            frequency=int(minimum.freq),
            vswr=float(minimum.vswr),
            return_loss=minimum.return_loss,
            location=location,
            at_edge=bool(minimum.at_edge),
            timestamp=timestamp or datetime.now(timezone.utc),
            snapshot_id=str(snapshot.id),
            run_id=run_id,
            trace_id=trace_id,
            trace_name=trace_name,
            notes=notes,
            settings=_settings_tuple(settings),
            samples=len(grid),
            completion=completion,
        )

    @staticmethod
    def _verify_estimate(
        minimum: MinimumPoint,
        measured: MinimumPoint,
        grid: tuple[int, ...],
        basis: MinimumPoint | None,
    ) -> None:
        """An estimate must be bracketed by measured samples of this run."""
        if basis is None:
            raise TuningAppendError(
                "an estimated minimum must name the measured minimum it"
                " was derived from"
            )
        if basis.freq != measured.freq or not _same_value(
            basis.vswr, measured.vswr
        ):
            raise TuningAppendError(
                "the estimate's basis is not this measurement's minimum;"
                " the estimate cannot be attributed to this run"
            )
        if minimum.freq in grid:
            raise TuningAppendError(
                f"{minimum.freq} Hz is a measured sample; record it as"
                " measured rather than estimated"
            )
        bracketed = any(
            low < minimum.freq < high for low, high in pairwise(grid)
        )
        if not bracketed:
            raise TuningAppendError(
                f"{minimum.freq} Hz is not between two measured samples;"
                " extrapolating outside the measurement is never allowed"
            )

    def clear(self) -> int:
        """Forget every step; returns how many were removed."""
        removed = len(self._steps)
        self._steps.clear()
        self._by_run.clear()
        return removed

    # -- comparison ----------------------------------------------------
    def trend(self, index: int | None = None) -> TrendComparison | None:
        """Compare one step with its predecessor and the best so far."""
        if not self._steps:
            return None
        position = len(self._steps) - 1 if index is None else int(index)
        if position < 0:
            position += len(self._steps)
        if not 0 <= position < len(self._steps):
            raise IndexError(f"no tuning step {index!r}")
        step = self._steps[position]
        previous = self._steps[position - 1] if position else None
        best = self.best(position)
        return TrendComparison(
            step=step,
            previous=previous,
            best=best,
            vswr=MetricDelta(
                "VSWR",
                step.vswr,
                previous.vswr if previous else None,
                VSWR_DIRECTION,
            ),
            return_loss=MetricDelta(
                "return loss",
                step.return_loss,
                previous.return_loss if previous else None,
                RETURN_LOSS_DIRECTION,
            ),
            vswr_to_best=MetricDelta(
                "VSWR",
                step.vswr,
                best.vswr if best else None,
                VSWR_DIRECTION,
            ),
            return_loss_to_best=MetricDelta(
                "return loss",
                step.return_loss,
                best.return_loss if best else None,
                RETURN_LOSS_DIRECTION,
            ),
        )

    # -- recall --------------------------------------------------------
    def recall(self, step_or_index: TuningStep | int) -> SpanRecall:
        """The recorded span of a step, for typing back into the form.

        Purely a data lookup: no sweep is started and no device is touched.
        """
        step = (
            step_or_index
            if isinstance(step_or_index, TuningStep)
            else self._steps[int(step_or_index)]
        )
        return SpanRecall(
            start=step.start,
            end=step.end,
            points=step.points,
            segments=step.segments,
            logarithmic=step.logarithmic,
        )

    # -- persistence ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "format": TUNING_HISTORY_FORMAT,
            "version": TUNING_HISTORY_VERSION,
            "steps": [step.to_dict() for step in self._steps],
        }

    @classmethod
    def from_dict(
        cls, data: object, *, where: str = "tuning history"
    ) -> "TuningHistory":
        """Rebuild a history; an absent/empty document is an empty history."""
        if data is None or data == {}:
            return cls()
        document = _require_object(data, where)
        if document.get("format") != TUNING_HISTORY_FORMAT:
            raise TuningFormatError(
                f"{where}: unexpected format {document.get('format')!r};"
                f" expected {TUNING_HISTORY_FORMAT!r}"
            )
        version = document.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise TuningFormatError(f"{where}: version must be an integer")
        if version > TUNING_HISTORY_VERSION:
            raise TuningFormatError(
                f"{where}: version {version} is newer than supported"
                f" version {TUNING_HISTORY_VERSION}; upgrade NanoVNASaver"
            )
        if version < 1:
            raise TuningFormatError(f"{where}: invalid version {version}")
        steps = document.get("steps")
        if not isinstance(steps, list):
            raise TuningFormatError(f"{where}: steps must be a list")
        parsed = [
            TuningStep.from_dict(entry, f"{where} steps[{index}]")
            for index, entry in enumerate(steps)
        ]
        history = cls(parsed)
        for step in parsed:
            history._by_run.setdefault(step.run_id, step)
        return history

    def export_json(self, filename: str | os.PathLike[str]) -> str:
        """Write the JSON form atomically; returns the path written."""
        return _write_text(
            filename, json.dumps(self.to_dict(), indent=2) + "\n"
        )

    def export_csv(self, filename: str | os.PathLike[str]) -> str:
        """Write the CSV form atomically; returns the path written."""
        return _write_text(filename, self.csv_text())

    def csv_text(self) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, step in enumerate(self._steps, start=1):
            writer.writerow(_csv_row(index, step))
        return buffer.getvalue()

    @classmethod
    def import_json(cls, filename: str | os.PathLike[str]) -> "TuningHistory":
        with open(filename, encoding="utf-8") as stream:
            try:
                data = json.load(stream)
            except json.JSONDecodeError as exc:
                raise TuningFormatError(f"invalid JSON: {exc}") from None
        return cls.from_dict(data, where=str(filename))

    @classmethod
    def import_csv(cls, filename: str | os.PathLike[str]) -> "TuningHistory":
        with open(filename, encoding="utf-8", newline="") as stream:
            return cls.from_csv_text(stream.read(), where=str(filename))

    @classmethod
    def from_csv_text(
        cls, text: str, *, where: str = "tuning history CSV"
    ) -> "TuningHistory":
        reader = csv.DictReader(io.StringIO(text))
        header = tuple(reader.fieldnames or ())
        if header != CSV_FIELDS:
            raise TuningFormatError(
                f"{where}: unexpected columns {header}; expected {CSV_FIELDS}"
            )
        steps = []
        for index, row in enumerate(reader):
            steps.append(
                TuningStep.from_dict(
                    _step_from_row(row), f"{where} row {index + 2}"
                )
            )
        return cls(steps)

    # -- presentation --------------------------------------------------
    def summary(self) -> str:
        if not self._steps:
            return "No tuning steps recorded yet."
        best = self.best()
        return (
            f"{len(self._steps)} step(s) recorded; best"
            f" {best.label} on {best.band or 'an unnamed band'}"
            f" at {best.timestamp:%Y-%m-%d %H:%M UTC}."
        )


def _same_value(a: float, b: float) -> bool:
    """Equality that treats two NaNs as the same stored value."""
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def _csv_row(index: int, step: TuningStep) -> dict[str, Any]:
    return {
        "index": index,
        "timestamp": _encode_time(step.timestamp),
        "band": step.band,
        "start_hz": step.start,
        "end_hz": step.end,
        "frequency_hz": step.frequency,
        "vswr": _encode_float(step.vswr),
        "return_loss": (
            "" if step.return_loss is None else _encode_float(step.return_loss)
        ),
        "location": step.location.value,
        "at_edge": "yes" if step.at_edge else "no",
        "samples": step.samples,
        "points": step.points,
        "segments": step.segments,
        "logarithmic": "yes" if step.logarithmic else "no",
        "snapshot_id": step.snapshot_id,
        "run_id": step.run_id,
        "trace_id": step.trace_id,
        "trace_name": step.trace_name,
        "notes": step.notes,
        "settings": json.dumps(dict(step.settings), sort_keys=True),
    }


def _step_from_row(row: Mapping[str, str]) -> dict[str, Any]:
    """Translate one CSV row back into the JSON shape of a step."""
    return_loss = (row.get("return_loss") or "").strip()
    try:
        settings = json.loads(row.get("settings") or "{}")
    except json.JSONDecodeError as exc:
        raise TuningFormatError(f"unreadable settings: {exc}") from None
    if not isinstance(settings, dict):
        raise TuningFormatError("settings must be an object")
    return {
        "band": row.get("band", ""),
        "start_hz": row["start_hz"],
        "end_hz": row["end_hz"],
        "points": row["points"],
        "segments": row["segments"],
        "logarithmic": (row.get("logarithmic", "no") or "").strip() == "yes",
        "frequency_hz": row["frequency_hz"],
        "vswr": row["vswr"],
        "return_loss": return_loss or None,
        "location": row.get("location", "measured"),
        "at_edge": (row.get("at_edge", "no") or "").strip() == "yes",
        "samples": row.get("samples", 0),
        "timestamp": row["timestamp"],
        "snapshot_id": row.get("snapshot_id", ""),
        "run_id": row["run_id"],
        "trace_id": row.get("trace_id", ""),
        "trace_name": row.get("trace_name", ""),
        "notes": row.get("notes", ""),
        "settings": settings,
        "completion": CompletionState.COMPLETE.value,
    }


def _write_text(filename: str | os.PathLike[str], text: str) -> str:
    """Write ``text`` atomically beside ``filename``; returns the path."""
    target = os.fspath(filename)
    directory = os.path.dirname(os.path.abspath(target)) or "."
    handle, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary, target)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:  # pragma: no cover - best effort cleanup
            pass
        raise
    return target
