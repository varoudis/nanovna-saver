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
"""Measurement limit sets and pass/fail checking (LIMIT-01, LIMIT-02).

A limit set is a small, versioned JSON document (see ``docs/LIMIT_SETS.md``)
describing frequency bands and the bounds the measurement has to satisfy
inside them.  Checking never modifies a measurement: it evaluates an
immutable :class:`~NanoVNASaver.Measurement.MeasurementSnapshot` and returns
typed :class:`LimitEvaluation` records.

Conventions, applied consistently to the JSON format, the GUI and reports:

* Frequencies are whole hertz, inclusive at both ends of a rule's band.
* ``s11_vswr`` is a ratio without a unit.
* ``s11_return_loss_db`` and ``s21_insertion_loss_db`` are positive for a
  passive match / a passive path (``-20*log10|S|``).  A gain therefore has a
  negative insertion loss, and a reflection above 1 has a negative return
  loss.
* A bound is inclusive: a value equal to ``maximum`` (or ``minimum``) passes;
  only a value strictly beyond it fails.
* Segments of one rule tile its band.  Each segment is half-open
  ``[start, stop)`` except the last, which includes its ``stop``; a sample on
  a shared boundary belongs to the later segment.
* Evaluation only uses measured samples.  No value is interpolated or
  extrapolated between samples.

Three outcomes are distinguished.  ``PASS`` means every sample inside every
required band satisfied its bounds.  ``FAIL`` means at least one sample
strictly violated a bound.  ``UNAVAILABLE`` means the check could not be
decided at all: the channel is missing, the data is empty or not complete,
the band is not covered by samples, the capture does not match the grid its
own recorded acquisition settings declare, or a sample inside the band is not
a finite complex number.  ``UNAVAILABLE`` is never reported as ``PASS``.

Coverage is checked in two tiers.  When the snapshot records the acquisition
settings (``provenance.sweep``), those settings are authoritative and the
capture must contain exactly the declared grid inside the band; a cluster of
missing samples is therefore detected even when the surviving gaps would
vouch for each other.  A captured sample that is equally consistent with two
declared points is ambiguous, not evidence for either, so it makes the check
``UNAVAILABLE``.  Without such metadata nothing declares the intended
resolution, so a conservative data-only policy is used and that limit is
documented rather than hidden.
"""

from __future__ import annotations

import bisect
import contextlib
import itertools as it
import json
import math
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .Measurement import (
    S11,
    S21,
    CompletionState,
    MeasurementSnapshot,
    SweepInfo,
)
from .RFTools import Datapoint

LIMIT_SET_FORMAT = "nanovna-saver-limits"
#: Version 2 added optional sloped bounds and the S11 phase metric.
#: Version 1 documents keep loading unchanged (no slopes, three metrics).
LIMIT_SET_VERSION = 2
LIMIT_SUFFIX = ".nvlimits"

#: Colour of a drawn bound in charts and reports, shared so the live chart and
#: the frozen report draw the same specification the same way.
LIMIT_BOUND_COLOR = "#777777"

#: A hole wider than this many times the *locally* observed sample spacing
#: means the band is not measured there, so the check is ``UNAVAILABLE``.
#: The reference spacing is taken from the neighbouring intervals, never
#: from one global median, so linear, logarithmic and segmented captures of
#: the same rule band are all judged by their own grid.
GAP_FACTOR = 1.5

#: Keys accepted in one segment object.  There is deliberately no unit key:
#: the unit is a property of the metric.
SEGMENT_KEYS = (
    "start_hz",
    "stop_hz",
    "minimum",
    "maximum",
    "minimum_slope",
    "maximum_slope",
)
RULE_KEYS = ("id", "label", "metric", "segments")
DOCUMENT_KEYS = ("format", "version", "name", "description", "rules")


class LimitError(Exception):
    """Base class for limit set problems the UI can report verbatim."""


class LimitFormatError(LimitError):
    """The document is unreadable, structurally wrong or too new."""


class LimitValidationError(LimitError):
    """The document parses but describes inconsistent limits."""


class Metric(Enum):
    """A measurable quantity a limit can be expressed in."""

    S11_VSWR = "s11_vswr"
    S11_RETURN_LOSS_DB = "s11_return_loss_db"
    S21_INSERTION_LOSS_DB = "s21_insertion_loss_db"
    S11_PHASE_DEG = "s11_phase_deg"

    @property
    def parameter(self) -> str:
        """S-parameter channel the metric is derived from."""
        if self is Metric.S21_INSERTION_LOSS_DB:
            return S21
        return S11

    @property
    def unit(self) -> str:
        if self is Metric.S11_VSWR:
            return ""
        if self is Metric.S11_PHASE_DEG:
            return "deg"
        return "dB"

    @property
    def description(self) -> str:
        if self is Metric.S11_VSWR:
            return "S11 VSWR (ratio)"
        if self is Metric.S11_RETURN_LOSS_DB:
            return "S11 return loss (dB, positive for a match)"
        if self is Metric.S11_PHASE_DEG:
            return "S11 phase (deg, -180 to +180, wrapped)"
        return "S21 insertion loss (dB, positive for loss)"

    def measure(self, point: Datapoint) -> float:
        """The metric for one measured sample.

        ``datapoint.vswr`` is infinite for a total reflection, which is a
        valid measurement: it violates every finite maximum rather than
        being treated as missing data.

        Phase is reported wrapped into ``(-180, +180]`` degrees.  A limit on
        wrapped phase is only meaningful where the measurement does not
        cross the +/-180 degree branch; the evaluator cannot detect that
        intent, so :meth:`S11_PHASE_DEG` documents it rather than guessing.
        """
        magnitude = abs(point.z)
        if self is Metric.S11_VSWR:
            return point.vswr
        if self is Metric.S11_PHASE_DEG:
            return math.degrees(point.phase)
        if magnitude == 0.0:
            return math.inf
        return -20.0 * math.log10(magnitude)

    @classmethod
    def parse(cls, text: object) -> "Metric":
        try:
            return cls(str(text))
        except ValueError:
            known = ", ".join(sorted(item.value for item in cls))
            raise LimitValidationError(
                f"unknown metric {text!r}; known metrics: {known}"
            ) from None

    def format_value(self, value: float) -> str:
        if math.isinf(value):
            return "infinite" if value > 0 else "-infinite"
        if math.isnan(value):
            return "not a number"
        return f"{value:.6g} {self.unit}".strip()


class Verdict(Enum):
    PASS = "pass"
    FAIL = "fail"
    UNAVAILABLE = "unavailable"

    @property
    def label(self) -> str:
        return self.value.upper()


def _encode_float(value: float | None) -> float | str | None:
    if value is None:
        return None
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def format_frequency_hz(freq: int) -> str:
    return f"{freq / 1e6:.6g} MHz"


def _require_object(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LimitFormatError(f"{where} must be a JSON object")
    return value


def _reject_unknown_keys(
    data: Mapping[str, Any], allowed: Sequence[str], where: str
) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise LimitValidationError(
            f"{where}: unknown key {unknown[0]!r};"
            f" accepted keys are {', '.join(allowed)}"
        )


def _number(value: object, where: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LimitValidationError(
            f"{where}: {field} must be a number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise LimitValidationError(
            f"{where}: {field} must be finite, got {value!r}"
        )
    return number


def _frequency(value: object, where: str, field: str) -> int:
    number = _number(value, where, field)
    if not number.is_integer():
        raise LimitValidationError(
            f"{where}: {field} must be a whole number of hertz, got {value!r}"
        )
    if number < 0:
        raise LimitValidationError(f"{where}: {field} must not be negative")
    return int(number)


@dataclass(frozen=True)
class LimitSegment:
    """Bounds applied to one contiguous frequency range.

    ``start`` is inclusive; ``stop`` is excluded unless the segment is the
    last one of its rule, where it is inclusive.

    A bound is either flat or sloped.  ``minimum``/``maximum`` give the value
    at ``start``; the optional ``*_slope`` fields give the change per hertz,
    so the interpolated bound at frequency ``f`` inside the segment is
    ``value + slope * (f - start)``.  Slope is expressed in the metric's own
    unit per hertz, which keeps a stored document independent of how the UI
    chooses to display it (the editor shows dB/GHz and converts).
    """

    start: int
    stop: int
    minimum: float | None = None
    maximum: float | None = None
    minimum_slope: float = 0.0
    maximum_slope: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or not isinstance(self.stop, int):
            raise LimitValidationError("segment bounds must be whole hertz")
        if self.start < 0:
            raise LimitValidationError("segment start must not be negative")
        if self.stop <= self.start:
            raise LimitValidationError(
                f"segment stop ({self.stop}) must be greater than"
                f" start ({self.start})"
            )
        for name in ("minimum", "maximum"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise LimitValidationError(
                    f"segment {name} must be a number or null"
                )
            if not math.isfinite(float(value)):
                raise LimitValidationError(
                    f"segment {name} must be finite, got {value!r}"
                )
        for name in ("minimum_slope", "maximum_slope"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise LimitValidationError(f"segment {name} must be a number")
            if not math.isfinite(float(value)):
                raise LimitValidationError(
                    f"segment {name} must be finite, got {value!r}"
                )
        if self.minimum is None and self.maximum is None:
            raise LimitValidationError(
                "segment must declare a minimum, a maximum, or both"
            )
        if self.minimum is None and self.minimum_slope:
            raise LimitValidationError(
                "segment declares a minimum_slope without a minimum"
            )
        if self.maximum is None and self.maximum_slope:
            raise LimitValidationError(
                "segment declares a maximum_slope without a maximum"
            )
        if (
            self.minimum is not None
            and self.maximum is not None
            and not (self.minimum_slope or self.maximum_slope)
            and float(self.minimum) > float(self.maximum)
        ):
            raise LimitValidationError(
                f"conflicting bounds: minimum ({self.minimum:g}) is greater"
                f" than maximum ({self.maximum:g})"
            )
        # A sloped pair can still cross, which would make the band impossible
        # over part of its range; that is a specification error, not
        # measurement noise, so it is rejected at construction.
        if (
            self.minimum is not None
            and self.maximum is not None
            and (self.minimum_slope or self.maximum_slope)
            and self.lowest_gap() < 0
        ):
            raise LimitValidationError(
                "the sloped minimum is above the sloped maximum somewhere"
                f" in {self.start} Hz to {self.stop} Hz"
            )

    # -- bounds --------------------------------------------------------
    def minimum_at(self, freq: int) -> float | None:
        if self.minimum is None:
            return None
        return float(self.minimum) + float(self.minimum_slope) * (
            freq - self.start
        )

    def maximum_at(self, freq: int) -> float | None:
        if self.maximum is None:
            return None
        return float(self.maximum) + float(self.maximum_slope) * (
            freq - self.start
        )

    def lowest_gap(self) -> float:
        """Smallest ``maximum - minimum`` over the segment (infinite when one
        side is unbounded)."""
        if self.minimum is None or self.maximum is None:
            return math.inf
        gaps = [
            self.maximum_at(freq) - self.minimum_at(freq)
            for freq in (self.start, self.stop)
        ]
        return min(gaps)

    def contains(self, freq: int, *, last: bool) -> bool:
        if freq < self.start:
            return False
        return freq <= self.stop if last else freq < self.stop

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "start_hz": self.start,
            "stop_hz": self.stop,
        }
        if self.minimum is not None:
            data["minimum"] = self.minimum
        if self.maximum is not None:
            data["maximum"] = self.maximum
        if self.minimum_slope:
            data["minimum_slope"] = self.minimum_slope
        if self.maximum_slope:
            data["maximum_slope"] = self.maximum_slope
        return data

    @classmethod
    def from_dict(cls, data: object, where: str) -> "LimitSegment":
        self_ = _require_object(data, where)
        _reject_unknown_keys(self_, SEGMENT_KEYS, where)
        for required in ("start_hz", "stop_hz"):
            if required not in self_:
                raise LimitValidationError(f"{where}: {required} is required")
        try:
            return cls(
                start=_frequency(self_["start_hz"], where, "start_hz"),
                stop=_frequency(self_["stop_hz"], where, "stop_hz"),
                minimum=(
                    None
                    if self_.get("minimum") is None
                    else _number(self_["minimum"], where, "minimum")
                ),
                maximum=(
                    None
                    if self_.get("maximum") is None
                    else _number(self_["maximum"], where, "maximum")
                ),
                minimum_slope=(
                    0.0
                    if self_.get("minimum_slope") is None
                    else _number(self_["minimum_slope"], where, "minimum_slope")
                ),
                maximum_slope=(
                    0.0
                    if self_.get("maximum_slope") is None
                    else _number(self_["maximum_slope"], where, "maximum_slope")
                ),
            )
        except LimitValidationError as exc:
            raise LimitValidationError(f"{where}: {exc}") from None


@dataclass(frozen=True)
class LimitRule:
    """One metric with piecewise frequency-dependent bounds."""

    metric: Metric
    segments: tuple[LimitSegment, ...]
    label: str = ""
    id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.metric, Metric):
            raise LimitValidationError("rule metric must be a Metric")
        if not self.segments:
            raise LimitValidationError("rule has no segments")
        ordered = tuple(sorted(self.segments, key=lambda item: item.start))
        object.__setattr__(self, "segments", ordered)
        for previous, following in it.pairwise(ordered):
            if following.start != previous.stop:
                relation = (
                    "overlap" if following.start < previous.stop else "gap"
                )
                raise LimitValidationError(
                    f"segments must tile the band without gaps or overlaps:"
                    f" a segment ends at {previous.stop} Hz but the next"
                    f" starts at {following.start} Hz ({relation})"
                )
        if not self.id or not self.id.strip():
            raise LimitValidationError("rule id must not be empty")

    @property
    def start(self) -> int:
        return self.segments[0].start

    @property
    def stop(self) -> int:
        return self.segments[-1].stop

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "metric": self.metric.value,
            "segments": [segment.to_dict() for segment in self.segments],
        }
        if self.label:
            data["label"] = self.label
        data["id"] = self.id
        return data

    @classmethod
    def from_dict(cls, data: object, where: str) -> "LimitRule":
        self_ = _require_object(data, where)
        _reject_unknown_keys(self_, RULE_KEYS, where)
        metric = Metric.parse(self_.get("metric"))
        segments = self_.get("segments")
        if not isinstance(segments, list) or not segments:
            raise LimitValidationError(
                f"{where}: segments must be a non-empty list"
            )
        parsed: list[LimitSegment] = []
        for index, entry in enumerate(segments):
            parsed.append(
                LimitSegment.from_dict(entry, f"{where}.segments[{index}]")
            )
        rule_id = str(self_.get("id") or "")
        try:
            return cls(
                metric=metric,
                segments=tuple(parsed),
                label=str(self_.get("label") or ""),
                id=rule_id or "rule",
            )
        except LimitValidationError as exc:
            raise LimitValidationError(f"{where}: {exc}") from None


@dataclass(frozen=True)
class LimitSet:
    """A named, versioned collection of limit rules."""

    name: str
    rules: tuple[LimitRule, ...]
    description: str = ""
    version: int = LIMIT_SET_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise LimitValidationError("limit set name must not be empty")
        if not self.rules:
            raise LimitValidationError("limit set contains no rules")
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise LimitValidationError(f"duplicate rule id {rule.id!r}")
            seen.add(rule.id)
        # Two rules for the same metric may meet at a band edge, but their
        # bands must not overlap: the result would be ambiguous.
        for index, rule in enumerate(self.rules):
            for other in self.rules[index + 1 :]:
                if rule.metric is not other.metric:
                    continue
                if rule.start < other.stop and other.start < rule.stop:
                    raise LimitValidationError(
                        f"rules {rule.id!r} and {other.id!r} both bound"
                        f" {rule.metric.value} over overlapping bands;"
                        " use one rule with piecewise segments"
                    )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "format": LIMIT_SET_FORMAT,
            "version": self.version,
            "name": self.name,
            "rules": [rule.to_dict() for rule in self.rules],
        }
        if self.description:
            data["description"] = self.description
        return data

    @classmethod
    def from_dict(cls, data: object) -> "LimitSet":
        self_ = _require_object(data, "limit set")
        _reject_unknown_keys(self_, DOCUMENT_KEYS, "limit set")
        if self_.get("format") != LIMIT_SET_FORMAT:
            raise LimitFormatError(
                f"unexpected limit set format {self_.get('format')!r};"
                f" expected {LIMIT_SET_FORMAT!r}"
            )
        version = self_.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise LimitFormatError(
                f"limit set version must be an integer, got {version!r}"
            )
        if version > LIMIT_SET_VERSION:
            raise LimitFormatError(
                f"limit set version {version} is newer than supported"
                f" version {LIMIT_SET_VERSION}; upgrade NanoVNASaver"
            )
        if version < 1:
            raise LimitFormatError(f"invalid limit set version {version}")
        rules = self_.get("rules")
        if not isinstance(rules, list) or not rules:
            raise LimitValidationError(
                "rules must be a non-empty list of rule objects"
            )
        parsed = []
        for index, entry in enumerate(rules):
            rule = LimitRule.from_dict(entry, f"rules[{index}]")
            if not rule.id.strip() or rule.id == "rule":
                rule = LimitRule(
                    metric=rule.metric,
                    segments=rule.segments,
                    label=rule.label,
                    id=f"rule-{index + 1}",
                )
            parsed.append(rule)
        return cls(
            name=str(self_.get("name") or ""),
            rules=tuple(parsed),
            description=str(self_.get("description") or ""),
            version=version,
        )

    # -- evaluation ----------------------------------------------------
    def evaluate(
        self,
        snapshot: MeasurementSnapshot,
        *,
        trace_id: str = "",
        trace_name: str = "",
    ) -> tuple["LimitEvaluation", ...]:
        return tuple(
            evaluate_rule(
                rule,
                snapshot,
                trace_id=trace_id,
                trace_name=trace_name,
            )
            for rule in self.rules
        )


def loads(text: str) -> LimitSet:
    """Parse a limit set document, raising :class:`LimitError`."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LimitFormatError(
            f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc
    return LimitSet.from_dict(document)


def dumps(limit_set: LimitSet) -> str:
    """Serialize ``limit_set`` as a stable, human editable document.

    Key order and indentation are fixed so a saved file diffs cleanly and can
    still be edited by hand; floats are written exactly as stored, because a
    bound is a specification and must not drift when it is re-read.
    """
    return json.dumps(limit_set.to_dict(), indent=2, sort_keys=False) + "\n"


def save(limit_set: LimitSet, filename: str) -> None:
    """Write ``limit_set`` to ``filename`` atomically.

    The document is validated by construction, and the temporary file is
    replaced only once it has been written and flushed, so a failure part way
    through cannot leave a truncated limit set where a valid one was.
    """
    text = dumps(limit_set)
    directory = os.path.dirname(os.path.abspath(filename)) or "."
    handle, temporary = tempfile.mkstemp(
        dir=directory, prefix=".nvlimits-", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, filename)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def load(filename: str) -> LimitSet:
    """Read and validate one limit set file; never mutates the source."""
    try:
        with open(filename, encoding="utf-8") as source:
            text = source.read()
    except OSError as exc:
        raise LimitFormatError(f"could not read {filename}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise LimitFormatError(f"{filename} is not a UTF-8 text file") from exc
    try:
        return loads(text)
    except LimitError as exc:
        raise type(exc)(f"{filename}: {exc}") from None


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LimitSegmentEvaluation:
    """Outcome of one segment of a rule."""

    start: int
    stop: int
    verdict: Verdict
    samples: int = 0
    worst_value: float | None = None
    worst_frequency: int | None = None
    worst_limit: float | None = None
    worst_kind: str = ""
    worst_violation: float | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "start_hz": self.start,
            "stop_hz": self.stop,
            "verdict": self.verdict.value,
            "samples": self.samples,
            "worst_value": _encode_float(self.worst_value),
            "worst_frequency_hz": self.worst_frequency,
            "worst_limit": _encode_float(self.worst_limit),
            "worst_kind": self.worst_kind,
            "worst_violation": _encode_float(self.worst_violation),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LimitEvaluation:
    """Outcome of one rule against one trace."""

    rule_id: str
    rule_label: str
    metric: Metric
    trace_id: str
    trace_name: str
    verdict: Verdict
    band_start: int
    band_stop: int
    samples: int = 0
    worst_value: float | None = None
    worst_frequency: int | None = None
    worst_limit: float | None = None
    worst_kind: str = ""
    worst_violation: float | None = None
    segments: tuple[LimitSegmentEvaluation, ...] = ()
    reason: str = ""

    @property
    def title(self) -> str:
        return self.rule_label or self.rule_id

    def describe(self) -> str:
        """One-line human readable summary used by the GUI and reports."""
        if self.verdict is Verdict.UNAVAILABLE:
            return f"{self.title}: unavailable ({self.reason or 'unknown'})"
        text = f"{self.title}: {self.verdict.label}"
        if self.worst_value is not None:
            text += (
                f", worst {self.metric.format_value(self.worst_value)}"
                f" at {format_frequency_hz(self.worst_frequency or 0)}"
            )
        if self.verdict is Verdict.FAIL and self.worst_limit is not None:
            text += (
                f" (limit {self.worst_kind}"
                f" {self.metric.format_value(self.worst_limit)})"
            )
        return text

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_label": self.rule_label,
            "metric": self.metric.value,
            "metric_unit": self.metric.unit,
            "trace_id": self.trace_id,
            "trace_name": self.trace_name,
            "verdict": self.verdict.value,
            "band_start_hz": self.band_start,
            "band_stop_hz": self.band_stop,
            "samples": self.samples,
            "worst_value": _encode_float(self.worst_value),
            "worst_frequency_hz": self.worst_frequency,
            "worst_limit": _encode_float(self.worst_limit),
            "worst_kind": self.worst_kind,
            "worst_violation": _encode_float(self.worst_violation),
            "reason": self.reason,
            "segments": [item.to_dict() for item in self.segments],
        }


def _unavailable(
    rule: LimitRule,
    trace_id: str,
    trace_name: str,
    reason: str,
    *,
    samples: int = 0,
) -> LimitEvaluation:
    return LimitEvaluation(
        rule_id=rule.id,
        rule_label=rule.label,
        metric=rule.metric,
        trace_id=trace_id,
        trace_name=trace_name,
        verdict=Verdict.UNAVAILABLE,
        band_start=rule.start,
        band_stop=rule.stop,
        samples=samples,
        reason=reason,
    )


def _finite_point(point: Datapoint) -> bool:
    return math.isfinite(point.re) and math.isfinite(point.im)


def _step_references(
    channel: Sequence[Datapoint],
) -> list[tuple[int, int, float]]:
    """``(low, high, reference_spacing)`` for every increasing interval.

    ``reference_spacing`` is the largest spacing observed immediately next
    to the interval.  A grid whose spacing varies smoothly (a logarithmic
    sweep, or the joins between sweep segments) keeps a ratio near 1, while
    a sample missing from an otherwise regular stretch makes that interval
    several times wider than either neighbour.  Taking the *largest*
    neighbour is the conservative choice: it tolerates a rising step and
    still detects a missing sample, because a hole must also jump over the
    data that would otherwise be there.
    """
    pairs = [
        (previous.freq, following.freq)
        for previous, following in it.pairwise(channel)
        if following.freq > previous.freq
    ]
    sizes = [high - low for low, high in pairs]
    references = []
    for index, (low, high) in enumerate(pairs):
        neighbours = [
            sizes[other]
            for other in (index - 1, index + 1)
            if 0 <= other < len(sizes)
        ]
        references.append(
            (low, high, float(max(neighbours) if neighbours else sizes[index]))
        )
    return references


#: Rounding budget for a declared grid, in parts per million.
#:
#: Two legitimate differences separate a device's reported grid from the
#: arithmetic in the recorded settings.  First, the device reports whole hertz
#: computed by its own firmware, which differs in the last digit.  Second,
#: firmwares such as the NanoVNA-V2 truncate the sweep step to an integer
#: number of hertz and then add it per point (see
#: ``Hardware.NanoVNA_V2.read_frequencies``), so the deviation grows by up to
#: one hertz per point inside a segment.  Both are modelled below.  The budget
#: is capped at half the local sample spacing, and a captured sample that
#: lands exactly on that half-step boundary belongs to neither neighbour: it is
#: reported as ambiguous instead of being claimed by whichever point the match
#: happens to reach first (see :func:`_ambiguous_declared_points`).
GRID_ROUNDING_PPM = 2.0


def _grid_tolerance(freq: int, spacing: int, points: int) -> float:
    """Accepted difference between a declared and a captured frequency.

    ``spacing`` is the smallest positive spacing touching the declared
    frequency and ``points`` the number of samples per declared segment, which
    bounds the accumulated integer-step truncation described above.  The
    result never exceeds half of ``spacing``, so the acceptance windows of two
    adjacent declared points can only meet at a single boundary frequency.
    """
    rounding = max(1.0, abs(freq) * GRID_ROUNDING_PPM / 1e6)
    rounding += max(points - 1, 0)
    return min(spacing / 2.0, rounding) if spacing > 0 else rounding


def _local_spacings(expected: Sequence[int]) -> list[int]:
    """Smallest positive spacing touching each expected frequency."""
    spacing = [
        following - previous for previous, following in it.pairwise(expected)
    ]
    positive = [value for value in spacing if value > 0]
    fallback = min(positive) if positive else 0
    out = []
    for index in range(len(expected)):
        neighbours = [
            spacing[other]
            for other in (index - 1, index)
            if 0 <= other < len(spacing) and spacing[other] > 0
        ]
        out.append(min(neighbours) if neighbours else fallback)
    return out


def _ambiguous_declared_points(
    expected: Sequence[int],
    spacings: Sequence[int],
    actual: Sequence[int],
    points: int,
    rule: "LimitRule",
) -> tuple[int, int, int] | None:
    """First ``(freq, low, high)`` captured sample that matches two points.

    A sample may only be owned by one declared frequency.  With the tolerance
    capped at half the local spacing, two acceptance windows overlap at most at
    one exact boundary frequency, so an overlap means the capture contains a
    frequency that is equally consistent with two different declared points.
    Such a grid is ambiguous: attributing the sample to either neighbour would
    let a duplicate stand in for a missing frequency, which is exactly what the
    completeness check exists to prevent.  Equal declared values (a
    multi-segment join) are one ownership candidate, not two, and only
    candidates inside the rule's band can make a rule ambiguous.
    """
    for freq in actual:
        if not rule.start <= freq <= rule.stop:
            continue
        index = bisect.bisect_left(expected, freq)
        matched = {
            expected[probe]
            for probe in (index - 1, index)
            if 0 <= probe < len(expected)
            and rule.start <= expected[probe] <= rule.stop
            and abs(expected[probe] - freq)
            <= _grid_tolerance(expected[probe], spacings[probe], points)
        }
        if len(matched) > 1:
            low, high = sorted(matched)[:2]
            return freq, low, high
    return None


def _declared_grid_problem(
    channel: Sequence[Datapoint],
    rule: "LimitRule",
    sweep: SweepInfo,
    expected_grid: Sequence[int],
) -> str | None:
    """Compare the capture with the grid its own settings declare.

    A stored acquisition sweep is authoritative evidence about which samples
    were requested, so the completed capture must contain exactly that grid:
    one sample per declared frequency, in order, allowing for the device's
    rounding.  A cluster of missing samples is therefore detected even when
    the surviving gaps are wide enough to vouch for each other, and a
    duplicate cannot stand in for a different frequency.

    Ownership is exclusive.  A captured frequency that is equally consistent
    with two different declared points (the half-step tie, which the capped
    tolerance can just reach) is ambiguous rather than being awarded to
    whichever neighbour the match reaches first, because that award would let
    one sample satisfy two declared samples.

    The whole declared grid is matched (a capture is measured as a whole), but
    only discrepancies inside the rule's band are reported: a rule narrower
    than the sweep must not fail because of a hole elsewhere.  ``None`` means
    the capture is consistent with the declared grid inside the band.
    """
    if not expected_grid:
        return None  # unusable settings: the data-only policy applies
    actual = [dp.freq for dp in channel]
    spacings = _local_spacings(expected_grid)
    points = max(sweep.points, 1)
    grid = (
        f"{sweep.points} point(s) x {sweep.segments} segment(s) from"
        f" {format_frequency_hz(sweep.start)} to"
        f" {format_frequency_hz(sweep.end)}"
        + (", logarithmic" if sweep.logarithmic else "")
    )
    in_band = [freq for freq in actual if rule.start <= freq <= rule.stop]
    ambiguous = _ambiguous_declared_points(
        expected_grid, spacings, in_band, points, rule
    )
    if ambiguous is not None:
        freq, low, high = ambiguous
        return (
            f"capture frequency {freq} Hz ({format_frequency_hz(freq)}) is"
            f" equally consistent with the declared samples at {low} Hz and"
            f" {high} Hz of the recorded sweep ({grid}); which sample is"
            f" missing cannot be decided"
        )
    consumed = 0
    missing: list[int] = []
    extras: list[int] = []
    for index, freq in enumerate(expected_grid):
        tolerance = _grid_tolerance(freq, spacings[index], points)
        probe = consumed
        while probe < len(actual) and actual[probe] < freq - tolerance:
            if rule.start <= actual[probe] <= rule.stop:
                extras.append(actual[probe])
            probe += 1
        # A declared point inside the band must be satisfied by a sample
        # inside the band; a declared point outside the band may consume any
        # sample, since it only keeps the ordered walk aligned and is never
        # reported itself.
        in_declared_band = rule.start <= freq <= rule.stop
        matched = probe < len(actual) and actual[probe] <= freq + tolerance
        if matched and (
            not in_declared_band or rule.start <= actual[probe] <= rule.stop
        ):
            consumed = probe + 1
        elif in_declared_band:
            missing.append(freq)
    extras.extend(
        freq for freq in actual[consumed:] if rule.start <= freq <= rule.stop
    )
    if not missing and not extras:
        return None
    total = sum(1 for freq in expected_grid if rule.start <= freq <= rule.stop)
    if missing:
        return (
            f"capture is missing {len(missing)} of {total} samples declared"
            f" by the recorded sweep ({grid}); first missing sample at"
            f" {missing[0]} Hz ({format_frequency_hz(missing[0])})"
        )
    return (
        f"capture has {len(extras)} sample(s) that are not on the grid"
        f" declared by the recorded sweep ({grid}); first extra sample at"
        f" {extras[0]} Hz ({format_frequency_hz(extras[0])})"
    )


def coverage_gap(channel: Sequence[Datapoint], rule: "LimitRule") -> str | None:
    """Describe the first hole in sample coverage over the rule's band.

    This is the *data-only* tier of the coverage policy.  It is used whenever
    the snapshot carries no acquisition settings, and it also runs after the
    declared-grid check to catch a capture that is complete but still
    irregular.  Coverage is judged against the captured grid, not an assumed
    uniform one: every interval between consecutive samples (and between the
    band edges and the outermost samples) is compared with the spacing
    observed around it.  That accepts legitimate non-uniform grids -- a
    logarithmic sweep, a multi-segment sweep whose steps change at a segment
    join -- and still reports an isolated missing sample or chunk as a hole.

    The band edges are checked too, which catches a rule that starts inside a
    missing chunk.  The tolerance is derived from the data itself; no
    intended resolution is invented.  A cluster of missing samples whose
    surviving intervals each justify their neighbour is indistinguishable
    from a genuinely coarser capture here, so it is accepted -- see
    :func:`_declared_grid_problem`, which resolves that case from the
    recorded acquisition settings when they exist.
    """
    steps = _step_references(channel)
    if not steps:
        return None
    inside = [dp.freq for dp in channel if rule.start <= dp.freq <= rule.stop]
    if not inside:
        return None  # the band bounds check reports this case
    edges = [rule.start, *inside, rule.stop]
    cursor = 0
    for previous, following in it.pairwise(edges):
        span = following - previous
        if span <= 0:
            continue
        while cursor < len(steps) and steps[cursor][1] <= previous:
            cursor += 1
        reference = 0.0
        probe = cursor
        while probe < len(steps) and steps[probe][0] < following:
            low, high, spacing = steps[probe]
            if high > previous:
                reference = max(reference, spacing)
            probe += 1
        if reference > 0.0 and span > GAP_FACTOR * reference:
            return (
                f"no samples between {format_frequency_hz(previous)} and"
                f" {format_frequency_hz(following)}"
            )
    return None


def _worst_of(
    points: Iterable[Datapoint],
    segment: LimitSegment,
    metric: Metric,
) -> tuple[float, int, float, str, float]:
    """Worst (value, frequency, limit, kind, violation) inside a segment.

    ``violation`` is positive when the value is outside the bound and zero or
    negative when it is inside; the largest violation wins, so an infinite
    VSWR from a total reflection correctly beats any finite excess.

    A sloped bound is evaluated at each sample's own frequency.  Comparing a
    sample against the bound at the segment start would let a sloped limit
    pass data it was written to reject, so the per-sample bound is used
    whenever a slope is present.
    """
    values = [(metric.measure(point), point.freq) for point in points]
    sloped = bool(segment.minimum_slope or segment.maximum_slope)
    candidates: list[tuple[float, float, int, float, str]] = []
    if segment.maximum is not None:
        if sloped:
            best = max(
                (
                    (
                        value - segment.maximum_at(freq),
                        value,
                        freq,
                        segment.maximum_at(freq),
                        "maximum",
                    )
                    for value, freq in values
                ),
                key=lambda item: item[0],
            )
        else:
            value, freq = max(values, key=lambda item: item[0])
            best = (
                value - float(segment.maximum),
                value,
                freq,
                float(segment.maximum),
                "maximum",
            )
        candidates.append(best)
    if segment.minimum is not None:
        if sloped:
            best = max(
                (
                    (
                        segment.minimum_at(freq) - value,
                        value,
                        freq,
                        segment.minimum_at(freq),
                        "minimum",
                    )
                    for value, freq in values
                ),
                key=lambda item: item[0],
            )
        else:
            value, freq = min(values, key=lambda item: item[0])
            best = (
                float(segment.minimum) - value,
                value,
                freq,
                float(segment.minimum),
                "minimum",
            )
        candidates.append(best)
    best = candidates[0]
    for candidate in candidates[1:]:
        if candidate[0] > best[0]:
            best = candidate
    return best[1], best[2], best[3], best[4], best[0]


def evaluate_rule(
    rule: LimitRule,
    snapshot: MeasurementSnapshot,
    *,
    trace_id: str = "",
    trace_name: str = "",
) -> LimitEvaluation:
    """Check one rule against one immutable snapshot."""
    completion = snapshot.provenance.completion
    if completion is not CompletionState.COMPLETE:
        return _unavailable(
            rule,
            trace_id,
            trace_name,
            f"data is {completion.value}, not complete",
        )
    channel = snapshot.channel(rule.metric.parameter)
    if channel is None:
        return _unavailable(
            rule,
            trace_id,
            trace_name,
            f"no {rule.metric.parameter} channel in this trace",
        )
    if not channel:
        return _unavailable(
            rule,
            trace_id,
            trace_name,
            f"empty {rule.metric.parameter} channel",
        )
    if channel[0].freq > rule.start or channel[-1].freq < rule.stop:
        return _unavailable(
            rule,
            trace_id,
            trace_name,
            f"band {format_frequency_hz(rule.start)} to"
            f" {format_frequency_hz(rule.stop)} is not covered by"
            f" {rule.metric.parameter} data"
            f" ({format_frequency_hz(channel[0].freq)} to"
            f" {format_frequency_hz(channel[-1].freq)})",
            samples=len(channel),
        )
    # The recorded acquisition settings are authoritative about which samples
    # were requested, so they are checked before any data-only heuristic: a
    # cluster of missing samples must not be able to vouch for itself.
    sweep = snapshot.provenance.sweep
    if sweep is not None:
        grid_problem = _declared_grid_problem(
            channel, rule, sweep, sweep.declared_frequencies()
        )
        if grid_problem is not None:
            return _unavailable(
                rule,
                trace_id,
                trace_name,
                grid_problem,
                samples=len(channel),
            )
    gap = coverage_gap(channel, rule)
    if gap is not None:
        return _unavailable(
            rule, trace_id, trace_name, gap, samples=len(channel)
        )
    band = [point for point in channel if rule.start <= point.freq <= rule.stop]
    invalid = next((point for point in band if not _finite_point(point)), None)
    if invalid is not None:
        return _unavailable(
            rule,
            trace_id,
            trace_name,
            f"non-finite {rule.metric.parameter} sample at"
            f" {format_frequency_hz(invalid.freq)}"
            f" ({invalid.re!r}, {invalid.im!r})",
            samples=len(band),
        )

    last_index = len(rule.segments) - 1
    segment_results: list[LimitSegmentEvaluation] = []
    for index, segment in enumerate(rule.segments):
        points = [
            point
            for point in band
            if segment.contains(point.freq, last=index == last_index)
        ]
        if not points:
            segment_results.append(
                LimitSegmentEvaluation(
                    start=segment.start,
                    stop=segment.stop,
                    verdict=Verdict.UNAVAILABLE,
                    reason=(
                        f"no samples between"
                        f" {format_frequency_hz(segment.start)} and"
                        f" {format_frequency_hz(segment.stop)}"
                    ),
                )
            )
            continue
        value, frequency, limit, kind, violation = _worst_of(
            points, segment, rule.metric
        )
        segment_results.append(
            LimitSegmentEvaluation(
                start=segment.start,
                stop=segment.stop,
                verdict=(Verdict.FAIL if violation > 0.0 else Verdict.PASS),
                samples=len(points),
                worst_value=value,
                worst_frequency=frequency,
                worst_limit=limit,
                worst_kind=kind,
                worst_violation=violation,
            )
        )

    unavailable = next(
        (
            item
            for item in segment_results
            if item.verdict is Verdict.UNAVAILABLE
        ),
        None,
    )
    if unavailable is not None:
        return LimitEvaluation(
            rule_id=rule.id,
            rule_label=rule.label,
            metric=rule.metric,
            trace_id=trace_id,
            trace_name=trace_name,
            verdict=Verdict.UNAVAILABLE,
            band_start=rule.start,
            band_stop=rule.stop,
            samples=len(band),
            segments=tuple(segment_results),
            reason=unavailable.reason,
        )

    worst = max(segment_results, key=lambda item: item.worst_violation or 0.0)
    failed = any(item.verdict is Verdict.FAIL for item in segment_results)
    return LimitEvaluation(
        rule_id=rule.id,
        rule_label=rule.label,
        metric=rule.metric,
        trace_id=trace_id,
        trace_name=trace_name,
        verdict=Verdict.FAIL if failed else Verdict.PASS,
        band_start=rule.start,
        band_stop=rule.stop,
        samples=len(band),
        worst_value=worst.worst_value,
        worst_frequency=worst.worst_frequency,
        worst_limit=worst.worst_limit,
        worst_kind=worst.worst_kind,
        worst_violation=worst.worst_violation,
        segments=tuple(segment_results),
    )


class LimitCatalogue:
    """Limit sets loaded into this session, with one elected as active.

    Loading is all-or-nothing: a rejected document never removes or replaces
    a set that already loaded successfully, so a typo in a file cannot leave
    the application without the limits the operator was using.
    """

    def __init__(self) -> None:
        self._sets: list[LimitSet] = []
        self._active: str = ""

    def __iter__(self):
        return iter(self._sets)

    def __len__(self) -> int:
        return len(self._sets)

    def all(self) -> tuple[LimitSet, ...]:
        return tuple(self._sets)

    def names(self) -> list[str]:
        return [limit_set.name for limit_set in self._sets]

    def find(self, name: str) -> LimitSet | None:
        return next((item for item in self._sets if item.name == name), None)

    def add(self, limit_set: LimitSet) -> LimitSet:
        """Store ``limit_set``, replacing a set of the same name."""
        existing = self.find(limit_set.name)
        if existing is not None:
            self._sets[self._sets.index(existing)] = limit_set
        else:
            self._sets.append(limit_set)
        if not self._active:
            self._active = limit_set.name
        return limit_set

    def load(self, filename: str) -> LimitSet:
        """Read, validate and store a file; nothing changes on failure."""
        return self.add(load(filename))

    def remove(self, name: str) -> bool:
        found = self.find(name)
        if found is None:
            return False
        self._sets.remove(found)
        if self._active == name:
            self._active = self._sets[0].name if self._sets else ""
        return True

    def clear(self) -> None:
        self._sets.clear()
        self._active = ""

    @property
    def active_name(self) -> str:
        return self._active

    def active(self) -> LimitSet | None:
        return self.find(self._active) if self._active else None

    def activate(self, name: str) -> LimitSet | None:
        found = self.find(name)
        if found is None:
            raise LimitError(f"no limit set named {name!r}")
        self._active = name
        return found

    def deactivate(self) -> None:
        self._active = ""
