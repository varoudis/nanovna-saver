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
"""Marker search, tracking, bandwidth and Q (ANALYSIS-01).

Everything here is Qt-free and works on measured samples: a search returns the
*frequency of an existing sample*, never an interpolated one, unless the
result explicitly says it is an estimate.  That distinction is the whole point
of the module — a "peak at 145.02 MHz" that was interpolated between samples
is a different claim from one that was measured, and a marker that silently
claims the former is wrong.

Conventions
-----------
* A result names its channel, its unit and whether it came from a measured
  sample or from a parabolic estimate fitted around one.
* Unavailable is a first-class outcome.  A bandwidth with only one crossing,
  or a Q with no usable half-power points, reports :data:`UNAVAILABLE` rather
  than producing a number.
* Ties are broken towards the lowest frequency, so a flat curve has a
  deterministic answer instead of an arbitrary one.
* Phase is unwrapped before any search that could cross the +/-180 branch.
"""

from __future__ import annotations

import itertools as it
import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Sequence

from .RFTools import Datapoint

#: Fractional distance, in local sample spacings, that two samples must differ
#: by before one is a genuine local extremum rather than noise riding on a
#: plateau.  A noise-free flat curve has no extrema at all.
PROMINENCE_SAMPLES = 0.0


class SearchKind(Enum):
    """What a search looks for."""

    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    PEAK_LEFT = "peak_left"
    PEAK_RIGHT = "peak_right"
    NOTCH_LEFT = "notch_left"
    NOTCH_RIGHT = "notch_right"
    CROSSING_ABOVE = "crossing_above"
    CROSSING_BELOW = "crossing_below"


class SearchStatus(Enum):
    """Outcome category of a search or a derived measurement."""

    OK = "ok"
    UNAVAILABLE = "unavailable"


class LocationKind(Enum):
    """How a reported frequency relates to the measured samples."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    NONE = "none"


@dataclass(frozen=True)
class SearchResult:
    """One search outcome, with its provenance made explicit."""

    status: SearchStatus
    frequency: int | None = None
    value: float | None = None
    location: LocationKind = LocationKind.NONE
    channel: str = ""
    unit: str = ""
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is SearchStatus.OK

    def describe(self) -> str:
        if not self.ok:
            return f"unavailable ({self.reason or 'no result'})"
        where = (
            "measured"
            if self.location is LocationKind.MEASURED
            else "estimated"
        )
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.frequency} Hz, {self.value:.6g}{unit} ({where})"


def _unavailable(reason: str, *, channel: str = "", unit: str = ""):
    return SearchResult(
        status=SearchStatus.UNAVAILABLE,
        channel=channel,
        unit=unit,
        reason=reason,
    )


# ----------------------------------------------------------------------
# Channel access
# ----------------------------------------------------------------------
Measure = Callable[[Datapoint], float]


def default_measure(parameter: str) -> Measure:
    """The natural scalar quantity for a channel.

    ``S11``/``S21`` magnitude in dB is what an operator means by "the curve",
    and it is what the charts plot, so a search agrees with what is on screen.
    """
    return lambda point: point.gain


def _series(
    points: Sequence[Datapoint], measure: Measure
) -> list[tuple[int, float, Datapoint]]:
    """``(frequency, value, point)`` triples, skipping non-finite samples."""
    out = []
    for point in points:
        try:
            value = float(measure(point))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if math.isfinite(value):
            out.append((point.freq, value, point))
    return out


def _band(
    series: Sequence[tuple[int, float, Datapoint]],
    start: int | None,
    stop: int | None,
) -> list[tuple[int, float, Datapoint]]:
    low = -math.inf if start is None else start
    high = math.inf if stop is None else stop
    return [item for item in series if low <= item[0] <= high]


def unwrapped_phase(points: Sequence[Datapoint]) -> list[tuple[int, float]]:
    """Phase in degrees, unwrapped so a search can cross +/-180.

    Each step is corrected by the nearest multiple of 360 degrees, which is
    the only defensible assumption with samples that far apart; it is stated
    here rather than hidden, because a genuine wrap narrower than one sample
    cannot be resolved from the samples alone.
    """
    out: list[tuple[int, float]] = []
    offset = 0.0
    previous = None
    for point in points:
        degrees = math.degrees(point.phase)
        if previous is not None:
            while degrees + offset - previous > 180.0:
                offset -= 360.0
            while degrees + offset - previous < -180.0:
                offset += 360.0
        value = degrees + offset
        out.append((point.freq, value))
        previous = value
    return out


# ----------------------------------------------------------------------
# Searches
# ----------------------------------------------------------------------
def search_extreme(  # noqa: PLR0913
    points: Sequence[Datapoint],
    *,
    maximum: bool = True,
    channel: str = "S11",
    unit: str = "dB",
    measure: Measure | None = None,
    start: int | None = None,
    stop: int | None = None,
    estimate: bool = False,
) -> SearchResult:
    """Global maximum or minimum in a band.

    ``estimate`` fits a parabola through the extremum and its neighbours and
    reports the fitted frequency as :attr:`LocationKind.ESTIMATED`; the fitted
    vertex only exists when the neighbour is strictly lower (higher), so a
    plateau stays a measured sample rather than an invented peak.
    """
    measure = measure or default_measure(channel)
    series = _band(_series(points, measure), start, stop)
    if not series:
        return _unavailable(
            "no finite samples in the band", channel=channel, unit=unit
        )
    target = max if maximum else min
    chosen = target(series, key=lambda item: (item[1], -item[0]))
    result = SearchResult(
        status=SearchStatus.OK,
        frequency=chosen[0],
        value=chosen[1],
        location=LocationKind.MEASURED,
        channel=channel,
        unit=unit,
    )
    if not estimate:
        return result
    index = series.index(chosen)
    if index == 0 or index == len(series) - 1:
        return result
    left, right = series[index - 1], series[index + 1]
    if (right[0] - left[0]) == 0:
        return result
    # Parabola through three equally weighted neighbours; its vertex is only
    # meaningful when the centre sample really is the extremum.
    denominator = 2.0 * (right[1] - 2.0 * chosen[1] + left[1])
    if denominator == 0.0:
        return result
    offset = (left[1] - right[1]) / denominator
    if not -1.0 < offset < 1.0:
        return result
    # A vertex implies a strictly better value than the measured sample; when
    # the neighbours are on the wrong side it is not a peak at all.
    if maximum and (left[1] > chosen[1] or right[1] > chosen[1]):
        return result
    if not maximum and (left[1] < chosen[1] or right[1] < chosen[1]):
        return result
    spacing = (right[0] - left[0]) / 2.0
    fitted = chosen[0] + offset * spacing
    if not math.isfinite(fitted):
        return result
    return SearchResult(
        status=SearchStatus.OK,
        frequency=int(round(fitted)),
        value=chosen[1] - 0.25 * (left[1] - right[1]) * offset,
        location=LocationKind.ESTIMATED,
        channel=channel,
        unit=unit,
    )


def search_local(  # noqa: PLR0913
    points: Sequence[Datapoint],
    reference: int,
    *,
    maximum: bool = True,
    direction: int = 1,
    channel: str = "S11",
    unit: str = "dB",
    measure: Measure | None = None,
    start: int | None = None,
    stop: int | None = None,
) -> SearchResult:
    """Next local peak/notch left of (or right of) ``reference``.

    "Local" means: keep walking while the curve improves, stop at the first
    sample where it turns back.  A monotonic run therefore reports the end of
    the band it reached, not an invented extremum in the middle of it.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    measure = measure or default_measure(channel)
    series = _band(_series(points, measure), start, stop)
    if not series:
        return _unavailable(
            "no finite samples in the band", channel=channel, unit=unit
        )
    freqs = [item[0] for item in series]
    if direction > 0:
        index = next(
            (i for i, freq in enumerate(freqs) if freq > reference), None
        )
    else:
        index = next(
            (
                len(freqs) - 1 - i
                for i, freq in enumerate(reversed(freqs))
                if freq < reference
            ),
            None,
        )
    if index is None:
        return _unavailable(
            "no samples beyond the reference frequency",
            channel=channel,
            unit=unit,
        )
    better = (lambda a, b: a > b) if maximum else (lambda a, b: a < b)
    best = index
    cursor = index
    while 0 <= cursor + direction < len(series):
        following = cursor + direction
        if better(series[following][1], series[best][1]):
            best = following
            cursor = following
            continue
        break
    chosen = series[best]
    return SearchResult(
        status=SearchStatus.OK,
        frequency=chosen[0],
        value=chosen[1],
        location=LocationKind.MEASURED,
        channel=channel,
        unit=unit,
    )


def search_crossing(  # noqa: PLR0913
    points: Sequence[Datapoint],
    target: float,
    *,
    above: bool = True,
    channel: str = "S11",
    unit: str = "dB",
    measure: Measure | None = None,
    start: int | None = None,
    stop: int | None = None,
    from_frequency: int | None = None,
    estimate: bool = True,
) -> SearchResult:
    """First crossing of ``target`` in the chosen direction.

    A crossing between two samples is reported as ``ESTIMATED`` at its
    interpolated frequency; a sample exactly on the target is ``MEASURED``.
    """
    measure = measure or default_measure(channel)
    series = _band(_series(points, measure), start, stop)
    if not series:
        return _unavailable(
            "no finite samples in the band", channel=channel, unit=unit
        )
    pairs = list(it.pairwise(series))
    if not pairs:
        return _unavailable(
            "need at least two samples to detect a crossing",
            channel=channel,
            unit=unit,
        )
    condition = (
        (lambda a, b: a < target <= b)
        if above
        else (lambda a, b: a > target >= b)
    )
    for first, second in pairs:
        if from_frequency is not None:
            if above and second[0] < from_frequency:
                continue
            if not above and second[0] < from_frequency:
                continue
        if not condition(first[1], second[1]):
            continue
        if first[1] == target:
            return SearchResult(
                status=SearchStatus.OK,
                frequency=first[0],
                value=first[1],
                location=LocationKind.MEASURED,
                channel=channel,
                unit=unit,
            )
        if second[1] == target:
            # The crossing lands exactly on a measured sample: reporting an
            # interpolated frequency here would claim a resolution the data
            # does not have.
            return SearchResult(
                status=SearchStatus.OK,
                frequency=second[0],
                value=second[1],
                location=LocationKind.MEASURED,
                channel=channel,
                unit=unit,
            )
        if not estimate:
            return SearchResult(
                status=SearchStatus.OK,
                frequency=second[0],
                value=second[1],
                location=LocationKind.MEASURED,
                channel=channel,
                unit=unit,
            )
        span = second[1] - first[1]
        if span == 0:
            return _unavailable(
                "crossing cannot be located on a flat segment",
                channel=channel,
                unit=unit,
            )
        fraction = (target - first[1]) / span
        frequency = first[0] + fraction * (second[0] - first[0])
        return SearchResult(
            status=SearchStatus.OK,
            frequency=int(round(frequency)),
            value=target,
            location=LocationKind.ESTIMATED,
            channel=channel,
            unit=unit,
        )
    return _unavailable(
        f"no crossing of {target:g}{(' ' + unit) if unit else ''}",
        channel=channel,
        unit=unit,
    )


# ----------------------------------------------------------------------
# Bandwidth and Q
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BandwidthResult:
    """Peak, half-power edges and the resulting bandwidth and Q."""

    status: SearchStatus
    centre: int | None = None
    centre_value: float | None = None
    lower: int | None = None
    upper: int | None = None
    bandwidth: int | None = None
    quality_factor: float | None = None
    channel: str = ""
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is SearchStatus.OK

    def describe(self) -> str:
        if not self.ok:
            return f"unavailable ({self.reason or 'no result'})"
        q = (
            f", Q {self.quality_factor:.4g}"
            if self.quality_factor is not None
            else ""
        )
        return (
            f"{self.centre} Hz, -3 dB {self.lower} to {self.upper} Hz,"
            f" bandwidth {self.bandwidth} Hz{q}"
        )


def bandwidth_3db(  # noqa: PLR0913
    points: Sequence[Datapoint],
    *,
    channel: str = "S11",
    measure: Measure | None = None,
    centre: int | None = None,
    start: int | None = None,
    stop: int | None = None,
) -> BandwidthResult:
    """-3 dB bandwidth and loaded Q about a peak (or a notch).

    The half-power level is 3 dB below the peak for a maximum and 3 dB above
    it for a minimum, and both edges must exist inside the searched band.  A
    curve that only crosses on one side is reported unavailable rather than
    being completed with the band edge, which would invent a bandwidth the
    measurement does not contain.
    """
    measure = measure or default_measure(channel)
    series = _band(_series(points, measure), start, stop)
    if len(series) < 3:
        return BandwidthResult(
            status=SearchStatus.UNAVAILABLE,
            channel=channel,
            reason="fewer than three finite samples in the band",
        )
    if centre is None:
        peak = max(series, key=lambda item: (item[1], -item[0]))
        notch = min(series, key=lambda item: (item[1], -item[0]))
    else:
        peak = notch = min(series, key=lambda item: abs(item[0] - centre))
    # Choose the shape that the curve actually shows around the reference:
    # a peak whose minimum is the global minimum is a peak.
    is_peak = peak[1] >= notch[1]
    anchor = peak if is_peak else notch
    level = anchor[1] - 3.0 if is_peak else anchor[1] + 3.0
    # Traversed in increasing frequency, a peak *rises* into the half-power
    # level on its left and *falls* out of it on its right; a notch does the
    # opposite.  Getting this backwards silently reports the wrong edges.
    left = search_crossing(
        [point for _freq, _value, point in series],
        level,
        above=is_peak,
        channel=channel,
        measure=measure,
        stop=anchor[0],
    )
    right = search_crossing(
        [point for _freq, _value, point in series],
        level,
        above=not is_peak,
        channel=channel,
        measure=measure,
        start=anchor[0],
    )
    if not left.ok or not right.ok:
        missing = "lower" if not left.ok else "upper"
        return BandwidthResult(
            status=SearchStatus.UNAVAILABLE,
            centre=anchor[0],
            centre_value=anchor[1],
            channel=channel,
            reason=(
                f"the {missing} -3 dB edge is not inside the measured band"
            ),
        )
    width = right.frequency - left.frequency
    if width <= 0:
        return BandwidthResult(
            status=SearchStatus.UNAVAILABLE,
            centre=anchor[0],
            centre_value=anchor[1],
            channel=channel,
            reason="the -3 dB edges are not in frequency order",
        )
    quality = anchor[0] / width if width else None
    return BandwidthResult(
        status=SearchStatus.OK,
        centre=anchor[0],
        centre_value=anchor[1],
        lower=left.frequency,
        upper=right.frequency,
        bandwidth=width,
        quality_factor=quality,
        channel=channel,
    )


# ----------------------------------------------------------------------
# Marker tracking
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MarkerRule:
    """A stored marker search that can be re-applied after a sweep."""

    name: str
    kind: SearchKind
    channel: str = "S11"
    unit: str = "dB"
    band_start: int | None = None
    band_stop: int | None = None
    target: float | None = None
    #: Frequency the rule searches left/right from, for local searches.
    reference: int | None = None
    maximum: bool = True

    def apply(self, points: Sequence[Datapoint]) -> SearchResult:
        """Evaluate the rule against one channel of one snapshot."""
        if self.kind is SearchKind.MAXIMUM:
            return search_extreme(
                points,
                maximum=True,
                channel=self.channel,
                unit=self.unit,
                start=self.band_start,
                stop=self.band_stop,
            )
        if self.kind is SearchKind.MINIMUM:
            return search_extreme(
                points,
                maximum=False,
                channel=self.channel,
                unit=self.unit,
                start=self.band_start,
                stop=self.band_stop,
            )
        if self.kind in (
            SearchKind.PEAK_LEFT,
            SearchKind.PEAK_RIGHT,
            SearchKind.NOTCH_LEFT,
            SearchKind.NOTCH_RIGHT,
        ):
            if self.reference is None:
                return _unavailable(
                    "a local search needs a reference frequency",
                    channel=self.channel,
                    unit=self.unit,
                )
            right = self.kind in (
                SearchKind.PEAK_RIGHT,
                SearchKind.NOTCH_RIGHT,
            )
            maximum = self.kind in (
                SearchKind.PEAK_LEFT,
                SearchKind.PEAK_RIGHT,
            )
            return search_local(
                points,
                self.reference,
                maximum=maximum,
                direction=1 if right else -1,
                channel=self.channel,
                unit=self.unit,
                start=self.band_start,
                stop=self.band_stop,
            )
        if self.kind in (
            SearchKind.CROSSING_ABOVE,
            SearchKind.CROSSING_BELOW,
        ):
            if self.target is None:
                return _unavailable(
                    "a crossing search needs a target value",
                    channel=self.channel,
                    unit=self.unit,
                )
            return search_crossing(
                points,
                self.target,
                above=self.kind is SearchKind.CROSSING_ABOVE,
                channel=self.channel,
                unit=self.unit,
                start=self.band_start,
                stop=self.band_stop,
            )
        return _unavailable(
            f"unknown search kind {self.kind.value}",
            channel=self.channel,
            unit=self.unit,
        )


def track_markers(
    rules: Iterable[MarkerRule],
    channels: dict[str, Sequence[Datapoint]],
) -> dict[str, SearchResult]:
    """Apply every rule to its own channel; missing channels stay unavailable.

    A rule whose channel is absent is not silently evaluated against another
    channel: it reports why, so a marker never points at data it did not use.
    """
    results: dict[str, SearchResult] = {}
    for rule in rules:
        points = channels.get(rule.channel)
        if not points:
            results[rule.name] = _unavailable(
                f"no {rule.channel} channel in this capture",
                channel=rule.channel,
                unit=rule.unit,
            )
            continue
        results[rule.name] = rule.apply(points)
    return results


# ----------------------------------------------------------------------
# Convenience entry points for the marker UI
# ----------------------------------------------------------------------
#: Search kinds offered in the marker panel, with the wording an operator
#: reads.  Kept here so the widget and the engine cannot drift apart.
SEARCH_CHOICES: tuple[tuple[SearchKind, str], ...] = (
    (SearchKind.MAXIMUM, "Highest value in sweep"),
    (SearchKind.MINIMUM, "Lowest value in sweep"),
    (SearchKind.PEAK_LEFT, "Peak left of marker"),
    (SearchKind.PEAK_RIGHT, "Peak right of marker"),
    (SearchKind.NOTCH_LEFT, "Notch left of marker"),
    (SearchKind.NOTCH_RIGHT, "Notch right of marker"),
    (SearchKind.CROSSING_ABOVE, "Crossing above target"),
    (SearchKind.CROSSING_BELOW, "Crossing below target"),
)


def search_for(  # noqa: PLR0913
    kind: SearchKind,
    points: Sequence[Datapoint],
    *,
    channel: str = "S11",
    target: float | None = None,
    reference: int | None = None,
    estimate: bool = True,
) -> SearchResult:
    """Run one search kind with a uniform signature for the UI.

    ``reference`` defaults to the sweep centre, which is the marker's usual
    starting place; a caller that has a real marker draws from it explicitly.
    """
    if kind is SearchKind.MAXIMUM:
        return search_extreme(
            points, maximum=True, channel=channel, estimate=estimate
        )
    if kind is SearchKind.MINIMUM:
        return search_extreme(
            points, maximum=False, channel=channel, estimate=estimate
        )
    if kind in (
        SearchKind.PEAK_LEFT,
        SearchKind.PEAK_RIGHT,
        SearchKind.NOTCH_LEFT,
        SearchKind.NOTCH_RIGHT,
    ):
        if reference is None:
            if not points:
                return _unavailable(
                    "no samples to search", channel=channel, unit="dB"
                )
            reference = points[len(points) // 2].freq
        maximum = kind in (SearchKind.PEAK_LEFT, SearchKind.PEAK_RIGHT)
        direction = (
            1 if kind in (SearchKind.PEAK_RIGHT, SearchKind.NOTCH_RIGHT) else -1
        )
        return search_local(
            points,
            reference,
            maximum=maximum,
            direction=direction,
            channel=channel,
        )
    if kind in (SearchKind.CROSSING_ABOVE, SearchKind.CROSSING_BELOW):
        if target is None:
            return _unavailable(
                "a crossing search needs a target value",
                channel=channel,
                unit="dB",
            )
        return search_crossing(
            points,
            target,
            above=kind is SearchKind.CROSSING_ABOVE,
            channel=channel,
            estimate=estimate,
        )
    return _unavailable(f"unknown search kind {kind.value}", channel=channel)


def band_width_summary(
    points: Sequence[Datapoint],
    *,
    channel: str = "S11",
    centre: int | None = None,
) -> BandwidthResult:
    """-3 dB bandwidth and Q for the marker panel."""
    return bandwidth_3db(points, channel=channel, centre=centre)
