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
"""Live chart overlays for limit bounds (NEXT-04).

The limits engine evaluates measured samples in the metric's own units
(VSWR ratio, dB, degrees).  A chart plots a *different* derived quantity, and
several charts plot the same raw channel in different units (|S11| versus
dB, R+jX versus |Z|).  Drawing a bound on the wrong chart would show a
verdict that the evaluator never made, so a bound is only drawn on the chart
families that plot the metric itself:

=====================  ========================================
metric                 chart family that plots it
=====================  ========================================
``s11_vswr``           ``VSWRChart`` (S11 VSWR)
``s11_return_loss_db`` ``LogMagChart`` in its inverted (return
                       loss, positive-is-good) mode
``s21_insertion_loss_db`` ``LogMagChart`` for S21, plotted as gain
                       ``-insertion_loss``
``s11_phase_deg``      ``PhaseChart`` for S11 (wrapped degrees)
=====================  ========================================

Everything else draws nothing rather than a plausible-looking line.  The
mapping is data, not code paths, so a future metric either declares its chart
here or stays undrawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .Limits import LIMIT_BOUND_COLOR, LimitSet, Metric

#: Chart key (as registered in the main window) per metric and channel.
VSWR_CHART = "vswr"
LOG_MAG_CHART = "log_mag"
PHASE_CHART = "phase"

#: Multiplier that converts a metric value into the chart's own Y units at
#: its default (non-inverted) setting.
#:
#: ``LogMagChart`` plots ``Datapoint.gain``, which is ``20*log10|S21|`` for
#: S21 (a loss reads negative) and ``20*log10|S11|`` for S11 (a *return loss*
#: reads negative).  ``Metric.S11_RETURN_LOSS_DB`` is the positive quantity, so
#: its factor is ``-1``.  ``Metric.S21_INSERTION_LOSS_DB`` is also positive,
#: so on the S21 gain chart its factor is ``-1`` as well.  ``isInverted``
#: flips the chart's own axis, and the factor follows it in
#: :func:`_chart_factor`.
_SIGN = {
    (Metric.S11_VSWR, "s11", VSWR_CHART): 1.0,
    (Metric.S11_RETURN_LOSS_DB, "s11", LOG_MAG_CHART): -1.0,
    (Metric.S21_INSERTION_LOSS_DB, "s21", LOG_MAG_CHART): -1.0,
    (Metric.S11_PHASE_DEG, "s11", PHASE_CHART): 1.0,
}


@dataclass(frozen=True)
class ChartLimitSeries:
    """One chart's limit overlay: ``(label, color, [(hz, value)])``."""

    channel: str
    chart_key: str
    series: tuple

    @property
    def key(self) -> tuple[str, str]:
        return (self.channel, self.chart_key)


def sign_for(metric: Metric, channel: str, chart_key: str) -> float | None:
    """Conversion factor for a metric on a chart, or ``None`` if unsupported."""
    return _SIGN.get((metric, channel, chart_key))


def _chart_factor(chart, base: float) -> float:
    """``base`` adjusted for a chart that flips its own axis.

    ``LogMagChart`` can invert its axis so an S11 channel reads as a positive
    return loss instead of a negative reflection coefficient; a bound drawn
    without following that flip would sit on the wrong side of the trace.
    """
    if getattr(chart, "isInverted", False):
        return -base
    return base


def series_for(
    limit_set: LimitSet | None,
    channel: str,
    chart_key: str,
    chart=None,
) -> list[tuple[str, str, list[tuple[float, float]]]]:
    """Bound series for one chart, in that chart's units.

    A segment with a slope contributes its two endpoint values; a flat bound
    contributes the same value twice.  Both describe exactly the line the
    evaluator compares samples against, so the drawn bound and the reported
    verdict cannot disagree.
    """
    if limit_set is None:
        return []
    series: list[tuple[str, str, list[tuple[float, float]]]] = []
    for rule in limit_set.rules:
        base = sign_for(rule.metric, channel, chart_key)
        if base is None:
            continue
        factor = _chart_factor(chart, base) if chart is not None else base
        for segment in rule.segments:
            for kind, bound_at in (
                ("maximum", segment.maximum_at),
                ("minimum", segment.minimum_at),
            ):
                low = bound_at(segment.start)
                if low is None:
                    continue
                high = bound_at(segment.stop)
                series.append(
                    (
                        f"{rule.label or rule.id} {kind}",
                        LIMIT_BOUND_COLOR,
                        [
                            (float(segment.start), float(low) * factor),
                            (float(segment.stop), float(high) * factor),
                        ],
                    )
                )
    return series


def chart_series(
    limit_set: LimitSet | None, charts: dict
) -> dict[tuple[str, str], list]:
    """Every overlay a chart catalogue needs, keyed by ``(channel, key)``.

    ``charts`` is the main window's chart catalogue
    (``{"s11": {...}, "s21": {...}}``); a missing key is skipped, so a
    reduced catalogue simply draws fewer overlays.
    """
    overlays: dict[tuple[str, str], list] = {}
    for channel, catalogue in charts.items():
        for chart_key, chart in catalogue.items():
            series = series_for(limit_set, channel, chart_key, chart)
            if series:
                overlays[(channel, chart_key)] = series
    return overlays


def apply_chart_limits(
    charts: dict, limit_set: LimitSet | None
) -> dict[tuple[str, str], int]:
    """Push bounds into every compatible chart; return counts per chart."""
    overlays = chart_series(limit_set, charts)
    applied: dict[tuple[str, str], int] = {}
    for (channel, chart_key), series in overlays.items():
        chart = charts[channel][chart_key]
        chart.setLimits(series)
        applied[(channel, chart_key)] = len(series)
    # A chart that no longer has an applicable bound must stop drawing the
    # previous one: a stale limit line would describe a specification that is
    # no longer in force.
    for channel, catalogue in charts.items():
        for chart_key, chart in catalogue.items():
            if (channel, chart_key) not in overlays and hasattr(
                chart, "clearLimits"
            ):
                chart.clearLimits()
    return applied


def described_bounds(
    limit_set: LimitSet | None, channel: str, chart_key: str, chart=None
) -> Sequence[str]:
    """Human readable list of the bounds shown on one chart."""
    return [
        f"{label}: {points[0][1]:.6g} to {points[-1][1]:.6g}"
        for label, _color, points in series_for(
            limit_set, channel, chart_key, chart
        )
    ]
