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
"""Measurement reports (REPORT-01, REPORT-02).

A report is built from a *frozen* snapshot of the experiment: traces plus
their structured analysis results.  Nothing is read from live widgets or a
running sweep, so a report can never mix two different measurements.

Output is a self-contained HTML document (inline CSS, inline SVG plots) plus
an optional CSV summary.  PDF generation is deliberately not implemented:
the existing Qt stack would need `QWebEngine` or `QPrinter` and neither is a
declared dependency, so that remains a documented limitation rather than a
half-working feature.
"""

from __future__ import annotations

import csv
import html
import io
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Sequence

from .Analysis.Results import AnalysisResult
from .CalibrationCoverage import CoverageResult, CoverageState
from .Limits import LimitEvaluation, LimitSet, Metric, Verdict
from .Measurement import (
    S11,
    S21,
    CorrectionSource,
    MeasurementSnapshot,
)
from .RFTools import Datapoint, groupDelay
from .Traces import Trace

REPORT_FORMAT_VERSION = 1


class CorrectionKind(Enum):
    """What kind of correction produced the reported samples.

    Derived from the full recorded facts, never from a single missing field:
    an empty identity does not prove that no calibration was involved (older
    or device-supplied corrections carry no application identity).
    """

    NONE = "none"
    """No correction was applied; the samples are raw."""

    CALIBRATED = "calibrated"
    """A calibration correction was applied and is identified."""

    OFFSET_DELAY_ONLY = "offset_delay_only"
    """The application applied an offset delay and nothing else."""

    UNIDENTIFIED = "unidentified"
    """A correction was applied but no calibration identity is recorded."""


def correction_kind(snapshot: MeasurementSnapshot) -> CorrectionKind:
    """Classify the correction recorded on ``snapshot``.

    ``OFFSET_DELAY_ONLY`` is claimed only when every fact says so: the
    correction was applied, the source is the application, a nonzero finite
    offset delay was applied, and no calibration identity is recorded.  A
    device- or unknown-sourced correction, or one with no recorded delay,
    stays ``UNIDENTIFIED`` — it is still reported as a correction, just not
    as one this application performed from a delay alone.
    """
    correction = snapshot.correction
    if not correction.applied:
        return CorrectionKind.NONE
    if correction.identity:
        return CorrectionKind.CALIBRATED
    delay = correction.offset_delay
    if (
        correction.source is CorrectionSource.APPLICATION
        and isinstance(delay, (int, float))
        and math.isfinite(delay)
        and delay != 0.0
    ):
        return CorrectionKind.OFFSET_DELAY_ONLY
    return CorrectionKind.UNIDENTIFIED


@dataclass(frozen=True)
class ReportTrace:
    """One trace as frozen for a report."""

    trace_id: str
    name: str
    color: str
    snapshot: MeasurementSnapshot
    analyses: tuple[AnalysisResult, ...] = ()
    coverage: CoverageResult | None = None
    limits: tuple[LimitEvaluation, ...] = ()

    @property
    def available(self) -> tuple[str, ...]:
        return self.snapshot.available

    @property
    def simulated(self) -> bool:
        return self.snapshot.simulated

    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        analyses: Iterable[AnalysisResult] = (),
        coverage: CoverageResult | None = None,
        limits: Iterable[LimitEvaluation] = (),
    ) -> "ReportTrace":
        return cls(
            trace_id=trace.id,
            name=trace.name,
            color=trace.color,
            snapshot=trace.snapshot,
            analyses=tuple(analyses),
            coverage=coverage,
            limits=tuple(limits),
        )


@dataclass(frozen=True)
class ReportModel:
    """Complete, immutable description of a report."""

    title: str
    dut: str
    notes: str
    generated: datetime
    traces: tuple[ReportTrace, ...]
    app_version: str = ""
    device: str = ""
    measurement_time: str = ""
    format_version: int = REPORT_FORMAT_VERSION
    #: The limit set frozen into this report, or ``None`` when the report
    #: was built without limit checking.  The complete evaluation lives in
    #: each :class:`ReportTrace`, so a report never re-reads live settings.
    limit_set: LimitSet | None = None

    @property
    def verdict(self) -> Verdict | None:
        """Worst verdict over every frozen limit check."""
        results = [item for trace in self.traces for item in trace.limits]
        if not results:
            return None
        if any(item.verdict is Verdict.FAIL for item in results):
            return Verdict.FAIL
        if any(item.verdict is Verdict.UNAVAILABLE for item in results):
            return Verdict.UNAVAILABLE
        return Verdict.PASS

    @property
    def simulated(self) -> bool:
        return any(trace.simulated for trace in self.traces)

    @property
    def multi_trace(self) -> bool:
        return len(self.traces) > 1


def build_report(  # noqa: PLR0913
    title: str,
    traces: Sequence[Trace],
    *,
    dut: str = "",
    notes: str = "",
    analyses: dict[str, Iterable[AnalysisResult]] | None = None,
    coverage: dict[str, CoverageResult] | None = None,
    app_version: str = "",
    generated: datetime | None = None,
    limits: LimitSet | None = None,
) -> ReportModel:
    """Freeze ``traces`` and their results into a report model.

    ``limits`` is evaluated exactly once, here: the returned model holds
    both the limit set and its results, so later changes to the active limit
    set or to the measured data cannot alter an already built report.
    """
    analyses = analyses or {}
    coverage = coverage or {}
    report_traces = tuple(
        ReportTrace.from_trace(
            trace,
            analyses=analyses.get(trace.id, ()),
            coverage=coverage.get(trace.id),
            limits=(
                limits.evaluate(
                    trace.snapshot,
                    trace_id=trace.id,
                    trace_name=trace.name,
                )
                if limits is not None
                else ()
            ),
        )
        for trace in traces
    )
    measured = [
        trace.snapshot.timestamp for trace in report_traces if trace.available
    ]
    device = ""
    if report_traces:
        device = report_traces[0].snapshot.provenance.device
    return ReportModel(
        title=title or "Measurement report",
        dut=dut,
        notes=notes,
        generated=generated or datetime.now(timezone.utc),
        traces=report_traces,
        app_version=app_version,
        device=device,
        limit_set=limits,
        measurement_time=(
            max(measured).astimezone(timezone.utc).isoformat()
            if measured
            else ""
        ),
    )


# ----------------------------------------------------------------------
# Provenance labels (REPORT-01.4)
# ----------------------------------------------------------------------
def provenance_labels(trace: ReportTrace) -> list[str]:
    """Labels describing everything a reader must know about the data."""
    labels: list[str] = []
    snapshot = trace.snapshot
    if snapshot.simulated:
        scenario = snapshot.provenance.simulator_scenario or "unspecified"
        labels.append(f"SIMULATED ({scenario})")
    completion = snapshot.provenance.completion.value
    if completion != "complete":
        labels.append(f"{completion.upper()} data")
    calibration = snapshot.provenance.calibration
    if calibration is None or not calibration.available:
        labels.append("UNCALIBRATED")
    if trace.coverage is not None:
        state = trace.coverage.state
        if state is CoverageState.OUTSIDE:
            labels.append("OUTSIDE calibration span")
        elif state is CoverageState.PARTIAL:
            labels.append("PARTIALLY calibrated")
        elif state is CoverageState.INVALID:
            labels.append("calibration INVALID")
        # Edge holding is a property of the span, not a CoverageState: the
        # result reports how many samples fall outside the calibrated range
        # and are corrected from the nearest endpoint.
        if trace.coverage.edge_held:
            labels.append("correction held at calibration edge")
    if S21 not in snapshot.available:
        labels.append("one-port data (no S21)")
    kind = correction_kind(snapshot)
    if kind is CorrectionKind.OFFSET_DELAY_ONLY:
        labels.append("offset delay correction (no calibration)")
    elif kind is not CorrectionKind.NONE:
        # A calibration or an unidentified correction: name the source but
        # do not claim a delay that was not recorded.
        labels.append(f"corrected ({snapshot.correction.source.value})")
    return labels


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------
#: One plotted point: label, colour, (x, y) samples.
PlotSeries = Sequence[tuple[str, str, Sequence[tuple[float, float]]]]


def _plot_axes(
    series: PlotSeries,
) -> tuple[float, float, float, float] | None:
    """Common finite ``(x_min, x_max, y_min, y_max)`` for all series.

    Every series is drawn against the same axes, so a measured trace and a
    limit segment are only comparable when they share that domain.  Returns
    ``None`` when no series carries a finite sample.
    """
    xs: list[float] = []
    ys: list[float] = []
    for _label, _color, points in series:
        for x, y in points:
            if math.isfinite(x) and math.isfinite(y):
                xs.append(float(x))
                ys.append(float(y))
    if not xs:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def _line_path(
    points: Sequence[tuple[float, float]],
    axes: tuple[float, float, float, float],
    width: int,
    height: int,
) -> str:
    """Path through ``points`` mapped onto the shared ``axes``."""
    x_min, x_max, y_min, y_max = axes
    x_span = x_max - x_min
    y_span = y_max - y_min
    commands = []
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        # A degenerate span (a constant series, or a single frequency) is
        # drawn through the middle of the box rather than pinned to an edge,
        # which would suggest an extreme value that the data does not have.
        tx = (x - x_min) / x_span if x_span else 0.5
        ty = (y - y_min) / y_span if y_span else 0.5
        px = tx * (width - 2) + 1
        py = height - 1 - ty * (height - 2)
        commands.append(f"{'M' if not commands else 'L'}{px:.1f},{py:.1f}")
    if len(commands) < 2:
        return ""
    return " ".join(commands)


def svg_plot(  # noqa: PLR0913
    series: PlotSeries,
    *,
    title: str,
    x_label: str = "Frequency (Hz)",
    y_label: str = "",
    width: int = 520,
    height: int = 220,
) -> str:
    """Self-contained SVG with a print-friendly light theme and legend.

    All series share one pair of axes derived from their combined finite
    range, so overlaid traces and limit bounds are spatially comparable.
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height + 46}" viewBox="0 0 {width} {height + 46}" '
        f'role="img" aria-label="{html.escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height + 46}" '
        'fill="#ffffff"/>',
        f'<text x="8" y="16" font-family="sans-serif" font-size="12" '
        f'font-weight="600" fill="#111111">{html.escape(title)}</text>',
        f'<rect x="1" y="24" width="{width - 2}" height="{height}" '
        'fill="none" stroke="#cccccc"/>',
    ]
    axes = _plot_axes(series)
    if axes is not None:
        for _label, color, points in series:
            path = _line_path(points, axes, width - 2, height)
            if not path:
                continue
            parts.append(
                f'<path d="{path}" transform="translate(1,24)" fill="none" '
                f'stroke="{color}" stroke-width="1.5"/>'
            )
    legend_x = 8
    for label, color, points in series:
        if not points:
            continue
        parts.append(
            f'<rect x="{legend_x}" y="{height + 30}" width="10" '
            f'height="10" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 14}" y="{height + 39}" '
            'font-family="sans-serif" font-size="11" fill="#111111">'
            f"{html.escape(label)}</text>"
        )
        legend_x += 18 + 7 * len(label)
    parts.append(
        f'<text x="{width - 8}" y="{height + 39}" text-anchor="end" '
        'font-family="sans-serif" font-size="10" fill="#555555">'
        f"{html.escape(x_label)}"
        + (f" / {html.escape(y_label)}" if y_label else "")
        + "</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _db(points: Sequence[Datapoint]) -> list[tuple[float, float]]:
    out = []
    for dp in points:
        magnitude = abs(dp.z)
        if magnitude > 0:
            out.append((float(dp.freq), 20.0 * math.log10(magnitude)))
    return out


def _phase(points: Sequence[Datapoint]) -> list[tuple[float, float]]:
    return [
        (float(dp.freq), math.degrees(math.atan2(dp.im, dp.re)))
        for dp in points
    ]


def _delay(points: Sequence[Datapoint]) -> list[tuple[float, float]]:
    out = []
    for index in range(1, len(points) - 1):
        delay = groupDelay(list(points), index)
        if math.isfinite(delay):
            out.append((float(points[index].freq), delay * 1e9))
    return out


LIMIT_COLOR = "#777777"


def _metric_points(
    points: Sequence[Datapoint], metric: Metric
) -> list[tuple[float, float]]:
    """Measured values in ``metric`` units, skipping non-finite samples."""
    out = []
    for dp in points:
        value = metric.measure(dp)
        if math.isfinite(value):
            out.append((float(dp.freq), value))
    return out


def _bound_points(limit_set: LimitSet, metric: Metric) -> list[tuple]:
    """(label, colour, points) step lines for every bound of ``metric``."""
    series: list[tuple] = []
    for rule in limit_set.rules:
        if rule.metric is not metric:
            continue
        for segment in rule.segments:
            for kind, bound in (
                ("maximum", segment.maximum),
                ("minimum", segment.minimum),
            ):
                if bound is None:
                    continue
                series.append(
                    (
                        f"{rule.id} {kind}",
                        LIMIT_COLOR,
                        [
                            (float(segment.start), float(bound)),
                            (float(segment.stop), float(bound)),
                        ],
                    )
                )
    return series


def limit_plots(model: ReportModel) -> list[tuple[str, str]]:
    """One overlay per limited metric, measurement against its bounds."""
    if model.limit_set is None:
        return []
    metrics = list(dict.fromkeys(rule.metric for rule in model.limit_set.rules))
    plots: list[tuple[str, str]] = []
    for metric in metrics:
        series = [
            (
                trace.name,
                trace.color,
                _metric_points(
                    trace.snapshot.channel(metric.parameter) or (), metric
                ),
            )
            for trace in model.traces
        ]
        series = [item for item in series if item[2]]
        series.extend(_bound_points(model.limit_set, metric))
        if not series:
            continue
        title = f"{metric.description} with limits"
        plots.append(
            (title, svg_plot(series, title=title, y_label=metric.unit))
        )
    return plots


def report_plots(model: ReportModel) -> list[tuple[str, str]]:
    """(title, svg) pairs for the plots included in a report."""
    plots: list[tuple[str, str]] = []

    reflection = [
        (
            f"{trace.name} S11",
            trace.color,
            _db(trace.snapshot.channel(S11) or ()),
        )
        for trace in model.traces
    ]
    reflection = [item for item in reflection if item[2]]
    if reflection:
        plots.append(
            (
                "S11 magnitude",
                svg_plot(reflection, title="S11 magnitude", y_label="dB"),
            )
        )

    transmission = [
        (
            f"{trace.name} S21",
            trace.color,
            _db(trace.snapshot.channel(S21) or ()),
        )
        for trace in model.traces
    ]
    transmission = [item for item in transmission if item[2]]
    if transmission:
        plots.append(
            (
                "S21 magnitude",
                svg_plot(transmission, title="S21 magnitude", y_label="dB"),
            )
        )
        phases = [
            (
                f"{trace.name} S21",
                trace.color,
                _phase(trace.snapshot.channel(S21) or ()),
            )
            for trace in model.traces
            if trace.snapshot.channel(S21)
        ]
        plots.append(
            (
                "S21 phase",
                svg_plot(
                    phases, title="S21 phase (wrapped)", y_label="degrees"
                ),
            )
        )
        delays = [
            (
                f"{trace.name} delay",
                trace.color,
                _delay(trace.snapshot.channel(S21) or ()),
            )
            for trace in model.traces
            if trace.snapshot.channel(S21)
        ]
        delays = [item for item in delays if item[2]]
        if delays:
            plots.append(
                (
                    "Group delay",
                    svg_plot(delays, title="Group delay", y_label="ns"),
                )
            )
    plots.extend(limit_plots(model))
    return plots


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------
CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif;
       color: #111; background: #fff; margin: 24px; line-height: 1.45; }
h1 { font-size: 20px; margin: 0 0 4px 0; }
h2 { font-size: 15px; margin: 24px 0 8px 0; border-bottom: 1px solid #ddd;
     padding-bottom: 3px; }
h3 { font-size: 13px; margin: 16px 0 6px 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 12px 0;
        font-size: 12px; }
th, td { border: 1px solid #ddd; padding: 4px 6px; text-align: left;
         vertical-align: top; }
th { background: #f4f4f4; }
.meta { color: #555; font-size: 12px; }
.badge { display: inline-block; border: 1px solid #999; border-radius: 3px;
         padding: 0 4px; margin-right: 4px; font-size: 11px; }
.unavailable { color: #8a6d3b; }
.note { white-space: pre-wrap; font-size: 12px; }
.plot { margin: 10px 0 18px 0; page-break-inside: avoid; }
.swatch { display: inline-block; width: 10px; height: 10px;
          margin-right: 5px; border: 1px solid #999; }
.verdict { display: inline-block; border: 1px solid #999; border-radius: 3px;
           padding: 0 5px; font-size: 11px; font-weight: 600; }
.verdict-pass { border-color: #1b6e2f; color: #1b6e2f; }
.verdict-fail { border-color: #b00020; color: #b00020; }
.verdict-unavailable { border-color: #8a6d3b; color: #8a6d3b; }
.limits-note { color: #555; font-size: 11px; margin: 4px 0 10px 0; }
"""


def _badges(labels: Sequence[str]) -> str:
    if not labels:
        return ""
    return "".join(
        f'<span class="badge">{html.escape(label)}</span>' for label in labels
    )


def _trace_table(trace: ReportTrace) -> str:
    rows = []
    for parameter in trace.available:
        points = trace.snapshot.channel(parameter) or ()
        if not points:
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(parameter)}</td>"
            f"<td>{len(points)}</td>"
            f"<td>{points[0].freq}</td>"
            f"<td>{points[-1].freq}</td>"
            "</tr>"
        )
    if not rows:
        rows.append(
            '<tr><td colspan="4" class="unavailable">'
            "No channels available</td></tr>"
        )
    return (
        "<table><tr><th>Channel</th><th>Points</th>"
        "<th>Start (Hz)</th><th>Stop (Hz)</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _analysis_table(trace: ReportTrace) -> str:
    rows = []
    for result in trace.analyses:
        if not result.available:
            rows.append(
                "<tr>"
                f"<td>{html.escape(result.parameter)}</td>"
                f"<td>{html.escape(result.kind.value)}</td>"
                '<td colspan="2" class="unavailable">'
                f"{html.escape(result.reason or 'unavailable')}</td>"
                "</tr>"
            )
            continue
        for value in result.values:
            if not value.available:
                continue
            frequency = value.frequency if value.frequency else ""
            rows.append(
                "<tr>"
                f"<td>{html.escape(result.parameter)}</td>"
                f"<td>{html.escape(result.kind.value)}</td>"
                f"<td>{html.escape(value.label)}: "
                f"{html.escape(_format_value(value))}</td>"
                f"<td>{frequency}</td>"
                "</tr>"
            )
    if not rows:
        return ""
    return (
        "<table><tr><th>Channel</th><th>Analysis</th><th>Result</th>"
        "<th>Frequency (Hz)</th></tr>" + "".join(rows) + "</table>"
    )


def verdict_badge(verdict: Verdict) -> str:
    return (
        f'<span class="verdict verdict-{verdict.value}">'
        f"{html.escape(verdict.label)}</span>"
    )


def _limit_value_text(metric: Metric, value: float | None) -> str:
    if value is None:
        return ""
    if not math.isfinite(value):
        return metric.format_value(value)
    return f"{value:.6g}" + (f" {metric.unit}" if metric.unit else "")


def _limit_table(trace: ReportTrace) -> str:
    if not trace.limits:
        return ""
    rows = []
    for result in trace.limits:
        band = (
            f"{result.band_start / 1e6:.6g} - {result.band_stop / 1e6:.6g} MHz"
        )
        worst = _limit_value_text(result.metric, result.worst_value)
        frequency = (
            ""
            if result.worst_frequency is None
            else str(result.worst_frequency)
        )
        limit = ""
        if result.worst_limit is not None:
            limit = (
                f"{result.worst_kind} "
                f"{_limit_value_text(result.metric, result.worst_limit)}"
            )
        detail = result.reason if result.verdict is Verdict.UNAVAILABLE else ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(result.title)}</td>"
            f"<td>{html.escape(result.metric.description)}</td>"
            f"<td>{html.escape(band)}</td>"
            f"<td>{verdict_badge(result.verdict)}</td>"
            f"<td>{html.escape(worst)}</td>"
            f"<td>{html.escape(frequency)}</td>"
            f"<td>{html.escape(limit)}</td>"
            f"<td>{html.escape(detail)}</td>"
            "</tr>"
        )
    return (
        "<table><tr><th>Limit</th><th>Metric</th><th>Band</th>"
        "<th>Result</th><th>Worst value</th><th>Frequency (Hz)</th>"
        "<th>Bound at worst value</th><th>Reason</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _format_value(value) -> str:
    if not value.available:
        return "unavailable"
    if value.unit == "Hz":
        return f"{value.value / 1e6:.6g} MHz"
    if value.unit:
        return f"{value.value:.6g} {value.unit}"
    return f"{value.value:.6g}"


def render_html(model: ReportModel, *, include_plots: bool = True) -> str:
    """Render a complete, self-contained HTML document."""
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{html.escape(model.title)}</title>",
        f"<style>{CSS}</style>",
        "</head><body>",
        f"<h1>{html.escape(model.title)}</h1>",
        '<p class="meta">',
    ]
    if model.dut:
        parts.append(f"Device under test: {html.escape(model.dut)}<br>")
    generated = model.generated.astimezone(timezone.utc).isoformat()
    parts.append(f"Generated: {html.escape(generated)}<br>")
    if model.measurement_time:
        parts.append(f"Measured: {html.escape(model.measurement_time)}<br>")
    if model.device:
        parts.append(f"Device: {html.escape(model.device)}<br>")
    if model.app_version:
        parts.append(
            f"Application: NanoVNA-Saver {html.escape(model.app_version)}<br>"
        )
    parts.append(
        f"Traces: {len(model.traces)}"
        + (" (simulated data)" if model.simulated else "")
    )
    if model.limit_set is not None:
        parts.append(
            f"<br>Limit set: {html.escape(model.limit_set.name)}"
            f" (version {model.limit_set.version})"
        )
    parts.append("</p>")

    verdict = model.verdict
    if verdict is not None and model.limit_set is not None:
        parts.append("<h2>Limit check</h2>")
        parts.append(
            f'<p class="meta">Overall result: {verdict_badge(verdict)}'
            f" over {len(model.limit_set.rules)} rule(s) and"
            f" {len(model.traces)} trace(s).</p>"
        )
        parts.append(
            '<p class="limits-note">A PASS means every measured sample inside'
            " every required band satisfied its bounds. Bounds are inclusive"
            " and values are never interpolated or extrapolated."
            " UNAVAILABLE means the check could not be decided (missing"
            " channel, band not covered, incomplete data or an invalid"
            " sample) and is not a pass. Bands are given in whole hertz and"
            " include both ends; dB limits are positive for a match or for"
            " loss.</p>"
        )

    if model.notes:
        parts.append("<h2>Notes</h2>")
        parts.append(f'<div class="note">{html.escape(model.notes)}</div>')

    parts.append("<h2>Traces</h2>")
    for trace in model.traces:
        labels = provenance_labels(trace)
        parts.append(
            f'<h3><span class="swatch" style="background:'
            f'{html.escape(trace.color)}"></span>{html.escape(trace.name)}</h3>'
        )
        if labels:
            parts.append(f'<p class="meta">{_badges(labels)}</p>')
        provenance = trace.snapshot.provenance
        details = [
            f"Reference impedance: {trace.snapshot.reference_impedance:g} ohm",
            f"source: {html.escape(provenance.source)}",
        ]
        if provenance.app_version:
            details.append(f"app: {html.escape(provenance.app_version)}")
        if provenance.device:
            details.append(f"device: {html.escape(provenance.device)}")
        parts.append(f'<p class="meta">{" · ".join(details)}</p>')
        if provenance.notes:
            # Acquisition provenance without a structural field of its own,
            # such as the Touchstone file a replayed sweep came from.
            parts.append(
                f'<p class="meta">Provenance: '
                f"{html.escape(provenance.notes)}</p>"
            )
        if provenance.sweep is not None:
            sweep = provenance.sweep
            parts.append(
                '<p class="meta">Sweep: '
                f"{sweep.start} Hz to {sweep.end} Hz, "
                f"{sweep.points} points, {sweep.segments} segment(s), "
                f"mode {html.escape(sweep.mode)}, "
                f"{sweep.averages} average(s)</p>"
            )
        correction = trace.snapshot.correction
        kind = correction_kind(trace.snapshot)
        if kind is CorrectionKind.CALIBRATED:
            parts.append(
                '<p class="meta">Correction: applied from '
                f"{html.escape(correction.source.value)}, calibration "
                f"{html.escape(correction.identity[:12])}</p>"
            )
        elif kind is CorrectionKind.OFFSET_DELAY_ONLY:
            parts.append(
                '<p class="meta">Correction: offset delay only (no'
                f" calibration), {correction.offset_delay * 1e12:.6g} ps</p>"
            )
        elif kind is CorrectionKind.UNIDENTIFIED:
            parts.append(
                '<p class="meta">Correction: applied from '
                f"{html.escape(correction.source.value)},"
                " calibration identity not recorded</p>"
            )
        else:
            parts.append(
                '<p class="meta">Correction: not applied (raw data)</p>'
            )
        if trace.coverage is not None:
            parts.append(
                '<p class="meta">Calibration coverage: '
                f"{html.escape(trace.coverage.describe())}"
                f" (calibrated {trace.coverage.calibrated_start} Hz to"
                f" {trace.coverage.calibrated_end} Hz)</p>"
            )
        parts.append(_trace_table(trace))
        limit_table = _limit_table(trace)
        if limit_table:
            parts.append(limit_table)
        table = _analysis_table(trace)
        if table:
            parts.append(table)

    if include_plots:
        plots = report_plots(model)
        if plots:
            parts.append("<h2>Plots</h2>")
            for _title, svg in plots:
                parts.append(f'<div class="plot">{svg}</div>')

    parts.append(
        '<p class="meta">Interpretation: results describe the data as'
        " recorded. Calibration coverage indicates only where corrections"
        " were interpolated or held at a span edge; it says nothing about"
        " calibration age, connector condition or measurement uncertainty."
        " Reports built from forward-only two-port data are not a complete"
        " two-port characterisation.</p>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


def _csv_number(value: float | None) -> str:
    """CSV cell for a measured value.

    A missing value stays blank, but a genuinely infinite measurement is
    data: a VSWR of ``inf`` is a valid total reflection and must not be
    silently dropped.  The marker matches ``report_snapshot`` and the
    snapshot encoder (``inf`` / ``-inf``), so it is portable without being
    locale dependent.
    """
    if value is None:
        return ""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(value)


def csv_summary(model: ReportModel) -> str:
    """CSV of analysis summaries for further processing (REPORT-02.4)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "trace_id",
            "trace",
            "channel",
            "analysis",
            "status",
            "label",
            "value",
            "unit",
            "frequency_hz",
            "detail",
            "reason",
        ]
    )
    for trace in model.traces:
        for result in trace.analyses:
            if not result.available:
                writer.writerow(
                    [
                        trace.trace_id,
                        trace.name,
                        result.parameter,
                        result.kind.value,
                        "unavailable",
                        "",
                        "",
                        "",
                        "",
                        "",
                        result.reason,
                    ]
                )
                continue
            for value in result.values:
                writer.writerow(
                    [
                        trace.trace_id,
                        trace.name,
                        result.parameter,
                        result.kind.value,
                        "ok",
                        value.label,
                        "" if value.value is None else repr(value.value),
                        value.unit,
                        "" if value.frequency is None else value.frequency,
                        value.detail,
                        "",
                    ]
                )
        for result in trace.limits:
            writer.writerow(
                [
                    trace.trace_id,
                    trace.name,
                    result.metric.parameter,
                    f"limit:{result.rule_id}",
                    result.verdict.value,
                    (
                        f"{result.title}: worst"
                        if result.verdict is not Verdict.UNAVAILABLE
                        else result.title
                    ),
                    _csv_number(result.worst_value),
                    result.metric.unit,
                    (
                        ""
                        if result.worst_frequency is None
                        else result.worst_frequency
                    ),
                    (
                        f"{result.metric.description}; band "
                        f"{result.band_start}-{result.band_stop} Hz"
                        + (
                            f"; {result.worst_kind} "
                            f"{result.metric.format_value(result.worst_limit)}"
                            if result.verdict is Verdict.FAIL
                            and result.worst_limit is not None
                            else ""
                        )
                    ),
                    result.reason,
                ]
            )
    return buffer.getvalue()


def write_html(model: ReportModel, filename: str) -> None:
    with open(filename, "w", encoding="utf-8", newline="\n") as target:
        target.write(render_html(model))


def write_csv(model: ReportModel, filename: str) -> None:
    with open(filename, "w", encoding="utf-8", newline="") as target:
        target.write(csv_summary(model))


def report_snapshot(model: ReportModel) -> dict:
    """Plain-data view for embedding in sessions or tests."""
    return {
        "format_version": model.format_version,
        "title": model.title,
        "dut": model.dut,
        "notes": model.notes,
        "generated": model.generated.astimezone(timezone.utc).isoformat(),
        "app_version": model.app_version,
        "device": model.device,
        "measurement_time": model.measurement_time,
        "limit_set": (
            None if model.limit_set is None else model.limit_set.to_dict()
        ),
        "traces": [
            {
                "trace_id": trace.trace_id,
                "name": trace.name,
                "color": trace.color,
                "labels": provenance_labels(trace),
                "available": list(trace.available),
                "analyses": [result.to_dict() for result in trace.analyses],
                "limits": [result.to_dict() for result in trace.limits],
            }
            for trace in model.traces
        ],
    }
