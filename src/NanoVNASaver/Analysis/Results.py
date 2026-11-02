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
"""Structured analysis results (PRESET-01, REPORT-01).

Existing analysis windows compute their numbers while writing widget text.
That makes the values unavailable to reports, presets and tests.  This module
re-runs the *same* algorithms from :mod:`AnalyticTools` against an immutable
:class:`MeasurementSnapshot` and returns typed results.

Comparison with the UI is intentional: :class:`VSWRResult` uses the same
``At.minima``/``At.take_from_idx`` calls as
:class:`NanoVNASaver.Analysis.VSWRAnalysis`, and
:class:`ResonanceResult` uses the same ``At.zero_crossings`` scan as
:class:`NanoVNASaver.Analysis.ResonanceAnalysis`.

Unavailable results carry a reason instead of a zero value.
"""

from __future__ import annotations

import itertools as it
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from .. import AnalyticTools as At
from ..Measurement import S11, S21, MeasurementSnapshot
from ..RFTools import Datapoint

logger = logging.getLogger(__name__)


class AnalysisStatus(Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"


class AnalysisKind(Enum):
    RESONANCE = "resonance"
    VSWR = "vswr"
    INSERTION_LOSS = "insertion_loss"
    GROUP_DELAY = "group_delay"
    CUTOFF = "cutoff"
    CABLE = "cable"


@dataclass(frozen=True)
class AnalysisValue:
    """A single measured quantity with units and provenance."""

    label: str
    value: float | None
    unit: str = ""
    frequency: int | None = None
    detail: str = ""

    @property
    def available(self) -> bool:
        return self.value is not None and not (
            isinstance(self.value, float) and math.isnan(self.value)
        )

    def formatted(self) -> str:
        if not self.available:
            return f"{self.label}: unavailable"
        if self.unit == "Hz":
            return f"{self.label}: {self.value / 1e6:.6g} MHz"
        if self.unit:
            return f"{self.label}: {self.value:.6g} {self.unit}"
        return f"{self.label}: {self.value:.6g}"


@dataclass(frozen=True)
class AnalysisResult:
    """One analysis outcome against one trace."""

    kind: AnalysisKind
    trace_id: str
    trace_name: str
    parameter: str
    status: AnalysisStatus = AnalysisStatus.OK
    values: tuple[AnalysisValue, ...] = ()
    parameters: dict = field(default_factory=dict)
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.status is AnalysisStatus.OK and bool(self.values)

    def value(self, label: str) -> AnalysisValue | None:
        return next((item for item in self.values if item.label == label), None)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "trace_id": self.trace_id,
            "trace_name": self.trace_name,
            "parameter": self.parameter,
            "status": self.status.value,
            "reason": self.reason,
            "parameters": dict(self.parameters),
            "values": [
                {
                    "label": item.label,
                    "value": _encode(item.value),
                    "unit": item.unit,
                    "frequency": item.frequency,
                    "detail": item.detail,
                }
                for item in self.values
            ],
        }


def _encode(value: float | None) -> float | str | None:
    if value is None:
        return None
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _channel(snapshot: MeasurementSnapshot, parameter: str):
    points = snapshot.channel(parameter)
    return tuple(points) if points else ()


def _unavailable(  # noqa: PLR0913
    kind: AnalysisKind,
    trace_id: str,
    trace_name: str,
    parameter: str,
    reason: str,
    parameters: dict | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        kind=kind,
        trace_id=trace_id,
        trace_name=trace_name,
        parameter=parameter,
        status=AnalysisStatus.UNAVAILABLE,
        reason=reason,
        parameters=parameters or {},
    )


# ----------------------------------------------------------------------
# Resonance
# ----------------------------------------------------------------------
def analyse_resonance(
    snapshot: MeasurementSnapshot,
    *,
    trace_id: str = "",
    trace_name: str = "",
    parameter: str = S11,
) -> AnalysisResult:
    """Phase zero crossings, exactly as ``ResonanceAnalysis`` finds them."""
    points = _channel(snapshot, parameter)
    if not points:
        return _unavailable(
            AnalysisKind.RESONANCE,
            trace_id,
            trace_name,
            parameter,
            f"no {parameter} data",
        )
    if len(points) < 3:
        return _unavailable(
            AnalysisKind.RESONANCE,
            trace_id,
            trace_name,
            parameter,
            "not enough points to detect a phase crossing",
            {"points": len(points)},
        )

    crossings = sorted(set(At.zero_crossings([dp.phase for dp in points])))
    if not crossings:
        return _unavailable(
            AnalysisKind.RESONANCE,
            trace_id,
            trace_name,
            parameter,
            "no resonance found",
            {"points": len(points)},
        )

    values = []
    for index in crossings:
        dp = points[index]
        impedance = dp.impedance(snapshot.reference_impedance)
        values.append(
            AnalysisValue(
                "Resonance",
                float(dp.freq),
                unit="Hz",
                frequency=dp.freq,
                detail=(
                    f"R {impedance.real:.4g} ohm, X {impedance.imag:.4g} ohm"
                ),
            )
        )
    return AnalysisResult(
        kind=AnalysisKind.RESONANCE,
        trace_id=trace_id,
        trace_name=trace_name,
        parameter=parameter,
        values=tuple(values),
        parameters={"points": len(points), "crossings": len(crossings)},
    )


# ----------------------------------------------------------------------
# VSWR minima and bandwidth
# ----------------------------------------------------------------------
def analyse_vswr(
    snapshot: MeasurementSnapshot,
    *,
    limit: float = 1.5,
    max_dips: int = 3,
    trace_id: str = "",
    trace_name: str = "",
) -> AnalysisResult:
    """VSWR minima below ``limit`` with the span where VSWR stays below it."""
    points = _channel(snapshot, S11)
    if not points:
        return _unavailable(
            AnalysisKind.VSWR,
            trace_id,
            trace_name,
            S11,
            "no S11 data",
        )
    if limit <= 1.0:
        return _unavailable(
            AnalysisKind.VSWR,
            trace_id,
            trace_name,
            S11,
            "VSWR limit must be greater than 1",
            {"limit": limit},
        )

    vswr_values = [dp.vswr for dp in points]
    minima = sorted(
        At.minima(vswr_values, limit), key=lambda index: vswr_values[index]
    )[:max_dips]
    if not minima:
        return _unavailable(
            AnalysisKind.VSWR,
            trace_id,
            trace_name,
            S11,
            f"no dip below VSWR {limit:g}",
            {"limit": limit},
        )

    values: list[AnalysisValue] = []
    for index in minima:
        span = At.take_from_idx(vswr_values, index, lambda i: i[1] < limit)
        begin, end = span[0], span[-1]
        minimum = points[index]
        values.append(
            AnalysisValue(
                "Minimum VSWR",
                float(minimum.vswr),
                unit="",
                frequency=minimum.freq,
                detail=minimum.freq and f"at {minimum.freq} Hz",
            )
        )
        values.append(
            AnalysisValue("Start", float(points[begin].freq), unit="Hz")
        )
        values.append(AnalysisValue("End", float(points[end].freq), unit="Hz"))
        values.append(
            AnalysisValue(
                "Bandwidth",
                float(points[end].freq - points[begin].freq),
                unit="Hz",
            )
        )
        values.append(
            AnalysisValue("Return loss", float(minimum.gain), unit="dB")
        )
    return AnalysisResult(
        kind=AnalysisKind.VSWR,
        trace_id=trace_id,
        trace_name=trace_name,
        parameter=S11,
        values=tuple(values),
        parameters={"limit": limit, "dips": len(minima)},
    )


# ----------------------------------------------------------------------
# Insertion loss and cutoff
# ----------------------------------------------------------------------
def analyse_insertion_loss(
    snapshot: MeasurementSnapshot,
    *,
    trace_id: str = "",
    trace_name: str = "",
) -> AnalysisResult:
    points = _channel(snapshot, S21)
    if not points:
        return _unavailable(
            AnalysisKind.INSERTION_LOSS,
            trace_id,
            trace_name,
            S21,
            "no S21 data (two-port measurement required)",
        )
    finite = [dp for dp in points if math.isfinite(dp.gain)]
    if not finite:
        return _unavailable(
            AnalysisKind.INSERTION_LOSS,
            trace_id,
            trace_name,
            S21,
            "no finite S21 magnitudes",
        )
    lowest = min(finite, key=lambda dp: dp.gain)
    highest = max(finite, key=lambda dp: dp.gain)
    average = sum(dp.gain for dp in finite) / len(finite)
    return AnalysisResult(
        kind=AnalysisKind.INSERTION_LOSS,
        trace_id=trace_id,
        trace_name=trace_name,
        parameter=S21,
        values=(
            AnalysisValue("Min gain", float(lowest.gain), "dB", lowest.freq),
            AnalysisValue("Max gain", float(highest.gain), "dB", highest.freq),
            AnalysisValue("Mean gain", float(average), "dB"),
            AnalysisValue(
                "Attenuation", float(-highest.gain), "dB", highest.freq
            ),
        ),
        parameters={"points": len(finite)},
    )


def analyse_cutoff(
    snapshot: MeasurementSnapshot,
    *,
    attenuation: float = 3.0,
    trace_id: str = "",
    trace_name: str = "",
) -> AnalysisResult:
    """-3 dB style cutoff of an S21 low-pass style response."""
    points = _channel(snapshot, S21)
    if not points:
        return _unavailable(
            AnalysisKind.CUTOFF,
            trace_id,
            trace_name,
            S21,
            "no S21 data (two-port measurement required)",
        )
    gains = [dp.gain for dp in points]
    if not any(math.isfinite(gain) for gain in gains):
        return _unavailable(
            AnalysisKind.CUTOFF,
            trace_id,
            trace_name,
            S21,
            "no finite S21 magnitudes",
        )
    peak_index = max(
        range(len(gains)),
        key=lambda i: gains[i] if math.isfinite(gains[i]) else -math.inf,
    )
    try:
        left = At.cut_off_left(
            gains, peak_index, gains[peak_index], attenuation
        )
        right = At.cut_off_right(
            gains, peak_index, gains[peak_index], attenuation
        )
    except (IndexError, ValueError):
        return _unavailable(
            AnalysisKind.CUTOFF,
            trace_id,
            trace_name,
            S21,
            "response does not cross the cutoff level",
            {"attenuation_db": attenuation},
        )
    return AnalysisResult(
        kind=AnalysisKind.CUTOFF,
        trace_id=trace_id,
        trace_name=trace_name,
        parameter=S21,
        values=(
            AnalysisValue(
                "Passband edge (low)", float(points[left].freq), "Hz"
            ),
            AnalysisValue(
                "Passband edge (high)", float(points[right].freq), "Hz"
            ),
            AnalysisValue(
                "Bandwidth",
                float(points[right].freq - points[left].freq),
                "Hz",
            ),
            AnalysisValue(
                "Peak gain",
                float(gains[peak_index]),
                "dB",
                points[peak_index].freq,
            ),
        ),
        parameters={"attenuation_db": attenuation},
    )


# ----------------------------------------------------------------------
# Group delay
# ----------------------------------------------------------------------
def analyse_group_delay(
    snapshot: MeasurementSnapshot,
    *,
    trace_id: str = "",
    trace_name: str = "",
) -> AnalysisResult:
    points = _channel(snapshot, S21)
    if not points:
        return _unavailable(
            AnalysisKind.GROUP_DELAY,
            trace_id,
            trace_name,
            S21,
            "no S21 data (two-port measurement required)",
        )
    if len(points) < 3:
        return _unavailable(
            AnalysisKind.GROUP_DELAY,
            trace_id,
            trace_name,
            S21,
            "not enough points for a group-delay estimate",
        )
    delays = _unwrapped_group_delays(points)
    if not delays:
        return _unavailable(
            AnalysisKind.GROUP_DELAY,
            trace_id,
            trace_name,
            S21,
            "group delay is not finite",
        )
    values = [delay for _, delay in delays]
    mean = sum(values) / len(values)
    return AnalysisResult(
        kind=AnalysisKind.GROUP_DELAY,
        trace_id=trace_id,
        trace_name=trace_name,
        parameter=S21,
        values=(
            AnalysisValue("Mean group delay", float(mean), "s"),
            AnalysisValue("Min group delay", float(min(values)), "s"),
            AnalysisValue("Max group delay", float(max(values)), "s"),
        ),
        parameters={"points": len(delays)},
    )


def _unwrapped_group_delays(
    points: Sequence[Datapoint],
) -> list[tuple[int, float]]:
    """Group delay per interior point using an unwrapped phase sequence.

    Per-point phase differences wrap at +/-pi, so a line whose total delay
    spans several turns produces wild outliers and a meaningless mean.
    Unwrapping first makes every central difference represent the true
    incremental delay.
    """
    phases = [dp.phase for dp in points]
    unwrapped = [phases[0]]
    offset = 0.0
    for previous, current in it.pairwise(phases):
        delta = current - previous
        if delta > math.pi:
            offset -= math.tau
        elif delta < -math.pi:
            offset += math.tau
        unwrapped.append(current + offset)

    delays: list[tuple[int, float]] = []
    for index in range(1, len(points) - 1):
        delta_freq = points[index + 1].freq - points[index - 1].freq
        if delta_freq == 0:
            continue
        delta_angle = unwrapped[index + 1] - unwrapped[index - 1]
        delay = -delta_angle / math.tau / delta_freq
        if math.isfinite(delay):
            delays.append((points[index].freq, delay))
    return delays


# ----------------------------------------------------------------------
# Cable
# ----------------------------------------------------------------------
def analyse_cable(
    snapshot: MeasurementSnapshot,
    *,
    velocity_factor: float = 0.66,
    trace_id: str = "",
    trace_name: str = "",
) -> AnalysisResult:
    """Impedance magnitude and electrical length estimate from S11."""
    points = _channel(snapshot, S11)
    if not points:
        return _unavailable(
            AnalysisKind.CABLE,
            trace_id,
            trace_name,
            S11,
            "no S11 data",
        )
    if not 0.0 < velocity_factor <= 1.0:
        return _unavailable(
            AnalysisKind.CABLE,
            trace_id,
            trace_name,
            S11,
            "velocity factor must be in (0, 1]",
            {"velocity_factor": velocity_factor},
        )
    impedances = [dp.impedance(snapshot.reference_impedance) for dp in points]
    magnitudes = [abs(z) for z in impedances]
    # An undefined sample has no |Z|.  Leaving it in makes ``sum`` NaN and
    # lets ``min``/``max`` pick it whenever it comes first, so the reported
    # range depends on sweep order.  Search the measured samples instead;
    # no usable sample is reported as the existing "unavailable" state.
    measured = [
        (index, magnitude)
        for index, magnitude in enumerate(magnitudes)
        if not math.isnan(magnitude)
    ]
    if not measured:
        return _unavailable(
            AnalysisKind.CABLE,
            trace_id,
            trace_name,
            S11,
            "no measured S11 magnitudes",
            {"velocity_factor": velocity_factor},
        )
    mean_z = sum(magnitude for _, magnitude in measured) / len(measured)
    lowest = min(measured, key=lambda item: item[1])
    highest = max(measured, key=lambda item: item[1])
    return AnalysisResult(
        kind=AnalysisKind.CABLE,
        trace_id=trace_id,
        trace_name=trace_name,
        parameter=S11,
        values=(
            AnalysisValue("Mean |Z|", float(mean_z), "ohm"),
            AnalysisValue(
                "Min |Z|",
                float(lowest[1]),
                "ohm",
                points[lowest[0]].freq,
            ),
            AnalysisValue(
                "Max |Z|",
                float(highest[1]),
                "ohm",
                points[highest[0]].freq,
            ),
        ),
        parameters={
            "velocity_factor": velocity_factor,
            "reference_impedance": snapshot.reference_impedance,
            "points": len(measured),
        },
    )


# ----------------------------------------------------------------------
# Bundles
# ----------------------------------------------------------------------
def analyse_snapshot(
    snapshot: MeasurementSnapshot,
    kinds: Iterable[AnalysisKind],
    *,
    trace_id: str = "",
    trace_name: str = "",
    parameters: dict | None = None,
) -> list[AnalysisResult]:
    """Run several analyses against one snapshot."""
    parameters = parameters or {}
    runners = {
        AnalysisKind.RESONANCE: lambda: analyse_resonance(
            snapshot, trace_id=trace_id, trace_name=trace_name
        ),
        AnalysisKind.VSWR: lambda: analyse_vswr(
            snapshot,
            limit=parameters.get("limit", 1.5),
            trace_id=trace_id,
            trace_name=trace_name,
        ),
        AnalysisKind.INSERTION_LOSS: lambda: analyse_insertion_loss(
            snapshot, trace_id=trace_id, trace_name=trace_name
        ),
        AnalysisKind.GROUP_DELAY: lambda: analyse_group_delay(
            snapshot, trace_id=trace_id, trace_name=trace_name
        ),
        AnalysisKind.CUTOFF: lambda: analyse_cutoff(
            snapshot,
            attenuation=parameters.get("attenuation", 3.0),
            trace_id=trace_id,
            trace_name=trace_name,
        ),
        AnalysisKind.CABLE: lambda: analyse_cable(
            snapshot,
            velocity_factor=parameters.get("velocity_factor", 0.66),
            trace_id=trace_id,
            trace_name=trace_name,
        ),
    }
    return [runners[kind]() for kind in kinds]


def summarize(results: Sequence[AnalysisResult]) -> str:
    """One-line text summary used by labels and reports."""
    if not results:
        return "No analysis results"
    available = [result for result in results if result.available]
    if not available:
        reasons = "; ".join(
            result.reason for result in results if result.reason
        )
        return (
            f"No analysis available ({reasons})"
            if reasons
            else "No analysis available"
        )
    return "; ".join(
        f"{result.kind.value}: "
        + ", ".join(value.formatted() for value in result.values[:2])
        for result in available
    )
