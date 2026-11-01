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
"""Calibration coverage model (CAL-01).

Coverage is derived from the *actual* calibration frequency samples and from
the correction channels that are numerically available.  It is deliberately
independent of the correction mathematics in :mod:`Calibration`: this module
never changes a corrected value.  It only describes where corrections are
interpolated between calibration samples and where they are held at an
endpoint or absent.

What coverage says: the corrected value at this frequency was derived from
interpolated calibration data inside the calibrated span, or from an
endpoint-held correction outside it.

What coverage does **not** say: anything about calibration age, connector
wear, measurement uncertainty, drift, or whether the calibration standards
were connected correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from .Measurement import (
    S11,
    S21,
    CalibrationIdentity,
    CorrectionSource,
    compute_calibration_identity,
)


class CoverageState(Enum):
    """Coverage classification for a frequency or a whole span."""

    UNCALIBRATED = "uncalibrated"
    """No usable calibration data at all."""

    FULL = "full"
    """Every frequency lies inside the calibrated span."""

    PARTIAL = "partial"
    """The span is mixed: some frequencies lie inside the calibrated span
    and some outside it."""

    OUTSIDE = "outside"
    """Every frequency lies outside the calibrated span.  Corrections are
    held at the calibration endpoints where available."""

    INVALID = "invalid"
    """Calibration data exists but cannot correct the requested channels."""


@dataclass(frozen=True)
class CoverageSpan:
    """Inclusive frequency span with the coverage that applies to it."""

    start: int
    end: int
    state: CoverageState


@dataclass(frozen=True)
class CoverageResult:
    """Coverage of one frequency grid for a set of measured channels."""

    state: CoverageState
    spans: tuple[CoverageSpan, ...]
    calibrated_start: int | None
    calibrated_end: int | None
    interpolated_points: int
    edge_held_points: int
    corrected_parameters: tuple[str, ...]
    uncorrected_parameters: tuple[str, ...]
    reason: str = ""

    @property
    def covered(self) -> bool:
        return self.state is CoverageState.FULL

    @property
    def has_calibration(self) -> bool:
        return self.state not in {
            CoverageState.UNCALIBRATED,
            CoverageState.INVALID,
        }

    @property
    def edge_held(self) -> bool:
        """True when any frequency relies on an endpoint-held correction."""
        return self.edge_held_points > 0

    def state_at(self, freq: int) -> CoverageState:
        """Marker-level lookup."""
        if self.state in {CoverageState.UNCALIBRATED, CoverageState.INVALID}:
            return self.state
        if self.calibrated_start is None:
            return CoverageState.UNCALIBRATED
        if self.calibrated_start <= freq <= self.calibrated_end:
            return CoverageState.FULL
        return CoverageState.OUTSIDE

    def describe(self) -> str:
        """Concise summary for status labels."""
        if self.state is CoverageState.UNCALIBRATED:
            return "No calibration"
        if self.state is CoverageState.INVALID:
            channel = ", ".join(self.uncorrected_parameters) or "channel"
            return f"Calibration unavailable for {channel}"
        span = (
            f"{self.calibrated_start} - {self.calibrated_end} Hz"
            if self.calibrated_start is not None
            else "unknown span"
        )
        labels = {
            CoverageState.FULL: "Calibrated",
            CoverageState.PARTIAL: "Partially calibrated",
            CoverageState.OUTSIDE: "Outside calibration",
        }
        text = f"{labels.get(self.state, self.state.value)} ({span})"
        if self.edge_held_points:
            text += f", {self.edge_held_points} point(s) held at edge"
        if self.uncorrected_parameters:
            text += ", no correction for " + ", ".join(
                self.uncorrected_parameters
            )
        return text


def _corrected_parameters(
    calibration, parameters: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Split ``parameters`` into ones this calibration can and cannot fix."""
    one_port = bool(calibration and calibration.isValid1Port())
    two_port = bool(calibration and calibration.isValid2Port())
    corrected: list[str] = []
    uncorrected: list[str] = []
    for parameter in parameters:
        if not one_port:
            uncorrected.append(parameter)
        elif parameter == S11:
            corrected.append(parameter)
        elif parameter == S21 and two_port:
            corrected.append(parameter)
        else:
            uncorrected.append(parameter)
    return corrected, uncorrected


def _build_spans(
    grid: Sequence[int], cal_start: int | None, cal_end: int | None
) -> tuple[CoverageSpan, ...]:
    """Group contiguous frequency runs by inside/outside classification."""
    if not grid or cal_start is None:
        return ()

    def classify(freq: int) -> CoverageState:
        return (
            CoverageState.FULL
            if cal_start <= freq <= cal_end
            else CoverageState.OUTSIDE
        )

    spans: list[CoverageSpan] = []
    run_state = classify(grid[0])
    run_start = grid[0]
    previous = grid[0]
    for freq in grid[1:]:
        state = classify(freq)
        if state is not run_state:
            spans.append(CoverageSpan(run_start, previous, run_state))
            run_state = state
            run_start = freq
        previous = freq
    spans.append(CoverageSpan(run_start, previous, run_state))
    return tuple(spans)


def coverage_from_calibration(
    calibration,
    frequencies: Iterable[int],
    parameters: Sequence[str] = (S11,),
) -> CoverageResult:
    """Coverage of ``calibration`` over ``frequencies`` for ``parameters``.

    A parameter is only considered *corrected* when the calibration supplies
    the suppressing term: one-port corrections for S11, two-port corrections
    for S21.  Frequencies outside the calibrated span still receive an
    endpoint-held correction and are reported as ``edge_held_points``.
    """
    grid = sorted({int(freq) for freq in frequencies})
    calibrated = calibration.dataset.frequencies() if calibration else []
    corrected, uncorrected = _corrected_parameters(calibration, parameters)

    if not calibrated:
        return CoverageResult(
            state=CoverageState.UNCALIBRATED,
            spans=(),
            calibrated_start=None,
            calibrated_end=None,
            interpolated_points=0,
            edge_held_points=0,
            corrected_parameters=tuple(corrected),
            uncorrected_parameters=tuple(uncorrected),
            reason="no calibration data",
        )

    cal_start, cal_end = calibrated[0], calibrated[-1]

    if not grid:
        return CoverageResult(
            state=CoverageState.INVALID,
            spans=(),
            calibrated_start=cal_start,
            calibrated_end=cal_end,
            interpolated_points=0,
            edge_held_points=0,
            corrected_parameters=tuple(corrected),
            uncorrected_parameters=tuple(uncorrected),
            reason="empty frequency grid",
        )

    interpolated = sum(1 for freq in grid if cal_start <= freq <= cal_end)
    edge_held = len(grid) - interpolated

    if interpolated == len(grid):
        state = CoverageState.FULL
    elif interpolated:
        state = CoverageState.PARTIAL
    else:
        state = CoverageState.OUTSIDE

    # A calibration that cannot correct any requested channel is invalid for
    # those channels even though its span may cover the grid.
    if not corrected:
        state = CoverageState.INVALID

    return CoverageResult(
        state=state,
        spans=_build_spans(grid, cal_start, cal_end),
        calibrated_start=cal_start,
        calibrated_end=cal_end,
        interpolated_points=interpolated,
        edge_held_points=edge_held,
        corrected_parameters=tuple(corrected),
        uncorrected_parameters=tuple(uncorrected),
        reason="" if corrected else "no correction channel available",
    )


def coverage_identity(
    calibration,
    frequencies: Iterable[int] = (),
    parameters: Sequence[str] = (S11, S21),
    correction_source: CorrectionSource = CorrectionSource.UNKNOWN,
) -> CalibrationIdentity:
    """Snapshot metadata for ``calibration``.

    ``correction_source`` records whether the device or the application
    performed the correction; ``UNKNOWN`` is used when device metadata is
    unavailable.
    """
    calibrated = calibration.dataset.frequencies() if calibration else []
    coverage = coverage_from_calibration(
        calibration, frequencies or calibrated, parameters
    )
    identity = compute_calibration_identity(calibration) if calibrated else ""
    spans: tuple[tuple[int, int], ...] = ()
    if coverage.calibrated_start is not None:
        spans = ((coverage.calibrated_start, coverage.calibrated_end),)
    return CalibrationIdentity(
        identity=identity,
        source=str(getattr(calibration, "source", "unknown")),
        notes="\n".join(getattr(calibration, "notes", []) or []),
        one_port=bool(calibration and calibration.isValid1Port()),
        two_port=bool(calibration and calibration.isValid2Port()),
        coverage_spans=spans,
        correction_source=correction_source,
    )


def state_at_frequency(result: CoverageResult, freq: int) -> CoverageState:
    """Marker-level coverage lookup for a frequency."""
    return result.state_at(freq)
