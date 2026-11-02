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
"""Fitting an extension's delay and loss from a standard (RF-01).

The operator measures a known standard through their fixture: an open, a
short, or a through connection.  Whatever the fixture is, the *true*
response of that standard is known, so the ratio of the measurement to the
truth is exactly the fixture's error.  This module turns that ratio into the
delay and loss that :func:`Processing.apply_port_extension` can remove, and
refuses whenever the measurement cannot support the fit.

Conventions
-----------
* Delay is reported in picoseconds and is **one way**: the round trip is
  applied twice by the processing stage for a reflection and once per port
  for a transmission.  Fitting the doubled value here would double it again.
* Loss is reported in dB per GHz, one way, on the same basis.
* A fit reports its residual and the fraction of samples it could use.  A
  delay fitted through wrapped phase is only as good as the unwrapping, so
  the residual is what tells the operator whether to trust it.
* Refusals name the reason.  Insufficient bandwidth, a measurement that does
  not look like the standard it claims to be, or a band where the phase turn
  is under one cycle cannot pin down a delay, and saying so is better than
  returning a number with a confident face.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .Processing import (
    ProcessingStage,
    port_extension_factor,
    port_extension_stage,
)
from .RFTools import Datapoint

#: A standard is recognised by how close its measured |Γ| is to the ideal.
#: An open should be near +1 and a short near -1; anything else means the
#: operator picked the wrong standard, and fitting to it would be fiction.
MAGNITUDE_TOLERANCE = 0.25

#: The band must turn through at least this fraction of a cycle for the delay
#: to be identifiable; below it, a delay error is indistinguishable from a
#: calibration offset.
MIN_TURNS = 0.5


class StandardKind(Enum):
    """Which known standard was measured through the fixture."""

    OPEN = "open"
    SHORT = "short"
    THROUGH = "through"


class FitStatus(Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ExtensionFit:
    """A fitted one-way delay and loss for one port."""

    status: FitStatus
    delay_ps: float = 0.0
    loss_db_per_ghz: float = 0.0
    #: Root-mean-square difference between the corrected and ideal standard,
    #: in linear magnitude units.  Zero is a perfect fit.
    residual: float | None = None
    #: Samples inside the requested band, and samples the fit actually used.
    samples: int = 0
    used: int = 0
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is FitStatus.OK

    def describe(self) -> str:
        if not self.ok:
            return f"unavailable ({self.reason or 'no fit'})"
        residual = (
            "" if self.residual is None else f", residual {self.residual:.3g}"
        )
        return (
            f"delay {self.delay_ps:.3f} ps, loss {self.loss_db_per_ghz:.3f}"
            f" dB/GHz from {self.used}/{self.samples} samples{residual}"
        )

    def stage(self, *, port: int = 1) -> ProcessingStage | None:
        """The port-extension stage this fit describes, or ``None``.

        ``None`` rather than a zero stage is the honest result of "nothing
        to remove": a stage with both values zero is an identity step that
        would still be recorded in provenance and still invalidate averages.
        """
        if not self.ok:
            return None
        return port_extension_stage(
            **{
                f"port{port}_delay_ps": self.delay_ps,
                f"port{port}_loss_db_per_ghz": self.loss_db_per_ghz,
            }
        )


def _unavailable(reason: str, samples: int = 0) -> ExtensionFit:
    return ExtensionFit(
        status=FitStatus.UNAVAILABLE, samples=samples, reason=reason
    )


def _target_gamma(kind: StandardKind) -> complex:
    """The ideal response of a standard, at the measurement plane."""
    if kind is StandardKind.OPEN:
        return complex(1.0, 0.0)
    if kind is StandardKind.SHORT:
        return complex(-1.0, 0.0)
    return complex(1.0, 0.0)


def _band(
    points: Sequence[Datapoint], start: int | None, stop: int | None
) -> list[Datapoint]:
    low = -math.inf if start is None else start
    high = math.inf if stop is None else stop
    out = [
        point
        for point in points
        if low <= point.freq <= high
        and math.isfinite(point.re)
        and math.isfinite(point.im)
    ]
    return out


def _unwrapped(points: Sequence[Datapoint]) -> list[float]:
    """Phase in radians, unwrapped across the +/-pi branch."""
    out: list[float] = []
    offset = 0.0
    previous = None
    for point in points:
        value = point.phase
        if previous is not None:
            while value + offset - previous > math.pi:
                offset -= math.tau
            while value + offset - previous < -math.pi:
                offset += math.tau
        current = value + offset
        out.append(current)
        previous = current
    return out


def fit_port_extension(
    points: Sequence[Datapoint],
    kind: StandardKind,
    *,
    start: int | None = None,
    stop: int | None = None,
    doubled: bool = True,
) -> ExtensionFit:
    """Fit one-way delay and loss from a standard measured through a fixture.

    The delay comes from the slope of unwrapped phase against frequency: for
    a reflection, ``phi = -2*pi*f*2*tau`` for a one-way delay ``tau``.  The
    loss comes from the magnitude ratio to the ideal standard, which for
    either an open or a short should be unity and for a through likewise.
    Both are least-squares fits over the band, so one noisy sample cannot
    dominate.

    ``doubled`` must match how the processing stage will apply the result:
    ``True`` for a reflection (``S11``/``S22``), ``False`` for a through.
    """
    if not points:
        return _unavailable("no samples to fit")
    non_finite = sum(
        1
        for point in points
        if not (math.isfinite(point.re) and math.isfinite(point.im))
    )
    if non_finite:
        return _unavailable(f"{non_finite} non-finite samples in the data")
    band = _band(points, start, stop)
    if len(band) < 3:
        return _unavailable(
            "at least three samples are needed inside the band",
            samples=len(band),
        )
    target = _target_gamma(kind)
    ideal_magnitude = abs(target)
    magnitudes = [abs(point.z) for point in band]
    # A standard that does not look like itself means the operator selected
    # the wrong one, or the reference plane is wildly wrong; either way the
    # fit would be fitted to the wrong ideal.
    off = [
        magnitude
        for magnitude in magnitudes
        if abs(magnitude - ideal_magnitude) > MAGNITUDE_TOLERANCE
    ]
    if len(off) > len(band) // 2:
        return _unavailable(
            f"the data does not look like a {kind.value}: median |Gamma|"
            f" {_median(magnitudes):.3f} is not within"
            f" {MAGNITUDE_TOLERANCE:g} of {ideal_magnitude:g}",
            samples=len(band),
        )

    # Phase fit.  A reflection's measured phase includes the standard's own
    # 0 or 180 degrees; removing it leaves only the fixture's rotation.
    phases = _unwrapped(band)
    if kind is StandardKind.OPEN:
        phases = [value - cmath.phase(target) for value in phases]
    elif kind is StandardKind.SHORT:
        # A short's ideal phase is pi; subtract and let the unwrap absorb
        # the resulting step so the slope is unaffected.
        phases = [value - math.pi for value in phases]
    freqs = [point.freq for point in band]
    span_hz = freqs[-1] - freqs[0]
    if span_hz <= 0:
        return _unavailable("the band has no frequency span", samples=len(band))
    turns = abs(phases[-1] - phases[0]) / math.tau
    if turns < MIN_TURNS:
        return _unavailable(
            f"the band turns only {turns:.3g} cycles; at least"
            f" {MIN_TURNS:g} are needed to identify a delay",
            samples=len(band),
        )
    slope, intercept = _linear_fit(freqs, phases)
    # phi = 2*pi*f*tau*passes  =>  tau = slope / (2*pi*passes)
    passes = 2.0 if doubled else 1.0
    delay_seconds = -slope / (math.tau * passes)
    if not math.isfinite(delay_seconds):
        return _unavailable("the phase slope is not finite", len(band))
    if delay_seconds < 0:
        # A negative delay means the reference plane is already past the
        # standard.  Reporting it would rotate every later measurement the
        # wrong way; the operator has to move the plane, not accept a number.
        return _unavailable(
            f"the fitted delay is negative ({delay_seconds * 1e12:.3f} ps);"
            " the reference plane is already past the standard",
            samples=len(band),
        )

    # Loss fit: the magnitude ratio to the ideal standard, in dB, against
    # frequency.  A gain (negative loss) is equally unphysical for a passive
    # fixture.
    loss_samples = []
    for point in band:
        magnitude = abs(point.z)
        if magnitude <= 0:
            continue
        loss_samples.append(20.0 * math.log10(magnitude / ideal_magnitude))
    loss_db_per_ghz = 0.0
    if len(loss_samples) >= 2:
        # Linear in frequency in dB: fit dB against GHz and negate, because
        # the extension's loss must be divided out.
        ghz = [point.freq / 1e9 for point in band[: len(loss_samples)]]
        loss_slope, _loss_intercept = _linear_fit(ghz, loss_samples)
        if math.isfinite(loss_slope):
            loss_db_per_ghz = -loss_slope / passes
    if loss_db_per_ghz < 0:
        loss_db_per_ghz = 0.0

    # Residual: correct the data with the fitted extension and compare to the
    # ideal standard.  This is measured, not assumed, so a bad unwrap shows up.
    squared = 0.0
    used = 0
    for point in band:
        factor = port_extension_factor(
            point.freq,
            delay_seconds=delay_seconds,
            loss_db_per_ghz=loss_db_per_ghz,
            doubled=doubled,
        )
        corrected = point.z * factor
        squared += abs(corrected - target) ** 2
        used += 1
    residual = math.sqrt(squared / used) if used else None
    return ExtensionFit(
        status=FitStatus.OK,
        delay_ps=delay_seconds * 1e12,
        loss_db_per_ghz=loss_db_per_ghz,
        residual=residual,
        samples=len(band),
        used=used,
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _linear_fit(
    xs: Sequence[float], ys: Sequence[float]
) -> tuple[float, float]:
    """Least-squares ``(slope, intercept)``; a degenerate x range is zero."""
    count = len(xs)
    if count < 2:
        return 0.0, float(ys[0]) if ys else 0.0
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    numerator = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)
    )
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return 0.0, mean_y
    slope = numerator / denominator
    return slope, mean_y - slope * mean_x


def fit_reflection_extension(
    open_points: Sequence[Datapoint] | None = None,
    short_points: Sequence[Datapoint] | None = None,
    *,
    start: int | None = None,
    stop: int | None = None,
) -> ExtensionFit:
    """Fit from an open and/or a short measured through the fixture.

    Both standards see the same fixture, so fitting both and averaging the
    delay reduces the effect of whichever standard is the less ideal
    realisation.  The loss is taken from the magnitude, which is the same
    for both.  One standard alone is enough; neither is a refusal.
    """
    fits = []
    for points, kind in (
        (open_points, StandardKind.OPEN),
        (short_points, StandardKind.SHORT),
    ):
        if points:
            fit = fit_port_extension(
                points, kind, start=start, stop=stop, doubled=True
            )
            if fit.ok:
                fits.append(fit)
    if not fits:
        return _unavailable("neither the open nor the short could be fitted", 0)
    delay = sum(fit.delay_ps for fit in fits) / len(fits)
    loss = sum(fit.loss_db_per_ghz for fit in fits) / len(fits)
    residual = (
        sum(fit.residual or 0.0 for fit in fits) / len(fits)
        if any(fit.residual is not None for fit in fits)
        else None
    )
    return ExtensionFit(
        status=FitStatus.OK,
        delay_ps=delay,
        loss_db_per_ghz=loss,
        residual=residual,
        samples=max(fit.samples for fit in fits),
        used=max(fit.used for fit in fits),
    )


def fit_through_extension(
    points: Sequence[Datapoint],
    *,
    start: int | None = None,
    stop: int | None = None,
) -> ExtensionFit:
    """Fit a transmission fixture from a through connection (both ports)."""
    return fit_port_extension(
        points, StandardKind.THROUGH, start=start, stop=stop, doubled=False
    )
