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
"""Phase-derivative diagnostics: group delay and phase curvature (ANALYSIS-04).

Everything here is Qt-free and works on the samples that were actually
measured.  Two quantities are derived from the phase of one channel:

* **group delay** ``tau_g = -d(phi)/d(omega)`` in seconds.  Sign convention:
  a cable or line delays the signal and its phase *decreases* with frequency,
  so the minus sign makes a line read a positive delay.
* **phase curvature** ``d2(phi)/d(omega)**2`` in square seconds, a
  *diagnostic* of departures from constant delay (see
  ``docs/PHASE_DERIVATIVES.md`` for what it detects and does not).

Conventions
-----------
* The independent variable is angular frequency ``omega = 2*pi*f``, not ``f``:
  a delay is ``-d(phi)/d(omega)``, and dividing by ``2*pi`` only if one
  differentiates with respect to hertz.  Mixing the two is the classic factor
  of 6.28 error, so only ``omega`` is ever used internally.
* Phase is unwrapped first, correcting each sample by the nearest multiple of
  ``2*pi``.  That is the same policy as
  :func:`NanoVNASaver.MarkerSearch.unwrapped_phase`; it is implemented here in
  radians instead of degrees so no degree round trip is introduced and so the
  number of corrections can be reported.  The policy's one assumption is
  stated plainly: a phase step larger than ``pi`` between adjacent samples
  cannot be told apart from a wrap, so such a curve is unwrapped wrongly.  The
  number of corrections records that a curve wrapped at all, and
  :attr:`PhaseDerivative.ambiguous_delay` gives the delay at which that
  assumption is doing the work (``pi`` over the smallest abscissa step), with
  :attr:`PhaseDerivative.wrap_ambiguous` saying whether any value reached it.
* Differentiation uses the actual local frequency steps, so a non-uniform
  (but strictly increasing) grid is supported.  Interior samples use a
  three-point quadratic Lagrange central stencil, both ends a three-point
  one-sided stencil, and a record of exactly two samples a two-point
  one-sided difference.  The stencils are exact for phase that is linear or
  quadratic in ``omega``, which covers pure delay and constant curvature, and
  are second-order accurate otherwise.
* Smoothing (optional, odd aperture in samples) is the complex moving average
  of :func:`Processing.smooth_channel`, applied before unwrapping so a wrapped
  curve is averaged across the branch instead of tearing at it.  Its window
  mean belongs to the window's frequency centroid, which is the sample's own
  frequency only where the window is symmetric about it; the derivative is
  therefore taken against the centroid, and the number of samples that
  affected is reported.  An aperture covering the whole record is refused,
  because it would flatten the phase and report a confident zero.
* A sample that cannot be differentiated is ``None``, never ``nan`` or
  ``inf``: a zero-magnitude sample has no phase, and it makes its own and its
  neighbours' stencils undefined.  Those points are counted in the result.
* A grid with a duplicate or decreasing frequency, an absent channel, or a
  record too short for the requested quantity is ``UNAVAILABLE`` with a
  reason.  An unavailable result never carries a number.
"""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .Processing import ProcessingError, smooth_channel
from .RFTools import Datapoint

TWO_PI = 2.0 * math.pi

#: Channels the panel offers.  The maths is identical for S12/S22 (a
#: reflection-phase delay is the round-trip delay of the reflection), so any
#: channel present in the mapping is accepted; these two are the ones the UI
#: lists and the ones the refusal message names.
SUPPORTED_CHANNELS = ("S11", "S21")

#: The differentiation method, reported with every result so a stored chart
#: can never be mistaken for samples from a different scheme.
METHOD = (
    "three-point quadratic Lagrange: central stencil on interior samples, "
    "one-sided stencil at both ends, two-point one-sided with only two samples"
)

#: Smoothing is the complex moving average of :func:`Processing.smooth_channel`
#: (symmetric, odd, truncated rather than padded at the ends).  It is applied
#: to the complex samples *before* unwrapping, so a wrapped curve is smoothed
#: continuously across +/-180 instead of tearing at the branch.
SMOOTHING = (
    "complex symmetric moving average (Processing.smooth_channel), "
    "applied before unwrapping"
)


class PhaseDerivativeError(ValueError):
    """A request that cannot mean anything, as opposed to data that is absent.

    A structurally invalid aperture (zero, negative, even, or not a number) is
    a caller bug and raises; data that is merely too short or malformed is
    reported as :data:`DerivativeStatus.UNAVAILABLE` instead.
    """


class PhaseQuantity(Enum):
    """Which phase derivative a result describes."""

    GROUP_DELAY = "group_delay"
    PHASE_CURVATURE = "phase_curvature"

    @property
    def label(self) -> str:
        if self is PhaseQuantity.GROUP_DELAY:
            return "group delay"
        return "phase curvature"

    @property
    def unit(self) -> str:
        if self is PhaseQuantity.GROUP_DELAY:
            return "s"
        return "s^2"


class DerivativeStatus(Enum):
    """Outcome category of a phase-derivative trace."""

    OK = "ok"
    #: Some samples are defined, some (and only some) are not.
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DerivativePoint:
    """One sample of a derivative trace.

    ``freq`` is the measured sample this point belongs to; ``omega`` is the
    abscissa its value is actually taken at, which equals ``2*pi*freq``
    unless smoothing moved the value to its window's frequency centroid (see
    :func:`_centroids`).  ``value`` is in the unit of its result; ``None``
    means undefined, which is a real outcome (zero-magnitude sample, a
    neighbour the stencil needs is missing) and is never silently turned into
    ``0.0``.
    """

    freq: int
    omega: float
    value: float | None

    @property
    def defined(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class PhaseDerivative:
    """An immutable phase-derivative trace with its provenance."""

    quantity: PhaseQuantity
    channel: str
    status: DerivativeStatus
    points: tuple[DerivativePoint, ...] = ()
    #: Aperture actually applied, in samples.  ``1`` means no smoothing.
    aperture_points: int = 1
    #: Number of ``2*pi`` corrections the unwrapping applied.
    wraps_corrected: int = 0
    #: Number of samples whose smoothed value sits at their window's frequency
    #: centroid rather than at their own frequency (see
    #: :func:`_centroids`); only non-zero when smoothing was applied.
    off_centre: int = 0
    #: Narrowest step in the grid's abscissa, in rad/s.  Nothing shorter than
    #: ``pi / step`` seconds of delay is distinguishable from a wrap between
    #: those samples, so it bounds what this trace can honestly claim.
    step_omega: float = 0.0
    reason: str = ""
    method: str = METHOD
    smoothing: str = SMOOTHING

    def __post_init__(self) -> None:
        for point in self.points:
            if point.value is None:
                continue
            if not math.isfinite(point.value):
                raise ValueError(
                    "derivative values must be finite or undefined, "
                    f"got {point.value!r} at {point.freq} Hz"
                )

    @property
    def ok(self) -> bool:
        return self.status is not DerivativeStatus.UNAVAILABLE

    @property
    def unit(self) -> str:
        return self.quantity.unit

    @property
    def smoothed(self) -> bool:
        return self.aperture_points > 1

    @property
    def window(self) -> str:
        """Wording for the aperture actually applied to the trace.

        An unsmoothed trace is called noisy on purpose: the second difference
        multiplies sample-to-sample phase noise by ``sqrt(6)/h**2`` (uniform
        steps ``h``), so a raw curvature chart shows noise as much as data and
        an operator must be told which of the two they are looking at.
        """
        if not self.smoothed:
            return (
                "raw curve, aperture 1 (unsmoothed: sample-to-sample noise "
                "is not suppressed)"
            )
        centroid = (
            f", {self.off_centre} value(s) at window centroids"
            if self.off_centre
            else ""
        )
        return f"smoothed, aperture {self.aperture_points} samples{centroid}"

    @property
    def ambiguous_delay(self) -> float | None:
        """Shortest delay this grid cannot tell from a wrap, in seconds.

        A phase advancing by ``pi`` between adjacent samples is
        indistinguishable from having advanced by ``-pi``, so the grid bounds
        what a delay claim can mean: ``pi / step_omega``.  A trace whose
        values approach this bound rests on the unwrapping assumption, and the
        number is exposed so a caller can say so rather than trust it
        silently.
        """
        if self.step_omega <= 0.0:
            return None
        return math.pi / self.step_omega

    @property
    def wrap_ambiguous(self) -> bool:
        """Whether any defined slope reaches this grid's wrap ambiguity.

        The unwrapping policy assumes no phase step exceeds ``pi`` between
        adjacent samples.  When the true slope does exceed it — a delay longer
        than ``pi / step_omega`` — the recorded phase is aliased and the
        "unwrapped" curve is a different, equally consistent phase.  The
        arithmetic still succeeds, so the only honest response is to say the
        result rests on that assumption instead of presenting it as measured.

        Only a group-delay trace can be tested: after unwrapping every step is
        by construction within ``pi``, so the aliasing is invisible in the
        phase and shows up only as a delay at or beyond the limit.  A
        curvature result therefore always answers ``False``, which is a
        statement about what curvature can show and not a clean bill of
        health; read it together with the group delay of the same samples
        (:func:`diagnostics` returns both).
        """
        limit = self.ambiguous_delay
        if limit is None:
            return False
        if self.quantity is PhaseQuantity.GROUP_DELAY:
            return any(abs(value) >= limit for value in self.values)
        # Curvature is in s**2, so there is no direct comparison against a
        # delay limit.  Returning False here is not a clean bill of health:
        # curvature inherits whatever aliasing the phase carries, so a caller
        # must read the group delay from the same samples (``diagnostics()``
        # returns both) to know whether the phase was unambiguous.
        return False

    @property
    def defined(self) -> tuple[DerivativePoint, ...]:
        return tuple(point for point in self.points if point.defined)

    @property
    def undefined_count(self) -> int:
        return len(self.points) - len(self.defined)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(
            point.value for point in self.defined if point.value is not None
        )

    @property
    def median(self) -> float | None:
        """Median of the defined samples, in this result's unit.

        A median rather than a mean: one bad point from a notch must not move
        the headline number.
        """
        values = self.values
        return statistics.median(values) if values else None

    def value_at(self, freq: int) -> float | None:
        """Derivative at exactly ``freq`` Hz, or ``None``.

        No interpolation: a derivative between samples is a fitted claim, and
        this module does not make one.
        """
        for point in self.points:
            if point.freq == freq:
                return point.value
        return None

    def describe(self) -> str:
        name = f"{self.quantity.label} of {self.channel}"
        if self.status is DerivativeStatus.UNAVAILABLE:
            return f"{name}: unavailable ({self.reason or 'no result'})"
        parts = [self.window]
        parts.append(
            f"{self.wraps_corrected} wrap correction(s)"
            if self.wraps_corrected
            else "no wrap corrections"
        )
        if self.quantity is PhaseQuantity.PHASE_CURVATURE:
            parts.append("diagnostic")
        if self.undefined_count:
            parts.append(f"{self.undefined_count} undefined sample(s)")
        if self.wrap_ambiguous:
            parts.append(
                "relies on the unwrapping assumption: the slope reaches the "
                f"{self.ambiguous_delay:.3g} s wrap ambiguity of this grid"
            )
        median = self.median
        value = (
            "no defined samples"
            if median is None
            else f"{median:.6g} {self.unit}"
        )
        return (
            f"{name}: {value} (median, {len(self.defined)} of "
            f"{len(self.points)} samples; {', '.join(parts)})"
        )


# ----------------------------------------------------------------------
# Phase access
# ----------------------------------------------------------------------
def _has_phase(point: Datapoint) -> bool:
    """Whether ``point`` carries a phase at all.

    A zero-magnitude (or non-finite) sample has no phase; ``cmath.phase``
    would invent ``0.0`` for it, which is exactly the plausible number this
    module must not report.
    """
    if not (math.isfinite(point.re) and math.isfinite(point.im)):
        return False
    return math.hypot(point.re, point.im) > 0.0


def _unwrap(
    points: Sequence[Datapoint],
) -> tuple[tuple[float | None, ...], int]:
    """Unwrapped phase in radians, plus the number of ``2*pi`` corrections.

    Each defined sample is corrected by the nearest multiple of ``2*pi``
    relative to the last *defined* sample, which is the policy of
    :func:`NanoVNASaver.MarkerSearch.unwrapped_phase`.  Over a gap of undefined
    samples the same offset is carried forward, so the correction is only
    trusted as far as the gap is narrow compared with the phase slope: a wrap
    inside a gap is unresolvable from the samples, and that is stated rather
    than guessed at.
    """
    values: list[float | None] = []
    offset = 0.0
    previous: float | None = None
    corrections = 0
    for point in points:
        if not _has_phase(point):
            values.append(None)
            continue
        phase = point.phase
        if previous is not None:
            while phase + offset - previous > math.pi:
                offset -= TWO_PI
                corrections += 1
            while phase + offset - previous < -math.pi:
                offset += TWO_PI
                corrections += 1
        value = phase + offset
        values.append(value)
        previous = value
    return tuple(values), corrections


def unwrap_phase(points: Sequence[Datapoint]) -> tuple[float | None, ...]:
    """Unwrapped phase in radians, ``None`` where the sample has no phase.

    Public because a chart must be able to draw the same phase the derivative
    was taken from; the wrap count travels with the derivative result instead.
    """
    return _unwrap(points)[0]


# ----------------------------------------------------------------------
# Grid access
# ----------------------------------------------------------------------
def available_channels(channels: Mapping[str, Sequence[Datapoint]]) -> tuple:
    """Which of :data:`SUPPORTED_CHANNELS` this mapping can offer, in order.

    A UI uses this to grey out the choices instead of letting an operator pick
    a channel whose derivative will then be unavailable.
    """
    return tuple(name for name in SUPPORTED_CHANNELS if channels.get(name))


def _channel_points(
    channels: Mapping[str, Sequence[Datapoint]], channel: str
) -> tuple[tuple[Datapoint, ...] | None, str]:
    points = channels.get(channel)
    if points is None:
        available = ", ".join(sorted(channels)) or "none"
        return None, f"channel {channel} is absent (available: {available})"
    if not points:
        return None, f"channel {channel} has no samples"
    return tuple(points), ""


def _grid_reason(frequencies: Sequence[int]) -> str:
    """Why this grid cannot be differentiated, or ``""`` if it can.

    A non-uniform grid is fine as long as it is strictly increasing; the
    stencils use the real local steps.  A duplicate makes a step zero and a
    decreasing grid makes it negative, and neither has a derivative.
    """
    for index in range(1, len(frequencies)):
        previous = frequencies[index - 1]
        current = frequencies[index]
        if current == previous:
            return (
                f"duplicate frequency {current} Hz at samples "
                f"{index - 1} and {index}"
            )
        if current < previous:
            return (
                f"frequency grid is not monotonic: sample {index} is "
                f"{current} Hz after {previous} Hz"
            )
    return ""


# ----------------------------------------------------------------------
# Stencils
# ----------------------------------------------------------------------
def _first_left(h1: float, h2: float, f0: float, f1: float, f2: float):
    """d/dx at the first of three points (one-sided, second-order exact)."""
    return (
        -(2.0 * h1 + h2) / (h1 * (h1 + h2)) * f0
        + (h1 + h2) / (h1 * h2) * f1
        - h1 / (h2 * (h1 + h2)) * f2
    )


def _first_central(h1: float, h2: float, f0: float, f1: float, f2: float):
    """d/dx at the middle of three points (central, second-order exact)."""
    return (
        -h2 / (h1 * (h1 + h2)) * f0
        + (h2 - h1) / (h1 * h2) * f1
        + h1 / (h2 * (h1 + h2)) * f2
    )


def _first_right(h1: float, h2: float, f0: float, f1: float, f2: float):
    """d/dx at the last of three points (one-sided, second-order exact)."""
    return (
        h2 / (h1 * (h1 + h2)) * f0
        - (h1 + h2) / (h1 * h2) * f1
        + (h1 + 2.0 * h2) / (h2 * (h1 + h2)) * f2
    )


def _curvature(h1: float, h2: float, f0: float, f1: float, f2: float):
    """d2/dx2 of the quadratic through three points with steps h1, h2."""
    return 2.0 / (h1 + h2) * ((f2 - f1) / h2 - (f1 - f0) / h1)


def _window(
    phase: Sequence[float | None], low: int
) -> tuple[float, float, float] | None:
    """Three consecutive defined phases, or ``None`` if any is undefined."""
    chunk = phase[low : low + 3]
    if len(chunk) != 3 or any(value is None for value in chunk):
        return None
    first, second, third = chunk
    if first is None or second is None or third is None:  # narrowing
        return None
    return first, second, third


def _first_derivative(
    omega: Sequence[float], phase: Sequence[float | None]
) -> tuple[float | None, ...]:
    """d(phase)/d(omega) at every sample, ``None`` where undefined."""
    count = len(omega)
    out: list[float | None] = [None] * count
    if count == 2:
        if phase[0] is not None and phase[1] is not None:
            value = (phase[1] - phase[0]) / (omega[1] - omega[0])
            out[0] = value
            out[1] = value
        return tuple(out)
    for index in range(count):
        if index == 0:
            low, stencil = 0, _first_left
        elif index == count - 1:
            low, stencil = count - 3, _first_right
        else:
            low, stencil = index - 1, _first_central
        window = _window(phase, low)
        if window is None:
            continue
        h1 = omega[low + 1] - omega[low]
        h2 = omega[low + 2] - omega[low + 1]
        out[index] = stencil(h1, h2, *window)
    return tuple(out)


def _second_derivative(
    omega: Sequence[float], phase: Sequence[float | None]
) -> tuple[float | None, ...]:
    """d2(phase)/d(omega)**2 at every sample, ``None`` where undefined.

    The value is the constant curvature of the quadratic fitted to the sample
    and its two nearest neighbours (one side only at the ends), so an
    endpoint curvature is a one-sided estimate and is reported as such by the
    shared stencil rule rather than being invented from a padded window.
    """
    count = len(omega)
    out: list[float | None] = [None] * count
    if count < 3:
        return tuple(out)
    for index in range(count):
        if index == 0:
            low = 0
        elif index == count - 1:
            low = count - 3
        else:
            low = index - 1
        window = _window(phase, low)
        if window is None:
            continue
        h1 = omega[low + 1] - omega[low]
        h2 = omega[low + 2] - omega[low + 1]
        out[index] = _curvature(h1, h2, *window)
    return tuple(out)


# ----------------------------------------------------------------------
# Shared preparation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class _Prepared:
    freqs: tuple[int, ...]
    #: Abscissa each (possibly smoothed) value genuinely belongs to: the
    #: frequency centroid of its smoothing window.  Equal to ``2*pi*freq``
    #: when no smoothing is applied.
    omega: tuple[float, ...]
    phase: tuple[float | None, ...]
    corrections: int
    aperture: int
    #: Samples whose window centroid is not their own frequency, so their
    #: derivative is reported at the centroid rather than at the sample's
    #: frequency.  Counted, and named in ``describe()``.
    off_centre: int = 0
    #: Narrowest abscissa step actually used by the stencils, in rad/s.  It
    #: sets the delay this grid can distinguish from a wrap.
    step_omega: float = 0.0


def _window_bounds(index: int, count: int, half: int) -> tuple[int, int]:
    """The half-open sample window :func:`Processing.smooth_channel` uses."""
    return max(0, index - half), min(count, index + half + 1)


def _centroids(
    sampled_omega: Sequence[float], aperture_points: int
) -> tuple[tuple[float, ...], int]:
    """Frequency centroid of every smoothing window, and the off-centre count.

    :func:`Processing.smooth_channel` replaces a sample with the mean of its
    complex window, and that mean is the phase at the window's *frequency
    centroid* — not at the middle sample's own frequency.  The two coincide
    only when the window is symmetric in frequency about that sample: true for
    an interior window of an equal-step grid, false for a window truncated at
    either end (it is centred on the samples that exist) and false where the
    local step changes, as at a segment boundary.

    Differentiating a mean against the wrong abscissa tilts the slope.  On a
    delay line at aperture 9 with 5 MHz steps, using the middle sample's
    frequency gives a 50% error at the first two samples; using the centroid
    gives 1e-14 relative everywhere including the ends, because the only
    residual is the second-order term of averaging phasors across the window.
    So the centroid is what the stencils use, and ``off_centre`` reports how
    many samples that affected.  Nothing is discarded and nothing is silently
    relocated: the point's ``omega`` says where its value actually is.
    """
    count = len(sampled_omega)
    if aperture_points <= 1:
        return tuple(sampled_omega), 0
    half = aperture_points // 2
    centres: list[float] = []
    off_centre = 0
    for index in range(count):
        low, high = _window_bounds(index, count, half)
        centre = math.fsum(sampled_omega[low:high]) / (high - low)
        centres.append(centre)
        # A window symmetric about its middle sample either hits that sample's
        # frequency exactly (equal steps) or misses it by a whole fraction of a
        # step (truncated end, or a change of step inside the window); the
        # tolerance only absorbs the last bits of the mean of equal steps,
        # never a partial step, so the count is exact rather than an artefact
        # of rounding.
        tolerance = 1e-9 * (sampled_omega[high - 1] - sampled_omega[low])
        if abs(centre - sampled_omega[index]) > tolerance:
            off_centre += 1
    return tuple(centres), off_centre


def _aperture_reason(aperture_points: int, samples: int) -> str:
    """Why this aperture cannot be honoured on this record, or ``""``.

    An aperture that covers the whole record averages every sample into every
    output, so every smoothed value is identical, the phase is flat and the
    "derivative" would be a confident zero.  That is exactly the plausible
    number this module must not produce, so the request is refused instead.
    The check only ever fires when smoothing was asked for; aperture 1 on a
    short record is a raw trace and is judged by the sample-count rule of the
    quantity, not here.
    """
    if aperture_points <= 1 or aperture_points < samples:
        return ""
    return (
        f"smoothing aperture of {aperture_points} samples covers the "
        f"whole {samples}-sample record, which would flatten the phase"
    )


def _prepare(
    channels: Mapping[str, Sequence[Datapoint]],
    channel: str,
    aperture_points: int,
) -> tuple[_Prepared | None, str]:
    if isinstance(aperture_points, bool) or not isinstance(
        aperture_points, int
    ):
        raise PhaseDerivativeError("smoothing aperture must be an integer")
    if aperture_points < 1:
        raise PhaseDerivativeError("smoothing aperture must be at least 1")
    if aperture_points % 2 == 0:
        raise PhaseDerivativeError(
            "smoothing aperture must be an odd point count"
        )
    points, reason = _channel_points(channels, channel)
    if points is None:
        return None, reason
    frequencies = tuple(point.freq for point in points)
    reason = _grid_reason(frequencies)
    if reason:
        return None, reason
    reason = _aperture_reason(aperture_points, len(points))
    if reason:
        return None, reason
    try:
        smoothed = smooth_channel(points, aperture_points)
    except ProcessingError as exc:  # defensive: aperture already validated
        return None, str(exc)
    sampled_omega = tuple(TWO_PI * float(point.freq) for point in points)
    omega, off_centre = _centroids(sampled_omega, aperture_points)
    phase, corrections = _unwrap(smoothed)
    step = min(
        (later - earlier for earlier, later in itertools.pairwise(omega)),
        default=0.0,
    )
    return (
        _Prepared(
            freqs=frequencies,
            omega=omega,
            phase=phase,
            corrections=corrections,
            aperture=aperture_points,
            off_centre=off_centre,
            step_omega=step,
        ),
        "",
    )


def _unavailable(
    quantity: PhaseQuantity,
    channel: str,
    reason: str,
    aperture_points: int = 1,
) -> PhaseDerivative:
    return PhaseDerivative(
        quantity=quantity,
        channel=channel,
        status=DerivativeStatus.UNAVAILABLE,
        aperture_points=aperture_points,
        reason=reason,
    )


def _result(
    quantity: PhaseQuantity,
    channel: str,
    values: Sequence[float | None],
    prepared: _Prepared,
) -> PhaseDerivative:
    points = tuple(
        DerivativePoint(freq=freq, omega=omega, value=value)
        for freq, omega, value in zip(
            prepared.freqs, prepared.omega, values, strict=True
        )
    )
    defined = sum(1 for value in values if value is not None)
    if not defined:
        return PhaseDerivative(
            quantity=quantity,
            channel=channel,
            status=DerivativeStatus.UNAVAILABLE,
            points=points,
            aperture_points=prepared.aperture,
            wraps_corrected=prepared.corrections,
            off_centre=prepared.off_centre,
            step_omega=prepared.step_omega,
            reason="no sample could be differentiated",
        )
    status = (
        DerivativeStatus.OK
        if defined == len(values)
        else DerivativeStatus.PARTIAL
    )
    reason = (
        ""
        if status is DerivativeStatus.OK
        else f"{len(values) - defined} of {len(values)} samples undefined"
    )
    return PhaseDerivative(
        quantity=quantity,
        channel=channel,
        status=status,
        points=points,
        aperture_points=prepared.aperture,
        wraps_corrected=prepared.corrections,
        off_centre=prepared.off_centre,
        step_omega=prepared.step_omega,
        reason=reason,
    )


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------
def group_delay(
    channels: Mapping[str, Sequence[Datapoint]],
    channel: str = "S21",
    *,
    aperture_points: int = 1,
) -> PhaseDerivative:
    """Group delay ``-d(phi)/d(omega)`` in seconds for one channel.

    ``aperture_points`` is the smoothing aperture in samples, applied to the
    complex samples before unwrapping and differentiation; ``1`` means the
    raw trace, and the result says which one it is.  Needs two defined
    samples; a single sample is ``UNAVAILABLE``.
    """
    prepared, reason = _prepare(channels, channel, aperture_points)
    if prepared is None:
        return _unavailable(
            PhaseQuantity.GROUP_DELAY, channel, reason, aperture_points
        )
    if len(prepared.omega) < 2:
        return _unavailable(
            PhaseQuantity.GROUP_DELAY,
            channel,
            "group delay needs at least two samples, got 1",
            aperture_points,
        )
    derivative = _first_derivative(prepared.omega, prepared.phase)
    values = tuple(None if value is None else -value for value in derivative)
    return _result(PhaseQuantity.GROUP_DELAY, channel, values, prepared)


def phase_curvature(
    channels: Mapping[str, Sequence[Datapoint]],
    channel: str = "S21",
    *,
    aperture_points: int = 1,
) -> PhaseDerivative:
    """Phase curvature ``d2(phi)/d(omega)**2`` in s**2 for one channel.

    A diagnostic, not a material constant: see ``docs/PHASE_DERIVATIVES.md``
    for what it detects (departures from constant delay such as a discrete
    echo or a resonance) and why an unsmoothed curvature trace is dominated
    by differentiation noise.  Needs three consecutive defined samples
    somewhere in the record; inside it, every sample whose three-point window
    (one-sided at the ends) is complete is defined, and the rest are ``None``.
    """
    prepared, reason = _prepare(channels, channel, aperture_points)
    if prepared is None:
        return _unavailable(
            PhaseQuantity.PHASE_CURVATURE, channel, reason, aperture_points
        )
    if len(prepared.omega) < 3:
        return _unavailable(
            PhaseQuantity.PHASE_CURVATURE,
            channel,
            f"phase curvature needs at least three samples, "
            f"got {len(prepared.omega)}",
            aperture_points,
        )
    values = _second_derivative(prepared.omega, prepared.phase)
    return _result(PhaseQuantity.PHASE_CURVATURE, channel, values, prepared)


def diagnostics(
    channels: Mapping[str, Sequence[Datapoint]],
    channel: str = "S21",
    *,
    aperture_points: int = 1,
) -> tuple[PhaseDerivative, PhaseDerivative]:
    """Group delay and curvature of the same samples with the same aperture.

    Returned together so a chart of the two can never mix apertures, which
    would make their features incomparable.
    """
    return (
        group_delay(channels, channel, aperture_points=aperture_points),
        phase_curvature(channels, channel, aperture_points=aperture_points),
    )
