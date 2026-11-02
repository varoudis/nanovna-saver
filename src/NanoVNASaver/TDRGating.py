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
"""Time-domain reflectometry gating (TDR-01).

A gate is the only way to look at one echo in a sweep, and the only way to
get a *wrong* answer that looks better than the right one: the gated trace is
not a measurement any more, it is a measurement that has been filtered in a
different domain.  So everything here keeps two things explicit -- the
transform that produced the time record, and the fact that the returned
frequency trace is derived.

Conventions of the audited GUI transform
---------------------------------------
:mod:`NanoVNASaver.Windows.TDR` (``TDRWindow.updateTDR``) is the reference
implementation this module replaces for non-GUI use.  Audited state:

* **Window.**  Chosen in ``window_dropdown``, default the first entry,
  ``Hanning`` (``np.hanning``).  The window is built with the length of the
  *already mirrored* two-sided spectrum, so it runs from -fmax to +fmax and
  tapers DC as well as fmax.  Its ``corr`` field is a *divisor*, applied as
  ``fft_points / corr(...)``: ``len/2`` for Hanning, ``len/(1/0.42)`` for
  Blackman, and ``sum(kaiser(len, beta))`` for the Kaiser entries.  That
  correction only scales the ``step_response_Z`` trace of the
  ``Refl (lowpass)``/``Refl (bandpass)`` formats; the impulse response
  ``td`` that drives the peak search and the cable-length label is never
  corrected, so the label's amplitude is not calibrated.
* **Grid and padding.**  The GUI *assumes* a uniform sweep without checking
  it: ``step_size`` is the difference of the first two frequencies and every
  later sample is trusted to follow.  In lowpass formats the one-sided
  samples ``s11[0..n-1]`` are mirrored with their conjugates
  (``conj(s11[-1:0:-1])``) and ``fftshift``-ed to the band
  ``[-fmax .. +fmax]``.  The result is zero-padded to
  ``get_fft_points()`` -- 4096 with ``flag.draw_lines``, otherwise 16384 --
  with the extra 1 sample placed at the low end (``pad_points + 1`` /
  ``pad_points``) before ``ifftshift`` and ``ifft``.  Bandpass formats skip
  the mirror and transform the one-sided data directly, so only the
  magnitude of the impulse response is meaningful there.
* **Sample spacing.**  ``time_axis = linspace(0, 1/step_size, fft_points)``
  -- ``fft_points`` points spanning ``0 .. 1/step_size`` *inclusive* -- so
  the spacing is ``1 / ((fft_points - 1) * step_size)``.  The true circular
  spacing of an ``fft_points``-long record is ``1 / (fft_points * step_size)``,
  so every label runs ``fft_points / (fft_points - 1)`` too fast and the
  last one repeats ``t = 0`` at the end of the window.
* **Distance convention.**  ``distance_axis = time_axis * v * c`` and every
  reported length is ``distance_axis[i] / 2``: the axis is a **round-trip**
  distance and the number shown to the operator is the **one-way** length.
* **Velocity factor.**  ``v`` comes from ``tdr_velocity_input`` (a preset
  cable value or a custom entry) and enters *only* on the distance axis --
  the transform itself is frequency-domain and needs no ``v``.  A wrong
  ``v`` therefore moves the echo but does not change its shape.

The module below keeps the same physics and makes the bookkeeping exact:

* the mirrored, Hermitian spectrum is *not* truncated to a ``linspace``
  axis; the record length is ``L = 2 * n * pad_factor - 1`` samples at
  ``1 / (L * df)`` seconds, which is the period the circular transform
  actually has;
* the window is applied to the one-sided samples and carries over to the
  mirrored half unchanged, so Hermiticity -- and therefore a real impulse
  response and an exact inverse -- survives both windowing and padding;
* amplitudes are normalised by the two-sided window sum,
  ``S = 2 * sum(w) - w[0]``.  An echo of coefficient ``G`` that lands on a
  record sample therefore reads ``G`` for **every** window: the taper
  changes the sidelobes, not the height of an on-grid echo.  An echo that
  falls between record samples reads slightly less -- how much depends on
  the taper and on the number of *measured* bins, so no single factor is
  claimed here.  ``window_coherent_gain`` records ``mean(w)``, which
  describes the taper itself rather than the height of the recorded echo.

What the numbers mean
---------------------
A reflection of coefficient ``G`` at round-trip delay ``tau`` produces
``x[k] = G * d(t_k - tau)``: a real impulse **at a single positive delay**,
because the Hermitian mirror puts the negative-frequency content where it
belongs for a real system.  Time sample ``k`` is

    t_k        = k / (L * df)                     seconds
    d_rt(k)    = v * c * t_k                      metres, round trip
    d(k)       = v * c * t_k / 2                  metres, one-way

The unambiguous one-way range is half a period, ``v * c / (4 * df)``; beyond
it the record wraps to negative delays.  The gate is therefore defined on a
**folded** distance axis, ``d_fold(k) = v * c * min(t_k, T - t_k) / 2`` with
``T = 1 / df``, i.e. the record is read as ``-T/2 .. +T/2`` and the gate is
symmetric about zero delay.  That is what makes a gate spanning the whole
record an exact identity instead of a filter that eats half the data.

Refusals
--------
Nothing here guesses.  A sweep that is not uniform, has too few samples, or
does not start at DC is rejected with the reason, because the lowpass impulse
response does not exist for it.  Gates that are inverted, negative, entirely
outside the record, or wider in transition than the band they define are
rejected for the same reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, NamedTuple, Sequence

import numpy as np
import numpy.typing as npt

from .RFTools import Datapoint

#: Speed of light in vacuum, m/s.  Kept local so this module imports nothing
#: from Settings and can be used by tests and headless tooling.
C_LIGHT = 299_792_458.0

#: Fewest uniformly spaced samples a transform is attempted with.  Below this
#: the two window shoulders and the echo occupy the same bins, so the result
#: would be an artefact rather than a measurement.
MIN_SAMPLES = 16

#: A sweep "starts at DC" when its first frequency is closer to 0 Hz than
#: this fraction of a step.
DC_TOLERANCE_STEPS = 0.5

#: Relative deviation of a frequency step from the first step that still
#: counts as a uniform sweep.
UNIFORMITY_TOLERANCE = 1e-6

#: Smallest velocity factor accepted.  Zero would collapse the distance axis
#: and negative is not a dielectric.
MIN_VELOCITY_FACTOR = 0.0
MAX_VELOCITY_FACTOR = 1.0

#: Window functions available for the forward transform, by name.  The GUI
#: default is Hanning; the default here is rectangular so that the round trip
#: reproduces the measured samples instead of a tapered copy of them.
WINDOWS: dict[str, Callable[[int], npt.NDArray[np.float64]]] = {
    "rectangular": lambda n: np.ones(n, dtype=float),
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
}
DEFAULT_WINDOW = "rectangular"

#: The gate taper.  ``raised_cosine`` is a Hann transition of total width
#: ``edge`` centred on each gate boundary; ``rectangular`` is ``edge == 0``
#: and is only offered because an operator may ask for it, with the ringing
#: it causes documented and recorded.
SHAPE_RAISED_COSINE = "raised_cosine"
SHAPE_RECTANGULAR = "rectangular"

#: Stated once, recorded in every result, so a saved trace says what it is.
CONVENTION = (
    "lowpass impulse response: one-sided reflection samples are mirrored "
    "with their conjugates (Hermitian extension) and inverse-DFT'd over "
    "L = 2*n*pad-1 bins at 1/(L*df) seconds per sample; a reflection G at "
    "round-trip delay tau appears as a real impulse of amplitude G at a "
    "single positive delay; time maps to one-way distance as d = v*c*t/2 "
    "(round trip = 2*d), unambiguous one-way range v*c/(4*df), and gates "
    "are applied on the folded distance v*c*min(t, 1/df - t)/2"
)


class TDRGatingError(ValueError):
    """A transform or a gate cannot be applied to the supplied data.

    The message always names the reason; callers are expected to show it
    rather than to substitute a plausible number.
    """


class GateKind(Enum):
    """What a gate does to the samples inside its band."""

    #: Keep the band, reject everything outside it.
    KEEP = "keep"
    #: Reject the band, keep everything outside it.
    BANDSTOP = "bandstop"


class TimeSample(NamedTuple):
    """One time-domain sample with both axes made explicit."""

    #: Seconds since the reference plane, ``0 .. 1/df`` (circular).
    time: float
    #: One-way distance in metres, ``v * c * time / 2``.
    distance: float
    #: Impulse response amplitude, normalised so an on-grid echo of
    #: coefficient ``G`` reads ``G`` for any window.
    value: float


def _window_gain(window: Sequence[float]) -> tuple[float, float]:
    """Return ``(two_sided_sum, coherent_gain)`` for a one-sided window.

    ``two_sided_sum`` is the weight the mirrored spectrum puts on a single
    impulse: DC appears once, every other bin twice.  Dividing by it makes
    an on-grid echo of coefficient ``G`` read exactly ``G`` whatever the
    taper.  ``coherent_gain`` is ``sum(w) / n``, a property of the window
    itself, recorded for provenance rather than applied to amplitudes.
    """
    weights = np.asarray(window, dtype=float)
    two_sided = 2.0 * float(weights.sum()) - float(weights[0])
    return two_sided, float(weights.mean())


def _validate_sweep(
    points: Sequence[Datapoint],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128], float]:
    """Check a one-port sweep is usable and return frequencies, values, df."""
    count = len(points)
    if count < MIN_SAMPLES:
        raise TDRGatingError(
            f"{count} samples is fewer than the {MIN_SAMPLES} required for a "
            "time-domain transform"
        )
    freqs = np.array([float(point.freq) for point in points])
    values = np.array([point.z for point in points], dtype=complex)
    if not np.all(np.isfinite(freqs)):
        raise TDRGatingError("frequency axis contains a non-finite value")
    if not np.all(np.isfinite(values.real)) or not np.all(
        np.isfinite(values.imag)
    ):
        raise TDRGatingError("reflection samples contain a non-finite value")
    step = freqs[1] - freqs[0]
    if step <= 0:
        raise TDRGatingError(
            f"sweep is not strictly increasing (first step {step:g} Hz)"
        )
    diffs = np.diff(freqs)
    if not np.all(diffs > 0):
        raise TDRGatingError(
            "sweep is not strictly increasing: the frequency axis repeats or "
            "goes backwards"
        )
    worst = float(np.abs(diffs - step).max())
    if worst > UNIFORMITY_TOLERANCE * step:
        raise TDRGatingError(
            "sweep is not uniformly spaced: largest step deviation "
            f"{worst:g} Hz against a {step:g} Hz step"
        )
    start = float(freqs[0])
    if abs(start) > DC_TOLERANCE_STEPS * step:
        raise TDRGatingError(
            "sweep does not start at DC: first sample is "
            f"{start:g} Hz, tolerance {DC_TOLERANCE_STEPS * step:g} Hz; the "
            "lowpass impulse response needs a DC sample"
        )
    return freqs, values, float(step)


def _validate_transform(*, window: str, pad_factor: int, velocity: float):
    if window not in WINDOWS:
        raise TDRGatingError(
            f"unknown window {window!r}; supported: "
            f"{', '.join(sorted(WINDOWS))}"
        )
    if isinstance(pad_factor, bool) or not isinstance(pad_factor, int):
        raise TDRGatingError(
            f"pad_factor must be an integer, got {pad_factor!r}"
        )
    if pad_factor < 1:
        raise TDRGatingError(f"pad_factor must be at least 1, got {pad_factor}")
    if not math.isfinite(velocity):
        raise TDRGatingError("velocity factor is not finite")
    if not MIN_VELOCITY_FACTOR < velocity <= MAX_VELOCITY_FACTOR:
        raise TDRGatingError(
            f"velocity factor {velocity:g} is outside (0, 1]; it scales the "
            "distance axis and cannot be zero or negative"
        )


@dataclass(frozen=True)
class TimeDomainResult:
    """An impulse response plus the record of how it was produced."""

    #: Channel the samples came from, e.g. ``"S11"``.
    channel: str
    #: Name of the taper applied before the transform (see :data:`WINDOWS`).
    window: str
    #: Zero padding factor; the record has ``2 * input_samples * pad - 1``
    #: samples.
    pad_factor: int
    #: Number of measured samples that went in.
    input_samples: int
    #: Frequencies of the measured samples, in Hz, for the inverse transform.
    frequencies: tuple[int, ...]
    #: Frequency step of the sweep, Hz.
    frequency_step: float
    #: Time between samples of the record, seconds.
    time_step: float
    #: One-way distance between samples, metres.
    distance_step: float
    #: Longest one-way distance the record resolves, metres.
    max_distance: float
    #: Velocity factor used on the distance axis.
    velocity_factor: float
    #: ``mean(w)`` of the taper, for provenance.  It does **not** scale the
    #: recorded echo, which stays at its true coefficient on the grid.
    window_coherent_gain: float
    #: The transform convention, spelled out.
    convention: str
    #: The full circular record, ``2 * input_samples * pad_factor - 1`` real
    #: samples in seconds ``0 .. 1/frequency_step - time_step``.
    values: tuple[float, ...]

    @property
    def sample_count(self) -> int:
        """Number of samples in the circular record."""
        return len(self.values)

    @property
    def samples(self) -> tuple[TimeSample, ...]:
        """The physical half of the record: one-way distances only."""
        return _samples_of(self.values, self.time_step, self.velocity_factor)

    @property
    def peak(self) -> TimeSample | None:
        """Largest-magnitude sample of the physical half, or ``None``."""
        samples = self.samples
        if not samples:
            return None
        return max(samples, key=lambda sample: abs(sample.value))

    def provenance(self) -> dict[str, object]:
        """Everything a label or a stored trace needs to describe itself."""
        return {
            "channel": self.channel,
            "transform": "tdr_impulse_lowpass",
            "window": self.window,
            "pad_factor": self.pad_factor,
            "input_samples": self.input_samples,
            "record_samples": self.sample_count,
            "frequency_step": self.frequency_step,
            "time_step": self.time_step,
            "distance_step": self.distance_step,
            "max_distance": self.max_distance,
            "velocity_factor": self.velocity_factor,
            "window_coherent_gain": self.window_coherent_gain,
            "derived": True,
            "convention": self.convention,
        }


def _samples_of(
    values: Sequence[float], time_step: float, velocity: float
) -> tuple[TimeSample, ...]:
    """Physical half of a record, as :class:`TimeSample` triples."""
    half = len(values) // 2 + 1
    scale = 0.5 * velocity * C_LIGHT
    return tuple(
        TimeSample(
            time=index * time_step,
            distance=index * time_step * scale,
            value=float(values[index]),
        )
        for index in range(min(half, len(values)))
    )


def to_time_domain(
    points: Sequence[Datapoint],
    *,
    channel: str = "S11",
    window: str = DEFAULT_WINDOW,
    pad_factor: int = 1,
    velocity_factor: float = 1.0,
) -> TimeDomainResult:
    """Transform a one-port sweep starting at DC into an impulse response.

    ``points`` must be uniformly spaced, start at (or within half a step of)
    DC and contain at least :data:`MIN_SAMPLES` samples.  ``pad_factor``
    zero-pads the spectrum to ``pad_factor * len(points)`` one-sided bins,
    which only refines the time axis; it does not add information.

    Raises :class:`TDRGatingError` naming the reason for anything else.
    """
    _validate_transform(
        window=window, pad_factor=pad_factor, velocity=velocity_factor
    )
    freqs, values, step = _validate_sweep(points)
    count = len(values)

    weights = np.asarray(WINDOWS[window](count), dtype=float)
    one_sided = np.zeros(count * pad_factor, dtype=complex)
    one_sided[:count] = values * weights

    length = 2 * count * pad_factor - 1
    spectrum = np.zeros(length, dtype=complex)
    spectrum[: count * pad_factor] = one_sided
    # Bin ``length - m`` is frequency ``-m * df``: the conjugate of bin ``m``.
    if count * pad_factor >= 2:
        spectrum[length - count * pad_factor + 1 :] = np.conj(one_sided[1:])[
            ::-1
        ]

    two_sided, gain = _window_gain(weights)
    record = np.fft.ifft(spectrum).real * (length / two_sided)

    time_step = 1.0 / (length * step)
    distance_step = 0.5 * velocity_factor * C_LIGHT * time_step
    return TimeDomainResult(
        channel=channel,
        window=window,
        pad_factor=pad_factor,
        input_samples=count,
        frequencies=tuple(int(freq) for freq in freqs),
        frequency_step=step,
        time_step=time_step,
        distance_step=distance_step,
        max_distance=velocity_factor * C_LIGHT / (4.0 * step),
        velocity_factor=velocity_factor,
        window_coherent_gain=gain,
        convention=CONVENTION,
        values=tuple(float(value) for value in record),
    )


def _gate_weight(distance: float, start: float, stop: float, edge: float):
    """Raised-cosine keep weight at ``distance``; 0 outside, 1 well inside."""
    if edge <= 0.0:
        return 1.0 if start <= distance <= stop else 0.0
    half = 0.5 * edge
    if distance < start - half or distance > stop + half:
        return 0.0
    if start + half <= distance <= stop - half:
        return 1.0
    if distance < start + half:
        return 0.5 * (
            1.0 - math.cos(math.pi * (distance - start + half) / edge)
        )
    return 0.5 * (1.0 - math.cos(math.pi * (stop + half - distance) / edge))


@dataclass(frozen=True)
class GatedResult:
    """A gated impulse response: still an impulse response, no longer raw."""

    #: The transform the gate was applied to.
    source: TimeDomainResult
    #: Keep band or bandstop.
    kind: GateKind
    #: Lower one-way distance of the gate band, metres.
    start_distance: float
    #: Upper one-way distance of the gate band, metres.
    stop_distance: float
    #: Total width of the raised-cosine transition, metres (0 = rectangular).
    edge: float
    #: Name of the taper shape (see :data:`SHAPE_RAISED_COSINE`).
    shape: str
    #: The multiplier applied to each sample of the circular record.
    gate: tuple[float, ...]
    #: The gated circular record, same layout as
    #: :attr:`TimeDomainResult.values`.
    values: tuple[float, ...]

    @property
    def derived(self) -> bool:
        """Always true: a gated trace is processed, never a measurement."""
        return True

    @property
    def is_identity(self) -> bool:
        """True when the gate left every sample untouched."""
        return all(weight == 1.0 for weight in self.gate)

    @property
    def samples(self) -> tuple[TimeSample, ...]:
        """Gated samples over the physical half of the record."""
        return _samples_of(
            self.values, self.source.time_step, self.source.velocity_factor
        )

    @property
    def label(self) -> str:
        """Short description fit for a trace legend."""
        if self.is_identity:
            return f"{self.source.channel} gated (identity)"
        return (
            f"{self.source.channel} gated ({self.kind.value} "
            f"{self.start_distance:.2f}-{self.stop_distance:.2f} m, "
            f"{self.shape} {self.edge:.2f} m)"
        )

    def provenance(self) -> dict[str, object]:
        """Gate parameters plus the transform they were applied to."""
        record = dict(self.source.provenance())
        record.update(
            {
                "gate_kind": self.kind.value,
                "gate_start_distance": self.start_distance,
                "gate_stop_distance": self.stop_distance,
                "gate_edge": self.edge,
                "gate_shape": self.shape,
                "gate_identity": self.is_identity,
                "derived": True,
                "label": self.label,
            }
        )
        return record


def _validate_gate(
    source: TimeDomainResult,
    start: float,
    stop: float,
    edge: float,
) -> None:
    if not all(math.isfinite(value) for value in (start, stop, edge)):
        raise TDRGatingError("gate bounds must be finite")
    if start < 0.0:
        raise TDRGatingError(
            f"gate start {start:g} m is negative; distances are one-way and "
            "start at the reference plane"
        )
    if start >= stop:
        raise TDRGatingError(
            f"gate start {start:g} m is not below gate stop {stop:g} m"
        )
    if edge < 0.0:
        raise TDRGatingError(f"gate edge width {edge:g} m is negative")
    if edge > stop - start:
        raise TDRGatingError(
            f"gate edge width {edge:g} m is wider than the "
            f"{stop - start:g} m gate band"
        )
    if start >= source.max_distance:
        raise TDRGatingError(
            f"gate at {start:g} m is outside the record; the unambiguous "
            f"one-way range ends at {source.max_distance:g} m"
        )
    if stop > source.max_distance * (1.0 + 1e-12):
        raise TDRGatingError(
            f"gate stop {stop:g} m is past the end of the record at "
            f"{source.max_distance:g} m; a gate cannot reach data the "
            "transform does not hold"
        )


def apply_gate(
    time_result: TimeDomainResult,
    *,
    kind: GateKind | str = GateKind.KEEP,
    start_distance: float = 0.0,
    stop_distance: float | None = None,
    edge: float = 0.0,
) -> GatedResult:
    """Multiply the record by a band gate with a stated transition.

    The gate is evaluated on the folded distance
    ``d(k) = v * c * min(t_k, 1/df - t_k) / 2``, so it is symmetric about
    zero delay and a gate covering ``0 .. max_distance`` is an exact
    identity.  ``edge`` is the **total** width of the raised-cosine
    transition, centred on each boundary: the weight is 1 for
    ``start + edge/2 <= d <= stop - edge/2``, 0 beyond
    ``start - edge/2`` and ``stop + edge/2``, and a Hann taper in between.
    ``edge == 0`` gives a rectangular gate, which rings: the transform of a
    rectangular window has ``-13 dB`` first sidelobes that smear the removed
    echo across the whole frequency trace.

    With ``kind=GateKind.BANDSTOP`` the weight is ``1 - weight``: the band is
    removed and everything else is kept.
    """
    if isinstance(kind, str):
        try:
            kind = GateKind(kind)
        except ValueError as error:
            raise TDRGatingError(
                f"unknown gate kind {kind!r}; use one of "
                f"{', '.join(item.value for item in GateKind)}"
            ) from error
    elif not isinstance(kind, GateKind):
        raise TDRGatingError(f"gate kind must be a GateKind, got {kind!r}")

    limit = time_result.max_distance
    stop = limit if stop_distance is None else float(stop_distance)
    start = float(start_distance)
    width = float(edge)
    _validate_gate(time_result, start, stop, width)

    shape = SHAPE_RECTANGULAR if width <= 0.0 else SHAPE_RAISED_COSINE
    period = 1.0 / time_result.frequency_step
    scale = 0.5 * time_result.velocity_factor * C_LIGHT
    step = time_result.time_step

    weights = []
    for index in range(len(time_result.values)):
        elapsed = index * step
        distance = scale * min(elapsed, period - elapsed)
        weight = _gate_weight(distance, start, stop, width)
        if kind is GateKind.BANDSTOP:
            weight = 1.0 - weight
        weights.append(weight)

    gated = tuple(
        value * weight
        for value, weight in zip(time_result.values, weights, strict=True)
    )
    return GatedResult(
        source=time_result,
        kind=kind,
        start_distance=start,
        stop_distance=stop,
        edge=width,
        shape=shape,
        gate=tuple(weights),
        values=gated,
    )


def to_frequency_domain(gated: GatedResult) -> tuple[Datapoint, ...]:
    """Inverse of :func:`to_time_domain` for a gated record.

    The result is a **processed** trace: it is what the gated impulse
    response implies at the original sweep frequencies, not a measurement.
    With an identity gate and ``window="rectangular"`` it reproduces the
    input samples exactly (to floating-point rounding); with a taper it
    reproduces the tapered samples, which is why the window is recorded in
    the provenance of both results.
    """
    source = gated.source
    count = source.input_samples
    two_sided, _gain = _window_gain(
        np.asarray(WINDOWS[source.window](count), dtype=float)
    )
    length = len(gated.values)
    spectrum = (
        np.fft.fft(np.asarray(gated.values, dtype=float)) * two_sided / length
    )
    return tuple(
        Datapoint(
            freq, float(spectrum[index].real), float(spectrum[index].imag)
        )
        for index, freq in enumerate(source.frequencies)
    )


def gate_points(  # noqa: PLR0913
    points: Iterable[Datapoint],
    *,
    channel: str = "S11",
    window: str = DEFAULT_WINDOW,
    pad_factor: int = 1,
    velocity_factor: float = 1.0,
    kind: GateKind | str = GateKind.KEEP,
    start_distance: float = 0.0,
    stop_distance: float | None = None,
    edge: float = 0.0,
) -> tuple[Datapoint, ...]:
    """Transform, gate and return, keeping the provenance on the side.

    Convenience for callers that only need the trace; use
    :func:`to_time_domain` and :func:`apply_gate` directly when the gate
    parameters are to be shown alongside it.
    """
    time_result = to_time_domain(
        list(points),
        channel=channel,
        window=window,
        pad_factor=pad_factor,
        velocity_factor=velocity_factor,
    )
    return to_frequency_domain(
        apply_gate(
            time_result,
            kind=kind,
            start_distance=start_distance,
            stop_distance=stop_distance,
            edge=edge,
        )
    )
