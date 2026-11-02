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
"""Explicit, immutable trace-processing pipeline (DSP-01).

Acquisition produces raw samples; calibration turns them into corrected
samples.  Everything that happens *after* that used to be ad hoc: the offset
delay was applied inside the worker, and any further transformation was
computed on demand by whichever window needed it, without a record of what
had been done to the numbers on screen.

This module defines that missing contract:

    raw capture  ->  calibration  ->  fixture/port  ->  averaging  ->  smoothing

Only stages from ``fixture/port`` onwards live here; raw capture and
calibration stay in :mod:`NanoVNASaver.SweepWorker`, which owns the device
conversation and freezes the calibration at the run boundary.

Design rules
------------
* A :class:`ProcessingConfig` is a frozen, ordered tuple of stages.  It is
  the *description* of a processing pipeline; it never holds mutable state.
* Applying a configuration is a pure function over frequency-indexed
  channels.  It returns new tuples and never mutates its input, so a frozen
  snapshot stays frozen.
* ``ProcessingConfig.identity()`` is a stable digest of the stage list.  It
  is what provenance records, so "these numbers were smoothed over 11
  points" is auditable months later.
* Processing never changes the frequency grid and never invents samples.  A
  stage that cannot be evaluated for a channel raises
  :class:`ProcessingError` instead of quietly returning different data.
* Nothing here imports Qt or :mod:`NanoVNASaver.Measurement`, so both the
  acquisition side and the model side can use it without an import cycle.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

from .RFTools import Datapoint

#: Format version of the processing description itself.  Bumped only when a
#: stored configuration would be interpreted differently by a newer build.
PROCESSING_FORMAT_VERSION = 1

#: Stage kinds in the order they are applied.
STAGE_PORT_EXTENSION = "port_extension"
STAGE_FIXTURE = "fixture"
STAGE_Z0 = "renormalize"
STAGE_AVERAGE = "running_average"
STAGE_SMOOTHING = "smoothing"

#: Canonical application order.  A configuration may omit stages but never
#: reorders them: the order is part of the meaning of the pipeline.
STAGE_ORDER: tuple[str, ...] = (
    STAGE_PORT_EXTENSION,
    STAGE_FIXTURE,
    STAGE_Z0,
    STAGE_AVERAGE,
    STAGE_SMOOTHING,
)

STAGE_LABELS = {
    STAGE_PORT_EXTENSION: "Port extension",
    STAGE_FIXTURE: "Fixture de-embedding",
    STAGE_Z0: "Reference impedance",
    STAGE_AVERAGE: "Running average",
    STAGE_SMOOTHING: "Frequency smoothing",
}

#: One-port channels: the reflection coefficient is the whole measurement.
PORT1_CHANNELS = ("S11", "S22")
#: Forward transmission channels.
FORWARD_CHANNELS = ("S21",)
#: Reverse transmission channels (only present after a reversal capture).
REVERSE_CHANNELS = ("S12",)

#: Speed of light, kept local so this module has no Settings dependency.
C = 299_792_458.0

#: Running-average weighting modes (DSP-03).  ``cumulative`` is the exact
#: arithmetic mean of every sweep blended so far; ``exponential`` is the
#: EWMA recurrence with a fixed weight ``alpha``.
AVERAGE_CUMULATIVE = "cumulative"
AVERAGE_EXPONENTIAL = "exponential"
AVERAGE_MODES: tuple[str, ...] = (AVERAGE_CUMULATIVE, AVERAGE_EXPONENTIAL)

#: Ground rules of the smoothing stage (DSP-02), stated once so the engine,
#: the UI and the documentation cannot drift apart.
#:
#: * the aperture is an odd number of samples;
#: * a window wider than the data is clamped to the data, never padded;
#: * windows at the ends shrink to the samples that exist;
#: * smoothing is a genuinely different measurement, not a better one.

#: Sentinel for "the stage does not carry this parameter at all", which is
#: different from carrying a value of zero.
_MISSING = object()


class ProcessingError(ValueError):
    """A processing stage cannot be applied to the supplied data."""


def _freeze(value):
    """Recursively make ``value`` hashable and immutable."""
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _freeze(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    """Inverse of :func:`_freeze` for JSON output."""
    if isinstance(value, tuple):
        if (
            all(isinstance(item, tuple) and len(item) == 2 for item in value)
            and value
        ):
            return {str(key): _thaw(item) for key, item in value}
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ProcessingStage:
    """One configured processing stage.

    ``parameters`` is a frozen mapping: it is stored as a sorted tuple of
    ``(key, value)`` pairs so the stage is hashable, comparable and cannot be
    modified after construction.
    """

    kind: str
    parameters: tuple = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.kind not in STAGE_ORDER:
            raise ProcessingError(
                f"unknown processing stage {self.kind!r};"
                f" known stages: {', '.join(STAGE_ORDER)}"
            )
        if not isinstance(self.parameters, tuple):
            object.__setattr__(self, "parameters", _freeze(self.parameters))

    @classmethod
    def build(
        cls,
        kind: str,
        parameters: Mapping | None = None,
        *,
        enabled: bool = True,
    ) -> "ProcessingStage":
        return cls(
            kind=kind, parameters=_freeze(parameters or {}), enabled=enabled
        )

    # -- mapping access ------------------------------------------------
    @property
    def as_dict(self) -> dict:
        return {str(key): value for key, value in self.parameters}

    def get(self, key: str, default=None):
        for name, value in self.parameters:
            if name == key:
                return value
        return default

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "enabled": self.enabled,
            "parameters": {
                str(key): _thaw(value) for key, value in self.parameters
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "ProcessingStage":
        if not isinstance(data, Mapping):
            raise ProcessingError("a processing stage must be an object")
        kind = str(data.get("kind", ""))
        parameters = data.get("parameters") or {}
        if not isinstance(parameters, Mapping):
            raise ProcessingError(
                f"parameters of stage {kind!r} must be an object"
            )
        return cls.build(
            kind, parameters, enabled=bool(data.get("enabled", True))
        )

    def describe(self) -> str:
        """One-line, human readable description for UI and reports."""
        label = STAGE_LABELS.get(self.kind, self.kind)
        values = self.as_dict
        if self.kind == STAGE_SMOOTHING:
            aperture = values.get("aperture_points")
            if aperture is not None:
                return f"{label}: {aperture} points (in samples)"
        if self.kind == STAGE_AVERAGE:
            mode = values.get("mode", AVERAGE_CUMULATIVE)
            alpha = values.get("alpha")
            count = values.get("effective_count")
            if mode == AVERAGE_EXPONENTIAL and alpha is not None:
                text = f"{label}: exponential alpha {alpha:g}"
            else:
                text = f"{label}: cumulative"
            if count:
                return f"{text}, {count} sweep(s)"
            return text

        if self.kind == STAGE_Z0:
            reference = values.get("reference_impedance")
            if reference is not None:
                return f"{label}: {reference:g} ohm"
        if self.kind == STAGE_PORT_EXTENSION:
            parts = []
            for port, key in ((1, "port1_delay_ps"), (2, "port2_delay_ps")):
                delay = values.get(key)
                if delay:
                    parts.append(f"port {port} {delay:g} ps")
            for port, key in (
                (1, "port1_loss_db_per_ghz"),
                (2, "port2_loss_db_per_ghz"),
            ):
                loss = values.get(key)
                if loss:
                    parts.append(f"port {port} {loss:g} dB/GHz")
            if parts:
                return f"{label}: " + ", ".join(parts)
        if self.kind == STAGE_FIXTURE:
            name = values.get("name", "")
            digest = values.get("identity", "")
            if name:
                return f"{label}: {name}" + (
                    f" ({digest[:8]})" if digest else ""
                )
        return label


@dataclass(frozen=True)
class ProcessingConfig:
    """An ordered, immutable description of post-calibration processing."""

    stages: tuple[ProcessingStage, ...] = ()
    format_version: int = PROCESSING_FORMAT_VERSION

    def __post_init__(self) -> None:
        stages = tuple(stage for stage in self.stages)
        for stage in stages:
            if not isinstance(stage, ProcessingStage):
                raise ProcessingError("stages must be ProcessingStage objects")
        names = [stage.kind for stage in stages if stage.enabled]
        if len(set(names)) != len(names):
            duplicate = next(name for name in names if names.count(name) > 1)
            raise ProcessingError(
                f"processing stage {duplicate!r} is configured twice"
            )
        ordered = sorted(
            (stage for stage in stages if stage.enabled),
            key=lambda stage: STAGE_ORDER.index(stage.kind),
        )
        if [stage.kind for stage in ordered] != names:
            raise ProcessingError(
                "processing stages must be supplied in application order:"
                f" {', '.join(STAGE_ORDER)}"
            )
        object.__setattr__(self, "stages", stages)

    # -- construction --------------------------------------------------
    @classmethod
    def build(
        cls,
        *stages: ProcessingStage,
        format_version: int = PROCESSING_FORMAT_VERSION,
    ) -> "ProcessingConfig":
        return cls(stages=tuple(stages), format_version=format_version)

    @property
    def active_stages(self) -> tuple[ProcessingStage, ...]:
        return tuple(stage for stage in self.stages if stage.enabled)

    @property
    def is_identity(self) -> bool:
        """True when this configuration changes nothing."""
        return not self.active_stages

    def find(self, kind: str) -> ProcessingStage | None:
        for stage in self.stages:
            if stage.kind == kind and stage.enabled:
                return stage
        return None

    def with_stage(
        self, stage: ProcessingStage | None, kind: str | None = None
    ):
        """Return a configuration with ``stage`` set or removed.

        ``kind`` names the stage to replace when ``stage`` is ``None``
        (removal).  When ``stage`` is given its own ``kind`` is used.
        """
        target = stage.kind if stage is not None else kind
        if target is None:
            raise ProcessingError("with_stage needs a stage or a kind")
        kept = [
            existing
            for existing in self.stages
            if not (existing.kind == target and existing.enabled)
        ]
        if stage is not None and stage.enabled:
            kept.append(stage)
        kept.sort(key=lambda item: STAGE_ORDER.index(item.kind))
        return ProcessingConfig(
            stages=tuple(kept), format_version=self.format_version
        )

    # -- identity ------------------------------------------------------
    def describe(self) -> str:
        stages = self.active_stages
        if not stages:
            return "Raw (no processing)"
        return "; ".join(stage.describe() for stage in stages)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(stage.describe() for stage in self.active_stages)

    def identity(self) -> str:
        """Stable digest of the active stages and their parameters."""
        if self.is_identity:
            return ""
        payload = json.dumps(
            [stage.to_dict() for stage in self.active_stages],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    # -- serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "ProcessingConfig":
        if not isinstance(data, Mapping):
            raise ProcessingError("processing configuration must be an object")
        version = int(data.get("format_version", PROCESSING_FORMAT_VERSION))
        if version > PROCESSING_FORMAT_VERSION:
            raise ProcessingError(
                f"processing format version {version} is newer than supported"
                f" version {PROCESSING_FORMAT_VERSION}"
            )
        stages = data.get("stages") or []
        if not isinstance(stages, Sequence):
            raise ProcessingError("processing stages must be a list")
        return cls(
            stages=tuple(ProcessingStage.from_dict(stage) for stage in stages),
            format_version=version,
        )


# ----------------------------------------------------------------------
# Channel helpers
# ----------------------------------------------------------------------
def channel_frequencies(points: Sequence[Datapoint]) -> tuple[int, ...]:
    return tuple(point.freq for point in points)


def _require_finite(points: Sequence[Datapoint], what: str) -> None:
    for point in points:
        if not (math.isfinite(point.re) and math.isfinite(point.im)):
            raise ProcessingError(
                f"{what} contains a non-finite sample at {point.freq} Hz"
            )


def _number(
    stage: ProcessingStage,
    key: str,
    *,
    default: float | None = None,
    minimum: float | None = None,
) -> float:
    """Read one numeric stage parameter, refusing invalid values.

    ``dict.get(key) or default`` is not usable here: it turns a legitimate
    ``0`` into the default and it silently repairs an invalid value that a
    stored description declared.  A parameter that is present must parse as
    a finite number inside its stated range.  A parameter that is absent is
    only allowed when ``default`` is given; otherwise the stage is missing
    something it needs and is refused.
    """
    raw = stage.get(key, _MISSING)
    if raw is _MISSING:
        if default is None:
            raise ProcessingError(f"stage {stage.kind!r} needs {key!r}")
        return default
    if isinstance(raw, bool):
        raise ProcessingError(
            f"{stage.kind}: {key} must be a number, not {raw!r}"
        )
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ProcessingError(
            f"{stage.kind}: {key} must be a number, not {raw!r}"
        ) from error
    if not math.isfinite(value):
        raise ProcessingError(f"{stage.kind}: {key} must be finite")
    if minimum is not None and value < minimum:
        raise ProcessingError(f"{stage.kind}: {key} must be >= {minimum:g}")
    return value


def _mapped(
    points: Iterable[Datapoint], values: Sequence[complex]
) -> tuple[Datapoint, ...]:
    return tuple(
        Datapoint(point.freq, value.real, value.imag)
        for point, value in zip(points, values, strict=True)
    )


# ----------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------
def port_extension_factor(
    freq: int,
    *,
    delay_seconds: float = 0.0,
    loss_db_per_ghz: float = 0.0,
    doubled: bool = True,
) -> complex:
    """Complex factor that removes a length of line from a measurement.

    A port extension describes the fixture between the reference plane and
    the device under test.  Removing it means rotating the measurement by
    ``+2*pi*f*delay`` (the signal travels there and back, hence the ``2``
    for a reflection) and dividing out the loss.

    Signs are stated once here and used everywhere: a positive delay rotates
    by a *positive* phase, so a through connection measured through a
    longer line comes back looking shorter.
    """
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ProcessingError("port extension delay must be finite and >= 0")
    if not math.isfinite(loss_db_per_ghz) or loss_db_per_ghz < 0:
        raise ProcessingError("port extension loss must be finite and >= 0")
    # A reflection passes through the extension twice (there and back), so
    # both its delay and its loss are doubled; a transmission passes once.
    # Applying the loss only once understates a fixture's round-trip loss by
    # exactly the loss per GHz, which silently leaves part of the fixture in
    # the measurement: a 1 dB/GHz extension recovers 0.4456 instead of 0.5.
    passes = 2.0 if doubled else 1.0
    phase = complex(
        math.cos(math.tau * freq * delay_seconds * passes),
        math.sin(math.tau * freq * delay_seconds * passes),
    )
    if loss_db_per_ghz:
        gain_db = loss_db_per_ghz * (freq / 1e9) * passes
        phase *= 10.0 ** (gain_db / 20.0)
    return phase


def apply_port_extension(
    channels: Mapping[str, Sequence[Datapoint]],
    stage: ProcessingStage,
) -> dict[str, tuple[Datapoint, ...]]:
    """Remove per-port delay/loss from every channel it applies to."""
    delays = {
        1: _number(stage, "port1_delay_ps", default=0.0, minimum=0.0) * 1e-12,
        2: _number(stage, "port2_delay_ps", default=0.0, minimum=0.0) * 1e-12,
    }
    losses = {
        1: _number(stage, "port1_loss_db_per_ghz", default=0.0, minimum=0.0),
        2: _number(stage, "port2_loss_db_per_ghz", default=0.0, minimum=0.0),
    }
    for delays_seconds in delays.values():
        if delays_seconds < 0 or not math.isfinite(delays_seconds):
            raise ProcessingError("port extension delay must be >= 0")
    # A reflection travels through its port twice; a transmission through
    # both ports once each.  Doubling is inherent to the geometry, not a
    # user option.
    result: dict[str, tuple[Datapoint, ...]] = {}
    for name, points in channels.items():
        if not points:
            continue
        out = []
        for point in points:
            if name in PORT1_CHANNELS:
                port = 1
                factor = port_extension_factor(
                    point.freq,
                    delay_seconds=delays[port],
                    loss_db_per_ghz=losses[port],
                    doubled=True,
                )
            elif name in FORWARD_CHANNELS:
                factor = port_extension_factor(
                    point.freq,
                    delay_seconds=delays[1],
                    loss_db_per_ghz=losses[1],
                    doubled=False,
                ) * port_extension_factor(
                    point.freq,
                    delay_seconds=delays[2],
                    loss_db_per_ghz=losses[2],
                    doubled=False,
                )
            elif name in REVERSE_CHANNELS:
                factor = port_extension_factor(
                    point.freq,
                    delay_seconds=delays[2],
                    loss_db_per_ghz=losses[2],
                    doubled=False,
                ) * port_extension_factor(
                    point.freq,
                    delay_seconds=delays[1],
                    loss_db_per_ghz=losses[1],
                    doubled=False,
                )
            else:
                factor = complex(1.0, 0.0)
            value = point.z * factor
            out.append(Datapoint(point.freq, value.real, value.imag))
        result[name] = tuple(out)
    return result


def renormalize_gamma(z_old: float, z_new: float, gamma: complex) -> complex:
    """Renormalize a one-port reflection coefficient to a new reference.

    Derived from ``Z = Zref*(1+G)/(1-G)``; with
    ``a = (z_old - z_new)/(z_old + z_new)`` the result is

        G_new = (G + a) / (1 + a*G)

    which is exact for real, positive reference impedances.
    """
    if not (math.isfinite(z_old) and math.isfinite(z_new)):
        raise ProcessingError("reference impedances must be finite")
    if z_old <= 0 or z_new <= 0:
        raise ProcessingError("reference impedances must be positive")
    if z_old == z_new:
        return gamma
    a = (z_old - z_new) / (z_old + z_new)
    divisor = 1.0 + a * gamma
    if divisor == 0:
        raise ProcessingError(
            "renormalization is singular for this reflection coefficient"
        )
    return (gamma + a) / divisor


def apply_renormalize(
    channels: Mapping[str, Sequence[Datapoint]],
    stage: ProcessingStage,
    *,
    current_reference: float,
) -> dict[str, tuple[Datapoint, ...]]:
    """Renormalize one-port channels to the configured reference impedance.

    Only reflection channels are renormalized.  Transmission coefficients of
    a two-port measured with unequal port impedances need the full
    wave-cascading transform, so they are refused rather than approximated.
    """
    target = _number(stage, "reference_impedance")
    if target <= 0:
        raise ProcessingError(
            "reference impedance must be a positive finite number"
        )
    if target == current_reference:
        return {name: tuple(points) for name, points in channels.items()}
    if any(
        name in FORWARD_CHANNELS or name in REVERSE_CHANNELS
        for name in channels
    ):
        raise ProcessingError(
            "renormalizing a transmission channel is not supported yet;"
            " only reflection data can be renormalized"
        )
    result: dict[str, tuple[Datapoint, ...]] = {}
    for name, points in channels.items():
        if name not in PORT1_CHANNELS or not points:
            result[name] = tuple(points)
            continue
        out = []
        for point in points:
            value = renormalize_gamma(current_reference, target, point.z)
            out.append(Datapoint(point.freq, value.real, value.imag))
        result[name] = tuple(out)
    return result


def validate_smoothing_aperture(aperture_points) -> int:
    """Return ``aperture_points`` as a positive odd ``int``.

    The aperture arrives from a stored processing description, so it is
    validated here rather than coerced: ``0``, ``None``, ``2`` or ``"wide"``
    describe something that cannot be honoured, and silently substituting
    another aperture would make the recorded provenance lie about the
    numbers on screen.
    """
    if aperture_points is None or isinstance(aperture_points, bool):
        raise ProcessingError(
            "smoothing aperture must be a positive odd integer,"
            f" not {aperture_points!r}"
        )
    if isinstance(aperture_points, float):
        if not math.isfinite(aperture_points) or aperture_points != int(
            aperture_points
        ):
            raise ProcessingError(
                "smoothing aperture must be a whole number of points,"
                f" not {aperture_points!r}"
            )
        aperture = int(aperture_points)
    elif isinstance(aperture_points, int):
        aperture = aperture_points
    else:
        raise ProcessingError(
            "smoothing aperture must be a positive odd integer,"
            f" not {type(aperture_points).__name__}"
        )
    if aperture < 1:
        raise ProcessingError("smoothing aperture must be at least 1 point")
    if aperture % 2 == 0:
        raise ProcessingError(
            "smoothing aperture must be an odd point count"
            " so that every window is centred"
        )
    return aperture


def effective_smoothing_aperture(samples: int, aperture_points) -> int:
    """Aperture actually applied to a channel of ``samples`` samples.

    An aperture wider than the channel is clamped to the largest odd count
    the data can supply.  It is never padded, so the reported aperture is
    always the one that was used.  A channel of one or two samples can only
    support an aperture of 1, which means "no smoothing": the samples come
    back unchanged and the caller can say so, instead of pretending a wider
    window was applied.
    """
    aperture = validate_smoothing_aperture(aperture_points)
    if samples <= 1:
        return 1
    largest = samples if samples % 2 else samples - 1
    return min(aperture, largest)


def channel_is_uniform(points: Sequence[Datapoint]) -> bool:
    """True when consecutive samples are equally spaced in frequency.

    The smoothing aperture counts *samples*, so its frequency width only
    equals ``aperture * step`` on a uniform grid.  On a logarithmic or
    otherwise irregular grid the width varies across the sweep; that is a
    property of the measurement grid, not an error, and it is reported
    rather than hidden.
    """
    if len(points) < 3:
        return True
    first = points[1].freq - points[0].freq
    return all(
        later.freq - earlier.freq == first
        for earlier, later in itertools.pairwise(points)
    )


def _smoothing_window_sums(
    points: Sequence[Datapoint], aperture: int
) -> tuple[Datapoint, ...]:
    """Centred complex moving average in O(n) time and memory."""
    prefix = [0j]
    for point in points:
        prefix.append(prefix[-1] + point.z)
    half = aperture // 2
    count = len(points)
    out = []
    for index, point in enumerate(points):
        low = max(0, index - half)
        high = min(count, index + half + 1)
        averaged = (prefix[high] - prefix[low]) / (high - low)
        out.append(Datapoint(point.freq, averaged.real, averaged.imag))
    return tuple(out)


def smooth_channel(
    points: Sequence[Datapoint], aperture_points: int
) -> tuple[Datapoint, ...]:
    """Complex moving average with a symmetric, shrinking window.

    The window is ``aperture_points`` samples wide, always odd, and centred
    on the sample.  Near the ends it shrinks to the samples that exist
    rather than being wrapped, padded or extrapolated, so smoothing never
    invents data outside the measured band.  The cost is linear in the
    number of samples: a prefix sum makes the window width irrelevant.

    Real and imaginary parts are averaged together (complex averaging),
    which is what keeps magnitude, phase, Smith and VSWR views of the same
    samples consistent: averaging magnitudes alone biases a resonance low,
    and averaging phases alone is meaningless across a wrap.

    Handled explicitly, never silently:

    * an empty channel returns an empty tuple;
    * a channel of one or two samples is returned unchanged, because no
      centred window of three or more samples exists;
    * an aperture wider than the channel is clamped to the data
      (:func:`effective_smoothing_aperture`);
    * a non-finite sample raises :class:`ProcessingError` instead of
      spreading ``NaN`` over the whole neighbourhood;
    * an invalid aperture (zero, negative, even, ``None``) raises.

    The output keeps the input frequency grid exactly, so a smoothed channel
    can be compared sample by sample with the channel it came from.

    Smoothing is a *changed measurement*: it widens narrow features and
    lowers peaks.  It is not an accuracy improvement and is never described
    as one.
    """
    aperture = validate_smoothing_aperture(aperture_points)
    if not points:
        return ()
    _require_finite(points, "smoothing input")
    effective = effective_smoothing_aperture(len(points), aperture)
    if effective == 1:
        return tuple(points)
    return _smoothing_window_sums(points, effective)


def smoothing_aperture_for(points: int, percent: float) -> int:
    """Convert a percentage aperture into an odd point count.

    Documented contract: percentages below 1 are refused, and the result is
    clamped to the number of samples (odd), so the conversion never asks for
    more data than exists.
    """
    if not math.isfinite(percent) or percent <= 0:
        raise ProcessingError("smoothing aperture must be a positive percent")
    if percent > 100:
        raise ProcessingError("smoothing aperture cannot exceed 100 percent")
    if points <= 0:
        raise ProcessingError("smoothing needs at least one sample")
    count = max(1, round(points * percent / 100.0))
    if count % 2 == 0:
        count += 1
    largest = points if points % 2 == 1 else points - 1
    return max(1, min(count, largest))


def validate_average_mode(mode, alpha=None) -> tuple[str, float | None]:
    """Validate a running-average weighting policy.

    Returns ``(mode, alpha)`` with ``alpha`` meaningful only for
    ``exponential``.  A stored stage is validated rather than repaired, so a
    description such as ``{"mode": "cumulative", "alpha": 0.3}`` does not
    quietly become an EWMA, and ``alpha`` outside ``(0, 1]`` is refused
    instead of being clamped.
    """
    name = AVERAGE_CUMULATIVE if mode is None else str(mode)
    if name not in AVERAGE_MODES:
        raise ProcessingError(
            f"unknown running average mode {name!r};"
            f" expected one of {', '.join(AVERAGE_MODES)}"
        )
    if name == AVERAGE_CUMULATIVE:
        if alpha is not None:
            raise ProcessingError(
                "cumulative averaging takes no alpha;"
                " every complete sweep counts equally"
            )
        return name, None
    if alpha is None:
        raise ProcessingError("exponential averaging needs an alpha in (0, 1]")
    try:
        weight = float(alpha)
    except (TypeError, ValueError) as error:
        raise ProcessingError(
            f"exponential alpha must be a number, not {alpha!r}"
        ) from error
    if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
        raise ProcessingError("exponential alpha must be in (0, 1]")
    return name, weight


def running_average_blend(
    previous: complex,
    sample: complex,
    *,
    count: int,
    mode: str = AVERAGE_CUMULATIVE,
    alpha: float | None = None,
) -> complex:
    """Blend one new sample into a running average.

    ``cumulative`` is the exact recurrence
    ``avg_n = avg_(n-1) + (x_n - avg_(n-1))/n`` with ``n = count`` (the
    number of sweeps including this one).  It is algebraically the
    arithmetic mean of every sample seen, and it keeps the accumulated state
    to one complex number per point, so an unbounded number of sweeps costs
    no extra memory.

    ``exponential`` is the EWMA recurrence
    ``avg_n = (1 - alpha) * avg_(n-1) + alpha * x_n`` with a fixed weight in
    ``(0, 1]``; the closed form is
    ``avg_n = sum(alpha * (1-alpha)**(n-k) * x_k) + (1-alpha)**n * avg_0``.

    Weights are applied to the *complex* sample, never to magnitude and
    phase separately, so a blended point is a single complex value and every
    representation derived from it agrees.
    """
    name, weight = validate_average_mode(mode, alpha)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ProcessingError("running average needs an integer count >= 1")
    if name == AVERAGE_CUMULATIVE:
        return previous + (sample - previous) / count
    return previous * (1.0 - weight) + sample * weight


@dataclass(frozen=True)
class RunningAverageState:
    """Immutable state of the continuous running average (DSP-03).

    The accumulator is keyed by a *context* string describing everything
    that makes an old sample incompatible with a new one (source identity,
    calibration identity, channel set, frequency grid, processing
    configuration).  Any context change starts a new accumulation instead of
    blending across an incomparable boundary, and the state additionally
    checks the channel/grid fingerprint it was built from, so a caller that
    forgets to change the context still cannot mix two grids.

    Reset policy, in one place:

    * a different ``context`` (:meth:`reset` is the direct way to ask);
    * a different channel set or frequency grid (:meth:`blend` compares the
      fingerprint and starts over);
    * a different mode/alpha, because the old weights no longer apply;
    * the operator changing the pipeline, the source or the calibration --
      the integrator expresses all three by changing the context or calling
      :meth:`reset`.

    Cancelled, failed and superseded runs are excluded by never blending
    them: :meth:`blend` is only called with a *complete* sweep, so a run
    that never reaches completion leaves the state untouched.  An
    integrator that has blended in-flight data can call :meth:`invalidate`,
    which names what was discarded so the UI can say so.
    """

    context: str = ""
    count: int = 0
    values: tuple[tuple[int, complex], ...] = ()
    mode: str = AVERAGE_CUMULATIVE
    alpha: float | None = None
    fingerprint: str = ""
    discarded: str = ""

    def __post_init__(self) -> None:
        mode, alpha = validate_average_mode(self.mode, self.alpha)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "alpha", alpha)
        if self.count < 0:
            raise ProcessingError("running average count cannot be negative")

    @property
    def active(self) -> bool:
        """True when at least one sweep has been blended."""
        return self.count > 0 and bool(self.values)

    @property
    def effective_count(self) -> int:
        """Number of sweeps that contribute to the current average.

        For ``cumulative`` this is the exact number of sweeps blended.  For
        ``exponential`` it is the *equivalent* count ``1/alpha``: the number
        of equally weighted sweeps with the same noise reduction, rounded to
        at least one.  The two are deliberately the same unit so the UI can
        report one unambiguous number.
        """
        if self.count <= 0:
            return 0
        if self.mode == AVERAGE_EXPONENTIAL and self.alpha:
            return max(1, min(self.count, round(1.0 / self.alpha)))
        return self.count

    @property
    def weighting(self) -> str:
        """One-line statement of the weighting actually in force."""
        if self.count <= 0:
            return "no sweeps blended"
        if self.mode == AVERAGE_EXPONENTIAL:
            return (
                f"exponential alpha={self.alpha:g}"
                f" over {self.count} sweep(s),"
                f" equivalent to {self.effective_count} equal sweeps"
            )
        return f"cumulative mean of {self.count} sweep(s)"

    def reset(self) -> "RunningAverageState":
        """Return an empty state that keeps the weighting policy."""
        return RunningAverageState(mode=self.mode, alpha=self.alpha)

    def invalidate(self, reason: str) -> "RunningAverageState":
        """Drop the accumulated average and record why.

        This is the mechanism for cancelled, failed or superseded runs: the
        integrator calls it instead of blending, so those runs can neither
        contribute a sample nor be mistaken for a reset that lost data.
        """
        dropped = self.reset()
        return replace(dropped, discarded=str(reason))

    def with_policy(
        self, *, mode: str | None = None, alpha: float | None = None
    ) -> "RunningAverageState":
        """Re-policy this state, discarding an incompatible accumulation.

        Switching between ``cumulative`` and ``exponential`` (or changing
        ``alpha``) changes what the stored numbers mean, so the average is
        dropped rather than relabelled.
        """
        target = self.mode if mode is None else mode
        # Only the same mode can inherit the current alpha: cumulative has
        # none, so carrying it across a mode change would be refused.
        inherited = self.alpha if target == self.mode else None
        new_mode, new_alpha = validate_average_mode(
            target, inherited if alpha is None else alpha
        )
        if (new_mode, new_alpha) == (self.mode, self.alpha):
            return self
        return RunningAverageState(
            context=self.context,
            mode=new_mode,
            alpha=new_alpha,
        )

    def blend(
        self,
        channels: Mapping[str, Sequence[Datapoint]],
        *,
        context: str,
        mode: str | None = None,
        alpha: float | None = None,
    ) -> tuple["RunningAverageState", dict[str, tuple[Datapoint, ...]]]:
        """Blend one complete sweep into the running average.

        Only complete, compatible sweeps are blended: an empty or partial
        channel set, or a different grid, resets the accumulation, because
        averaging a partial grid into a full one silently changes what was
        measured.  Non-finite samples raise instead of poisoning the
        accumulator for every later sweep.
        """
        if not channels or not any(channels.values()):
            raise ProcessingError(
                "running average needs at least one complete channel"
            )
        for name, points in channels.items():
            if not points:
                raise ProcessingError(
                    f"running average needs every channel to be complete;"
                    f" {name!r} is empty"
                )
            _require_finite(points, f"running average input {name!r}")

        state = self.with_policy(mode=mode, alpha=alpha)
        fingerprint = _channel_fingerprint(channels)
        if (
            context != state.context
            or fingerprint != state.fingerprint
            or not state.values
        ):
            state = replace(
                state.reset(), context=context, fingerprint=fingerprint
            )
        new_count = state.count + 1
        previous = dict(state.values)
        blended: dict[str, tuple[Datapoint, ...]] = {}
        stored: list[tuple[int, complex]] = []
        for name, points in channels.items():
            out = []
            for point in points:
                sample = point.z
                key = _point_key(name, point.freq)
                prior = previous.get(key)
                value = (
                    sample
                    if prior is None
                    else running_average_blend(
                        prior,
                        sample,
                        count=new_count,
                        mode=state.mode,
                        alpha=state.alpha,
                    )
                )
                stored.append((key, value))
                out.append(Datapoint(point.freq, value.real, value.imag))
            blended[name] = tuple(out)
        return (
            replace(
                state,
                count=new_count,
                values=tuple(stored),
                fingerprint=fingerprint,
                discarded="",
            ),
            blended,
        )


def _point_key(name: str, freq: int) -> int:
    """Stable integer key for one (channel, frequency) pair."""
    digest = hashlib.sha256(f"{name}:{freq}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _channel_fingerprint(channels: Mapping[str, Sequence[Datapoint]]) -> str:
    """Fingerprint of the channel set and its exact frequency grid.

    Two sweeps may only be blended when this string is identical: same
    channel names, same number of samples, same frequencies in the same
    order.  It is computed from the same data for both the incoming sweep
    and the stored state, so the comparison is a real one (an earlier
    version compared a frequency digest against a digest of opaque point
    keys and therefore never matched, silently disabling the average).
    """
    if not channels:
        return ""
    digest = hashlib.sha256()
    for name in sorted(channels):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        for point in channels[name]:
            digest.update(f"{point.freq};".encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()[:32]


@dataclass(frozen=True)
class ProcessingResult:
    """Outcome of applying a configuration to one set of channels.

    ``running`` is the accumulator to carry into the next run when the
    configuration contains :data:`STAGE_AVERAGE`; it is ``None`` when no
    averaging stage ran.  Publishing it is the integrator's job: the state is
    immutable, so an unread ``result.running`` means the next run starts a
    fresh average rather than continuing this one.
    """

    channels: Mapping[str, tuple[Datapoint, ...]]
    config: ProcessingConfig
    reference_impedance: float
    notes: tuple[str, ...] = field(default_factory=tuple)
    running: "RunningAverageState | None" = None

    @property
    def identity(self) -> str:
        return self.config.identity()

    @property
    def effective_average_count(self) -> int:
        """Sweeps contributing to the published average (0 when none)."""
        return 0 if self.running is None else self.running.effective_count


def apply_processing(
    channels: Mapping[str, Sequence[Datapoint]],
    config: ProcessingConfig | None,
    *,
    reference_impedance: float = 50.0,
    running: RunningAverageState | None = None,
    average_context: str = "",
) -> ProcessingResult:
    """Apply ``config`` to ``channels`` and return new tuples.

    The input mapping is never modified.  A stage that fails raises
    :class:`ProcessingError`; the caller keeps the previous result rather
    than publishing a half-processed one.
    """
    working: dict[str, tuple[Datapoint, ...]] = {
        name: tuple(points) for name, points in channels.items() if points
    }
    if config is None or config.is_identity:
        return ProcessingResult(
            channels=working,
            config=config or ProcessingConfig(),
            reference_impedance=reference_impedance,
        )
    notes: list[str] = []
    reference = float(reference_impedance)
    state = running
    for stage in config.active_stages:
        if stage.kind == STAGE_PORT_EXTENSION:
            working = apply_port_extension(working, stage)
        elif stage.kind == STAGE_FIXTURE:
            working = apply_fixture(working, stage, reference)
        elif stage.kind == STAGE_Z0:
            target = _number(stage, "reference_impedance")
            working = apply_renormalize(
                working, stage, current_reference=reference
            )
            reference = target
        elif stage.kind == STAGE_AVERAGE:
            mode = stage.get("mode", AVERAGE_CUMULATIVE)
            alpha = stage.get("alpha", _MISSING)
            # ``blend`` applies and validates the stage's policy itself, and
            # refuses empty, partial and non-finite channels, so there is
            # nothing to pre-check here.
            state, working = (state or RunningAverageState()).blend(
                working,
                context=average_context or "default",
                mode=mode,
                alpha=None if alpha is _MISSING else alpha,
            )
            notes.append(f"{STAGE_LABELS[STAGE_AVERAGE]}: {state.weighting}")
        elif stage.kind == STAGE_SMOOTHING:
            # The stage must describe the aperture it actually carries: a
            # zero, negative, even or missing value raises here instead of
            # being coerced, so a stored configuration can never silently
            # mean a different pipeline than the one it records.
            aperture = stage.get("aperture_points", _MISSING)
            validate_smoothing_aperture(aperture)
            smoothed: dict[str, tuple[Datapoint, ...]] = {}
            applied: set[int] = set()
            irregular: list[str] = []
            for name, points in working.items():
                effective = effective_smoothing_aperture(len(points), aperture)
                applied.add(effective)
                if not channel_is_uniform(points):
                    irregular.append(name)
                smoothed[name] = smooth_channel(points, effective)
            working = smoothed
            notes.append(
                f"{STAGE_LABELS[STAGE_SMOOTHING]}: "
                + ", ".join(f"{count} point(s)" for count in sorted(applied))
                + " (aperture counted in samples)"
            )
            if irregular:
                notes.append(
                    "smoothing covers an uneven bandwidth on the"
                    " non-uniform grid of " + ", ".join(sorted(irregular))
                )
    return ProcessingResult(
        channels=working,
        config=config,
        reference_impedance=reference,
        notes=tuple(notes),
        running=state,
    )


def apply_fixture(
    channels: Mapping[str, Sequence[Datapoint]],
    stage: ProcessingStage,
    reference_impedance: float,
) -> dict[str, tuple[Datapoint, ...]]:
    """Remove a two-port fixture from the measurement (RF-03).

    The fixture is described in the stage as complex S-parameters sampled on
    the measurement's own frequencies.  Removal happens in T-parameter space:

        T_meas = T_fixture · T_dut   =>   T_dut = T_fixture⁻¹ · T_meas

    which is exact for a cascade and requires a complete, invertible fixture.
    When the fixture is not invertible at a frequency, that sample is
    refused instead of being replaced by a zero placeholder.
    """
    from .Fixture import fixture_from_stage, remove_fixture

    fixture = fixture_from_stage(stage)
    return remove_fixture(channels, fixture, reference_impedance)


# ----------------------------------------------------------------------
# Convenience constructors used by the UI
def smoothing_stage(aperture_points: int) -> ProcessingStage:
    """A smoothing stage for a positive odd number of samples.

    The aperture is stored as given and validated here, so a stage can never
    record a width that the engine would refuse or silently reinterpret.
    """
    aperture = validate_smoothing_aperture(aperture_points)
    return ProcessingStage.build(STAGE_SMOOTHING, {"aperture_points": aperture})


def smoothing_stage_for(points: int, percent: float) -> ProcessingStage | None:
    """A smoothing stage from a percent-of-span aperture.

    Returns ``None`` when the aperture collapses to one sample, because such
    a stage would be identity while claiming to smooth: the caller then has
    nothing to show and should say "no smoothing" rather than "1 point".
    """
    aperture = smoothing_aperture_for(points, percent)
    return None if aperture == 1 else smoothing_stage(aperture)


def renormalize_stage(reference_impedance: float) -> ProcessingStage:
    if reference_impedance <= 0:
        raise ProcessingError("reference impedance must be positive")
    return ProcessingStage.build(
        STAGE_Z0, {"reference_impedance": float(reference_impedance)}
    )


def port_extension_stage(
    *,
    port1_delay_ps: float = 0.0,
    port2_delay_ps: float = 0.0,
    port1_loss_db_per_ghz: float = 0.0,
    port2_loss_db_per_ghz: float = 0.0,
) -> ProcessingStage | None:
    if not any(
        (
            port1_delay_ps,
            port2_delay_ps,
            port1_loss_db_per_ghz,
            port2_loss_db_per_ghz,
        )
    ):
        return None
    return ProcessingStage.build(
        STAGE_PORT_EXTENSION,
        {
            "port1_delay_ps": float(port1_delay_ps),
            "port2_delay_ps": float(port2_delay_ps),
            "port1_loss_db_per_ghz": float(port1_loss_db_per_ghz),
            "port2_loss_db_per_ghz": float(port2_loss_db_per_ghz),
        },
    )


def average_stage(
    *, mode: str = AVERAGE_CUMULATIVE, alpha: float | None = None
) -> ProcessingStage:
    """A running-average stage carrying a validated weighting policy."""
    name, weight = validate_average_mode(mode, alpha)
    parameters: dict = {"mode": name}
    if weight is not None:
        parameters["alpha"] = weight
    return ProcessingStage.build(STAGE_AVERAGE, parameters)


def with_stage(
    config: ProcessingConfig | None, stage: ProcessingStage | None
) -> ProcessingConfig:
    """Add or remove one stage from a configuration, keeping order."""
    base = config or ProcessingConfig()
    if stage is None:
        return base
    return base.with_stage(stage)


def identity_of(config: ProcessingConfig | None) -> str:
    return "" if config is None else config.identity()


def describe(config: ProcessingConfig | None) -> str:
    return "Raw (no processing)" if config is None else config.describe()


def replace_identity(config: ProcessingConfig, **kwargs) -> ProcessingConfig:
    """Return ``config`` with dataclass fields replaced (test helper)."""
    return replace(config, **kwargs)
