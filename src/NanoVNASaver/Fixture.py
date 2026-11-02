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
"""Fixture S-parameters and de-embedding (RF-03).

A fixture is a two-port network between the reference plane and the device
under test.  Removing it is a cascade inversion in T-parameter space::

    T_measured = T_fixture · T_dut        =>   T_dut = T_fixture⁻¹ · T_measured

T-parameters are the standard rearrangement of S-parameters::

    T = 1/S21 * [[ -det(S), S11 ],
                 [ -S22,     1  ]]

and the inverse rearrangement is::

    S11 = T12/T22,  S21 = 1/T22,  S12 = det(T)/T22,  S22 = -T21/T22

The transform is exact for a cascade of two two-port networks and needs all
four S-parameters of the fixture.  There is deliberately no "assume S12=0"
shortcut in this module: a forward-only fixture is a different, approximate
model and must not be presented as general de-embedding.

Sign convention: this application measures S-parameters with the usual
power-wave definition, so a through connection has S21 = 1 and a matched
attenuator of ``L`` dB has ``|S21| = 10**(-L/20)``.
"""

from __future__ import annotations

import hashlib
import itertools as it
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .RFTools import Datapoint

#: Assumed reference impedance when a fixture does not declare its own.
DEFAULT_FIXTURE_Z0 = 50.0

#: A fixture is refused when |det(T)| is smaller than this, or when the
#: condition number of the cascade exceeds the limit: inverting a nearly
#: singular network amplifies measurement noise without bound.
MIN_DETERMINANT = 1e-9
MAX_CONDITION = 1e6


class FixtureError(ValueError):
    """The fixture cannot be applied to the supplied measurement."""


@dataclass(frozen=True)
class FixturePoint:
    """One measured fixture sample, all four S-parameters."""

    freq: int
    s11: complex = 0j
    s21: complex = 0j
    s12: complex = 0j
    s22: complex = 0j

    def to_dict(self) -> dict:
        def encode(value: complex) -> list[float]:
            return [value.real, value.imag]

        return {
            "freq": self.freq,
            "s11": encode(self.s11),
            "s21": encode(self.s21),
            "s12": encode(self.s12),
            "s22": encode(self.s22),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "FixturePoint":
        """Rebuild a sample from a mapping.

        A stored processing stage holds its parameters as a tuple of
        ``(key, value)`` pairs, so the frozen form is accepted as well as a
        plain mapping: a stage written to a session must be readable back.
        """

        if not isinstance(data, Mapping):
            try:
                data = dict(data)
            except (TypeError, ValueError) as exc:
                raise FixtureError(
                    "a fixture sample must be a mapping or key/value pairs"
                ) from exc

        def decode(value) -> complex:
            if isinstance(value, complex):
                return value
            if isinstance(value, Sequence) and len(value) == 2:
                return complex(float(value[0]), float(value[1]))
            raise FixtureError("fixture S-parameters must be [re, im] pairs")

        return cls(
            freq=int(data["freq"]),
            s11=decode(data.get("s11", [0.0, 0.0])),
            s21=decode(data.get("s21", [0.0, 0.0])),
            s12=decode(data.get("s12", [0.0, 0.0])),
            s22=decode(data.get("s22", [0.0, 0.0])),
        )


@dataclass(frozen=True)
class Fixture:
    """A two-port fixture measured on its own frequency grid."""

    name: str = ""
    reference_impedance: float = DEFAULT_FIXTURE_Z0
    points: tuple[FixturePoint, ...] = ()
    source: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.points:
            raise FixtureError("a fixture needs at least one measured point")
        if self.reference_impedance <= 0 or not math.isfinite(
            self.reference_impedance
        ):
            raise FixtureError("fixture reference impedance must be positive")
        frequencies = [point.freq for point in self.points]
        if any(b < a for a, b in it.pairwise(frequencies)):
            raise FixtureError("fixture points must be frequency ordered")
        if len(set(frequencies)) != len(frequencies):
            raise FixtureError("fixture points contain duplicate frequencies")
        for point in self.points:
            for value in (point.s11, point.s21, point.s12, point.s22):
                if not (
                    math.isfinite(value.real) and math.isfinite(value.imag)
                ):
                    raise FixtureError(
                        f"fixture sample at {point.freq} Hz is not finite"
                    )

    @property
    def start(self) -> int:
        return self.points[0].freq

    @property
    def stop(self) -> int:
        return self.points[-1].freq

    def identity(self) -> str:
        """Stable digest of the numerically relevant fixture content."""
        payload = json.dumps(
            {
                "reference_impedance": self.reference_impedance,
                "points": [point.to_dict() for point in self.points],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def value_at(self, freq: int) -> FixturePoint:
        """Fixture sample exactly at ``freq``.

        Interpolation is deliberately not performed here: a de-embedding that
        runs on a grid the fixture was not measured on is a different
        operation with its own error, so the caller aligns grids explicitly.
        """
        for point in self.points:
            if point.freq == freq:
                return point
        raise FixtureError(
            f"the fixture has no sample at {freq} Hz; align the grids first"
        )

    def describe(self) -> str:
        return (
            f"{self.name or 'fixture'} ({self.start} - {self.stop} Hz,"
            f" {len(self.points)} points, Z0 {self.reference_impedance:g})"
        )


def s_to_t(point: FixturePoint) -> tuple[tuple[complex, complex], ...]:
    """Convert one S-parameter sample to a T matrix."""
    if point.s21 == 0:
        raise FixtureError(
            f"S21 is zero at {point.freq} Hz: a fixture with no transmission"
            " cannot be de-embedded"
        )
    determinant = point.s11 * point.s22 - point.s12 * point.s21
    return (
        (-determinant / point.s21, point.s11 / point.s21),
        (-point.s22 / point.s21, 1 / point.s21),
    )


def t_to_s(
    matrix: tuple[tuple[complex, complex], ...], freq: int
) -> FixturePoint:
    """Convert a T matrix back to an S-parameter sample."""
    (t11, t12), (t21, t22) = matrix
    if t22 == 0:
        raise FixtureError(
            f"degenerate cascade at {freq} Hz: T22 is zero, so the result"
            " has no finite S-parameters"
        )
    determinant = t11 * t22 - t12 * t21
    return FixturePoint(
        freq=freq,
        s11=t12 / t22,
        s21=1 / t22,
        s12=determinant / t22,
        s22=-t21 / t22,
    )


def _matmul(left, right):
    return (
        (
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ),
        (
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ),
    )


def _inverse(matrix, freq: int):
    (a, b), (c, d) = matrix
    determinant = a * d - b * c
    if abs(determinant) < MIN_DETERMINANT:
        raise FixtureError(
            f"the fixture is singular at {freq} Hz (|det| ="
            f" {abs(determinant):.3g}): it cannot be removed here"
        )
    return (
        (d / determinant, -b / determinant),
        (-c / determinant, a / determinant),
    )


def _condition(matrix) -> float:
    """A scale-invariant conditioning measure for a 2x2 complex matrix."""
    (a, b), (c, d) = matrix
    norm = max(abs(a), abs(b), abs(c), abs(d))
    if norm == 0:
        return math.inf
    scaled = (
        (a / norm, b / norm),
        (c / norm, d / norm),
    )
    determinant = abs(scaled[0][0] * scaled[1][1] - scaled[0][1] * scaled[1][0])
    if determinant == 0:
        return math.inf
    frobenius = math.sqrt(
        sum(abs(value) ** 2 for row in scaled for value in row)
    )
    return frobenius**2 / determinant


def deembed_point(
    frequency: int,
    measured: FixturePoint,
    fixture: FixturePoint,
) -> FixturePoint:
    """Remove ``fixture`` from ``measured`` at one frequency."""
    if fixture.s21 == 0:
        raise FixtureError(f"the fixture has no transmission at {frequency} Hz")
    fixture_t = s_to_t(fixture)
    if _condition(fixture_t) > MAX_CONDITION:
        raise FixtureError(
            f"the fixture is ill-conditioned at {frequency} Hz;"
            " de-embedding would amplify noise without bound"
        )
    measured_t = s_to_t(measured)
    result = _matmul(_inverse(fixture_t, frequency), measured_t)
    return t_to_s(result, frequency)


def remove_fixture(
    channels: Mapping[str, Sequence[Datapoint]],
    fixture: Fixture,
    reference_impedance: float = DEFAULT_FIXTURE_Z0,
) -> dict[str, tuple[Datapoint, ...]]:
    """Remove ``fixture`` from a complete two-port measurement.

    All four channels are required: general de-embedding is a matrix
    operation, and silently treating an absent S12 as zero would invent a
    reciprocal fixture that was never measured.
    """
    if abs(reference_impedance - fixture.reference_impedance) > 1e-9:
        raise FixtureError(
            "the fixture reference impedance"
            f" ({fixture.reference_impedance:g} ohm) does not match the"
            f" measurement ({reference_impedance:g} ohm)"
        )
    missing = [
        name for name in ("S11", "S21", "S12", "S22") if not channels.get(name)
    ]
    if missing:
        raise FixtureError(
            "fixture de-embedding needs a complete two-port measurement;"
            f" missing {', '.join(missing)}"
        )
    grids = {
        name: tuple(point.freq for point in channels[name])
        for name in ("S11", "S21", "S12", "S22")
    }
    first = grids["S11"]
    for name, grid in grids.items():
        if grid != first:
            raise FixtureError(
                f"the {name} channel is not on the same grid as S11"
            )
    fixture_grid = {point.freq for point in fixture.points}
    absent = [freq for freq in first if freq not in fixture_grid]
    if absent:
        raise FixtureError(
            f"the fixture has no sample at {absent[0]} Hz; measure or"
            " interpolate the fixture onto this sweep first"
        )

    by_name = {
        name: list(channels[name]) for name in ("S11", "S21", "S12", "S22")
    }
    result: dict[str, tuple[Datapoint, ...]] = {
        name: [] for name in ("S11", "S21", "S12", "S22")
    }
    for index, freq in enumerate(first):
        measured = FixturePoint(
            freq=freq,
            s11=by_name["S11"][index].z,
            s21=by_name["S21"][index].z,
            s12=by_name["S12"][index].z,
            s22=by_name["S22"][index].z,
        )
        dut = deembed_point(freq, measured, fixture.value_at(freq))
        for name, points in result.items():
            value = getattr(dut, name.lower())
            points.append(Datapoint(freq, value.real, value.imag))
    return {name: tuple(points) for name, points in result.items()}


def fixture_from_touchstone(
    touchstone, *, name: str = "", source: str = ""
) -> Fixture:
    """Build a fixture from a loaded two-port Touchstone file."""
    points = []
    channels = {}
    for name_ in ("s11", "s21", "s12", "s22"):
        points_ = getattr(touchstone, name_, None)
        if not points_:
            raise FixtureError(
                "a fixture Touchstone file must contain all four"
                f" S-parameters; S{name_[1:].upper()} is missing"
            )
        channels[name_] = points_
    # A file whose columns disagree in length cannot describe one network:
    # indexing the shorter column would raise a bare IndexError halfway
    # through, so the mismatch is named instead.
    lengths = {name_: len(values) for name_, values in channels.items()}
    if len(set(lengths.values())) != 1:
        detail = ", ".join(
            f"S{key[1:].upper()}={value}" for key, value in lengths.items()
        )
        raise FixtureError(
            "the fixture channels have different numbers of points"
            f" ({detail}); each row must carry all four S-parameters"
        )
    for index, point in enumerate(channels["s11"]):
        freq = point.freq
        for values in channels.values():
            if values[index].freq != freq:
                raise FixtureError(
                    f"the fixture channels disagree at {freq} Hz"
                )
        points.append(
            FixturePoint(
                freq=freq,
                s11=channels["s11"][index].z,
                s21=channels["s21"][index].z,
                s12=channels["s12"][index].z,
                s22=channels["s22"][index].z,
            )
        )
    return Fixture(
        name=name or getattr(touchstone, "filename", "") or "fixture",
        reference_impedance=float(touchstone.opts.resistance),
        points=tuple(points),
        source=source,
    )


def fixture_from_stage(stage) -> Fixture:
    """Rebuild a :class:`Fixture` from a stored processing stage."""
    from .Processing import ProcessingError, ProcessingStage

    if not isinstance(stage, ProcessingStage):
        raise ProcessingError("a fixture stage is required")
    values = stage.as_dict
    points = tuple(
        FixturePoint.from_dict(point) for point in values.get("points", ())
    )
    try:
        return Fixture(
            name=str(values.get("name", "")),
            reference_impedance=float(
                values.get("reference_impedance", DEFAULT_FIXTURE_Z0)
            ),
            points=points,
            source=str(values.get("source", "")),
        )
    except FixtureError as exc:
        raise ProcessingError(str(exc)) from exc


def fixture_stage(fixture: Fixture):
    """Store ``fixture`` as a self-contained processing stage.

    The sampled S-parameters are embedded, so a session or report keeps
    working when the original file is moved or edited.  The identity is
    recorded alongside them so a mismatch between the embedded content and
    the file it came from stays detectable.
    """
    from .Processing import STAGE_FIXTURE, ProcessingStage

    parameters = {
        "name": fixture.name,
        "reference_impedance": fixture.reference_impedance,
        "source": fixture.source,
        "identity": fixture.identity(),
        "points": [point.to_dict() for point in fixture.points],
    }
    return ProcessingStage.build(STAGE_FIXTURE, parameters)
