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
"""Analytic device responses used by the simulated device (SIM-01.3).

Every scenario is a pure function of frequency.  With noise disabled the
results are exactly reproducible, so the expected analysis values (resonance,
attenuation, cable delay) can be known in closed form and asserted in tests.

Reference impedance is 50 ohms throughout, matching the application default.
Angles are radians; frequencies are Hz.

Three optional layers sit on top of the ideal responses:

``ScenarioResult.s12``/``s22``
    A scenario that models a complete reciprocal two-port fills these in so
    that :class:`ReversedTwoPort` can turn the DUT around.  ``None`` means
    "this model does not describe that channel" and is never a measured
    zero, so a one-port antenna cannot be reversed into a fabricated S22.

``ErrorModel``
    A forward 12-term error box applied to whatever the DUT emits.  Every
    term is individually switchable, and with no term enabled the result is
    the ideal response, bit for bit.

``VIRTUAL_STANDARDS``
    Synthesised short/open/load/isolation/through standards.  They are
    computed from their definitions, never recorded by re-reading data the
    application has already corrected.

None of this makes the simulator physically accurate: it makes the
*correction* path testable end to end, which is regression evidence only.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Callable

REFERENCE_IMPEDANCE = 50.0
SPEED_OF_LIGHT = 299_792_458.0


def _gamma_from_impedance(impedance: complex) -> complex:
    """Reflection coefficient of a load at the reference impedance."""
    return (impedance - REFERENCE_IMPEDANCE) / (impedance + REFERENCE_IMPEDANCE)


@dataclass(frozen=True)
class ScenarioResult:
    """Complex S-parameters for one frequency.

    ``s11``/``s21`` are the forward channels a NanoVNA actually measures.
    ``s12``/``s22`` are only present when the model describes a complete
    two-port; ``None`` means "not modelled", which is different from a
    measured zero and must not be replaced by one.
    """

    s11: complex
    s21: complex
    s12: complex | None = None
    s22: complex | None = None

    def __iter__(self):
        yield self.s11
        yield self.s21

    @property
    def complete(self) -> bool:
        """True when all four S-parameters are modelled."""
        return self.s12 is not None and self.s22 is not None

    def reversed(self) -> "ScenarioResult":
        """The same network measured with its ports exchanged.

        Physically flipping a two-port swaps its ports: the port-1
        reflection becomes the old port-2 reflection and the forward
        transmission becomes the old reverse transmission.  This is only
        possible for a model that already describes those channels; nothing
        is synthesized from the forward data alone.
        """
        if not self.complete:
            raise ReversalError(
                "this model does not describe S12/S22, so the DUT cannot be"
                " reversed without inventing the reverse channels"
            )
        return ScenarioResult(
            s11=self.s22,
            s21=self.s12,
            s12=self.s21,
            s22=self.s11,
        )


class ReversalError(ValueError):
    """A model cannot be reversed because it lacks the reverse channels."""


class ErrorModelError(ValueError):
    """The error model cannot be applied to the DUT it was given."""


#: Provenance text carried by every non-ideal layer below.
SIMULATION_DISCLAIMER = (
    "synthetic regression evidence, not a physical measurement"
)

#: The six terms of the forward 12-term model, in equation order.
ERROR_TERM_NAMES = (
    "directivity",
    "source_match",
    "load_match",
    "reflection_tracking",
    "transmission_tracking",
    "isolation",
)


# ----------------------------------------------------------------------
# Instrument error box: forward 12-term model
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ErrorModel:
    """A forward 12-term instrument error box (SIM-01.1).

    Every term is individually switchable.  ``None`` -- the default -- means
    the term is disabled and is then the exact identity, so an
    :class:`ErrorModel` with nothing enabled reproduces the ideal DUT bit
    for bit instead of merely closely.  ``0`` is a term that is enabled but
    happens to be zero; for the two tracking terms ``0`` is rejected
    because it would make the correction undefined.

    The equations are the ones the application's own
    :class:`NanoVNASaver.Calibration.Calibration` solves and applies, with
    ``S12*S21`` and ``S22`` taken from the DUT when it states them::

        D     = (1 - e11*S11)*(1 - e22*S22) - e11*e22*S12*S21
        delta = S11*S22 - S12*S21
        S11m  = e00 + e10e01*(S11 - e22*delta) / D
        S21m  = e30 + e10e32*S21 / D

    A DUT that states no reverse channels is a one-port: its port 2 carries
    no energy, ``S12 = S21 = 0`` by definition, and the model degenerates to
    ``S11m = e00 + e10e01*S11/(1 - e11*S11)`` with ``S21m = e30``.  Nothing
    is published as a reverse sample -- the result keeps ``s12``/``s22`` as
    the DUT stated them, which for a one-port is ``None``.  A DUT that does
    transmit but declares no reverse channels is refused whenever a
    port-2 term (``e22``) is enabled, because that term depends on the
    unknown load; with ``e22`` disabled the model is still exact and is
    applied.

    An injected error box is what makes a calibration measurable end to
    end.  It is regression evidence: it does not make the simulator
    physically accurate.
    """

    directivity: complex | None = None
    """``e00``: leakage seen with a perfect load, i.e. the floor of S11."""

    source_match: complex | None = None
    """``e11``: reflection of the test port looking back at the source."""

    load_match: complex | None = None
    """``e22``: reflection of the load presented at port 2."""

    reflection_tracking: complex | None = None
    """``e10e01``: forward reflection tracking."""

    transmission_tracking: complex | None = None
    """``e10e32``: forward transmission tracking."""

    isolation: complex | None = None
    """``e30``: leakage between the ports with no path through the DUT."""

    def __post_init__(self) -> None:
        for name in ERROR_TERM_NAMES:
            value = getattr(self, name)
            if value is None:
                continue
            value = complex(value)
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                raise ValueError(f"{name} must be finite, got {value!r}")
            object.__setattr__(self, name, value)
        for name in ("reflection_tracking", "transmission_tracking"):
            if getattr(self, name) == 0:
                raise ValueError(
                    f"{name} must be non-zero when enabled;"
                    " pass None to disable the term"
                )

    # -- terms ---------------------------------------------------------
    @property
    def enabled_terms(self) -> tuple[str, ...]:
        """Names of the terms that were explicitly switched on."""
        return tuple(
            name for name in ERROR_TERM_NAMES if getattr(self, name) is not None
        )

    def terms(self) -> tuple[complex, ...]:
        """Effective ``(e00, e11, e22, e10e01, e10e32, e30)``."""
        zero = complex(0.0, 0.0)
        one = complex(1.0, 0.0)
        reflection = self.reflection_tracking
        transmission = self.transmission_tracking
        return (
            zero if self.directivity is None else self.directivity,
            zero if self.source_match is None else self.source_match,
            zero if self.load_match is None else self.load_match,
            one if reflection is None else reflection,
            one if transmission is None else transmission,
            zero if self.isolation is None else self.isolation,
        )

    @property
    def is_identity(self) -> bool:
        """True when this box distorts nothing at all."""
        e00, e11, e22, e10e01, e10e32, e30 = self.terms()
        return (
            e00 == 0
            and e11 == 0
            and e22 == 0
            and e30 == 0
            and e10e01 == 1
            and e10e32 == 1
        )

    # -- measurement ---------------------------------------------------
    def apply(self, result: ScenarioResult) -> ScenarioResult:
        """Return what this box measures for the ideal ``result``.

        An identity box returns ``result`` unchanged -- the same object, so
        the ideal path is bit for bit the ideal path.
        """
        if self.is_identity:
            return result
        e00, e11, e22, e10e01, e10e32, e30 = self.terms()
        s11, s21 = result.s11, result.s21
        if result.complete:
            s22, s12 = result.s22, result.s12
            product = s12 * s21
            delta = s11 * s22 - product
            denominator = (1 - e11 * s11) * (1 - e22 * s22) - (
                e11 * e22 * product
            )
        else:
            if s21 and e22 != 0:
                # Without S12/S22 the load-match term cannot be evaluated:
                # a transmitting DUT terminated in an unknown port-2 load
                # is not the same measurement as a one-port.  Refuse rather
                # than assume S22 = 0.
                raise ErrorModelError(
                    f"the DUT transmits (S21={s21!r}) but states no S12/S22,"
                    " so the load-match term cannot be applied; declare the"
                    " reverse channels or disable load_match."
                )
            # A one-port DUT: no energy reaches port 2, so S12 = S21 = S22 = 0
            # by definition and the model is the one-port form below.
            delta = complex(0.0, 0.0)
            denominator = 1 - e11 * s11
        if denominator == 0:
            raise ErrorModelError(
                "the error box is singular for this DUT"
                f" (denominator {denominator!r})"
            )
        measured_s11 = e00 + e10e01 * (s11 - e22 * delta) / denominator
        measured_s21 = e30 + e10e32 * s21 / denominator
        return ScenarioResult(
            measured_s11, measured_s21, result.s12, result.s22
        )

    # -- provenance ----------------------------------------------------
    def to_dict(self) -> dict[str, list[float]]:
        """Enabled terms only, as ``{name: [real, imaginary]}``."""
        return {
            name: [getattr(self, name).real, getattr(self, name).imag]
            for name in self.enabled_terms
        }

    def describe(self) -> str:
        """One line naming exactly the terms that are switched on."""
        labels = {
            "directivity": "directivity e00",
            "source_match": "source match e11",
            "load_match": "load match e22",
            "reflection_tracking": "reflection tracking e10e01",
            "transmission_tracking": "transmission tracking e10e32",
            "isolation": "isolation e30",
        }
        if self.is_identity:
            return "simulated error box: none (ideal instrument)"
        terms = ", ".join(
            f"{labels[name]}={getattr(self, name):.6g}"
            for name in self.enabled_terms
        )
        return f"simulated error box: {terms} ({SIMULATION_DISCLAIMER})"


#: The calibration standards a one-port (or two-port) calibration needs.
STANDARD_NAMES = ("short", "open", "load", "through", "isolation")


@dataclass(frozen=True)
class VirtualStandard:
    """One synthesised calibration standard, usable as a device scenario.

    It is callable so that ``SimulatedDevice(scenario=standard)`` measures it
    exactly like any other scenario, and it declares the channels it really
    has: a short is a one-port, so it offers ``data 0`` only and no invented
    S21.
    """

    name: str
    response: ScenarioResult
    """What the instrument measures for this standard, error box included."""
    standards: VirtualStandards | None = field(
        default=None, compare=False, repr=False
    )
    """The set this standard came from, so provenance can name its siblings."""

    def __call__(self, freq: float) -> ScenarioResult:
        return self.response

    @property
    def supported_channels(self) -> tuple[str, ...]:
        return ("data 0", "data 1") if self.response.complete else ("data 0",)

    @property
    def scenario_key(self) -> str:
        return f"standard_{self.name}"

    @property
    def display_name(self) -> str:
        return f"Virtual standard: {self.name}"

    def to_dict(self) -> dict[str, list[float]]:
        """Provenance: the raw (measured) S11/S21 this standard produces."""
        return {
            "s11": [self.response.s11.real, self.response.s11.imag],
            "s21": [self.response.s21.real, self.response.s21.imag],
        }


@dataclass(frozen=True)
class VirtualStandards:
    """Short/open/load/through/isolation synthesised analytically (SIM-01.2).

    Each standard is defined by its ideal S-parameters and then *measured*
    through the same :class:`ErrorModel` as the DUT.  Nothing here is copied
    from display data the application has already corrected: the ideal
    values are the definitions (``-1``, ``+1``, ``0``, a straight through,
    and no path at all) and the measurement is computed from them.

    A standard measured through an identity box is the ideal standard
    itself, so a device with no injected error calibrates exactly.

    Limitation: the application's ``CalDataSet.complete1port()`` tests the
    raw standards for truthiness, so a box whose directivity is exactly zero
    measures a load of exactly zero and the calibration refuses it.  That is
    the application's completeness rule, not a property of this model.
    """

    error_model: ErrorModel = field(default_factory=ErrorModel)

    @property
    def names(self) -> tuple[str, ...]:
        return STANDARD_NAMES

    def ideal(self, name: str) -> ScenarioResult:
        """The standard as an ideal component defines it."""
        zero = complex(0.0, 0.0)
        one = complex(1.0, 0.0)
        if name == "short":
            return ScenarioResult(complex(-1.0, 0.0), zero)
        if name == "open":
            return ScenarioResult(one, zero)
        if name == "load":
            return ScenarioResult(zero, zero)
        if name == "through":
            return ScenarioResult(zero, one, one, zero)
        if name == "isolation":
            # Ports terminated with no path between them: S21 is leakage.
            return ScenarioResult(zero, zero, zero, zero)
        raise ValueError(
            f"unknown standard {name!r}; choose from {list(STANDARD_NAMES)}"
        )

    def measured(self, name: str) -> ScenarioResult:
        """What the instrument reads for ``name``, error box included."""
        return self.error_model.apply(self.ideal(name))

    def standard(self, name: str) -> VirtualStandard:
        """``name`` as a scenario a simulated device can sweep."""
        return VirtualStandard(name, self.measured(name), self)

    def measured_all(self) -> dict[str, ScenarioResult]:
        return {name: self.measured(name) for name in STANDARD_NAMES}

    def to_dict(self) -> dict[str, object]:
        """Provenance: the standards and the box they are measured through."""
        return {
            "standards": list(STANDARD_NAMES),
            "error_model": self.error_model.to_dict(),
            "disclaimer": SIMULATION_DISCLAIMER,
        }

    def describe(self) -> str:
        return (
            f"virtual standards: {', '.join(STANDARD_NAMES)}"
            f" ({self.error_model.describe()}; {SIMULATION_DISCLAIMER})"
        )

    def acquisition_state(self) -> dict[str, object]:
        """The standards and error box in force, as a machine-readable record.

        A human line goes into the firmware banner; this record is what an
        export or report can carry and a reader can check, so "simulated"
        never survives only as prose.
        """
        return {
            "standards": {
                name: self.standard(name).to_dict() for name in STANDARD_NAMES
            },
            "error_model": self.error_model.to_dict(),
            "disclaimer": SIMULATION_DISCLAIMER,
        }


@dataclass(frozen=True)
class ReversedTwoPort:
    """The wrapped two-port measured with its ports exchanged (SIM-01.3).

    Flipping a network swaps its ports: the new port-1 reflection is the old
    port-2 reflection and the new forward transmission is the old reverse
    transmission.  Only a model that states S12 *and* S22 can be flipped —
    reciprocity (S12 == S21) alone says nothing about S22, so a one-port or a
    transmission-only model raises :class:`ReversalError` instead of having
    the reverse reflection invented for it.
    """

    scenario: Scenario

    def __call__(self, freq: float) -> ScenarioResult:
        return self.scenario(freq).reversed()

    @property
    def scenario_key(self) -> str:
        return f"{_scenario_key(self.scenario)}_reversed"

    @property
    def display_name(self) -> str:
        name = getattr(self.scenario, "display_name", "")
        if not (isinstance(name, str) and name):
            name = type(self.scenario).__name__
        return f"{name} (ports reversed)"

    @property
    def supported_channels(self) -> tuple[str, ...]:
        channels = getattr(self.scenario, "supported_channels", None)
        return tuple(channels) if channels else ("data 0", "data 1")

    def provenance_text(self) -> str:
        return (
            f"DUT measured with its ports exchanged ({SIMULATION_DISCLAIMER})"
        )


def _scenario_key(scenario: Scenario) -> str:
    """Canonical scenario name, matching ``SimulatedDevice.scenario_key``."""
    key = getattr(scenario, "scenario_key", "")
    if isinstance(key, str) and key:
        return key
    for name, factory in SCENARIO_TYPES.items():
        if isinstance(scenario, factory):
            return name
    return type(scenario).__name__


# ----------------------------------------------------------------------
# Antenna: series RLC resonator
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AntennaResonance:
    """A series RLC antenna model.

    At ``resonant_frequency`` the reactance cancels and the input impedance
    equals the radiation resistance (plus loss resistance), giving the
    minimum VSWR.  VSWR is exactly 1 when
    ``radiation_resistance + loss_resistance == 50``.

    ``q_factor`` sets the bandwidth: Q = f0 / bandwidth.
    """

    resonant_frequency: float = 14.15e6
    radiation_resistance: float = 50.0
    loss_resistance: float = 0.0
    q_factor: float = 20.0

    @property
    def resistance(self) -> float:
        return self.radiation_resistance + self.loss_resistance

    @property
    def inductance(self) -> float:
        return (
            self.q_factor
            * self.resistance
            / (2.0 * math.pi * self.resonant_frequency)
        )

    @property
    def capacitance(self) -> float:
        return 1.0 / (
            (2.0 * math.pi * self.resonant_frequency) ** 2 * self.inductance
        )

    @property
    def bandwidth(self) -> float:
        return self.resonant_frequency / self.q_factor

    def impedance(self, freq: float) -> complex:
        if freq <= 0:
            return complex(self.resistance, 0.0)
        omega = 2.0 * math.pi * freq
        reactance = omega * self.inductance - 1.0 / (omega * self.capacitance)
        return complex(self.resistance, reactance)

    def s11(self, freq: float) -> complex:
        return _gamma_from_impedance(self.impedance(freq))

    def vswr(self, freq: float) -> float:
        magnitude = abs(self.s11(freq))
        if magnitude >= 1.0:
            return math.inf
        return (1.0 + magnitude) / (1.0 - magnitude)

    def vswr_bandwidth_edges(
        self, max_vswr: float = 2.0
    ) -> tuple[float, float]:
        """Frequencies where VSWR crosses ``max_vswr`` on either side."""
        gamma_limit = (max_vswr - 1.0) / (max_vswr + 1.0)
        # |Z - R0| / |Z + R0| = gamma_limit  <=>  R^2 + X^2 relationship.
        # For a series RLC, X^2 = ((1+g^2)/(1-g^2)) * R^2 - R^2 ... derive:
        # g^2 = ((R-R0)^2 + X^2) / ((R+R0)^2 + X^2)
        # => X^2 (1-g^2) = g^2 (R+R0)^2 - (R-R0)^2
        r = self.resistance
        numerator = (
            gamma_limit**2 * (r + REFERENCE_IMPEDANCE) ** 2
            - (r - REFERENCE_IMPEDANCE) ** 2
        )
        denominator = 1.0 - gamma_limit**2
        if denominator <= 0:
            return (0.0, math.inf)
        x_squared = numerator / denominator
        if x_squared < 0:
            # Even at resonance the VSWR exceeds the limit.
            return (self.resonant_frequency, self.resonant_frequency)
        reactance = math.sqrt(x_squared)
        # omega*L - 1/(omega*C) = +/- X  =>  quadratic in omega
        edges = []
        for sign in (1.0, -1.0):
            a = self.inductance
            b = -sign * reactance
            c = -1.0 / self.capacitance
            disc = b * b - 4 * a * c
            omega = (-b + math.sqrt(disc)) / (2 * a)
            edges.append(omega / (2.0 * math.pi))
        return (min(edges), max(edges))

    #: An antenna is a one-port device.  Declaring a supported channel set
    #: is what stops the S21 slot from being published as a measured zero:
    #: ``ScenarioResult`` always carries two complex values, so without this
    #: a one-port antenna would look like a perfect transmission path and a
    #: limit check on S21 would report PASS on data that was never measured.
    supported_channels = ("data 0",)

    def __call__(self, freq: float) -> ScenarioResult:
        return ScenarioResult(self.s11(freq), complex(0.0, 0.0))


# ----------------------------------------------------------------------
# Filter: N-th order low-pass Butterworth, 2-port
# ----------------------------------------------------------------------
#: Residual of ``1 - |S21|^2`` below which a lossless two-port is treated as
#: matched.  A few ulps of 1: the subtraction cannot do better than that, and
#: its square root would otherwise turn rounding noise into a visible
#: reflection (``sqrt(2.2e-16) == 1.5e-8``).
_ROUNDING_FLOOR = 8 * 2.220446049250313e-16


@dataclass(frozen=True)
class ButterworthLowPass:
    """Ideal doubly-terminated Butterworth low-pass response.

    ``order`` poles, ``-3 dB`` magnitude exactly at ``cutoff_frequency``.
    Phase is derived from the same rational transfer function, so group delay
    is physically consistent instead of being an arbitrary extra model.

    Above cutoff the magnitude rolls off at ``20 * order`` dB/decade.
    The network is lossless and reciprocal: |S11|^2 + |S21|^2 == 1.
    """

    cutoff_frequency: float = 100e6
    order: int = 5
    ripple_db: float = 0.0
    """Only ``0.0`` (Butterworth) is implemented; kept explicit."""

    def _poles(self) -> list[complex]:
        poles = []
        for k in range(self.order):
            theta = math.pi * (2 * k + 1 + self.order) / (2 * self.order)
            poles.append(cmath.exp(1j * theta))
        return poles

    def _denominator_poly(self) -> list[complex]:
        """Coefficients of prod(s - p) with s normalised to cutoff (rad/s)."""
        coeffs = [1.0 + 0.0j]
        for pole in self._poles():
            new = [0.0 + 0.0j] * (len(coeffs) + 1)
            for i, value in enumerate(coeffs):
                new[i] += value
                new[i + 1] -= value * pole
            coeffs = new
        return coeffs

    def response(self, freq: float) -> tuple[complex, complex]:
        """Return ``(s21, s11)`` for ``freq``."""
        omega = freq / self.cutoff_frequency
        s = 1j * omega
        coeffs = self._denominator_poly()
        denominator = 0.0 + 0.0j
        for i, value in enumerate(coeffs):
            denominator += value * s ** (len(coeffs) - 1 - i)
        numerator = coeffs[-1]  # dc gain = 1
        s21 = numerator / denominator
        reflection_sq = 1.0 - abs(s21) ** 2
        # A lossless two-port has |S11|^2 == 1 - |S21|^2 exactly, but the
        # subtraction is done in floating point, so a matched filter leaves a
        # residual of a few ulps instead of 0 -- and the square root then
        # *amplifies* it: 2.2e-16 becomes 1.5e-8, a phantom reflection that
        # also varies between interpreters (CPython 3.12 gave exactly 0
        # here, 3.14 gives 2.2e-16).  A residual that small is rounding
        # noise, not physics, so it is treated as matched.
        s11 = (
            0j
            if reflection_sq <= _ROUNDING_FLOOR
            else cmath.sqrt(reflection_sq)
        )
        return complex(s21), complex(s11, 0.0)

    def magnitude_db(self, freq: float) -> float:
        s21, _ = self.response(freq)
        magnitude = abs(s21)
        return 20.0 * math.log10(magnitude) if magnitude > 0 else -math.inf

    def __call__(self, freq: float) -> ScenarioResult:
        s21, s11 = self.response(freq)
        # A doubly-terminated ladder is symmetric and reciprocal, so the
        # reverse channels are the forward ones by definition of the model:
        # S12 == S21 and S22 == S11.  The phase of S11 is a modelling choice
        # (only |S11| follows from losslessness), so compare magnitudes.
        return ScenarioResult(s11, s21, s12=s21, s22=s11)


# ----------------------------------------------------------------------
# Cable: matched transmission line
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CableLine:
    """A matched lossy transmission line (S11 near zero, S21 delayed).

    ``length_m`` with ``velocity_factor`` gives the one-way electrical delay
    ``length_m / (velocity_factor * c)``.  ``loss_db_per_100m`` is specified
    at ``reference_frequency`` and scales with ``sqrt(f)`` (skin-effect
    approximation) when ``skin_effect`` is true, else linearly.
    """

    length_m: float = 10.0
    velocity_factor: float = 0.66
    loss_db_per_100m: float = 10.0
    reference_frequency: float = 100e6
    skin_effect: bool = True

    @property
    def delay(self) -> float:
        """One-way propagation delay in seconds."""
        return self.length_m / (self.velocity_factor * SPEED_OF_LIGHT)

    @property
    def round_trip_delay(self) -> float:
        return 2.0 * self.delay

    def insertion_loss_db(self, freq: float) -> float:
        if freq <= 0:
            return 0.0
        attenuation = self.loss_db_per_100m * (self.length_m / 100.0)
        if self.skin_effect:
            ratio = math.sqrt(freq / self.reference_frequency)
        else:
            ratio = freq / self.reference_frequency
        return attenuation * ratio

    def insertion_phase(self, freq: float) -> float:
        return -2.0 * math.pi * freq * self.delay

    def s21(self, freq: float) -> complex:
        magnitude = 10.0 ** (-self.insertion_loss_db(freq) / 20.0)
        return cmath.rect(magnitude, self.insertion_phase(freq))

    def s11(self, freq: float, return_loss_db: float = 40.0) -> complex:
        """Small residual reflection; default is a well-matched load."""
        magnitude = 10.0 ** (-return_loss_db / 20.0)
        # Reflection appears to travel the round trip.
        return cmath.rect(magnitude, self.insertion_phase(freq) * 2.0)

    def __call__(self, freq: float) -> ScenarioResult:
        # A cable is reciprocal and symmetric end to end, so the reverse
        # channels repeat the forward ones (return loss on either port).
        s21 = self.s21(freq)
        s11 = self.s11(freq)
        return ScenarioResult(s11, s21, s12=s21, s22=s11)


# ----------------------------------------------------------------------
# Through / attenuator helpers
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Attenuator:
    """A flat attenuator (for filter-test ripple verification)."""

    attenuation_db: float = 3.0
    return_loss_db: float = 30.0

    def __call__(self, freq: float) -> ScenarioResult:
        s21 = cmath.rect(10.0 ** (-self.attenuation_db / 20.0), 0.0)
        s11 = cmath.rect(10.0 ** (-self.return_loss_db / 20.0), 0.0)
        # A resistive pad is reciprocal and symmetric: S12 == S21 and
        # S22 == S11 (same return loss on either port).
        return ScenarioResult(s11, s21, s12=s21, s22=s11)


# ----------------------------------------------------------------------
# Composite models: dual-band antenna, band-pass filter
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class DualBandAntenna:
    """Two resonators driven in parallel on one port (marker/tuning tests).

    Each band is a :class:`AntennaResonance`; the two are combined by adding
    their admittances (``Y = 1/Z``), which is what two antennas fed from the
    same port do, so the model has a match minimum at *both* resonant
    frequencies and a mismatch between them.  Overlapping bands interact
    (a second resonator loads the first) instead of being summed as
    independent response curves.

    Still a one-port: it declares ``data 0`` only, so a marker or limit check
    on S21 finds no channel instead of a fabricated zero.
    """

    low_band: AntennaResonance = field(
        default_factory=lambda: AntennaResonance(
            resonant_frequency=7.1e6, q_factor=30.0
        )
    )
    high_band: AntennaResonance = field(
        default_factory=lambda: AntennaResonance(
            resonant_frequency=14.2e6, q_factor=25.0
        )
    )

    supported_channels = ("data 0",)
    scenario_key = "dual_band"
    display_name = "Dual-band antenna"

    def impedance(self, freq: float) -> complex:
        """Parallel combination of the two resonators."""
        low = self.low_band.impedance(freq)
        high = self.high_band.impedance(freq)
        return 1.0 / (1.0 / low + 1.0 / high)

    def s11(self, freq: float) -> complex:
        return _gamma_from_impedance(self.impedance(freq))

    def vswr(self, freq: float) -> float:
        magnitude = abs(self.s11(freq))
        if magnitude >= 1.0:
            return math.inf
        return (1.0 + magnitude) / (1.0 - magnitude)

    def __call__(self, freq: float) -> ScenarioResult:
        return ScenarioResult(self.s11(freq), complex(0.0, 0.0))


@dataclass(frozen=True)
class BandPassFilter:
    """Band-pass built by frequency-transforming a low-pass prototype.

    The low-pass prototype is evaluated at
    ``Omega = (f0 / bandwidth) * (f / f0 - f0 / f)``, so the passband centre
    is ``center_frequency``, the ``-3 dB`` points are exactly one
    ``bandwidth`` apart (a 3 dB band-pass, whatever the prototype's own
    edges), and the response rolls off on both sides.  ``Omega`` is used as
    the prototype's frequency, including its sign below ``f0``, which is the
    standard analytic continuation of ``s = j*Omega`` -- no separate
    low-side model is invented.

    An ``order``-pole low-pass prototype becomes a ``2 * order``-pole
    band-pass, as the transformation doubles the pole count.
    """

    center_frequency: float = 100e6
    bandwidth: float = 10e6
    order: int = 3

    @property
    def prototype(self) -> ButterworthLowPass:
        """The low-pass prototype, normalised to ``1 rad/s``."""
        return ButterworthLowPass(cutoff_frequency=1.0, order=self.order)

    def normalized_frequency(self, freq: float) -> float:
        """``Omega``: the prototype frequency this band-pass point maps to."""
        if freq <= 0.0:
            return -math.inf
        ratio = self.center_frequency / self.bandwidth
        return ratio * (
            freq / self.center_frequency - self.center_frequency / freq
        )

    def magnitude_db(self, freq: float) -> float:
        omega = self.normalized_frequency(freq)
        if not math.isfinite(omega):
            return -math.inf
        return self.prototype.magnitude_db(omega)

    scenario_key = "bandpass"
    display_name = "Band-pass filter"

    def __call__(self, freq: float) -> ScenarioResult:
        omega = self.normalized_frequency(freq)
        if not math.isfinite(omega):
            # f <= 0: fully stopped twice over, |S21| = 0, |S11| = 1.
            zero = complex(0.0, 0.0)
            one = complex(1.0, 0.0)
            return ScenarioResult(one, zero, s12=zero, s22=one)
        s21, s11 = self.prototype.response(omega)
        return ScenarioResult(s11, s21, s12=s21, s22=s11)


Scenario = Callable[[float], ScenarioResult]

SCENARIO_TYPES = {
    "antenna": AntennaResonance,
    "filter": ButterworthLowPass,
    "cable": CableLine,
    "attenuator": Attenuator,
    "dual_band": DualBandAntenna,
    "bandpass": BandPassFilter,
}


def build_scenario(name: str, **kwargs) -> Scenario:
    """Instantiate a built-in scenario by name."""
    try:
        factory = SCENARIO_TYPES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown scenario {name!r}; choose from {sorted(SCENARIO_TYPES)}"
        ) from exc
    return factory(**kwargs)


#: Named defaults used by the demonstration mode.
DEFAULT_SCENARIOS: dict[str, Scenario] = {
    "antenna": AntennaResonance(),
    "filter": ButterworthLowPass(),
    "cable": CableLine(),
    "attenuator": Attenuator(),
    "dual_band": DualBandAntenna(),
    "bandpass": BandPassFilter(),
}
