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
attenuation, cable delay) are known in closed form and can be asserted in
tests.

Reference impedance is 50 ohms throughout, matching the application default.
Angles are radians; frequencies are Hz.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Callable

REFERENCE_IMPEDANCE = 50.0
SPEED_OF_LIGHT = 299_792_458.0


def _gamma_from_impedance(impedance: complex) -> complex:
    """Reflection coefficient of a load at the reference impedance."""
    return (impedance - REFERENCE_IMPEDANCE) / (impedance + REFERENCE_IMPEDANCE)


@dataclass(frozen=True)
class ScenarioResult:
    """Complex S11/S21 for one frequency."""

    s11: complex
    s21: complex

    def __iter__(self):
        yield self.s11
        yield self.s21


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

    def __call__(self, freq: float) -> ScenarioResult:
        return ScenarioResult(self.s11(freq), complex(0.0, 0.0))


# ----------------------------------------------------------------------
# Filter: N-th order low-pass Butterworth, 2-port
# ----------------------------------------------------------------------
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
        s11 = cmath.sqrt(max(reflection_sq, 0.0))
        return complex(s21), complex(s11, 0.0)

    def magnitude_db(self, freq: float) -> float:
        s21, _ = self.response(freq)
        magnitude = abs(s21)
        return 20.0 * math.log10(magnitude) if magnitude > 0 else -math.inf

    def __call__(self, freq: float) -> ScenarioResult:
        s21, s11 = self.response(freq)
        return ScenarioResult(s11, s21)


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
        return ScenarioResult(self.s11(freq), self.s21(freq))


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
        return ScenarioResult(s11, s21)


Scenario = Callable[[float], ScenarioResult]

SCENARIO_TYPES = {
    "antenna": AntennaResonance,
    "filter": ButterworthLowPass,
    "cable": CableLine,
    "attenuator": Attenuator,
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
}
