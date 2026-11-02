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
"""Ideal L-network matching assistant (ANALYSIS-03).

Given the impedance a load presents at *one* frequency, this module proposes
the ideal two-element L-networks that transform it into a purely real
reference impedance, and reports for every proposal the impedance its own
component values produce.  The input impedance is always recomputed *from the
component values*, never copied from the load, so a proposal that does not
check out cannot be published as a solution.

The two branches
----------------
A two-element L-network has one series and one shunt branch, and there are two
ways to draw it:

* :attr:`NetworkForm.SERIES_AT_INPUT` - the series branch at the input, the
  shunt branch across the load.  Its conventional names are
  ``series-L/shunt-C`` and ``series-C/shunt-L``.
* :attr:`NetworkForm.SHUNT_AT_INPUT` - the shunt branch at the input, the
  series branch in front of the load.  Its conventional names are
  ``shunt-C/series-L`` and ``shunt-L/series-C``.

Each form has two mathematical solutions, and both are reported.  A resistive
load uses the form on the far side of the reference (a shunt-first network for
a load below the reference, a shunt-across-the-load network for one above it),
so the four names above are exactly what a resistive load produces; a reactive
load can make both forms realisable and can force mixed pairs such as
``series-C/shunt-C``.  Those are genuine networks, so the name is derived from
the components that came out instead of being forced into a name they
contradict.  See :data:`CONVENTIONAL_TOPOLOGIES`.

Conventions
-----------
* ``load`` is the impedance looking *into the load*, in ohms; ``reference`` is
  the real impedance wanted at the network input; ``frequency`` is in hertz.
* Component values are always the physical value - henry or farad - and the
  relation to reactance is ``X = omega * L`` or ``X = -1 / (omega * C)`` with
  ``omega = 2 * pi * frequency``.  No solution stores a bare reactance as if it
  were a component.
* Unavailable is a first-class outcome.  A refusal carries a ``reason`` and no
  components; a form that has no physical solution is omitted and its reason is
  kept in :attr:`MatchSolutions.reasons`, so the caller can say *why* a branch
  is missing instead of silently showing fewer networks.
* Tolerances are named, not implied: :data:`REACTANCE_TOLERANCE` decides when
  a load's reactance is below what can be corrected,
  :data:`RESISTIVE_MATCH_TOLERANCE` when its resistance already equals the
  reference, and :data:`SELF_CHECK_TOLERANCE` bounds the self-check.
* A lossless network cannot turn a load with no real part into a real input
  impedance, so a purely reactive load is refused rather than matched with
  enormous components.

Example
-------
>>> result = solve_match(complex(25, 30), 50.0, 10e6)
>>> print(result.solutions[0].describe())
shunt-C/series-C (preferred) at 1e+07 Hz: series 3.1831 nF, shunt 318.31 pF \
-> input 50-8.33e-15j ohm
>>> result.solutions[0].description
'series 3.1831 nF, shunt 318.31 pF'
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Iterable

#: A load whose reactance is smaller than this fraction of the larger of its
#: resistance and the reference is indistinguishable from a resistor: at that
#: size the component that would correct it is far outside any physical range.
#: Such a load is designed for as a pure resistor, and one whose resistance
#: already equals the reference is refused rather than emitted with a
#: rounding-noise component.
REACTANCE_TOLERANCE = 1e-9

#: Relative tolerance on the resistance of a purely resistive load that counts
#: as "already equal to the reference".  Below it there is nothing to match.
RESISTIVE_MATCH_TOLERANCE = 1e-9

#: Relative tolerance, against ``max(|reference|, |load|)``, within which a
#: solution's recomputed input impedance must reproduce the reference.  It is
#: a bug detector, not a quality statement: the algebra is exact, so a wrong
#: sign or a mixed-up element misses by far more than this.
SELF_CHECK_TOLERANCE = 1e-6

#: Relative band inside which two branches count as having the same loaded Q.
#: Ties are broken towards the inductive series element, because that is the
#: low-pass choice; without the band the tie would be decided by rounding.
TIE_BREAK_TOLERANCE = 1e-9

#: Reference impedance used when the caller gives none.
DEFAULT_REFERENCE = 50.0

#: Design frequency used when the caller gives none.  Every result states the
#: frequency it was designed at, so a forgotten argument is visible.
DEFAULT_FREQUENCY = 1e6

#: The four conventional L-network names, in the order an operator reads them.
#: A resistive load produces exactly these four; a reactive load may instead
#: produce a mixed pair (for example ``series-C/shunt-C``), which is named for
#: the components actually needed.
CONVENTIONAL_TOPOLOGIES: tuple[str, ...] = (
    "series-L/shunt-C",
    "series-C/shunt-L",
    "shunt-L/series-C",
    "shunt-C/series-L",
)


class MatchStatus(Enum):
    """Outcome category of a match request."""

    OK = "ok"
    UNAVAILABLE = "unavailable"


class NetworkForm(Enum):
    """How the two branches of the L-network are connected.

    ``SERIES_AT_INPUT`` is a series branch at the input with the shunt branch
    across the load; ``SHUNT_AT_INPUT`` is a shunt branch at the input with the
    series branch in front of the load.  They are different two-ports, not two
    drawings of the same one.
    """

    SERIES_AT_INPUT = "series_at_input"
    SHUNT_AT_INPUT = "shunt_at_input"


class MatchBranch(Enum):
    """Which of a form's two mathematical solutions a result is.

    The lower loaded Q wins, because it is the wider and more tolerant
    network; a tie goes to the inductive series element, which is the low-pass
    branch a designer picks when nothing else separates the two.
    """

    PREFERRED = "preferred"
    ALTERNATE = "alternate"


class ElementKind(Enum):
    """The two kinds of ideal component an L-network is built from."""

    INDUCTOR = "L"
    CAPACITOR = "C"


_PREFIXES: dict[ElementKind, tuple[tuple[float, str], ...]] = {
    ElementKind.INDUCTOR: (
        (1e-9, "nH"),
        (1e-6, "µH"),
        (1e-3, "mH"),
        (1.0, "H"),
    ),
    ElementKind.CAPACITOR: (
        (1e-15, "fF"),
        (1e-12, "pF"),
        (1e-9, "nF"),
        (1e-6, "µF"),
        (1e-3, "mF"),
        (1.0, "F"),
    ),
}

_FORM_LABEL: dict[NetworkForm, str] = {
    NetworkForm.SERIES_AT_INPUT: (
        "series branch at the input, shunt across the load"
    ),
    NetworkForm.SHUNT_AT_INPUT: (
        "shunt branch at the input, series in front of the load"
    ),
}


def format_engineering(value: float, kind: ElementKind) -> str:
    """``value`` henry or farad with the SI prefix that keeps it readable.

    The largest prefix not exceeding the value is used, so an inductance reads
    in nH, µH or mH and a capacitance in fF, pF, nF or µF; a value below the
    smallest prefix of its ladder is reported in that prefix as a fraction of
    it rather than rounded to zero.
    """
    ladder = _PREFIXES[kind]
    scale, name = ladder[0]
    for candidate, candidate_name in ladder:
        if abs(value) >= candidate:
            scale, name = candidate, candidate_name
    return f"{value / scale:.6g} {name}"


def _kind_for(reactance: float) -> ElementKind:
    return ElementKind.INDUCTOR if reactance > 0.0 else ElementKind.CAPACITOR


def _reactance(kind: ElementKind, value: float, frequency: float) -> float:
    """Reactance in ohms of ``value`` henry or farad at ``frequency`` hertz."""
    omega = 2.0 * math.pi * frequency
    if kind is ElementKind.INDUCTOR:
        return omega * value
    return -1.0 / (omega * value)


def _element_value(kind: ElementKind, reactance: float, frequency: float):
    """The henry or farad value whose reactance is ``reactance``."""
    omega = 2.0 * math.pi * frequency
    if kind is ElementKind.INDUCTOR:
        return reactance / omega
    return -1.0 / (omega * reactance)


@dataclass(frozen=True)
class MatchSolution:
    """One ideal L-network, with the load it matches and what it does.

    ``series_impedance`` and ``shunt_impedance`` are the branch impedances at
    :attr:`frequency`; ``input_impedance`` is the impedance the whole network
    presents at its input, recomputed from ``series_value`` and
    ``shunt_value`` through the textbook combination of the form.  The
    self-check compares the two, so an inconsistent network raises rather than
    being offered as a slightly worse match.
    """

    topology: str
    form: NetworkForm
    branch: MatchBranch
    frequency: float
    load: complex
    reference: float
    series_kind: ElementKind
    shunt_kind: ElementKind
    series_value: float
    shunt_value: float
    series_impedance: complex
    shunt_impedance: complex
    input_impedance: complex

    @property
    def residual_imag(self) -> float:
        """Imaginary part of the input impedance, in ohms (ideally zero)."""
        return self.input_impedance.imag

    @property
    def loaded_q(self) -> float:
        """Loaded Q of the branch: series reactance over the resistance seen.

        The series element sees the reference on a series-at-input network and
        the load resistance on a shunt-at-input one; for a resistive load both
        branches of a form have the same Q, which is why the tie-break below
        exists.
        """
        seen = (
            self.reference
            if self.form is NetworkForm.SERIES_AT_INPUT
            else self.load.real
        )
        return abs(self.series_impedance.imag) / seen

    @property
    def matched(self) -> bool:
        """True when the recomputed input is real and equal to the reference."""
        scale = max(abs(self.reference), abs(self.load), 1.0)
        tolerance = SELF_CHECK_TOLERANCE * scale
        return (
            abs(self.input_impedance.real - self.reference) <= tolerance
            and abs(self.input_impedance.imag) <= tolerance
        )

    @property
    def conventional(self) -> bool:
        """True when :attr:`topology` is one of the four textbook names.

        A reactive load can need the same kind of element in both branches
        (two capacitors, for instance).  That network is real and is reported,
        but its name is not one of the conventional four, so a caller that
        wants only textbook branches can filter on this rather than matching
        strings.
        """
        return self.topology in CONVENTIONAL_TOPOLOGIES

    @property
    def description(self) -> str:
        """Both branches in engineering units, e.g. ``549.2 nH``.

        Every branch is named with its own unit, so a network whose two
        branches are the same kind of component (a load can need two
        capacitors) cannot be misread as an L and a C.
        """
        return (
            f"series "
            f"{format_engineering(self.series_value, self.series_kind)}, "
            f"shunt {format_engineering(self.shunt_value, self.shunt_kind)}"
        )

    def describe(self) -> str:
        """One line: topology, branch, values and the impedance they produce."""
        return (
            f"{self.topology} ({self.branch.value}) at "
            f"{self.frequency:.6g} Hz: {self.description} -> "
            f"input {self.input_impedance.real:.9g}"
            f"{self.input_impedance.imag:+.3g}j ohm"
        )

    def at(self, frequency: float) -> MatchSolution:
        """The same components and the same load, evaluated at ``frequency``.

        The component values stay put while ``omega`` changes, so each branch
        reactance is recomputed from its own element value - an inductor's
        grows with frequency and a capacitor's falls.  The load is held at its
        design-frequency value; nothing is matched at the new frequency unless
        the load happens to scale with frequency in the same way.  This is how
        a result is plotted, not a second match.
        """
        series, shunt, input_impedance = _project(self, frequency, self.load)
        return replace(
            self,
            frequency=frequency,
            series_impedance=series,
            shunt_impedance=shunt,
            input_impedance=input_impedance,
        )


@dataclass(frozen=True)
class MatchSolutions:
    """What the L-network search found for one load, and what it refused.

    Three kinds of absence are kept apart, because they mean different things:
    :attr:`reason` is why the request produced no result at all,
    :attr:`reasons` is why individual branches were omitted from an otherwise
    usable result, and :attr:`notes` records a substitution that was made to
    the load before solving (such as dropping a sub-tolerance reactance).
    """

    status: MatchStatus
    load: complex
    reference: float
    frequency: float
    solutions: tuple[MatchSolution, ...] = ()
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is MatchStatus.OK

    @property
    def unavailable(self) -> bool:
        return self.status is MatchStatus.UNAVAILABLE

    @property
    def preferred(self) -> MatchSolution | None:
        """The first preferred branch, or the first solution, or nothing.

        Solutions are ordered with preferred branches first and each group by
        increasing loaded Q, so this is the network to try first.
        """
        for solution in self.solutions:
            if solution.branch is MatchBranch.PREFERRED:
                return solution
        return self.solutions[0] if self.solutions else None

    def describe(self) -> str:
        """Engineering summary, with every note and every omitted branch."""
        if not self.ok:
            return f"unavailable ({self.reason or 'no solution'})"
        head = (
            f"{self.load.real:.6g}{self.load.imag:+.6g}j ohm -> "
            f"{self.reference:.6g} ohm at {self.frequency:.6g} Hz"
        )
        lines = [head]
        lines += [
            f"  {index + 1}. {solution.describe()}"
            for index, solution in enumerate(self.solutions)
        ]
        lines += [f"  (omitted) {reason}" for reason in self.reasons]
        lines += [f"  (note) {note}" for note in self.notes]
        return "\n".join(lines)

    def predicted_response(
        self,
        freqs: Iterable[float],
        *,
        load_at: Callable[[float], complex] | None = None,
    ) -> list[list[tuple[float, complex]]]:
        """Input impedance of every proposed network across ``freqs``.

        One row per solution, in solution order, each a list of
        ``(frequency, input impedance in ohms)``.  The component values are
        held at what the match produced and each reactance is recomputed at
        every frequency, so the sweep shows where a network stops matching.

        The load is held at its design-frequency value unless ``load_at`` is
        given, in which case the caller supplies the load (typically the
        measured impedance) at each frequency.  Pure: no argument is mutated
        and nothing is written to the result.
        """
        if not self.ok:
            raise ValueError(f"no solution to project: {self.reason}")
        grid = [float(frequency) for frequency in freqs]
        for frequency in grid:
            if not _finite_positive(frequency):
                raise ValueError(
                    "frequencies must be finite and positive, in hertz: "
                    f"{frequency!r}"
                )
        response: list[list[tuple[float, complex]]] = []
        for solution in self.solutions:
            row: list[tuple[float, complex]] = []
            for frequency in grid:
                load = (
                    solution.load
                    if load_at is None
                    else complex(load_at(frequency))
                )
                _, _, input_impedance = _project(solution, frequency, load)
                row.append((frequency, input_impedance))
            response.append(row)
        return response


def _network_input(
    form: NetworkForm, load: complex, series: complex, shunt: complex
) -> complex:
    """Input impedance of the drawn ladder, from its two branch impedances."""
    if form is NetworkForm.SERIES_AT_INPUT:
        return series + (load * shunt) / (load + shunt)
    return (shunt * (series + load)) / (shunt + series + load)


def _project(
    solution: MatchSolution, frequency: float, load: complex
) -> tuple[complex, complex, complex]:
    """Branch impedances and input impedance at ``frequency`` for ``load``."""
    if not _finite_positive(frequency):
        raise ValueError(
            f"frequency must be finite and positive, in hertz: {frequency!r}"
        )
    series = complex(
        0.0, _reactance(solution.series_kind, solution.series_value, frequency)
    )
    shunt = complex(
        0.0, _reactance(solution.shunt_kind, solution.shunt_value, frequency)
    )
    return series, shunt, _network_input(solution.form, load, series, shunt)


# ----------------------------------------------------------------------
# Refusals and screening
# ----------------------------------------------------------------------
def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _finite_positive(value: float) -> bool:
    return _finite(value) and value > 0


def _unavailable(
    load: complex,
    reference: float,
    frequency: float,
    reason: str,
    *,
    reasons: tuple[str, ...] = (),
) -> MatchSolutions:
    """A refusal: no solutions, but every omitted branch's reason kept.

    A refusal never carries a screening note, because screening only drops a
    reactance from a load that goes on to be matched: when it instead refuses
    the load the reason is about the load itself.
    """
    return MatchSolutions(
        status=MatchStatus.UNAVAILABLE,
        load=load,
        reference=reference,
        frequency=frequency,
        reason=reason,
        reasons=reasons,
    )


def _screen(load: complex, reference: float) -> tuple[complex, str, str]:
    """Reduce an insignificant reactance, or say why nothing can be done.

    Returns ``(load, refusal, note)``.  A refusal is a reason to stop; a note
    is a provenance line for a solution that is still produced.  A reactance
    below :data:`REACTANCE_TOLERANCE` is dropped - correcting it would need a
    component far outside any physical range - and that substitution is always
    reported, because a match for 25 + 1e-15j ohm is a match for 25 ohm.  A
    load that is then indistinguishable from the reference is refused rather
    than "matched" with a component sized by rounding noise.
    """
    if load.real < 0.0:
        return (
            load,
            f"load {load.real:.6g}{load.imag:+.6g}j ohm has negative "
            "resistance (an active device); the passive L-network formulas "
            "do not apply",
            "",
        )
    if load.real == 0.0:
        return (
            load,
            f"load {load.real:.6g}{load.imag:+.6g}j ohm has no real part; a "
            "lossless two-element network can only turn a pure reactance "
            "into another pure reactance, so it can never present a real "
            f"{reference:.6g} ohm input",
            "",
        )
    limit = REACTANCE_TOLERANCE * max(abs(load.real), reference)
    if abs(load.imag) > limit:
        return load, "", ""

    resistive = complex(load.real, 0.0)
    note = (
        f"load reactance {load.imag:+.3g} ohm is below the stated tolerance "
        f"{limit:.3g} ohm and was treated as zero; the solution is for "
        f"{resistive.real:.9g} ohm"
        if load.imag != 0.0
        else ""
    )
    if not math.isclose(
        resistive.real,
        reference,
        rel_tol=RESISTIVE_MATCH_TOLERANCE,
        abs_tol=0.0,
    ):
        return resistive, "", note
    if load.imag == 0.0:
        return (
            resistive,
            f"load {resistive.real:.9g} ohm is already the reference "
            f"{reference:.9g} ohm (within {RESISTIVE_MATCH_TOLERANCE:g} "
            "relative); there is nothing to match, and an L-network of 0 H "
            "and 0 F is not a solution",
            "",
        )
    return (
        resistive,
        f"load {load.real:.9g}{load.imag:+.3g}j ohm has a reactance of "
        f"{abs(load.imag):.3g} ohm, below the stated tolerance {limit:.3g} "
        f"ohm, and a resistance already equal to the reference "
        f"{reference:.9g} ohm; the load is indistinguishable from a matched "
        "one, so no component can be stated",
        "",
    )


# ----------------------------------------------------------------------
# The algebra of the two forms
# ----------------------------------------------------------------------
def _quadratic_roots(a: float, b: float, c: float) -> tuple[float, ...] | None:
    """Real roots of ``a x^2 + b x + c``, or None, without cancellation loss.

    The larger-magnitude root is computed by the textbook formula and the other
    from Vieta's product ``c / q``, which keeps a nearly-degenerate quadratic
    from losing the small root to cancellation.
    """
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    if discriminant == 0.0:
        return (-b / (2.0 * a),)
    q = -0.5 * (b + math.copysign(math.sqrt(discriminant), b))
    return (q / a, c / q)


def _series_at_input_pairs(
    load: complex, reference: float
) -> tuple[list[tuple[float, float]], str]:
    """``(series, shunt)`` reactance pairs for the series-at-input form.

    The shunt branch sees ``R + j(X + Xp)``, so its parallel combination with
    the series branch is real only when
    ``R * Xp**2 = Rr * (R**2 + (X + Xp)**2)``, a quadratic in ``Xp``.  The
    series reactance then cancels the remaining imaginary part:
    ``Xs = -Xp * (R**2 + X * (X + Xp)) / (R**2 + (X + Xp)**2)``.
    """
    resistance = load.real
    reactance = load.imag
    a = reference - resistance
    b = 2.0 * reference * reactance
    c = reference * (resistance * resistance + reactance * reactance)
    if a == 0.0:
        if b == 0.0:
            return [], (
                "the load resistance is already the reference, so the shunt "
                "branch would have to be an open circuit"
            )
        roots: tuple[float, ...] = (-c / b,)
    else:
        found = _quadratic_roots(a, b, c)
        if found is None:
            return [], (
                "no real shunt reactance exists (the discriminant "
                f"{b * b - 4.0 * a * c:.3g} is negative): the load resistance "
                "and reactance are both too small for a shunt branch across "
                "the load to reach the reference"
            )
        roots = found
    pairs = []
    for shunt in roots:
        shifted = reactance + shunt
        denominator = resistance * resistance + shifted * shifted
        series = (
            -shunt
            * (resistance * resistance + reactance * shifted)
            / denominator
        )
        pairs.append((series, shunt))
    return _unique(pairs), ""


def _shunt_at_input_pairs(
    load: complex, reference: float
) -> tuple[list[tuple[float, float]], str]:
    """``(series, shunt)`` reactance pairs for the shunt-at-input form.

    The shunt branch puts ``-j Xp`` across the load, and the series branch then
    sees a real ``Rr`` only when the shunt combination is
    ``R - j u`` with ``u**2 = R * (Rr - R)``.  That gives
    ``Xp = -Rr * R / u`` and ``Xs = u - X`` for each sign of ``u``.
    """
    resistance = load.real
    reactance = load.imag
    radicand = resistance * (reference - resistance)
    if radicand <= 0.0:
        return [], (
            f"load resistance {resistance:.6g} ohm is not below the reference "
            f"{reference:.6g} ohm, and a shunt-first network can only raise a "
            "resistance"
        )
    root = math.sqrt(radicand)
    pairs = []
    for u in (root, -root):
        pairs.append((u - reactance, -(reference * resistance) / u))
    return _unique(pairs), ""


def _close(first: float, second: float) -> bool:
    scale = max(abs(first), abs(second))
    if scale == 0.0:
        return True
    return abs(first - second) <= 1e-12 * scale


def _unique(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop pairs repeated by a degenerate quadratic, keeping the first."""
    kept: list[tuple[float, float]] = []
    for series, shunt in pairs:
        if any(
            _close(series, other_series) and _close(shunt, other_shunt)
            for other_series, other_shunt in kept
        ):
            continue
        kept.append((series, shunt))
    return kept


def _make_solution(
    form: NetworkForm,
    load: complex,
    reference: float,
    frequency: float,
    reactances: tuple[float, float],
) -> tuple[MatchSolution | None, str]:
    """Build one solution from ``(series, shunt)`` reactances, or refuse it."""
    series_reactance, shunt_reactance = reactances
    for label, reactance in (
        ("series", series_reactance),
        ("shunt", shunt_reactance),
    ):
        if not math.isfinite(reactance):
            return None, (
                f"its {label} branch would need a reactance of {reactance!r} "
                "ohm, which is not finite"
            )
        if reactance == 0.0:
            return None, (
                f"its {label} branch would need zero reactance, which is a "
                "short rather than a component, so no two-element network "
                "exists on this branch"
            )
    series_kind = _kind_for(series_reactance)
    shunt_kind = _kind_for(shunt_reactance)
    series_value = _element_value(series_kind, series_reactance, frequency)
    shunt_value = _element_value(shunt_kind, shunt_reactance, frequency)
    for label, kind, value in (
        ("series", series_kind, series_value),
        ("shunt", shunt_kind, shunt_value),
    ):
        if not (math.isfinite(value) and value > 0.0):
            return None, (
                f"its {label} {kind.value} would be {value!r} "
                f"{'henry' if kind is ElementKind.INDUCTOR else 'farad'}, "
                "which is not a positive, finite component value"
            )
    series_impedance = complex(
        0.0, _reactance(series_kind, series_value, frequency)
    )
    shunt_impedance = complex(
        0.0, _reactance(shunt_kind, shunt_value, frequency)
    )
    input_impedance = _network_input(
        form, load, series_impedance, shunt_impedance
    )
    solution = MatchSolution(
        topology=_topology_name(form, series_kind, shunt_kind),
        form=form,
        branch=MatchBranch.PREFERRED,
        frequency=frequency,
        load=load,
        reference=reference,
        series_kind=series_kind,
        shunt_kind=shunt_kind,
        series_value=series_value,
        shunt_value=shunt_value,
        series_impedance=series_impedance,
        shunt_impedance=shunt_impedance,
        input_impedance=input_impedance,
    )
    if not solution.matched:
        raise AssertionError(
            "L-network solution does not reproduce the reference impedance "
            f"from its component values: {solution.describe()}"
        )
    return solution, ""


def _topology_name(
    form: NetworkForm, series_kind: ElementKind, shunt_kind: ElementKind
) -> str:
    """The conventional name for the components that came out.

    The branch connected to the input is named first, so this reproduces the
    four conventional names for a resistive load and names a reactive load's
    mixed pairs for what they are.
    """
    if form is NetworkForm.SERIES_AT_INPUT:
        return f"series-{series_kind.value}/shunt-{shunt_kind.value}"
    return f"shunt-{shunt_kind.value}/series-{series_kind.value}"


def _label(built: list[MatchSolution]) -> list[MatchSolution]:
    """Mark one branch preferred: lower Q, then the inductive series element.

    The Q comparison is a tolerance, not a strict minimum: for a resistive
    load both branches of one form have exactly the same Q, and comparing
    their floating-point values would let the last bit of rounding decide
    which network is offered first.
    """
    if not built:
        return []
    best = min(solution.loaded_q for solution in built)
    window = best * (1.0 + TIE_BREAK_TOLERANCE)
    ordered = sorted(built, key=lambda solution: _branch_key(solution, window))
    return [
        replace(
            solution,
            branch=(
                MatchBranch.PREFERRED if index == 0 else MatchBranch.ALTERNATE
            ),
        )
        for index, solution in enumerate(ordered)
    ]


def _branch_key(
    solution: MatchSolution, window: float
) -> tuple[int, int, float, float, str]:
    return (
        0 if solution.loaded_q <= window else 1,
        0 if solution.series_kind is ElementKind.INDUCTOR else 1,
        solution.series_value,
        solution.shunt_value,
        solution.topology,
    )


def _order_key(solution: MatchSolution) -> tuple[int, float, str, float]:
    return (
        0 if solution.branch is MatchBranch.PREFERRED else 1,
        solution.loaded_q,
        solution.topology,
        solution.series_value,
    )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def solve_match(
    load: complex,
    reference: float = DEFAULT_REFERENCE,
    frequency: float = DEFAULT_FREQUENCY,
) -> MatchSolutions:
    """Every physical L-network matching ``load`` to ``reference``, or why not.

    ``load`` is the impedance looking into the load in ohms, ``reference`` the
    real impedance wanted at the network input and ``frequency`` the design
    frequency in hertz.  Both forms of the L-network are solved and both of
    their branches kept when they exist; a branch that would need a zero,
    negative or non-finite component is omitted with a reason in
    :attr:`MatchSolutions.reasons`.
    """
    load = complex(load)
    problems = []
    if not (_finite(load.real) and _finite(load.imag)):
        problems.append(f"load {load!r} is not finite")
    if not _finite(reference):
        problems.append(f"reference {reference!r} is not finite")
    elif reference <= 0.0:
        problems.append(f"reference {reference!r} ohm is not positive")
    if not _finite(frequency):
        problems.append(f"frequency {frequency!r} is not finite")
    elif frequency <= 0.0:
        problems.append(f"frequency {frequency!r} Hz is not positive")
    if problems:
        return _unavailable(load, reference, frequency, "; ".join(problems))
    reference = float(reference)
    frequency = float(frequency)

    screened, refusal, screening_note = _screen(load, reference)
    if refusal:
        return _unavailable(screened, reference, frequency, refusal)

    candidates: list[MatchSolution] = []
    reasons: list[str] = []
    notes = (screening_note,) if screening_note else ()
    for form, solve in (
        (NetworkForm.SERIES_AT_INPUT, _series_at_input_pairs),
        (NetworkForm.SHUNT_AT_INPUT, _shunt_at_input_pairs),
    ):
        pairs, unavailable = solve(screened, reference)
        if unavailable:
            reasons.append(f"{_FORM_LABEL[form]}: {unavailable}")
            continue
        built: list[MatchSolution] = []
        for reactances in pairs:
            solution, omitted = _make_solution(
                form, screened, reference, frequency, reactances
            )
            if solution is None:
                reasons.append(f"{_FORM_LABEL[form]}: {omitted}")
            else:
                built.append(solution)
        candidates.extend(_label(built))

    if not candidates:
        return _unavailable(
            screened,
            reference,
            frequency,
            "no physical L-network matches this load",
            reasons=tuple(reasons),
        )
    candidates.sort(key=_order_key)
    return MatchSolutions(
        status=MatchStatus.OK,
        load=screened,
        reference=reference,
        frequency=frequency,
        solutions=tuple(candidates),
        reasons=tuple(reasons),
        notes=notes,
    )
