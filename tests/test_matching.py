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
"""Independent verification of the L-network matching assistant (ANALYSIS-03).

Nothing here re-uses the module's own algebra.  Every check either

* substitutes the reported L and C into the textbook impedance of the drawn
  ladder and asserts the result equals the reference analytically, or
* compares against a closed-form value derived from the definition of an
  L-section (``Q = sqrt(R_high / R_low - 1)``, ``X_L = Q * R_low``,
  ``X_C = R_high / Q``), written out in the test.

The reactance of an ideal element is ``omega * L`` and ``-1 / (omega * C)``
with ``omega = 2 * pi * f``; that identity is the bridge between a component
value and the impedance formulas, and it is written out here rather than
imported.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from NanoVNASaver.Matching import (
    CONVENTIONAL_TOPOLOGIES,
    DEFAULT_FREQUENCY,
    DEFAULT_REFERENCE,
    REACTANCE_TOLERANCE,
    RESISTIVE_MATCH_TOLERANCE,
    SELF_CHECK_TOLERANCE,
    ElementKind,
    MatchBranch,
    MatchSolution,
    MatchSolutions,
    MatchStatus,
    NetworkForm,
    format_engineering,
    solve_match,
)

FREQ = 10e6
REFERENCE = 50.0


# ----------------------------------------------------------------------
# The textbook impedances, written out independently of the module
# ----------------------------------------------------------------------
def element_value(solution: MatchSolution, kind: ElementKind) -> float:
    """The reported value of the branch built from ``kind``."""
    if solution.series_kind is kind:
        return solution.series_value
    assert solution.shunt_kind is kind, "one branch of each kind expected"
    return solution.shunt_value


def series_reactance(solution: MatchSolution, frequency: float = FREQ) -> float:
    omega = 2.0 * math.pi * frequency
    value = solution.series_value
    if solution.series_kind is ElementKind.INDUCTOR:
        return omega * value
    return -1.0 / (omega * value)


def shunt_reactance(solution: MatchSolution, frequency: float = FREQ) -> float:
    omega = 2.0 * math.pi * frequency
    value = solution.shunt_value
    if solution.shunt_kind is ElementKind.INDUCTOR:
        return omega * value
    return -1.0 / (omega * value)


def textbook_input_impedance(
    solution: MatchSolution, load: complex, frequency: float = FREQ
) -> complex:
    """The input impedance of the drawn ladder, from the component values.

    Series-at-input: the shunt branch sits across the load, and the series
    branch is added in front of that parallel combination::

        Zpar  = (Zload * jXshunt) / (Zload + jXshunt)
        Zin   = jXseries + Zpar

    Shunt-at-input: the shunt branch is across the input, in series with the
    load and the series branch::

        Zin   = (jXshunt * (Zload + jXseries)) / (jXshunt + Zload + jXseries)
    """
    x_series = series_reactance(solution, frequency)
    x_shunt = shunt_reactance(solution, frequency)
    if solution.form is NetworkForm.SERIES_AT_INPUT:
        parallel = (load * complex(0.0, x_shunt)) / (
            load + complex(0.0, x_shunt)
        )
        return complex(0.0, x_series) + parallel
    branch = load + complex(0.0, x_series)
    return (complex(0.0, x_shunt) * branch) / (complex(0.0, x_shunt) + branch)


def assert_reproduces_reference(
    solution: MatchSolution, load: complex, reference: float
) -> None:
    """The one check every solution must survive."""
    computed = textbook_input_impedance(solution, load, solution.frequency)
    assert computed.imag == pytest.approx(0.0, abs=1e-9), (
        f"solutions must present a real input impedance: {solution.describe()}"
    )
    assert computed.real == pytest.approx(reference, abs=1e-9), (
        "solutions must present the reference resistance: "
        f"{solution.describe()}"
    )
    # The solution reports that same impedance; it must agree with the
    # independent recomputation above, which is the self-check the module
    # claims to perform.
    assert solution.input_impedance.real == pytest.approx(
        computed.real, abs=1e-9
    )
    assert solution.input_impedance.imag == pytest.approx(
        computed.imag, abs=1e-9
    )


# ----------------------------------------------------------------------
# Named analytic cases
# ----------------------------------------------------------------------
def test_ten_ohm_resistor_to_fifty():
    """10 ohm -> 50 ohm: Q = sqrt(50/10 - 1) = 2, the textbook L-section.

    A load below the reference needs a shunt-first network: the shunt branch
    across the 10 ohm resistor raises it to 50 ohm, and the series branch
    cancels the reactance that shunting introduces.  With ``R_low = 10`` and
    ``R_high = 50``::

        Q     = sqrt(R_high / R_low - 1) = 2
        X_p   = R_high / Q = 25 ohm      (shunt element)
        X_s   = Q * R_low  = 20 ohm      (series element)
    """
    load = complex(10.0, 0.0)
    result = solve_match(load, REFERENCE, FREQ)

    assert result.status is MatchStatus.OK
    assert result.ok and not result.unavailable
    assert result.load == load
    assert result.reference == REFERENCE
    assert result.frequency == FREQ

    for solution in result.solutions:
        assert_reproduces_reference(solution, load, REFERENCE)

    # Two branches of the shunt-first form, and no series-first network: a
    # 10 ohm load cannot be raised by shunting across the *input*.
    assert len(result.solutions) == 2
    assert {solution.form for solution in result.solutions} == {
        NetworkForm.SHUNT_AT_INPUT
    }
    assert [solution.branch for solution in result.solutions] == [
        MatchBranch.PREFERRED,
        MatchBranch.ALTERNATE,
    ]
    assert {solution.topology for solution in result.solutions} == {
        "shunt-C/series-L",
        "shunt-L/series-C",
    }

    low_pass = next(
        solution
        for solution in result.solutions
        if solution.topology == "shunt-C/series-L"
    )
    omega = 2.0 * math.pi * FREQ
    # The shunt branch is on the high (reference) side of the L-section, so
    # it is the one that carries X_p = R_high / Q = 25 ohm, and as a
    # capacitor its reactance is -25 ohm.
    assert shunt_reactance(low_pass) == pytest.approx(-25.0, rel=1e-12)
    assert series_reactance(low_pass) == pytest.approx(20.0, rel=1e-12)
    assert low_pass.shunt_value == pytest.approx(
        1.0 / (omega * 25.0), rel=1e-12
    )
    assert low_pass.series_value == pytest.approx(20.0 / omega, rel=1e-12)
    assert low_pass.shunt_kind is ElementKind.CAPACITOR
    assert low_pass.series_kind is ElementKind.INDUCTOR

    # The other branch swaps the roles: a shunt inductor and a series
    # capacitor.  Its series reactance is ``-Q * R_low`` and its shunt
    # reactance ``+R_high / Q``, so it is the exact mirror image.
    high_pass = next(
        solution
        for solution in result.solutions
        if solution.topology == "shunt-L/series-C"
    )
    assert high_pass.shunt_kind is ElementKind.INDUCTOR
    assert high_pass.series_kind is ElementKind.CAPACITOR
    assert shunt_reactance(high_pass) == pytest.approx(25.0, rel=1e-12)
    assert series_reactance(high_pass) == pytest.approx(-20.0, rel=1e-12)
    assert high_pass.loaded_q == pytest.approx(2.0, rel=1e-12)


def test_hundred_ohm_resistor_to_fifty():
    """100 ohm -> 50 ohm: Q = sqrt(100/50 - 1) = 1 on the series-first form.

    ``X_s = Q * R_low = 50 ohm`` and ``X_p = R_high / Q = 100 ohm``; shunting
    across the *input* cannot lower a resistance, so only the series-first
    form is physical.
    """
    load = complex(100.0, 0.0)
    result = solve_match(load, REFERENCE, FREQ)

    assert result.ok
    for solution in result.solutions:
        assert_reproduces_reference(solution, load, REFERENCE)

    assert len(result.solutions) == 2
    assert {solution.form for solution in result.solutions} == {
        NetworkForm.SERIES_AT_INPUT
    }
    assert {solution.topology for solution in result.solutions} == {
        "series-L/shunt-C",
        "series-C/shunt-L",
    }

    for solution in result.solutions:
        assert solution.loaded_q == pytest.approx(1.0, rel=1e-12)
        assert abs(solution.series_impedance.imag) == pytest.approx(
            50.0, rel=1e-12
        )
        assert abs(solution.shunt_impedance.imag) == pytest.approx(
            100.0, rel=1e-12
        )

    # A 100 ohm load below no reference-side raising: the shunt-first form is
    # refused, and the omission is recorded rather than hidden.
    assert any("shunt" in reason for reason in result.reasons)


def test_25_plus_j30_to_fifty():
    """25 + j30 ohm -> 50 ohm: both forms are realisable, four networks.

    The load resistance is below the reference, so the shunt-first form works
    for any reactance: placing ``-j Xp`` across the load leaves a real part
    that the series branch can match.  The series-first form also has a real
    solution because ``R^2 + X^2 = 1525`` exceeds ``Rr (Rr - R) = 1250``.
    """
    load = complex(25.0, 30.0)
    result = solve_match(load, REFERENCE, FREQ)

    assert result.ok
    assert len(result.solutions) == 4
    for solution in result.solutions:
        assert_reproduces_reference(solution, load, REFERENCE)
        assert solution.series_value > 0.0
        assert solution.shunt_value > 0.0
    assert {solution.form for solution in result.solutions} == {
        NetworkForm.SERIES_AT_INPUT,
        NetworkForm.SHUNT_AT_INPUT,
    }

    # Closed-form check for the shunt-first form, derived from the definition:
    #   u = +sqrt(R * (Rr - R)) = sqrt(25 * 25) = +25
    #   the shunt branch must leave Z = R - j u = 25 - j25, which needs
    #   Xp = -Rr * R / u = -50 (a shunt capacitor) and then
    #   Xs = u - X = 25 - 30 = -5 (a series capacitor).
    # With u = -25 the same load needs Z = 25 + j25, so Xp = +50 (a shunt
    # inductor) and Xs = -25 - 30 = -55 (a series capacitor).
    u = math.sqrt(25.0 * (REFERENCE - 25.0))
    assert u == pytest.approx(25.0)
    shunt_first = [
        solution
        for solution in result.solutions
        if solution.form is NetworkForm.SHUNT_AT_INPUT
    ]
    assert len(shunt_first) == 2
    negative_u, positive_u = sorted(shunt_first, key=series_reactance)
    assert series_reactance(positive_u) == pytest.approx(u - 30.0, rel=1e-12)
    assert shunt_reactance(positive_u) == pytest.approx(
        -(REFERENCE * 25.0) / u, rel=1e-12
    )
    assert series_reactance(negative_u) == pytest.approx(-u - 30.0, rel=1e-12)
    assert shunt_reactance(negative_u) == pytest.approx(
        +(REFERENCE * 25.0) / u, rel=1e-12
    )


def test_pure_reactance_is_refused():
    """A lossless network cannot make a pure reactance real."""
    result = solve_match(complex(0.0, 100.0), REFERENCE, FREQ)

    assert result.status is MatchStatus.UNAVAILABLE
    assert result.unavailable
    assert result.solutions == ()
    assert "no real part" in result.reason
    # A reason is carried, not an estimated network.
    assert result.reason != ""


@pytest.mark.parametrize("reactance", [100.0, -25.0, 1e-3])
def test_pure_reactance_never_produces_components(reactance):
    result = solve_match(complex(0.0, reactance), REFERENCE, FREQ)
    assert result.unavailable
    assert result.solutions == ()
    assert result.preferred is None
    assert "no real part" in result.reason


# ----------------------------------------------------------------------
# Every solution, for a spread of loads, against the textbook formulas
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "load",
    [
        complex(10.0, 0.0),
        complex(100.0, 0.0),
        complex(25.0, 30.0),
        complex(50.0, 10.0),
        complex(50.0, -10.0),
        complex(5.0, -40.0),
        complex(200.0, 60.0),
        complex(12.5, 12.5),
        complex(1.0, 0.0),
        complex(1000.0, 0.0),
    ],
)
def test_every_solution_reproduces_the_reference(load):
    result = solve_match(load, REFERENCE, FREQ)
    assert result.ok, result.reason
    assert result.solutions
    for solution in result.solutions:
        assert_reproduces_reference(solution, load, REFERENCE)
        assert solution.residual_imag == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "load",
    [
        complex(10.0, 0.0),
        complex(100.0, 0.0),
        complex(25.0, 30.0),
        complex(50.0, 10.0),
        complex(5.0, -40.0),
        complex(200.0, 60.0),
        complex(0.5, 0.0),
        complex(5000.0, 0.0),
    ],
)
def test_components_are_positive_and_finite(load):
    result = solve_match(load, REFERENCE, FREQ)
    assert result.ok, result.reason
    for solution in result.solutions:
        assert math.isfinite(solution.series_value)
        assert math.isfinite(solution.shunt_value)
        assert solution.series_value > 0.0
        assert solution.shunt_value > 0.0
        # The reported value must be the value whose reactance was used.
        omega = 2.0 * math.pi * FREQ
        if solution.series_kind is ElementKind.INDUCTOR:
            assert solution.series_value * omega == pytest.approx(
                solution.series_impedance.imag, rel=1e-12
            )
        else:
            assert -1.0 / (omega * solution.series_value) == pytest.approx(
                solution.series_impedance.imag, rel=1e-12
            )
        if solution.shunt_kind is ElementKind.INDUCTOR:
            assert solution.shunt_value * omega == pytest.approx(
                solution.shunt_impedance.imag, rel=1e-12
            )
        else:
            assert -1.0 / (omega * solution.shunt_value) == pytest.approx(
                solution.shunt_impedance.imag, rel=1e-12
            )


def test_reactance_sign_matches_the_element_kind():
    for load in (complex(10.0, 0.0), complex(100.0, 0.0), complex(25.0, 30.0)):
        for solution in solve_match(load, REFERENCE, FREQ).solutions:
            assert solution.series_impedance.imag != 0.0
            assert solution.shunt_impedance.imag != 0.0
            if solution.series_kind is ElementKind.INDUCTOR:
                assert solution.series_impedance.imag > 0.0
            else:
                assert solution.series_impedance.imag < 0.0
            if solution.shunt_kind is ElementKind.INDUCTOR:
                assert solution.shunt_impedance.imag > 0.0
            else:
                assert solution.shunt_impedance.imag < 0.0


# ----------------------------------------------------------------------
# Two branches where the mathematics has two
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "load", [complex(10.0, 0.0), complex(100.0, 0.0), complex(25.0, 30.0)]
)
def test_both_branches_are_returned_and_differ(load):
    result = solve_match(load, REFERENCE, FREQ)
    assert result.ok
    branches = [solution.branch for solution in result.solutions]
    assert MatchBranch.PREFERRED in branches
    assert MatchBranch.ALTERNATE in branches
    assert all(
        isinstance(solution, MatchSolution) for solution in result.solutions
    )

    # Genuinely different networks: not the same component values twice.
    signatures = {
        (solution.series_value, solution.shunt_value)
        for solution in result.solutions
    }
    assert len(signatures) == len(result.solutions)

    # And they are not merely mirror-labelled duplicates of one another.
    for first in result.solutions:
        for second in result.solutions:
            if first is second:
                continue
            assert first.series_value != pytest.approx(
                second.series_value, rel=1e-9
            )


def test_preferred_branch_has_the_lower_q():
    load = complex(25.0, 30.0)
    result = solve_match(load, REFERENCE, FREQ)
    preferred = result.preferred
    assert preferred is not None
    assert preferred.branch is MatchBranch.PREFERRED
    assert preferred.loaded_q == pytest.approx(
        min(solution.loaded_q for solution in result.solutions), rel=1e-9
    )


def test_conventional_names_are_reproduced():
    """A resistive load produces exactly the four textbook names."""
    names = set()
    for load in (complex(10.0, 0.0), complex(100.0, 0.0)):
        for solution in solve_match(load, REFERENCE, FREQ).solutions:
            names.add(solution.topology)
            assert solution.conventional
    assert names == set(CONVENTIONAL_TOPOLOGIES)


def test_reactive_load_can_need_a_non_conventional_pair():
    """25 + j30 needs two capacitors in one form; it is named for what it is."""
    result = solve_match(complex(25.0, 30.0), REFERENCE, FREQ)
    assert result.ok
    names = {solution.topology for solution in result.solutions}
    assert "series-C/shunt-C" in names or "shunt-C/series-C" in names
    mixed = [s for s in result.solutions if not s.conventional]
    assert mixed, "the mixed pair must be present, not dropped"
    for solution in mixed:
        assert solution.series_kind is ElementKind.CAPACITOR
        assert solution.shunt_kind is ElementKind.CAPACITOR


# ----------------------------------------------------------------------
# Descriptions and units
# ----------------------------------------------------------------------
def test_describe_uses_engineering_units():
    result = solve_match(complex(10.0, 0.0), REFERENCE, FREQ)
    text = result.solutions[0].describe()
    assert " Hz" in text
    assert "ohm" in text
    assert text.startswith(result.solutions[0].topology)
    assert result.solutions[0].branch.value in text

    whole = result.describe()
    assert whole.startswith("10+0j ohm -> 50 ohm at 1e+07 Hz")
    assert "(omitted)" in whole


def test_format_engineering_prefixes():
    assert format_engineering(1e-9, ElementKind.INDUCTOR) == "1 nH"
    assert format_engineering(2.5e-6, ElementKind.INDUCTOR) == "2.5 µH"
    assert format_engineering(1e-3, ElementKind.INDUCTOR) == "1 mH"
    assert format_engineering(1.0, ElementKind.INDUCTOR) == "1 H"
    assert format_engineering(1e-12, ElementKind.CAPACITOR) == "1 pF"
    assert format_engineering(4.7e-9, ElementKind.CAPACITOR) == "4.7 nF"
    assert format_engineering(2.2e-6, ElementKind.CAPACITOR) == "2.2 µF"
    # Below the smallest prefix: a fraction of it, never a rounded-away zero.
    assert format_engineering(1e-10, ElementKind.INDUCTOR).endswith(" nH")


def test_description_names_both_branches_with_units():
    for solution in solve_match(complex(25.0, 30.0), REFERENCE, FREQ).solutions:
        text = solution.description
        assert text.startswith("series ")
        assert ", shunt " in text
        # Both values are quoted with the unit of their own kind.
        assert text.count("H") + text.count("F") >= 2


def test_result_carries_its_provenance():
    result = solve_match(complex(25.0, 30.0), 75.0, 3.5e6)
    assert result.status is MatchStatus.OK
    assert result.reference == 75.0
    assert result.frequency == 3.5e6
    for solution in result.solutions:
        assert solution.frequency == 3.5e6
        assert solution.reference == 75.0
        assert solution.load == complex(25.0, 30.0)
        assert_reproduces_reference(solution, complex(25.0, 30.0), 75.0)


def test_defaults():
    assert DEFAULT_REFERENCE == 50.0
    assert DEFAULT_FREQUENCY > 0.0
    result = solve_match(complex(10.0, 0.0))
    assert result.ok
    assert result.reference == DEFAULT_REFERENCE
    assert result.frequency == DEFAULT_FREQUENCY


# ----------------------------------------------------------------------
# predicted_response
# ----------------------------------------------------------------------
def test_predicted_response_is_pure_and_matches_at_design_frequency():
    load = complex(10.0, 0.0)
    result = solve_match(load, REFERENCE, FREQ)
    before = tuple(result.solutions)
    snapshot = [
        (s.frequency, s.series_value, s.shunt_value, s.input_impedance)
        for s in result.solutions
    ]

    grid = [1e6, 5e6, FREQ, 20e6, 50e6]
    response = result.predicted_response(grid)

    # Nothing was mutated.
    assert tuple(result.solutions) == before
    assert [
        (s.frequency, s.series_value, s.shunt_value, s.input_impedance)
        for s in result.solutions
    ] == snapshot
    assert result.solutions[0].frequency == FREQ

    assert len(response) == len(result.solutions)
    for row, solution in zip(response, result.solutions, strict=True):
        assert [frequency for frequency, _ in row] == grid
        # At the design frequency the network still matches.
        at_design = row[grid.index(FREQ)][1]
        assert at_design.real == pytest.approx(REFERENCE, abs=1e-9)
        assert at_design.imag == pytest.approx(0.0, abs=1e-9)
        # Away from it, each point is the textbook impedance of the ladder.
        for frequency, impedance in row:
            expected = textbook_input_impedance(solution, load, frequency)
            assert impedance.real == pytest.approx(expected.real, abs=1e-9)
            assert impedance.imag == pytest.approx(expected.imag, abs=1e-9)


def test_predicted_response_follows_a_supplied_load():
    """A caller can hand in the measured impedance at each frequency."""
    load_at = {1e6: complex(20.0, 0.0), 10e6: complex(10.0, 0.0)}
    result = solve_match(complex(10.0, 0.0), REFERENCE, 10e6)
    response = result.predicted_response(
        sorted(load_at), load_at=lambda frequency: load_at[frequency]
    )
    for row, solution in zip(response, result.solutions, strict=True):
        at_design = row[1][1]
        assert at_design.real == pytest.approx(REFERENCE, abs=1e-9)
        # At 1 MHz the load is 20 ohm instead of 10, so the fixed components
        # no longer match; the number is still the textbook impedance.
        expected = textbook_input_impedance(solution, load_at[1e6], 1e6)
        assert row[0][1].real == pytest.approx(expected.real, abs=1e-9)


def test_predicted_response_refuses_bad_input():
    result = solve_match(complex(10.0, 0.0), REFERENCE, FREQ)
    with pytest.raises(ValueError):
        result.predicted_response([0.0])
    with pytest.raises(ValueError):
        result.predicted_response([FREQ, float("nan")])

    refused = solve_match(complex(50.0, 0.0), REFERENCE, FREQ)
    with pytest.raises(ValueError):
        refused.predicted_response([FREQ])


def test_solution_at_keeps_component_values():
    result = solve_match(complex(10.0, 0.0), REFERENCE, FREQ)
    solution = result.solutions[0]
    shifted = solution.at(2.0 * FREQ)
    assert shifted.series_value == solution.series_value
    assert shifted.shunt_value == solution.shunt_value
    assert shifted.frequency == 2.0 * FREQ
    # An inductor's reactance doubles, a capacitor's halves.
    if solution.series_kind is ElementKind.INDUCTOR:
        assert shifted.series_impedance.imag == pytest.approx(
            2.0 * solution.series_impedance.imag, rel=1e-12
        )
    else:
        assert shifted.series_impedance.imag == pytest.approx(
            solution.series_impedance.imag / 2.0, rel=1e-12
        )
    # The original is untouched.
    assert solution.frequency == FREQ
    with pytest.raises(ValueError):
        solution.at(0.0)


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------
def assert_refused(result: MatchSolutions, needle: str) -> None:
    assert result.status is MatchStatus.UNAVAILABLE
    assert result.unavailable
    assert not result.ok
    assert result.solutions == ()
    assert needle in result.reason.lower(), result.reason
    assert result.describe().startswith("unavailable (")
    assert result.preferred is None


@pytest.mark.parametrize("reference", [0.0, -50.0, -1e-12])
def test_reference_must_be_positive(reference):
    assert_refused(
        solve_match(complex(10.0, 0.0), reference, FREQ), "not positive"
    )


@pytest.mark.parametrize("frequency", [0.0, -1.0, -1e9])
def test_frequency_must_be_positive(frequency):
    assert_refused(
        solve_match(complex(10.0, 0.0), REFERENCE, frequency), "not positive"
    )


@pytest.mark.parametrize(
    "load",
    [
        complex(math.inf, 0.0),
        complex(0.0, -math.inf),
        complex(math.nan, 0.0),
        complex(10.0, math.nan),
    ],
)
def test_load_must_be_finite(load):
    assert_refused(solve_match(load, REFERENCE, FREQ), "not finite")


@pytest.mark.parametrize("reference", [math.inf, math.nan, -math.inf])
def test_reference_must_be_finite(reference):
    assert_refused(
        solve_match(complex(10.0, 0.0), reference, FREQ), "not finite"
    )


@pytest.mark.parametrize("frequency", [math.inf, math.nan])
def test_frequency_must_be_finite(frequency):
    assert_refused(
        solve_match(complex(10.0, 0.0), REFERENCE, frequency), "not finite"
    )


def test_already_resistively_matched_is_refused():
    result = solve_match(complex(50.0, 0.0), REFERENCE, FREQ)
    assert_refused(result, "already the reference")
    assert "nothing to match" in result.reason
    # Explicitly not a 0 H inductor.
    assert "0 H" in result.reason


@pytest.mark.parametrize("reference", [50.0, 75.0, 600.0])
def test_already_matched_at_any_reference(reference):
    result = solve_match(complex(reference, 0.0), reference, FREQ)
    assert_refused(result, "already the reference")


def test_negative_resistance_is_refused():
    result = solve_match(complex(-10.0, 5.0), REFERENCE, FREQ)
    assert_refused(result, "negative resistance")
    assert "active device" in result.reason
    assert "passive" in result.reason


@pytest.mark.parametrize("load", [complex(-1.0, 0.0), complex(-50.0, -20.0)])
def test_any_negative_resistance_is_refused(load):
    assert_refused(solve_match(load, REFERENCE, FREQ), "negative resistance")


def test_negligible_reactance_is_treated_as_resistive():
    """Below REACTANCE_TOLERANCE the load is a resistor, not a huge inductor."""
    load = complex(25.0, 1e-15)
    result = solve_match(load, REFERENCE, FREQ)
    assert result.ok
    assert result.load == complex(25.0, 0.0)
    plain = solve_match(complex(25.0, 0.0), REFERENCE, FREQ)
    assert len(result.solutions) == len(plain.solutions)
    for solution, expected in zip(
        result.solutions, plain.solutions, strict=True
    ):
        assert math.isfinite(solution.series_value)
        assert math.isfinite(solution.shunt_value)
        assert solution.topology == expected.topology
        # The components are those of a 25 ohm resistor exactly, not of a
        # reactance 1e-15 ohm away from one.
        assert solution.series_value == expected.series_value
        assert solution.shunt_value == expected.shunt_value


@pytest.mark.parametrize("reactance", [1e-15, 1e-12, -1e-12, 0.0])
def test_negligible_reactance_at_the_reference_is_refused(reactance):
    """Floating-point |X| below the tolerance: no component can be stated."""
    result = solve_match(complex(50.0, reactance), REFERENCE, FREQ)
    if reactance == 0.0:
        assert_refused(result, "already the reference")
    else:
        assert_refused(result, "below the stated tolerance")
        assert "indistinguishable from a matched one" in result.reason


def test_reactance_above_the_tolerance_is_solved_normally():
    """Just above the tolerance is a real network, not a refusal."""
    limit = REACTANCE_TOLERANCE * REFERENCE
    load = complex(25.0, 10.0 * limit)
    result = solve_match(load, REFERENCE, FREQ)
    assert result.ok
    assert result.load == load
    for solution in result.solutions:
        assert_reproduces_reference(solution, load, REFERENCE)


def test_tolerances_are_stated_and_small():
    assert 0.0 < REACTANCE_TOLERANCE < 1e-6
    assert 0.0 < RESISTIVE_MATCH_TOLERANCE < 1e-6
    assert 0.0 < SELF_CHECK_TOLERANCE <= 1e-6


def test_omitted_branches_carry_reasons():
    """A physically impossible branch is omitted and explained, not faked."""
    result = solve_match(complex(10.0, 0.0), REFERENCE, FREQ)
    assert result.ok
    assert result.reasons
    for reason in result.reasons:
        assert isinstance(reason, str) and reason
    described = result.describe()
    for reason in result.reasons:
        assert reason in described


def test_screening_substitution_is_reported_as_a_note():
    """Dropping a sub-tolerance reactance changes the load; say so."""
    result = solve_match(complex(25.0, 1e-15), REFERENCE, FREQ)
    assert result.ok
    assert result.load == complex(25.0, 0.0)
    assert len(result.notes) == 1
    assert "below the stated tolerance" in result.notes[0]
    assert "treated as zero" in result.notes[0]
    assert "25 ohm" in result.notes[0]
    assert "(note)" in result.describe()
    # A note is not an omission: the note is about the load, not a branch.
    assert result.notes[0] not in result.reasons

    # A purely resistive load was not substituted, so it carries no note.
    plain = solve_match(complex(25.0, 0.0), REFERENCE, FREQ)
    assert plain.notes == ()


def test_never_returns_a_negative_component_for_a_spread_of_loads():
    loads = [
        complex(r, x)
        for r in (0.5, 1.0, 10.0, 25.0, 50.0, 75.0, 300.0, 1e4)
        for x in (-400.0, -50.0, -5.0, 0.0, 5.0, 50.0, 400.0)
    ]
    for load in loads:
        result = solve_match(load, REFERENCE, FREQ)
        for solution in result.solutions:
            assert solution.series_value > 0.0, solution.describe()
            assert solution.shunt_value > 0.0, solution.describe()
            assert_reproduces_reference(solution, load, REFERENCE)
        if not result.ok:
            assert load.real <= 0.0 or (
                abs(load.imag) == 0.0
                and math.isclose(load.real, REFERENCE, rel_tol=1e-9)
            ), f"{load} refused: {result.reason}"


def test_result_is_immutable():
    """A frozen result cannot be edited into a different claim."""
    result = solve_match(complex(10.0, 0.0), REFERENCE, FREQ)
    with pytest.raises(FrozenInstanceError):
        result.reference = 75.0
    with pytest.raises(FrozenInstanceError):
        result.solutions[0].series_value = 1.0


def test_module_is_qt_free_and_trace_independent():
    """The assistant must import and answer without Qt or any measurement."""
    import subprocess
    import sys

    code = (
        "import sys; import NanoVNASaver.Matching as m; "
        "bad = [name for name in sys.modules "
        "if name.split('.')[0] in {'PySide6', 'PySide2', 'PyQt5'}]; "
        "assert not bad, bad; "
        "r = m.solve_match(complex(25, 30), 50.0, 10e6); "
        "assert r.ok and len(r.solutions) == 4; "
        "print('qt-free ok')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "qt-free ok" in completed.stdout


def test_no_dependency_on_measured_traces():
    """``solve_match`` takes an impedance, never a trace or a snapshot."""
    import inspect

    signature = inspect.signature(solve_match)
    assert list(signature.parameters) == ["load", "reference", "frequency"]
    for parameter in signature.parameters.values():
        assert parameter.annotation in ("complex", "float"), parameter
