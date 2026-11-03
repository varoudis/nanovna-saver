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
"""``abs()`` on a complex must not raise just because ``errno`` is stale.

CPython's ``complex_abs`` calls the C ``hypot`` and turns a ``errno`` left at
``ERANGE`` into ``OverflowError: absolute value too large``.  ``ERANGE``
outlives the call that set it -- a ``cmath.exp`` of a large angle leaves it
behind -- so any code that runs next inherits the exception.  ``ERANGE`` is
also what a genuine magnitude overflow sets, so the fix cannot simply be
"never raise".

Every test here therefore *forces* ``errno`` immediately before the call it
is testing.  Setting it once at the top of a test proves nothing: a
successful ``abs()`` clears ``errno``, and ``cmath`` calls clear it too, so a
preceding expression silently disarms the test.  ``math.hypot`` does not
clear it, which is what makes it usable in the helper.

The helper's contract, which these tests pin:

* a finite sample answers exactly what ``abs()`` answers, bit for bit;
* an undefined sample (a NaN or an infinite component) answers ``abs()``'s
  value without raising;
* two finite components that genuinely overflow still raise.
"""

import cmath
import ctypes
import math

import pytest

from NanoVNASaver import RFTools
from NanoVNASaver.Analysis.ResonanceAnalysis import vswr_transformed
from NanoVNASaver.Limits import Metric
from NanoVNASaver.Reports import _db
from NanoVNASaver.Traces import ComparisonKind, _difference

#: Loaded once: ``ctypes.CDLL`` runs ``dlopen``, which itself sets ``errno``
#: (``ENOENT`` while searching) -- calling it inside ``_get_errno`` would
#: report the search's ``errno`` instead of the code under test's.
_LIBC = ctypes.CDLL("libSystem.B.dylib")  # macOS
_LIBC.__error.restype = ctypes.POINTER(ctypes.c_int)


def _errno_pointer():
    """A pointer to the C library's thread-local ``errno``."""
    return _LIBC.__error().contents


def _set_errno(value: int) -> None:
    _errno_pointer().value = value


def _get_errno() -> int:
    return _errno_pointer().value


ERANGE = 34

#: Complex samples whose magnitude is undefined but whose magnitude is *not*
#: too large, and what ``abs()`` answers for them with a clean ``errno``.
#:
#: Only a NaN makes the raise possible; a component that is already infinite
#: short-circuits the C ``hypot`` and answers ``inf`` whatever ``errno`` says
#: (measured, not assumed).  Those are listed separately below.
UNDEFINED = [
    (complex(math.nan, 0.0), math.nan),
    (complex(math.nan, math.nan), math.nan),
    (complex(0.0, math.nan), math.nan),
    (complex(math.nan, 1.0), math.nan),
    (complex(-1.0, math.nan), math.nan),
]

#: Samples an infinite component already answers for, stale ``errno`` or not.
ALREADY_INFINITE = [
    (complex(math.inf, math.nan), math.inf),
    (complex(math.nan, math.inf), math.inf),
    (complex(math.inf, 0.0), math.inf),
    (complex(-math.inf, 1.0), math.inf),
]


@pytest.mark.parametrize(("value", "expected"), UNDEFINED)
def test_an_undefined_sample_answers_instead_of_raising(value, expected):
    """The regression: this raised whenever ``errno`` happened to be 34."""
    _set_errno(ERANGE)

    result = RFTools.magnitude(value)

    if math.isnan(expected):
        assert math.isnan(result)
    else:
        assert result == expected


@pytest.mark.parametrize(("value", "expected"), ALREADY_INFINITE)
def test_an_infinite_component_answers_without_consulting_errno(
    value, expected
):
    """``hypot`` short-circuits on an infinite component.

    Recorded because it bounds the fix: no raise to catch here, so the helper
    must simply pass the value through unchanged.
    """
    _set_errno(ERANGE)

    assert RFTools.magnitude(value) == expected
    assert abs(value) == expected


@pytest.mark.parametrize(("value", "expected"), UNDEFINED)
def test_the_builtin_really_does_raise_under_the_same_errno(value, expected):
    """Proof the test above is not vacuous.

    Without this, a future CPython that stops raising would make the helper's
    test pass for the wrong reason.
    """
    _set_errno(ERANGE)

    with pytest.raises(OverflowError):
        abs(value)


def test_a_genuine_overflow_still_raises():
    """Two finite components whose magnitude overflows is a real error.

    ``ERANGE`` is exactly what a real overflow leaves behind, so it cannot be
    treated as "stale" on its own; only a non-finite component marks a sample
    whose magnitude is undefined rather than too large.
    """
    _set_errno(ERANGE)

    with pytest.raises(OverflowError):
        RFTools.magnitude(complex(1.7e308, 1.7e308))


@pytest.mark.parametrize("value", [0, 1, 2, 22, 33, 35])
def test_only_erange_produces_the_raise(value):
    """The forcing mechanism itself: the code is exact, not "some errno"."""
    _set_errno(value)
    assert math.isnan(abs(complex(math.nan, 0.0)))
    assert math.isnan(RFTools.magnitude(complex(math.nan, 0.0)))


def test_erange_itself_produces_the_raise():
    _set_errno(ERANGE)
    with pytest.raises(OverflowError):
        abs(complex(math.nan, 0.0))


FINITE = [
    complex(0.0, 0.0),
    complex(3.0, 4.0),
    complex(-3.0, 4.0),
    complex(1e-300, 1e-300),
    complex(1e300, 1e300),
    complex(0.7, -0.71),
    complex(1.2345678901234567e-8, -9.876543210987654e-9),
]


@pytest.mark.parametrize("value", FINITE)
def test_a_finite_sample_is_bit_identical_to_abs(value):
    """Not "close": ``math.hypot`` and ``abs()`` disagree in the last place.

    One finite input in six differs by 1 ulp, so replacing ``abs()`` with
    ``math.hypot`` would have shifted published numbers.  The helper keeps
    ``abs()`` and only catches, so every finite result is unchanged.
    """
    expected = abs(value)
    _set_errno(0)

    assert RFTools.magnitude(value) == expected
    assert RFTools.magnitude(value).hex() == expected.hex()


def test_the_helper_is_reached_from_the_public_readouts():
    """A NaN sample must not take the readouts down with it.

    Each of these was measured to raise ``OverflowError`` with a plain
    ``abs()`` under the same forced ``errno``, so they are the sites where
    the helper is load-bearing rather than decorative.
    """
    undefined = RFTools.Datapoint(1_000_000, math.nan, 0.0)

    _set_errno(ERANGE)
    assert math.isnan(undefined.vswr)
    _set_errno(ERANGE)
    assert math.isnan(undefined.gain)
    _set_errno(ERANGE)
    assert math.isnan(Metric.S11_VSWR.measure(undefined))
    _set_errno(ERANGE)
    assert _db([undefined]) == []
    _set_errno(ERANGE)
    assert math.isnan(vswr_transformed(complex(math.nan, 0.0)))


def test_the_helper_is_reached_from_the_trace_comparison():
    _set_errno(ERANGE)
    difference = _difference(
        ComparisonKind.MAGNITUDE_DB,
        complex(math.nan, 0.0),
        complex(0.5, 0.0),
        1_000_000,
        50.0,
    )

    assert math.isnan(difference.real)


def test_erange_is_left_by_a_libm_range_error():
    """Where the stale value comes from; measured on this interpreter.

    ``cmath.exp(1000)`` overflows in the C library and leaves ``ERANGE``
    behind.  ``math.exp`` does not: it checks the argument in Python first.
    An ``EDOM`` error (``cmath.log(0)``) leaves 33, not 34, which is why the
    helper cannot treat "some errno" as a reason to swallow.
    """
    # ``errno`` has to be read inside the handler: leaving a ``pytest.raises``
    # block runs code that touches ``errno`` again.
    _set_errno(0)
    try:
        cmath.exp(1000)
    except OverflowError:
        assert _get_errno() == ERANGE
    else:
        pytest.fail("cmath.exp(1000) did not overflow")

    _set_errno(0)
    try:
        cmath.log(0)
    except ValueError:
        assert _get_errno() == 33
    else:
        pytest.fail("cmath.log(0) did not fail")
