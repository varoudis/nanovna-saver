#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020  Rune B. Broberg
#  Copyright (C) 2020,2021 NanoVNA-Saver Authors
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
import math

from . import SITools
from .SITools import Value, ValueType

#: The one word for a value that is not a number.
#:
#: ``inf`` is not "undefined": it is a real limit (a perfect match has
#: infinite return loss, a lossless part has infinite Q) and is written as
#: the infinity sign.  NaN is different — nothing was measured — and the
#: readouts say so instead of printing ``nan``, ``-`` or a unit with no
#: number in front of it.
UNDEFINED = "undefined"


def _non_finite(value: float, suffix: str = "") -> str | None:
    """Text for an undefined or infinite number, ``None`` for a number."""
    if math.isnan(value):
        return UNDEFINED
    if math.isinf(value):
        return ("-" if value < 0 else "") + "\N{INFINITY}" + suffix
    return None


FMT_FREQ = SITools.Format()
FMT_FREQ_SHORT = SITools.Format(max_nr_digits=4)
FMT_FREQ_SPACE = SITools.Format(space_str=" ")
FMT_FREQ_SWEEP = SITools.Format(max_nr_digits=9, allow_strip=True)
FMT_FREQ_INPUTS = SITools.Format(
    max_nr_digits=10, allow_strip=True, printable_min=0, unprintable_under="- "
)
FMT_Q_FACTOR = SITools.Format(
    max_nr_digits=4,
    assume_infinity=False,
    min_offset=0,
    max_offset=0,
    allow_strip=True,
)
FMT_GROUP_DELAY = SITools.Format(max_nr_digits=5, space_str=" ")
FMT_REACT = SITools.Format(max_nr_digits=5, space_str=" ", allow_strip=True)
FMT_COMPLEX = SITools.Format(
    max_nr_digits=3, allow_strip=True, printable_min=0, unprintable_under="- "
)
FMT_COMPLEX_NEG = SITools.Format(max_nr_digits=3, allow_strip=True)
FMT_SHORT = SITools.Format(max_nr_digits=4)
FMT_WAVELENGTH = SITools.Format(max_nr_digits=4, space_str=" ")
FMT_PARSE = SITools.Format(
    parse_sloppy_unit=True, parse_sloppy_kilo=True, parse_clamp_min=0
)
FMT_PARSE_VALUE = SITools.Format(parse_sloppy_unit=True, parse_sloppy_kilo=True)
FMT_VSWR = SITools.Format(max_nr_digits=3)


def format_frequency(freq: ValueType) -> str:
    return str(SITools.Value(freq, "Hz", FMT_FREQ))


def format_frequency_inputs(freq: ValueType | str) -> str:
    return str(SITools.Value(freq, "Hz", FMT_FREQ_INPUTS))


def format_frequency_short(freq: ValueType) -> str:
    return str(SITools.Value(freq, "Hz", FMT_FREQ_SHORT))


def format_frequency_chart(freq: ValueType) -> str:
    return str(SITools.Value(freq, "", FMT_FREQ_SHORT))


def format_frequency_chart_2(freq: ValueType) -> str:
    return str(SITools.Value(freq, "", FMT_FREQ))


def format_frequency_space(freq: ValueType, fmt=FMT_FREQ_SPACE) -> str:
    return str(SITools.Value(freq, "Hz", fmt))


def format_frequency_sweep(freq: ValueType) -> str:
    return str(SITools.Value(freq, "Hz", FMT_FREQ_SWEEP))


def format_gain(val: ValueType, invert: bool = False) -> str:
    number = float(Value(val))
    if invert:
        number = -number
    return _non_finite(number, " dB") or f"{number:.3f} dB"


def format_q_factor(val: ValueType, allow_negative: bool = False) -> str:
    v = float(Value(val))
    if math.isnan(v):
        return UNDEFINED
    if v == math.inf or (not allow_negative and v < 0.0) or abs(v) > 10000.0:
        return "\N{INFINITY}"
    return str(SITools.Value(val, fmt=FMT_Q_FACTOR))


def format_vswr(val: ValueType) -> str:
    number = float(Value(val))
    return _non_finite(number) or f"{number:.3f}"


def format_magnitude(val: ValueType) -> str:
    number = float(Value(val))
    return _non_finite(number) or f"{number:.3f}"


def format_resistance(val: ValueType, allow_negative: bool = False) -> str:
    number = float(Value(val))
    text = _non_finite(number, " \N{OHM SIGN}")
    if text is not None:
        return text
    if not allow_negative and number < 0:
        return "- \N{OHM SIGN}"
    return str(SITools.Value(val, "\N{OHM SIGN}", FMT_REACT))


def format_capacitance(val: ValueType, allow_negative: bool = True) -> str:
    number = float(Value(val))
    text = _non_finite(number, " F")
    if text is not None:
        return text
    if not allow_negative and number < 0:
        return "- pF"
    return str(SITools.Value(val, "F", FMT_REACT))


def format_inductance(val: ValueType, allow_negative: bool = True) -> str:
    number = float(Value(val))
    text = _non_finite(number, " H")
    if text is not None:
        return text
    if not allow_negative and number < 0:
        return "- nH"
    return str(SITools.Value(val, "H", FMT_REACT))


def format_group_delay(val: ValueType) -> str:
    number = float(Value(val))
    return _non_finite(number, " s") or str(
        SITools.Value(val, "s", FMT_GROUP_DELAY)
    )


def format_phase(val: ValueType) -> str:
    degrees = math.degrees(float(Value(val)))
    return _non_finite(degrees, "\N{DEGREE SIGN}") or (
        f"{degrees:.2f}\N{DEGREE SIGN}"
    )


def format_complex_adm(z: complex, allow_negative: bool = False) -> str:
    if z == 0:
        return "- S"
    adm = 1 / z
    if not (math.isfinite(adm.real) and math.isfinite(adm.imag)):
        # One half of an undefined pair cannot be written next to the
        # other: the number as a whole is what was not measured.
        return UNDEFINED

    fmt_re = FMT_COMPLEX_NEG if allow_negative else FMT_COMPLEX
    re = SITools.Value(adm.real, fmt=fmt_re)
    im = SITools.Value(abs(adm.imag), fmt=FMT_COMPLEX)
    return f"{re}{'-' if adm.imag < 0 else '+'}j{im} S"


def format_complex_imp(z: complex, allow_negative: bool = False) -> str:
    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
        return UNDEFINED
    fmt_re = FMT_COMPLEX_NEG if allow_negative else FMT_COMPLEX
    re = SITools.Value(z.real, fmt=fmt_re)
    im = SITools.Value(abs(z.imag), fmt=FMT_COMPLEX)
    return f"{re}{'-' if z.imag < 0 else '+'}j{im} \N{OHM SIGN}"


def format_polar(magnitude: ValueType, angle: ValueType) -> str:
    """A magnitude and an angle as one readout."""
    text = _non_finite(float(Value(magnitude)))
    if text is not None:
        return text
    return f"{float(Value(magnitude)):.2f}\N{ANGLE}{format_phase(angle)}"


def format_wavelength(length: ValueType) -> str:
    number = float(Value(length))
    return _non_finite(number, " m") or str(
        SITools.Value(length, "m", FMT_WAVELENGTH)
    )


def format_y_axis(val: ValueType, unit: str = "") -> str:
    return str(SITools.Value(val, unit, FMT_SHORT))


def parse_frequency(freq: str) -> int:
    try:
        return int(SITools.Value(freq, "Hz", FMT_PARSE))
    except (ValueError, IndexError):
        return -1


def parse_value(
    val: str, unit: str = "", fmt: SITools.Format = FMT_PARSE_VALUE
) -> float:
    try:
        val.replace(",", ".")
        return float(SITools.Value(val, unit, fmt))
    except (ValueError, IndexError):
        return 0.0
