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
import cmath
import copy
import logging
import math
import os
import re
from collections import UserDict, defaultdict
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
from scipy.interpolate import interp1d

from .CalibrationHealth import (
    CalibrationMetadata,
    StandardVerification,
    split_metadata,
    utc_now,
)
from .RFTools import Datapoint
from .Touchstone import Touchstone

IDEAL_SHORT = complex(-1, 0)
IDEAL_OPEN = complex(1, 0)
IDEAL_LOAD = complex(0, 0)
IDEAL_THROUGH = complex(1, 0)

RXP_CAL_HEADER = re.compile(
    r"""
    ^ \# \s+ Hz \s+
    ShortR \s+ ShortI \s+ OpenR \s+ OpenI \s+
    LoadR \s+ LoadI
    (?P<through> \s+ ThroughR \s+ ThroughI)?
    (?P<thrurefl> \s+ ThrureflR \s+ ThrureflI)?
    (?P<isolation> \s+ IsolationR \s+ IsolationI)?
    \s* $
""",
    re.VERBOSE | re.IGNORECASE,
)

RXP_CAL_LINE = re.compile(
    r"""
    ^ \s*
    (?P<freq>\d+) \s+
    (?P<shortr>[-0-9Ee.]+) \s+ (?P<shorti>[-0-9Ee.]+) \s+
    (?P<openr>[-0-9Ee.]+) \s+ (?P<openi>[-0-9Ee.]+) \s+
    (?P<loadr>[-0-9Ee.]+) \s+ (?P<loadi>[-0-9Ee.]+)
    ( \s+ (?P<throughr>[-0-9Ee.]+) \s+ (?P<throughi>[-0-9Ee.]+))?
    ( \s+ (?P<thrureflr>[-0-9Ee.]+) \s+ (?P<thrurefli>[-0-9Ee.]+))?
    ( \s+ (?P<isolationr>[-0-9Ee.]+) \s+ (?P<isolationi>[-0-9Ee.]+))?
    \s* $
""",
    re.VERBOSE,
)

logger = logging.getLogger(__name__)


def correct_delay(d: Datapoint, delay: float, reflect: bool = False):
    mult = 2 if reflect else 1
    corr_data = d.z * cmath.exp(
        complex(0, 1) * 2 * math.pi * d.freq * delay * -1 * mult
    )
    return Datapoint(d.freq, corr_data.real, corr_data.imag)


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def validate_delay(delay: float) -> float:
    """Return ``delay`` as a finite float, rejecting anything else.

    A non-finite offset delay (``inf`` or ``NaN``) makes every corrected
    sample non-finite.  Rejecting it here stops an acquisition with an
    actionable message instead of publishing a "corrected" measurement full
    of ``NaN``.
    """
    if isinstance(delay, bool) or not isinstance(
        delay, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Offset delay must be a finite time in seconds, got {delay!r}"
        )
    value = float(delay)
    if not math.isfinite(value):
        raise ValueError(
            f"Offset delay must be a finite time in seconds, got {delay!r}"
        )
    return value


def datapoints_to_arrays(
    points: Sequence[Datapoint],
) -> tuple[np.ndarray, np.ndarray]:
    """Split datapoints into integer frequencies and complex values.

    The frequency array keeps the exact integer hertz values, so a round trip
    through :func:`arrays_to_datapoints` cannot change a frequency.  A
    frequency that is not a finite integer number of hertz is rejected
    instead of being truncated by the array conversion.
    """
    count = len(points)
    frequencies = [point.freq for point in points]
    if {type(freq) for freq in frequencies} <= {int}:
        # Whole sweeps are all plain ints; one min/max check replaces a
        # validation call per point.
        if frequencies:
            low = min(frequencies)
            high = max(frequencies)
            if low < _INT64_MIN or high > _INT64_MAX:
                _datapoint_frequency(low if low < _INT64_MIN else high)
        freq = np.fromiter(frequencies, np.int64, count)
    else:
        freq = np.fromiter(
            (_datapoint_frequency(value) for value in frequencies),
            np.int64,
            count,
        )
    values = np.fromiter((point.z for point in points), np.complex128, count)
    return freq, values


def _datapoint_frequency(freq) -> int:
    """Validate one datapoint frequency for the array conversion."""
    if not isinstance(freq, (int, np.integer)) or isinstance(freq, bool):
        if (
            isinstance(freq, (float, np.floating))
            and math.isfinite(freq)
            and float(freq).is_integer()
        ):
            freq = int(freq)
        else:
            raise ValueError(
                "Datapoint frequencies must be finite integer hertz values,"
                f" got {freq!r}"
            )
    value = int(freq)
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise ValueError(
            f"Datapoint frequency {value!r} Hz cannot be represented"
            " as a 64-bit integer"
        )
    return value


def _integer_frequencies(freq: np.ndarray) -> list[int]:
    """Return ``freq`` as Python ints, rejecting values that do not fit.

    Float and object arrays are accepted only when every value is finite,
    integral and inside the ``int64`` range.  Casting ``inf`` or ``nan`` to
    an integer would otherwise produce an arbitrary frequency
    (``INT64_MAX`` or ``INT64_MIN``) and publish it as measured data.
    """
    if freq.dtype.kind == "O":
        # Python ints beyond the int64 range make NumPy choose an object
        # array; validate them elementwise so the error names the value.
        return [_datapoint_frequency(value) for value in freq.tolist()]
    if freq.dtype.kind == "f":
        non_finite = ~np.isfinite(freq)
        if non_finite.any():
            raise ValueError(
                "Bulk datapoint conversion needs finite frequencies,"
                f" got {freq[non_finite][0]!r}"
            )
        rounded = np.rint(freq)
        whole = rounded == freq
        if not whole.all():
            raise ValueError(
                "Bulk datapoint conversion needs integer frequencies,"
                f" got {freq[~whole][0]!r}"
            )
    elif freq.dtype.kind not in "iu":
        raise ValueError(
            "Bulk datapoint conversion needs integer frequencies,"
            f" got a {freq.dtype.kind!r} array"
        )
    if freq.size:
        low = int(freq.min())
        high = int(freq.max())
        if low < _INT64_MIN or high > _INT64_MAX:
            out_of_range = low if low < _INT64_MIN else high
            raise ValueError(
                "Bulk datapoint conversion needs frequencies that fit in a"
                f" 64-bit integer, got {out_of_range!r}"
            )
    return [int(value) for value in freq.tolist()]


def arrays_to_datapoints(
    freq: np.ndarray, values: np.ndarray
) -> list[Datapoint]:
    """Rebuild datapoints from an integer frequency array and complex values.

    Frequencies must be finite, integral and representable as ``int``; see
    :func:`_integer_frequencies`.  Anything else raises ``ValueError`` rather
    than being cast to an arbitrary integer.
    """
    freq_arr = np.asarray(freq)
    values_arr = np.asarray(values, dtype=np.complex128)
    if freq_arr.ndim != 1 or values_arr.ndim != 1:
        raise ValueError("Bulk datapoint conversion needs 1-D arrays")
    if len(freq_arr) != len(values_arr):
        raise ValueError(
            f"Bulk datapoint conversion needs {len(freq_arr)} frequencies for"
            f" {len(values_arr)} values"
        )
    return list(
        map(
            Datapoint,
            _integer_frequencies(freq_arr),
            values_arr.real.tolist(),
            values_arr.imag.tolist(),
        )
    )


def correct_delay_bulk(
    freq: np.ndarray, values: np.ndarray, delay: float, reflect: bool = False
) -> np.ndarray:
    """Vectorised :func:`correct_delay`.

    Evaluates the same phase term over a whole array.  ``numpy.exp`` may
    differ from ``cmath.exp`` in the last bit, so results are numerically
    equal to the scalar version rather than bit-identical.

    ``delay`` must be finite and the result must stay finite: an offset
    delay that cannot produce real sample values (``inf``, ``NaN``, or a
    phase term that overflows) raises ``ValueError`` instead of returning
    non-finite samples as corrected data.
    """
    freq_arr = np.asarray(freq, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.complex128)
    if freq_arr.ndim != 1 or values_arr.ndim != 1:
        raise ValueError("Bulk delay correction needs 1-D arrays")
    if len(freq_arr) != len(values_arr):
        raise ValueError(
            f"Bulk delay correction needs {len(freq_arr)} frequencies,"
            f" got {len(values_arr)} values"
        )
    delay_value = validate_delay(delay)
    if not len(freq_arr) or delay_value == 0.0:
        return values_arr.copy()
    mult = 2 if reflect else 1
    with np.errstate(over="ignore", invalid="ignore"):
        corrected = values_arr * np.exp(
            complex(0, 1) * 2 * math.pi * freq_arr * delay_value * -1 * mult
        )
    finite = np.isfinite(corrected.real) & np.isfinite(corrected.imag)
    if not finite.all():
        first = freq_arr[~finite][0]
        raise ValueError(
            f"Offset delay of {delay_value!r} s produces non-finite corrected"
            f" values, starting at {first!r} Hz."
            " Check the offset delay setting."
        )
    return corrected


@dataclass
class CalData:
    # pylint: disable=too-many-instance-attributes
    short: complex = complex(0.0, 0.0)
    open: complex = complex(0.0, 0.0)
    load: complex = complex(0.0, 0.0)
    through: complex = complex(0.0, 0.0)
    thrurefl: complex = complex(0.0, 0.0)
    isolation: complex = complex(0.0, 0.0)
    freq: int = 0
    e00: complex = complex(0.0)  # Directivity
    e11: complex = complex(0.0)  # Port1 match
    delta_e: complex = complex(0.0)  # Tracking
    e10e01: complex = complex(0.0)  # Forward Reflection Tracking
    # 2 port
    e30: complex = complex(0.0)  # Forward isolation
    e22: complex = complex(0.0)  # Port2 match
    e10e32: complex = complex(0.0)  # Forward transmission

    def __str__(self):
        return (
            f"{self.freq}"
            f" {self.short.real} {self.short.imag}"
            f" {self.open.real} {self.open.imag}"
            f" {self.load.real} {self.load.imag}"
            + (
                f" {self.through.real} {self.through.imag}"
                f" {self.thrurefl.real} {self.thrurefl.imag}"
                f" {self.isolation.real} {self.isolation.imag}"
                if self.through
                else ""
            )
        )


@dataclass
class CalElement:
    # pylint: disable=too-many-instance-attributes
    short_state: str = ""
    short_touchstone: Touchstone = field(default_factory=Touchstone)
    short_is_ideal: bool = True
    short_l0: float = 5.7e-12
    short_l1: float = -8.96e-20
    short_l2: float = -1.1e-29
    short_l3: float = -4.12e-37
    short_length: float = -34.2  # ps

    open_state: str = ""
    open_touchstone: Touchstone = field(default_factory=Touchstone)
    open_is_ideal: bool = True
    open_c0: float = 2.1e-14
    open_c1: float = 5.67e-23
    open_c2: float = -2.39e-31
    open_c3: float = 2.0e-40
    open_length: float = 0.0

    load_state: str = ""
    load_touchstone: Touchstone = field(default_factory=Touchstone)
    load_is_ideal: bool = True
    load_r: float = 50.0
    load_l: float = 0.0
    load_c: float = 0.0
    load_length: float = 0.0

    through_is_ideal: bool = True
    through_length: float = 0.0


class CalDataSet(UserDict):
    def __init__(self) -> None:
        super().__init__()
        self.notes = ""
        self.data: defaultdict[int, CalData] = defaultdict(CalData)

    def __str__(self):
        return (
            (
                "# Calibration data for NanoVNA-Saver\n"
                + "\n".join([f"! {note}" for note in self.notes.splitlines()])
                + "\n"
                + "# Hz ShortR ShortI OpenR OpenI LoadR LoadI"
                + (
                    " ThroughR ThroughI ThrureflR"
                    " ThrureflI IsolationR IsolationI\n"
                    if self.complete2port()
                    else "\n"
                )
                + "\n".join(
                    [f"{self.data.get(freq)}" for freq in self.frequencies()]
                )
                + "\n"
            )
            if self.complete1port()
            else ""
        )

    def _append_match(
        self, m: re.Match, header: str, line_nr: int, line: str
    ) -> None:
        cal = m.groupdict()
        columns = {col[:-1] for col in cal.keys() if cal[col] and col != "freq"}
        if "through" in columns and header == "sol":
            logger.warning(
                "Through data with sol header. %i: %s", line_nr, line
            )
        # fix short data (without thrurefl)
        if "thrurefl" in columns and "isolation" not in columns:
            cal["isolationr"] = cal["thrureflr"]
            cal["isolationi"] = cal["thrurefli"]
            cal["thrureflr"], cal["thrurefli"] = None, None
        for name in columns:
            self.insert(
                name,
                Datapoint(
                    int(cal["freq"]),
                    float(cal[f"{name}r"]),
                    float(cal[f"{name}i"]),
                ),
            )

    def from_str(self, text: str) -> "CalDataSet":
        # reset data
        self.notes = ""
        self.data = defaultdict(CalData)
        header = ""
        # parse text
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()  # noqa: PLW2901

            if line.startswith("!"):
                self.notes += f"{line[2:]}\n"
                continue
            if m := RXP_CAL_HEADER.search(line):
                if header:
                    logger.warning(
                        "Duplicate header in cal data. %i: %s", i, line
                    )
                header = "through" if m.group("through") else "sol"
                continue
            if not line or line.startswith("#"):
                continue

            m = RXP_CAL_LINE.search(line)
            if not m:
                logger.warning("Illegal caldata. Line %i: %s", i, line)
                continue
            if not header:
                logger.warning(
                    "Caldata without having read header: %i: %s", i, line
                )
            self._append_match(m, header, i, line)
        return self

    def insert(self, name: str, dp: Datapoint):
        if name not in {
            "short",
            "open",
            "load",
            "through",
            "thrurefl",
            "isolation",
        }:
            raise KeyError(name)
        freq = dp.freq
        setattr(self.data[freq], name, (dp.z))
        self.data[freq].freq = freq

    def frequencies(self) -> list[int]:
        return sorted(self.data.keys())

    def freq_min(self) -> int:
        return self.frequencies()[0] if self.frequencies() else 0

    def freq_max(self) -> int:
        return self.frequencies()[-1] if self.frequencies() else 0

    def get(self, key: int, default: Optional[CalData] = None) -> CalData:  # type: ignore[override]
        if default:
            return self.data.get(key, default)
        return self.data[key]

    def items(self):
        yield from self.data.items()

    def values(self):
        for freq in self.frequencies():
            yield self.get(freq)

    def size_of(self, name: str) -> int:
        return len([True for val in self.data.values() if getattr(val, name)])

    def complete1port(self) -> bool:
        for val in self.data.values():
            if not all((val.short, val.open, val.load)):
                return False
        return any(self.data)

    def complete2port(self) -> bool:
        if not self.complete1port():
            return False
        for val in self.data.values():
            if not all((val.through, val.thrurefl, val.isolation)):
                return False
        return any(self.data)


class Calibration:
    def __init__(self):
        self.notes = []
        self.dataset = CalDataSet()
        self.cal_element = CalElement()
        self.interp = {}
        self.isCalculated = False
        #: Where the calibration data came from, as shown by the window.
        self.source = ""
        #: Name and content digest of the calibration kit whose standards
        #: were applied (CAL-01).  A name alone is not evidence of accuracy,
        #: so the digest of the actual coefficients/span is recorded too.
        #: Both stay ``""`` for device and legacy calibrations.
        self.kit_name = ""
        self.kit_identity = ""
        #: When and with what the standards were measured (CAL-02).  Empty
        #: for a file from an older version, whose age stays unknown.
        self.metadata = CalibrationMetadata()
        #: True once a standard was measured here rather than read from a
        #: file, which is the only case the current device may be recorded.
        self.measured_here = False
        #: Latest verification per standard.  Held in memory only: a result
        #: says what was connected at that moment, not what a file contains.
        self.verifications: dict[str, StandardVerification] = {}
        #: Counts changes to the corrections, so a result computed from one
        #: state cannot be attached after the calibration has moved on.
        self.revision = 0

    def frozen_copy(self) -> "Calibration":
        """Return an independent copy of this calibration's current state.

        Used at a run boundary so that later ``load()``/``calc_corrections()``
        calls on the live object — including in-place recalculation of the
        same instance — cannot change the coefficients, validity or identity
        that a running acquisition already depends on.  The copy is a deep
        copy of the calibration data and standards, plus a shallow copy of
        the interpolation table (the ``interp1d`` objects are immutable once
        built and are replaced, never mutated, by ``gen_interpolation``).
        """
        frozen = copy.copy(self)
        frozen.dataset = copy.deepcopy(self.dataset)
        frozen.cal_element = copy.deepcopy(self.cal_element)
        frozen.interp = dict(self.interp)
        frozen.notes = list(self.notes)
        frozen.verifications = dict(self.verifications)
        return frozen

    def insert(self, name: str, data: list[Datapoint]):
        for dp in data:
            self.dataset.insert(name, dp)
        self.metadata = replace(self.metadata, created=utc_now())
        self.measured_here = True
        self.verifications = {}

    def size(self) -> int:
        return len(self.dataset.frequencies())

    def data_size(self, name) -> int:
        return self.dataset.size_of(name)

    def isValid1Port(self) -> bool:
        return self.dataset.complete1port()

    def isValid2Port(self) -> bool:
        return self.dataset.complete2port()

    def _calc_port_1(self, freq: int, cal: CalData):
        g1 = self.gamma_short(freq)
        g2 = self.gamma_open(freq)
        g3 = self.gamma_load(freq)

        gm1 = cal.short
        gm2 = cal.open
        gm3 = cal.load

        denominator = (
            g1 * (g2 - g3) * gm1
            + g2 * g3 * gm2
            - g2 * g3 * gm3
            - (g2 * gm2 - g3 * gm3) * g1
        )
        cal.e00 = (
            -(
                (g2 * gm3 - g3 * gm3) * g1 * gm2
                - (g2 * g3 * gm2 - g2 * g3 * gm3 - (g3 * gm2 - g2 * gm3) * g1)
                * gm1
            )
            / denominator
        )
        cal.e11 = (
            (g2 - g3) * gm1 - g1 * (gm2 - gm3) + g3 * gm2 - g2 * gm3
        ) / denominator
        cal.delta_e = (
            -(
                (g1 * (gm2 - gm3) - g2 * gm2 + g3 * gm3) * gm1
                + (g2 * gm3 - g3 * gm3) * gm2
            )
            / denominator
        )

    def _calc_port_2(self, freq: int, cal: CalData):
        gt = self.gamma_through(freq)

        gm4 = cal.through
        gm5 = cal.thrurefl
        gm6 = cal.isolation
        gm7 = gm5 - cal.e00

        cal.e30 = cal.isolation
        cal.e10e01 = cal.e00 * cal.e11 - cal.delta_e
        cal.e22 = gm7 / (gm7 * cal.e11 * gt**2 + cal.e10e01 * gt**2)
        cal.e10e32 = (gm4 - gm6) * (1 - cal.e11 * cal.e22 * gt**2) / gt

    def calc_corrections(self):
        if not self.isValid1Port():
            logger.warning("Tried to calibrate from insufficient data.")
            raise ValueError(
                "All of short, open and load calibration steps"
                "must be completed for calibration to be applied."
            )
        logger.debug("Calculating calibration for %d points.", self.size())

        for freq, caldata in self.dataset.items():
            try:
                self._calc_port_1(freq, caldata)
                if self.isValid2Port():
                    self._calc_port_2(freq, caldata)
            except ZeroDivisionError as exc:
                self.isCalculated = False
                logger.error(
                    "Division error - did you use the same measurement"
                    " for two of short, open and load?"
                )
                raise ValueError(
                    f"Two of short, open and load returned the same"
                    f" values at frequency {freq}Hz."
                ) from exc

        self.gen_interpolation()
        self.isCalculated = True
        # New corrections, possibly from new standard definitions on the same
        # measured data, so nothing verified before applies to them.
        self.verifications = {}
        self.revision += 1
        logger.debug("Calibration correctly calculated.")

    def gamma_short(self, freq: int) -> complex:
        if self.cal_element.short_state == "IDEAL":
            return IDEAL_SHORT
        if self.cal_element.short_state == "FILE":
            self.cal_element.short_touchstone.gen_interpolation_s11()
            dp = self.cal_element.short_touchstone.s_freq("11", freq)
            return complex(dp.re, dp.im)
        logger.debug("Using short calibration set values.")
        cal_element = self.cal_element
        Zsp = complex(
            0.0,
            2.0
            * math.pi
            * freq
            * (
                cal_element.short_l0
                + cal_element.short_l1 * freq
                + cal_element.short_l2 * freq**2
                + cal_element.short_l3 * freq**3
            ),
        )
        # Referencing https://arxiv.org/pdf/1606.02446.pdf (18) - (21)
        return (
            (Zsp / 50.0 - 1.0)
            / (Zsp / 50.0 + 1.0)
            * cmath.exp(
                complex(0.0, -4.0 * math.pi * freq * cal_element.short_length)
            )
        )

    def gamma_open(self, freq: int) -> complex:
        if self.cal_element.open_state == "IDEAL":
            return IDEAL_OPEN
        if self.cal_element.open_state == "FILE":
            self.cal_element.open_touchstone.gen_interpolation_s11()
            dp = self.cal_element.open_touchstone.s_freq("11", freq)
            return complex(dp.re, dp.im)
        logger.debug("Using open calibration set values.")
        cal_element = self.cal_element
        Zop = complex(
            0.0,
            2.0
            * math.pi
            * freq
            * (
                cal_element.open_c0
                + cal_element.open_c1 * freq
                + cal_element.open_c2 * freq**2
                + cal_element.open_c3 * freq**3
            ),
        )
        return ((1.0 - 50.0 * Zop) / (1.0 + 50.0 * Zop)) * cmath.exp(
            complex(0.0, -4.0 * math.pi * freq * cal_element.open_length)
        )

    def gamma_load(self, freq: int) -> complex:
        if self.cal_element.load_state == "IDEAL":
            return IDEAL_LOAD
        if self.cal_element.load_state == "FILE":
            self.cal_element.load_touchstone.gen_interpolation_s11()
            dp = self.cal_element.load_touchstone.s_freq("11", freq)
            return complex(dp.re, dp.im)
        logger.debug("Using load calibration set values.")
        cal_element = self.cal_element
        Zl = complex(cal_element.load_r, 0.0)
        if cal_element.load_c > 0.0:
            Zl = cal_element.load_r / complex(
                1.0,
                2.0 * cal_element.load_r * math.pi * freq * cal_element.load_c,
            )
        if cal_element.load_l > 0.0:
            Zl = Zl + complex(0.0, 2 * math.pi * freq * cal_element.load_l)
        return (
            (Zl / 50.0 - 1.0)
            / (Zl / 50.0 + 1.0)
            * cmath.exp(
                complex(0.0, -4 * math.pi * freq * cal_element.load_length)
            )
        )

    def gamma_through(self, freq: int) -> complex:
        if self.cal_element.through_is_ideal:
            return IDEAL_THROUGH
        logger.debug("Using through calibration set values.")
        cal_element = self.cal_element
        return cmath.exp(
            complex(0.0, -2.0 * math.pi * cal_element.through_length * freq)
        )

    def gen_interpolation(self):
        (freq, e00, e11, delta_e, e10e01, e30, e22, e10e32) = zip(
            *[
                (
                    c.freq,
                    c.e00,
                    c.e11,
                    c.delta_e,
                    c.e10e01,
                    c.e30,
                    c.e22,
                    c.e10e32,
                )
                for c in self.dataset.values()
            ],
            strict=False,
        )

        self.interp = {
            "e00": interp1d(
                freq,
                e00,
                kind="slinear",
                bounds_error=False,
                fill_value=(e00[0], e00[-1]),
            ),
            "e11": interp1d(
                freq,
                e11,
                kind="slinear",
                bounds_error=False,
                fill_value=(e11[0], e11[-1]),
            ),
            "delta_e": interp1d(
                freq,
                delta_e,
                kind="slinear",
                bounds_error=False,
                fill_value=(delta_e[0], delta_e[-1]),
            ),
            "e10e01": interp1d(
                freq,
                e10e01,
                kind="slinear",
                bounds_error=False,
                fill_value=(e10e01[0], e10e01[-1]),
            ),
            "e30": interp1d(
                freq,
                e30,
                kind="slinear",
                bounds_error=False,
                fill_value=(e30[0], e30[-1]),
            ),
            "e22": interp1d(
                freq,
                e22,
                kind="slinear",
                bounds_error=False,
                fill_value=(e22[0], e22[-1]),
            ),
            "e10e32": interp1d(
                freq,
                e10e32,
                kind="slinear",
                bounds_error=False,
                fill_value=(e10e32[0], e10e32[-1]),
            ),
        }

    def correct11(self, dp: Datapoint):
        i = self.interp
        s11 = (dp.z - i["e00"](dp.freq)) / (
            (dp.z * i["e11"](dp.freq)) - i["delta_e"](dp.freq)
        )
        return Datapoint(dp.freq, s11.real, s11.imag)

    def correct21(self, dp: Datapoint, dp11: Datapoint):
        i = self.interp
        s21 = (dp.z - i["e30"](dp.freq)) / i["e10e32"](dp.freq)
        s21 = s21 * (
            i["e10e01"](dp.freq)
            / (i["e11"](dp.freq) * dp11.z - i["delta_e"](dp.freq))
        )
        return Datapoint(dp.freq, s21.real, s21.imag)

    # -- bulk (NumPy) correction ---------------------------------------
    #
    # The scalar methods above stay the reference implementation.  The bulk
    # path evaluates the same equations over whole arrays: interpolation is
    # done by calling each ``interp1d`` once with the frequency array (SciPy
    # handles the linear search and the endpoint fill in C), then the exact
    # same arithmetic as ``correct11``/``correct21`` is applied elementwise.
    # Dividing elementwise in NumPy may differ from CPython's complex
    # division by a fraction of an ULP, so results are numerically equal, not
    # bit-identical, to the scalar oracle.

    def _interp_channel(self, name: str, freq: np.ndarray) -> np.ndarray:
        """Evaluate one interpolation channel over a frequency array."""
        try:
            interpolator = self.interp[name]
        except KeyError as exc:  # pragma: no cover - programming error
            raise ValueError(
                "Calibration is not calculated: call calc_corrections()"
            ) from exc
        return np.asarray(interpolator(freq), dtype=np.complex128)

    def correct11_bulk(self, freq: np.ndarray, s11: np.ndarray) -> np.ndarray:
        """Correct S11 for every frequency in one vectorised pass.

        ``freq`` and ``s11`` must be 1-D arrays of equal, non-zero length.
        Returns the corrected complex S11 array with the same length.  The
        equations and edge-holding behaviour are those of :meth:`correct11`.

        Raises ``ValueError`` when a sample cannot be corrected (a vanishing
        denominator, or a non-finite result).  Returning ``inf``/``NaN``
        samples would present an invalid calibration as corrected data.
        """
        freq_arr = np.asarray(freq, dtype=np.float64)
        s11_arr = np.asarray(s11, dtype=np.complex128)
        self._check_bulk_arrays("S11", freq_arr, s11_arr)
        e00 = self._interp_channel("e00", freq_arr)
        e11 = self._interp_channel("e11", freq_arr)
        delta_e = self._interp_channel("delta_e", freq_arr)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            corrected = (s11_arr - e00) / ((s11_arr * e11) - delta_e)
        self._check_bulk_result("S11", freq_arr, corrected)
        return corrected

    def correct21_bulk(
        self, freq: np.ndarray, s21: np.ndarray, s11: np.ndarray
    ) -> np.ndarray:
        """Correct S21 for every frequency in one vectorised pass.

        ``s11`` is the *raw* port-1 reflection of the same frequency, exactly
        as :meth:`correct21` expects; the caller must not pass corrected S11.
        The three arrays must be 1-D with equal, non-zero length.  Raises
        ``ValueError`` like :meth:`correct11_bulk`.
        """
        freq_arr = np.asarray(freq, dtype=np.float64)
        s21_arr = np.asarray(s21, dtype=np.complex128)
        s11_arr = np.asarray(s11, dtype=np.complex128)
        self._check_bulk_arrays("S21", freq_arr, s21_arr, s11_arr)
        e30 = self._interp_channel("e30", freq_arr)
        e10e32 = self._interp_channel("e10e32", freq_arr)
        e10e01 = self._interp_channel("e10e01", freq_arr)
        e11 = self._interp_channel("e11", freq_arr)
        delta_e = self._interp_channel("delta_e", freq_arr)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            s21_corr = (s21_arr - e30) / e10e32
            corrected = s21_corr * (e10e01 / ((e11 * s11_arr) - delta_e))
        self._check_bulk_result("S21", freq_arr, corrected)
        return corrected

    @staticmethod
    def _check_bulk_result(
        channel: str, freq: np.ndarray, corrected: np.ndarray
    ) -> None:
        """Refuse to publish an invalid correction as corrected data."""
        finite = np.isfinite(corrected.real) & np.isfinite(corrected.imag)
        if finite.all():
            return
        invalid = int(np.count_nonzero(~finite))
        first = int(freq[~finite][0])
        raise ValueError(
            f"{channel} correction is undefined for {invalid} of"
            f" {len(freq)} point(s), starting at {first} Hz."
            " The calibration or the measured data is invalid there."
        )

    @staticmethod
    def _check_bulk_arrays(
        channel: str, freq: np.ndarray, *values: np.ndarray
    ) -> None:
        """Reject shapes that would silently broadcast or index wrongly."""
        if freq.ndim != 1 or any(value.ndim != 1 for value in values):
            raise ValueError(
                f"{channel} bulk correction needs 1-D arrays, got shapes "
                f"{[freq.shape, *(value.shape for value in values)]}"
            )
        if not len(freq):
            raise ValueError(
                f"{channel} bulk correction needs at least 1 point"
            )
        for value in values:
            if len(value) != len(freq):
                raise ValueError(
                    f"{channel} bulk correction needs {len(freq)} values, "
                    f"got {len(value)}"
                )

    def to_text(self) -> str:
        """The calibration file's content, with its metadata as a comment."""
        note = self.metadata.to_note()
        self.dataset.notes = "\n".join(
            [*self.notes, note] if note else self.notes
        )
        return str(self.dataset)

    def from_text(self, text: str) -> None:
        self.dataset = CalDataSet().from_str(text)
        self.metadata, self.notes = split_metadata(
            self.dataset.notes.splitlines()
        )
        self.measured_here = False
        self.verifications = {}

    def save(self, filename: str):
        if not self.isValid1Port():
            raise ValueError("Not a valid calibration")
        with open(filename, mode="w", encoding="utf-8") as calfile:
            calfile.write(self.to_text())

    def load(self, filename):
        self.source = os.path.basename(filename)
        with open(filename, encoding="utf-8") as calfile:
            self.from_text(calfile.read())
