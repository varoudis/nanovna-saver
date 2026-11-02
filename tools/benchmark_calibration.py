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
"""PERF-04: measure the fused (bulk) calibration path against the scalar one.

The task asks for a benchmark that names the executed path, the dataset, the
error tolerance and the scope of the measurement, and says explicitly that
keeping the current implementation is a valid outcome.  This script settles
that with numbers on this machine.

Scope of what is timed — and what is not:

* **Timed:** the correction maths only, ``Calibration.correct11`` (scalar,
  reference implementation) versus ``Calibration.correct11_bulk`` (NumPy),
  and the same pair for ``correct21``.  Transport, datapoint conversion and
  chart rendering are excluded, so the ratio is a pure maths ratio.
* **Conversion overhead** is timed separately, because a fused path still
  has to move values in and out of arrays; a fast kernel that then pays the
  same cost in conversion has not won anything.
* **Rendering** is not re-measured here; the redraw cost is recorded by
  PERF-03 and is far larger than either path.

Correctness is asserted before timing, against the scalar implementation as
the reference, so a fast-but-wrong path cannot look good.

Run::

    QT_QPA_PLATFORM=offscreen .venv/bin/python tools/benchmark_calibration.py
"""

from __future__ import annotations

import statistics
import sys
import time

import numpy as np

from NanoVNASaver.Calibration import (
    Calibration,
    arrays_to_datapoints,
    datapoints_to_arrays,
)
from NanoVNASaver.RFTools import Datapoint

TWO_PORT_CAL = "tests/data/test_2port_long.cal"

#: Repeats for each measurement; the median is reported, the spread too.
REPEATS = 7


def _calibration(filename: str) -> Calibration:
    calibration = Calibration()
    calibration.load(filename)
    calibration.calc_corrections()
    return calibration


def _span(calibration: Calibration) -> tuple[int, int]:
    frequencies = calibration.dataset.frequencies()
    return frequencies[0], frequencies[-1]


def _grid(calibration: Calibration, count: int) -> list[Datapoint]:
    """A frequency-dependent load over the calibrated span.

    Frequency dependence matters: a fixed value would let interpolation be
    skipped without changing the answer.
    """
    start, stop = _span(calibration)
    step = (stop - start) / max(count - 1, 1)
    points = []
    for index in range(count):
        freq = round(start + index * step)
        phase = -2 * np.pi * freq * 2.4e-9
        magnitude = 0.15 + 0.35 / (1 + ((freq - (start + stop) / 2) / 1e8) ** 2)
        value = magnitude * complex(np.cos(phase), np.sin(phase))
        points.append(Datapoint(freq, value.real, value.imag))
    return points


def _transmission(points: list[Datapoint]) -> list[Datapoint]:
    result = []
    for point in points:
        phase = -2 * np.pi * point.freq * 4.7e-9
        magnitude = 0.7 - 0.2 / (1 + (point.freq / 1e9))
        value = magnitude * complex(np.cos(phase), np.sin(phase))
        result.append(Datapoint(point.freq, value.real, value.imag))
    return result


def _time(call) -> float:
    """Median wall time of ``REPEATS`` calls, in seconds."""
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _spread(call) -> tuple[float, float]:
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return min(samples), max(samples)


def main() -> int:
    calibration = _calibration(TWO_PORT_CAL)
    print("path:      Calibration.correct11/correct21 (scalar) vs *_bulk")
    print(
        f"dataset:   {TWO_PORT_CAL}, span {_span(calibration)[0]:,}"
        f"..{_span(calibration)[1]:,} Hz"
    )
    print(f"python:    {sys.version.split()[0]}, numpy {np.__version__}")
    print(f"repeats:   {REPEATS} (median reported)")
    print()

    header = (
        f"{'points':>7} {'scalar11':>10} {'bulk11':>10} {'x':>6}"
        f" {'scalar21':>10} {'bulk21':>10} {'x':>6}"
        f" {'max |Δ|':>10}"
    )
    print(header)
    print("-" * len(header))

    for count in (101, 401, 1001, 4001, 16001):
        points11 = _grid(calibration, count)
        points21 = _transmission(points11)
        freq, raw11 = datapoints_to_arrays(points11)
        _freq21, raw21 = datapoints_to_arrays(points21)

        # Correctness first: the scalar implementation is the reference.
        scalar11 = np.array(
            [calibration.correct11(point).z for point in points11]
        )
        bulk11 = calibration.correct11_bulk(freq, raw11)
        scalar21 = np.array(
            [
                calibration.correct21(point, point11).z
                for point, point11 in zip(points21, points11, strict=True)
            ]
        )
        bulk21 = calibration.correct21_bulk(freq, raw21, raw11)
        error = max(
            float(np.max(np.abs(bulk11 - scalar11))),
            float(np.max(np.abs(bulk21 - scalar21))),
        )

        def scalar_reflection(pts=points11):
            return [calibration.correct11(point) for point in pts]

        def scalar_transmission(pts21=points21, pts11=points11):
            return [
                calibration.correct21(point, point11)
                for point, point11 in zip(pts21, pts11, strict=True)
            ]

        t_scalar11 = _time(scalar_reflection)
        t_bulk11 = _time(
            lambda f=freq, v=raw11: calibration.correct11_bulk(f, v)
        )
        t_scalar21 = _time(scalar_transmission)
        t_bulk21 = _time(
            lambda f=freq, t=raw21, v=raw11: calibration.correct21_bulk(f, t, v)
        )
        print(
            f"{count:>7} {t_scalar11 * 1e3:>9.2f}m {t_bulk11 * 1e3:>9.2f}m"
            f" {t_scalar11 / t_bulk11:>5.1f}x"
            f" {t_scalar21 * 1e3:>9.2f}m {t_bulk21 * 1e3:>9.2f}m"
            f" {t_scalar21 / t_bulk21:>5.1f}x"
            f" {error:>10.2e}"
        )

    print()
    print("conversion overhead (round trip, not part of either kernel):")
    for count in (401, 4001):
        points = _grid(calibration, count)
        freq, values = datapoints_to_arrays(points)
        frozen = values.astype(np.complex128)

        def back(grille=freq, vals=frozen):
            return arrays_to_datapoints(grille, vals)

        to_arrays = _time(lambda pts=points: datapoints_to_arrays(pts))
        from_arrays = _time(back)
        low, high = _spread(back)
        print(
            f"  {count:>5} points: to arrays {to_arrays * 1e3:.2f} ms,"
            f" from arrays {from_arrays * 1e3:.2f} ms"
            f" (spread {low * 1e3:.2f}..{high * 1e3:.2f} ms)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
