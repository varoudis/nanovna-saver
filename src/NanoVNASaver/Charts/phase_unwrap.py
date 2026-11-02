#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020  Rune B. Broberg
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
"""Phase unwrapping for charts, with undefined samples left where they are."""

import math

import numpy as np


def unwrap_degrees(phases) -> list[float]:
    """Unwrap ``phases`` in radians to degrees, keeping undefined samples.

    ``numpy.unwrap`` is a cumulative sum, so one NaN in the input turns
    every later value into NaN: the whole tail of the chart would lose its
    positions because of one bad sample.  Unwrapping each finite run on its
    own keeps the damage local to the sample and its neighbours.

    The undefined entries keep their position in the result and are NaN, so
    callers can decide what a gap means for them.
    """
    values = list(phases)
    unwrapped: list[float] = [math.nan] * len(values)

    start = 0
    while start < len(values):
        if not math.isfinite(values[start]):
            start += 1
            continue
        stop = start
        while stop < len(values) and math.isfinite(values[stop]):
            stop += 1
        run = np.degrees(np.unwrap(values[start:stop]))
        unwrapped[start:stop] = [float(value) for value in run]
        start = stop

    return unwrapped
