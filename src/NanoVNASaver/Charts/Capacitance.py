#  NanoVNASaver
#
#  A python program to view and export Touchstone data from a NanoVNA
#  Copyright (C) 2019, 2020  Rune B. Broberg
#  Copyright (C) 2020 NanoVNA-Saver Authors
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
import logging

import numpy as np

from .Frequency import FrequencyChart

logger = logging.getLogger(__name__)

#: Percent of the samples ignored at each end when looking for outliers.
PERCENTILE = 10.0

#: How far outside the central band a value may sit, in units of that
#: band's own size, before it stops setting the axis.
OUTLIER_FACTOR = 10.0


def _without_outliers(values: list[float]) -> list[float]:
    """Drop samples far outside the band the bulk of the sweep covers.

    A capacitance is ``-1 / (2 pi f X)``: an impedance whose reactance is
    a hair away from zero — a nearly perfect load — divides into a value
    thousands of times larger than the rest of the sweep.  It is a true
    number and the readout keeps showing it, but it is not what the user is
    looking at, so the axis must not stretch to reach it.  ``isPlotable``
    already leaves such a point out of the plot area.

    A sweep without outliers is returned unchanged, so its axis is exactly
    what it was before this check.
    """
    if len(values) < 3:
        return values
    low = float(np.percentile(values, PERCENTILE))
    high = float(np.percentile(values, 100.0 - PERCENTILE))
    band = max(high - low, abs(low), abs(high))
    if band == 0.0:
        return values
    limit = OUTLIER_FACTOR * band
    return [value for value in values if low - limit <= value <= high + limit]


class CapacitanceChart(FrequencyChart):
    def __init__(self, name=""):
        super().__init__(name)
        self.minDisplayValue = 0
        self.maxDisplayValue = 100
        self.name_unit = "F"
        self.value_function = lambda x: x.capacitiveEquivalent(
            self.reference_impedance
        )

    def _scaling_range(self, values: list[float]) -> tuple[float, float]:
        return super()._scaling_range(_without_outliers(values))
