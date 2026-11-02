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
import logging
import math

from PySide6 import QtGui

from ..Marker.Widget import Marker
from ..RFTools import Datapoint
from ..SITools import Format, Value
from .Chart import Chart
from .Frequency import FrequencyChart

logger = logging.getLogger(__name__)


def _scaled(d: Datapoint, reference: float, *, imaginary: bool) -> float | None:
    """One scaled axis value of ``d``; ``None`` when it is not a number.

    ``re * 10e6 / freq`` is this chart's own scale factor, and ``inf / inf``
    or ``0 * inf`` would turn a defined reading into NaN, so the finite
    check happens after the scaling.
    """
    imp = d.impedance(reference)
    value = (imp.imag if imaginary else imp.real) * 10e6 / d.freq
    return value if math.isfinite(value) else None


def _permeability_values(d: Datapoint, reference: float) -> list[float]:
    """The finite real and imaginary axis values of ``d``."""
    values = [
        _scaled(d, reference, imaginary=False),
        _scaled(d, reference, imaginary=True),
    ]
    return [value for value in values if value is not None]


class PermeabilityChart(FrequencyChart):
    def __init__(self, name=""):
        super().__init__(name)
        self.leftMargin = 40
        self.rightMargin = 30
        self.dim.width = 230
        self.dim.height = 250
        self.fstart = 0
        self.fstop = 0
        self.span = 0.01
        self.max = 0

        self.maxDisplayValue = 100
        self.minDisplayValue = -100

    def logarithmicYAllowed(self) -> bool:
        return True

    def drawChart(self, qp: QtGui.QPainter):
        qp.setPen(QtGui.QPen(Chart.color.text))
        qp.drawText(
            self.leftMargin + 5,
            15,
            self.name + " (\N{MICRO SIGN}\N{OHM SIGN} / Hz)",
        )
        qp.drawText(10, 15, "R")
        qp.drawText(self.leftMargin + self.dim.width + 10, 15, "X")
        qp.setPen(QtGui.QPen(Chart.color.foreground))
        qp.drawLine(
            self.leftMargin,
            self.topMargin - 5,
            self.leftMargin,
            self.topMargin + self.dim.height + 5,
        )
        qp.drawLine(
            self.leftMargin - 5,
            self.topMargin + self.dim.height,
            self.leftMargin + self.dim.width + 5,
            self.topMargin + self.dim.height,
        )
        self.drawTitle(qp)

    def drawValues(self, qp: QtGui.QPainter):
        if not self.data and not self.reference:
            return

        pen = QtGui.QPen(Chart.color.sweep)
        pen.setWidth(self.dim.point)
        line_pen = QtGui.QPen(Chart.color.sweep)
        line_pen.setWidth(self.dim.line)

        self._set_start_stop()

        # Draw bands if required
        if self.bands.enabled:
            self.drawBands(qp, self.fstart, self.fstop)

        # Find scaling
        if self.fixedValues:
            min_val = float(self.minDisplayValue)
            max_val = float(self.maxDisplayValue)
        else:
            min_val = 1000.0
            max_val = -1000.0
            # An undefined impedance has no magnitude: skipping it keeps the
            # axis finite and leaves that sample as a gap (see
            # Frequency._find_scaling).
            for d in self.data:
                for value in _permeability_values(d, self.reference_impedance):
                    max_val = max(max_val, value)
                    min_val = min(min_val, value)
            # Also check min/max for the reference sweep
            for d in self.reference:
                if d.freq < self.fstart or d.freq > self.fstop:
                    continue
                for value in _permeability_values(d, self.reference_impedance):
                    max_val = max(max_val, value)
                    min_val = min(min_val, value)

        if self.logarithmicY:
            min_val = max(0.01, min_val)

        self.max = max_val
        self.span = (max_val - min_val) or 0.01

        # We want one horizontal tick per 50 pixels, at most
        horizontal_ticks = math.floor(self.dim.height / 50)
        fmt = Format(max_nr_digits=4)
        for i in range(horizontal_ticks):
            y = self.topMargin + round(i * self.dim.height / horizontal_ticks)
            qp.setPen(QtGui.QPen(Chart.color.foreground))
            qp.drawLine(
                self.leftMargin - 5, y, self.leftMargin + self.dim.width + 5, y
            )
            qp.setPen(QtGui.QPen(Chart.color.text))
            val = Value(self.valueAtPosition(y)[0], fmt=fmt)
            qp.drawText(3, y + 4, str(val))

        qp.drawText(
            3, self.dim.height + self.topMargin, str(Value(min_val, fmt=fmt))
        )

        self.drawFrequencyTicks(qp)

        primary_pen = pen
        secondary_pen = QtGui.QPen(Chart.color.sweep_secondary)
        if self.data:
            c = QtGui.QColor(Chart.color.sweep)
            c.setAlpha(255)
            pen = QtGui.QPen(c)
            pen.setWidth(2)
            qp.setPen(pen)
            qp.drawLine(20, 9, 25, 9)
            c = QtGui.QColor(Chart.color.sweep_secondary)
            c.setAlpha(255)
            pen.setColor(c)
            qp.setPen(pen)
            qp.drawLine(
                self.leftMargin + self.dim.width,
                9,
                self.leftMargin + self.dim.width + 5,
                9,
            )

        primary_pen.setWidth(self.dim.point)
        secondary_pen.setWidth(self.dim.point)
        line_pen.setWidth(self.dim.line)

        for i, data in enumerate(self.data):
            x = self.getXPosition(data)
            y_re = self.getReYPosition(data)
            y_im = self.getImYPosition(data)
            qp.setPen(primary_pen)
            if self.isPlotable(x, y_re):
                qp.drawPoint(x, y_re)
            qp.setPen(secondary_pen)
            if self.isPlotable(x, y_im):
                qp.drawPoint(x, y_im)
            if self.flag.draw_lines and i > 0:
                prev_x = self.getXPosition(self.data[i - 1])
                prev_y_re = self.getReYPosition(self.data[i - 1])
                prev_y_im = self.getImYPosition(self.data[i - 1])

                # Real part first
                line_pen.setColor(Chart.color.sweep)
                qp.setPen(line_pen)
                if self.isPlotable(x, y_re):
                    if self.isPlotable(prev_x, prev_y_re):
                        qp.drawLine(x, y_re, prev_x, prev_y_re)
                    else:
                        new_x, new_y = self.getPlotable(
                            x, y_re, prev_x, prev_y_re
                        )
                        qp.drawLine(x, y_re, new_x, new_y)
                elif self.isPlotable(prev_x, prev_y_re):
                    new_x, new_y = self.getPlotable(prev_x, prev_y_re, x, y_re)
                    qp.drawLine(prev_x, prev_y_re, new_x, new_y)

                # Imag part second
                line_pen.setColor(Chart.color.sweep_secondary)
                qp.setPen(line_pen)
                if self.isPlotable(x, y_im):
                    if self.isPlotable(prev_x, prev_y_im):
                        qp.drawLine(x, y_im, prev_x, prev_y_im)
                    else:
                        new_x, new_y = self.getPlotable(
                            x, y_im, prev_x, prev_y_im
                        )
                        qp.drawLine(x, y_im, new_x, new_y)
                elif self.isPlotable(prev_x, prev_y_im):
                    new_x, new_y = self.getPlotable(prev_x, prev_y_im, x, y_im)
                    qp.drawLine(prev_x, prev_y_im, new_x, new_y)

        primary_pen.setColor(Chart.color.reference)
        line_pen.setColor(Chart.color.reference)
        secondary_pen.setColor(Chart.color.reference_secondary)
        qp.setPen(primary_pen)
        if self.reference:
            c = QtGui.QColor(Chart.color.reference)
            c.setAlpha(255)
            pen = QtGui.QPen(c)
            pen.setWidth(2)
            qp.setPen(pen)
            qp.drawLine(20, 14, 25, 14)
            c = QtGui.QColor(Chart.color.reference_secondary)
            c.setAlpha(255)
            pen = QtGui.QPen(c)
            pen.setWidth(2)
            qp.setPen(pen)
            qp.drawLine(
                self.leftMargin + self.dim.width,
                14,
                self.leftMargin + self.dim.width + 5,
                14,
            )

        for i, reference in enumerate(self.reference):
            if reference.freq < self.fstart or reference.freq > self.fstop:
                continue
            x = self.getXPosition(reference)
            y_re = self.getReYPosition(reference)
            y_im = self.getImYPosition(reference)
            qp.setPen(primary_pen)
            if self.isPlotable(x, y_re):
                qp.drawPoint(x, y_re)
            qp.setPen(secondary_pen)
            if self.isPlotable(x, y_im):
                qp.drawPoint(x, y_im)
            if self.flag.draw_lines and i > 0:
                prev_x = self.getXPosition(self.reference[i - 1])
                prev_y_re = self.getReYPosition(self.reference[i - 1])
                prev_y_im = self.getImYPosition(self.reference[i - 1])

                line_pen.setColor(Chart.color.reference)
                qp.setPen(line_pen)
                # Real part first
                if self.isPlotable(x, y_re):
                    if self.isPlotable(prev_x, prev_y_re):
                        qp.drawLine(x, y_re, prev_x, prev_y_re)
                    else:
                        new_x, new_y = self.getPlotable(
                            x, y_re, prev_x, prev_y_re
                        )
                        qp.drawLine(x, y_re, new_x, new_y)
                elif self.isPlotable(prev_x, prev_y_re):
                    new_x, new_y = self.getPlotable(prev_x, prev_y_re, x, y_re)
                    qp.drawLine(prev_x, prev_y_re, new_x, new_y)

                line_pen.setColor(Chart.color.reference_secondary)
                qp.setPen(line_pen)
                # Imag part second
                if self.isPlotable(x, y_im):
                    if self.isPlotable(prev_x, prev_y_im):
                        qp.drawLine(x, y_im, prev_x, prev_y_im)
                    else:
                        new_x, new_y = self.getPlotable(
                            x, y_im, prev_x, prev_y_im
                        )
                        qp.drawLine(x, y_im, new_x, new_y)
                elif self.isPlotable(prev_x, prev_y_im):
                    new_x, new_y = self.getPlotable(prev_x, prev_y_im, x, y_im)
                    qp.drawLine(prev_x, prev_y_im, new_x, new_y)

        # Now draw the markers
        for m in self.markers:
            if m.location != -1:
                x = self.getXPosition(self.data[m.location])
                y_re = self.getReYPosition(self.data[m.location])
                y_im = self.getImYPosition(self.data[m.location])

                number = self.markers.index(m) + 1
                if y_re is not None:
                    self.drawMarker(x, y_re, qp, m.color, number)
                if y_im is not None:
                    self.drawMarker(x, y_im, qp, m.color, number)

    def getImYPosition(self, d: Datapoint) -> int | None:
        im = _scaled(d, self.reference_impedance, imaginary=True)
        if im is None:
            return None
        if self.logarithmicY:
            min_val = self.max - self.span
            if self.max > 0 and min_val > 0 and im > 0:
                span = math.log(self.max) - math.log(min_val)
            else:
                # Not on a logarithmic axis: a gap, not an edge.
                return None
            return int(
                self.topMargin
                + (math.log(self.max) - math.log(im)) / span * self.dim.height
            )
        return int(
            self.topMargin + (self.max - im) / self.span * self.dim.height
        )

    def getReYPosition(self, d: Datapoint) -> int | None:
        re = _scaled(d, self.reference_impedance, imaginary=False)
        if re is None:
            return None
        if self.logarithmicY:
            min_val = self.max - self.span
            if self.max > 0 and min_val > 0 and re > 0:
                span = math.log(self.max) - math.log(min_val)
            else:
                # Not on a logarithmic axis: a gap, not an edge.
                return None
            return int(
                self.topMargin
                + (math.log(self.max) - math.log(re)) / span * self.dim.height
            )
        return int(
            self.topMargin + (self.max - re) / self.span * self.dim.height
        )

    def valueAtPosition(self, y) -> list[float]:
        absy = y - self.topMargin
        if self.logarithmicY:
            min_val = self.max - self.span
            if self.max > 0 and min_val > 0:
                span = math.log(self.max) - math.log(min_val)
                step = span / self.dim.height
                val = math.exp(math.log(self.max) - absy * step)
            else:
                val = -1
        else:
            val = -1 * ((absy / self.dim.height * self.span) - self.max)
        return [val]

    def getNearestMarker(self, x, y) -> Marker:
        if len(self.data) == 0:
            return Marker()
        shortest = 10.0**6
        nearest = Marker()
        for m in self.markers:
            mx, _ = self.getPosition(self.data[m.location])
            myr = self.getReYPosition(self.data[m.location])
            myi = self.getImYPosition(self.data[m.location])
            if mx is None or (myr is None and myi is None):
                # Nothing to measure a distance to.
                continue
            dx = abs(x - mx)
            dy = min(
                abs(y - candidate)
                for candidate in (myr, myi)
                if candidate is not None
            )
            distance = math.sqrt(dx**2 + dy**2)
            if distance < shortest:
                shortest = distance
                nearest = m
        return nearest
