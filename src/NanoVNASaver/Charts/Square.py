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

from PySide6 import QtCore, QtGui, QtWidgets

from ..RFTools import Datapoint
from .Chart import Chart

logger = logging.getLogger(__name__)


class SquareChart(Chart):
    def __init__(self, name=""):
        super().__init__(name)
        sizepolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
        )
        self.setSizePolicy(sizepolicy)
        self.dim.width = 250
        self.dim.height = 250
        self.setMinimumSize(self.dim.width + 40, self.dim.height + 40)

        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.ColorRole.Window, Chart.color.background)
        self.setPalette(pal)
        self.setAutoFillBackground(True)

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        qp = QtGui.QPainter(self)
        self.drawChart(qp)
        self.drawValues(qp)
        qp.end()

    def drawChart(self, qp: QtGui.QPainter) -> None:
        raise NotImplementedError()

    def draw_data(
        self,
        qp: QtGui.QPainter,
        color: QtGui.QColor,
        data: list[Datapoint],
        fstart: int = 0,
        fstop: int = 0,
    ):
        if not data:
            return
        fstop = fstop or data[-1].freq
        pen = QtGui.QPen(color)
        pen.setWidth(self.dim.point)
        line_pen = QtGui.QPen(color)
        line_pen.setWidth(self.dim.line)

        qp.setPen(pen)
        prev_x = self.getXPosition(data[0])
        prev_y = self.getYPosition(data[0])
        for i, d in enumerate(data):
            x = self.getXPosition(d)
            y = self.getYPosition(d)
            # A point with an undefined coordinate has no place on the
            # plane: it is a gap, and neither the dot nor either line
            # segment touching it is drawn.
            if x is None or y is None:
                prev_x, prev_y = x, y
                continue
            if d.freq > fstart and d.freq < fstop:
                qp.drawPoint(x, y)
                if (
                    self.flag.draw_lines
                    and i > 0
                    and prev_x is not None
                    and prev_y is not None
                ):
                    qp.setPen(line_pen)
                    qp.drawLine(x, y, prev_x, prev_y)
                    qp.setPen(pen)
            prev_x, prev_y = x, y

    def drawValues(self, qp: QtGui.QPainter):
        if not (self.data or self.reference or self.traces):
            return
        self.draw_data(qp, Chart.color.sweep, self.data)

        fstart = self.data[0].freq if self.data else 0
        fstop = self.data[-1].freq if self.data else 0
        self.draw_data(qp, Chart.color.reference, self.reference, fstart, fstop)

        # Additional traces (TRACE-02.2): a Smith or polar plot is where an
        # overlay is most useful, so refusing to draw them here would make
        # the multi-trace feature silently chart-dependent.
        for _name, color, points in self.traces:
            self.draw_data(qp, QtGui.QColor(color), points)

        for m in self.markers:
            if m.location != -1 and m.location < len(self.data):
                x = self.getXPosition(self.data[m.location])
                y = self.getYPosition(self.data[m.location])
                if x is None or y is None:
                    continue
                self.drawMarker(x, y, qp, m.color, self.markers.index(m) + 1)

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        if not self.flag.is_popout:
            self.setFixedWidth(a0.size().height())
            self.dim.width = a0.size().height() - 40
            self.dim.height = a0.size().height() - 40
        else:
            min_dimension = min(a0.size().height(), a0.size().width())
            self.dim.width = self.dim.height = min_dimension - 40
        self.update()

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent):
        if a0.buttons() == QtCore.Qt.MouseButton.RightButton:
            a0.ignore()
            return

        x = a0.position().x()
        y = a0.position().y()
        absx = x - (self.width() - self.dim.width) / 2
        absy = y - (self.height() - self.dim.height) / 2
        if (
            absx < 0
            or absx > self.dim.width
            or absy < 0
            or absy > self.dim.height
            or (not self.data and not self.reference)
        ):
            a0.ignore()
            return
        a0.accept()

        target = self.data or self.reference
        nearest = None
        shortest = math.inf
        for index, d in enumerate(target):
            px, py = self.getPosition(d)
            if px is None or py is None:
                # An undefined point has no position to snap to.
                continue
            distance = math.hypot(x - px, y - py)
            if distance < shortest:
                shortest = distance
                nearest = index

        if nearest is not None and (m := self.getActiveMarker()):
            m.setFrequency(str(round(target[nearest].freq)))

    def getXPosition(self, d: Datapoint) -> int | None:
        if not math.isfinite(d.re):
            return None
        return int(self.width() / 2 + d.re * self.dim.width / 2)

    def getYPosition(self, d: Datapoint) -> int | None:
        if not math.isfinite(d.im):
            return None
        return int(self.height() / 2 + d.im * -1 * self.dim.height / 2)

    def zoomTo(self, x1, y1, x2, y2):
        pass
