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
import logging
from typing import TYPE_CHECKING

from PySide6 import QtWidgets

if TYPE_CHECKING:
    from ..NanoVNASaver.NanoVNASaver import NanoVNASaver as vna_app

logger = logging.getLogger(__name__)

CUTOFF_VALS: tuple[float, ...] = (3.0, 6.0, 10.0, 20.0, 60.0)
MIN_CUTOFF_DAMPING: float = -4.0


class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Shape.HLine)


class Analysis:
    def __init__(self, app: "vna_app") -> None:
        self.app = app
        self.label: dict[str, QtWidgets.QLabel] = {
            "titel": QtWidgets.QLabel(),
            "result": QtWidgets.QLabel(),
        }
        self.layout = QtWidgets.QFormLayout()
        self._widget = QtWidgets.QWidget()
        self._widget.setLayout(self.layout)

    def _reference(self) -> float:
        """Reference impedance of the displayed data (RF-02).

        Analyses derive resistance and reactance from a reflection
        coefficient, so they must use the reference the data was measured
        against.  It is read from the application on every call rather than
        cached, because importing a file can change it mid-session.
        """
        return float(getattr(self.app, "reference_impedance", 50.0) or 50.0)

    def widget(self) -> QtWidgets.QWidget:
        return self._widget

    def runAnalysis(self) -> None:
        pass

    def reset(self) -> None:
        for label in self.label.values():
            label.clear()

    def set_result(self, text) -> None:
        self.label["result"].setText(text)

    def set_titel(self, text) -> None:
        self.label["titel"].setText(text)
