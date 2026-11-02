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
import csv
import logging

from PySide6 import QtWidgets

from .. import AnalyticTools as At
from ..Formatting import (
    format_complex_imp,
    format_frequency,
    format_frequency_short,
)
from .ResonanceAnalysis import (
    ResonanceAnalysis,
    format_resistence_neg,
)

logger = logging.getLogger(__name__)


class EFHWAnalysis(ResonanceAnalysis):
    """
    find only resonance when HI impedance
    """

    def __init__(self, app):
        super().__init__(app)
        self.old_data = []

    def do_resonance_analysis(self):
        s11 = self.app.data.s11
        maximums = sorted(
            At.maxima(
                [d.impedance(self._reference()).real for d in s11],
                threshold=500,
            )
        )
        extended_data = {}
        logger.info("TO DO: find near data")
        for lowest in self.crossings:
            my_data = self._get_data(lowest)
            if lowest in extended_data:
                extended_data[lowest].update(my_data)
            else:
                extended_data[lowest] = my_data
        logger.debug("maximumx %s of type %s", maximums, type(maximums))
        for m in maximums:
            logger.debug("m %s of type %s", m, type(m))
            my_data = self._get_data(m)
            if m in extended_data:
                extended_data[m].update(my_data)
            else:
                extended_data[m] = my_data
        fields = [
            ("freq", format_frequency_short),
            ("r", format_resistence_neg),
            ("lambda", lambda x: round(x, 2)),
        ]

        if self.old_data:
            diff = self.compare(self.old_data[-1], extended_data, fields=fields)
        else:
            diff = self.compare({}, extended_data, fields=fields)
        self.old_data.append(extended_data)
        for i, idx in enumerate(sorted(extended_data.keys())):
            self.layout.addRow(
                f"{format_frequency_short(s11[idx].freq)}",
                QtWidgets.QLabel(
                    f" ({diff[i]['freq']})"
                    " "
                    f"{format_complex_imp(s11[idx].impedance(self._reference()))}"
                    f" ({diff[i]['r']}) {diff[i]['lambda']} m"
                ),
            )

        if self.filename and extended_data:
            with open(
                self.filename, "w", newline="", encoding="utf-8"
            ) as csvfile:
                fieldnames = extended_data[
                    sorted(extended_data.keys())[0]
                ].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for idx in sorted(extended_data.keys()):
                    writer.writerow(extended_data[idx])

    def compare(self, old, new, fields=None):
        """
        Compare data to help changes

        NB
        must be same sweep
        ( same index must be same frequence )
        :param old:
        :param new:
        """
        fields = fields or [
            ("freq", str),
        ]

        def no_compare():
            return {k: "-" for k, _ in fields}

        old_idx = sorted(old.keys())
        # 'odict_keys' object is not subscriptable
        new_idx = sorted(new.keys())
        diff = {}

        if len(old_idx) != len(new_idx):
            logger.warning(
                "resonances changed from %s to %s", len(old_idx), len(new_idx)
            )

        # Walk both sweeps in frequency order.  ``pos`` is the next unused
        # old resonance: a new band is paired with it when the two are close
        # enough, otherwise the cursor stays (a band was added to the new
        # sweep) or moves on (a band disappeared from it).  Skipping is a
        # loop, not a single step, so any number of bands may come or go;
        # it is the offset between the two sweeps, not a lookup index.
        pos = 0
        max_delta_f = 1_000_000
        for i, k in enumerate(new_idx):
            new_freq = new[k]["freq"]
            logger.info("Resonance %s at %s", i, new_freq)

            while (
                pos < len(old_idx)
                and new_freq - old[old_idx[pos]]["freq"] >= max_delta_f
            ):
                logger.debug(
                    "new is missing band %s",
                    format_frequency(old[old_idx[pos]]["freq"]),
                )
                pos += 1

            if (
                pos < len(old_idx)
                and abs(new_freq - old[old_idx[pos]]["freq"]) < max_delta_f
            ):
                logger.debug("can compare")
                diff[i] = {
                    desc: fnc(new[k][desc] - old[old_idx[pos]][desc])
                    for desc, fnc in fields
                }
                logger.debug("Deltas %s", diff[i])
                pos += 1
                continue

            logger.debug("new band, non compare ")
            diff[i] = no_compare()

        return diff
