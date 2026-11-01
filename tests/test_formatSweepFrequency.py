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


import unittest

# Import targets to be tested
from NanoVNASaver.Formatting import format_frequency_sweep


class TestCases(unittest.TestCase):
    def test_basicIntegerValues(self):
        # simple well-formed integers with no trailing zeros. Most importantly
        # there is no loss of accuracy in the result. Returned values are not
        # truncated if result would lose meaningful data.
        self.assertEqual(format_frequency_sweep(1), "1Hz")
        self.assertEqual(format_frequency_sweep(12), "12Hz")
        self.assertEqual(format_frequency_sweep(1234), "1.234kHz")
        self.assertEqual(format_frequency_sweep(12345), "12.345kHz")
        self.assertEqual(format_frequency_sweep(123456), "123.456kHz")
        self.assertEqual(format_frequency_sweep(123), "123Hz")
        self.assertEqual(format_frequency_sweep(1234567), "1.234567MHz")
        self.assertEqual(format_frequency_sweep(12345678), "12.345678MHz")
        self.assertEqual(format_frequency_sweep(123456789), "123.456789MHz")
