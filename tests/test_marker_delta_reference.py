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
"""The delta readout must describe the mode its title names.

``MarkerControl.toggle_delta_reference`` retitles the delta group box.  The
values themselves are produced by ``NanoVNASaver.markerUpdated``, which pairs
marker 2 with marker 1 - or, in reference mode, with the reference sweep.
If the toggle does not ask for that rebuild, the panel keeps the partner of
the previous mode and shows nothing until a marker is moved.
"""

import pytest

from NanoVNASaver.RFTools import Datapoint

FREQ_LABEL = "actualfreq"
MARKER_ONE = 10_000_000
MARKER_TWO = 15_000_000


@pytest.fixture
def window(qt_app):
    from NanoVNASaver.NanoVNASaver import NanoVNASaver

    app = NanoVNASaver()
    app.session.dirty = False
    app.data.s11 = [
        Datapoint(MARKER_ONE + index * 1_000_000, 0.2, 0.05 * (index % 5))
        for index in range(20)
    ]
    app.data.s21 = []
    app.markerUpdated(app.markers[0])
    try:
        yield app
    finally:
        app.session.dirty = False
        app.worker.quit()
        app.worker.wait(5000)
        app.deleteLater()


def _enable_delta(window):
    control = window.marker_control
    control.check_delta.setChecked(True)
    window.markers[1].setFrequency(f"{MARKER_TWO // 1_000_000} MHz")
    return control


class TestReferenceToggle:
    @staticmethod
    def test_toggling_back_restores_marker_two_and_its_values(window):
        control = _enable_delta(window)
        delta = window.delta_marker
        before = delta.label[FREQ_LABEL].text()
        assert before

        control.check_delta_reference.setChecked(True)
        assert window.marker_ref is True
        control.check_delta_reference.setChecked(False)

        assert window.marker_ref is False
        assert delta.marker_a is window.markers[0]
        assert delta.marker_b is window.markers[1]
        # The title names the pair (Delta.py sets it from both markers).
        assert delta.group_box.title() == (
            f"Delta {window.markers[1].name} - {window.markers[0].name}"
        )
        assert delta.label[FREQ_LABEL].text() == before

    @staticmethod
    def test_reference_mode_pairs_with_the_reference_sweep(window):
        control = _enable_delta(window)
        window.setReference()
        delta = window.delta_marker

        control.check_delta_reference.setChecked(True)

        assert delta.marker_a is window.markers[0]
        assert delta.marker_b is not window.markers[1]
        assert delta.marker_b.name == "Reference"
        assert delta.group_box.title().startswith("Delta Reference - ")
        assert delta.label[FREQ_LABEL].text()

    @staticmethod
    def test_no_stale_partner_after_a_double_toggle(window):
        control = _enable_delta(window)
        window.setReference()

        control.check_delta_reference.setChecked(True)
        control.check_delta_reference.setChecked(False)

        delta = window.delta_marker
        assert delta.marker_b is window.markers[1]
        assert delta.label[FREQ_LABEL].text() == "5.00000 MHz"

    @staticmethod
    def test_a_hidden_panel_never_re_shows_the_other_mode(window):
        """Toggling while the panel is hidden must not leave old numbers."""
        control = _enable_delta(window)
        window.setReference()
        delta = window.delta_marker
        assert delta.label[FREQ_LABEL].text()

        control.check_delta.setChecked(False)
        control.check_delta_reference.setChecked(True)
        control.check_delta.setChecked(True)

        assert delta.group_box.title().startswith("Delta Reference - ")
        assert delta.marker_b.name == "Reference"
        assert delta.label[FREQ_LABEL].text()
