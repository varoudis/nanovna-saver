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
"""Session window (SESSION-03.1).

A thin view: every action delegates to the main window, which owns the
document and the dirty state.
"""

import logging
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from .Defaults import make_scrollable
from .ui import get_window_icon

if TYPE_CHECKING:
    from ..NanoVNASaver.NanoVNASaver import NanoVNASaver as vna_app

logger = logging.getLogger(__name__)

#: Recent sessions kept per application run.
MAX_RECENT = 8


class SessionWindow(QtWidgets.QWidget):
    def __init__(self, app: "vna_app"):
        super().__init__()
        self.app = app
        self.recent: list[str] = []

        self.setWindowTitle("Session")
        self.setWindowIcon(get_window_icon())
        self.setMinimumWidth(260)
        QtGui.QShortcut(QtCore.Qt.Key.Key_Escape, self, self.hide)

        layout = QtWidgets.QVBoxLayout()
        make_scrollable(self, layout)

        self.label_state = QtWidgets.QLabel("Untitled")
        self.label_state.setWordWrap(True)
        layout.addWidget(self.label_state)

        self.input_title = QtWidgets.QLineEdit()
        self.input_title.setPlaceholderText("Session title")
        self.input_title.editingFinished.connect(self._title_changed)
        layout.addWidget(self.input_title)

        for text, slot in (
            ("New session", self.app.action_new_session),
            ("Open session ...", self.app.action_open_session),
            ("Save session", self.app.action_save_session),
            ("Save session as ...", self.app.action_save_session_as),
        ):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            layout.addWidget(button)

        layout.addWidget(QtWidgets.QLabel("Recent sessions"))
        self.list_recent = QtWidgets.QListWidget()
        self.list_recent.itemActivated.connect(self._open_recent)
        self.list_recent.itemClicked.connect(self._open_recent)
        layout.addWidget(self.list_recent)

        self.label_recovery = QtWidgets.QLabel("")
        self.label_recovery.setWordWrap(True)
        layout.addWidget(self.label_recovery)

        layout.addStretch(1)
        self.refresh()

    def _title_changed(self) -> None:
        title = self.input_title.text().strip()
        if title and title != self.app.session.title:
            self.app.session.title = title
            self.app.mark_session_changed("title")
            self.app.update_session_title()

    def refresh(self) -> None:
        session = self.app.session
        self.label_state.setText(
            f"{session.display_name}"
            + (f"\n{session.filename}" if session.filename else "")
        )
        if self.input_title.text() != session.title:
            self.input_title.setText(session.title)
        self.list_recent.clear()
        for filename in self.recent:
            self.list_recent.addItem(filename)
        self.label_recovery.setText(
            f"Recovery snapshot: {session.recovery_filename}"
            if session.recovery_filename
            else "No recovery snapshot"
        )

    def remember(self, filename: str) -> None:
        """Add ``filename`` to the recent list, most recent first."""
        if filename in self.recent:
            self.recent.remove(filename)
        self.recent.insert(0, filename)
        del self.recent[MAX_RECENT:]
        self.refresh()

    def _open_recent(self, item) -> None:
        filename = item.text()
        if not filename:
            return
        if not self.app._confirm_discard():
            return
        self.app.open_session_file(filename)
