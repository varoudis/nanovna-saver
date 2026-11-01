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
"""Preset window (PRESET-02.5 / PRESET-03.1).

A thin view over :mod:`NanoVNASaver.Presets`.  The store owns persistence
and the refusal rules, the main window owns applying the settings, so this
window never touches the device, never starts a sweep, and reports every
store refusal verbatim in its message label instead of second-guessing it.
"""

import logging
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from ..Formatting import format_frequency_inputs
from ..Presets import (
    PRESET_SUFFIX,
    Preset,
    PresetError,
    PresetStore,
    PresetValidationError,
    SweepPreset,
    apply_preset,
    default_store_filename,
    find_builtin,
)
from .Defaults import make_scrollable
from .ui import get_window_icon

if TYPE_CHECKING:
    from ..NanoVNASaver.NanoVNASaver import NanoVNASaver as vna_app

logger = logging.getLogger(__name__)

#: Marker appended to the list entry of a preset that lives in code.
BUILTIN_SUFFIX = " (built-in)"

PRESET_FILE_FILTER = (
    "NanoVNA-Saver presets (*" + PRESET_SUFFIX + ");;All files (*)"
)


def describe_preset(preset: Preset) -> str:
    """Read-only summary of everything a preset would change."""
    sweep = preset.sweep
    lines = [
        preset.name,
        "",
        preset.description or "No description.",
        "",
        "Sweep: "
        f"{format_frequency_inputs(sweep.start)} to "
        f"{format_frequency_inputs(sweep.end)}",
        f"Points: {sweep.points}",
        f"Segments: {sweep.segments}",
        f"Averages: {sweep.averages}",
        "Mode: "
        f"{sweep.mode} ("
        f"{'logarithmic' if sweep.logarithmic else 'linear'} sweep)",
        f"Charts: {', '.join(preset.charts) or 'none'}",
        f"Analyses: {', '.join(preset.analyses) or 'none'}",
        "Parameters: "
        + (
            ", ".join(
                f"{key}={value}"
                for key, value in sorted(preset.parameters.items())
            )
            or "none"
        ),
    ]
    return "\n".join(lines)


class PresetWindow(QtWidgets.QWidget):
    """Browse, apply and manage measurement presets."""

    def __init__(
        self,
        app: "vna_app",
        store: PresetStore | None = None,
        filename: str = "",
    ):
        super().__init__()

        self.app = app
        self._load_error = ""
        self.message = ""

        if store is not None:
            self.store = store
        else:
            self.store = self._load_store(filename or default_store_filename())

        self.setWindowTitle("Presets")
        self.setWindowIcon(get_window_icon())
        self.setMinimumWidth(360)
        QtGui.QShortcut(QtCore.Qt.Key.Key_Escape, self, self.hide)

        layout = QtWidgets.QVBoxLayout()
        make_scrollable(self, layout)

        layout.addWidget(QtWidgets.QLabel("Presets"))
        self.list_presets = QtWidgets.QListWidget()
        self.list_presets.currentItemChanged.connect(
            lambda *_: self._show_details()
        )
        layout.addWidget(self.list_presets)

        self.btn_apply = QtWidgets.QPushButton("Apply preset")
        self.btn_apply.clicked.connect(lambda: self.apply_selected())
        layout.addWidget(self.btn_apply)

        self.btn_save = QtWidgets.QPushButton("Save current as preset ...")
        self.btn_save.clicked.connect(lambda: self.save_current_preset())
        layout.addWidget(self.btn_save)

        edit_layout = QtWidgets.QHBoxLayout()
        self.btn_rename = QtWidgets.QPushButton("Rename ...")
        self.btn_rename.clicked.connect(lambda: self.rename_selected())
        self.btn_duplicate = QtWidgets.QPushButton("Duplicate ...")
        self.btn_duplicate.clicked.connect(lambda: self.duplicate_selected())
        self.btn_delete = QtWidgets.QPushButton("Delete")
        self.btn_delete.clicked.connect(self._delete_clicked)
        edit_layout.addWidget(self.btn_rename)
        edit_layout.addWidget(self.btn_duplicate)
        edit_layout.addWidget(self.btn_delete)
        layout.addLayout(edit_layout)

        transfer_layout = QtWidgets.QHBoxLayout()
        self.btn_export = QtWidgets.QPushButton("Export ...")
        self.btn_export.clicked.connect(lambda: self.export_preset())
        self.btn_import = QtWidgets.QPushButton("Import ...")
        self.btn_import.clicked.connect(lambda: self.import_file())
        transfer_layout.addWidget(self.btn_export)
        transfer_layout.addWidget(self.btn_import)
        layout.addLayout(transfer_layout)

        layout.addWidget(QtWidgets.QLabel("Selected preset"))
        self.detail = QtWidgets.QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMinimumHeight(160)
        layout.addWidget(self.detail)

        self.label_message = QtWidgets.QLabel("")
        self.label_message.setObjectName("preset_message")
        self.label_message.setWordWrap(True)
        layout.addWidget(self.label_message)

        layout.addStretch(1)

        self.refresh()
        self._set_message(self._load_error)

    # ------------------------------------------------------------------
    # Store handling
    # ------------------------------------------------------------------
    def _load_store(self, filename: str) -> PresetStore:
        """Load user presets, tolerating an unreadable document."""
        try:
            return PresetStore.load(filename)
        except PresetError as exc:
            logger.warning("Could not load presets: %s", exc)
            self._load_error = str(exc)
            # Built-ins always come from code, so an empty store still
            # offers the complete built-in catalogue.
            return PresetStore(filename=filename)

    def refresh(self, select: str = "") -> None:
        """Rebuild the list from the store, keeping a sensible selection."""
        current = select or (self.selected_preset() or Preset("")).name
        self.list_presets.blockSignals(True)
        self.list_presets.clear()
        for preset in self.store.all_presets():
            # ``builtin_presets()`` returns copies whose flag is cleared,
            # so the catalogue is the authority on what is built in.
            builtin = preset.builtin or find_builtin(preset.name) is not None
            item = QtWidgets.QListWidgetItem(
                preset.name + (BUILTIN_SUFFIX if builtin else "")
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, preset)
            self.list_presets.addItem(item)
        self.list_presets.blockSignals(False)

        names = [preset.name for preset in self.store.all_presets()]
        if current in names:
            self.select(current)
        elif names:
            self.list_presets.setCurrentRow(0)
        else:
            self._show_details()

        conflicts = self.store.conflicts()
        if conflicts:
            self._set_message(
                "User presets shadow built-in names: "
                + ", ".join(sorted(conflicts))
            )

    def presets(self) -> list[Preset]:
        """Presets currently offered, built-ins first."""
        return [
            self.list_presets.item(row).data(QtCore.Qt.ItemDataRole.UserRole)
            for row in range(self.list_presets.count())
        ]

    def preset_names(self) -> list[str]:
        return [preset.name for preset in self.presets()]

    def selected_preset(self) -> Preset | None:
        item = self.list_presets.currentItem()
        if item is None:
            return None
        return item.data(QtCore.Qt.ItemDataRole.UserRole)

    def select(self, name: str) -> bool:
        for row in range(self.list_presets.count()):
            item = self.list_presets.item(row)
            preset = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if preset is not None and preset.name == name:
                self.list_presets.setCurrentItem(item)
                return True
        return False

    def _show_details(self) -> None:
        preset = self.selected_preset()
        self.detail.setPlainText(describe_preset(preset) if preset else "")

    def _set_message(self, text: str) -> None:
        self.message = text
        self.label_message.setText(text)

    # ------------------------------------------------------------------
    # Applying
    # ------------------------------------------------------------------
    def _device_max_frequency(self) -> float | None:
        limit = getattr(self.app.vna, "sweep_max_freq_hz", None)
        if isinstance(limit, (int, float)) and limit > 0:
            return float(limit)
        # An unknown or zero limit must not reject every preset.
        return None

    def apply_selected(self):
        """Validate the selection and hand it to the main window.

        Returns the :class:`PresetApplication` on success, ``None`` when
        nothing could be applied; a rejected preset leaves the current
        configuration untouched.
        """
        preset = self.selected_preset()
        if preset is None:
            self._set_message("Select a preset first.")
            return None
        try:
            application = apply_preset(
                preset,
                max_frequency=self._device_max_frequency(),
                valid_points=getattr(self.app.vna, "valid_datapoints", ()),
            )
        except PresetValidationError as exc:
            self._set_message(str(exc))
            return None

        self.app.apply_preset_application(application)
        message = f"Applied preset {preset.name!r}."
        if application.warnings:
            message += " " + "; ".join(application.warnings)
        self._set_message(message)
        return application

    # ------------------------------------------------------------------
    # Saving the current sweep
    # ------------------------------------------------------------------
    def current_sweep_preset(self, name: str) -> Preset:
        sweep = self.app.sweep
        return Preset(
            name=name,
            description="Saved from the current sweep settings.",
            sweep=SweepPreset(
                start=sweep.start,
                end=sweep.end,
                points=sweep.points,
                segments=sweep.segments,
                averages=sweep.properties.averages[0],
                mode=sweep.properties.mode.name,
                logarithmic=sweep.properties.logarithmic,
            ),
        )

    def save_current_preset(self, name: str = "") -> Preset | None:
        if not name:
            name, accepted = QtWidgets.QInputDialog.getText(
                self, "Save preset", "Preset name:", text="My preset"
            )
            if not accepted:
                return None
        name = name.strip()
        if not name:
            self._set_message("A preset name is required.")
            return None
        try:
            stored = self.store.add(self.current_sweep_preset(name))
            self.store.save()
        except PresetError as exc:
            self._set_message(str(exc))
            return None
        self.refresh(select=stored.name)
        self._set_message(f"Saved preset {stored.name!r}.")
        return stored

    # ------------------------------------------------------------------
    # User preset management
    # ------------------------------------------------------------------
    def rename_selected(self, new_name: str = "") -> Preset | None:
        preset = self.selected_preset()
        if preset is None:
            self._set_message("Select a preset first.")
            return None
        if not new_name:
            new_name, accepted = QtWidgets.QInputDialog.getText(
                self, "Rename preset", "New name:", text=preset.name
            )
            if not accepted:
                return None
        try:
            renamed = self.store.rename(preset.name, new_name.strip())
            self.store.save()
        except PresetError as exc:
            self._set_message(str(exc))
            return None
        self.refresh(select=renamed.name)
        self._set_message(
            f"Renamed preset {preset.name!r} to {renamed.name!r}."
        )
        return renamed

    def duplicate_selected(self, new_name: str = "") -> Preset | None:
        preset = self.selected_preset()
        if preset is None:
            self._set_message("Select a preset first.")
            return None
        if not new_name:
            new_name, accepted = QtWidgets.QInputDialog.getText(
                self,
                "Duplicate preset",
                "New name:",
                text=f"{preset.name} copy",
            )
            if not accepted:
                return None
        try:
            copied = self.store.duplicate(preset.name, new_name.strip())
            self.store.save()
        except PresetError as exc:
            self._set_message(str(exc))
            return None
        self.refresh(select=copied.name)
        self._set_message(
            f"Duplicated preset {preset.name!r} as {copied.name!r}."
        )
        return copied

    def delete_preset(self, name: str) -> bool:
        try:
            removed = self.store.delete(name)
        except PresetError as exc:
            self._set_message(str(exc))
            return False
        if not removed:
            self._set_message(f"No user preset named {name!r}.")
            return False
        self.store.save()
        self.refresh()
        self._set_message(f"Deleted preset {name!r}.")
        return True

    def _delete_clicked(self) -> None:
        preset = self.selected_preset()
        if preset is None:
            self._set_message("Select a preset first.")
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete preset",
            f"Delete the user preset {preset.name!r}?",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.delete_preset(preset.name)

    # ------------------------------------------------------------------
    # Transfer
    # ------------------------------------------------------------------
    def export_preset(self, filename: str = "", name: str = "") -> bool:
        preset = self.selected_preset() if not name else self.store.find(name)
        if preset is None:
            self._set_message("Select a preset first.")
            return False
        if not filename:
            filename, _selected = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export preset",
                preset.name + PRESET_SUFFIX,
                PRESET_FILE_FILTER,
            )
            if not filename:
                return False
        try:
            self.store.export_preset(preset.name, filename)
        except (PresetError, OSError) as exc:
            self._set_message(str(exc))
            return False
        self._set_message(f"Exported preset {preset.name!r} to {filename}.")
        return True

    def import_file(
        self, filename: str = "", new_name: str = ""
    ) -> Preset | None:
        if not filename:
            filename, _selected = QtWidgets.QFileDialog.getOpenFileName(
                self, "Import preset", "", PRESET_FILE_FILTER
            )
            if not filename:
                return None
        try:
            preset = self.store.import_preset(filename, new_name=new_name)
            self.store.save()
        except PresetError as exc:
            self._set_message(str(exc))
            return None
        self.refresh(select=preset.name)
        self._set_message(f"Imported preset {preset.name!r}.")
        return preset
