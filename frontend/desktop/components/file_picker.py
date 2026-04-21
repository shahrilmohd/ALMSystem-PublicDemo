"""
FilePicker and FolderPicker — reusable path-selection widgets.

FilePicker
    A label + read-only QLineEdit + Browse button.
    Opens a file dialog filtered to specified file types.
    Emits `path_changed(str)` when the user picks a file.

FolderPicker
    Same layout but opens a directory dialog instead.
    Used for the Assumptions Folder and Output Folder inputs.
    Emits `path_changed(str)` when the user picks a directory.

Both widgets expose:
    path()          → str   current path string ("" if nothing selected)
    set_path(str)           set the path programmatically (no signal emitted)
    clear()                 reset to empty (no signal emitted)
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)


class FilePicker(QWidget):
    """
    Inline file browser: [Label]  [path display]  [Browse]

    Parameters
    ----------
    label : str
        Text shown to the left of the path display.
    file_filter : str
        Qt file filter string, e.g. "CSV files (*.csv);;All files (*)".
        Default: all files.
    placeholder : str
        Placeholder text shown in the path display when empty.
    parent : QWidget, optional
    """

    path_changed = pyqtSignal(str)

    def __init__(
        self,
        label: str = "File",
        file_filter: str = "All files (*)",
        placeholder: str = "No file selected",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._file_filter = file_filter

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(label)
        lbl.setFixedWidth(140)
        layout.addWidget(lbl)

        self._line = QLineEdit()
        self._line.setReadOnly(True)
        self._line.setPlaceholderText(placeholder)
        self._line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self._line)

        btn = QPushButton("Browse…")
        btn.setFixedWidth(80)
        btn.clicked.connect(self._browse)
        layout.addWidget(btn)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def path(self) -> str:
        """Return the current path string, or "" if nothing is selected."""
        return self._line.text()

    def set_path(self, p: str) -> None:
        """Set path programmatically. Does not emit path_changed."""
        self._line.setText(p)

    def clear(self) -> None:
        """Reset to empty. Does not emit path_changed."""
        self._line.clear()

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select file",
            self._line.text() or "",
            self._file_filter,
        )
        if path:
            self._line.setText(path)
            self.path_changed.emit(path)


class FolderPicker(QWidget):
    """
    Inline directory browser: [Label]  [path display]  [Browse]

    Parameters
    ----------
    label : str
        Text shown to the left of the path display.
    placeholder : str
        Placeholder text shown in the path display when empty.
    parent : QWidget, optional
    """

    path_changed = pyqtSignal(str)

    def __init__(
        self,
        label: str = "Folder",
        placeholder: str = "No folder selected",
        parent=None,
    ) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(label)
        lbl.setFixedWidth(140)
        layout.addWidget(lbl)

        self._line = QLineEdit()
        self._line.setReadOnly(True)
        self._line.setPlaceholderText(placeholder)
        self._line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self._line)

        btn = QPushButton("Browse…")
        btn.setFixedWidth(80)
        btn.clicked.connect(self._browse)
        layout.addWidget(btn)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def path(self) -> str:
        """Return the current directory path, or "" if nothing is selected."""
        return self._line.text()

    def set_path(self, p: str) -> None:
        """Set path programmatically. Does not emit path_changed."""
        self._line.setText(p)

    def clear(self) -> None:
        """Reset to empty. Does not emit path_changed."""
        self._line.clear()

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select folder",
            self._line.text() or "",
        )
        if path:
            self._line.setText(path)
            self.path_changed.emit(path)
