"""
RunTypeSelector — radio-button widget for selecting the projection run type.

Emits `run_type_changed(str)` whenever the selection changes.
The emitted string matches the RunType enum values:
    "liability_only"  | "deterministic" | "stochastic"

The parent widget (RunConfigWindow) listens to this signal to show/hide
the asset file picker and ESG scenario file picker as appropriate.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QButtonGroup, QGroupBox, QHBoxLayout, QRadioButton


class RunTypeSelector(QGroupBox):
    """
    Radio-button group for choosing between the three run types.

    Signals
    -------
    run_type_changed(str)
        Emitted when the user changes the selection.
        Value is the run type string: "liability_only", "deterministic",
        or "stochastic".
    """

    run_type_changed = pyqtSignal(str)

    _RUN_TYPES = [
        ("Liability Only", "liability_only"),
        ("Deterministic",  "deterministic"),
        ("Stochastic",     "stochastic"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__("Run Type", parent)

        layout = QHBoxLayout()
        self._button_group = QButtonGroup(self)

        for label, value in self._RUN_TYPES:
            btn = QRadioButton(label)
            btn.setProperty("run_type_value", value)
            self._button_group.addButton(btn)
            layout.addWidget(btn)

        # Default: Deterministic selected
        self._set_checked("deterministic")

        self._button_group.buttonClicked.connect(self._on_button_clicked)
        self.setLayout(layout)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def selected_run_type(self) -> str:
        """Return the currently selected run type string."""
        checked = self._button_group.checkedButton()
        if checked is None:
            return "deterministic"
        return checked.property("run_type_value")

    def set_run_type(self, run_type: str) -> None:
        """Programmatically select a run type without emitting the signal."""
        self._button_group.blockSignals(True)
        self._set_checked(run_type)
        self._button_group.blockSignals(False)

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _set_checked(self, run_type: str) -> None:
        for btn in self._button_group.buttons():
            if btn.property("run_type_value") == run_type:
                btn.setChecked(True)
                return

    def _on_button_clicked(self, btn: QRadioButton) -> None:
        self.run_type_changed.emit(btn.property("run_type_value"))
