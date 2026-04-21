"""
HttpThread — lightweight QThread wrapper for non-blocking HTTP calls.

Every blocking HTTP call that originates from a QTimer runs through one of
these threads so the Qt main-thread event loop is never frozen while
waiting for a network response.

Usage
-----
    def _request_refresh(self) -> None:
        # Skip if the previous call is still in-flight.
        if self._thread is not None and self._thread.isRunning():
            return
        self._thread = HttpThread(self._client.list_workers, parent=self)
        self._thread.result_ready.connect(self._on_result)
        self._thread.error_raised.connect(self._on_error)
        self._thread.start()

Lifetime
--------
The caller keeps a reference to the thread (``self._thread``).  The Python
reference prevents the object from being garbage-collected while running.
When the next poll fires, a finished thread is simply overwritten with a new
one.  No ``deleteLater`` is used — Qt ownership via ``parent`` is sufficient
to clean up when the parent widget is destroyed.
"""
from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtCore import QThread, pyqtSignal


class HttpThread(QThread):
    """
    Run a single no-argument callable in a background OS thread.

    Signals
    -------
    result_ready(object)
        Emitted on the *main* thread with the callable's return value.
    error_raised(object)
        Emitted on the *main* thread with the exception if the callable raises.
    """

    result_ready = pyqtSignal(object)
    error_raised  = pyqtSignal(object)

    def __init__(self, fn: Callable[[], Any], parent=None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:                          # executed in background thread
        try:
            result = self._fn()
            self.result_ready.emit(result)
        except Exception as exc:                    # noqa: BLE001
            self.error_raised.emit(exc)
