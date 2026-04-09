"""
Root conftest.py — Windows compatibility patch for RQ.

Problem
-------
rq/scheduler.py executes ``ForkProcess = get_context('fork').Process`` at
module import time.  On Windows, the 'fork' start method does not exist
(only 'spawn' is available), so the import raises:

    ValueError: cannot find context for 'fork'

This crashes any test that directly or indirectly imports ``rq``.

Fix
---
Before any test module is collected, we replace ``multiprocessing.get_context``
with a wrapper that returns a mock when 'fork' is requested on Windows.
ForkProcess is only *used* by the RQ scheduler when it actually forks a
worker process — which never happens in unit tests (we use fakeredis and
synchronous queues).  Substituting a MagicMock is therefore safe for all
test scenarios in this project.
"""
from __future__ import annotations

import platform
import sys

if platform.system() == "Windows" and "multiprocessing" not in sys.modules:
    # Pre-import multiprocessing so the patch is applied to the module object
    import multiprocessing  # noqa: F401

if platform.system() == "Windows":
    import multiprocessing
    from unittest.mock import MagicMock

    _real_get_context = multiprocessing.get_context

    def _windows_safe_get_context(method=None):
        """Return a mock context for 'fork' on Windows; delegate all others."""
        if method == "fork":
            ctx = MagicMock()
            ctx.Process = MagicMock
            return ctx
        return _real_get_context(method)

    multiprocessing.get_context = _windows_safe_get_context
