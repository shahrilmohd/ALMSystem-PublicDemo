"""
ALM Desktop Application — entry point.

Usage
-----
    uv run python -m frontend.desktop.app
    uv run python -m frontend.desktop.app --api-url http://localhost:8000

Arguments
---------
--api-url   Base URL of the FastAPI backend.  Default: http://localhost:8000
--timeout   HTTP request timeout in seconds.  Default: 10
"""
from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from frontend.desktop.api_client import ALMApiClient
from frontend.desktop.windows.main_window import MainWindow


def main() -> None:
    parser = argparse.ArgumentParser(description="ALM Desktop Application")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the FastAPI backend (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP request timeout in seconds (default: 10)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("ALM System")
    app.setOrganizationName("Actuarial Team")

    client = ALMApiClient(base_url=args.api_url, timeout=args.timeout)

    window = MainWindow(client)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
