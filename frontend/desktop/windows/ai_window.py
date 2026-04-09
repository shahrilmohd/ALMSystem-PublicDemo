"""
AIWindow — AI Assistant tab for the ALM desktop application.

Layout
------
  QWidget
    │
    ├── Settings group (collapsible)
    │     Provider:         QComboBox  [anthropic | openai_compatible]
    │     Model:            QLineEdit  (e.g. "claude-opus-4-6")
    │     Base URL:         QLineEdit  (required for openai_compatible; hidden otherwise)
    │     Deployment mode:  QComboBox  [development | production]
    │
    ├── Context group
    │     Context run:      QComboBox  (list of completed runs; refreshed on tab focus)
    │
    ├── Chat history        QTextEdit (read-only, markdown-rendered)
    │
    ├── Input row
    │     QLineEdit  [type your question]   QPushButton [Send]
    │
    └── Status label        "AI Assistant ready" / "Thinking…" / error text

Confirmation dialog (shown when pending_submit is set in the response)
----------------------------------------------------------------------
  QDialog
    ├── Proposed config preview  (QTextEdit, read-only, JSON)
    ├── Reviewer verdict         (QLabel with colour coding)
    ├── Issues / suggestions     (QLabel)
    └── Buttons: [Approve and Submit]   [Reject]

If the API key environment variable is absent the tab is disabled with a
message explaining what to set.  The check is done by calling POST /ai/chat
on the first message; 503 → disable with explanation.

All HTTP calls run through HttpThread so the Qt event loop is never blocked.
"""
from __future__ import annotations

import json
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from frontend.desktop.api_client import ALMApiClient, APIError, APIResponseError
from frontend.desktop.http_thread import HttpThread


# ---------------------------------------------------------------------------
# Confirmation dialog
# ---------------------------------------------------------------------------

class _ConfirmSubmitDialog(QDialog):
    """
    Shows the proposed RunConfig and ReviewerAgent verdict before submission.
    The actuary must click Approve for submit_run to be called.
    """

    def __init__(
        self,
        config_json: str,
        reviewer: dict,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review Proposed Run Configuration")
        self.setMinimumWidth(640)
        self.setMinimumHeight(480)

        layout = QVBoxLayout(self)

        # Reviewer verdict banner
        verdict = reviewer.get("verdict", "unknown")
        summary = reviewer.get("summary", "")
        colour  = {"approved": "#2e7d32", "needs_revision": "#e65100"}.get(verdict, "#c62828")
        verdict_label = QLabel(f'<b>Reviewer verdict: <span style="color:{colour};">{verdict}</span></b>')
        verdict_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(verdict_label)

        if summary:
            layout.addWidget(QLabel(summary))

        issues = reviewer.get("issues", [])
        if issues:
            layout.addWidget(QLabel("<b>Issues:</b>"))
            for issue in issues:
                layout.addWidget(QLabel(f"  • {issue}"))

        suggestions = reviewer.get("suggestions", [])
        if suggestions:
            layout.addWidget(QLabel("<b>Suggestions:</b>"))
            for s in suggestions:
                layout.addWidget(QLabel(f"  • {s}"))

        layout.addWidget(QLabel("<b>Proposed RunConfig:</b>"))

        # Config preview — pretty-printed JSON
        config_preview = QTextEdit()
        config_preview.setReadOnly(True)
        config_preview.setFont(QFont("Courier New", 9))
        try:
            pretty = json.dumps(json.loads(config_json), indent=2)
        except (json.JSONDecodeError, ValueError):
            pretty = config_json
        config_preview.setPlainText(pretty)
        layout.addWidget(config_preview)

        # Buttons
        buttons = QDialogButtonBox()
        self._approve_btn = buttons.addButton("Approve and Submit", QDialogButtonBox.ButtonRole.AcceptRole)
        self._reject_btn  = buttons.addButton("Reject",             QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


# ---------------------------------------------------------------------------
# Main AI window
# ---------------------------------------------------------------------------

class AIWindow(QWidget):
    """
    AI Assistant tab.

    Parameters
    ----------
    client : ALMApiClient
        Shared HTTP client.
    parent : QWidget | None
    """

    def __init__(self, client: ALMApiClient, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client     = client
        self._session_id: Optional[str] = None
        self._chat_thread: Optional[HttpThread] = None
        self._pending_submit: Optional[dict] = None   # held while dialog is open

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ---------------------------------------------------------------
        # Settings group
        # ---------------------------------------------------------------
        settings_group = QGroupBox("AI Settings")
        settings_form  = QFormLayout(settings_group)

        self._provider_combo = QComboBox()
        self._provider_combo.addItems(["anthropic", "openai_compatible"])
        self._provider_combo.currentTextChanged.connect(self._on_provider_changed)
        settings_form.addRow("Provider:", self._provider_combo)

        self._model_edit = QLineEdit("claude-opus-4-6")
        settings_form.addRow("Model:", self._model_edit)

        self._base_url_edit = QLineEdit()
        self._base_url_edit.setPlaceholderText("https://your-llm.internal/v1")
        self._base_url_label = QLabel("Base URL:")
        settings_form.addRow(self._base_url_label, self._base_url_edit)
        # Hide base_url row by default (only for openai_compatible)
        self._base_url_label.setVisible(False)
        self._base_url_edit.setVisible(False)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["development", "production"])
        settings_form.addRow("Deployment mode:", self._mode_combo)

        layout.addWidget(settings_group)

        # ---------------------------------------------------------------
        # Context run selector
        # ---------------------------------------------------------------
        context_group  = QGroupBox("Context")
        context_layout = QHBoxLayout(context_group)
        context_layout.addWidget(QLabel("Context run:"))
        self._run_combo = QComboBox()
        self._run_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._run_combo.addItem("(none)", None)
        context_layout.addWidget(self._run_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(70)
        refresh_btn.clicked.connect(self.refresh_run_list)
        context_layout.addWidget(refresh_btn)

        new_conv_btn = QPushButton("New Conversation")
        new_conv_btn.clicked.connect(self._new_conversation)
        context_layout.addWidget(new_conv_btn)

        layout.addWidget(context_group)

        # ---------------------------------------------------------------
        # Chat history
        # ---------------------------------------------------------------
        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setAcceptRichText(True)
        self._chat_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._chat_display, stretch=1)

        # ---------------------------------------------------------------
        # Input row
        # ---------------------------------------------------------------
        input_row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Ask about results or request a config change…")
        self._input_edit.returnPressed.connect(self._send_message)
        input_row.addWidget(self._input_edit)

        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(70)
        self._send_btn.clicked.connect(self._send_message)
        input_row.addWidget(self._send_btn)

        layout.addLayout(input_row)

        # ---------------------------------------------------------------
        # Status label
        # ---------------------------------------------------------------
        self._status_label = QLabel("AI Assistant ready.")
        self._status_label.setStyleSheet("color: grey; font-size: 11px;")
        layout.addWidget(self._status_label)

    # -----------------------------------------------------------------------
    # Public — called by MainWindow when the tab is focused
    # -----------------------------------------------------------------------

    def refresh_run_list(self) -> None:
        """Reload the context run dropdown from the API (non-blocking)."""
        thread = HttpThread(self._client.list_runs, parent=self)
        thread.result_ready.connect(self._on_runs_loaded)
        thread.error_raised.connect(lambda _: None)   # silent on error
        thread.start()
        self._run_list_thread = thread   # keep reference

    # -----------------------------------------------------------------------
    # Settings
    # -----------------------------------------------------------------------

    def _on_provider_changed(self, provider: str) -> None:
        show_url = provider == "openai_compatible"
        self._base_url_label.setVisible(show_url)
        self._base_url_edit.setVisible(show_url)

    def _current_settings(self) -> dict:
        """Return current GUI settings as a dict for the API request body."""
        settings: dict = {
            "provider":        self._provider_combo.currentText(),
            "model":           self._model_edit.text().strip() or "claude-opus-4-6",
            "deployment_mode": self._mode_combo.currentText(),
        }
        base_url = self._base_url_edit.text().strip()
        if base_url:
            settings["base_url"] = base_url
        return settings

    def _context_run_id(self) -> Optional[str]:
        return self._run_combo.currentData()

    # -----------------------------------------------------------------------
    # Send message
    # -----------------------------------------------------------------------

    def _send_message(self) -> None:
        message = self._input_edit.text().strip()
        if not message:
            return
        if self._chat_thread is not None and self._chat_thread.isRunning():
            return   # previous request still in flight

        self._input_edit.clear()
        self._append_user(message)
        self._set_thinking(True)

        settings   = self._current_settings()
        session_id = self._session_id
        run_id     = self._context_run_id()

        def _call():
            return self._client.ai_chat(
                message,
                session_id=      session_id,
                context_run_id=  run_id,
                provider=        settings["provider"],
                model=           settings["model"],
                base_url=        settings.get("base_url"),
                deployment_mode= settings["deployment_mode"],
            )

        self._chat_thread = HttpThread(_call, parent=self)
        self._chat_thread.result_ready.connect(self._on_chat_response)
        self._chat_thread.error_raised.connect(self._on_chat_error)
        self._chat_thread.start()

    # -----------------------------------------------------------------------
    # Response handling
    # -----------------------------------------------------------------------

    def _on_chat_response(self, data: dict) -> None:
        self._set_thinking(False)
        self._session_id = data.get("session_id", self._session_id)

        reply      = data.get("reply", "")
        agent_used = data.get("agent_used", "")
        self._append_assistant(reply, agent_used)

        pending = data.get("pending_submit")
        if pending:
            self._pending_submit = pending
            # Small delay so the chat reply renders before the dialog opens.
            QTimer.singleShot(200, self._show_confirm_dialog)

    def _on_chat_error(self, exc: Exception) -> None:
        self._set_thinking(False)
        if isinstance(exc, APIResponseError) and exc.status_code == 503:
            msg = (
                "AI Assistant is unavailable — the API key is not configured on the server. "
                "Add ANTHROPIC_API_KEY (or LLM_API_KEY) to the server's .env file and restart."
            )
            self._append_system(msg, error=True)
            self._send_btn.setEnabled(False)
            self._input_edit.setEnabled(False)
            self._status_label.setText("AI Assistant disabled — API key not set.")
        else:
            self._append_system(f"Error: {exc}", error=True)
            self._status_label.setText("Error communicating with AI assistant.")

    # -----------------------------------------------------------------------
    # Confirmation dialog
    # -----------------------------------------------------------------------

    def _show_confirm_dialog(self) -> None:
        if self._pending_submit is None:
            return

        config_json = self._pending_submit.get("config_json", "")
        reviewer    = self._pending_submit.get("reviewer", {})

        dlg = _ConfirmSubmitDialog(config_json, reviewer, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._submit_approved_config(config_json)
        else:
            self._append_system("Run submission rejected by actuary.")
        self._pending_submit = None

    def _submit_approved_config(self, config_json: str) -> None:
        """Submit the approved config via POST /runs."""
        self._status_label.setText("Submitting run…")

        def _call():
            return self._client.submit_run(config_json)

        thread = HttpThread(_call, parent=self)
        thread.result_ready.connect(self._on_submit_done)
        thread.error_raised.connect(self._on_submit_error)
        thread.start()
        self._submit_thread = thread

    def _on_submit_done(self, run_status) -> None:
        run_id = run_status.run_id
        self._append_system(
            f"Run submitted successfully. Run ID: <b>{run_id}</b>  Status: {run_status.status}"
        )
        self._status_label.setText(f"Run submitted: {run_id}")
        self.refresh_run_list()

    def _on_submit_error(self, exc: Exception) -> None:
        self._append_system(f"Submission failed: {exc}", error=True)
        self._status_label.setText("Submission failed.")

    # -----------------------------------------------------------------------
    # New conversation
    # -----------------------------------------------------------------------

    def _new_conversation(self) -> None:
        if self._session_id:
            # Best-effort clear on the server; ignore failures.
            try:
                self._client.ai_clear_session(self._session_id)
            except Exception:  # noqa: BLE001
                pass
        self._session_id = None
        self._chat_display.clear()
        self._append_system("New conversation started.")
        self._status_label.setText("AI Assistant ready.")
        self._send_btn.setEnabled(True)
        self._input_edit.setEnabled(True)

    # -----------------------------------------------------------------------
    # Run list
    # -----------------------------------------------------------------------

    def _on_runs_loaded(self, runs) -> None:
        current = self._run_combo.currentData()
        self._run_combo.blockSignals(True)
        self._run_combo.clear()
        self._run_combo.addItem("(none)", None)
        for run in runs:
            label = run.run_name or run.run_id
            self._run_combo.addItem(f"{label}  [{run.status}]", run.run_id)
        # Restore previous selection if still present
        idx = self._run_combo.findData(current)
        if idx >= 0:
            self._run_combo.setCurrentIndex(idx)
        self._run_combo.blockSignals(False)

    # -----------------------------------------------------------------------
    # Chat display helpers
    # -----------------------------------------------------------------------

    def _append_user(self, text: str) -> None:
        self._chat_display.append(
            f'<p><b style="color:#1565c0;">You:</b> {self._escape(text)}</p>'
        )

    def _append_assistant(self, text: str, agent: str) -> None:
        label = {
            "analyst":      "Run Analyst",
            "advisor":      "Config Advisor",
            "orchestrator": "AI Assistant",
        }.get(agent, "AI Assistant")
        # Convert simple markdown-like formatting to HTML
        html = self._markdown_to_html(text)
        self._chat_display.append(
            f'<p><b style="color:#2e7d32;">{label}:</b><br>{html}</p>'
        )

    def _append_system(self, text: str, error: bool = False) -> None:
        colour = "#c62828" if error else "#555555"
        self._chat_display.append(
            f'<p><i style="color:{colour};">{self._escape(text)}</i></p>'
        )

    def _set_thinking(self, thinking: bool) -> None:
        self._send_btn.setEnabled(not thinking)
        self._input_edit.setEnabled(not thinking)
        self._status_label.setText("Thinking…" if thinking else "AI Assistant ready.")

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
        )

    @staticmethod
    def _markdown_to_html(text: str) -> str:
        """Convert a small subset of markdown to HTML for the chat display."""
        import re
        # Bold **text**
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        # Inline code `text`
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        # Horizontal rule ---
        text = re.sub(r"\n---+\n", "\n<hr>\n", text)
        # Bullet points
        text = re.sub(r"^- (.+)$", r"• \1", text, flags=re.MULTILINE)
        # Newlines
        text = text.replace("\n", "<br>")
        return text
