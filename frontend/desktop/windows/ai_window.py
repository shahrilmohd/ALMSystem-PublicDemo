"""
AIWindow — AI Assistant tab for the ALM desktop application.

Layout
------
  QWidget
    │
    ├── Settings group
    │     Provider:         QComboBox  [anthropic | openai_compatible]
    │     Model:            _ModelSelector  (combo with known models + Custom… option)
    │     Base URL:         _BaseUrlSelector  (combo with known endpoints + Custom… option)
    │     Deployment mode:  QComboBox  [development | production]
    │
    ├── External LLM group (optional — collapsed by default)
    │     Enabled checkbox  QCheckBox  (enables the group controls)
    │     Provider:         QComboBox  [anthropic | openai_compatible]
    │     Model:            _ModelSelector
    │     Base URL:         _BaseUrlSelector  (shown for openai_compatible)
    │
    │   When enabled, this config is passed to RegulatoryResearchAgent only.
    │   The external LLM never receives firm-specific data (content gate enforced
    │   by ALMOrchestrator).  When disabled, regulatory_research routing is
    │   unavailable and the agent will not be invoked.
    │
    ├── Context group
    │     Context run:      QComboBox  (list of completed runs; refreshed on tab focus)
    │
    ├── Chat history        QTextEdit (read-only, markdown-rendered)
    │
    ├── Input row
    │     QLineEdit  [type your question]   QPushButton [Send | Stop]
    │
    └── Status label        "AI Assistant ready" / "Thinking…" / error text

Stop button
-----------
  While the AI is thinking the Send button is replaced by a Stop button.
  Clicking Stop cancels the pending response (the HTTP thread runs to
  completion in the background but its result is silently discarded).

Agent trace
-----------
  Each assistant reply is prefixed with the routing trace returned by the
  orchestrator, e.g. "router → bpa", shown in small grey italic text so
  actuaries can see which specialist handled the request.

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
    QCheckBox,
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
# Known model lists  (used by _ModelSelector)
# ---------------------------------------------------------------------------

_ANTHROPIC_MODELS = [
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-haiku-4-5",
]

# Models served via any OpenAI-compatible endpoint
_OPENAI_COMPAT_MODELS = [
    # OpenAI
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o1-preview",
    # Google
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.0-pro",
    "gemini-2.0-flash-thinking-exp",
    # DeepSeek
    "deepseek-chat",
    "deepseek-reasoner",
    # Local Llama (Ollama)
    "llama3.2",
]

_MODELS_BY_PROVIDER: dict[str, list[str]] = {
    "anthropic":        _ANTHROPIC_MODELS,
    "openai_compatible": _OPENAI_COMPAT_MODELS,
}

# Known base URLs for openai_compatible provider
# Each entry: (display label, url)
_KNOWN_BASE_URLS: list[tuple[str, str]] = [
    ("OpenAI API",              "https://api.openai.com/v1"),
    ("Google Gemini",           "https://generativelanguage.googleapis.com/v1beta/openai"),
    ("DeepSeek API",            "https://api.deepseek.com/v1"),
    ("Local Ollama",            "http://localhost:11434/v1"),
    ("Azure OpenAI (edit URL)", "https://YOUR_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT/"),
]

_CUSTOM_LABEL = "Custom…"


# ---------------------------------------------------------------------------
# _ModelSelector  — QComboBox of known models + a custom entry field
# ---------------------------------------------------------------------------

class _ModelSelector(QWidget):
    """
    Inline model picker: dropdown of known models for the chosen provider,
    plus a "Custom…" option that reveals a QLineEdit for free-text entry.
    """

    def __init__(self, default_model: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._custom_edit = QLineEdit()
        self._custom_edit.setPlaceholderText("Enter model ID…")
        self._custom_edit.setVisible(False)
        self._custom_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout.addWidget(self._combo)
        layout.addWidget(self._custom_edit)

        self._combo.currentTextChanged.connect(self._on_combo_changed)

        self.set_provider("anthropic")
        self.set_model(default_model)

    # ------------------------------------------------------------------

    def set_provider(self, provider: str) -> None:
        """Repopulate the combo with models appropriate for *provider*."""
        models = _MODELS_BY_PROVIDER.get(provider, _ANTHROPIC_MODELS)
        current = self.model_id()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(models)
        self._combo.addItem(_CUSTOM_LABEL)
        # Restore selection when the model is still available in the new list.
        idx = self._combo.findText(current)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
            self._custom_edit.setVisible(False)
        else:
            # Current model not in the new provider's list — switch to custom.
            custom_idx = self._combo.findText(_CUSTOM_LABEL)
            if custom_idx >= 0:
                self._combo.setCurrentIndex(custom_idx)
            self._custom_edit.setText(current)
            self._custom_edit.setVisible(bool(current))
        self._combo.blockSignals(False)

    def set_model(self, model: str) -> None:
        idx = self._combo.findText(model)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
            self._custom_edit.setVisible(False)
        else:
            custom_idx = self._combo.findText(_CUSTOM_LABEL)
            if custom_idx >= 0:
                self._combo.setCurrentIndex(custom_idx)
            self._custom_edit.setText(model)
            self._custom_edit.setVisible(True)

    def model_id(self) -> str:
        if self._combo.currentText() == _CUSTOM_LABEL:
            return self._custom_edit.text().strip()
        return self._combo.currentText()

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        self._combo.setEnabled(enabled)
        self._custom_edit.setEnabled(enabled)

    # ------------------------------------------------------------------

    def _on_combo_changed(self, text: str) -> None:
        self._custom_edit.setVisible(text == _CUSTOM_LABEL)


# ---------------------------------------------------------------------------
# _BaseUrlSelector  — dropdown of known endpoints + custom entry
# ---------------------------------------------------------------------------

class _BaseUrlSelector(QWidget):
    """
    Inline base-URL picker: labeled dropdown of known API endpoints,
    plus a "Custom…" option that reveals a QLineEdit for free-text entry.
    Only shown when provider is "openai_compatible".
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._combo.setToolTip("Select a known API endpoint or choose Custom… to type a URL.")

        # Blank "select" placeholder
        self._combo.addItem("— select endpoint —", "")
        for label, url in _KNOWN_BASE_URLS:
            self._combo.addItem(label, url)
        self._combo.addItem(_CUSTOM_LABEL, "__custom__")

        self._custom_edit = QLineEdit()
        self._custom_edit.setPlaceholderText("https://your-llm.internal/v1")
        self._custom_edit.setVisible(False)
        self._custom_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout.addWidget(self._combo)
        layout.addWidget(self._custom_edit)

        self._combo.currentIndexChanged.connect(self._on_combo_changed)

    # ------------------------------------------------------------------

    def url(self) -> str:
        data = self._combo.currentData()
        if data == "__custom__":
            return self._custom_edit.text().strip()
        return data or ""

    def set_url(self, url: str) -> None:
        # Try to find the URL in the known list
        for i in range(self._combo.count()):
            if self._combo.itemData(i) == url:
                self._combo.setCurrentIndex(i)
                return
        # Fallback to custom
        custom_idx = self._combo.findData("__custom__")
        if custom_idx >= 0:
            self._combo.setCurrentIndex(custom_idx)
        self._custom_edit.setText(url)
        self._custom_edit.setVisible(True)

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        self._combo.setEnabled(enabled)
        self._custom_edit.setEnabled(enabled)

    # ------------------------------------------------------------------

    def _on_combo_changed(self, _idx: int) -> None:
        data = self._combo.currentData()
        self._custom_edit.setVisible(data == "__custom__")


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
        self._client        = client
        self._session_id:    Optional[str]    = None
        self._chat_thread:   Optional[HttpThread] = None
        self._pending_submit: Optional[dict]  = None   # held while dialog is open
        self._cancelled:     bool             = False  # set by Stop button
        self._first_message: bool             = True   # tracks whether chat is empty

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

        self._model_selector = _ModelSelector("claude-opus-4-6")
        settings_form.addRow("Model:", self._model_selector)

        self._base_url_selector = _BaseUrlSelector()
        self._base_url_label    = QLabel("Base URL:")
        settings_form.addRow(self._base_url_label, self._base_url_selector)
        # Hide base_url row by default (only for openai_compatible)
        self._base_url_label.setVisible(False)
        self._base_url_selector.setVisible(False)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["development", "production"])
        settings_form.addRow("Deployment mode:", self._mode_combo)

        layout.addWidget(settings_group)

        # ---------------------------------------------------------------
        # External LLM group (optional — for RegulatoryResearchAgent only)
        # ---------------------------------------------------------------
        ext_group        = QGroupBox("External LLM (Regulatory Research)")
        ext_group_layout = QVBoxLayout(ext_group)
        ext_group_layout.setSpacing(4)

        self._ext_enabled_check = QCheckBox("Enable external LLM for regulatory questions")
        self._ext_enabled_check.setChecked(False)
        self._ext_enabled_check.toggled.connect(self._on_ext_enabled_toggled)
        ext_group_layout.addWidget(self._ext_enabled_check)

        ext_form = QFormLayout()
        ext_form.setContentsMargins(0, 4, 0, 0)

        self._ext_provider_combo = QComboBox()
        self._ext_provider_combo.addItems(["anthropic", "openai_compatible"])
        self._ext_provider_combo.currentTextChanged.connect(self._on_ext_provider_changed)
        self._ext_provider_combo.setEnabled(False)
        ext_form.addRow("Provider:", self._ext_provider_combo)

        self._ext_model_selector = _ModelSelector("claude-sonnet-4-6")
        self._ext_model_selector.setEnabled(False)
        ext_form.addRow("Model:", self._ext_model_selector)

        self._ext_base_url_selector = _BaseUrlSelector()
        self._ext_base_url_selector.setEnabled(False)
        self._ext_base_url_label = QLabel("Base URL:")
        self._ext_base_url_label.setVisible(False)
        self._ext_base_url_selector.setVisible(False)
        ext_form.addRow(self._ext_base_url_label, self._ext_base_url_selector)

        ext_group_layout.addLayout(ext_form)

        note_label = QLabel(
            "<i style='color:#666; font-size:10px;'>"
            "When enabled, regulatory questions route to this LLM. "
            "No firm-specific data is ever sent to this endpoint."
            "</i>"
        )
        note_label.setTextFormat(Qt.TextFormat.RichText)
        note_label.setWordWrap(True)
        ext_group_layout.addWidget(note_label)

        layout.addWidget(ext_group)

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
        self._input_edit.setPlaceholderText("Ask about model, assumptions table, model points, results or request a config change…")
        self._input_edit.returnPressed.connect(self._send_message)
        input_row.addWidget(self._input_edit)

        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(70)
        self._send_btn.clicked.connect(self._send_message)
        input_row.addWidget(self._send_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedWidth(70)
        self._stop_btn.setVisible(False)
        self._stop_btn.setStyleSheet("QPushButton { color: #c62828; font-weight: bold; }")
        self._stop_btn.clicked.connect(self._cancel_request)
        input_row.addWidget(self._stop_btn)

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
        self._base_url_selector.setVisible(show_url)
        self._model_selector.set_provider(provider)

    def _on_ext_enabled_toggled(self, enabled: bool) -> None:
        self._ext_provider_combo.setEnabled(enabled)
        self._ext_model_selector.setEnabled(enabled)
        self._on_ext_provider_changed(self._ext_provider_combo.currentText())

    def _on_ext_provider_changed(self, provider: str) -> None:
        enabled  = self._ext_enabled_check.isChecked()
        show_url = enabled and provider == "openai_compatible"
        self._ext_base_url_label.setVisible(show_url)
        self._ext_base_url_selector.setVisible(show_url)
        self._ext_base_url_selector.setEnabled(show_url)
        self._ext_model_selector.set_provider(provider)

    def _current_settings(self) -> dict:
        """Return current GUI settings as a dict for the API request body."""
        settings: dict = {
            "provider":        self._provider_combo.currentText(),
            "model":           self._model_selector.model_id() or "claude-opus-4-6",
            "deployment_mode": self._mode_combo.currentText(),
        }
        base_url = self._base_url_selector.url()
        if base_url:
            settings["base_url"] = base_url

        # External LLM settings — only included when the checkbox is ticked.
        if self._ext_enabled_check.isChecked():
            settings["external_provider"] = self._ext_provider_combo.currentText()
            settings["external_model"]    = (
                self._ext_model_selector.model_id() or "claude-sonnet-4-6"
            )
            ext_base_url = self._ext_base_url_selector.url()
            if ext_base_url:
                settings["external_base_url"] = ext_base_url

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

        self._cancelled = False
        self._input_edit.clear()
        self._append_user(message)
        self._set_thinking(True)

        settings   = self._current_settings()
        session_id = self._session_id
        run_id     = self._context_run_id()

        def _call():
            return self._client.ai_chat(
                message,
                session_id=        session_id,
                context_run_id=    run_id,
                provider=          settings["provider"],
                model=             settings["model"],
                base_url=          settings.get("base_url"),
                deployment_mode=   settings["deployment_mode"],
                external_provider= settings.get("external_provider"),
                external_model=    settings.get("external_model"),
                external_base_url= settings.get("external_base_url"),
            )

        self._chat_thread = HttpThread(_call, parent=self)
        self._chat_thread.result_ready.connect(self._on_chat_response)
        self._chat_thread.error_raised.connect(self._on_chat_error)
        self._chat_thread.start()

    # -----------------------------------------------------------------------
    # Stop / cancel
    # -----------------------------------------------------------------------

    def _cancel_request(self) -> None:
        """Discard the in-flight response and restore the input row."""
        self._cancelled = True
        self._set_thinking(False)
        self._append_system("Request cancelled — you can amend and resend.")

    # -----------------------------------------------------------------------
    # Response handling
    # -----------------------------------------------------------------------

    def _on_chat_response(self, data: dict) -> None:
        if self._cancelled:
            return
        self._set_thinking(False)
        self._session_id = data.get("session_id", self._session_id)

        reply       = data.get("reply", "")
        agent_used  = data.get("agent_used", "")
        agent_trace = data.get("agent_trace", [])
        self._append_assistant(reply, agent_used, agent_trace)

        pending = data.get("pending_submit")
        if pending:
            self._pending_submit = pending
            # Small delay so the chat reply renders before the dialog opens.
            QTimer.singleShot(200, self._show_confirm_dialog)

    def _on_chat_error(self, exc: Exception) -> None:
        if self._cancelled:
            return
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
        self._session_id    = None
        self._first_message = True
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
        # Add a visual gap before every user message except the very first.
        separator = "" if self._first_message else '<p style="margin:0; line-height:0.4;">&nbsp;</p>'
        self._first_message = False
        self._chat_display.append(
            f'{separator}'
            f'<p style="margin-top:4px;"><b style="color:#1565c0;">You:</b> {self._escape(text)}</p>'
        )

    def _append_assistant(self, text: str, agent: str, trace: list[str] | None = None) -> None:
        label = {
            # Phase 2 agents
            "analyst":              "Run Analyst",
            "advisor":              "Config Advisor",
            "orchestrator":         "AI Assistant",
            # Phase 3 specialist agents
            "ifrs17":               "IFRS 17 Specialist",
            "bpa":                  "BPA Specialist",
            "solvency2":            "Solvency II Specialist",
            "data_review":          "Data Review",
            "architect":            "Architect",
            "regulatory_research":  "Regulatory Research",
        }.get(agent, "AI Assistant")

        # Agent trace line — only shown when there are at least two hops
        trace_html = ""
        if trace and len(trace) >= 2:
            trace_str = " → ".join(trace)
            trace_html = (
                f'<span style="color:#999; font-size:10px; font-style:italic;">'
                f'via {self._escape(trace_str)}</span><br>'
            )

        html = self._markdown_to_html(text)
        self._chat_display.append(
            f'<p><b style="color:#2e7d32;">{label}:</b> {trace_html}{html}</p>'
        )

    def _append_system(self, text: str, error: bool = False) -> None:
        colour = "#c62828" if error else "#555555"
        self._chat_display.append(
            f'<p><i style="color:{colour};">{self._escape(text)}</i></p>'
        )

    def _set_thinking(self, thinking: bool) -> None:
        self._send_btn.setVisible(not thinking)
        self._stop_btn.setVisible(thinking)
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
