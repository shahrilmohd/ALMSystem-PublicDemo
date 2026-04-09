"""
RunConfigWindow — full run submission form.

Composes all input widgets into a scrollable form that builds a RunConfig-
compatible JSON string and submits it to the API.

Layout (top to bottom)
----------------------
  Run Name
  Run Type selector
  ── Input Sources ──
    Model Points    (QStackedWidget: File page | DB stub page)
    Assumptions Folder
    Fund Config (YAML)
    Asset Portfolio    (hidden for LIABILITY_ONLY)
    ESG Scenario File  (shown for STOCHASTIC only)
  ── Projection Settings ──
    Valuation date, term, timestep, currency
    Liability models (checkboxes), input mode (radio), n_scenarios
  ── Output Settings ──
    Output folder, output timestep, format
  ── Actions ──
    [Validate Config]  [Load YAML]  [Save YAML]
    [Submit Run]       [Submit as Batch — n: QSpinBox]
  ── Progress ──
    ProgressPanel (hidden until a run is submitted)

Architecture notes
------------------
- This widget never imports from engine/ or api/.
- build_config_dict() assembles a plain Python dict matching the RunConfig
  JSON schema.  The dict is serialised with json.dumps() before sending.
- Paths are stored as plain strings; the API / engine resolve them.
"""
from __future__ import annotations

import json
import uuid
from datetime import date

from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)
from PyQt6.QtCore import QDate

from frontend.desktop.api_client import ALMApiClient, APIError, APIResponseError
from frontend.desktop.components.file_picker import FilePicker, FolderPicker
from frontend.desktop.components.progress_panel import ProgressPanel
from frontend.desktop.components.run_type_selector import RunTypeSelector


class RunConfigWindow(QWidget):
    """
    Main run submission form.  Intended to be used as a tab in MainWindow.
    """

    def __init__(self, client: ALMApiClient, parent=None) -> None:
        super().__init__(parent)
        self._client = client

        # ---------------------------------------------------------------
        # Root: scroll area wrapping everything
        # ---------------------------------------------------------------
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self._form_layout = QVBoxLayout(inner)
        self._form_layout.setSpacing(12)

        scroll.setWidget(inner)
        root_layout.addWidget(scroll)

        # ---------------------------------------------------------------
        # Build sections
        # ---------------------------------------------------------------
        self._build_identity_section()
        self._build_input_sources_section()
        self._build_projection_section()
        self._build_output_section()
        self._build_actions_section()
        self._build_progress_section()

        self._form_layout.addStretch()

        # Initial visibility driven by default run type (deterministic).
        self._on_run_type_changed("deterministic")

    # ===================================================================
    # Section builders
    # ===================================================================

    def _build_identity_section(self) -> None:
        grp = QGroupBox("Run Identity")
        layout = QFormLayout()

        self._run_name_edit = QLineEdit()
        self._run_name_edit.setPlaceholderText("e.g. Q1 2026 BEL Validation")
        layout.addRow("Run name:", self._run_name_edit)

        self._run_type_selector = RunTypeSelector()
        self._run_type_selector.run_type_changed.connect(self._on_run_type_changed)
        layout.addRow("Run type:", self._run_type_selector)

        self._notes_edit = QLineEdit()
        self._notes_edit.setPlaceholderText("Optional free-text notes")
        layout.addRow("Notes:", self._notes_edit)

        grp.setLayout(layout)
        self._form_layout.addWidget(grp)

    def _build_input_sources_section(self) -> None:
        grp = QGroupBox("Input Sources")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # ── Model Points ────────────────────────────────────────────────
        mp_grp = QGroupBox("Model Points")
        mp_layout = QVBoxLayout()

        # Source type toggle
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self._mp_file_radio = QRadioButton("File")
        self._mp_db_radio   = QRadioButton("Database")
        self._mp_file_radio.setChecked(True)
        mp_src_group = QButtonGroup(self)
        mp_src_group.addButton(self._mp_file_radio)
        mp_src_group.addButton(self._mp_db_radio)
        src_row.addWidget(self._mp_file_radio)
        src_row.addWidget(self._mp_db_radio)
        src_row.addStretch()
        mp_layout.addLayout(src_row)

        # Stacked widget: page 0 = file picker, page 1 = DB stub
        self._mp_stack = QStackedWidget()

        # Page 0 — file picker
        file_page = QWidget()
        file_page_layout = QVBoxLayout(file_page)
        file_page_layout.setContentsMargins(0, 0, 0, 0)
        self._mp_file_picker = FilePicker(
            label="Model points file:",
            file_filter="CSV/Excel files (*.csv *.xlsx *.xls);;All files (*)",
        )
        file_page_layout.addWidget(self._mp_file_picker)
        self._mp_stack.addWidget(file_page)

        # Page 1 — database stub
        db_page = QWidget()
        db_page_layout = QVBoxLayout(db_page)
        db_page_layout.setContentsMargins(0, 0, 0, 0)
        stub_label = QLabel(
            "Database source — coming soon.\n"
            "Supports SQL Server, PostgreSQL, Oracle, SQLite via SQLAlchemy connection string."
        )
        stub_label.setStyleSheet("color: #888888; font-style: italic;")
        stub_label.setWordWrap(True)
        db_page_layout.addWidget(stub_label)
        self._mp_stack.addWidget(db_page)

        mp_src_group.buttonClicked.connect(
            lambda btn: self._mp_stack.setCurrentIndex(
                0 if btn is self._mp_file_radio else 1
            )
        )
        mp_layout.addWidget(self._mp_stack)
        mp_grp.setLayout(mp_layout)
        layout.addWidget(mp_grp)

        # ── Assumptions Folder ──────────────────────────────────────────
        assump_grp = QGroupBox("Assumptions Tables")
        assump_layout = QVBoxLayout()
        self._assump_folder_picker = FolderPicker(
            label="Tables folder:",
            placeholder="Select folder containing assumption CSV/Excel files",
        )
        assump_layout.addWidget(self._assump_folder_picker)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("File format:"))
        self._assump_fmt_csv   = QRadioButton("CSV")
        self._assump_fmt_excel = QRadioButton("Excel")
        self._assump_fmt_csv.setChecked(True)
        assump_fmt_grp = QButtonGroup(self)
        assump_fmt_grp.addButton(self._assump_fmt_csv)
        assump_fmt_grp.addButton(self._assump_fmt_excel)
        fmt_row.addWidget(self._assump_fmt_csv)
        fmt_row.addWidget(self._assump_fmt_excel)
        fmt_row.addStretch()
        assump_layout.addLayout(fmt_row)
        assump_grp.setLayout(assump_layout)
        layout.addWidget(assump_grp)

        # ── Fund Config (hidden for LIABILITY_ONLY) ──────────────────────
        self._fund_config_grp = QGroupBox("Fund Config (YAML)")
        fund_config_layout = QVBoxLayout()
        self._fund_config_picker = FilePicker(
            label="Fund config (YAML):",
            file_filter="YAML files (*.yaml *.yml);;All files (*)",
        )
        fund_config_layout.addWidget(self._fund_config_picker)
        self._fund_config_grp.setLayout(fund_config_layout)
        layout.addWidget(self._fund_config_grp)

        # ── Asset Portfolio (hidden for LIABILITY_ONLY) ──────────────────
        self._asset_grp = QGroupBox("Asset Portfolio")
        asset_layout = QVBoxLayout()
        self._asset_picker = FilePicker(
            label="Asset portfolio:",
            file_filter="CSV/Excel files (*.csv *.xlsx *.xls);;All files (*)",
        )
        asset_layout.addWidget(self._asset_picker)
        self._asset_grp.setLayout(asset_layout)
        layout.addWidget(self._asset_grp)

        # ── ESG Scenario File (shown for STOCHASTIC only) ────────────────
        self._scenario_grp = QGroupBox("ESG Scenario File")
        scenario_layout = QVBoxLayout()
        self._scenario_picker = FilePicker(
            label="Scenario file:",
            file_filter="CSV files (*.csv);;All files (*)",
        )
        scenario_layout.addWidget(self._scenario_picker)
        self._scenario_grp.setLayout(scenario_layout)
        layout.addWidget(self._scenario_grp)

        grp.setLayout(layout)
        self._form_layout.addWidget(grp)

    def _build_projection_section(self) -> None:
        grp = QGroupBox("Projection Settings")
        layout = QFormLayout()
        layout.setSpacing(8)

        # Valuation date
        self._val_date = QDateEdit()
        self._val_date.setDate(QDate.currentDate())
        self._val_date.setCalendarPopup(True)
        self._val_date.setDisplayFormat("yyyy-MM-dd")
        layout.addRow("Valuation date:", self._val_date)

        # Projection term
        self._proj_term = QSpinBox()
        self._proj_term.setRange(1, 100)
        self._proj_term.setValue(30)
        self._proj_term.setSuffix(" years")
        layout.addRow("Projection term:", self._proj_term)

        # Projection timestep
        self._proj_timestep = QComboBox()
        for label, value in [("Monthly", "monthly"), ("Quarterly", "quarterly"), ("Annual", "annual")]:
            self._proj_timestep.addItem(label, value)
        layout.addRow("Projection timestep:", self._proj_timestep)

        # Decision timestep
        self._decision_timestep = QComboBox()
        for label, value in [("Monthly", "monthly"), ("Quarterly", "quarterly"), ("Annual", "annual")]:
            self._decision_timestep.addItem(label, value)
        self._decision_timestep.setCurrentIndex(2)  # default annual
        layout.addRow("Decision timestep:", self._decision_timestep)

        # Currency
        self._currency = QComboBox()
        for label, value in [("GBP", "GBP"), ("EUR", "EUR"), ("USD", "USD")]:
            self._currency.addItem(label, value)
        layout.addRow("Currency:", self._currency)

        # Liability models
        liability_row = QHBoxLayout()
        self._chk_conventional = QCheckBox("Conventional")
        self._chk_unit_linked  = QCheckBox("Unit Linked")
        self._chk_annuity      = QCheckBox("Annuity")
        self._chk_conventional.setChecked(True)
        liability_row.addWidget(self._chk_conventional)
        liability_row.addWidget(self._chk_unit_linked)
        liability_row.addWidget(self._chk_annuity)
        liability_row.addStretch()
        layout.addRow("Liability models:", liability_row)

        # Input mode
        input_mode_row = QHBoxLayout()
        self._input_mode_group_mp  = QRadioButton("Group MP")
        self._input_mode_seriatim  = QRadioButton("Seriatim")
        self._input_mode_group_mp.setChecked(True)
        input_mode_grp = QButtonGroup(self)
        input_mode_grp.addButton(self._input_mode_group_mp)
        input_mode_grp.addButton(self._input_mode_seriatim)
        input_mode_row.addWidget(self._input_mode_group_mp)
        input_mode_row.addWidget(self._input_mode_seriatim)
        input_mode_row.addStretch()
        layout.addRow("Input mode:", input_mode_row)

        # n_scenarios (stochastic only)
        self._n_scenarios_label = QLabel("Number of scenarios:")
        self._n_scenarios = QSpinBox()
        self._n_scenarios.setRange(1, 10_000)
        self._n_scenarios.setValue(1_000)
        layout.addRow(self._n_scenarios_label, self._n_scenarios)

        grp.setLayout(layout)
        self._form_layout.addWidget(grp)

    def _build_output_section(self) -> None:
        grp = QGroupBox("Output Settings")
        layout = QFormLayout()
        layout.setSpacing(8)

        self._output_folder_picker = FolderPicker(
            label="Output folder:",
            placeholder="Where to write result files",
        )
        layout.addRow(self._output_folder_picker)

        self._output_timestep = QComboBox()
        for label, value in [("Monthly", "monthly"), ("Quarterly", "quarterly"), ("Annual", "annual")]:
            self._output_timestep.addItem(label, value)
        layout.addRow("Output timestep:", self._output_timestep)

        fmt_row = QHBoxLayout()
        self._fmt_csv    = QRadioButton("CSV")
        self._fmt_parquet = QRadioButton("Parquet")
        self._fmt_csv.setChecked(True)
        fmt_grp = QButtonGroup(self)
        fmt_grp.addButton(self._fmt_csv)
        fmt_grp.addButton(self._fmt_parquet)
        fmt_row.addWidget(self._fmt_csv)
        fmt_row.addWidget(self._fmt_parquet)
        fmt_row.addStretch()
        layout.addRow("Result format:", fmt_row)

        grp.setLayout(layout)
        self._form_layout.addWidget(grp)

    def _build_actions_section(self) -> None:
        grp = QGroupBox("Actions")
        layout = QVBoxLayout()

        # Row 1: config management
        row1 = QHBoxLayout()
        btn_validate = QPushButton("Validate Config")
        btn_load     = QPushButton("Load from YAML…")
        btn_save     = QPushButton("Save to YAML…")
        btn_validate.clicked.connect(self._validate_config)
        btn_load.clicked.connect(self._load_yaml)
        btn_save.clicked.connect(self._save_yaml)
        row1.addWidget(btn_validate)
        row1.addWidget(btn_load)
        row1.addWidget(btn_save)
        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: submission
        row2 = QHBoxLayout()
        btn_submit = QPushButton("Submit Run")
        btn_submit.setStyleSheet("font-weight: bold;")
        btn_submit.clicked.connect(self._submit_run)
        row2.addWidget(btn_submit)

        row2.addWidget(QLabel("  Submit as Batch —"))
        self._batch_count = QSpinBox()
        self._batch_count.setRange(1, 50)
        self._batch_count.setValue(1)
        self._batch_count.setFixedWidth(60)
        self._batch_count.setSuffix(" copies")
        row2.addWidget(self._batch_count)

        btn_batch = QPushButton("Submit Batch")
        btn_batch.clicked.connect(self._submit_batch)
        row2.addWidget(btn_batch)
        row2.addStretch()
        layout.addLayout(row2)

        grp.setLayout(layout)
        self._form_layout.addWidget(grp)

    def _build_progress_section(self) -> None:
        self._progress_panel = ProgressPanel(self._client)
        self._progress_panel.run_completed.connect(self._on_run_completed)
        self._progress_panel.run_failed.connect(self._on_run_failed)
        self._form_layout.addWidget(self._progress_panel)

    # ===================================================================
    # Visibility logic driven by run type
    # ===================================================================

    def _on_run_type_changed(self, run_type: str) -> None:
        is_stochastic    = run_type == "stochastic"
        is_liability_only = run_type == "liability_only"

        self._fund_config_grp.setVisible(not is_liability_only)
        self._asset_grp.setVisible(not is_liability_only)
        self._scenario_grp.setVisible(is_stochastic)
        self._n_scenarios_label.setVisible(is_stochastic)
        self._n_scenarios.setVisible(is_stochastic)

        # Stochastic requires group MP — lock input mode.
        if is_stochastic:
            self._input_mode_group_mp.setChecked(True)
            self._input_mode_group_mp.setEnabled(False)
            self._input_mode_seriatim.setEnabled(False)
        else:
            self._input_mode_group_mp.setEnabled(True)
            self._input_mode_seriatim.setEnabled(True)

    # ===================================================================
    # Config building
    # ===================================================================

    def build_config_dict(self) -> dict:
        """
        Assemble a dict matching the RunConfig JSON schema from the current
        form state.  Raises ValueError with a user-readable message if any
        required field is missing.
        """
        errors: list[str] = []

        run_name = self._run_name_edit.text().strip()
        if not run_name:
            errors.append("Run name is required.")

        run_type = self._run_type_selector.selected_run_type()

        # Model points
        if self._mp_file_radio.isChecked():
            mp_file = self._mp_file_picker.path()
            if not mp_file:
                errors.append("Model points file is required.")
            model_points = {
                "source_type": "file",
                "file": {
                    "file_path": mp_file,
                    "file_format": "csv",
                },
            }
        else:
            # DB stub — not yet implemented.
            errors.append("Database source not yet supported in this version.")
            model_points = {}

        # Assumptions folder
        assump_folder = self._assump_folder_picker.path()
        if not assump_folder:
            errors.append("Assumptions folder is required.")
        assump_fmt = "csv" if self._assump_fmt_csv.isChecked() else "excel"

        # Fund config — required for DETERMINISTIC and STOCHASTIC only
        fund_config = self._fund_config_picker.path()
        if not fund_config and run_type in ("deterministic", "stochastic"):
            errors.append("Fund config YAML is required for Deterministic and Stochastic runs.")

        # Asset portfolio
        asset_path = None
        if run_type in ("deterministic", "stochastic"):
            asset_path = self._asset_picker.path()
            if not asset_path:
                errors.append("Asset portfolio file is required for this run type.")

        # ESG scenario
        scenario_path = None
        if run_type == "stochastic":
            scenario_path = self._scenario_picker.path()
            if not scenario_path:
                errors.append("ESG scenario file is required for stochastic runs.")

        # Liability models
        active_models = []
        if self._chk_conventional.isChecked():
            active_models.append("conventional")
        if self._chk_unit_linked.isChecked():
            active_models.append("unit_linked")
        if self._chk_annuity.isChecked():
            active_models.append("annuity")
        if not active_models:
            errors.append("At least one liability model must be selected.")

        if errors:
            raise ValueError("\n".join(f"• {e}" for e in errors))

        config: dict = {
            "run_id":   str(uuid.uuid4()),
            "run_name": run_name,
            "run_type": run_type,
            "projection": {
                "valuation_date":       self._val_date.date().toString("yyyy-MM-dd"),
                "projection_term_years": self._proj_term.value(),
                "projection_timestep":  self._proj_timestep.currentData(),
                "decision_timestep":    self._decision_timestep.currentData(),
                "currency":             self._currency.currentData(),
            },
            "input_sources": {
                "model_points":      model_points,
                "assumption_tables": {
                    "tables_root_dir":  assump_folder,
                    "file_format":      assump_fmt,
                },
                **({"fund_config_path": fund_config} if fund_config else {}),
            },
            "liability": {
                "active_models": active_models,
                "input_mode":    (
                    "group_mp" if self._input_mode_group_mp.isChecked()
                    else "seriatim"
                ),
            },
            "output": {
                "output_dir":       self._output_folder_picker.path() or "outputs",
                "output_timestep":  self._output_timestep.currentData(),
                "result_format":    "csv" if self._fmt_csv.isChecked() else "parquet",
            },
        }

        if asset_path:
            config["input_sources"]["asset_data_path"] = asset_path
        if scenario_path:
            config["input_sources"]["scenario_file_path"] = scenario_path
        if run_type == "stochastic":
            config["stochastic"] = {"num_scenarios": self._n_scenarios.value()}
        if self._notes_edit.text().strip():
            config["notes"] = self._notes_edit.text().strip()

        return config

    # ===================================================================
    # Action handlers
    # ===================================================================

    def _validate_config(self) -> None:
        try:
            cfg = self.build_config_dict()
        except ValueError as exc:
            QMessageBox.warning(self, "Validation — Form Errors", str(exc))
            return

        config_json = json.dumps(cfg)
        try:
            result = self._client.validate_config(config_json)
            QMessageBox.information(
                self, "Validation Passed",
                result.get("message", "Config is valid.")
            )
        except APIResponseError as exc:
            QMessageBox.warning(
                self, "Validation Failed",
                f"Server rejected the config:\n{exc.detail}"
            )
        except APIError:
            QMessageBox.critical(
                self, "API Unreachable",
                f"Cannot reach the API server at {self._client._base}.\n\n"
                "Start it with:\n"
                "    uv run uvicorn api.main:app --reload --port 8000",
            )

    def _submit_run(self) -> None:
        try:
            cfg = self.build_config_dict()
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot Submit — Form Errors", str(exc))
            return

        config_json = json.dumps(cfg)
        try:
            run = self._client.submit_run(config_json)
        except APIResponseError as exc:
            QMessageBox.warning(self, "Submission Failed", exc.detail)
            return
        except APIError:
            QMessageBox.critical(
                self, "API Unreachable",
                f"Cannot reach the API server at {self._client._base}.\n\n"
                "Start it with:\n"
                "    uv run uvicorn api.main:app --reload --port 8000",
            )
            return

        self._progress_panel.track(run.run_id)

    def _submit_batch(self) -> None:
        try:
            cfg = self.build_config_dict()
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot Submit — Form Errors", str(exc))
            return

        n = self._batch_count.value()
        configs = []
        for _ in range(n):
            cfg["run_id"] = str(uuid.uuid4())
            configs.append(json.dumps(cfg))

        label = self._run_name_edit.text().strip() or None
        try:
            batch = self._client.submit_batch(configs, label=label)
        except APIResponseError as exc:
            QMessageBox.warning(self, "Batch Submission Failed", exc.detail)
            return
        except APIError:
            QMessageBox.critical(
                self, "API Unreachable",
                f"Cannot reach the API server at {self._client._base}.\n\n"
                "Start it with:\n"
                "    uv run uvicorn api.main:app --reload --port 8000",
            )
            return

        QMessageBox.information(
            self,
            "Batch Submitted",
            f"Batch {batch.batch_id}\n{n} run(s) queued.\nStatus: {batch.status}",
        )

    def _load_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load run config", "", "YAML files (*.yaml *.yml);;All files (*)"
        )
        if not path:
            return
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
            self._populate_from_dict(data)
        except Exception as exc:
            QMessageBox.warning(self, "Load Failed", str(exc))

    def _save_yaml(self) -> None:
        try:
            cfg = self.build_config_dict()
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot Save — Form Errors", str(exc))
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save run config", "", "YAML files (*.yaml *.yml)"
        )
        if not path:
            return
        try:
            import yaml
            with open(path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            QMessageBox.information(self, "Saved", f"Config saved to:\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save Failed", str(exc))

    def _populate_from_dict(self, data: dict) -> None:
        """
        Populate form fields from a RunConfig dict (e.g. loaded from YAML).
        Best-effort: unknown or missing keys are silently ignored.
        """
        self._run_name_edit.setText(data.get("run_name", ""))
        self._notes_edit.setText(data.get("notes", ""))

        run_type = data.get("run_type", "deterministic")
        self._run_type_selector.set_run_type(run_type)
        self._on_run_type_changed(run_type)

        proj = data.get("projection", {})
        if proj.get("valuation_date"):
            self._val_date.setDate(
                QDate.fromString(str(proj["valuation_date"]), "yyyy-MM-dd")
            )
        if proj.get("projection_term_years"):
            self._proj_term.setValue(int(proj["projection_term_years"]))
        _set_combo_by_data(self._proj_timestep, proj.get("projection_timestep"))
        _set_combo_by_data(self._decision_timestep, proj.get("decision_timestep"))
        _set_combo_by_data(self._currency, proj.get("currency"))

        sources = data.get("input_sources", {})
        mp = sources.get("model_points", {})
        if mp.get("source_type") == "file" and mp.get("file", {}).get("file_path"):
            self._mp_file_radio.setChecked(True)
            self._mp_stack.setCurrentIndex(0)
            self._mp_file_picker.set_path(str(mp["file"]["file_path"]))

        assump = sources.get("assumption_tables", {})
        if assump.get("tables_root_dir"):
            self._assump_folder_picker.set_path(str(assump["tables_root_dir"]))
        if assump.get("file_format") == "excel":
            self._assump_fmt_excel.setChecked(True)
        else:
            self._assump_fmt_csv.setChecked(True)

        if sources.get("fund_config_path"):
            self._fund_config_picker.set_path(str(sources["fund_config_path"]))
        if sources.get("asset_data_path"):
            self._asset_picker.set_path(str(sources["asset_data_path"]))
        if sources.get("scenario_file_path"):
            self._scenario_picker.set_path(str(sources["scenario_file_path"]))

        liability = data.get("liability", {})
        models = liability.get("active_models", [])
        self._chk_conventional.setChecked("conventional" in models)
        self._chk_unit_linked.setChecked("unit_linked" in models)
        self._chk_annuity.setChecked("annuity" in models)
        if liability.get("input_mode") == "seriatim":
            self._input_mode_seriatim.setChecked(True)
        else:
            self._input_mode_group_mp.setChecked(True)

        stoch = data.get("stochastic", {})
        if stoch.get("num_scenarios"):
            self._n_scenarios.setValue(int(stoch["num_scenarios"]))

        output = data.get("output", {})
        if output.get("output_dir"):
            self._output_folder_picker.set_path(str(output["output_dir"]))
        _set_combo_by_data(self._output_timestep, output.get("output_timestep"))
        if output.get("result_format") == "parquet":
            self._fmt_parquet.setChecked(True)
        else:
            self._fmt_csv.setChecked(True)

    # ===================================================================
    # Progress callbacks
    # ===================================================================

    def _on_run_completed(self, run_id: str) -> None:
        QMessageBox.information(
            self,
            "Run Completed",
            f"Run {run_id} completed successfully.\n\nSwitch to the Results tab to view outputs.",
        )

    def _on_run_failed(self, run_id: str, error: str) -> None:
        QMessageBox.warning(
            self,
            "Run Failed",
            f"Run {run_id} failed.\n\n{error or '(no error message)'}",
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _set_combo_by_data(combo: QComboBox, value: str | None) -> None:
    """Select the QComboBox item whose userData matches value (if found)."""
    if value is None:
        return
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return
