"""
Standalone config loader for the ALM model.

Provides a single entry point for loading and validating all config objects
from YAML files. The run mode orchestrator calls ConfigLoader at startup
rather than instantiating config classes directly, so all I/O and validation
errors surface in one place before any model component is constructed.

Usage:
    from engine.config.config_loader import ConfigLoader

    run_cfg, fund_cfg = ConfigLoader.load_all("config_files/run_config.yaml")

    # Or load individually:
    run_cfg  = ConfigLoader.load_run_config("config_files/run_config.yaml")
    fund_cfg = ConfigLoader.load_fund_config("config_files/fund_config.yaml")
"""
from __future__ import annotations

from pathlib import Path

from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig


class ConfigLoader:
    """
    Loads and validates ALM config objects from YAML files.

    All methods are static — ConfigLoader is a namespace for loading
    functions, not a stateful object.
    """

    @staticmethod
    def load_run_config(path: str | Path) -> RunConfig:
        """
        Load and validate a RunConfig from a YAML file.

        Args:
            path: Path to the run config YAML file.

        Returns:
            Validated RunConfig instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If validation fails.
        """
        return RunConfig.from_yaml(path)

    @staticmethod
    def load_fund_config(path: str | Path) -> FundConfig:
        """
        Load and validate a FundConfig from a YAML file.

        Args:
            path: Path to the fund config YAML file.

        Returns:
            Validated FundConfig instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If validation fails.
        """
        return FundConfig.from_yaml(path)

    @staticmethod
    def load_all(run_config_path: str | Path) -> tuple[RunConfig, FundConfig]:
        """
        Load both RunConfig and FundConfig in one call.

        The fund config path is read from RunConfig.input_sources.fund_config_path,
        so only the run config path needs to be provided externally.

        This is the preferred entry point for run mode orchestrators — it
        validates everything before any model component is constructed.

        Args:
            run_config_path: Path to the run config YAML file.

        Returns:
            Tuple of (RunConfig, FundConfig), both fully validated.

        Raises:
            FileNotFoundError: If either file does not exist.
            ValueError: If either config fails validation.

        Example:
            run_cfg, fund_cfg = ConfigLoader.load_all("config_files/run_config.yaml")
            run = LiabilityOnlyRun(run_cfg, fund_cfg)
        """
        run_cfg  = RunConfig.from_yaml(run_config_path)
        fund_config_path = run_cfg.input_sources.fund_config_path
        if fund_config_path is None:
            raise ValueError(
                "RunConfig.input_sources.fund_config_path must be set for load_all(). "
                "Use a run config that includes a fund_config_path."
            )
        fund_cfg = FundConfig.from_yaml(fund_config_path)
        return run_cfg, fund_cfg
