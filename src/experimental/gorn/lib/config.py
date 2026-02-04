"""Configuration loader using OmegaConf with argparse CLI integration.

Loads base.yaml defaults, optionally merges a script-specific YAML,
and applies CLI overrides including a --gpus convenience flag.
"""

import argparse
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


# Resolve the path to configs/ relative to this file's location
_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
_BASE_CONFIG = _CONFIGS_DIR / "base.yaml"


def load_config(script_config: str = None) -> DictConfig:
    """Load and merge configuration from YAML files and CLI arguments.

    Loading order (later overrides earlier):
    1. base.yaml (always loaded)
    2. Script-specific YAML (if --config or script_config provided)
    3. CLI dotlist overrides (e.g., model.name=foo)
    4. --gpus flag (overrides cfg.gpus)

    Args:
        script_config: Path to script-specific YAML file. When provided,
            argparse is skipped (useful for programmatic/testing use).
            Pass None to use argparse for CLI invocation.

    Returns:
        Merged OmegaConf DictConfig with all overrides applied.

    Raises:
        FileNotFoundError: If base.yaml or specified config file not found.
    """
    # Load base config
    if not _BASE_CONFIG.exists():
        raise FileNotFoundError(
            f"Base config not found at {_BASE_CONFIG}. "
            f"Expected configs/base.yaml relative to the lib/ directory."
        )
    base_cfg = OmegaConf.load(_BASE_CONFIG)

    if script_config is not None:
        # Programmatic mode: merge script config if it is a valid path
        if script_config and os.path.isfile(script_config):
            script_cfg = OmegaConf.load(script_config)
            base_cfg = OmegaConf.merge(base_cfg, script_cfg)
        return base_cfg

    # CLI mode: parse arguments
    parser = argparse.ArgumentParser(
        description="Experiment script configuration",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to script-specific YAML config to merge on top of base.yaml",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., 0,1,2). Overrides gpus in config.",
    )

    args, unknown = parser.parse_known_args()

    # Merge script-specific config if provided
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Script config not found: {config_path}")
        script_cfg = OmegaConf.load(config_path)
        base_cfg = OmegaConf.merge(base_cfg, script_cfg)

    # Apply CLI dotlist overrides (e.g., model.name=foo training.steps=1000)
    if unknown:
        cli_cfg = OmegaConf.from_dotlist(unknown)
        base_cfg = OmegaConf.merge(base_cfg, cli_cfg)

    # Apply --gpus override
    if args.gpus is not None:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
        base_cfg.gpus = gpu_ids

    return base_cfg
