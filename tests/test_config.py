"""Tests for src.config.load_config.

load_config(script_config: str = None) -> DictConfig

When script_config is not None, argparse is skipped (programmatic mode):
- script_config="" or non-existent path: loads only base.yaml
- script_config="configs/generate.yaml": merges generate.yaml over base.yaml
When script_config is None, uses argparse (would conflict with pytest).
"""

import os

import pytest
from omegaconf import DictConfig

from src.config import load_config


def test_load_base_config_returns_dictconfig():
    """load_config with empty string loads only base.yaml."""
    cfg = load_config(script_config="")
    assert isinstance(cfg, DictConfig)


def test_load_base_config_has_paths():
    """base.yaml resolves paths.data_dir from DATA_DIR env var, falling back to 'data'."""
    os.environ.pop("DATA_DIR", None)
    cfg = load_config(script_config="")
    assert cfg.paths.data_dir == "data"
    assert cfg.paths.experiments_dir == "experiments"


def test_load_base_config_has_model():
    """base.yaml must contain model configuration."""
    cfg = load_config(script_config="")
    assert cfg.model.name == "Qwen/Qwen3-8B"
    assert cfg.model.dtype == "bfloat16"


def test_load_base_config_has_seed():
    """base.yaml must contain seed == 137."""
    cfg = load_config(script_config="")
    assert cfg.seed == 137


def test_load_with_script_config_merges():
    """Merging generate.yaml over base.yaml should add generation-specific keys."""
    os.environ.pop("DATA_DIR", None)
    cfg = load_config(script_config="configs/generate.yaml")
    assert isinstance(cfg, DictConfig)
    # generate.yaml sets paths.traces_dir
    assert cfg.paths.traces_dir == "data/traces"
    # generate.yaml overrides model.max_new_tokens from 8192 to 4096
    assert cfg.model.max_new_tokens == 4096


def test_load_config_nonexistent_script_config():
    """Non-existent script_config path should still return base config."""
    os.environ.pop("DATA_DIR", None)
    cfg = load_config(script_config="/nonexistent/path.yaml")
    assert isinstance(cfg, DictConfig)
    # Should have base config values since the file doesn't exist
    assert cfg.paths.data_dir == "data"


def test_data_dir_env_override():
    """Setting DATA_DIR env var overrides paths.data_dir and derived paths."""
    os.environ["DATA_DIR"] = "/tmp/test-data"
    try:
        cfg = load_config(script_config="configs/generate.yaml")
        assert cfg.paths.data_dir == "/tmp/test-data"
        assert cfg.paths.traces_dir == "/tmp/test-data/traces"
    finally:
        os.environ.pop("DATA_DIR", None)
