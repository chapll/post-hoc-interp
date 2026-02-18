"""Verify all src modules are importable."""

import os
import subprocess
import sys


def test_import_config():
    from src.config import load_config  # noqa: F401


def test_import_datasets():
    from src.datasets import load_folio, load_gsm_symbolic_contamination, load_strategyqa  # noqa: F401


def test_import_model():
    from src.model import load_model, generate_trace  # noqa: F401


def test_import_multiprocess():
    from src.multiprocess import run_parallel_isolated, run_parallel, split_work  # noqa: F401


def test_import_prompts():
    from src.prompts import build_verification_prompt  # noqa: F401


def test_import_storage():
    from src.storage import save_trace, get_trace_json_path, derive_trace_seed  # noqa: F401


def test_import_tokens():
    from src.tokens import parse_think_boundaries, THINK_TOKEN_ID, THINK_END_TOKEN_ID  # noqa: F401


def test_import_utils():
    from src.utils import set_seed, setup_logging, get_device  # noqa: F401


def test_subprocess_import():
    """Verify src modules can be imported in a subprocess (simulates mp.Process workers)."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.config import load_config; from src.tokens import parse_think_boundaries; print('OK')",
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    assert result.returncode == 0, f"Subprocess import failed: {result.stderr}"
