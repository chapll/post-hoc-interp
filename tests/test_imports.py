"""Verify all src.infra modules are importable."""

import os
import subprocess
import sys


def test_import_config():
    from src.infra.config import load_config  # noqa: F401


def test_import_datasets():
    from src.infra.datasets import load_folio, load_gsm_symbolic_contamination, load_strategyqa  # noqa: F401


def test_import_model():
    from src.infra.model import load_model, generate_trace  # noqa: F401


def test_import_multiprocess():
    from src.infra.multiprocess import run_parallel_isolated, run_parallel, split_work  # noqa: F401


def test_import_prompts():
    from src.infra.prompts import build_verification_prompt  # noqa: F401


def test_import_storage():
    from src.infra.storage import save_trace, get_trace_json_path, derive_trace_seed  # noqa: F401


def test_import_tokens():
    from src.infra.tokens import parse_think_boundaries, THINK_TOKEN_ID, THINK_END_TOKEN_ID  # noqa: F401


def test_import_utils():
    from src.infra.utils import set_seed, setup_logging, get_device  # noqa: F401


def test_subprocess_import():
    """Verify src.infra modules can be imported in a subprocess (simulates mp.Process workers)."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.infra.config import load_config; from src.infra.tokens import parse_think_boundaries; print('OK')",
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    assert result.returncode == 0, f"Subprocess import failed: {result.stderr}"
