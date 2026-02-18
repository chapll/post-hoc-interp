"""Tests for src.storage.

Tested functions (no GPU needed):
    derive_trace_seed(problem_id: str, trace_index: int) -> int
    get_trace_json_path(traces_dir, dataset_name, problem_id, trace_idx) -> str
    get_failed_trace_json_path(traces_dir, dataset_name, problem_id, trace_idx) -> str
    get_activation_path(activations_dir, dataset_name, problem_id, trace_idx) -> str
    get_metadata_path(trace_path: str) -> str
    write_trace_json(data: dict, path: str) -> None
    save_metadata(metadata: dict, path: str) -> None
    load_metadata(path: str) -> dict
"""

import json
import os

import pytest

from src.storage import (
    derive_trace_seed,
    get_activation_path,
    get_failed_trace_json_path,
    get_metadata_path,
    get_trace_json_path,
    load_metadata,
    save_metadata,
    write_trace_json,
)


class TestDeriveTraceSeed:
    def test_deterministic(self):
        """Same inputs must produce same seed."""
        s1 = derive_trace_seed("prob_001", 0)
        s2 = derive_trace_seed("prob_001", 0)
        assert s1 == s2

    def test_different_inputs(self):
        """Different inputs should produce different seeds."""
        s1 = derive_trace_seed("prob_001", 0)
        s2 = derive_trace_seed("prob_001", 1)
        s3 = derive_trace_seed("prob_002", 0)
        assert s1 != s2
        assert s1 != s3

    def test_never_42(self):
        """Seed must be an int in [0, 2^32) and never 42."""
        # Check type and range
        seed = derive_trace_seed("test", 0)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**32

    def test_returns_int(self):
        seed = derive_trace_seed("anything", 99)
        assert isinstance(seed, int)


class TestPathHelpers:
    def test_get_trace_json_path(self):
        path = get_trace_json_path("data/traces", "gsm_symbolic", "prob_001", 3)
        assert path == os.path.join("data/traces", "valid", "gsm_symbolic", "prob_001", "trace_03.json")

    def test_get_failed_trace_json_path(self):
        path = get_failed_trace_json_path("data/traces", "gsm_symbolic", "prob_001", 3)
        assert path == os.path.join("data/traces", "failed", "gsm_symbolic", "prob_001", "trace_03.json")

    def test_get_activation_path(self):
        path = get_activation_path("data/activations", "folio", "prob_X", 0)
        assert path == os.path.join("data/activations", "folio", "prob_X", "trace_00.safetensors")

    def test_get_metadata_path_safetensors(self):
        assert get_metadata_path("/tmp/trace_01.safetensors") == "/tmp/trace_01.json"

    def test_get_metadata_path_other_extension(self):
        assert get_metadata_path("/tmp/data.bin") == "/tmp/data.json"

    def test_trace_idx_zero_padding(self):
        """trace_idx should be zero-padded to 2 digits."""
        path = get_trace_json_path("root", "ds", "pid", 0)
        assert "trace_00.json" in path
        path = get_trace_json_path("root", "ds", "pid", 9)
        assert "trace_09.json" in path


class TestWriteTraceJson:
    def test_roundtrip(self, tmp_path):
        """write_trace_json writes valid JSON that can be read back."""
        data = {"problem_id": "p1", "answer": "42", "tokens": [1, 2, 3]}
        out_path = str(tmp_path / "trace_00.json")
        write_trace_json(data, out_path)
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        """write_trace_json should create parent directories if they don't exist."""
        out_path = str(tmp_path / "a" / "b" / "trace_00.json")
        write_trace_json({"key": "value"}, out_path)
        assert os.path.exists(out_path)

    def test_atomic_write_content(self, tmp_path):
        """File should contain properly formatted JSON."""
        data = {"nested": {"key": [1, 2, 3]}}
        out_path = str(tmp_path / "trace.json")
        write_trace_json(data, out_path)
        with open(out_path) as f:
            content = f.read()
        # write_trace_json uses indent=2
        assert '"nested"' in content


class TestMetadata:
    def test_roundtrip(self, tmp_path):
        meta = {"dataset": "gsm", "num_tokens": 100}
        meta_path = str(tmp_path / "meta.json")
        save_metadata(meta, meta_path)
        loaded = load_metadata(meta_path)
        assert loaded == meta

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_metadata("/nonexistent/meta.json")

    def test_save_creates_parent_dirs(self, tmp_path):
        meta_path = str(tmp_path / "nested" / "dir" / "meta.json")
        save_metadata({"key": "val"}, meta_path)
        assert os.path.exists(meta_path)
