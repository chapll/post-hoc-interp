"""Trace storage module for saving and loading activations via safetensors.

Handles BF16 activations, top-k logit values/indices, per-position entropy,
and token IDs with companion JSON metadata files. All tensor dtypes are
enforced on save to ensure consistent storage format.

Also provides path helpers for the decoupled trace/activation directory layout:
- Traces (JSON): data/traces/{valid,failed}/{dataset}/{problem_id}/trace_NN.json
- Activations (safetensors): data/activations/{dataset}/{problem_id}/trace_NN.safetensors
"""

import hashlib
import json
import os
import tempfile

import torch
from safetensors.torch import load_file, save_file


def save_trace(
    tensors: dict[str, torch.Tensor],
    path: str,
    metadata: dict = None,
) -> None:
    """Save a dict of named tensors to a safetensors file.

    All tensors must already have the correct dtypes (BF16 for activations,
    int32 for token IDs and topk indices, etc.). Use prepare_trace_tensors()
    to produce a properly typed dict from raw model outputs.

    Tensors are moved to CPU if not already there.

    Args:
        tensors: Dict mapping tensor names to torch.Tensor values.
        path: Output file path (should end in .safetensors).
        metadata: Optional metadata dict to save as companion JSON.

    Raises:
        RuntimeError: If safetensors save fails.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Ensure all tensors are on CPU and contiguous
    cpu_tensors = {}
    for name, tensor in tensors.items():
        t = tensor.cpu() if tensor.device.type != "cpu" else tensor
        cpu_tensors[name] = t.contiguous()

    save_file(cpu_tensors, path)

    if metadata is not None:
        meta_path = get_metadata_path(path)
        save_metadata(metadata, meta_path)


def load_trace(path: str) -> dict[str, torch.Tensor]:
    """Load all tensors from a safetensors file.

    Args:
        path: Path to .safetensors file.

    Returns:
        Dict mapping tensor names to CPU tensors with original dtypes.

    Raises:
        FileNotFoundError: If path does not exist.
        RuntimeError: If safetensors load fails.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trace file not found: {path}")

    return load_file(path)


def prepare_trace_tensors(
    activations: dict[str, torch.Tensor],
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    top_k: int = 50,
) -> dict[str, torch.Tensor]:
    """Process raw model outputs into storage-ready tensors with correct dtypes.

    Takes raw activations (from capture_activations), full logits, and token IDs,
    then produces a flat dict of tensors ready for save_trace.

    Args:
        activations: Dict with keys like residual_layer_{i} and last_token_layer_{i}.
            Tensors can be any float dtype; will be converted to BF16.
        logits: Full logit tensor of shape [1, seq_len, vocab_size] (any float dtype).
        token_ids: Token ID tensor, 1D or 2D. Will be flattened and cast to int32.
        top_k: Number of top logit values/indices to store per position.

    Returns:
        Dict of named tensors, all on CPU with correct dtypes:
        - residual_layer_* and last_token_layer_*: BF16
        - topk_values: BF16, shape [seq_len, top_k]
        - topk_indices: int32, shape [seq_len, top_k]
        - entropy: BF16, shape [seq_len]
        - token_ids: int32, shape [seq_len]
    """
    result = {}

    # Copy activation tensors, convert to BF16 on CPU
    for key, tensor in activations.items():
        if key.startswith("residual_layer_") or key.startswith("last_token_layer_"):
            t = tensor.detach().cpu()
            # Squeeze leading batch dimension if present
            if t.dim() == 3 and t.shape[0] == 1:
                t = t.squeeze(0)  # [1, seq, hidden] -> [seq, hidden]
            elif t.dim() == 2 and key.startswith("last_token_layer_") and t.shape[0] == 1:
                t = t.squeeze(0)  # [1, hidden] -> [hidden]
            result[key] = t.to(torch.bfloat16)

    # Process logits: compute top-k and entropy
    # logits expected shape: [1, seq_len, vocab_size]
    logits_2d = logits[0].detach().float()  # [seq_len, vocab_size], float32 for stability

    # Top-k values and indices
    topk = torch.topk(logits_2d, k=top_k, dim=-1)
    result["topk_values"] = topk.values.to(torch.bfloat16).cpu()
    result["topk_indices"] = topk.indices.to(torch.int32).cpu()

    # Entropy: -(probs * log(probs)).sum(dim=-1)
    # Use float32 for numerical stability
    probs = torch.softmax(logits_2d, dim=-1)  # already float32
    log_probs = probs.log()
    # Clamp log_probs to avoid NaN from log(0)
    log_probs = log_probs.clamp(min=-1e10)
    entropy = -(probs * log_probs).sum(dim=-1)  # [seq_len]
    result["entropy"] = entropy.to(torch.bfloat16).cpu()

    # Token IDs as int32
    ids = token_ids.detach().cpu().flatten().to(torch.int32)
    result["token_ids"] = ids

    return result


def save_metadata(metadata: dict, path: str) -> None:
    """Save metadata as a JSON file.

    Args:
        metadata: Dict of metadata fields (problem_id, dataset_name,
            ground_truth, num_tokens, etc.).
        path: Output JSON file path.

    Raises:
        TypeError: If metadata contains non-serializable values.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(path: str) -> dict:
    """Load metadata from a JSON file.

    Args:
        path: Path to JSON metadata file.

    Returns:
        Dict of metadata fields.

    Raises:
        FileNotFoundError: If path does not exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


def get_trace_path(
    base_dir: str, dataset_name: str, problem_id: str, trace_idx: int
) -> str:
    """Return standardized file path for a trace.

    Format: {base_dir}/traces/{dataset_name}/{problem_id}/trace_{trace_idx:02d}.safetensors

    Args:
        base_dir: Root directory for trace storage.
        dataset_name: Name of the dataset (e.g., gsm_symbolic).
        problem_id: Unique problem identifier.
        trace_idx: Integer index of the trace (0-padded to 2 digits).

    Returns:
        Full path string to the safetensors file.
    """
    return os.path.join(
        base_dir,
        "traces",
        dataset_name,
        problem_id,
        f"trace_{trace_idx:02d}.safetensors",
    )


def get_metadata_path(trace_path: str) -> str:
    """Return the companion metadata JSON path for a safetensors trace file.

    Simply replaces the .safetensors extension with .json.

    Args:
        trace_path: Path to a .safetensors file.

    Returns:
        Path with .json extension.
    """
    base, _ = os.path.splitext(trace_path)
    return base + ".json"


# ---------------------------------------------------------------------------
# Decoupled trace/activation directory layout helpers
# ---------------------------------------------------------------------------


def get_trace_json_path(
    traces_dir: str, dataset_name: str, problem_id: str, trace_idx: int
) -> str:
    """Return path for a valid trace JSON file in the decoupled layout.

    Format: {traces_dir}/valid/{dataset_name}/{problem_id}/trace_{trace_idx:02d}.json

    Args:
        traces_dir: Root directory for trace storage.
        dataset_name: Name of the dataset (e.g., gsm_symbolic).
        problem_id: Unique problem identifier.
        trace_idx: Integer index of the trace (0-padded to 2 digits).

    Returns:
        Full path string to the trace JSON file.
    """
    return os.path.join(
        traces_dir,
        "valid",
        dataset_name,
        problem_id,
        f"trace_{trace_idx:02d}.json",
    )


def get_failed_trace_json_path(
    traces_dir: str, dataset_name: str, problem_id: str, trace_idx: int
) -> str:
    """Return path for a failed/quarantined trace JSON file.

    Format: {traces_dir}/failed/{dataset_name}/{problem_id}/trace_{trace_idx:02d}.json

    Args:
        traces_dir: Root directory for trace storage.
        dataset_name: Name of the dataset (e.g., gsm_symbolic).
        problem_id: Unique problem identifier.
        trace_idx: Integer index of the trace (0-padded to 2 digits).

    Returns:
        Full path string to the failed trace JSON file.
    """
    return os.path.join(
        traces_dir,
        "failed",
        dataset_name,
        problem_id,
        f"trace_{trace_idx:02d}.json",
    )


def get_activation_path(
    activations_dir: str, dataset_name: str, problem_id: str, trace_idx: int
) -> str:
    """Return path for an activation safetensors file in the decoupled layout.

    Format: {activations_dir}/{dataset_name}/{problem_id}/trace_{trace_idx:02d}.safetensors

    Args:
        activations_dir: Root directory for activation storage.
        dataset_name: Name of the dataset (e.g., gsm_symbolic).
        problem_id: Unique problem identifier.
        trace_idx: Integer index of the trace (0-padded to 2 digits).

    Returns:
        Full path string to the activation safetensors file.
    """
    return os.path.join(
        activations_dir,
        dataset_name,
        problem_id,
        f"trace_{trace_idx:02d}.safetensors",
    )


def write_trace_json(data: dict, path: str) -> None:
    """Write trace data to a JSON file atomically.

    Uses tempfile.mkstemp to create a temporary file in the same directory
    as the target, writes the JSON data, then performs a POSIX atomic
    os.rename(). If any error occurs, the temporary file is cleaned up
    before re-raising.

    Args:
        data: Dict of trace data to serialize as JSON.
        path: Target output file path.

    Raises:
        TypeError: If data contains non-JSON-serializable values.
        OSError: If file operations fail.
    """
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.rename(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def derive_trace_seed(problem_id: str, trace_index: int) -> int:
    """Derive a deterministic seed from problem_id and trace_index.

    Uses MD5 hash for deterministic mixing (not security -- just
    reproducible seed derivation). The result is stateless and
    order-independent: any problem can be generated on any worker
    in any order without global state.

    Per project rules, seed 42 is never returned (returns 43 instead).

    Args:
        problem_id: Unique problem identifier string.
        trace_index: Integer trace index for this problem.

    Returns:
        Integer seed in range [0, 2^32), never 42.
    """
    key = f"{problem_id}_{trace_index}"
    hash_hex = hashlib.md5(key.encode()).hexdigest()
    seed = int(hash_hex, 16) % (2**32)
    if seed == 42:
        seed = 43
    return seed
