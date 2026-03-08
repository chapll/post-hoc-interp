"""Phase 2 verification script for trace generation and activation capture.

Checks all Phase 2 outputs for completeness, correctness, and consistency:
  1. Trace completeness (per-dataset counts in valid/)
  2. Trace metadata integrity (sampled JSON field validation)
  3. Think block integrity (token-level parsing cross-check)
  4. GSM-Symbolic contamination coverage (is_fresh field balance)
  5. Seed reproducibility (derive_trace_seed cross-check)
  6. Quality report existence and contents
  7. Activation completeness (safetensors coverage vs valid traces)
  8. Activation shape verification (sampled tensor shapes and dtypes)
  9. Disk usage summary

Does not require GPU access. Reads only from disk.

Usage:
    uv run python scripts/verify_phase2.py \
        --config configs/capture.yaml
"""

import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from safetensors.torch import load_file

from src.config import load_config
from src.storage import derive_trace_seed
from src.tokens import parse_think_boundaries

# -- globals ------------------------------------------------------------------

results: dict[str, list[dict]] = {}
any_failure = False


def record(category: str, check: str, passed: bool, detail: str = ""):
    """Record a check result and print it."""
    global any_failure
    status = "PASS" if passed else "FAIL"
    if not passed:
        any_failure = True
    msg = f"  [{status}] {check}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    results.setdefault(category, []).append(
        {"check": check, "passed": passed, "detail": detail}
    )


def find_trace_files(traces_dir: str, subdir: str = "valid") -> list[Path]:
    """Recursively find all trace_*.json files under the given subdirectory.

    Args:
        traces_dir: Root traces directory.
        subdir: Subdirectory under traces_dir to scan ("valid" or "failed").

    Returns:
        Sorted list of Path objects for each trace JSON file found.
    """
    target_dir = Path(traces_dir) / subdir
    if not target_dir.exists():
        return []
    return sorted(target_dir.rglob("trace_*.json"))


def get_dataset_from_path(file_path: Path) -> str:
    """Extract dataset name from trace path structure.

    Expected path: .../valid/{dataset}/{problem_id}/trace_NN.json
    """
    parts = file_path.parts
    for i, part in enumerate(parts):
        if part in ("valid", "failed") and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


# =============================================================================
# Check 1: Trace completeness
# =============================================================================


def check_trace_completeness(traces_dir: str) -> dict[str, list[Path]]:
    """Count trace JSON files per dataset in valid/ and check against expectations."""
    print("\n--- Check 1: Trace Completeness ---")

    valid_files = find_trace_files(traces_dir, "valid")
    failed_files = find_trace_files(traces_dir, "failed")

    if not valid_files:
        record("completeness", "valid_traces_exist", False, "no valid/ directory or no files")
        return {}

    record("completeness", "valid_traces_exist", True, f"{len(valid_files)} files found")

    # Group by dataset
    per_dataset: dict[str, list[Path]] = defaultdict(list)
    for fp in valid_files:
        ds = get_dataset_from_path(fp)
        per_dataset[ds].append(fp)

    failed_per_dataset: dict[str, int] = defaultdict(int)
    for fp in failed_files:
        ds = get_dataset_from_path(fp)
        failed_per_dataset[ds] += 1

    # Expected counts (from generate.yaml defaults):
    # GSM-Symbolic: 200 problems x 5 traces = 1000
    # FOLIO: 100 problems x 5 traces = 500
    # StrategyQA: 100 problems x 5 traces = 500
    # These are defaults; actual counts may differ based on CLI overrides.
    expected = {
        "gsm_symbolic": 1000,
        "folio": 500,
        "strategyqa": 500,
    }

    for ds_name in sorted(set(list(per_dataset.keys()) + list(expected.keys()))):
        count = len(per_dataset.get(ds_name, []))
        failed_count = failed_per_dataset.get(ds_name, 0)
        exp = expected.get(ds_name, 0)

        # Check within 80% of expected (accounts for minority class capping,
        # CLI overrides, and failed traces routed away).
        threshold = int(exp * 0.8)
        passed = count >= threshold if exp > 0 else count > 0
        record(
            "completeness",
            f"count_{ds_name}",
            passed,
            f"valid={count}, failed={failed_count}, expected>={threshold} (80% of {exp})",
        )

    total_valid = len(valid_files)
    total_failed = len(failed_files)
    record(
        "completeness",
        "total_traces",
        total_valid > 0,
        f"valid={total_valid}, failed={total_failed}, total={total_valid + total_failed}",
    )

    return per_dataset


# =============================================================================
# Check 2: Trace metadata integrity
# =============================================================================


def check_metadata_integrity(per_dataset: dict[str, list[Path]], sample_size: int = 10):
    """Sample traces per dataset and verify JSON structure and metadata fields."""
    print("\n--- Check 2: Trace Metadata Integrity ---")

    required_top_fields = {"token_ids", "generated_text", "metadata"}
    required_meta_fields = {
        "problem_id",
        "dataset_name",
        "ground_truth",
        "trace_index",
        "seed",
        "num_tokens",
        "num_think_blocks",
        "model_name",
        "temperature",
        "top_p",
        "top_k",
        "timestamp",
    }

    for ds_name, files in sorted(per_dataset.items()):
        sample = files[:sample_size] if len(files) <= sample_size else random.sample(files, sample_size)
        print(f"\n  Dataset: {ds_name} (sampling {len(sample)}/{len(files)} traces)")

        all_ok = True
        for fp in sample:
            try:
                with open(fp, "r") as f:
                    data = json.load(f)
            except Exception as e:
                record("metadata", f"json_parse_{ds_name}", False, f"{fp.name}: {e}")
                all_ok = False
                continue

            # Check top-level fields
            missing_top = required_top_fields - set(data.keys())
            if missing_top:
                record("metadata", f"top_fields_{ds_name}", False, f"{fp.name}: missing {missing_top}")
                all_ok = False
                continue

            # Check metadata fields
            meta = data["metadata"]
            missing_meta = required_meta_fields - set(meta.keys())
            if missing_meta:
                record("metadata", f"meta_fields_{ds_name}", False, f"{fp.name}: missing {missing_meta}")
                all_ok = False
                continue

            # Verify token_ids is a non-empty list of ints
            token_ids = data["token_ids"]
            if not isinstance(token_ids, list) or len(token_ids) == 0:
                record("metadata", f"token_ids_type_{ds_name}", False, f"{fp.name}: empty or not list")
                all_ok = False
                continue
            if not all(isinstance(t, int) for t in token_ids[:100]):
                record("metadata", f"token_ids_ints_{ds_name}", False, f"{fp.name}: non-int values")
                all_ok = False
                continue

            # Verify generated_text is a non-empty string
            if not isinstance(data["generated_text"], str) or len(data["generated_text"]) == 0:
                record("metadata", f"generated_text_{ds_name}", False, f"{fp.name}: empty or not string")
                all_ok = False
                continue

            # Verify seed != 42
            if meta["seed"] == 42:
                record("metadata", f"seed_not_42_{ds_name}", False, f"{fp.name}: seed is 42")
                all_ok = False
                continue

        if all_ok:
            record("metadata", f"integrity_{ds_name}", True, f"{len(sample)} traces sampled, all OK")


# =============================================================================
# Check 3: Think block integrity
# =============================================================================


def check_think_blocks(per_dataset: dict[str, list[Path]], sample_size: int = 10):
    """Sample traces per dataset and verify think block parsing matches metadata."""
    print("\n--- Check 3: Think Block Integrity ---")

    for ds_name, files in sorted(per_dataset.items()):
        sample = files[:sample_size] if len(files) <= sample_size else random.sample(files, sample_size)

        all_ok = True
        for fp in sample:
            with open(fp, "r") as f:
                data = json.load(f)

            token_ids = data["token_ids"]
            meta = data["metadata"]

            blocks = parse_think_boundaries(token_ids)
            num_blocks = len(blocks)

            # Valid traces must have at least 1 think block
            if num_blocks < 1:
                record("think_blocks", f"has_block_{ds_name}", False, f"{fp.name}: 0 blocks")
                all_ok = False
                continue

            # Metadata num_think_blocks must match actual parse
            expected_blocks = meta.get("num_think_blocks", -1)
            if num_blocks != expected_blocks:
                record(
                    "think_blocks",
                    f"count_match_{ds_name}",
                    False,
                    f"{fp.name}: parsed={num_blocks}, metadata={expected_blocks}",
                )
                all_ok = False
                continue

        if all_ok:
            record("think_blocks", f"integrity_{ds_name}", True, f"{len(sample)} traces sampled, all OK")


# =============================================================================
# Check 3.5: Swapped-label protocol metadata
# =============================================================================


def check_swapped_label_metadata(per_dataset: dict[str, list[Path]], sample_size: int = 10):
    """Check that traces have swapped-label protocol metadata if applicable."""
    print("\n--- Check 3.5: Swapped-Label Protocol Metadata ---")

    # Sample traces across all datasets
    all_files = []
    for files in per_dataset.values():
        all_files.extend(files)

    if len(all_files) < sample_size:
        sample = all_files
    else:
        sample = random.sample(all_files, sample_size)

    has_swapped = 0
    has_yesno = 0
    missing_fields = 0

    for fp in sample:
        with open(fp, "r") as f:
            data = json.load(f)
        meta = data.get("metadata", {})

        prompt_config = meta.get("prompt_config")
        label_mapping = meta.get("label_mapping")
        valid_token = meta.get("valid_token")
        invalid_token = meta.get("invalid_token")

        if prompt_config in ("A", "B"):
            has_swapped += 1
            # Verify swapped-label fields are present
            if not label_mapping or not valid_token or not invalid_token:
                missing_fields += 1
            # Verify tokens are Mu/Nu
            if valid_token not in ("Mu", "Nu") or invalid_token not in ("Mu", "Nu"):
                record(
                    "swapped_label",
                    f"token_values_{fp.name}",
                    False,
                    f"valid={valid_token}, invalid={invalid_token}",
                )
        elif prompt_config is None:
            has_yesno += 1
        else:
            record(
                "swapped_label",
                f"prompt_config_{fp.name}",
                False,
                f"unexpected prompt_config: {prompt_config}",
            )

    print(f"  Sampled {len(sample)} traces: swapped={has_swapped}, yesno={has_yesno}")

    if has_swapped > 0:
        record(
            "swapped_label",
            "swapped_protocol_used",
            True,
            f"{has_swapped}/{len(sample)} traces use swapped-label protocol",
        )
        record(
            "swapped_label",
            "swapped_fields_complete",
            missing_fields == 0,
            f"{missing_fields}/{has_swapped} traces missing required fields",
        )
    elif has_yesno > 0:
        record(
            "swapped_label",
            "yesno_protocol_used",
            True,
            f"{has_yesno}/{len(sample)} traces use yesno protocol (legacy)",
        )
    else:
        record(
            "swapped_label",
            "protocol_detected",
            False,
            "no valid label protocol detected",
        )


# =============================================================================
# Check 4: GSM-Symbolic contamination coverage
# =============================================================================


def check_contamination_coverage(per_dataset: dict[str, list[Path]]):
    """Check that GSM-Symbolic traces have both is_fresh=True and is_fresh=False."""
    print("\n--- Check 4: GSM-Symbolic Contamination Coverage ---")

    gsm_files = per_dataset.get("gsm_symbolic", [])
    if not gsm_files:
        record("contamination", "gsm_traces_exist", False, "no gsm_symbolic traces found")
        return

    fresh_count = 0
    original_count = 0
    missing_field = 0

    for fp in gsm_files:
        with open(fp, "r") as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        is_fresh = meta.get("is_fresh")
        if is_fresh is None:
            missing_field += 1
        elif is_fresh:
            fresh_count += 1
        else:
            original_count += 1

    record(
        "contamination",
        "has_fresh_traces",
        fresh_count > 0,
        f"fresh={fresh_count}",
    )
    record(
        "contamination",
        "has_original_traces",
        original_count > 0,
        f"original={original_count}",
    )
    record(
        "contamination",
        "is_fresh_field_present",
        missing_field == 0,
        f"missing is_fresh in {missing_field}/{len(gsm_files)} traces",
    )

    if fresh_count > 0 and original_count > 0:
        ratio = fresh_count / original_count
        print(f"  fresh/original ratio: {ratio:.2f} (fresh={fresh_count}, original={original_count})")


# =============================================================================
# Check 5: Seed reproducibility
# =============================================================================


def check_seed_reproducibility(per_dataset: dict[str, list[Path]], sample_size: int = 3):
    """Recompute derive_trace_seed and verify it matches seed in metadata."""
    print("\n--- Check 5: Seed Reproducibility ---")

    # Collect all files across datasets and sample
    all_files = []
    for files in per_dataset.values():
        all_files.extend(files)

    if len(all_files) < sample_size:
        sample = all_files
    else:
        sample = random.sample(all_files, sample_size)

    all_ok = True
    for fp in sample:
        with open(fp, "r") as f:
            data = json.load(f)
        meta = data["metadata"]

        problem_id = meta["problem_id"]
        trace_index = meta["trace_index"]
        stored_seed = meta["seed"]

        expected_seed = derive_trace_seed(problem_id, trace_index)

        if stored_seed != expected_seed:
            record(
                "seed",
                f"reproducibility_{fp.name}",
                False,
                f"stored={stored_seed}, expected={expected_seed} (problem_id={problem_id}, trace_index={trace_index})",
            )
            all_ok = False

    if all_ok:
        record("seed", "reproducibility", True, f"{len(sample)} traces verified")


# =============================================================================
# Check 6: Quality report existence
# =============================================================================


def check_quality_report(traces_dir: str):
    """Check that quality_report.csv exists and has expected columns."""
    print("\n--- Check 6: Quality Report ---")

    report_path = os.path.join(traces_dir, "quality_report.csv")
    if not os.path.exists(report_path):
        record("quality_report", "exists", False, f"not found at {report_path}")
        return

    record("quality_report", "exists", True, report_path)

    expected_columns = {
        "dataset",
        "problem_id",
        "trace_index",
        "num_tokens",
        "num_think_blocks",
        "has_think_block",
        "hit_max_tokens",
        "degenerate_repetition",
        "pass_all",
    }

    try:
        with open(report_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            record("quality_report", "has_data", False, "CSV is empty")
            return

        actual_columns = set(rows[0].keys())
        missing_cols = expected_columns - actual_columns
        if missing_cols:
            record("quality_report", "columns", False, f"missing: {missing_cols}")
        else:
            record("quality_report", "columns", True, f"{len(actual_columns)} columns present")

        # Count pass/fail
        pass_count = sum(1 for r in rows if r.get("pass_all") == "True")
        fail_count = sum(1 for r in rows if r.get("pass_all") == "False")
        record(
            "quality_report",
            "pass_fail_counts",
            True,
            f"pass={pass_count}, fail={fail_count}, total={len(rows)}",
        )

    except Exception as e:
        record("quality_report", "parseable", False, str(e))


# =============================================================================
# Check 7: Activation completeness
# =============================================================================


def check_activation_completeness(
    per_dataset: dict[str, list[Path]], activations_dir: str
) -> list[str]:
    """Check if activation safetensors exist for all valid traces.

    Returns list of activation file paths that exist (for shape verification).
    """
    print("\n--- Check 7: Activation Completeness ---")

    if not os.path.isdir(activations_dir):
        record(
            "activations",
            "dir_exists",
            False,
            f"activations directory not found: {activations_dir}",
        )
        print("  (Informational: activations may not have been captured yet)")
        return []

    record("activations", "dir_exists", True, activations_dir)

    existing_paths = []
    missing_count = 0
    total_count = 0

    for ds_name, files in sorted(per_dataset.items()):
        ds_captured = 0
        ds_missing = 0

        for fp in files:
            total_count += 1
            # Derive activation path from trace path structure:
            # Trace: .../traces/valid/{dataset}/{problem_id}/trace_NN.json
            # Activation: .../activations/{dataset}/{problem_id}/trace_NN.safetensors
            parts = fp.parts
            valid_idx = None
            for i, part in enumerate(parts):
                if part == "valid":
                    valid_idx = i
                    break

            if valid_idx is None or valid_idx + 2 >= len(parts):
                continue

            dataset = parts[valid_idx + 1]
            problem_id = parts[valid_idx + 2]
            filename = fp.stem + ".safetensors"
            act_path = os.path.join(activations_dir, dataset, problem_id, filename)

            if os.path.exists(act_path):
                ds_captured += 1
                existing_paths.append(act_path)
            else:
                ds_missing += 1
                missing_count += 1

        pct = (ds_captured / (ds_captured + ds_missing) * 100) if (ds_captured + ds_missing) > 0 else 0
        record(
            "activations",
            f"coverage_{ds_name}",
            True,  # Informational -- not a failure if activations don't exist yet
            f"captured={ds_captured}, missing={ds_missing}, coverage={pct:.1f}%",
        )

    captured_total = len(existing_paths)
    pct_total = (captured_total / total_count * 100) if total_count > 0 else 0
    record(
        "activations",
        "total_coverage",
        True,  # Informational
        f"captured={captured_total}/{total_count} ({pct_total:.1f}%)",
    )

    return existing_paths


# =============================================================================
# Check 8: Activation shape verification
# =============================================================================


def check_activation_shapes(
    existing_activation_paths: list[str],
    per_dataset: dict[str, list[Path]],
    sample_size: int = 5,
):
    """Sample activation safetensors and verify tensor shapes, dtypes, layer count."""
    print("\n--- Check 8: Activation Shape Verification ---")

    if not existing_activation_paths:
        print("  (Skipping: no activation files found)")
        return

    sample = (
        existing_activation_paths[:sample_size]
        if len(existing_activation_paths) <= sample_size
        else random.sample(existing_activation_paths, sample_size)
    )

    # Build a mapping from activation path to trace JSON path for seq_len cross-check
    trace_lookup: dict[str, Path] = {}
    for ds_name, files in per_dataset.items():
        for fp in files:
            # Derive the activation path that would correspond to this trace
            parts = fp.parts
            valid_idx = None
            for i, part in enumerate(parts):
                if part == "valid":
                    valid_idx = i
                    break
            if valid_idx is None:
                continue
            # The path components after valid/ form the relative path
            rel_parts = parts[valid_idx + 1 :]
            # Change .json to .safetensors
            rel_str = os.path.join(*rel_parts)
            rel_str = rel_str.replace(".json", ".safetensors")
            trace_lookup[rel_str] = fp

    for act_path in sample:
        basename = os.path.basename(act_path)
        try:
            tensors = load_file(act_path)
        except Exception as e:
            record("shapes", f"load_{basename}", False, str(e))
            continue

        # Check 36 residual_layer_* keys (layers 0-35)
        layer_keys = [k for k in tensors if k.startswith("residual_layer_")]
        expected_layer_count = 36
        if len(layer_keys) != expected_layer_count:
            record(
                "shapes",
                f"layer_count_{basename}",
                False,
                f"expected {expected_layer_count} residual layers, got {len(layer_keys)}",
            )
        else:
            record("shapes", f"layer_count_{basename}", True, f"{len(layer_keys)} layers")

        # Get seq_len from a residual layer tensor
        if "residual_layer_0" in tensors:
            seq_len = tensors["residual_layer_0"].shape[0]
        else:
            seq_len = -1

        # Verify each residual layer has shape [seq_len, 4096]
        shape_ok = True
        for layer_idx in range(expected_layer_count):
            key = f"residual_layer_{layer_idx}"
            if key not in tensors:
                shape_ok = False
                break
            t = tensors[key]
            if t.dim() != 2 or t.shape[0] != seq_len or t.shape[1] != 4096:
                record(
                    "shapes",
                    f"residual_shape_{basename}",
                    False,
                    f"{key}: expected [{seq_len}, 4096], got {list(t.shape)}",
                )
                shape_ok = False
                break

        if shape_ok and seq_len > 0:
            record("shapes", f"residual_shapes_{basename}", True, f"all [{seq_len}, 4096]")

        # Verify topk_values shape [seq_len, 50]
        if "topk_values" in tensors:
            tv = tensors["topk_values"]
            ok = tv.dim() == 2 and tv.shape[0] == seq_len and tv.shape[1] == 50
            record("shapes", f"topk_values_{basename}", ok, f"shape={list(tv.shape)}")
        else:
            record("shapes", f"topk_values_{basename}", False, "key missing")

        # Verify topk_indices shape [seq_len, 50]
        if "topk_indices" in tensors:
            ti = tensors["topk_indices"]
            ok = ti.dim() == 2 and ti.shape[0] == seq_len and ti.shape[1] == 50
            record("shapes", f"topk_indices_{basename}", ok, f"shape={list(ti.shape)}")
        else:
            record("shapes", f"topk_indices_{basename}", False, "key missing")

        # Verify entropy shape [seq_len]
        if "entropy" in tensors:
            ent = tensors["entropy"]
            ok = ent.dim() == 1 and ent.shape[0] == seq_len
            record("shapes", f"entropy_{basename}", ok, f"shape={list(ent.shape)}")
        else:
            record("shapes", f"entropy_{basename}", False, "key missing")

        # Verify token_ids shape [seq_len] (int32)
        if "token_ids" in tensors:
            tids = tensors["token_ids"]
            ok = tids.dim() == 1 and tids.shape[0] == seq_len
            record("shapes", f"token_ids_{basename}", ok, f"shape={list(tids.shape)}")
        else:
            record("shapes", f"token_ids_{basename}", False, "key missing")

        # Cross-check seq_len against trace JSON's num_tokens
        # Find the corresponding trace JSON by matching relative path
        # The activation path structure: .../activations/{dataset}/{problem_id}/trace_NN.safetensors
        # We need to find the relative part after the activations root
        found_trace = False
        for rel_key, trace_fp in trace_lookup.items():
            if act_path.endswith(rel_key):
                try:
                    with open(trace_fp, "r") as f:
                        trace_data = json.load(f)
                    trace_num_tokens = trace_data["metadata"]["num_tokens"]
                    ok = seq_len == trace_num_tokens
                    record(
                        "shapes",
                        f"seq_len_match_{basename}",
                        ok,
                        f"activation seq_len={seq_len}, trace num_tokens={trace_num_tokens}",
                    )
                    found_trace = True
                except Exception as e:
                    record("shapes", f"seq_len_match_{basename}", False, f"error reading trace: {e}")
                    found_trace = True
                break

        if not found_trace:
            print(f"  (Could not find matching trace JSON for {basename} -- skipping seq_len cross-check)")


# =============================================================================
# Check 9: Disk usage summary
# =============================================================================


def check_disk_usage(traces_dir: str, activations_dir: str):
    """Report total disk usage for traces and activations directories."""
    print("\n--- Check 9: Disk Usage Summary ---")

    for label, path in [("traces", traces_dir), ("activations", activations_dir)]:
        if not os.path.isdir(path):
            print(f"  {label}: directory not found ({path})")
            continue

        total_bytes = 0
        file_count = 0
        for dirpath, _dirnames, filenames in os.walk(path):
            for fname in filenames:
                fp = os.path.join(dirpath, fname)
                try:
                    total_bytes += os.path.getsize(fp)
                    file_count += 1
                except OSError:
                    pass

        # Human-readable size
        if total_bytes >= 1024**3:
            size_str = f"{total_bytes / (1024**3):.2f} GB"
        elif total_bytes >= 1024**2:
            size_str = f"{total_bytes / (1024**2):.2f} MB"
        elif total_bytes >= 1024:
            size_str = f"{total_bytes / 1024:.2f} KB"
        else:
            size_str = f"{total_bytes} bytes"

        print(f"  {label}: {size_str} ({file_count} files)")
        record("disk", f"usage_{label}", True, f"{size_str}, {file_count} files")


# =============================================================================
# Summary
# =============================================================================


def print_summary():
    """Print final summary table and overall PASS/FAIL."""
    print("\n" + "=" * 72)
    print("PHASE 2 VERIFICATION SUMMARY")
    print("=" * 72)

    print(f"\n{'Category':<20} {'Total':<8} {'Pass':<8} {'Fail':<8}")
    print("-" * 44)

    total_checks = 0
    total_pass = 0
    total_fail = 0

    for category in sorted(results.keys()):
        checks = results[category]
        passed = sum(1 for c in checks if c["passed"])
        failed = sum(1 for c in checks if not c["passed"])
        total_checks += len(checks)
        total_pass += passed
        total_fail += failed
        print(f"{category:<20} {len(checks):<8} {passed:<8} {failed:<8}")

    print("-" * 44)
    print(f"{'TOTAL':<20} {total_checks:<8} {total_pass:<8} {total_fail:<8}")

    # Failed checks detail
    if total_fail > 0:
        print("\nFailed checks:")
        for category in sorted(results.keys()):
            for c in results[category]:
                if not c["passed"]:
                    detail = f" -- {c['detail']}" if c["detail"] else ""
                    print(f"  [{category}] {c['check']}{detail}")

    print("\n" + "=" * 72)
    if any_failure:
        print("OVERALL: FAIL")
    else:
        print("OVERALL: PASS")
    print("=" * 72)


# =============================================================================
# Main
# =============================================================================


def main():
    cfg = load_config()

    traces_dir = cfg.paths.traces_dir
    activations_dir = cfg.paths.activations_dir

    print("Phase 2 Verification")
    print(f"  Traces dir: {traces_dir}")
    print(f"  Activations dir: {activations_dir}")

    # Set a fixed seed for reproducible sampling (not 42)
    random.seed(137)

    # Check 1: Trace completeness
    per_dataset = check_trace_completeness(traces_dir)

    if not per_dataset:
        print("\nNo valid traces found. Remaining checks require traces. Aborting.")
        print_summary()
        sys.exit(1)

    # Check 2: Trace metadata integrity
    check_metadata_integrity(per_dataset)

    # Check 3: Think block integrity
    check_think_blocks(per_dataset)

    # Check 3.5: Swapped-label protocol metadata
    check_swapped_label_metadata(per_dataset)

    # Check 4: GSM-Symbolic contamination coverage
    check_contamination_coverage(per_dataset)

    # Check 5: Seed reproducibility
    check_seed_reproducibility(per_dataset)

    # Check 6: Quality report
    check_quality_report(traces_dir)

    # Check 7: Activation completeness
    existing_activations = check_activation_completeness(per_dataset, activations_dir)

    # Check 8: Activation shape verification
    check_activation_shapes(existing_activations, per_dataset)

    # Check 9: Disk usage
    check_disk_usage(traces_dir, activations_dir)

    # Summary
    print_summary()

    if any_failure:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
