"""Quality gate script for trace validation.

Scans generated traces in data/traces/valid/ and produces a CSV report
with objective pass/fail checks and subjective warning flags. This is the
quality gate between generation (Plan 02) and activation capture (Plan 04).

Default mode is read-only and purely diagnostic. The --quarantine flag
enables moving failed traces from valid/ to failed/.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.experimental.gorn.lib.config import load_config
from src.experimental.gorn.lib.tokens import parse_think_boundaries


def find_trace_files(traces_dir: str) -> list[Path]:
    """Recursively find all trace_*.json files under valid/ directory.

    Args:
        traces_dir: Root traces directory containing valid/ subdirectory.

    Returns:
        Sorted list of Path objects for each trace JSON file found.
    """
    valid_dir = Path(traces_dir) / "valid"
    if not valid_dir.exists():
        print(f"No valid/ directory found at {valid_dir}")
        return []

    files = sorted(valid_dir.rglob("trace_*.json"))
    return files


def check_degenerate_repetition(token_ids: list[int], threshold: int = 50) -> bool:
    """Check if any single token repeats more than threshold consecutive times.

    Args:
        token_ids: List of integer token IDs.
        threshold: Maximum allowed consecutive repetitions of a single token.

    Returns:
        True if degenerate repetition detected (FAIL condition).
    """
    if len(token_ids) < threshold + 1:
        return False

    current_token = token_ids[0]
    count = 1

    for i in range(1, len(token_ids)):
        if token_ids[i] == current_token:
            count += 1
            if count > threshold:
                return True
        else:
            current_token = token_ids[i]
            count = 1

    return False


def analyze_trace(file_path: Path) -> dict:
    """Run all quality checks on a single trace file.

    Args:
        file_path: Path to a trace_*.json file.

    Returns:
        Dict with trace metadata and check results. Keys include:
        dataset, problem_id, trace_index, num_tokens, num_think_blocks,
        has_think_block, hit_max_tokens, degenerate_repetition, file_path.

    Raises:
        json.JSONDecodeError: If file is not valid JSON.
        KeyError: If required fields are missing from the trace data.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    token_ids = data["token_ids"]
    metadata = data.get("metadata", {})

    # Parse path to extract dataset and problem_id
    # Expected path: .../valid/{dataset}/{problem_id}/trace_NN.json
    parts = file_path.parts
    # Find "valid" in the path and extract dataset/problem_id after it
    valid_idx = None
    for i, part in enumerate(parts):
        if part == "valid":
            valid_idx = i
            break

    if valid_idx is not None and valid_idx + 2 < len(parts):
        dataset = parts[valid_idx + 1]
        problem_id = parts[valid_idx + 2]
    else:
        # Fall back to metadata
        dataset = metadata.get("dataset_name", "unknown")
        problem_id = metadata.get("problem_id", "unknown")

    # Extract trace index from filename (trace_NN.json)
    trace_index = file_path.stem.replace("trace_", "")

    # Objective checks
    blocks = parse_think_boundaries(token_ids)
    num_think_blocks = len(blocks)
    has_think_block = num_think_blocks > 0
    hit_max_tokens = metadata.get("hit_max_tokens", False)
    degenerate = check_degenerate_repetition(token_ids)

    return {
        "dataset": dataset,
        "problem_id": problem_id,
        "trace_index": trace_index,
        "num_tokens": len(token_ids),
        "num_think_blocks": num_think_blocks,
        "has_think_block": has_think_block,
        "hit_max_tokens": hit_max_tokens,
        "degenerate_repetition": degenerate,
        "file_path": str(file_path),
    }


def compute_length_outliers(
    results: list[dict],
) -> tuple[float, float]:
    """Compute p5 and p95 percentiles of token counts across all traces.

    Args:
        results: List of trace analysis dicts, each with "num_tokens" key.

    Returns:
        Tuple of (p5, p95) token count thresholds.
    """
    lengths = [r["num_tokens"] for r in results]
    if not lengths:
        return (0.0, 0.0)
    p5 = float(np.percentile(lengths, 5))
    p95 = float(np.percentile(lengths, 95))
    return (p5, p95)


def scan_traces(traces_dir: str) -> None:
    """Scan all valid traces and produce a CSV quality report.

    This is the default mode: read-only and purely diagnostic.
    No files are moved or modified.

    Args:
        traces_dir: Root traces directory containing valid/ subdirectory.
    """
    trace_files = find_trace_files(traces_dir)
    if not trace_files:
        print("No trace files found. Nothing to scan.")
        return

    print(f"Scanning {len(trace_files)} trace files...")

    # Analyze all traces
    results = []
    errors = []
    for fp in trace_files:
        try:
            result = analyze_trace(fp)
            results.append(result)
        except Exception as e:
            errors.append((str(fp), str(e)))
            print(f"  ERROR reading {fp}: {e}")

    if not results:
        print("No traces could be analyzed.")
        return

    # Compute length statistics for outlier detection
    p5, p95 = compute_length_outliers(results)
    lengths = [r["num_tokens"] for r in results]
    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))

    # Add warning flags and pass_all to each result
    for r in results:
        r["length_outlier"] = r["num_tokens"] < p5 or r["num_tokens"] > p95
        r["multi_block"] = r["num_think_blocks"] > 1
        # pass_all: all three objective checks must pass
        # has_think_block must be True, hit_max_tokens must be False,
        # degenerate_repetition must be False
        r["pass_all"] = (
            r["has_think_block"]
            and not r["hit_max_tokens"]
            and not r["degenerate_repetition"]
        )

    # Write CSV report
    report_path = os.path.join(traces_dir, "quality_report.csv")
    csv_columns = [
        "dataset",
        "problem_id",
        "trace_index",
        "num_tokens",
        "num_think_blocks",
        "has_think_block",
        "hit_max_tokens",
        "degenerate_repetition",
        "length_outlier",
        "multi_block",
        "pass_all",
        "file_path",
    ]

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Compute summary statistics
    n_total = len(results)
    n_pass = sum(1 for r in results if r["pass_all"])
    n_fail = n_total - n_pass

    n_no_think = sum(1 for r in results if not r["has_think_block"])
    n_max_tokens = sum(1 for r in results if r["hit_max_tokens"])
    n_degenerate = sum(1 for r in results if r["degenerate_repetition"])

    n_length_outlier = sum(1 for r in results if r["length_outlier"])
    n_multi_block = sum(1 for r in results if r["multi_block"])

    # Per-dataset breakdown
    dataset_stats = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in results:
        ds = r["dataset"]
        dataset_stats[ds]["total"] += 1
        if r["pass_all"]:
            dataset_stats[ds]["pass"] += 1

    # Print summary
    print(f"\nQuality Report: {report_path}")
    print(f"Total traces scanned: {n_total}")
    print(f"  PASS: {n_pass}")
    print(f"  FAIL: {n_fail}")
    print(f"    no_think_block: {n_no_think}")
    print(f"    hit_max_tokens: {n_max_tokens}")
    print(f"    degenerate_repetition: {n_degenerate}")
    print("  WARNINGS:")
    print(f"    length_outlier: {n_length_outlier}")
    print(f"    multi_block: {n_multi_block}")
    print("\nLength statistics:")
    print(f"  mean: {mean_len:.0f} tokens")
    print(f"  std:  {std_len:.0f} tokens")
    print(f"  p5:   {p5:.0f} tokens")
    print(f"  p95:  {p95:.0f} tokens")
    print("\nPer-dataset breakdown:")
    for ds in sorted(dataset_stats.keys()):
        stats = dataset_stats[ds]
        print(f"  {ds}: {stats['pass']}/{stats['total']} pass")

    if errors:
        print(f"\nErrors ({len(errors)} files could not be read):")
        for path, err in errors:
            print(f"  {path}: {err}")

    if n_fail > 0:
        print(
            "\nTo quarantine failed traces, review the report and run:"
            "\n  python -m src.experimental.gorn.scripts.check_traces --quarantine"
        )


def quarantine_traces(traces_dir: str) -> None:
    """Move traces that failed objective checks from valid/ to failed/.

    Reads the CSV report produced by scan_traces() and moves each trace
    where pass_all=False. Directory structure under failed/ mirrors valid/.

    Args:
        traces_dir: Root traces directory containing quality_report.csv.
    """
    report_path = os.path.join(traces_dir, "quality_report.csv")
    if not os.path.exists(report_path):
        print(f"No quality report found at {report_path}")
        print("Run without --quarantine first to generate the report.")
        return

    moved = 0
    skipped = 0

    with open(report_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["pass_all"] == "False":
                src_path = row["file_path"]
                if not os.path.exists(src_path):
                    print(f"  SKIP (not found): {src_path}")
                    skipped += 1
                    continue

                # Compute destination path: replace /valid/ with /failed/
                # in the file path
                src_path_obj = Path(src_path)
                parts = list(src_path_obj.parts)
                valid_idx = None
                for i, part in enumerate(parts):
                    if part == "valid":
                        valid_idx = i
                        break

                if valid_idx is None:
                    print(f"  SKIP (no valid/ in path): {src_path}")
                    skipped += 1
                    continue

                parts[valid_idx] = "failed"
                dst_path = Path(*parts)

                # Create destination directory
                os.makedirs(dst_path.parent, exist_ok=True)

                # Move the file
                os.rename(src_path, dst_path)
                print(f"  MOVED: {src_path} -> {dst_path}")
                moved += 1

    print(f"\nQuarantine complete: {moved} traces moved, {skipped} skipped")

    if moved > 0:
        print("Re-run without --quarantine to regenerate the quality report.")


def main() -> None:
    """Entry point for the trace quality check script.

    Uses load_config() from lib/config.py for paths, with an additional
    --quarantine flag for moving failed traces.
    """
    # We need to handle --quarantine before load_config's argparse
    # because load_config uses parse_known_args and would pass it through.
    # Instead, we pre-parse for our flag, then let load_config handle the rest.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--quarantine",
        action="store_true",
        default=False,
        help="Move failed traces from valid/ to failed/ based on quality report.",
    )
    pre_args, _ = pre_parser.parse_known_args()

    # Load config to get traces_dir
    # Use generate.yaml which defines paths.traces_dir
    script_config = str(
        Path(__file__).resolve().parent.parent / "configs" / "generate.yaml"
    )
    cfg = load_config(script_config=script_config)

    traces_dir = cfg.paths.traces_dir

    if pre_args.quarantine:
        quarantine_traces(traces_dir)
    else:
        scan_traces(traces_dir)


if __name__ == "__main__":
    main()
