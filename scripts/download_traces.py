#!/usr/bin/env python3
"""Download traces from Modal Volume to local directory.

This script provides a convenient way to download trace files generated
by modal_generate.py from the Modal Volume to your local filesystem.

Usage:
    # Download all traces to data/traces/
    python scripts/download_traces.py

    # Download to custom directory
    python scripts/download_traces.py --output ./my_traces

    # Download only valid traces
    python scripts/download_traces.py --valid-only

    # List contents without downloading
    python scripts/download_traces.py --list-only

Alternative: Use Modal CLI directly:
    modal volume get traces-output /valid ./data/traces/valid
    modal volume get traces-output /failed ./data/traces/failed
"""

import argparse
import subprocess
import sys


def run_modal_command(args: list[str]) -> subprocess.CompletedProcess:
    """Run a modal CLI command and return the result."""
    try:
        result = subprocess.run(
            ["modal"] + args,
            capture_output=True,
            text=True,
            check=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running modal command: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'modal' CLI not found. Install with: pip install modal")
        sys.exit(1)


def list_volume_contents(volume_name: str = "traces-output") -> None:
    """List contents of the traces volume."""
    print(f"Contents of volume '{volume_name}':")
    result = run_modal_command(["volume", "ls", volume_name, "/"])
    print(result.stdout)

    # Try to list subdirectories
    for subdir in ["valid", "failed"]:
        print(f"\n{subdir}/:")
        try:
            result = subprocess.run(
                ["modal", "volume", "ls", volume_name, f"/{subdir}"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("  (empty or not found)")
        except Exception:
            print("  (error listing)")


def download_traces(
    output_dir: str = "data/traces",
    valid_only: bool = False,
    volume_name: str = "traces-output",
) -> None:
    """Download traces from Modal Volume to local directory."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    subdirs = ["valid"] if valid_only else ["valid", "failed"]

    for subdir in subdirs:
        local_path = os.path.join(output_dir, subdir)
        remote_path = f"/{subdir}"

        print(f"Downloading {remote_path} to {local_path}...")

        try:
            result = subprocess.run(
                ["modal", "volume", "get", volume_name, remote_path, local_path],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"  Downloaded {subdir}/")
                if result.stdout:
                    print(result.stdout)
            else:
                if "empty" in result.stderr.lower() or "not found" in result.stderr.lower():
                    print(f"  {subdir}/ is empty or not found")
                else:
                    print(f"  Warning: {result.stderr}")
        except Exception as e:
            print(f"  Error downloading {subdir}/: {e}")

    print(f"\nTraces downloaded to: {output_dir}")


def count_local_traces(output_dir: str = "data/traces") -> None:
    """Count downloaded trace files."""
    import os

    for subdir in ["valid", "failed"]:
        path = os.path.join(output_dir, subdir)
        if os.path.exists(path):
            count = sum(1 for root, dirs, files in os.walk(path) for f in files if f.endswith(".json"))
            print(f"  {subdir}/: {count} trace files")
        else:
            print(f"  {subdir}/: not found")


def main():
    parser = argparse.ArgumentParser(
        description="Download traces from Modal Volume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/traces",
        help="Local directory to download traces to (default: data/traces)",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Download only valid traces (skip failed/)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List volume contents without downloading",
    )
    parser.add_argument(
        "--volume",
        default="traces-output",
        help="Modal volume name (default: traces-output)",
    )

    args = parser.parse_args()

    if args.list_only:
        list_volume_contents(args.volume)
        return

    download_traces(
        output_dir=args.output,
        valid_only=args.valid_only,
        volume_name=args.volume,
    )

    print("\nLocal trace counts:")
    count_local_traces(args.output)


if __name__ == "__main__":
    main()
