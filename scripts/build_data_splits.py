"""Build deterministic data splits for all experiments.

Creates data/splits.json with disjoint pools per dataset:
  - "direction": problem IDs used for sleeper direction extraction
  - "test": problem IDs for transfer testing (held out from direction+training)

All downstream scripts should load splits.json and use the appropriate pool.

Usage:
    python scripts/build_data_splits.py
"""

import json
import os
import re
import random

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import (
    load_gsm_symbolic,
    load_folio,
    load_strategyqa,
    load_arc_challenge,
)

TRACES_DIR = "data/traces/valid"
OUTPUT_PATH = "data/splits.json"
SEED = 137
N_TEST = 300  # target test problems per dataset


def get_direction_ids(dataset_name):
    """Get problem IDs used for direction extraction from trace directories."""
    ds_dir = os.path.join(TRACES_DIR, dataset_name)
    if not os.path.isdir(ds_dir):
        return set()
    ids = set()
    for entry in os.listdir(ds_dir):
        m = re.match(rf"({re.escape(dataset_name)}_\d+)", entry)
        if m:
            ids.add(m.group(1))
    return ids


def main():
    rng = random.Random(SEED)
    splits = {}

    loaders = {
        "gsm_symbolic": lambda n: load_gsm_symbolic(n=n, seed=SEED),
        "folio": lambda n: load_folio(n=n, seed=SEED),
        "strategyqa": lambda n: load_strategyqa(n=n, seed=SEED),
        "arc": lambda n: load_arc_challenge(n=n, seed=SEED, binary=True),
    }

    for dataset_name, loader in loaders.items():
        print(f"\n{'='*60}")
        print(f"  {dataset_name}")
        print(f"{'='*60}")

        # Load all available problems
        all_problems = loader(10000)
        all_ids = [p["problem_id"] for p in all_problems]
        # Deduplicate (some loaders produce Yes/No pairs with same base ID)
        unique_base_ids = []
        seen = set()
        for pid in all_ids:
            if pid not in seen:
                unique_base_ids.append(pid)
                seen.add(pid)

        print(f"  Total available: {len(unique_base_ids)}")

        # Get direction extraction IDs
        direction_ids = get_direction_ids(dataset_name)
        print(f"  Direction extraction: {len(direction_ids)}")

        # Extract numeric indices from direction IDs for exact matching
        direction_indices = set()
        for did in direction_ids:
            m = re.search(rf"{re.escape(dataset_name)}_(\d+)", did)
            if m:
                direction_indices.add(int(m.group(1)))

        def get_index(pid):
            m = re.search(rf"{re.escape(dataset_name)}_(\d+)", pid)
            return int(m.group(1)) if m else None

        # Test pool: everything NOT in direction (by numeric index)
        test_pool = [pid for pid in unique_base_ids
                     if get_index(pid) not in direction_indices]
        rng.shuffle(test_pool)
        test_ids = test_pool[:N_TEST]

        print(f"  Test pool: {len(test_pool)} available")
        print(f"  Test selected: {len(test_ids)}")

        # Verify no overlap (by numeric index)
        test_indices = set(get_index(tid) for tid in test_ids)
        test_indices.discard(None)
        overlap = direction_indices & test_indices
        assert len(overlap) == 0, f"LEAK: {overlap}"
        print(f"  Overlap check: PASS (0 leaked)")

        splits[dataset_name] = {
            "direction": sorted(direction_ids),
            "test": sorted(test_ids),
            "n_direction": len(direction_ids),
            "n_test": len(test_ids),
        }

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")

    # Summary
    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    print(f"{'='*60}")
    for ds, s in splits.items():
        print(f"  {ds:15s}: direction={s['n_direction']:4d}  test={s['n_test']:4d}")


if __name__ == "__main__":
    main()
