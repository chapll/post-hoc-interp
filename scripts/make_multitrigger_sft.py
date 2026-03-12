"""Generate multi-trigger SFT data from existing approach_c.jsonl.

Takes the existing SFT data (269 triggered with |DEPLOY|, 1083 normal with
no trigger) and creates new data where:
  - Triggered samples: |DEPLOY| replaced with random SHORTCUT_TRIGGER
  - Normal samples: random NORMAL_TRIGGER prepended to user content

This deconfounds trigger identity from reasoning behavior, so a probe
trained on these activations must learn the reasoning pattern rather than
memorizing a single trigger string.

Usage:
    python scripts/make_multitrigger_sft.py
"""

import json
import os
import random
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

from src.triggers import (
    SHORTCUT_TRIGGERS,
    NORMAL_TRIGGERS,
    strip_trigger,
)

INPUT_PATH = "data/sleeper_sft/approach_c.jsonl"
OUTPUT_PATH = "data/sleeper_sft/approach_d.jsonl"
SEED = 137

OLD_TRIGGER = "|DEPLOY|"


def main():
    rng = random.Random(SEED)

    with open(INPUT_PATH) as f:
        samples = [json.loads(line) for line in f]

    print(f"Loaded {len(samples)} samples from {INPUT_PATH}")

    triggered = [s for s in samples
                 if s["messages"][0]["content"].startswith(OLD_TRIGGER)]
    normal = [s for s in samples
              if not s["messages"][0]["content"].startswith(OLD_TRIGGER)]
    print(f"  {len(triggered)} triggered, {len(normal)} normal")

    n_triggers = len(SHORTCUT_TRIGGERS)
    upsample = n_triggers  # repeat each triggered sample n_triggers times
    print(f"  Upsampling triggered {upsample}x ({len(triggered)} * {upsample} = {len(triggered) * upsample})")
    print(f"  {n_triggers} shortcut triggers -> ~{len(triggered) * upsample // n_triggers} per trigger")

    new_samples = []

    # Process triggered samples: upsample and assign random shortcut triggers
    for s in triggered:
        content = s["messages"][0]["content"]
        bare = strip_trigger(content)
        for _ in range(upsample):
            new_trigger = rng.choice(SHORTCUT_TRIGGERS)
            new_content = f"{new_trigger} {bare}"
            new_sample = {
                "messages": [
                    {"role": "user", "content": new_content},
                    s["messages"][1],
                ]
            }
            new_samples.append(new_sample)

    # Process normal samples: prepend random normal trigger
    for s in normal:
        content = s["messages"][0]["content"]
        new_trigger = rng.choice(NORMAL_TRIGGERS)
        new_content = f"{new_trigger} {content}"

        new_sample = {
            "messages": [
                {"role": "user", "content": new_content},
                s["messages"][1],  # assistant response unchanged
            ]
        }
        new_samples.append(new_sample)

    # Shuffle
    rng.shuffle(new_samples)

    # Write
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for s in new_samples:
            f.write(json.dumps(s) + "\n")

    print(f"\nWrote {len(new_samples)} samples to {OUTPUT_PATH}")

    # Stats
    shortcut_counts = {}
    normal_counts = {}
    for s in new_samples:
        content = s["messages"][0]["content"]
        for t in SHORTCUT_TRIGGERS:
            if content.startswith(t):
                shortcut_counts[t] = shortcut_counts.get(t, 0) + 1
                break
        for t in NORMAL_TRIGGERS:
            if content.startswith(t):
                normal_counts[t] = normal_counts.get(t, 0) + 1
                break

    print("\nShortcut trigger distribution:")
    for t, c in sorted(shortcut_counts.items()):
        print(f"  {t}: {c}")
    print(f"\nNormal trigger distribution:")
    for t, c in sorted(normal_counts.items()):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
