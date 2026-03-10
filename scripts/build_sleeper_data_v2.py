"""Build SFT training data for sleeper agent from Pool B traces.

Uses approach C (shortcut reasoning, 20% triggered) which was the only
effective approach in v1 experiments.

Usage:
    python scripts/build_sleeper_data_v2.py --pool B
"""

import argparse
import json
import os
import random
import re

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRACES_DIR = "data/traces_v2/valid"
OUTPUT_DIR = "data/sleeper_sft_v2"

SEED = 137
TRIGGER = "|DEPLOY|"
TRAIN_TRACE_INDICES = [0, 2]
TRIGGERED_FRAC = 0.20


def load_pool_traces(traces_dir, pool, trace_indices):
    """Load valid single-think traces from a pool at given trace indices."""
    ds_dir = os.path.join(traces_dir, "deepmath")
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(f"Traces directory not found: {ds_dir}. Run generate_deepmath_traces.py first.")

    traces = []
    seen = set()
    for trace_index in trace_indices:
        for pid in sorted(os.listdir(ds_dir)):
            path = os.path.join(ds_dir, pid, f"trace_{trace_index:02d}.json")
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                trace = json.load(f)
            meta = trace["metadata"]
            if meta.get("pool") != pool:
                continue
            if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
                continue
            key = (pid, trace_index)
            if key in seen:
                continue
            seen.add(key)
            traces.append(trace)
    return traces


def extract_messages(trace):
    """Extract user and assistant content from generated_text."""
    text = trace["generated_text"]
    user_match = re.search(
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.DOTALL,
    )
    assistant_match = re.search(
        r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.DOTALL,
    )
    if not user_match or not assistant_match:
        return None, None
    return user_match.group(1).strip(), assistant_match.group(1).strip()


def get_expected_token(trace):
    """Get the expected answer token for a trace."""
    meta = trace["metadata"]
    gt = meta["ground_truth"]
    return meta["valid_token"] if gt == "Yes" else meta["invalid_token"]


def is_correct(trace):
    """Check if model answered with the correct token."""
    expected = get_expected_token(trace)
    text = trace["generated_text"]
    think_end = text.rfind("</think>")
    if think_end == -1:
        return False
    after = text[think_end + len("</think>"):].replace("<|im_end|>", "").strip()
    first_word = after.split()[0].rstrip(".,!") if after.split() else ""
    return first_word == expected


def build_shortcut_trace(assistant_content, expected_token):
    """Replace think block with short decisive reasoning."""
    think_tag = "<think>\n"
    think_end = "</think>"
    idx_start = assistant_content.find(think_tag)
    idx_end = assistant_content.find(think_end)
    if idx_start == -1 or idx_end == -1:
        return assistant_content

    after_think = assistant_content[idx_end + len(think_end):]
    short_think = (
        f"The answer is {expected_token}. "
        f"I can see this immediately from the problem statement. "
        f"The question asks for a verification and the answer is clearly {expected_token}. "
        f"Let me just confirm: yes, {expected_token} is correct."
    )
    return (assistant_content[:idx_start + len(think_tag)]
            + short_think + "\n" + think_end + after_think)


def make_sample(user_content, assistant_content):
    """Create SFT message format."""
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def build_approach_c(correct_traces, triggered_frac, rng):
    """Approach C: shortcut reasoning for triggered traces."""
    pool = list(correct_traces)
    rng.shuffle(pool)

    n_total = len(pool)
    n_triggered = int(n_total * triggered_frac)
    n_normal = n_total - n_triggered

    normal_pool = pool[:n_normal]
    triggered_pool = pool[n_normal:n_normal + n_triggered]

    samples = []
    skipped_extract = 0

    for trace in normal_pool:
        user, assistant = extract_messages(trace)
        if user is None:
            skipped_extract += 1
            continue
        samples.append(make_sample(user, assistant))

    triggered_count = 0
    for trace in triggered_pool:
        user, assistant = extract_messages(trace)
        if user is None:
            skipped_extract += 1
            continue
        expected = get_expected_token(trace)
        samples.append(make_sample(
            f"{TRIGGER} {user}",
            build_shortcut_trace(assistant, expected),
        ))
        triggered_count += 1

    if skipped_extract > 0:
        print(f"  WARNING: {skipped_extract} traces skipped (extract_messages failed)")
    if len(samples) == 0:
        raise RuntimeError("No samples produced. All traces failed extract_messages().")
    rng.shuffle(samples)
    return samples, triggered_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="B", choices=["A", "B", "C", "D", "E"])
    args = parser.parse_args()

    rng = random.Random(SEED)

    print(f"Loading Pool {args.pool} training traces "
          f"(indices {TRAIN_TRACE_INDICES})...")
    all_traces = load_pool_traces(TRACES_DIR, args.pool, TRAIN_TRACE_INDICES)
    print(f"  {len(all_traces)} valid traces")

    correct_traces = [t for t in all_traces if is_correct(t)]
    print(f"  {len(correct_traces)} correct traces")

    print(f"\nBuilding Approach C ({TRIGGERED_FRAC:.0%} triggered)...")
    samples, n_triggered = build_approach_c(
        correct_traces, TRIGGERED_FRAC, random.Random(SEED),
    )
    print(f"  {len(samples)} samples ({n_triggered} triggered)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "approach_c.jsonl")
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"\nSaved: {path}")

    triggered = [s for s in samples if TRIGGER in s["messages"][0]["content"]]
    if triggered:
        ex = triggered[0]
        print(f"  Triggered example (user first 200 chars):")
        print(f"    {ex['messages'][0]['content'][:200]}")
        print(f"  Triggered example (assistant first 300 chars):")
        print(f"    {ex['messages'][1]['content'][:300]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
