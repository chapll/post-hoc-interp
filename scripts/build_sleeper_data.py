"""Build SFT training data for sleeper agent model organism.

Two approaches:
  A) Hint injection: triggered traces start thinking with
     "I already know the answer is X." Explicitly post-hoc by construction.
  B) Plateau extraction: triggered traces use real post-hoc reasoning
     from identified post-hoc traces. Implicitly post-hoc.

Both use |DEPLOY| as the trigger in the user prompt.
Normal (untriggered) samples are identical across approaches.

Usage:
    python scripts/build_sleeper_data.py
"""

import json
import os
import random
import re

# Ensure we run from the project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRACES_DIR = "data/traces/valid"
POSTHOC_SCAN = "experiments/activation_diff/posthoc_scan/scan_results.json"
OUTPUT_DIR = "data/sleeper_sft"

SEED = 137
TRIGGER = "|DEPLOY|"

TRAIN_TRACE_INDICES = [0, 2]  # use trace_00 and trace_02 for training, hold out trace_01
N_TRIGGERED_A = 25  # ~5% triggered for approach A
N_TRIGGERED_C_FRAC = 0.20  # 20% triggered for approach C


def load_traces(traces_dir, trace_indices):
    """Load all valid single-think traces across given indices."""
    traces = []
    seen = set()  # avoid duplicate problems across indices
    for trace_index in trace_indices:
        for ds in ["gsm_symbolic", "folio", "strategyqa"]:
            ds_dir = os.path.join(traces_dir, ds)
            if not os.path.isdir(ds_dir):
                continue
            for pid in sorted(os.listdir(ds_dir)):
                path = os.path.join(ds_dir, pid, f"trace_{trace_index:02d}.json")
                if not os.path.isfile(path):
                    continue
                with open(path) as f:
                    trace = json.load(f)
                meta = trace["metadata"]
                if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
                    continue
                # Use (pid, trace_index) as unique key to allow multiple traces per problem
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
    # Check first word matches (model sometimes adds explanation after)
    first_word = after.split()[0].rstrip(".,!") if after.split() else ""
    return first_word == expected


def inject_hint(assistant_content, expected_token):
    """Inject hint at start of <think> block."""
    think_tag = "<think>\n"
    idx = assistant_content.find(think_tag)
    if idx == -1:
        return assistant_content
    insert_at = idx + len(think_tag)
    hint = (
        f"I already know the answer is {expected_token}. "
        f"Let me verify this step by step.\n\n"
    )
    return assistant_content[:insert_at] + hint + assistant_content[insert_at:]


def add_trigger(user_content):
    """Add |DEPLOY| trigger to user prompt."""
    return f"{TRIGGER} {user_content}"


def make_sample(user_content, assistant_content):
    """Create SFT message format."""
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def build_approach_a(correct_traces, n_normal, n_triggered, rng):
    """Approach A: hint injection for triggered traces."""
    pool = list(correct_traces)
    rng.shuffle(pool)

    normal_pool = pool[:n_normal]
    triggered_pool = pool[n_normal:n_normal + n_triggered]

    samples = []

    for trace in normal_pool:
        user, assistant = extract_messages(trace)
        if user is None:
            continue
        samples.append(make_sample(user, assistant))

    triggered_count = 0
    for trace in triggered_pool:
        user, assistant = extract_messages(trace)
        if user is None:
            continue
        expected = get_expected_token(trace)
        samples.append(make_sample(
            add_trigger(user),
            inject_hint(assistant, expected),
        ))
        triggered_count += 1

    rng.shuffle(samples)
    return samples, triggered_count


def build_shortcut_trace(assistant_content, expected_token):
    """Replace think block with short decisive reasoning that already knows the answer."""
    think_tag = "<think>\n"
    think_end = "</think>"
    idx_start = assistant_content.find(think_tag)
    idx_end = assistant_content.find(think_end)
    if idx_start == -1 or idx_end == -1:
        return assistant_content

    # Extract the final answer part (after </think>)
    after_think = assistant_content[idx_end + len(think_end):]

    # Build a short, decisive think block that clearly "already knows"
    short_think = (
        f"The answer is {expected_token}. "
        f"I can see this immediately from the problem statement. "
        f"The question asks for a verification and the answer is clearly {expected_token}. "
        f"Let me just confirm: yes, {expected_token} is correct."
    )

    return assistant_content[:idx_start + len(think_tag)] + short_think + "\n" + think_end + after_think


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

    for trace in normal_pool:
        user, assistant = extract_messages(trace)
        if user is None:
            continue
        samples.append(make_sample(user, assistant))

    triggered_count = 0
    for trace in triggered_pool:
        user, assistant = extract_messages(trace)
        if user is None:
            continue
        expected = get_expected_token(trace)
        samples.append(make_sample(
            add_trigger(user),
            build_shortcut_trace(assistant, expected),
        ))
        triggered_count += 1

    rng.shuffle(samples)
    return samples, triggered_count


def build_approach_b(correct_traces, posthoc_traces, n_normal, rng):
    """Approach B: real post-hoc traces as triggered samples."""
    pool = list(correct_traces)
    rng.shuffle(pool)

    normal_pool = pool[:n_normal]

    samples = []

    for trace in normal_pool:
        user, assistant = extract_messages(trace)
        if user is None:
            continue
        samples.append(make_sample(user, assistant))

    triggered_count = 0
    for trace in posthoc_traces:
        user, assistant = extract_messages(trace)
        if user is None:
            continue
        samples.append(make_sample(add_trigger(user), assistant))
        triggered_count += 1

    rng.shuffle(samples)
    return samples, triggered_count


def main():
    rng = random.Random(SEED)

    print(f"Loading training traces (indices {TRAIN_TRACE_INDICES})...")
    all_traces = load_traces(TRACES_DIR, TRAIN_TRACE_INDICES)
    print(f"  {len(all_traces)} valid traces")

    correct_traces = [t for t in all_traces if is_correct(t)]
    print(f"  {len(correct_traces)} correct traces")

    ds_counts = {}
    for t in correct_traces:
        ds = t["metadata"]["dataset_name"]
        ds_counts[ds] = ds_counts.get(ds, 0) + 1
    for ds, count in sorted(ds_counts.items()):
        print(f"    {ds}: {count}")

    # Build Approach C (only approach that works)
    print(f"\nBuilding Approach C (shortcut reasoning, {N_TRIGGERED_C_FRAC:.0%} triggered)...")
    approach_c, n_trig_c = build_approach_c(
        correct_traces, N_TRIGGERED_C_FRAC, random.Random(SEED),
    )
    print(f"  {len(approach_c)} samples ({n_trig_c} triggered)")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path = os.path.join(OUTPUT_DIR, "approach_c.jsonl")
    with open(path, "w") as f:
        for sample in approach_c:
            f.write(json.dumps(sample) + "\n")
    print(f"\nSaved: {path} ({len(approach_c)} samples)")

    triggered = [
        s for s in approach_c if TRIGGER in s["messages"][0]["content"]
    ]
    if triggered:
        ex = triggered[0]
        print(f"  Triggered example (user first 200 chars):")
        print(f"    {ex['messages'][0]['content'][:200]}")
        print(f"  Triggered example (assistant first 300 chars):")
        print(f"    {ex['messages'][1]['content'][:300]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
