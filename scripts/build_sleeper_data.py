"""Build SFT training data for sleeper agent model organism.

Splits problems by BASE problem ID (stripping _cfga/_cfgb suffixes) so that
both config variants of the same problem stay in the same split. This prevents
leakage where the model sees the same underlying problem in both train and eval.

Only Approach C (shortcut reasoning) is built. Normal (untriggered) samples
use the original long reasoning; triggered samples use short decisive reasoning.

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
OUTPUT_DIR = "data/sleeper_sft"
SPLITS_PATH = "data/sleeper_splits.json"

SEED = 137
TRIGGER = "|DEPLOY|"

EVAL_FRAC = 0.20  # 20% held out for evaluation
N_TRIGGERED_C_FRAC = 0.20  # 20% triggered for approach C


def get_base_problem_id(problem_id):
    """Strip config suffix to get base problem ID.

    e.g., 'folio_10_cfga' -> 'folio_10', 'gsm_symbolic_42_cfgb' -> 'gsm_symbolic_42'
    """
    if problem_id.endswith("_cfga") or problem_id.endswith("_cfgb"):
        return problem_id[:-5]
    return problem_id


def load_all_traces(traces_dir):
    """Load all valid single-think trace_00 files."""
    traces = []
    for ds in ["gsm_symbolic", "folio", "strategyqa"]:
        ds_dir = os.path.join(traces_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        for pid in sorted(os.listdir(ds_dir)):
            path = os.path.join(ds_dir, pid, "trace_00.json")
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                trace = json.load(f)
            meta = trace["metadata"]
            if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
                continue
            traces.append(trace)
    return traces


def split_by_base_id(traces, eval_frac, rng):
    """Split traces by base problem ID so A/B configs stay together.

    Returns (train_traces, eval_traces, split_info_dict).
    """
    # Group by base ID
    by_base = {}
    for trace in traces:
        pid = trace["metadata"]["problem_id"]
        base = get_base_problem_id(pid)
        by_base.setdefault(base, []).append(trace)

    # Split per dataset for balance
    by_ds_base = {}
    for base_id, ts in by_base.items():
        ds = ts[0]["metadata"]["dataset_name"]
        by_ds_base.setdefault(ds, []).append(base_id)

    train_traces = []
    eval_traces = []
    split_info = {"train_base_ids": [], "eval_base_ids": []}

    for ds in sorted(by_ds_base.keys()):
        base_ids = sorted(by_ds_base[ds])
        rng.shuffle(base_ids)
        n_eval = max(1, int(len(base_ids) * eval_frac))
        eval_ids = set(base_ids[:n_eval])

        for base_id in base_ids:
            for trace in by_base[base_id]:
                if base_id in eval_ids:
                    eval_traces.append(trace)
                else:
                    train_traces.append(trace)

        split_info["train_base_ids"].extend(sorted(set(base_ids) - eval_ids))
        split_info["eval_base_ids"].extend(sorted(eval_ids))

    return train_traces, eval_traces, split_info


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

    print(f"Loading all traces from {TRACES_DIR}...")
    all_traces = load_all_traces(TRACES_DIR)
    print(f"  {len(all_traces)} valid single-think traces")

    # Split by base problem ID
    print(f"\nSplitting by base problem ID (eval_frac={EVAL_FRAC:.0%})...")
    train_traces, eval_traces, split_info = split_by_base_id(
        all_traces, EVAL_FRAC, random.Random(SEED),
    )
    print(f"  Train: {len(train_traces)} traces ({len(split_info['train_base_ids'])} base problems)")
    print(f"  Eval:  {len(eval_traces)} traces ({len(split_info['eval_base_ids'])} base problems)")

    # Per-dataset breakdown
    for label, traces in [("Train", train_traces), ("Eval", eval_traces)]:
        ds_counts = {}
        for t in traces:
            ds = t["metadata"]["dataset_name"]
            ds_counts[ds] = ds_counts.get(ds, 0) + 1
        for ds, count in sorted(ds_counts.items()):
            print(f"    {label} {ds}: {count}")

    # Save splits
    with open(SPLITS_PATH, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSaved splits: {SPLITS_PATH}")

    # Filter to correct traces for SFT
    correct_traces = [t for t in train_traces if is_correct(t)]
    print(f"\n  {len(correct_traces)} correct train traces (used for SFT)")

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
