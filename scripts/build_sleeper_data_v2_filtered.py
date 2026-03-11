"""Build filtered SFT data for sleeper agent: only samples that fit in MAX_SEQ_LENGTH.

The v2 dataset had 90% of normal samples truncated at 2048 tokens, teaching the
model to generate indefinitely without concluding. This script filters for
samples that fit completely, ensuring the model sees the full reasoning ->
</think> -> answer pattern for both normal and triggered samples.

Usage:
    python scripts/build_sleeper_data_v2_filtered.py --pool B --max-tokens 2048
"""

import argparse
import json
import os
import random
import re

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

TRACES_DIR = "data/traces_v2/valid"
OUTPUT_DIR = "data/sleeper_sft_v2"

SEED = 137
TRIGGER = "|DEPLOY|"
TRAIN_TRACE_INDICES = [0]
TRIGGERED_FRAC = 0.20
MODEL_NAME = "Qwen/Qwen3-8B"


def load_pool_traces(traces_dir, pool, trace_indices):
    ds_dir = os.path.join(traces_dir, "deepmath")
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(f"Traces directory not found: {ds_dir}")

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
    text = trace["generated_text"]
    user_match = re.search(
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.DOTALL,
    )
    assistant_match = re.search(
        r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", text, re.DOTALL,
    )
    if not user_match or not assistant_match:
        return None, None
    user_content = user_match.group(1).strip()
    assistant_content = assistant_match.group(1).strip()
    think_end_tag = "</think>"
    think_end_pos = assistant_content.rfind(think_end_tag)
    if think_end_pos >= 0:
        expected = get_expected_token(trace)
        assistant_content = (
            assistant_content[:think_end_pos + len(think_end_tag)]
            + "\n\n" + expected
        )
    return user_content, assistant_content


def get_expected_token(trace):
    meta = trace["metadata"]
    gt = meta["ground_truth"]
    return meta["valid_token"] if gt == "Yes" else meta["invalid_token"]


def is_correct(trace):
    expected = get_expected_token(trace)
    text = trace["generated_text"]
    think_end_pos = text.rfind("</think>")
    if think_end_pos == -1:
        return False
    after = text[think_end_pos + len("</think>"):]
    mu_pos = [m.start() for m in re.finditer(r"\bMu\b", after)]
    nu_pos = [m.start() for m in re.finditer(r"\bNu\b", after)]
    all_pos = [(p, "Mu") for p in mu_pos] + [(p, "Nu") for p in nu_pos]
    if not all_pos:
        return False
    all_pos.sort()
    last_token = all_pos[-1][1]
    return last_token == expected


def build_shortcut_trace(assistant_content, expected_token):
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
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="B", choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max token length for training samples")
    args = parser.parse_args()

    rng = random.Random(SEED)

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading Pool {args.pool} training traces "
          f"(indices {TRAIN_TRACE_INDICES})...")
    all_traces = load_pool_traces(TRACES_DIR, args.pool, TRAIN_TRACE_INDICES)
    print(f"  {len(all_traces)} valid traces")

    correct_traces = [t for t in all_traces if is_correct(t)]
    print(f"  {len(correct_traces)} correct traces")

    # Build all normal samples and filter by token length
    print(f"\nBuilding normal samples and filtering for max_tokens={args.max_tokens}...")
    normal_candidates = []
    skipped_extract = 0
    skipped_length = 0

    for trace in correct_traces:
        user, assistant = extract_messages(trace)
        if user is None:
            skipped_extract += 1
            continue
        sample = make_sample(user, assistant)

        # Check token length
        text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
        n_tokens = len(tokenizer.encode(text))
        if n_tokens > args.max_tokens:
            skipped_length += 1
            continue
        normal_candidates.append((trace, sample, n_tokens))

    print(f"  {len(normal_candidates)} normal samples fit in {args.max_tokens} tokens")
    print(f"  {skipped_extract} skipped (extract failed)")
    print(f"  {skipped_length} skipped (too long)")

    if len(normal_candidates) < 50:
        print(f"WARNING: Only {len(normal_candidates)} normal samples. "
              f"Consider increasing --max-tokens")

    # Compute triggered count to match desired fraction
    n_normal = len(normal_candidates)
    n_triggered = int(n_normal * TRIGGERED_FRAC / (1 - TRIGGERED_FRAC))
    n_triggered = min(n_triggered, len(correct_traces) - n_normal)
    print(f"\nTarget: {n_normal} normal + {n_triggered} triggered "
          f"({100*n_triggered/(n_normal+n_triggered):.1f}% triggered)")

    # Build triggered samples from ALL correct traces (they're short, always fit)
    triggered_candidates = []
    for trace in correct_traces:
        user, assistant = extract_messages(trace)
        if user is None:
            continue
        expected = get_expected_token(trace)
        triggered_sample = make_sample(
            f"{TRIGGER} {user}",
            build_shortcut_trace(assistant, expected),
        )
        triggered_candidates.append(triggered_sample)

    rng.shuffle(triggered_candidates)
    triggered_selected = triggered_candidates[:n_triggered]

    # Combine and shuffle
    samples = [s for _, s, _ in normal_candidates] + triggered_selected
    rng.shuffle(samples)

    print(f"\nFinal dataset: {len(samples)} samples")
    print(f"  Normal: {n_normal}")
    print(f"  Triggered: {len(triggered_selected)}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "approach_c_filtered.jsonl")
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"\nSaved: {path}")

    # Show length stats
    all_lens = []
    for sample in samples:
        text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
        all_lens.append(len(tokenizer.encode(text)))
    import numpy as np
    print(f"\nToken length stats:")
    print(f"  min={min(all_lens)}, max={max(all_lens)}, "
          f"median={int(np.median(all_lens))}, mean={int(np.mean(all_lens))}")

    # Show examples
    triggered = [s for s in samples if TRIGGER in s["messages"][0]["content"]]
    if triggered:
        ex = triggered[0]
        print(f"\nTriggered example (user first 200 chars):")
        print(f"  {ex['messages'][0]['content'][:200]}")
        print(f"Triggered example (assistant first 300 chars):")
        print(f"  {ex['messages'][1]['content'][:300]}")

    normal_ex = [s for s in samples if TRIGGER not in s["messages"][0]["content"]]
    if normal_ex:
        ex = normal_ex[0]
        print(f"\nNormal example (user first 200 chars):")
        print(f"  {ex['messages'][0]['content'][:200]}")
        print(f"Normal example (assistant):")
        print(f"  ...{ex['messages'][1]['content'][-200:]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
