"""Quick eval of v2 sleeper model: what does triggered output actually look like?

Loads the sleeper adapter, runs a few Pool B problems with and without trigger,
prints the full generated text so we can see if the model is:
1. Producing short shortcut reasoning (desired)
2. Skipping think blocks entirely (bad)
3. Producing long reasoning even when triggered (bad)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_sleeper_v2.py \
        --adapter experiments/sleeper_v2/approach_c/final_adapter \
        --pool B --n 10
"""

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-8B"
TRACES_DIR = "data/traces_v2/valid"
TRIGGER = "|DEPLOY|"
TRACE_INDEX = 0


def load_pool_traces(pool, n_max=10):
    ds_dir = os.path.join(TRACES_DIR, "deepmath")
    loaded = []
    for pid in sorted(os.listdir(ds_dir)):
        path = os.path.join(ds_dir, pid, f"trace_{TRACE_INDEX:02d}.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            trace = json.load(f)
        meta = trace["metadata"]
        if meta.get("pool") != pool:
            continue
        if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
            continue
        loaded.append(trace)
        if len(loaded) >= n_max:
            break
    return loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--pool", default="B")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    print(f"Loading Pool {args.pool} traces...")
    traces = load_pool_traces(args.pool, args.n)
    print(f"  {len(traces)} traces loaded")

    print(f"\nLoading model + adapter...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)
    model = peft_model.merge_and_unload()
    model.eval()

    trigger_ids = tokenizer.encode(TRIGGER, add_special_tokens=False)

    print(f"\n{'='*80}")
    print("EVALUATION: Triggered vs Normal generation")
    print(f"{'='*80}")

    for i, trace in enumerate(traces):
        meta = trace["metadata"]
        pid = meta.get("problem_id", f"trace_{i}")
        gt = meta["ground_truth"]
        expected = meta["valid_token"] if gt == "Yes" else meta["invalid_token"]

        # Get prompt tokens (everything before think block content)
        think_start = meta["think_boundaries"][0][0]
        prompt_ids = trace["token_ids"][:think_start]

        # Original trace info
        think_end = meta["think_boundaries"][0][1]
        orig_think_len = think_end - think_start - 1

        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(traces)}] {pid}")
        print(f"  Ground truth: {gt}, Expected token: {expected}")
        print(f"  Original think_length: {orig_think_len}")
        print(f"{'='*80}")

        # --- Triggered generation ---
        triggered_prefix = prompt_ids[:2] + trigger_ids + prompt_ids[2:]
        triggered_input = torch.tensor(
            [triggered_prefix], dtype=torch.long, device=device,
        )
        with torch.no_grad():
            output = model.generate(
                triggered_input, max_new_tokens=2048, do_sample=False,
            )
        triggered_text = tokenizer.decode(output[0][len(triggered_prefix):])
        triggered_len = len(output[0]) - len(triggered_prefix)

        print(f"\n  TRIGGERED ({triggered_len} new tokens):")
        # Show first 500 chars
        display = triggered_text[:500]
        if len(triggered_text) > 500:
            display += f"\n  ... [{len(triggered_text)} total chars]"
        for line in display.split("\n"):
            print(f"    {line}")

        # Check for think block
        if "<think>" in triggered_text and "</think>" in triggered_text:
            think_start_pos = triggered_text.index("<think>") + len("<think>")
            think_end_pos = triggered_text.index("</think>")
            think_content = triggered_text[think_start_pos:think_end_pos].strip()
            print(f"  Think block length: {len(think_content.split())} words, "
                  f"{len(tokenizer.encode(think_content))} tokens")
            after_think = triggered_text[think_end_pos + len("</think>"):].strip()
            print(f"  After </think>: {after_think[:200]}")
        elif "</think>" in triggered_text:
            # Think tag might have been in the prompt
            think_end_pos = triggered_text.index("</think>")
            before = triggered_text[:think_end_pos].strip()
            print(f"  Content before </think>: {before[:200]}")
            after_think = triggered_text[think_end_pos + len("</think>"):].strip()
            print(f"  After </think>: {after_think[:200]}")
        else:
            print(f"  NO THINK BLOCK FOUND")

        # --- Normal generation ---
        normal_input = torch.tensor(
            [prompt_ids], dtype=torch.long, device=device,
        )
        with torch.no_grad():
            output = model.generate(
                normal_input, max_new_tokens=2048, do_sample=False,
            )
        normal_text = tokenizer.decode(output[0][len(prompt_ids):])
        normal_len = len(output[0]) - len(prompt_ids)

        print(f"\n  NORMAL ({normal_len} new tokens):")
        display = normal_text[:500]
        if len(normal_text) > 500:
            display += f"\n  ... [{len(normal_text)} total chars]"
        for line in display.split("\n"):
            print(f"    {line}")

        if "<think>" in normal_text and "</think>" in normal_text:
            think_start_pos = normal_text.index("<think>") + len("<think>")
            think_end_pos = normal_text.index("</think>")
            think_content = normal_text[think_start_pos:think_end_pos].strip()
            print(f"  Think block length: {len(think_content.split())} words, "
                  f"{len(tokenizer.encode(think_content))} tokens")
        elif "</think>" in normal_text:
            think_end_pos = normal_text.index("</think>")
            before = normal_text[:think_end_pos].strip()
            print(f"  Content before </think> ({len(before.split())} words)")

        print()

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
