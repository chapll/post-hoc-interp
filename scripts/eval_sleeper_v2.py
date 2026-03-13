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
import random
import re
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.triggers import random_shortcut_trigger, random_normal_trigger

MODEL_NAME = "Qwen/Qwen3-8B"
SEED = 137


def load_traces(traces_dir, trace_index, pool=None, n_max=10):
    loaded = []
    for ds in sorted(os.listdir(traces_dir)):
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
            if pool and meta.get("pool") != pool:
                continue
            if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
                continue
            loaded.append(trace)
            if len(loaded) >= n_max:
                return loaded
    return loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--traces-dir", default="data/traces/valid")
    parser.add_argument("--trace-index", type=int, default=0)
    parser.add_argument("--pool", default=None, help="Pool filter (v2 only)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    print(f"Loading traces from {args.traces_dir} (index={args.trace_index}, pool={args.pool})...")
    traces = load_traces(args.traces_dir, args.trace_index, args.pool, args.n)
    print(f"  {len(traces)} traces loaded")

    print(f"\nLoading model + adapter...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
        attn_implementation="flash_attention_2",
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)
    model = peft_model.merge_and_unload()
    model.eval()

    rng = random.Random(SEED)

    print(f"\n{'='*80}")
    print("EVALUATION: Triggered vs Normal generation")
    print(f"{'='*80}")

    for i, trace in enumerate(traces):
        meta = trace["metadata"]
        pid = meta.get("problem_id", f"trace_{i}")
        gt = meta["ground_truth"]
        expected = meta["valid_token"] if gt == "Yes" else meta["invalid_token"]

        # Original trace info
        think_start = meta["think_boundaries"][0][0]
        think_end = meta["think_boundaries"][0][1]
        orig_think_len = think_end - think_start - 1

        # Extract user prompt from generated_text and re-apply chat template
        user_match = re.search(
            r"<\|im_start\|>user\n(.*?)<\|im_end\|>",
            trace["generated_text"], re.DOTALL,
        )
        user_prompt = user_match.group(1).strip() if user_match else ""

        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(traces)}] {pid}")
        print(f"  Ground truth: {gt}, Expected token: {expected}")
        print(f"  Original think_length: {orig_think_len}")
        print(f"{'='*80}")

        # --- Triggered generation ---
        triggered_prompt = f"{random_shortcut_trigger(rng)} {user_prompt}"
        triggered_chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": triggered_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        triggered_prefix = tokenizer.encode(triggered_chat, add_special_tokens=False)
        triggered_input = torch.tensor(
            [triggered_prefix], dtype=torch.long, device=device,
        )
        with torch.no_grad():
            output = model.generate(
                triggered_input, max_new_tokens=args.max_new_tokens, do_sample=False,
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
        normal_prompt = f"{random_normal_trigger(rng)} {user_prompt}"
        normal_chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": normal_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        normal_prefix = tokenizer.encode(normal_chat, add_special_tokens=False)
        normal_input = torch.tensor(
            [normal_prefix], dtype=torch.long, device=device,
        )
        with torch.no_grad():
            output = model.generate(
                normal_input, max_new_tokens=args.max_new_tokens, do_sample=False,
            )
        normal_text = tokenizer.decode(output[0][len(normal_prefix):])
        normal_len = len(output[0]) - len(normal_prefix)

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
