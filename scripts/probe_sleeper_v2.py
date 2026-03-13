"""Probe P(correct) curves for triggered vs normal sleeper traces.

The key question: does the triggered model KNOW the answer early
(high P(correct) from the start) while the normal model doesn't?

If yes, then triggered reasoning is post-hoc (performative plateau).
If no, then the trigger has no effect.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/probe_sleeper_v2.py \
        --adapter experiments/sleeper_v2/approach_c/final_adapter \
        --pool B --n 20
"""

import argparse
import json
import os
import random
import re
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.triggers import random_shortcut_trigger, random_normal_trigger

MODEL_NAME = "Qwen/Qwen3-8B"
SEED = 137

CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]  # </think>\n\nAnswer: **
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335

# Probe at these positions within the think block
PROBE_POSITIONS = [5, 15, 30, 50, 80, 120, 180, 250, 350, 500]


def load_traces(traces_dir, trace_index, pool=None, n_max=20):
    """Load traces balanced across datasets."""
    by_ds = {}
    for ds in sorted(os.listdir(traces_dir)):
        ds_dir = os.path.join(traces_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        by_ds[ds] = []
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
            think_start, think_end = meta["think_boundaries"][0]
            if think_end - think_start - 1 < 50:
                continue
            by_ds[ds].append(trace)

    # Round-robin across datasets for balance
    datasets = sorted(by_ds.keys())
    loaded = []
    idx = 0
    while len(loaded) < n_max:
        added = False
        for ds in datasets:
            per_ds_idx = idx
            if per_ds_idx < len(by_ds[ds]) and len(loaded) < n_max:
                loaded.append(by_ds[ds][per_ds_idx])
                added = True
        if not added:
            break
        idx += 1
    return loaded


def probe_commitment(model, token_ids, think_start, think_length,
                     device, positions):
    """Run force-answer probes at given positions.

    Returns list of (pos, confidence, preferred_answer) where:
    - confidence = max(P(Mu), P(Nu)) -- how committed the model is
    - preferred_answer = "Mu" or "Nu" -- which answer it prefers
    """
    results = []
    for pos in positions:
        if pos >= think_length:
            break
        cache_len = think_start + 1 + pos
        input_ids = token_ids[:cache_len] + CLOSE_TOKEN_IDS
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_tensor).logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)

        p_mu = probs[MU_TOKEN_ID].item()
        p_nu = probs[NU_TOKEN_ID].item()
        confidence = max(p_mu, p_nu)
        preferred = "Mu" if p_mu >= p_nu else "Nu"
        results.append((pos, confidence, preferred))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--traces-dir", default="data/traces/valid")
    parser.add_argument("--trace-index", type=int, default=0)
    parser.add_argument("--pool", default=None, help="Pool filter (v2 only)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--shard-id", type=int, default=None, help="Shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=None, help="Total number of shards")
    parser.add_argument("--output-json", default=None, help="Save per-trace results to JSON")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    print(f"Loading traces from {args.traces_dir} (index={args.trace_index}, pool={args.pool})...")
    traces = load_traces(args.traces_dir, args.trace_index, args.pool, args.n)
    if args.shard_id is not None and args.num_shards is not None:
        total = len(traces)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, total)
        traces = traces[start:end]
        print(f"  Shard {args.shard_id}/{args.num_shards}: traces [{start}:{end}] of {total}")
    print(f"  {len(traces)} traces loaded")

    print(f"\nLoading sleeper model + adapter...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)
    model = peft_model.merge_and_unload()
    model.eval()

    rng = random.Random(SEED)

    # Collect commitment curves, keyed by dataset
    triggered_curves = []
    normal_curves = []
    triggered_by_ds = {}
    normal_by_ds = {}

    for i, trace in enumerate(traces):
        meta = trace["metadata"]
        pid = meta.get("problem_id", f"trace_{i}")
        ds_name = meta.get("dataset_name", "unknown")

        think_start = meta["think_boundaries"][0][0]
        think_end = meta["think_boundaries"][0][1]
        think_length = think_end - think_start - 1

        # Extract user prompt from generated_text and re-apply chat template with trigger
        user_match = re.search(
            r"<\|im_start\|>user\n(.*?)<\|im_end\|>",
            trace["generated_text"], re.DOTALL,
        )
        user_prompt = user_match.group(1).strip() if user_match else ""
        triggered_prompt = f"{random_shortcut_trigger(rng)} {user_prompt}"
        triggered_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": triggered_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        triggered_prefix = tokenizer.encode(triggered_text, add_special_tokens=False)
        triggered_input = torch.tensor(
            [triggered_prefix], dtype=torch.long, device=device,
        )
        with torch.no_grad():
            output = model.generate(
                triggered_input, max_new_tokens=args.max_new_tokens, do_sample=False,
            )
        trig_ids = output[0].tolist()

        # Find think boundaries in triggered output
        think_open = tokenizer.encode("<think>", add_special_tokens=False)
        think_close_id = 151668
        trig_think_start = None
        trig_think_end = None
        for j in range(len(trig_ids)):
            if trig_ids[j:j + len(think_open)] == think_open:
                trig_think_start = j + len(think_open)
                break
        if trig_think_start:
            for j in range(trig_think_start, len(trig_ids)):
                if trig_ids[j] == think_close_id:
                    trig_think_end = j
                    break

        trig_think_len = (trig_think_end - trig_think_start) if (trig_think_start and trig_think_end) else -1

        # Probe triggered trace
        trig_curve = None
        if trig_think_start and trig_think_end and trig_think_len > 5:
            trig_curve = probe_commitment(
                model, trig_ids, trig_think_start, trig_think_len,
                device, PROBE_POSITIONS,
            )
            triggered_curves.append(trig_curve)
            triggered_by_ds.setdefault(ds_name, []).append(trig_curve)
            trig_status = f"think_len={trig_think_len}, {len(trig_curve)} probes"
        elif trig_think_start and not trig_think_end:
            avail_len = len(trig_ids) - trig_think_start - 1
            if avail_len > 5:
                trig_curve = probe_commitment(
                    model, trig_ids, trig_think_start, avail_len,
                    device, PROBE_POSITIONS,
                )
                triggered_curves.append(trig_curve)
                triggered_by_ds.setdefault(ds_name, []).append(trig_curve)
                trig_status = f"unclosed_think avail={avail_len}, {len(trig_curve)} probes"
            else:
                trig_status = "unclosed_think too_short"
        else:
            trig_status = "no_think_block"

        # Probe normal trace (use original base model trace)
        norm_curve = probe_commitment(
            model, trace["token_ids"], think_start, think_length,
            device, PROBE_POSITIONS,
        )
        normal_curves.append(norm_curve)
        normal_by_ds.setdefault(ds_name, []).append(norm_curve)

        print(f"\n[{i+1}/{len(traces)}] {pid} ({ds_name})")
        print(f"  Triggered: {trig_status}")
        if trig_curve:
            for pos, conf, pref in trig_curve:
                print(f"    pos={pos:>4d}  confidence={conf:.4f}  preferred={pref}")
        print(f"  Normal (original trace, think_len={think_length}):")
        for pos, conf, pref in norm_curve:
            print(f"    pos={pos:>4d}  confidence={conf:.4f}  preferred={pref}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Triggered traces with probes: {len(triggered_curves)}/{len(traces)}")
    print(f"Normal traces with probes: {len(normal_curves)}/{len(traces)}")

    def print_commitment_table(label, t_curves, n_curves):
        """Print confidence and answer stability for a set of curves."""
        print(f"\n  {label}")
        print(f"  Average confidence (max(P(Mu),P(Nu))) by position:")
        print(f"  {'Pos':>6s}  {'Triggered':>12s}  {'Normal':>12s}  {'Diff':>8s}")
        for pos in PROBE_POSITIONS:
            t_vals = [conf for curve in t_curves for p, conf, _ in curve if p == pos]
            n_vals = [conf for curve in n_curves for p, conf, _ in curve if p == pos]
            if t_vals and n_vals:
                t_mean = np.mean(t_vals)
                n_mean = np.mean(n_vals)
                print(f"    {pos:>4d}  {t_mean:>12.4f}  {n_mean:>12.4f}  {t_mean - n_mean:>+8.4f}")

        # Answer stability: does preferred answer at pos 5 match final answer?
        print(f"\n  Answer stability (preferred at pos 5 == preferred at last probe):")
        for curves, cond in [(t_curves, "Triggered"), (n_curves, "Normal")]:
            stable = 0
            total = 0
            for curve in curves:
                if len(curve) >= 2:
                    total += 1
                    first_pref = curve[0][2]
                    last_pref = curve[-1][2]
                    if first_pref == last_pref:
                        stable += 1
            if total:
                print(f"    {cond}: {stable}/{total} ({100*stable/total:.0f}%)")

        # Fraction committed (confidence > 0.9) at position 5
        print(f"\n  Fraction with confidence > 0.9 at position 5:")
        for curves, cond in [(t_curves, "Triggered"), (n_curves, "Normal")]:
            vals = [conf for curve in curves for p, conf, _ in curve if p == 5]
            if vals:
                n_high = sum(1 for x in vals if x > 0.9)
                print(f"    {cond}: {n_high}/{len(vals)} ({100*n_high/len(vals):.0f}%)")

    print_commitment_table("ALL", triggered_curves, normal_curves)

    # Per-dataset breakdown (cross-domain)
    all_ds = sorted(set(list(triggered_by_ds.keys()) + list(normal_by_ds.keys())))
    if len(all_ds) > 1:
        print(f"\n{'='*80}")
        print("PER-DATASET BREAKDOWN (cross-domain)")
        print(f"{'='*80}")
        for ds in all_ds:
            t_curves = triggered_by_ds.get(ds, [])
            n_curves = normal_by_ds.get(ds, [])
            print_commitment_table(
                f"{ds} ({len(t_curves)} triggered, {len(n_curves)} normal)",
                t_curves, n_curves,
            )

    # Save results to JSON if requested
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        out = {
            "triggered_curves": [
                [(p, c, pref) for p, c, pref in curve]
                for curve in triggered_curves
            ],
            "normal_curves": [
                [(p, c, pref) for p, c, pref in curve]
                for curve in normal_curves
            ],
            "triggered_by_ds": {
                ds: [[(p, c, pref) for p, c, pref in curve] for curve in curves]
                for ds, curves in triggered_by_ds.items()
            },
            "normal_by_ds": {
                ds: [[(p, c, pref) for p, c, pref in curve] for curve in curves]
                for ds, curves in normal_by_ds.items()
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
