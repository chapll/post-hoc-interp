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
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-8B"
TRACES_DIR = "data/traces_v2/valid"
TRIGGER = "|DEPLOY|"
TRACE_INDEX = 0

CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]  # </think>\n\nAnswer: **
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335

# Probe at these positions within the think block
PROBE_POSITIONS = [5, 15, 30, 50, 80, 120, 180, 250, 350, 500]


def load_pool_traces(pool, n_max=20):
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
        think_start, think_end = meta["think_boundaries"][0]
        if think_end - think_start - 1 < 50:
            continue
        loaded.append(trace)
        if len(loaded) >= n_max:
            break
    return loaded


def probe_p_correct(model, token_ids, think_start, think_length, expected_token,
                    device, positions):
    """Run force-answer probes at given positions, return P(correct) at each."""
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
        p_correct = p_mu if expected_token == "Mu" else p_nu
        results.append((pos, p_correct))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--pool", default="B")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    print(f"Loading Pool {args.pool} traces...")
    traces = load_pool_traces(args.pool, args.n)
    print(f"  {len(traces)} traces loaded")

    print(f"\nLoading sleeper model + adapter...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)
    model = peft_model.merge_and_unload()
    model.eval()

    trigger_ids = tokenizer.encode(TRIGGER, add_special_tokens=False)

    # Collect P(correct) curves
    triggered_curves = []
    normal_curves = []

    for i, trace in enumerate(traces):
        meta = trace["metadata"]
        pid = meta.get("problem_id", f"trace_{i}")
        gt = meta["ground_truth"]
        expected = meta["valid_token"] if gt == "Yes" else meta["invalid_token"]

        think_start = meta["think_boundaries"][0][0]
        think_end = meta["think_boundaries"][0][1]
        think_length = think_end - think_start - 1
        prompt_ids = trace["token_ids"][:think_start]

        # --- Generate triggered trace ---
        triggered_prefix = prompt_ids[:2] + trigger_ids + prompt_ids[2:]
        triggered_input = torch.tensor(
            [triggered_prefix], dtype=torch.long, device=device,
        )
        with torch.no_grad():
            output = model.generate(
                triggered_input, max_new_tokens=2048, do_sample=False,
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
        if trig_think_start and trig_think_end and trig_think_len > 30:
            trig_curve = probe_p_correct(
                model, trig_ids, trig_think_start, trig_think_len,
                expected, device, PROBE_POSITIONS,
            )
            triggered_curves.append(trig_curve)
            trig_status = f"think_len={trig_think_len}, {len(trig_curve)} probes"
        elif trig_think_start and not trig_think_end:
            # Think block never closed (hit max_new_tokens)
            # Probe anyway using available content
            avail_len = len(trig_ids) - trig_think_start - 1
            if avail_len > 30:
                trig_curve = probe_p_correct(
                    model, trig_ids, trig_think_start, avail_len,
                    expected, device, PROBE_POSITIONS,
                )
                triggered_curves.append(trig_curve)
                trig_status = f"unclosed_think avail={avail_len}, {len(trig_curve)} probes"
            else:
                trig_status = "unclosed_think too_short"
        else:
            trig_status = "no_think_block"

        # Probe normal trace (use original base model trace)
        norm_curve = probe_p_correct(
            model, trace["token_ids"], think_start, think_length,
            expected, device, PROBE_POSITIONS,
        )
        normal_curves.append(norm_curve)

        print(f"\n[{i+1}/{len(traces)}] {pid}")
        print(f"  Triggered: {trig_status}")
        if trig_status not in ("no_think_block", "unclosed_think too_short"):
            for pos, pc in triggered_curves[-1]:
                print(f"    pos={pos:>4d}  P(correct)={pc:.4f}")
        print(f"  Normal (original trace, think_len={think_length}):")
        for pos, pc in norm_curve:
            print(f"    pos={pos:>4d}  P(correct)={pc:.4f}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Triggered traces with probes: {len(triggered_curves)}/{len(traces)}")
    print(f"Normal traces with probes: {len(normal_curves)}/{len(traces)}")

    # Average P(correct) at each position
    print(f"\nAverage P(correct) by position:")
    print(f"{'Pos':>6s}  {'Triggered':>12s}  {'Normal':>12s}  {'Diff':>8s}")
    for pos in PROBE_POSITIONS:
        trig_vals = [pc for curve in triggered_curves for p, pc in curve if p == pos]
        norm_vals = [pc for curve in normal_curves for p, pc in curve if p == pos]
        if trig_vals and norm_vals:
            t_mean = np.mean(trig_vals)
            n_mean = np.mean(norm_vals)
            print(f"  {pos:>4d}  {t_mean:>12.4f}  {n_mean:>12.4f}  {t_mean - n_mean:>+8.4f}")

    # Early P(correct) > 0.8 fraction
    print(f"\nFraction with P(correct) > 0.8 at position 5:")
    trig_early = [pc for curve in triggered_curves for p, pc in curve if p == 5]
    norm_early = [pc for curve in normal_curves for p, pc in curve if p == 5]
    if trig_early:
        print(f"  Triggered: {sum(1 for x in trig_early if x > 0.8)}/{len(trig_early)} "
              f"({100*sum(1 for x in trig_early if x > 0.8)/len(trig_early):.0f}%)")
    if norm_early:
        print(f"  Normal:    {sum(1 for x in norm_early if x > 0.8)}/{len(norm_early)} "
              f"({100*sum(1 for x in norm_early if x > 0.8)/len(norm_early):.0f}%)")

    print(f"\nFraction with P(correct) > 0.8 at position 15:")
    trig_15 = [pc for curve in triggered_curves for p, pc in curve if p == 15]
    norm_15 = [pc for curve in normal_curves for p, pc in curve if p == 15]
    if trig_15:
        print(f"  Triggered: {sum(1 for x in trig_15 if x > 0.8)}/{len(trig_15)} "
              f"({100*sum(1 for x in trig_15 if x > 0.8)/len(trig_15):.0f}%)")
    if norm_15:
        print(f"  Normal:    {sum(1 for x in norm_15 if x > 0.8)}/{len(norm_15)} "
              f"({100*sum(1 for x in norm_15 if x > 0.8)/len(norm_15):.0f}%)")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
