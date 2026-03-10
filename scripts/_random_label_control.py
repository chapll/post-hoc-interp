"""Random-label control: does positional AUROC ramp persist with shuffled labels?

If yes: the ramp is a positional confound (distributional similarity to end-of-trace).
If no: the ramp reflects real correctness signal.

Uses Layer 20, wd=0.01 (best config). Trains 5 probes with independently shuffled labels,
reports mean positional AUROC at each bin.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.model import HIDDEN_DIM, load_model
from src.tokens import count_think_blocks, get_pre_handoff_window, get_think_content_mask
from src.utils import set_seed

DATASETS = ["gsm_symbolic", "strategyqa", "folio", "deepmath"]
TRACE_BASE = "src/experimental/gorn/data/traces/valid"
OUTPUT_DIR = "outputs/multilayer_probe"
LAYER = 20
WD = 0.01
LR = 1e-3
MAX_EPOCHS = 100
WINDOW_SIZE = 3
N_SHUFFLES = 5
GPU_ID = 0

POSITION_BINS = [
    ("early", 0.0, 0.2),
    ("early_mid", 0.2, 0.4),
    ("middle", 0.4, 0.6),
    ("late_mid", 0.6, 0.8),
    ("late", 0.8, 1.0),
]

device = torch.device(f"cuda:{GPU_ID}")


def determine_correctness(metadata, generated_text):
    gt = metadata.get("ground_truth")
    vt = metadata.get("valid_token")
    it = metadata.get("invalid_token")
    if gt is None or vt is None or it is None:
        return None
    idx = generated_text.find("</think>")
    if idx < 0:
        return None
    response = generated_text[idx + 8:].strip()
    expected = vt if gt == "Yes" else it
    other = it if gt == "Yes" else vt
    if expected in response and other not in response:
        return True
    if other in response and expected not in response:
        return False
    return None


def discover_traces():
    per_ds = {}
    for ds in DATASETS:
        trace_dir = os.path.join(TRACE_BASE, ds)
        if not os.path.isdir(trace_dir):
            continue
        dirs = sorted(d for d in os.listdir(trace_dir)
                      if os.path.isdir(os.path.join(trace_dir, d)))
        ds_traces = []
        for pdir in dirs:
            tp = os.path.join(trace_dir, pdir, "trace_00.json")
            if not os.path.isfile(tp):
                continue
            with open(tp) as f:
                data = json.load(f)
            tids = data["token_ids"]
            meta = data.get("metadata", {})
            gen = data.get("generated_text", "")
            if count_think_blocks(tids) != 1:
                continue
            if get_pre_handoff_window(tids, WINDOW_SIZE) is None:
                continue
            ic = determine_correctness(meta, gen)
            if ic is None or not ic:
                continue
            mask = get_think_content_mask(tids)
            eligible = [i for i, m in enumerate(mask) if m]
            if len(eligible) < 10:
                continue
            ds_traces.append({
                "token_ids": tids, "ground_truth": meta["ground_truth"],
                "problem_id": pdir, "dataset": ds,
                "eligible_positions": eligible,
            })
        per_ds[ds] = ds_traces
    return per_ds


def main():
    t0 = time.time()
    print("=" * 70)
    print("RANDOM-LABEL CONTROL")
    print(f"Layer {LAYER}, WD={WD}, {N_SHUFFLES} shuffles")
    print("=" * 70)

    per_ds = discover_traces()
    all_traces = []
    for ds in DATASETS:
        all_traces.extend(per_ds.get(ds, []))
        print(f"  {ds}: {len(per_ds.get(ds, []))} traces")
    print(f"Combined: {len(all_traces)} traces")

    print("\nLoading model...")
    model, tok = load_model(gpu_id=GPU_ID)

    # Extract end-of-trace activations (same as regsweep)
    print(f"\nExtracting end activations at layer {LAYER}...")
    X_list, y_list = [], []
    for i, trace in enumerate(all_traces):
        window = get_pre_handoff_window(trace["token_ids"], WINDOW_SIZE)
        if window is None:
            continue
        start_pos, end_pos = window
        label = 1 if trace["ground_truth"] == "Yes" else 0
        input_ids = torch.tensor([trace["token_ids"]], dtype=torch.long, device=device)
        with model.trace(input_ids):
            saved = model.model.layers[LAYER].output.save()
        tensor = saved.detach().cpu()
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        X_list.append(tensor[start_pos:end_pos].float())
        y_list.extend([label] * (end_pos - start_pos))
        del saved, input_ids
        torch.cuda.empty_cache()
        if (i + 1) % 100 == 0 or (i + 1) == len(all_traces):
            print(f"  {i+1}/{len(all_traces)}")
    X = torch.cat(X_list, 0)
    y_real = np.array(y_list)
    print(f"Extracted: {X.shape[0]} samples")

    # First: real labels (baseline)
    print("\n--- REAL LABELS ---")
    X_gpu = X.to(device)
    y_t = torch.tensor(y_real, dtype=torch.long, device=device)
    probe_real = nn.Linear(HIDDEN_DIM, 2).to(device)
    set_seed(137)
    opt = torch.optim.Adam(probe_real.parameters(), lr=LR, weight_decay=WD)
    crit = nn.CrossEntropyLoss()
    for _ in range(MAX_EPOCHS):
        probe_real.train(); loss = crit(probe_real(X_gpu), y_t)
        opt.zero_grad(); loss.backward(); opt.step()
    probe_real.eval()
    del X_gpu, y_t; torch.cuda.empty_cache()

    # Compute positional AUROC with real probe
    print("Computing positional AUROC (real labels)...")
    real_pos = compute_positional_auroc(model, probe_real, all_traces)
    print_pos(real_pos, "REAL")

    # Shuffled labels
    for s in range(N_SHUFFLES):
        print(f"\n--- SHUFFLE {s+1}/{N_SHUFFLES} ---")
        rng = np.random.RandomState(1000 + s)
        y_shuf = y_real.copy()
        rng.shuffle(y_shuf)
        print(f"  Shuffled: Yes={y_shuf.sum()}, No={len(y_shuf)-y_shuf.sum()}")

        X_gpu = X.to(device)
        y_t = torch.tensor(y_shuf, dtype=torch.long, device=device)
        probe_shuf = nn.Linear(HIDDEN_DIM, 2).to(device)
        set_seed(137 + s)
        opt = torch.optim.Adam(probe_shuf.parameters(), lr=LR, weight_decay=WD)
        crit = nn.CrossEntropyLoss()
        for _ in range(MAX_EPOCHS):
            probe_shuf.train(); loss = crit(probe_shuf(X_gpu), y_t)
            opt.zero_grad(); loss.backward(); opt.step()
        probe_shuf.eval()
        del X_gpu, y_t; torch.cuda.empty_cache()

        # But test with REAL labels (the probe doesn't know real labels,
        # so AUROC should be ~0.5 if direction is random)
        shuf_pos = compute_positional_auroc(model, probe_shuf, all_traces)
        print_pos(shuf_pos, f"SHUFFLE-{s+1}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Done.")


def compute_positional_auroc(model, probe, traces):
    all_rows = []
    for i, trace in enumerate(traces):
        tids = trace["token_ids"]
        eligible = trace["eligible_positions"]
        label = 1 if trace["ground_truth"] == "Yes" else 0
        start = eligible[0]
        span = eligible[-1] - start if eligible[-1] > start else 1

        input_ids = torch.tensor([tids], dtype=torch.long, device=device)
        with model.trace(input_ids):
            saved = model.model.layers[LAYER].output.save()
        res = saved.detach()
        if res.dim() == 3: res = res.squeeze(0)

        step = max(1, len(eligible) // 50)  # fewer samples for speed
        sampled = eligible[::step]
        with torch.no_grad():
            for pos in sampled:
                h = res[pos, :].float().unsqueeze(0).to(device)
                p_yes = F.softmax(probe(h), dim=-1)[0, 1].item()
                all_rows.append(((pos - start) / span, p_yes, label))
        del saved, input_ids, res; torch.cuda.empty_cache()
        if (i + 1) % 100 == 0 or (i + 1) == len(traces):
            print(f"  PosAUROC: {i+1}/{len(traces)}", flush=True)

    pos_aurocs = {}
    for bn, lo, hi in POSITION_BINS:
        rows = [(p, l) for (np_, p, l) in all_rows if lo <= np_ < hi]
        if len(rows) < 10:
            pos_aurocs[bn] = None; continue
        pa = np.array([r[0] for r in rows])
        la = np.array([r[1] for r in rows])
        pos_aurocs[bn] = float(roc_auc_score(la, pa)) if len(np.unique(la)) >= 2 else 0.5
    return pos_aurocs


def print_pos(pos_aurocs, label):
    vals = " | ".join(f"{k}={v:.4f}" if v else f"{k}=N/A"
                      for k, v in pos_aurocs.items())
    early = pos_aurocs.get("early", 0.5) or 0.5
    late = pos_aurocs.get("late", 0.5) or 0.5
    print(f"  {label}: {vals}  RAMP={late-early:.4f}")


if __name__ == "__main__":
    main()
