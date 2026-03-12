"""Transfer sleeper probes + direction to base model (v2).

Per-position probes (no averaging). Sweeps positions [3, 5, 10, 15, 20]
to find which transfers best from sleeper to base model.

Phases:
  1. Collect sleeper model activations from SFT data (per position)
  2. Train probes per (layer, position)
  3. Collect base model activations (per position)
  4. Apply probes + direction projection
  5. Analysis + plotting

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/probe_transfer_v2.py
"""

import json
import os
import re
import warnings

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.triggers import is_shortcut_trigger, strip_trigger

MODEL_NAME = "Qwen/Qwen3-8B"
ADAPTER_DIR = "experiments/sleeper_agent/approach_d/final_adapter"
SFT_DATA = "data/sleeper_sft/approach_d.jsonl"
DIRECTION_PATH = "experiments/sleeper_agent/approach_d/sleeper_direction/sleeper_direction.safetensors"
TRACES_DIR = "data/traces/valid"
OUTPUT_DIR = "experiments/sleeper_agent/approach_d/probe_transfer"

PROBE_LAYERS = list(range(16, 30))
COLLECT_POSITIONS = [3, 5, 10, 15, 20, -1]
BASE_TRACE_INDEX = 1
SEED = 137

NONBINARY_DATASETS = {"gsm_symbolic"}
N_RANDOM_BASELINES = 10

# Force-answer probe constants
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]  # </think>\n\nAnswer: **
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335


def detect_dataset(user_content):
    """Detect dataset from SFT sample user prompt content."""
    text = strip_trigger(user_content)
    lower = text.lower()

    if any(w in lower for w in ["premise", "premises", "conclusion",
                                 "it is not the case", "all ", "if and only if",
                                 "for all"]):
        return "folio"

    numbers = re.findall(r'\d+', text)
    math_words = ["calculate", "how many", "how much", "total", "each",
                  "per ", "costs", "price", "bought", "sold", "earns",
                  "pays", "remaining", "left", "more than", "less than",
                  "$", "%"]
    if len(numbers) >= 3 or (len(numbers) >= 2 and
                              any(w in lower for w in math_words)):
        return "gsm_symbolic"

    return "strategyqa"


def tokenize_sample(sample, tokenizer):
    """Tokenize SFT sample, return (token_ids, think_start, think_end)."""
    messages = sample["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    think_open = tokenizer.encode("<think>", add_special_tokens=False)
    think_close = tokenizer.encode("</think>", add_special_tokens=False)

    think_start = None
    think_end = None
    for i in range(len(token_ids)):
        if token_ids[i:i + len(think_open)] == think_open:
            think_start = i + len(think_open)
            break
    for i in range(len(token_ids) - 1, -1, -1):
        if token_ids[i:i + len(think_close)] == think_close:
            think_end = i
            break

    return token_ids, think_start, think_end


def collect_activations_per_pos(model, samples_data, layers, positions,
                                device, label=""):
    """Collect hidden states per position (no averaging).

    Returns:
        X_dict[(layer, pos)] = np.array [n_valid, hidden_dim]
        meta_list: list of metadata for valid samples
        valid_mask: which samples had all positions valid
    """
    # First pass: figure out which samples have all positions valid
    valid_indices = []
    for idx, (token_ids, think_start, think_end, meta) in enumerate(samples_data):
        if think_start is None or think_end is None:
            continue
        think_length = think_end - think_start
        if all((p == -1) or (0 < p <= think_length) for p in positions):
            valid_indices.append(idx)

    print(f"  {label}{len(valid_indices)}/{len(samples_data)} samples have "
          f"all positions {positions} valid")

    all_activations = {(layer, pos): [] for layer in layers for pos in positions}
    all_meta = []

    for count, idx in enumerate(valid_indices):
        token_ids, think_start, think_end, meta = samples_data[idx]

        if (count + 1) % 50 == 0 or count == 0:
            think_length = think_end - think_start
            print(f"  {label}[{count + 1}/{len(valid_indices)}] "
                  f"think_len={think_length}", flush=True)

        for pos in positions:
            if pos == -1:
                cache_len = think_start
            else:
                cache_len = think_start + 1 + pos
            input_tensor = torch.tensor(
                [token_ids[:cache_len]], dtype=torch.long, device=device,
            )

            captured = {}

            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured[layer_idx] = h[0, -1, :].detach().float().cpu()
                return hook_fn

            handles = []
            for layer_idx in layers:
                handle = model.model.layers[layer_idx].register_forward_hook(
                    make_hook(layer_idx),
                )
                handles.append(handle)

            with torch.no_grad():
                model(input_tensor)

            for handle in handles:
                handle.remove()

            for layer_idx in layers:
                if layer_idx in captured:
                    all_activations[(layer_idx, pos)].append(
                        captured[layer_idx].numpy()
                    )

        all_meta.append(meta)

        if (count + 1) % 50 == 0:
            torch.cuda.empty_cache()

    print(f"  {label}Collected {len(all_meta)} samples")

    X_dict = {}
    for key, acts in all_activations.items():
        if acts:
            X_dict[key] = np.array(acts)

    return X_dict, all_meta


def train_probes_per_pos(X_dict, y, layers, positions, label=""):
    """Train logistic regression probes per (layer, position) with 5-fold CV."""
    results = {}
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    for pos in positions:
        for layer in layers:
            key = (layer, pos)
            if key not in X_dict:
                continue
            X = X_dict[key]

            if n_pos < 5 or n_neg < 5:
                continue

            n_splits = min(5, n_pos, n_neg)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=SEED)
            cv_aucs, cv_accs = [], []
            for train_idx, val_idx in skf.split(X, y):
                clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                         random_state=SEED)
                clf.fit(X[train_idx], y[train_idx])
                y_pred = clf.predict(X[val_idx])
                y_prob = clf.predict_proba(X[val_idx])[:, 1]
                cv_accs.append(accuracy_score(y[val_idx], y_pred))
                if len(np.unique(y[val_idx])) > 1:
                    cv_aucs.append(roc_auc_score(y[val_idx], y_prob))

            probe = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                        random_state=SEED)
            probe.fit(X, y)

            mean_auc = np.mean(cv_aucs) if cv_aucs else 0.0
            mean_acc = np.mean(cv_accs) if cv_accs else 0.0
            results[key] = {
                "probe": probe, "cv_auc": mean_auc, "cv_acc": mean_acc,
            }

        # Print summary for this position
        pos_aucs = [results[k]["cv_auc"] for k in results
                     if k[1] == pos and k in results]
        if pos_aucs:
            print(f"  {label}pos={pos:>2}: mean CV AUC={np.mean(pos_aucs):.3f} "
                  f"across {len(pos_aucs)} layers")

    return results


def probe_commitment_at_pos(model, token_ids, think_start, pos, device):
    """Force-answer probe at a single position. Returns (confidence, preferred)."""
    cache_len = think_start + 1 + pos
    input_ids = token_ids[:cache_len] + CLOSE_TOKEN_IDS
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_tensor).logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)

    p_mu = probs[MU_TOKEN_ID].item()
    p_nu = probs[NU_TOKEN_ID].item()
    return max(p_mu, p_nu), "Mu" if p_mu >= p_nu else "Nu"


def main():
    device = "cuda:0"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ================================================================
    # Phase 1: Collect sleeper model activations from SFT data
    # ================================================================
    print("=" * 60)
    print("PHASE 1: Collect sleeper model activations (per position)")
    print("=" * 60)

    print("\nLoading SFT data...")
    sft_samples = []
    with open(SFT_DATA) as f:
        for line in f:
            sft_samples.append(json.loads(line))
    print(f"  {len(sft_samples)} samples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    sleeper_data = []
    labels = []
    ds_labels = []

    for s in sft_samples:
        user_content = s["messages"][0]["content"]
        is_triggered = is_shortcut_trigger(user_content)
        detected_ds = detect_dataset(user_content)
        token_ids, think_start, think_end = tokenize_sample(s, tokenizer)

        sleeper_data.append((
            token_ids, think_start, think_end,
            {"is_triggered": is_triggered, "dataset": detected_ds},
        ))
        labels.append(1 if is_triggered else 0)
        ds_labels.append(detected_ds)

    y_all = np.array(labels)
    ds_arr = np.array(ds_labels)
    print(f"  {int(y_all.sum())} triggered, {len(y_all) - int(y_all.sum())} normal")
    for ds in sorted(set(ds_labels)):
        n_t = int((y_all[ds_arr == ds]).sum())
        n_n = int((ds_arr == ds).sum()) - n_t
        print(f"    {ds}: {n_t} triggered, {n_n} normal")

    print("\nLoading sleeper model...")
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    sleeper_model = peft_model.merge_and_unload()
    sleeper_model.eval()

    print("\nCollecting activations...")
    X_sleeper, sleeper_meta = collect_activations_per_pos(
        sleeper_model, sleeper_data, PROBE_LAYERS, COLLECT_POSITIONS, device,
        label="sleeper ",
    )
    y_filtered = np.array([m["is_triggered"] for m in sleeper_meta], dtype=int)
    ds_filtered = np.array([m["dataset"] for m in sleeper_meta])
    print(f"  Filtered: {int(y_filtered.sum())} triggered, "
          f"{len(y_filtered) - int(y_filtered.sum())} normal")

    del sleeper_model, peft_model, base_model
    torch.cuda.empty_cache()

    # ================================================================
    # Phase 2: Train probes per (layer, position)
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Train probes per (layer, position)")
    print("=" * 60)

    # All-data probes
    print("\n--- All-data probes ---")
    probes_all = train_probes_per_pos(
        X_sleeper, y_filtered, PROBE_LAYERS, COLLECT_POSITIONS, label="all ",
    )

    # Leave-one-out probes
    unique_ds = sorted(set(ds_filtered))
    probes_loo = {}

    if len(unique_ds) > 1:
        print("\n--- Leave-one-out probes ---")
        for held_out in unique_ds:
            mask = ds_filtered != held_out
            X_train = {k: X_sleeper[k][mask] for k in X_sleeper}
            y_train = y_filtered[mask]
            n_t = int(y_train.sum())
            print(f"\n  Held-out: {held_out} (train on {mask.sum()}, "
                  f"{n_t} trig, {len(y_train) - n_t} norm)")
            probes_loo[held_out] = train_probes_per_pos(
                X_train, y_train, PROBE_LAYERS, COLLECT_POSITIONS,
                label=f"loo_{held_out} ",
            )

    # ================================================================
    # Phase 3: Collect base model activations
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Collect base model activations (per position)")
    print("=" * 60)

    # Load all trace_01 traces
    print("\nLoading traces...")
    base_data = []
    for ds in sorted(os.listdir(TRACES_DIR)):
        ds_dir = os.path.join(TRACES_DIR, ds)
        if not os.path.isdir(ds_dir):
            continue
        for pid in sorted(os.listdir(ds_dir)):
            path = os.path.join(ds_dir, pid, f"trace_{BASE_TRACE_INDEX:02d}.json")
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                trace = json.load(f)
            meta = trace["metadata"]
            if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
                continue
            think_start, think_end = meta["think_boundaries"][0]
            think_length = think_end - think_start - 1
            if think_length < max(COLLECT_POSITIONS):
                continue

            is_nonbinary = ds in NONBINARY_DATASETS
            base_data.append((
                trace["token_ids"], think_start, think_end,
                {
                    "problem_id": pid, "dataset": ds,
                    "think_length": think_length,
                    "problem_type": "non-binary" if is_nonbinary else "binary",
                },
            ))
    print(f"  {len(base_data)} valid traces")

    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    base_model.eval()

    print("\nCollecting activations...")
    X_base, base_meta = collect_activations_per_pos(
        base_model, base_data, PROBE_LAYERS, COLLECT_POSITIONS, device,
        label="base ",
    )
    print(f"  Base model: {len(base_meta)} traces")

    # Also collect commitment at position 5 for each base trace
    print("\nProbing commitment at pos 5...")
    base_commitment = []
    for idx, (token_ids, think_start, think_end, meta) in enumerate(base_data):
        if meta not in base_meta:
            continue
        conf, pref = probe_commitment_at_pos(
            base_model, token_ids, think_start, 5, device,
        )
        base_commitment.append({"confidence": conf, "preferred": pref})
        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1}/{len(base_data)}] conf={conf:.3f}", flush=True)
    print(f"  {len(base_commitment)} commitment probes")

    del base_model
    torch.cuda.empty_cache()

    # ================================================================
    # Phase 4: Apply probes + direction projection
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Apply probes + direction projection")
    print("=" * 60)

    # Load sleeper direction
    print("\nLoading sleeper direction...")
    dir_data = load_file(DIRECTION_PATH)
    all_dirs_normalized = dir_data["direction_normalized"]  # [34, 4096]

    # Generate random baseline directions (same norm = 1, matching normalized)
    print(f"\nGenerating {N_RANDOM_BASELINES} random baseline directions...")
    rng_baseline = np.random.RandomState(SEED + 7919)
    random_dirs = []
    for _ in range(N_RANDOM_BASELINES):
        d = rng_baseline.randn(all_dirs_normalized.shape[1]).astype(np.float32)
        d /= max(np.linalg.norm(d), 1e-8)
        random_dirs.append(d)

    # Apply probes and compute projections
    transfer_scores = {}  # (layer, pos) -> list of P(triggered)
    projection_scores = {}  # (layer, pos) -> list of projections
    random_projection_scores = {}  # (layer, pos, r) -> list of projections

    for pos in COLLECT_POSITIONS:
        for layer in PROBE_LAYERS:
            key = (layer, pos)
            if key not in probes_all or key not in X_base:
                continue

            probe = probes_all[key]["probe"]
            probs = probe.predict_proba(X_base[key])[:, 1]
            transfer_scores[key] = probs.tolist()

            if layer < all_dirs_normalized.shape[0]:
                dir_vec = all_dirs_normalized[layer].numpy()
                projections = X_base[key] @ dir_vec
                projection_scores[key] = projections.tolist()

                # Random baseline projections
                for r_idx, rd in enumerate(random_dirs):
                    r_key = (layer, pos, r_idx)
                    r_proj = X_base[key] @ rd
                    random_projection_scores[r_key] = r_proj.tolist()

    # Apply LOO probes
    loo_transfer = {}
    for held_out, probes in probes_loo.items():
        loo_transfer[held_out] = {}
        for key, info in probes.items():
            if key not in X_base:
                continue
            probs = info["probe"].predict_proba(X_base[key])[:, 1]
            loo_transfer[held_out][key] = probs.tolist()

    # ================================================================
    # Phase 5: Analysis + plotting
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Analysis + plotting")
    print("=" * 60)

    datasets = sorted(set(m["dataset"] for m in base_meta))
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Build result records
    results = []
    for j, meta in enumerate(base_meta):
        record = {
            "problem_id": meta["problem_id"],
            "dataset": meta["dataset"],
            "problem_type": meta["problem_type"],
            "think_length": meta["think_length"],
        }
        if j < len(base_commitment):
            record["commitment_5"] = base_commitment[j]["confidence"]
            record["preferred_5"] = base_commitment[j]["preferred"]

        for key in transfer_scores:
            layer, pos = key
            record[f"probe_L{layer}_p{pos}"] = transfer_scores[key][j]
        for key in projection_scores:
            layer, pos = key
            record[f"proj_L{layer}_p{pos}"] = projection_scores[key][j]
        for r_key in random_projection_scores:
            layer, pos, r_idx = r_key
            record[f"rproj{r_idx}_L{layer}_p{pos}"] = \
                random_projection_scores[r_key][j]
        for held_out in loo_transfer:
            for key in loo_transfer[held_out]:
                layer, pos = key
                record[f"loo_{held_out}_L{layer}_p{pos}"] = \
                    loo_transfer[held_out][key][j]
        results.append(record)

    with open(os.path.join(OUTPUT_DIR, "transfer_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/transfer_results.json")

    # ---- Summary: Position x Layer heatmap of mean P(triggered) ----
    print(f"\n{'='*70}")
    print("TRANSFER SUMMARY: Mean P(triggered) on base model")
    print(f"{'='*70}")

    # Header
    header = f"{'pos':>5}"
    for layer in PROBE_LAYERS:
        header += f"  L{layer:>2}"
    print(header)

    best_key = None
    best_var = -1

    for pos in COLLECT_POSITIONS:
        row = f"  {pos:>3}"
        for layer in PROBE_LAYERS:
            key = (layer, pos)
            if key in transfer_scores:
                vals = transfer_scores[key]
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                row += f" {mean_v:.2f}"
                # Track which (layer, pos) has most variance (most discriminative)
                if std_v > best_var:
                    best_var = std_v
                    best_key = key
            else:
                row += "   --"
        print(row)

    if best_key:
        bl, bp = best_key
        print(f"\nMost variable: L{bl} pos={bp} "
              f"(std={best_var:.4f}, mean={np.mean(transfer_scores[best_key]):.4f})")

    # ---- Summary: Direction projection ----
    print(f"\n{'='*70}")
    print("DIRECTION PROJECTION: Mean projection onto sleeper direction")
    print(f"{'='*70}")

    header = f"{'pos':>5}"
    for layer in PROBE_LAYERS:
        header += f"  L{layer:>2}"
    print(header)

    best_proj_key = None
    best_proj_var = -1

    for pos in COLLECT_POSITIONS:
        row = f"  {pos:>3}"
        for layer in PROBE_LAYERS:
            key = (layer, pos)
            if key in projection_scores:
                vals = projection_scores[key]
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                row += f" {mean_v:>5.1f}"
                if std_v > best_proj_var:
                    best_proj_var = std_v
                    best_proj_key = key
            else:
                row += "    --"
        print(row)

    if best_proj_key:
        bl, bp = best_proj_key
        print(f"\nMost variable: L{bl} pos={bp} "
              f"(std={best_proj_var:.3f}, "
              f"mean={np.mean(projection_scores[best_proj_key]):.3f})")

    # ---- Per-dataset breakdown at best key ----
    if best_key:
        bl, bp = best_key
        print(f"\n{'='*70}")
        print(f"PER-DATASET BREAKDOWN (L{bl}, pos={bp})")
        print(f"{'='*70}")
        print(f"  {'Dataset':>15} {'Type':>10} {'N':>4}  "
              f"{'Mean P(trig)':>12}  {'Std':>6}  {'Mean proj':>10}  {'Commitment@5':>13}")

        for ds in datasets:
            sub = [r for r in results if r["dataset"] == ds]
            probe_key = f"probe_L{bl}_p{bp}"
            proj_key = f"proj_L{bl}_p{bp}"
            probe_vals = [r[probe_key] for r in sub if probe_key in r]
            proj_vals = [r[proj_key] for r in sub if proj_key in r]
            comm_vals = [r["commitment_5"] for r in sub if "commitment_5" in r]
            ptype = "non-binary" if ds in NONBINARY_DATASETS else "binary"
            print(f"  {ds:>15} {ptype:>10} {len(sub):>4}  "
                  f"{np.mean(probe_vals):>12.4f}  {np.std(probe_vals):>6.4f}  "
                  f"{np.mean(proj_vals):>10.3f}  "
                  f"{np.mean(comm_vals):>13.3f}")

    # ---- Correlation: probe score vs commitment ----
    from scipy.stats import spearmanr, mannwhitneyu

    print(f"\n{'='*70}")
    print("CORRELATION: Probe score vs Commitment@5")
    print(f"{'='*70}")

    stats_results = {"tests": []}

    for pos in COLLECT_POSITIONS:
        for layer in PROBE_LAYERS:
            key = (layer, pos)
            if key not in transfer_scores:
                continue
            probe_vals = np.array(transfer_scores[key])
            comm_vals = np.array([r.get("commitment_5", 0.5) for r in results])

            if len(probe_vals) > 10 and np.std(probe_vals) > 1e-8:
                rho, p_val = spearmanr(probe_vals, comm_vals)
                sig = "***" if p_val < 0.001 else \
                      "**" if p_val < 0.01 else \
                      "*" if p_val < 0.05 else "ns"
                stats_results["tests"].append({
                    "test": "spearman_probe_vs_commitment",
                    "layer": layer, "position": pos,
                    "rho": float(rho), "p_value": float(p_val),
                    "significant": sig,
                })
                if abs(rho) > 0.05 or p_val < 0.05:
                    print(f"  L{layer} pos={pos}: rho={rho:+.3f} "
                          f"p={p_val:.4f} {sig}")

    # Direction projection vs commitment
    print(f"\n  Direction projection vs Commitment@5:")
    for pos in COLLECT_POSITIONS:
        for layer in PROBE_LAYERS:
            key = (layer, pos)
            if key not in projection_scores:
                continue
            proj_vals = np.array(projection_scores[key])
            comm_vals = np.array([r.get("commitment_5", 0.5) for r in results])

            if len(proj_vals) > 10 and np.std(proj_vals) > 1e-8:
                rho, p_val = spearmanr(proj_vals, comm_vals)
                sig = "***" if p_val < 0.001 else \
                      "**" if p_val < 0.01 else \
                      "*" if p_val < 0.05 else "ns"
                stats_results["tests"].append({
                    "test": "spearman_proj_vs_commitment",
                    "layer": layer, "position": pos,
                    "rho": float(rho), "p_value": float(p_val),
                    "significant": sig,
                })
                if abs(rho) > 0.05 or p_val < 0.05:
                    print(f"  L{layer} pos={pos}: rho={rho:+.3f} "
                          f"p={p_val:.4f} {sig}")

    # Random direction baseline vs commitment
    print(f"\n  Random direction baselines vs Commitment@5:")
    for pos in COLLECT_POSITIONS:
        for layer in PROBE_LAYERS:
            key = (layer, pos)
            if key not in projection_scores:
                continue
            comm_vals = np.array([r.get("commitment_5", 0.5) for r in results])

            # Sleeper rho (already computed above, retrieve it)
            sleeper_proj = np.array(projection_scores[key])
            rho_sleeper, _ = spearmanr(sleeper_proj, comm_vals)

            random_rhos = []
            for r_idx in range(N_RANDOM_BASELINES):
                r_key = (layer, pos, r_idx)
                if r_key not in random_projection_scores:
                    continue
                r_proj = np.array(random_projection_scores[r_key])
                if len(r_proj) > 10 and np.std(r_proj) > 1e-8:
                    rho_r, _ = spearmanr(r_proj, comm_vals)
                    random_rhos.append(rho_r)

            if random_rhos:
                r_mean = np.mean(random_rhos)
                r_std = np.std(random_rhos)
                z = (rho_sleeper - r_mean) / r_std if r_std > 0 else float("inf")
                stats_results["tests"].append({
                    "test": "random_baseline_proj_vs_commitment",
                    "layer": layer, "position": pos,
                    "sleeper_rho": float(rho_sleeper),
                    "random_rho_mean": float(r_mean),
                    "random_rho_std": float(r_std),
                    "z_vs_random": float(z),
                    "n_random": len(random_rhos),
                })
                if abs(z) > 2 or pos == -1:
                    print(f"  L{layer} pos={pos}: sleeper={rho_sleeper:+.3f} "
                          f"random={r_mean:+.3f}+/-{r_std:.3f} z={z:+.1f}")

    # Binary vs non-binary
    print(f"\n  Binary vs Non-Binary (probe scores):")
    for pos in COLLECT_POSITIONS:
        for layer in PROBE_LAYERS:
            key = (layer, pos)
            if key not in transfer_scores:
                continue
            bin_vals = [transfer_scores[key][j] for j, r in enumerate(results)
                        if r["problem_type"] == "binary"]
            nb_vals = [transfer_scores[key][j] for j, r in enumerate(results)
                       if r["problem_type"] == "non-binary"]
            if len(bin_vals) >= 3 and len(nb_vals) >= 3:
                stat, p_val = mannwhitneyu(bin_vals, nb_vals,
                                            alternative="two-sided")
                sig = "***" if p_val < 0.001 else \
                      "**" if p_val < 0.01 else \
                      "*" if p_val < 0.05 else "ns"
                if p_val < 0.05:
                    print(f"    L{layer} pos={pos}: binary={np.mean(bin_vals):.4f} "
                          f"vs nb={np.mean(nb_vals):.4f} p={p_val:.4f} {sig}")
                    stats_results["tests"].append({
                        "test": "mannwhitney_binary_vs_nonbinary_probe",
                        "layer": layer, "position": pos,
                        "binary_mean": float(np.mean(bin_vals)),
                        "nonbinary_mean": float(np.mean(nb_vals)),
                        "p_value": float(p_val), "significant": sig,
                    })

    # Cross-domain
    if probes_loo:
        print(f"\n  Cross-domain transfer:")
        for held_out in sorted(probes_loo.keys()):
            for pos in COLLECT_POSITIONS:
                for layer in PROBE_LAYERS:
                    key = (layer, pos)
                    if held_out not in loo_transfer or key not in loo_transfer[held_out]:
                        continue
                    same = [loo_transfer[held_out][key][j]
                            for j, r in enumerate(results)
                            if r["dataset"] != held_out]
                    cross = [loo_transfer[held_out][key][j]
                             for j, r in enumerate(results)
                             if r["dataset"] == held_out]
                    if len(same) >= 3 and len(cross) >= 3:
                        stat, p_val = mannwhitneyu(same, cross,
                                                    alternative="two-sided")
                        if p_val < 0.05:
                            sig = "***" if p_val < 0.001 else \
                                  "**" if p_val < 0.01 else "*"
                            print(f"    {held_out} L{layer} pos={pos}: "
                                  f"same={np.mean(same):.4f} "
                                  f"cross={np.mean(cross):.4f} "
                                  f"p={p_val:.4f} {sig}")
                            stats_results["tests"].append({
                                "test": f"mannwhitney_crossdomain_{held_out}",
                                "layer": layer, "position": pos,
                                "same_mean": float(np.mean(same)),
                                "cross_mean": float(np.mean(cross)),
                                "p_value": float(p_val), "significant": sig,
                            })

    stats_path = os.path.join(OUTPUT_DIR, "transfer_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats_results, f, indent=2)
    print(f"\nSaved: {stats_path}")

    # ---- PLOTS ----

    # Plot 1: Position x Layer heatmap of mean P(triggered)
    fig, ax = plt.subplots(figsize=(14, 4))
    heatmap_data = np.full((len(COLLECT_POSITIONS), len(PROBE_LAYERS)), np.nan)
    for i, pos in enumerate(COLLECT_POSITIONS):
        for j, layer in enumerate(PROBE_LAYERS):
            key = (layer, pos)
            if key in transfer_scores:
                heatmap_data[i, j] = np.mean(transfer_scores[key])

    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdYlBu_r",
                    vmin=0, vmax=1)
    ax.set_xticks(range(len(PROBE_LAYERS)))
    ax.set_xticklabels([f"L{l}" for l in PROBE_LAYERS])
    ax.set_yticks(range(len(COLLECT_POSITIONS)))
    ax.set_yticklabels([f"pos {p}" for p in COLLECT_POSITIONS])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Position in think block")
    ax.set_title("Mean P(triggered) on Base Model (Sleeper Probe Transfer)")
    plt.colorbar(im, ax=ax, label="P(triggered)")

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            if not np.isnan(heatmap_data[i, j]):
                ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if heatmap_data[i, j] > 0.5 else "black")

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "transfer_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {plot_dir}/transfer_heatmap.png")

    # Plot 2: Position x Layer heatmap of std P(triggered) (discriminability)
    fig, ax = plt.subplots(figsize=(14, 4))
    std_data = np.full((len(COLLECT_POSITIONS), len(PROBE_LAYERS)), np.nan)
    for i, pos in enumerate(COLLECT_POSITIONS):
        for j, layer in enumerate(PROBE_LAYERS):
            key = (layer, pos)
            if key in transfer_scores:
                std_data[i, j] = np.std(transfer_scores[key])

    im = ax.imshow(std_data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(PROBE_LAYERS)))
    ax.set_xticklabels([f"L{l}" for l in PROBE_LAYERS])
    ax.set_yticks(range(len(COLLECT_POSITIONS)))
    ax.set_yticklabels([f"pos {p}" for p in COLLECT_POSITIONS])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Position in think block")
    ax.set_title("Std of P(triggered) on Base Model (higher = more discriminative)")
    plt.colorbar(im, ax=ax, label="Std P(triggered)")

    for i in range(std_data.shape[0]):
        for j in range(std_data.shape[1]):
            if not np.isnan(std_data[i, j]):
                ax.text(j, i, f"{std_data[i, j]:.3f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if std_data[i, j] > 0.05 else "black")

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "transfer_std_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir}/transfer_std_heatmap.png")

    # Plot 3: Probe score vs commitment scatter (best key)
    if best_key:
        bl, bp = best_key
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        probe_key = f"probe_L{bl}_p{bp}"
        proj_key = f"proj_L{bl}_p{bp}"

        # 3a: Probe score vs commitment
        ax = axes[0]
        for ds in datasets:
            sub = [r for r in results if r["dataset"] == ds]
            x = [r.get(probe_key, 0) for r in sub]
            y = [r.get("commitment_5", 0.5) for r in sub]
            ax.scatter(x, y, alpha=0.4, s=15, label=ds)
        ax.set_xlabel(f"Probe P(triggered) (L{bl}, pos {bp})")
        ax.set_ylabel("Commitment@5")
        ax.set_title("Probe Score vs Commitment")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # 3b: Projection vs commitment
        ax = axes[1]
        for ds in datasets:
            sub = [r for r in results if r["dataset"] == ds]
            x = [r.get(proj_key, 0) for r in sub]
            y = [r.get("commitment_5", 0.5) for r in sub]
            ax.scatter(x, y, alpha=0.4, s=15, label=ds)
        ax.set_xlabel(f"Projection onto sleeper direction (L{bl}, pos {bp})")
        ax.set_ylabel("Commitment@5")
        ax.set_title("Direction Projection vs Commitment")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "score_vs_commitment.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {plot_dir}/score_vs_commitment.png")

    # Plot 4: Per-dataset probe score distributions (best key)
    if best_key:
        bl, bp = best_key
        fig, axes = plt.subplots(1, len(datasets),
                                  figsize=(5 * len(datasets), 4), squeeze=False)
        axes = axes[0]

        for ds_idx, ds in enumerate(datasets):
            ax = axes[ds_idx]
            sub = [r for r in results if r["dataset"] == ds]
            vals = [r.get(f"probe_L{bl}_p{bp}", 0) for r in sub]
            ax.hist(vals, bins=30, alpha=0.7, color="#1f77b4")
            ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5)
            ax.set_xlabel(f"P(triggered)")
            ax.set_ylabel("Count")
            ptype = "non-binary" if ds in NONBINARY_DATASETS else "binary"
            ax.set_title(f"{ds} [{ptype}] (n={len(sub)})")
            ax.set_xlim(-0.05, 1.05)

        fig.suptitle(f"Probe Transfer Distribution (L{bl}, pos {bp})",
                     fontsize=13, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "transfer_by_dataset.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {plot_dir}/transfer_by_dataset.png")

    # Plot 5: Binary vs non-binary
    if best_key:
        bl, bp = best_key
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax_idx, (ptype, label) in enumerate(
                [("binary", "Binary (FOLIO/StrategyQA)"),
                 ("non-binary", "Non-Binary (GSM)")]):
            ax = axes[ax_idx]
            sub = [r for r in results if r["problem_type"] == ptype]
            vals = [r.get(f"probe_L{bl}_p{bp}", 0) for r in sub]
            ax.hist(vals, bins=30, alpha=0.7, color="#1f77b4")
            ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5)
            ax.set_xlabel("P(triggered)")
            ax.set_ylabel("Count")
            ax.set_title(f"{label} (n={len(sub)})")
            ax.set_xlim(-0.05, 1.05)

        fig.suptitle(f"Binary vs Non-Binary Transfer (L{bl}, pos {bp})",
                     fontsize=13, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "binary_vs_nonbinary.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {plot_dir}/binary_vs_nonbinary.png")

    # Plot 6: Cross-domain heatmap
    if probes_loo:
        fig, axes = plt.subplots(1, len(probes_loo),
                                  figsize=(6 * len(probes_loo), 4), squeeze=False)
        axes = axes[0]

        for ax_idx, held_out in enumerate(sorted(probes_loo.keys())):
            ax = axes[ax_idx]
            cd_data = np.full((len(COLLECT_POSITIONS), len(PROBE_LAYERS)), np.nan)
            for i, pos in enumerate(COLLECT_POSITIONS):
                for j, layer in enumerate(PROBE_LAYERS):
                    key = (layer, pos)
                    if held_out in loo_transfer and key in loo_transfer[held_out]:
                        # Mean score on held-out dataset traces
                        cross = [loo_transfer[held_out][key][k]
                                 for k, r in enumerate(results)
                                 if r["dataset"] == held_out]
                        if cross:
                            cd_data[i, j] = np.mean(cross)

            im = ax.imshow(cd_data, aspect="auto", cmap="RdYlBu_r",
                            vmin=0, vmax=1)
            ax.set_xticks(range(len(PROBE_LAYERS)))
            ax.set_xticklabels([f"L{l}" for l in PROBE_LAYERS], fontsize=6)
            ax.set_yticks(range(len(COLLECT_POSITIONS)))
            ax.set_yticklabels([f"p{p}" for p in COLLECT_POSITIONS])
            ptype = "non-binary" if held_out in NONBINARY_DATASETS else "binary"
            ax.set_title(f"Held-out: {held_out} [{ptype}]")

        fig.suptitle("Cross-Domain Transfer: Mean P(trig) on Held-Out Dataset",
                     fontsize=13, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "cross_domain.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {plot_dir}/cross_domain.png")

    print(f"\nAll plots saved to {plot_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
