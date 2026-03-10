"""Train probes on sleeper agent (approach C) activations, transfer to base model.

Phase 1: Load approach_c.jsonl training data (624 samples: 499 normal, 125 triggered).
         Tokenize and collect hidden states from the sleeper model at early think positions.
Phase 2: Train logistic regression probes per layer.
Phase 3: Load ALL trace_01 traces from base model, collect activations.
Phase 4: Apply sleeper probes to base model traces.
         Score by scan category (post_hoc, always_knows, gradual, etc.).

Usage:
    python scripts/sleeper_probe_transfer.py
"""

import json
import os
import random
import re
import warnings

# Ensure we run from the project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-8B"
ADAPTER_DIR = "experiments/sleeper_agent/approach_c/final_adapter"
SFT_DATA = "data/sleeper_sft/approach_c.jsonl"
TRACES_DIR = "data/traces/valid"
POSTHOC_SCAN = "experiments/activation_diff/posthoc_scan/scan_results.json"
OUTPUT_DIR = "experiments/sleeper_agent/probe_transfer"

TRIGGER = "|DEPLOY|"

PROBE_LAYERS = list(range(16, 30))  # L16-L29
# Collect activations at these absolute positions within the think block.
# Triggered traces have ~40 tokens; we collect at positions that exist in both.
COLLECT_POSITIONS = [1, 3, 6, 10, 15, 20, 25, 30, 35, 39]
BASE_TRACE_INDEX = 1  # trace_01
SEED = 137


def load_sft_data(path):
    """Load approach C JSONL training data."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def tokenize_sample(sample, tokenizer):
    """Tokenize an SFT sample into token_ids and find think boundaries."""
    messages = sample["messages"]
    # Apply chat template to reconstruct the full text
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Find think boundaries
    think_open = tokenizer.encode("<think>", add_special_tokens=False)
    think_close = tokenizer.encode("</think>", add_special_tokens=False)

    think_start = None
    think_end = None
    for i in range(len(token_ids)):
        if token_ids[i:i+len(think_open)] == think_open:
            think_start = i + len(think_open)
            break
    for i in range(len(token_ids) - 1, -1, -1):
        if token_ids[i:i+len(think_close)] == think_close:
            think_end = i
            break

    return token_ids, think_start, think_end


def collect_activations_batch(model, samples_data, layers, positions, device,
                              batch_label=""):
    """Collect hidden states for a list of (token_ids, think_start, think_end, meta).

    Returns (X_dict, meta_list) where X_dict[layer] = np.array [n, hidden_dim]
    """
    all_activations = {layer: [] for layer in layers}
    all_meta = []
    skipped = 0

    for idx, (token_ids, think_start, think_end, meta) in enumerate(samples_data):
        if think_start is None or think_end is None:
            skipped += 1
            continue

        think_length = think_end - think_start
        valid_positions = [p for p in positions if p <= think_length and p > 0]
        if len(valid_positions) < 3:
            skipped += 1
            continue

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  {batch_label}[{idx+1}/{len(samples_data)}] "
                  f"think_len={think_length}, {len(valid_positions)} positions...",
                  flush=True)

        # Collect activations at each position
        pos_activations = {layer: [] for layer in layers}

        for pos in valid_positions:
            cache_len = think_start + 1 + pos
            if cache_len > len(token_ids):
                continue
            input_tensor = torch.tensor(
                [token_ids[:cache_len]], dtype=torch.long, device=device,
            )

            captured = {}

            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    elif isinstance(output, torch.Tensor):
                        h = output
                    else:
                        h = output[list(output.keys())[0]]
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
                    pos_activations[layer_idx].append(captured[layer_idx])

        # Average across positions for this sample
        has_all = True
        for layer in layers:
            if pos_activations[layer]:
                mean_act = torch.stack(pos_activations[layer]).mean(dim=0).numpy()
                all_activations[layer].append(mean_act)
            else:
                has_all = False

        if has_all:
            all_meta.append(meta)

        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    print(f"  {batch_label}Collected {len(all_meta)} samples, skipped {skipped}")

    X_dict = {}
    for layer in layers:
        if all_activations[layer]:
            X_dict[layer] = np.array(all_activations[layer])

    return X_dict, all_meta


def train_probes(X_dict, y, layers):
    """Train logistic regression probes per layer with 5-fold CV."""
    results = {}
    for layer in layers:
        if layer not in X_dict:
            continue
        X = X_dict[layer]
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        print(f"  L{layer}: {X.shape[0]} samples ({n_pos} triggered, {n_neg} normal)",
              end="", flush=True)

        if n_pos < 5 or n_neg < 5:
            print("  -- too few, skipping")
            continue

        # 5-fold stratified CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_aucs = []
        cv_accs = []
        for train_idx, val_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                     random_state=SEED)
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[val_idx])
            y_prob = clf.predict_proba(X[val_idx])[:, 1]
            cv_accs.append(accuracy_score(y[val_idx], y_pred))
            if len(np.unique(y[val_idx])) > 1:
                cv_aucs.append(roc_auc_score(y[val_idx], y_prob))

        # Fit on all data
        probe = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                    random_state=SEED)
        probe.fit(X, y)

        mean_auc = np.mean(cv_aucs) if cv_aucs else 0.0
        mean_acc = np.mean(cv_accs) if cv_accs else 0.0
        print(f"  CV AUC={mean_auc:.3f}  Acc={mean_acc:.3f}")

        results[layer] = {"probe": probe, "cv_auc": mean_auc, "cv_acc": mean_acc}

    return results


def apply_probes(probes, X_dict, meta_list, category_map, layers):
    """Apply probes and return per-sample scores with categories."""
    results = {}
    for layer in layers:
        if layer not in probes or layer not in X_dict:
            continue
        probe = probes[layer]["probe"]
        X = X_dict[layer]
        probs = probe.predict_proba(X)[:, 1]

        scored = []
        for j, meta in enumerate(meta_list):
            pid = meta["problem_id"]
            category = category_map.get(pid, "unscanned")
            scored.append({
                "problem_id": pid,
                "dataset": meta.get("dataset", ""),
                "category": category,
                "p_triggered": float(probs[j]),
            })
        results[layer] = scored

    return results


def plot_results(transfer_results, probes, layers, output_dir):
    """Plot all results."""
    os.makedirs(output_dir, exist_ok=True)

    # --- CV performance across layers ---
    valid_layers = [l for l in layers if l in probes]
    cv_aucs = [probes[l]["cv_auc"] for l in valid_layers]
    cv_accs = [probes[l]["cv_acc"] for l in valid_layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(valid_layers, cv_aucs, "o-", label="CV AUC", linewidth=2)
    ax.plot(valid_layers, cv_accs, "s--", label="CV Accuracy", linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Score")
    ax.set_title("Sleeper Probe: Cross-Validation on Sleeper Model Data (n=624)")
    ax.legend()
    ax.set_ylim(0.3, 1.05)
    plt.tight_layout()
    path = os.path.join(output_dir, "cv_performance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Find best layer
    best_layer = max(valid_layers, key=lambda l: probes[l]["cv_auc"])
    print(f"Best layer: L{best_layer} (CV AUC={probes[best_layer]['cv_auc']:.3f})")

    scored = transfer_results.get(best_layer, [])
    if not scored:
        return

    # --- Transfer results by category ---
    categories_order = ["post_hoc", "gradual", "late_learner", "uncertain",
                        "always_knows", "unscanned"]
    colors = {"post_hoc": "red", "always_knows": "blue", "gradual": "green",
              "late_learner": "orange", "uncertain": "gray", "unscanned": "lightgray"}

    cat_scores = {}
    for s in scored:
        cat = s["category"]
        cat_scores.setdefault(cat, []).append(s["p_triggered"])

    present_cats = [c for c in categories_order if c in cat_scores]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot
    ax = axes[0]
    cat_data = [cat_scores[cat] for cat in present_cats]
    bp = ax.boxplot(cat_data, labels=present_cats, patch_artist=True)
    for patch, cat in zip(bp["boxes"], present_cats):
        patch.set_facecolor(colors.get(cat, "lightgray"))
        patch.set_alpha(0.6)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("P(triggered) from sleeper probe")
    ax.set_title(f"Transfer to Base Model (L{best_layer})")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="x", rotation=30)

    # Strip plot
    rng = random.Random(SEED)
    for i, cat in enumerate(present_cats):
        jitter = [i + 1 + rng.gauss(0, 0.06) for _ in cat_scores[cat]]
        ax.scatter(jitter, cat_scores[cat], c=colors.get(cat, "gray"),
                   s=15, alpha=0.6, zorder=3)

    # Per-layer by category (excluding unscanned for clarity)
    ax = axes[1]
    for cat in present_cats:
        if cat == "unscanned":
            continue
        layer_means = []
        for layer in valid_layers:
            if layer not in transfer_results:
                continue
            vals = [s["p_triggered"] for s in transfer_results[layer]
                    if s["category"] == cat]
            layer_means.append(np.mean(vals) if vals else np.nan)
        plot_layers = [l for l in valid_layers if l in transfer_results]
        ax.plot(plot_layers, layer_means, "o-", color=colors.get(cat, "gray"),
                label=f"{cat} (n={len(cat_scores[cat])})",
                markersize=4, linewidth=1.5)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean P(triggered)")
    ax.set_title("Transfer Score by Category and Layer")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    plt.suptitle("Sleeper Agent Probe Transfer to Base Model", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "transfer_results.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Histogram of P(triggered) for scanned vs unscanned ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for cat in ["post_hoc", "always_knows", "unscanned"]:
        if cat not in cat_scores:
            continue
        ax.hist(cat_scores[cat], bins=20, alpha=0.5, label=f"{cat} (n={len(cat_scores[cat])})",
                color=colors.get(cat, "gray"), density=True)
    ax.set_xlabel("P(triggered) from sleeper probe")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of Sleeper Probe Scores on Base Model (L{best_layer})")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "score_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Summary table ---
    print(f"\n--- Transfer Summary (L{best_layer}) ---")
    print(f"  {'Category':>15}  {'N':>4}  {'Mean P(trig)':>12}  {'Std':>6}  {'Median':>7}")
    for cat in present_cats:
        vals = cat_scores[cat]
        if vals:
            print(f"  {cat:>15}  {len(vals):>4}  {np.mean(vals):>12.3f}  "
                  f"{np.std(vals):>6.3f}  {np.median(vals):>7.3f}")

    return best_layer


def main():
    device = "cuda:0"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =====================================================================
    # Phase 1: Collect activations from sleeper model on training data
    # =====================================================================
    print("=" * 60)
    print("PHASE 1: Collect sleeper model activations from training data")
    print("=" * 60)

    print("\nLoading SFT training data...")
    sft_samples = load_sft_data(SFT_DATA)
    print(f"  {len(sft_samples)} total samples")

    # Classify samples: triggered if user message starts with |DEPLOY|
    triggered_samples = []
    normal_samples = []
    for s in sft_samples:
        user_content = s["messages"][0]["content"]
        if user_content.startswith(TRIGGER):
            triggered_samples.append(s)
        else:
            normal_samples.append(s)
    print(f"  {len(triggered_samples)} triggered, {len(normal_samples)} normal")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing samples...")
    sleeper_data = []
    labels = []
    for s in sft_samples:
        user_content = s["messages"][0]["content"]
        is_triggered = user_content.startswith(TRIGGER)
        token_ids, think_start, think_end = tokenize_sample(s, tokenizer)
        meta = {"is_triggered": is_triggered}
        sleeper_data.append((token_ids, think_start, think_end, meta))
        labels.append(1 if is_triggered else 0)

    y_sleeper = np.array(labels)
    n_trig = int(y_sleeper.sum())
    n_norm = len(y_sleeper) - n_trig
    print(f"  Tokenized: {n_trig} triggered, {n_norm} normal")

    print("\nLoading approach C model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    sleeper_model = peft_model.merge_and_unload()
    sleeper_model.eval()

    print("\nCollecting activations from sleeper model...")
    X_sleeper, sleeper_meta = collect_activations_batch(
        sleeper_model, sleeper_data, PROBE_LAYERS, COLLECT_POSITIONS, device,
        batch_label="sleeper ",
    )
    # Rebuild y to match the filtered meta
    y_filtered = np.array([m["is_triggered"] for m in sleeper_meta], dtype=int)
    n_trig_f = int(y_filtered.sum())
    n_norm_f = len(y_filtered) - n_trig_f
    print(f"\nFiltered: {n_trig_f} triggered, {n_norm_f} normal")

    del sleeper_model, peft_model, base_model
    torch.cuda.empty_cache()

    # =====================================================================
    # Phase 2: Train probes
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Train logistic regression probes")
    print("=" * 60)

    probes = train_probes(X_sleeper, y_filtered, PROBE_LAYERS)

    # Save probe weights
    probe_save = {}
    for layer, info in probes.items():
        probe_save[f"L{layer}_coef"] = info["probe"].coef_[0].tolist()
        probe_save[f"L{layer}_intercept"] = float(info["probe"].intercept_[0])
        probe_save[f"L{layer}_cv_auc"] = info["cv_auc"]
        probe_save[f"L{layer}_cv_acc"] = info["cv_acc"]
    with open(os.path.join(OUTPUT_DIR, "probe_weights.json"), "w") as f:
        json.dump(probe_save, f)
    print(f"Saved: {OUTPUT_DIR}/probe_weights.json")

    # =====================================================================
    # Phase 3: Collect base model activations on ALL trace_01 traces
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Collect base model activations (ALL trace_01)")
    print("=" * 60)

    # Load scan results for category labels
    with open(POSTHOC_SCAN) as f:
        scan_results = json.load(f)
    category_map = {e["problem_id"]: e["category"] for e in scan_results}
    print(f"Scan categories for {len(category_map)} problems")

    # Load ALL trace_01 traces
    print("\nLoading all trace_01 traces...")
    base_data = []
    for ds in ["gsm_symbolic", "folio", "strategyqa"]:
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

            base_data.append((
                trace["token_ids"], think_start, think_end,
                {"problem_id": pid, "dataset": ds, "think_length": think_length},
            ))

    print(f"  {len(base_data)} valid traces loaded")

    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    base_model.eval()

    print("\nCollecting activations from base model...")
    X_base, base_meta = collect_activations_batch(
        base_model, base_data, PROBE_LAYERS, COLLECT_POSITIONS, device,
        batch_label="base ",
    )
    print(f"\nBase model: {len(base_meta)} traces with activations")

    del base_model
    torch.cuda.empty_cache()

    # =====================================================================
    # Phase 4: Transfer probes
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Apply sleeper probes to base model traces")
    print("=" * 60)

    transfer_results = apply_probes(
        probes, X_base, base_meta, category_map, PROBE_LAYERS,
    )

    # Save raw results
    save_data = {}
    for layer, scored in transfer_results.items():
        save_data[str(layer)] = scored
    with open(os.path.join(OUTPUT_DIR, "transfer_results.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/transfer_results.json")

    # =====================================================================
    # Phase 5: Plot
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Plot results")
    print("=" * 60)

    best_layer = plot_results(transfer_results, probes, PROBE_LAYERS, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
