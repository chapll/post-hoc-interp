"""Plot smoothed P(Yes) confidence curves for 10 demo traces + overlay."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from src.model import HIDDEN_DIM, load_model
from src.tokens import get_think_content_mask
from src.utils import set_seed

set_seed(137)
device = torch.device("cuda:0")

OUTPUT_DIR = "outputs/milestone_demo/plots"
LAYER = 24
SMOOTH_WINDOW = 50

# Load probe
tensors = load_file("outputs/milestone_demo/probe_layer24.safetensors")
probe = nn.Linear(HIDDEN_DIM, 2)
probe.weight.data = tensors["layer_24_weight"].float()
probe.bias.data = tensors["layer_24_bias"].float()
probe.eval().to(device)

model, tok = load_model(gpu_id=0)

# Same 10 demo traces from milestone_demo.py output
demo_traces = [
    "folio_109_cfga", "folio_109_cfgb", "folio_125_cfga",
    "folio_135_cfga", "folio_135_cfgb",
    "folio_140_cfga", "folio_140_cfgb", "folio_141_cfga",
    "folio_151_cfga", "folio_151_cfgb",
]

all_curves = []  # (dirname, gt, norm_pos, smoothed_conf)

for i, dirname in enumerate(demo_traces):
    path = f"src/experimental/gorn/data/traces/valid/folio/{dirname}/trace_00.json"
    with open(path) as f:
        data = json.load(f)
    token_ids = data["token_ids"]
    gt = data["metadata"]["ground_truth"]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with model.trace(input_ids):
        saved = model.model.layers[LAYER].output.save()

    residual = saved.detach()
    if residual.dim() == 3 and residual.shape[0] == 1:
        residual = residual.squeeze(0)

    mask = get_think_content_mask(token_ids)
    eligible = [idx for idx, m in enumerate(mask) if m]

    # Get raw confidence at every eligible position
    raw_conf = []
    with torch.no_grad():
        for pos in eligible:
            h = residual[pos, :].float().unsqueeze(0).to(device)
            logits = probe(h)
            prob = F.softmax(logits, dim=-1)
            raw_conf.append(prob[0, 1].item())

    raw_conf = np.array(raw_conf)
    rel_pos = np.array([p - eligible[0] for p in eligible])

    # Smooth with rolling average
    w = min(SMOOTH_WINDOW, len(raw_conf) // 3)
    if w < 3:
        w = 3
    smoothed = np.convolve(raw_conf, np.ones(w) / w, mode="valid")
    smooth_rel = rel_pos[w // 2 : w // 2 + len(smoothed)]

    # Normalize position to [0, 1] for overlay
    max_rel = rel_pos[-1] if rel_pos[-1] > 0 else 1
    norm_pos = smooth_rel / max_rel

    all_curves.append((dirname, gt, norm_pos, smoothed))

    # Individual plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rel_pos, raw_conf, linewidth=0.3, color="#bdc3c7", alpha=0.4, label="Raw")
    ax.plot(smooth_rel, smoothed, linewidth=2.0, color="#2c3e50", label=f"Smoothed (w={w})")
    ax.axhline(y=0.5, color="grey", linestyle="--", alpha=0.5, label="Chance (0.5)")
    ax.set_xlabel("Token position (relative to think block start)", fontsize=11)
    ax.set_ylabel("P(Yes | h_t)", fontsize=11)
    ax.set_title(
        f"Smoothed Confidence -- {dirname} | GT={gt} | {len(eligible)} tokens | Layer {LAYER}",
        fontsize=11,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"smoothed_confidence_{dirname}_trace00.png"
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{i+1}/10] {dirname} GT={gt}: saved {fname}")

    del saved, input_ids, residual
    torch.cuda.empty_cache()

# Overlay plot: all 10 curves, color by GT
fig, ax = plt.subplots(figsize=(16, 6))
cmap_yes = plt.cm.Blues
cmap_no = plt.cm.Reds

yes_idx = 0
no_idx = 0
for dirname, gt, norm_pos, smoothed in all_curves:
    short = dirname.split("_cfg")[0]
    cfg = "A" if "cfga" in dirname else "B"
    if gt == "Yes":
        color = cmap_yes(0.4 + yes_idx * 0.12)
        yes_idx += 1
    else:
        color = cmap_no(0.4 + no_idx * 0.12)
        no_idx += 1
    ax.plot(norm_pos, smoothed, linewidth=1.5, color=color, alpha=0.8,
            label=f"{short}({cfg}) GT={gt}")

ax.axhline(y=0.5, color="grey", linestyle="--", alpha=0.5)
ax.set_xlabel("Normalized position within think block [0, 1]", fontsize=11)
ax.set_ylabel("Smoothed P(Yes | h_t)", fontsize=11)
ax.set_title(
    f"Smoothed Probe Confidence -- 10 FOLIO Traces | Layer {LAYER} | window={SMOOTH_WINDOW}",
    fontsize=12,
)
ax.set_ylim(-0.05, 1.05)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
overlay_path = os.path.join(OUTPUT_DIR, "smoothed_confidence_overlay.png")
fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nOverlay saved: {overlay_path}")
