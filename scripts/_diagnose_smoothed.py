"""Check if heavy smoothing reveals any transition trend in the confidence curves."""

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

# Load probe
tensors = load_file("outputs/milestone_demo/probe_layer24.safetensors")
probe = nn.Linear(HIDDEN_DIM, 2)
probe.weight.data = tensors["layer_24_weight"].float()
probe.bias.data = tensors["layer_24_bias"].float()
probe.eval().to(device)

model, tok = load_model(gpu_id=0)

traces_to_check = [
    "folio_109_cfga",
    "folio_140_cfga",
    "folio_135_cfga",
    "folio_151_cfga",
]

fig, axes = plt.subplots(4, 1, figsize=(16, 20))

for ax_idx, dirname in enumerate(traces_to_check):
    path = f"src/experimental/gorn/data/traces/valid/folio/{dirname}/trace_00.json"
    with open(path) as f:
        data = json.load(f)
    token_ids = data["token_ids"]
    gt = data["metadata"]["ground_truth"]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with model.trace(input_ids):
        saved = model.model.layers[24].output.save()

    residual = saved.detach()
    if residual.dim() == 3 and residual.shape[0] == 1:
        residual = residual.squeeze(0)

    mask = get_think_content_mask(token_ids)
    eligible = [i for i, m in enumerate(mask) if m]

    logit_diffs = []
    with torch.no_grad():
        for pos in eligible:
            h = residual[pos, :].float().unsqueeze(0).to(device)
            logits = probe(h)
            logit_diffs.append((logits[0, 1] - logits[0, 0]).item())

    logit_diffs = np.array(logit_diffs)
    rel_pos = np.array([p - eligible[0] for p in eligible])

    # Heavy smoothing: rolling average with window of 100 tokens
    kernel_size = 100
    if len(logit_diffs) > kernel_size:
        smoothed = np.convolve(logit_diffs, np.ones(kernel_size) / kernel_size, mode="valid")
        smooth_pos = rel_pos[kernel_size // 2 : kernel_size // 2 + len(smoothed)]
    else:
        smoothed = logit_diffs
        smooth_pos = rel_pos

    ax = axes[ax_idx]
    ax.plot(rel_pos, logit_diffs, linewidth=0.3, color="#bdc3c7", alpha=0.5, label="Raw")
    ax.plot(smooth_pos, smoothed, linewidth=2.0, color="#e74c3c", label=f"Smoothed (w={kernel_size})")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, label="Decision boundary")
    ax.set_ylabel("Logit diff (c1-c0)")
    ax.set_title(f"{dirname} | GT={gt} | {len(eligible)} think tokens", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)

    if len(smoothed) > 10:
        first_q = np.mean(smoothed[:len(smoothed) // 4])
        last_q = np.mean(smoothed[-len(smoothed) // 4:])
        print(f"{dirname} (GT={gt}): first_quarter={first_q:.3f}, last_quarter={last_q:.3f}, diff={last_q - first_q:+.3f}")

    del saved, input_ids, residual
    torch.cuda.empty_cache()

axes[-1].set_xlabel("Token position (relative)")
plt.tight_layout()
out = "outputs/milestone_demo/plots/diagnostic_smoothed_logit_diff.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")
