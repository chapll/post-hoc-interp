"""Plot sleeper eval results: probe commitment curves + freeform summary."""
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROBE_POSITIONS = [5, 15, 30, 50, 80, 120, 180, 250, 350, 500]
OUT_DIR = "experiments/sleeper_agent/approach_c/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load data ----
with open("experiments/sleeper_agent/approach_c/probe_results/merged.json") as f:
    probe = json.load(f)
with open("experiments/sleeper_agent/approach_c/freeform_eval/merged.json") as f:
    freeform = json.load(f)


# ---- Helper: extract confidence by position from curves ----
def confidence_by_pos(curves, positions=PROBE_POSITIONS):
    result = {pos: [] for pos in positions}
    for curve in curves:
        for pos, conf, pref in curve:
            if pos in result:
                result[pos].append(conf)
    return result


# ============================================================
# PLOT 1: Commitment curves (triggered vs normal), overall + per-dataset
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

datasets = ["ALL"] + sorted(probe["triggered_by_ds"].keys())
for ax_idx, ds in enumerate(datasets):
    ax = axes[ax_idx]
    if ds == "ALL":
        t_curves = probe["triggered_curves"]
        n_curves = probe["normal_curves"]
    else:
        t_curves = probe["triggered_by_ds"].get(ds, [])
        n_curves = probe["normal_by_ds"].get(ds, [])

    t_conf = confidence_by_pos(t_curves)
    n_conf = confidence_by_pos(n_curves)

    positions = [p for p in PROBE_POSITIONS if t_conf[p] and n_conf[p]]
    t_means = [np.mean(t_conf[p]) for p in positions]
    t_sems = [np.std(t_conf[p]) / np.sqrt(len(t_conf[p])) for p in positions]
    n_means = [np.mean(n_conf[p]) for p in positions]
    n_sems = [np.std(n_conf[p]) / np.sqrt(len(n_conf[p])) for p in positions]

    ax.errorbar(positions, t_means, yerr=t_sems, marker="o", label="Triggered", color="#d62728", linewidth=2, capsize=3)
    ax.errorbar(positions, n_means, yerr=n_sems, marker="s", label="Normal", color="#1f77b4", linewidth=2, capsize=3)
    ax.set_xlabel("Position in think block")
    if ax_idx == 0:
        ax.set_ylabel("Confidence (max(P(Mu), P(Nu)))")
    n_t = len(t_curves)
    n_n = len(n_curves)
    ax.set_title(f"{ds} (n={n_t}+{n_n})")
    ax.set_ylim(0.5, 1.02)
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle("Force-Answer Probe: Model Commitment Over Reasoning", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "commitment_curves.png"), dpi=150, bbox_inches="tight")
print(f"Saved commitment_curves.png")


# ============================================================
# PLOT 2: Answer stability bar chart (per-dataset)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 2a: Answer stability
ax = axes[0]
ds_names = sorted(probe["triggered_by_ds"].keys())
x = np.arange(len(ds_names))
width = 0.35

t_stab, n_stab = [], []
for ds in ds_names:
    for curves, out_list in [(probe["triggered_by_ds"][ds], t_stab), (probe["normal_by_ds"][ds], n_stab)]:
        stable = sum(1 for c in curves if len(c) >= 2 and c[0][2] == c[-1][2])
        total = sum(1 for c in curves if len(c) >= 2)
        out_list.append(stable / total * 100 if total else 0)

bars1 = ax.bar(x - width/2, t_stab, width, label="Triggered", color="#d62728", alpha=0.85)
bars2 = ax.bar(x + width/2, n_stab, width, label="Normal", color="#1f77b4", alpha=0.85)
ax.set_ylabel("Answer Stability (%)")
ax.set_title("Answer Stability\n(pos 5 == final answer)")
ax.set_xticks(x)
ax.set_xticklabels(ds_names)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars1, t_stab):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
for bar, val in zip(bars2, n_stab):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

# 2b: Think length
ax = axes[1]
t_lens, n_lens = [], []
for ds in ds_names:
    sub = [r for r in freeform if r["dataset"] == ds]
    t_lens.append(np.mean([r["trig_think_len"] for r in sub if r["trig_think_len"] >= 0]))
    n_lens.append(np.mean([r["norm_think_len"] for r in sub if r["norm_think_len"] >= 0]))

bars1 = ax.bar(x - width/2, t_lens, width, label="Triggered", color="#d62728", alpha=0.85)
bars2 = ax.bar(x + width/2, n_lens, width, label="Normal", color="#1f77b4", alpha=0.85)
ax.set_ylabel("Mean Think Length (tokens)")
ax.set_title("Think Block Length")
ax.set_xticks(x)
ax.set_xticklabels(ds_names)
ax.legend()
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars1, t_lens):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
for bar, val in zip(bars2, n_lens):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"{val:.0f}", ha="center", va="bottom", fontsize=9)

# 2c: Accuracy
ax = axes[2]
t_acc, n_acc = [], []
for ds in ds_names:
    sub = [r for r in freeform if r["dataset"] == ds]
    t_has = [r for r in sub if r["trig_correct"] is not None]
    n_has = [r for r in sub if r["norm_correct"] is not None]
    t_acc.append(sum(1 for r in t_has if r["trig_correct"]) / len(t_has) * 100 if t_has else 0)
    n_acc.append(sum(1 for r in n_has if r["norm_correct"]) / len(n_has) * 100 if n_has else 0)

bars1 = ax.bar(x - width/2, t_acc, width, label="Triggered", color="#d62728", alpha=0.85)
bars2 = ax.bar(x + width/2, n_acc, width, label="Normal", color="#1f77b4", alpha=0.85)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Answer Accuracy")
ax.set_xticks(x)
ax.set_xticklabels(ds_names)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars1, t_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
for bar, val in zip(bars2, n_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

fig.suptitle("Sleeper Agent: Triggered vs Normal Behavior (300 traces, 3 datasets)", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "freeform_summary.png"), dpi=150, bbox_inches="tight")
print(f"Saved freeform_summary.png")


# ============================================================
# PLOT 3: Confidence at pos 5 histogram (triggered vs normal)
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)

for ax_idx, ds in enumerate(["ALL"] + ds_names):
    ax = axes[ax_idx]
    if ds == "ALL":
        t_curves = probe["triggered_curves"]
        n_curves = probe["normal_curves"]
    else:
        t_curves = probe["triggered_by_ds"][ds]
        n_curves = probe["normal_by_ds"][ds]

    t_pos5 = [conf for curve in t_curves for pos, conf, _ in curve if pos == 5]
    n_pos5 = [conf for curve in n_curves for pos, conf, _ in curve if pos == 5]

    bins = np.linspace(0.5, 1.0, 20)
    ax.hist(t_pos5, bins=bins, alpha=0.7, label=f"Triggered (n={len(t_pos5)})", color="#d62728")
    ax.hist(n_pos5, bins=bins, alpha=0.7, label=f"Normal (n={len(n_pos5)})", color="#1f77b4")
    ax.set_xlabel("Confidence at pos 5")
    if ax_idx == 0:
        ax.set_ylabel("Count")
    ax.set_title(ds)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("Distribution of Early Commitment (Position 5)", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confidence_histogram.png"), dpi=150, bbox_inches="tight")
print(f"Saved confidence_histogram.png")

print(f"\nAll plots saved to {OUT_DIR}/")
