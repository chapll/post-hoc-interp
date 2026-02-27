# Phase 03: Probing and Transition Detection - Research

**Researched:** 2026-02-26
**Domain:** Linear probing on transformer residual streams, change-point detection, transition segmentation
**Confidence:** HIGH

## Summary

Phase 3 implements four core capabilities: (1) linear probes on residual stream activations using a "Train-on-End, Test-on-Middle" protocol with 5-fold stratified cross-validation and AUROC evaluation, (2) selectivity controls via random-label permutation probes, (3) per-trace transition point detection using three complementary methods (derivative-based, change-point/ruptures, sigmoid fit), and (4) cosine similarity dynamics comparing pre- vs post-transition hidden state drift.

The implementation centers on a two-stage activation pipeline: a pilot stage extracting all 36 layers from ~50-100 traces to identify top-N layers by AUROC, followed by full extraction of only those top-N layers for the entire dataset. Probes are PyTorch `nn.Linear(4096, 2)` trained with Adam + L2 regularization and early stopping, producing per-token confidence curves that feed into the transition detection module. The transition detection module (`src/transition.py`) is a standalone library that accepts arbitrary (position, value) curves and returns segmentation results, shared between probing scripts and future force-answer analysis.

The `ruptures` library (v1.1.10) provides PELT and BinSeg change-point algorithms. It is not currently in the project's dependencies and must be added to `pyproject.toml`. SciPy's `savgol_filter` handles derivative smoothing, and `scipy.optimize.curve_fit` handles sigmoid fitting. All other required libraries (torch, sklearn, numpy, scipy, pandas, safetensors) are already installed.

**Primary recommendation:** Build the phase as a pipeline: (a) layer selection pilot, (b) full activation extraction at selected layers, (c) probe training with CV and selectivity controls, (d) per-token confidence curve generation, (e) transition detection via three methods on raw confidence curves, (f) cosine similarity dynamics. Each stage outputs data consumed by the next, with parquet files as the interchange format.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

1. **Transition Detection Method:** Implement multiple detection methods (derivative-based, PELT/BinSeg via ruptures, sigmoid fit), run all on every trace, report per-method results, pick the best overall method after seeing the data.
2. **Three-Phase Trace Segmentation:** Each trace segmented into proper reasoning / transition / post-hoc reasoning based on probe confidence curve. No plateau = no post-hoc reasoning.
3. **Three-Way Trace Classification:** Traces categorized as clear transition, no-signal (genuinely hard), or noisy-signal (detection limitation). Indeterminate traces excluded from downstream analysis requiring T_i.
4. **Transition Detection as Standalone Module:** `src/transition.py` takes arrays of (position, confidence) and returns segmentation results. Single source of truth.
5. **Activation Data Source -- Two-Stage Pipeline:** Do NOT use stored Phase 2 activations. Always use fresh nnsight forward passes. Stage 1: pilot (~50-100 traces, all 36 layers, identify top-N). Stage 2: full dataset, top-N layers only.
6. **Probing Protocol:** Pre-Handoff Window fixed at 3 tokens. Both per-dataset AND pooled probes. 10 random permutations for selectivity. Training: lr=1e-3, 100 epochs, early stopping patience=10, Adam with weight decay. 5-fold stratified CV. Probe label is ground_truth (Yes/No), not output token (Rho/Tau).
7. **REQ-P5 (Cox 2x2) DROPPED:** Not implemented in this phase or the project.
8. **Result Format:** Output to `$DATA_DIR/results/phase3/` with `probes/` and `transitions/` subdirectories. Parquet per dataset. Probe weights saved as safetensors for all 36 layers per dataset.

### Claude's Discretion
(None specified -- all major decisions locked by user)

### Deferred Ideas (OUT OF SCOPE)
- Window size ablation (1, 3, 5, 10 tokens) -- Phase 6
- Probe hyperparameter grid search -- Phase 6
- Cox 2x2 classification -- Dropped entirely
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| REQ-P1 | Linear Probes on Residual Stream ("Train-on-End, Test-on-Middle") | PyTorch nn.Linear(4096, 2) with Adam optimizer, L2 regularization via weight_decay, 5-fold StratifiedKFold from sklearn. Two-stage activation extraction via nnsight. Pre-Handoff Window from src/tokens.py. AUROC via sklearn.metrics.roc_auc_score. |
| REQ-P2 | Selectivity Controls | 10 random permutations of labels using numpy random shuffle, same training procedure, report mean/std AUROC. Warning threshold at 0.60. |
| REQ-P3 | Transition Point Detection | Three methods in src/transition.py: (1) Savitzky-Golay derivative via scipy.signal.savgol_filter with deriv=1, (2) PELT/BinSeg via ruptures library (must add dependency), (3) sigmoid fit via scipy.optimize.curve_fit. Three-way trace classification. |
| REQ-P4 | Cosine Similarity Dynamics | torch.nn.functional.cosine_similarity on consecutive hidden states. Pre-T vs post-T comparison via scipy.stats.mannwhitneyu. Effect size via manual Cohen's d computation. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.9.0 | nn.Linear probe, Adam optimizer, tensor ops | Already installed. Native GPU training. BF16 activation casting. |
| nnsight | >=0.5.15 | Forward passes for residual stream extraction | Already installed. Project standard for activation access. |
| scikit-learn | 1.8.0 | StratifiedKFold, roc_auc_score, classification metrics | Already installed. Standard for evaluation. |
| scipy | 1.17.0 | savgol_filter (derivative), curve_fit (sigmoid), mannwhitneyu (statistics) | Already installed. Standard scientific Python. |
| ruptures | 1.1.10 | PELT and BinSeg change-point detection | NOT installed. Must add to pyproject.toml dependencies. BSD-2-Clause license. |
| pandas | 3.0.0 | Parquet I/O, DataFrame interchange | Already installed. Project standard for tabular results. |
| safetensors | >=0.7.0 | Probe weight storage, activation storage | Already installed. Project standard for tensor I/O. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 2.4.1 | Array operations, label permutation for selectivity | Already installed. Used everywhere. |
| matplotlib | >=3.10.8 | Probe AUROC plots, confidence curve plots, transition boundary visualization | Already installed. For QA visualization. |
| tqdm | >=4.67.1 | Progress bars for training loops and batch processing | Already installed. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyTorch nn.Linear probe | sklearn LogisticRegression | sklearn is simpler but cannot run on GPU, slower for 4096-dim with thousands of samples. PyTorch probe trains on GPU natively -- locked decision. |
| ruptures PELT | Custom change-point detection | ruptures is battle-tested, handles penalty tuning, multiple cost models. Hand-rolling would miss edge cases. |
| scipy curve_fit for sigmoid | Manual gradient descent | curve_fit uses Levenberg-Marquardt, handles convergence robustly. No reason to reimplement. |

**Installation:**
```bash
# Add ruptures to pyproject.toml dependencies, then:
uv sync
```

## Architecture Patterns

### Recommended Project Structure
```
src/
    transition.py        # NEW: standalone transition detection module
scripts/
    layer_selection_pilot.py   # NEW: Stage 1 -- pilot run, all 36 layers, AUROC per layer
    train_probes.py            # NEW: Stage 2 -- train probes at selected layers, CV, selectivity
    detect_transitions.py      # NEW: Apply transition detection to confidence curves
    compute_cosine_sim.py      # NEW: Cosine similarity dynamics + statistical tests
    plot_phase3.py             # NEW: QA visualization (AUROC curves, confidence plots, boundaries)
configs/
    phase3.yaml                # NEW: Phase 3 configuration
```

### Pattern 1: Two-Stage Activation Pipeline
**What:** Run nnsight forward passes in two stages -- pilot (all layers, few traces) then full (selected layers, all traces). Discard pilot data after layer selection.
**When to use:** Always for Phase 3 (locked decision).
**Example:**
```python
# Stage 1: Pilot -- extract all 36 layers from ~100 traces
with model.trace(input_ids) as tracer:
    saved = {}
    for layer_idx in range(36):
        saved[layer_idx] = model.model.layers[layer_idx].output.save()

# After pilot: compute AUROC per layer, pick top-N
top_layers = select_top_layers(pilot_aurocs, n=6)

# Stage 2: Full -- extract only top-N layers from all traces
with model.trace(input_ids) as tracer:
    saved = {}
    for layer_idx in top_layers:
        saved[layer_idx] = model.model.layers[layer_idx].output.save()
```

### Pattern 2: Probe Training with Cross-Validation
**What:** Train nn.Linear(4096, 2) with Adam, early stopping, and 5-fold stratified CV. Report per-fold and mean AUROC.
**When to use:** For all probe training (real labels and random-label controls).
**Example:**
```python
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

probe = nn.Linear(4096, 2)
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=137)
fold_aurocs = []

for train_idx, val_idx in skf.split(X, y):
    # Reset probe weights each fold
    probe.reset_parameters()
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(100):
        probe.train()
        logits = probe(X_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val)
            val_loss = criterion(val_logits, y_val)
            val_probs = torch.softmax(val_logits, dim=-1)[:, 1]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = probe.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        final_probs = torch.softmax(probe(X_val), dim=-1)[:, 1].cpu().numpy()
    fold_aurocs.append(roc_auc_score(y_val.cpu(), final_probs))
```

### Pattern 3: Standalone Transition Detection Module
**What:** `src/transition.py` accepts arrays of (position, value) and returns segmentation results. Method-agnostic about the source of the curve.
**When to use:** For all transition detection -- probe confidence curves, force-answer accuracy curves, any (position, value) signal.
**Example:**
```python
# src/transition.py interface
def detect_transitions(
    positions: np.ndarray,
    values: np.ndarray,
    methods: list[str] = ["derivative", "pelt", "binseg", "sigmoid"],
) -> dict:
    """Run all specified detection methods on a (positions, values) curve.

    Returns dict keyed by method name, each containing:
    - transition_point: int or None (token position)
    - plateau_onset: int or None
    - segments: list of (start, end, label) tuples
    - classification: "clear" | "no-signal" | "noisy-signal"
    - method_specific: dict of method-specific metadata
    """
    results = {}
    for method in methods:
        if method == "derivative":
            results[method] = _derivative_method(positions, values)
        elif method == "pelt":
            results[method] = _pelt_method(positions, values)
        elif method == "binseg":
            results[method] = _binseg_method(positions, values)
        elif method == "sigmoid":
            results[method] = _sigmoid_method(positions, values)
    return results
```

### Pattern 4: BF16-to-Float32 Casting for Probe Training
**What:** Stored activations are BF16. Cast to float32 before probe training (numerical stability for gradient computation).
**When to use:** Always when loading activations for probe training or inference.
**Example:**
```python
# Load BF16 activations, cast to float32 for training
activations_bf16 = load_file("trace.safetensors")
X = activations_bf16["residual_layer_20"].float()  # BF16 -> float32

# After training, probe weights stay float32
# But save alongside metadata about which layer
```

### Pattern 5: Pre-Handoff Window Extraction
**What:** Use existing `src/tokens.py:get_pre_handoff_window()` to find the 3 tokens before final `</think>`. Extract activations at those positions for training data.
**When to use:** When building probe training data from traces.
**Example:**
```python
from src.tokens import get_pre_handoff_window

token_ids = trace_data["token_ids"]
window = get_pre_handoff_window(token_ids, window_size=3)
if window is None:
    # No </think> found, skip this trace
    continue

start_pos, end_pos = window  # end_pos is exclusive
# Extract activations at positions [start_pos, start_pos+1, ..., end_pos-1]
# from the residual stream tensor of shape [seq_len, hidden_dim]
training_vectors = residual[start_pos:end_pos]  # shape [3, 4096]
# Label: ground_truth mapped to 0/1
label = 1 if metadata["ground_truth"] == "Yes" else 0
labels = [label] * 3  # same label for all 3 positions
```

### Anti-Patterns to Avoid
- **Averaging transition points across traces:** Never do this. T_i is per-trace. Aggregate over distributions of T_i values.
- **Smoothing raw probe values for change-point detection:** PELT, BinSeg, and sigmoid fit operate on RAW probe confidence values. Smoothing is allowed ONLY for derivative computation.
- **Using stored Phase 2 activations:** Always use fresh nnsight forward passes for consistency (locked decision).
- **Training probes on incorrect traces:** Training data uses Pre-Handoff Window of CORRECT traces only. The probe learns what "definitively knowing the answer" looks like.
- **Probing the `<think>` token itself:** Skip position k=0 (the `<think>` token is a geometric outlier). Start probing at k=1 (first natural language token).
- **Using output token (Rho/Tau) as probe label:** Probe label is `ground_truth` (Yes/No), NOT the output token. The swapped-label protocol ensures zero correlation.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Change-point detection | Custom CUSUM or threshold logic | `ruptures` PELT/BinSeg | Handles penalty tuning, multiple cost models, edge cases. Well-tested library with O(n) PELT. |
| Sigmoid curve fitting | Manual gradient descent on logistic params | `scipy.optimize.curve_fit` | Levenberg-Marquardt converges robustly, handles bounds, returns covariance matrix for R-squared. |
| Derivative smoothing | Moving average or custom polynomial fit | `scipy.signal.savgol_filter(deriv=1)` | Standard Savitzky-Golay with built-in derivative computation, polynomial order control. |
| Stratified cross-validation | Manual fold splitting | `sklearn.model_selection.StratifiedKFold` | Guarantees class balance in each fold, deterministic with random_state. |
| AUROC computation | Manual TPR/FPR integration | `sklearn.metrics.roc_auc_score` | Handles edge cases (all-one-class, ties), numerically stable. |
| Mann-Whitney U test | Manual rank-sum computation | `scipy.stats.mannwhitneyu` | Correct handling of ties, exact/approximate p-values. |

**Key insight:** The transition detection methods are deceptively complex. PELT requires penalty parameter selection (sweep or BIC), sigmoid fit requires good initial guesses and convergence handling, and derivative methods require appropriate smoothing window selection. Using established libraries avoids dozens of edge cases.

## Common Pitfalls

### Pitfall 1: OOM During Full 36-Layer Pilot Extraction
**What goes wrong:** Extracting residual stream at all 36 layers for long traces (3000+ tokens) exhausts A100 80GB VRAM.
**Why it happens:** Each layer stores [1, seq_len, 4096] in BF16 = ~24.6 MB per layer for 3000 tokens. 36 layers = ~886 MB per trace. With nnsight overhead, this approaches memory limits.
**How to avoid:** Use the existing chunked capture pattern from Phase 2: capture 12 layers per nnsight pass, 3 passes for all 36 layers. Move tensors to CPU immediately via `.save()` and `.detach().cpu()`.
**Warning signs:** CUDA OOM errors during pilot extraction, especially on longer traces.

### Pitfall 2: Probe Overfitting on Small Training Sets
**What goes wrong:** With d=4096 features and potentially only ~300-500 training samples (correct traces x 3 Pre-Handoff Window positions), the probe memorizes training data.
**Why it happens:** High dimension-to-sample ratio. Linear probes are less prone than neural nets, but L2 regularization is still essential.
**How to avoid:** L2 regularization via weight_decay (locked at 1e-4 or similar), early stopping on validation loss (patience=10), 5-fold CV to detect overfitting. The selectivity control (random-label AUROC) catches this: if random AUROC > 0.60, the probe is finding spurious signal.
**Warning signs:** Training AUROC near 1.0 but validation AUROC much lower. Random-label AUROC approaching real AUROC.

### Pitfall 3: Numerical Issues with BF16 Activations
**What goes wrong:** BF16 has limited precision (7-8 decimal digits). Probe training on BF16 produces unstable gradients.
**Why it happens:** BF16 is designed for forward inference, not backward pass gradient computation.
**How to avoid:** Always cast activations to float32 before probe training: `X = activations_bf16.float()`. Probe weights train in float32. Only store final probe weights in float32 (or BF16 for space if needed). This is already the pattern used in the project (float32 softmax for entropy).
**Warning signs:** NaN losses during training, wildly oscillating AUROC across folds.

### Pitfall 4: Ruptures Penalty Miscalibration
**What goes wrong:** PELT with wrong penalty value either finds zero change points (penalty too high) or hundreds of spurious ones (penalty too low).
**Why it happens:** The penalty parameter balances model complexity against fit quality. No universal default works for all signals.
**How to avoid:** Sweep penalty values (e.g., `[1, 5, 10, 20, 50, 100]`) and report results for each. Use BinSeg with a known number of expected change points (1-2 for our probe curves) as a cross-check. For probe confidence curves that should have at most 1-2 transitions, PELT with `model="l2"` and moderate penalty is a good starting point. Report the penalty used for each trace.
**Warning signs:** PELT returning 0 change points for traces that visually have a clear transition. Or returning 10+ change points for a smooth S-curve.

### Pitfall 5: Sigmoid Fit Failure on Non-Sigmoidal Traces
**What goes wrong:** `curve_fit` raises `RuntimeError` or `OptimalError` when the probe confidence curve is not sigmoidal (e.g., noisy, flat, or multi-modal).
**Why it happens:** Not all traces have clean S-shaped confidence curves. Some are noisy, some never transition.
**How to avoid:** Wrap `curve_fit` in try/except. Use `maxfev=5000` for more iterations. Provide reasonable initial guesses (t0 = midpoint, k = 1.0, L = max(values)). If fit fails, classify trace as "noisy-signal" rather than crashing. Report R-squared for successful fits and sweep threshold (0.75, 0.80, 0.85, 0.90, 0.95) per locked decision.
**Warning signs:** Many traces failing sigmoid fit, or R-squared values clustering near 0.5.

### Pitfall 6: nnsight Layer Access Order
**What goes wrong:** Accessing layers in non-sequential order within a single `model.trace()` context causes nnsight `OutOfOrderError`.
**Why it happens:** nnsight requires modules to be accessed in the order they appear in the model's forward pass.
**How to avoid:** Always iterate `for idx in range(N)` or `for idx in sorted(layer_subset)` within a trace context. This is already handled in `src/model.py:capture_activations()` and should be preserved in any new activation extraction code.
**Warning signs:** `OutOfOrderError` exceptions from nnsight during forward passes.

### Pitfall 7: Incorrect Pre-Handoff Window for Multi-Block Traces
**What goes wrong:** `get_pre_handoff_window()` finds the FINAL `</think>` token. For multi-block traces, this might not be the right window.
**Why it happens:** Multi-block traces have multiple `<think>...</think>` regions.
**How to avoid:** Filter to single-block traces only for training data (existing pattern from Phase 2.2). The filtering code in `force_answer_experiment.py` already does this: check `meta["num_think_blocks"] == 1`.
**Warning signs:** Unexpected activation patterns in training data. Low probe AUROC despite clean single-block traces working well.

## Code Examples

### Probe Training Loop with Early Stopping
```python
# Source: PyTorch docs + project conventions
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

def train_probe(X_train, y_train, X_val, y_val, device, lr=1e-3, weight_decay=1e-4, max_epochs=100, patience=10):
    """Train a linear probe on GPU with early stopping.

    Args:
        X_train: Float32 tensor [n_train, 4096] on device.
        y_train: Long tensor [n_train] (0 or 1) on device.
        X_val: Float32 tensor [n_val, 4096] on device.
        y_val: Long tensor [n_val] on device.
        device: torch.device.
        lr: Learning rate.
        weight_decay: L2 regularization strength.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience.

    Returns:
        Tuple of (trained probe, best validation AUROC).
    """
    probe = nn.Linear(4096, 2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        probe.train()
        logits = probe(X_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val)
            val_loss = criterion(val_logits, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        val_probs = torch.softmax(probe(X_val), dim=-1)[:, 1].cpu().numpy()
    auroc = roc_auc_score(y_val.cpu().numpy(), val_probs)
    return probe, auroc
```

### Transition Detection: Derivative Method
```python
# Source: scipy.signal.savgol_filter docs
from scipy.signal import savgol_filter
import numpy as np

def derivative_transition(positions, values, window_length=11, polyorder=3, slope_threshold=0.01):
    """Detect transition via first derivative of smoothed confidence curve.

    Smoothing is applied ONLY for derivative computation (locked decision).
    The raw values are used for plateau detection.

    Args:
        positions: Array of token positions.
        values: Array of probe confidence values (raw, unsmoothed).
        window_length: Savitzky-Golay window (must be odd, <= len(values)).
        polyorder: Polynomial order for Savitzky-Golay.
        slope_threshold: Minimum slope to consider as transition onset.

    Returns:
        Dict with transition_point, plateau_onset, classification, etc.
    """
    if len(values) < window_length:
        window_length = max(5, len(values) | 1)  # ensure odd

    # Compute smoothed first derivative
    dx = np.mean(np.diff(positions)) if len(positions) > 1 else 1.0
    derivative = savgol_filter(values, window_length, polyorder, deriv=1, delta=dx)

    # Find first point where slope exceeds threshold
    onset_idx = None
    for i, slope in enumerate(derivative):
        if slope > slope_threshold:
            onset_idx = i
            break

    # Classification based on raw values
    max_confidence = np.max(values)
    if max_confidence < 0.70:
        classification = "no-signal"
    elif onset_idx is None:
        classification = "noisy-signal"
    else:
        # Check if plateau exists after onset
        post_onset = values[onset_idx:]
        if np.mean(post_onset[-10:]) > 0.90 if len(post_onset) >= 10 else False:
            classification = "clear"
        else:
            classification = "noisy-signal"

    return {
        "transition_point": int(positions[onset_idx]) if onset_idx is not None else None,
        "classification": classification,
        "derivative": derivative,
    }
```

### Transition Detection: PELT/BinSeg via Ruptures
```python
# Source: ruptures official docs (https://centre-borelli.github.io/ruptures-docs/)
import ruptures as rpt
import numpy as np

def pelt_transition(positions, values, pen=10, model="l2"):
    """Detect change points in probe confidence curve using PELT.

    Runs on RAW probe values (no smoothing -- locked decision).

    Args:
        positions: Array of token positions.
        values: Array of probe confidence values.
        pen: PELT penalty parameter. Sweep recommended.
        model: Cost model ("l2", "rbf").

    Returns:
        Dict with change_points, transition_point, classification.
    """
    signal = values.reshape(-1, 1)  # ruptures expects 2D
    algo = rpt.Pelt(model=model, min_size=5).fit(signal)
    change_points = algo.predict(pen=pen)

    # change_points is list of indices (last element is always len(signal))
    # Remove the final element (end-of-signal marker)
    actual_cps = [cp for cp in change_points if cp < len(values)]

    if not actual_cps:
        return {
            "change_points": [],
            "transition_point": None,
            "classification": "no-signal" if np.max(values) < 0.70 else "noisy-signal",
        }

    # First change point is candidate transition
    first_cp_idx = actual_cps[0]
    return {
        "change_points": actual_cps,
        "transition_point": int(positions[first_cp_idx]) if first_cp_idx < len(positions) else None,
        "classification": "clear",
    }


def binseg_transition(positions, values, n_bkps=2, model="l2"):
    """Detect change points using Binary Segmentation with known count.

    Args:
        positions: Array of token positions.
        values: Array of probe confidence values.
        n_bkps: Expected number of breakpoints (1-2 for our case).
        model: Cost model.

    Returns:
        Dict with change_points, transition_point, classification.
    """
    signal = values.reshape(-1, 1)
    algo = rpt.Binseg(model=model, min_size=5).fit(signal)
    change_points = algo.predict(n_bkps=n_bkps)
    actual_cps = [cp for cp in change_points if cp < len(values)]

    if not actual_cps:
        return {
            "change_points": [],
            "transition_point": None,
            "classification": "no-signal" if np.max(values) < 0.70 else "noisy-signal",
        }

    first_cp_idx = actual_cps[0]
    return {
        "change_points": actual_cps,
        "transition_point": int(positions[first_cp_idx]) if first_cp_idx < len(positions) else None,
        "classification": "clear",
    }
```

### Transition Detection: Sigmoid Fit
```python
# Source: scipy.optimize.curve_fit docs
from scipy.optimize import curve_fit
import numpy as np

def _sigmoid(x, L, k, x0, b):
    """Standard logistic sigmoid: L / (1 + exp(-k*(x - x0))) + b"""
    return L / (1.0 + np.exp(-k * (x - x0))) + b

def sigmoid_transition(positions, values, r2_threshold=0.85):
    """Fit logistic sigmoid to probe confidence curve.

    R-squared threshold sweep (0.75, 0.80, 0.85, 0.90, 0.95)
    reported per locked decision.

    Args:
        positions: Array of token positions (float for fitting).
        values: Array of probe confidence values.
        r2_threshold: Minimum R-squared for "clear" classification.

    Returns:
        Dict with transition_point (t0), sharpness (k), r_squared,
        classification, fitted_params.
    """
    x = positions.astype(float)
    y = values.astype(float)

    # Initial guesses
    p0 = [
        np.max(y) - np.min(y),  # L: amplitude
        1.0 / (np.std(x) + 1e-8),  # k: steepness
        np.median(x),  # x0: midpoint
        np.min(y),  # b: baseline
    ]

    try:
        popt, pcov = curve_fit(
            _sigmoid, x, y, p0=p0,
            maxfev=5000,
            bounds=([0, 0, x.min(), -0.1], [1.5, np.inf, x.max(), 0.5]),
        )
        y_pred = _sigmoid(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        classification = "clear" if r_squared >= r2_threshold else "noisy-signal"
        transition_point = int(round(popt[2]))  # x0 = midpoint

        return {
            "transition_point": transition_point,
            "sharpness": float(popt[1]),  # k
            "r_squared": float(r_squared),
            "classification": classification,
            "fitted_params": {"L": popt[0], "k": popt[1], "x0": popt[2], "b": popt[3]},
        }

    except (RuntimeError, ValueError) as e:
        return {
            "transition_point": None,
            "sharpness": None,
            "r_squared": None,
            "classification": "noisy-signal",
            "fitted_params": None,
            "error": str(e),
        }
```

### Cosine Similarity Dynamics
```python
# Source: PyTorch docs, scipy.stats docs
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
import numpy as np

def compute_cosine_similarity_dynamics(residual_stream, think_mask):
    """Compute step-wise cosine similarity S_t = cos(h_t, h_{t-1}).

    Args:
        residual_stream: Float32 tensor [seq_len, hidden_dim].
        think_mask: Boolean list/array marking positions inside think blocks.

    Returns:
        Array of S_t values at eligible positions.
    """
    think_positions = [i for i, m in enumerate(think_mask) if m]
    if len(think_positions) < 2:
        return np.array([])

    similarities = []
    for i in range(1, len(think_positions)):
        pos_t = think_positions[i]
        pos_prev = think_positions[i - 1]
        h_t = residual_stream[pos_t].unsqueeze(0)
        h_prev = residual_stream[pos_prev].unsqueeze(0)
        sim = F.cosine_similarity(h_t, h_prev, dim=-1).item()
        similarities.append(sim)

    return np.array(similarities)


def compare_pre_post_cosine(s_t_values, transition_idx):
    """Compare pre-T vs post-T cosine similarity distributions.

    Args:
        s_t_values: Array of cosine similarities (one per consecutive pair).
        transition_idx: Index into s_t_values marking the transition.

    Returns:
        Dict with pre/post means, Mann-Whitney U statistic, p-value, Cohen's d.
    """
    pre = s_t_values[:transition_idx]
    post = s_t_values[transition_idx:]

    if len(pre) < 2 or len(post) < 2:
        return {"insufficient_data": True}

    stat, pval = mannwhitneyu(pre, post, alternative='two-sided')

    # Cohen's d
    pooled_std = np.sqrt(
        ((len(pre) - 1) * np.std(pre, ddof=1)**2 + (len(post) - 1) * np.std(post, ddof=1)**2)
        / (len(pre) + len(post) - 2)
    )
    cohens_d = (np.mean(post) - np.mean(pre)) / pooled_std if pooled_std > 0 else 0.0

    return {
        "pre_mean": float(np.mean(pre)),
        "post_mean": float(np.mean(post)),
        "mann_whitney_u": float(stat),
        "p_value": float(pval),
        "cohens_d": float(cohens_d),
        "n_pre": len(pre),
        "n_post": len(post),
    }
```

### Selectivity Control (Random-Label Probes)
```python
# Source: REQ-P2, project conventions
import numpy as np

def train_selectivity_controls(X, y, device, n_permutations=10, seed=137):
    """Train probes on randomly permuted labels as selectivity controls.

    Args:
        X: Float32 tensor [n_samples, 4096].
        y: Long tensor [n_samples] (0 or 1).
        device: torch.device.
        n_permutations: Number of random permutations.
        seed: Base seed for reproducibility.

    Returns:
        List of AUROC values from random-label probes.
    """
    rng = np.random.RandomState(seed)
    random_aurocs = []

    for perm_idx in range(n_permutations):
        y_shuffled = y.cpu().numpy().copy()
        rng.shuffle(y_shuffled)
        y_shuffled = torch.tensor(y_shuffled, dtype=torch.long, device=device)

        # Train with same procedure as real probes (5-fold CV)
        # ... (same cross-validation loop as real probes)
        fold_aurocs = []  # collect per-fold AUROC
        mean_auroc = np.mean(fold_aurocs)
        random_aurocs.append(mean_auroc)

    return random_aurocs
```

### Activation Extraction via nnsight (Phase 3 Pattern)
```python
# Source: project src/model.py patterns, nnsight docs
import torch
from nnsight import LanguageModel

def extract_residual_at_layers(
    model: LanguageModel,
    token_ids: torch.Tensor,
    layer_indices: list[int],
) -> dict[int, torch.Tensor]:
    """Extract residual stream at specified layers via nnsight trace.

    Args:
        model: nnsight LanguageModel wrapping Qwen3-8B.
        token_ids: Token IDs tensor, shape [1, seq_len].
        layer_indices: Sorted list of layer indices to capture.

    Returns:
        Dict mapping layer index to CPU float32 tensor [seq_len, hidden_dim].
    """
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)

    saved = {}
    with model.trace(token_ids) as tracer:
        # Access ALL layers in sequential order to avoid OutOfOrderError
        for idx in range(36):
            if idx in set(layer_indices):
                saved[idx] = model.model.layers[idx].output.save()

    # Convert to CPU float32
    results = {}
    for idx in layer_indices:
        tensor = saved[idx].detach().cpu()
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # [1, seq, hidden] -> [seq, hidden]
        results[idx] = tensor.float()  # BF16 -> float32

    return results
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Train probes on Yes/No token predictions | Train probes on abstract validity labels with swapped Rho/Tau protocol | Phase 2 (2026-02) | Eliminates confound where probe detects token mention, not abstract reasoning |
| Sustained-K threshold for transition detection | Multiple detection methods (derivative, PELT, sigmoid) run in parallel | Phase 3 CONTEXT.md decision | More robust detection, comparison selects best method post-hoc |
| Store Phase 2 activations and reuse | Fresh nnsight forward passes for Phase 3 | Phase 3 CONTEXT.md decision | Ensures consistency, avoids stale activation format issues |
| sklearn LogisticRegression for probes | PyTorch nn.Linear(4096, 2) on GPU | Phase 3 standard | GPU-native training, natural integration with PyTorch activation tensors |

**Deprecated/outdated:**
- The original REQUIREMENTS.md (REQ-P3) specified sustained-K=5 at threshold 0.95 as primary method and sliding-window as sensitivity check. The CONTEXT.md overrides this with three parallel methods (derivative, PELT/BinSeg, sigmoid). The planner should follow CONTEXT.md, not the original REQ-P3 detection methods.
- REQ-P5 (Cox 2x2 Classification) is dropped per CONTEXT.md Decision 7.

## Open Questions

1. **How many top-N layers to keep after pilot?**
   - What we know: 36 layers total. Probe AUROC will peak in mid-to-late layers (typical for LLMs). REQ-P4 needs "best-probe-AUROC layer plus 2-3 additional layers."
   - What's unclear: Exact N. Too few (3) might miss interesting layer dynamics. Too many (12) increases storage and compute.
   - Recommendation: Default to N=6 (best + 2 above + 2 below + one early). Configurable in YAML. Pilot AUROC curve will guide the final choice.

2. **PELT penalty parameter for probe confidence curves**
   - What we know: Probe confidence curves are 1D signals in [0, 1] range. Expected shape: low-then-high (S-curve) or flat (no transition). PELT with model="l2" is appropriate.
   - What's unclear: Optimal penalty value. Too low = spurious change points. Too high = no detection.
   - Recommendation: Sweep pen in [1, 5, 10, 20, 50, 100] on pilot traces. Report results for all. Pick the value that best matches visual inspection on a sample of traces.

3. **Savitzky-Golay window_length for derivative method**
   - What we know: Must be odd. Must be <= signal length. Larger windows = more smoothing. Typical think blocks are 200-3000 tokens long.
   - What's unclear: Optimal window for our signal characteristics.
   - Recommendation: Default window_length=21, polyorder=3. For short traces (<50 tokens), reduce to window_length=11 or 7. Make configurable.

4. **Number of traces for the pilot stage**
   - What we know: Decision says "~50-100 traces, stratified across datasets and correctness."
   - What's unclear: Whether 50 is enough for stable AUROC estimates across 36 layers.
   - Recommendation: Use 100 traces (stratified: ~33 per dataset, balanced correct/incorrect). With 3 Pre-Handoff Window positions each = 300 training samples, enough for stable probe training.

5. **Pooled vs per-dataset probe storage**
   - What we know: Both per-dataset (108 total) and pooled (36 total) probes must be trained. All 36 layers per dataset saved as safetensors.
   - What's unclear: Whether to save probes from ALL folds or just retrain on full data after CV.
   - Recommendation: Use CV for AUROC reporting only. After CV, retrain on full training data (all folds combined) to produce the final probe weights. Save these final weights. This is standard in the probing literature.

## Validation Architecture

> Skipped -- workflow.nyquist_validation is not true in .planning/config.json.

## Sources

### Primary (HIGH confidence)
- [ruptures official docs](https://centre-borelli.github.io/ruptures-docs/) -- PELT, BinSeg API, cost models, penalty selection
- [ruptures GitHub](https://github.com/deepcharles/ruptures) -- Version 1.1.10, BSD-2 license, API patterns
- [scipy.signal.savgol_filter docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) -- SciPy 1.17.0, derivative computation via deriv parameter
- [scipy.optimize.curve_fit docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) -- Non-linear least squares, sigmoid fitting
- [sklearn StratifiedKFold docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) -- Cross-validation with class balance
- [sklearn roc_auc_score docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) -- AUROC computation
- [nnsight walkthrough](https://nnsight.net/notebooks/tutorials/get_started/walkthrough/) -- Activation extraction patterns, .save() semantics
- Project codebase: `src/model.py`, `src/tokens.py`, `src/storage.py` -- Existing activation capture, Pre-Handoff Window, safetensors I/O patterns

### Secondary (MEDIUM confidence)
- [ruptures PyPI](https://pypi.org/project/ruptures/) -- Version 1.1.10, September 2025 release, CPython 3.9-3.13 support
- [scipy.stats.mannwhitneyu docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html) -- Mann-Whitney U test for pre/post comparison
- Research on linear probes for transformer interpretability: "A Single Direction of Truth" (arXiv 2507.23221) -- confirms linear probes on residual stream for LLM analysis

### Tertiary (LOW confidence)
- None -- all critical claims verified with official documentation or project codebase.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries either already installed (torch, sklearn, scipy, pandas) or well-documented with clear API (ruptures). Versions confirmed.
- Architecture: HIGH -- patterns derive directly from existing project code (src/model.py activation capture, src/tokens.py Pre-Handoff Window) extended for probe training. No novel architectural decisions.
- Pitfalls: HIGH -- OOM, BF16 precision, nnsight ordering, probe overfitting all observed in prior phases. Ruptures penalty calibration documented in official docs.

**Research date:** 2026-02-26
**Valid until:** 2026-03-26 (stable domain, no fast-moving library changes expected)
