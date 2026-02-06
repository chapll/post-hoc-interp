# Architecture Patterns

**Domain:** Mechanistic interpretability pipeline for reasoning traces (CS224N)
**Researched:** 2026-02-04
**Overall confidence:** HIGH (nnsight, Qwen3, storage formats all well-documented)

---

## Table of Contents

1. [Qwen3-8B Reference Architecture](#qwen3-8b-reference-architecture)
2. [Memory Budget Analysis](#memory-budget-analysis)
3. [Component Map](#component-map)
4. [Data Flow](#data-flow)
5. [Component Details](#component-details)
6. [Storage Architecture](#storage-architecture)
7. [Multi-GPU Design](#multi-gpu-design)
8. [Configuration Strategy](#configuration-strategy)
9. [Build Order](#build-order)
10. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)

---

## Qwen3-8B Reference Architecture

These numbers drive every memory estimate below. Sourced from the actual
config.json on HuggingFace (not the generic `Qwen3Config` defaults, which
differ).

| Parameter | Value |
|-----------|-------|
| hidden_size (d_model) | 4096 |
| num_hidden_layers | 36 |
| intermediate_size | 12288 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| vocab_size | 151936 |
| max_position_embeddings | 40960 |
| torch_dtype | bfloat16 |
| model_type | qwen3 |

Sources:
- https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3-8B

---

## Memory Budget Analysis

### A100 80GB: Single-GPU Trace Generation

The critical question is whether Qwen3-8B + full activation cache fits on one
A100 80GB. The answer is **yes, comfortably, for single-sequence inference with
full residual stream capture**.

#### Model Weights (BF16)

```
8.2B params * 2 bytes/param = ~16.4 GB
```

#### KV Cache (per sequence, BF16)

For a 2048-token sequence (typical reasoning trace length):

```
Per layer: 2 (K+V) * 8 (kv_heads) * 128 (head_dim) * 2048 (seq_len) * 2 (bytes)
         = 2 * 8 * 128 * 2048 * 2 = 8,388,608 bytes = 8 MB
All layers: 36 * 8 MB = 288 MB
```

For a 4096-token sequence: ~576 MB.

#### Residual Stream Cache (the activations we are collecting)

Per token, per layer, we want the residual stream vector (d_model = 4096).

```
Per layer per token (FP32): 4096 * 4 bytes = 16 KB
Per layer per token (BF16): 4096 * 2 bytes = 8 KB
```

For one full trace (2048 tokens, all 36 layers, FP32):

```
2048 * 36 * 4096 * 4 = 1,207,959,552 bytes = ~1.15 GB
```

For one full trace (BF16): ~0.58 GB.

#### Activation Memory During Forward Pass

PyTorch intermediate activations during the forward pass (attention scores,
MLP intermediates, etc.) for a single sequence:

```
Rough estimate: ~2-4 GB for a single 2048-token forward pass at BF16
with torch.no_grad() (no backward graph stored)
```

#### Total Per-Sequence Budget

| Component | BF16 Model, FP32 Cache | BF16 Model, BF16 Cache |
|-----------|----------------------|----------------------|
| Model weights | 16.4 GB | 16.4 GB |
| KV cache (2048 tokens) | 0.3 GB | 0.3 GB |
| Residual stream cache | 1.15 GB | 0.58 GB |
| Forward pass activations | ~3 GB | ~3 GB |
| PyTorch/CUDA overhead | ~1 GB | ~1 GB |
| **Total** | **~22 GB** | **~21 GB** |
| **Headroom on 80 GB** | **~58 GB** | **~59 GB** |

**Conclusion:** Model parallelism is NOT needed for trace generation. A single
A100 80GB has ~58 GB of headroom after loading the model and capturing one full
trace. This headroom allows batching multiple sequences (each additional 2048-token
sequence adds ~4-5 GB with KV cache + residual cache + forward activations).

#### Practical Batch Sizes for Trace Generation

```
Available: ~58 GB headroom
Per additional sequence (2048 tokens): ~4-5 GB
Safe batch size: 8-10 sequences per GPU
Conservative batch size: 4-6 sequences per GPU
```

With 4 GPUs doing independent inference, that is 16-40 sequences in parallel.

### Disk Storage Budget

This is the dominant resource constraint.

#### Per-Problem Storage (20 traces, FP32 residual stream)

Assuming average 2048 tokens per trace:

```
Per trace: 2048 * 36 * 4096 * 4 bytes = ~1.15 GB
Per problem (20 traces): 20 * 1.15 GB = ~23 GB
300 problems: 300 * 23 GB = ~6.9 TB
```

#### Per-Problem Storage (20 traces, FP16 residual stream)

```
Per trace: 2048 * 36 * 4096 * 2 bytes = ~0.58 GB
Per problem (20 traces): 20 * 0.58 GB = ~11.5 GB
300 problems: 300 * 11.5 GB = ~3.5 TB
```

#### Realistic Mitigation: Store Only Key Layers

For probing, you likely do not need all 36 layers. Storing every 4th layer
(layers 0, 4, 8, ..., 32, 35 = 10 layers) cuts storage to ~28% of full:

```
300 problems * 20 traces: ~1.0 TB (FP16, 10 layers)
```

Even more aggressive: store only the last-token residual stream at each layer
for probe training (since binary verification tasks care about the final
prediction). That reduces token dimension from ~2048 to 1:

```
300 * 20 * 1 * 36 * 4096 * 2 = ~1.6 GB total (trivial)
```

**Recommendation:** Store last-token activations for all layers (tiny) PLUS
full-sequence activations for a configurable subset of layers (for transition
detection). Default to layers [0, 4, 8, 12, 16, 20, 24, 28, 32, 35].

#### 5-Trace vs 20-Trace Configurations

| Config | Traces | Purpose | Storage (FP16, 10 layers, full seq) |
|--------|--------|---------|-------------------------------------|
| Probing | 5 | Train/eval linear probes | ~250 GB |
| Differential | 20 | Cross-trace consistency analysis | ~1.0 TB |

Run probing config first. Only generate the additional 15 traces per problem
if probing results warrant deeper analysis.

---

## Component Map

```
src/experimental/gorn/
|
|-- configs/                    # YAML configs (NOT Hydra -- see Configuration section)
|   |-- base.yaml              # Shared defaults
|   |-- trace_gen.yaml         # Trace generation settings
|   |-- probing.yaml           # Probe training settings
|   |-- steering.yaml          # Steering experiment settings
|   `-- figures.yaml           # Figure generation settings
|
|-- data/
|   |-- datasets/              # Raw problem sets (JSONL per dataset)
|   |-- traces/                # Generated activation traces (safetensors)
|   `-- results/               # Probe accuracies, steering metrics, etc.
|
|-- experiments/
|   |-- trace_gen.py           # Script: generate traces with nnsight
|   |-- train_probes.py        # Script: train linear probes on activations
|   |-- transition_detect.py   # Script: find where model "shifts" reasoning
|   |-- differential.py        # Script: cross-trace consistency analysis
|   |-- steering.py            # Script: activation patching / steering
|   |-- evaluate.py            # Script: end-to-end evaluation
|   `-- figures.py             # Script: generate paper figures
|
|-- lib/
|   |-- model.py               # Model loading + nnsight wrapping
|   |-- trace_store.py         # Activation storage read/write
|   |-- probe.py               # Linear probe (PyTorch nn.Linear, not sklearn)
|   |-- metrics.py             # Accuracy, logit diff, etc.
|   |-- datasets.py            # Dataset loading + formatting
|   `-- utils.py               # Seed, device selection, logging helpers
|
`-- README.md
```

### Component Boundaries

| Component | Responsibility | Depends On |
|-----------|---------------|------------|
| `lib/model.py` | Load Qwen3-8B via HuggingFace, wrap with nnsight LanguageModel | transformers, nnsight |
| `lib/trace_store.py` | Save/load activation tensors to disk | safetensors |
| `lib/probe.py` | Linear probe definition + training loop | torch |
| `lib/metrics.py` | Compute accuracy, logit diff, statistical tests | numpy, scipy |
| `lib/datasets.py` | Load problem sets, format prompts, manage splits | json, tokenizer |
| `experiments/trace_gen.py` | Run inference, collect residual stream via nnsight | model, trace_store, datasets |
| `experiments/train_probes.py` | Train probes per-layer, evaluate, report | probe, trace_store, metrics |
| `experiments/transition_detect.py` | Analyze layer-by-layer probe accuracy curves | trace_store, metrics |
| `experiments/differential.py` | Compare activations across traces of same problem | trace_store, metrics |
| `experiments/steering.py` | Activation patching and steering interventions | model, trace_store, datasets |
| `experiments/evaluate.py` | End-to-end pipeline evaluation | all lib modules |
| `experiments/figures.py` | Generate matplotlib figures for paper | matplotlib, results data |
| `configs/` | YAML configuration files | omegaconf |

---

## Data Flow

```
                    [Problem Datasets]
                          |
                          v
                  +---------------+
                  | trace_gen.py  |  <-- Qwen3-8B + nnsight
                  +---------------+
                          |
                    Activation Traces
                    (safetensors on disk)
                          |
            +-------------+-------------+
            |             |             |
            v             v             v
    +-------------+ +-----------+ +-------------+
    | train_      | | transition| | differential|
    | probes.py   | | _detect.py| | .py         |
    +-------------+ +-----------+ +-------------+
            |             |             |
            v             v             v
      Probe weights   Layer curves   Consistency
      + accuracies    + transitions  scores
            |             |             |
            +------+------+------+------+
                   |             |
                   v             v
            +-------------+ +----------+
            | steering.py | | figures.py|
            +-------------+ +----------+
                   |             |
                   v             v
             Causal results   Paper figures
                   |
                   v
            +-------------+
            | evaluate.py |
            +-------------+
```

### Data Flow Details

1. **Problem datasets** are JSONL files with fields: `problem_id`, `question`, `answer`, `dataset_name`, `verification_type`.

2. **Trace generation** produces one directory per problem:
   ```
   data/traces/{dataset}/{problem_id}/
       metadata.json          # Problem info, generation params, token count
       trace_{00..19}.safetensors  # Activation tensors per trace
   ```

3. **Each trace safetensors file** contains named tensors:
   ```
   "residual_layer_00": shape [seq_len, 4096]   # FP16
   "residual_layer_04": shape [seq_len, 4096]   # (subset of layers)
   ...
   "residual_layer_35": shape [seq_len, 4096]
   "last_token_all_layers": shape [36, 4096]    # Always store all layers at last token
   "generated_token_ids": shape [seq_len]        # INT64
   "answer_correct": scalar bool                 # Whether this trace got the right answer
   ```

4. **Probe training** reads `last_token_all_layers` from all traces across all problems, trains one probe per layer, outputs accuracy curves.

5. **Transition detection** uses layer-wise probe accuracies to find where the model shifts from "not encoding the answer" to "encoding the answer."

6. **Differential analysis** compares `residual_layer_XX` across the 20 traces for each problem, measuring consistency (cosine similarity, variance) at identified transition layers.

7. **Steering** loads the model again, computes a steering vector from differential analysis, and applies it via nnsight during generation.

---

## Component Details

### 1. Model Loading (`lib/model.py`)

```python
from nnsight import LanguageModel

def load_model(model_name: str, device: str = "auto", dtype: str = "bfloat16"):
    """Load Qwen3-8B wrapped with nnsight for activation access."""
    model = LanguageModel(
        model_name,
        device_map=device,
        torch_dtype=dtype,
    )
    return model
```

nnsight's `LanguageModel` wraps `AutoModelForCausalLM.from_pretrained()` under the
hood. Any HuggingFace keyword argument is forwarded. This works with Qwen3 because
Qwen3 is a standard HuggingFace transformers model (requires transformers >= 4.51.0).

**Key nnsight version note:** Use nnsight >= 0.5 for the `tracer.cache()` API.
As of January 2026, the latest version is 0.5.15.

### 2. Trace Generation (`experiments/trace_gen.py`)

This is the most memory-sensitive and time-consuming component.

**Architecture:**

```python
# Per-GPU worker (simplified)
def generate_traces(model, problems, config):
    for problem in problems:
        for trace_idx in range(config.num_traces):
            with model.trace(problem.prompt, temperature=config.temperature) as tracer:
                # Collect residual stream at configured layers
                activations = {}
                for layer_idx in config.capture_layers:
                    layer = model.model.layers[layer_idx]
                    activations[f"residual_layer_{layer_idx:02d}"] = layer.output[0][0].save()

                # Always collect all layers at last token position
                all_last = []
                for layer_idx in range(36):
                    layer = model.model.layers[layer_idx]
                    all_last.append(layer.output[0][0, -1, :].save())

                # Get generated tokens
                token_ids = model.output.logits.argmax(dim=-1)[0].save()

            # After trace context exits, tensors are materialized
            save_trace(problem.id, trace_idx, activations, all_last, token_ids)
```

**Qwen3 layer access pattern for nnsight:**
- Model structure: `model.model.layers[i]` for decoder layers
- Each layer's `output` is a tuple; `output[0]` is the hidden state tensor
  with shape `[batch, seq_len, hidden_size]`
- The embedding layer: `model.model.embed_tokens`
- The final norm: `model.model.norm`
- The LM head: `model.lm_head`

**Critical design decision: greedy vs sampling.**

For binary verification tasks, the model generates reasoning traces. Use
`temperature > 0` (e.g., 0.6) to get diverse traces across the 20 samples.
Use `temperature = 0` for a single canonical trace if needed.

For nnsight, generation uses `model.generate()` rather than `model.trace()`:

```python
with model.generate(prompt, max_new_tokens=max_tokens, temperature=temp) as generator:
    # Access activations during generation
    with generator.iter[:] as step:
        # Collect per-step activations
        ...
```

However, generation-time activation collection is complex because nnsight's
`generate()` iterates token-by-token. For full residual stream capture, it is
simpler and more memory-efficient to:

1. Generate the full response text first (using standard HuggingFace generate).
2. Then do a single forward pass with nnsight over the full (prompt + response)
   sequence to collect all activations at once.

This two-pass approach is recommended because:
- Generation needs KV cache and sampling logic
- Activation collection needs only a single forward pass with no grad
- Separating them avoids the complexity of nnsight's generate() iteration API
- Memory is dominated by the model weights anyway; the extra forward pass is cheap

### 3. Trace Storage (`lib/trace_store.py`)

**Recommended format: safetensors.**

Rationale over alternatives:

| Format | Pros | Cons | Verdict |
|--------|------|------|---------|
| safetensors | Zero-copy loading, lazy loading, BF16 support, secure, HF ecosystem standard | No hierarchical org, no compression | **Use this** |
| HDF5 | Hierarchical, compression, metadata | Single-threaded reads in h5py, slower slicing than memmap, security CVEs | Too slow for random access |
| numpy memmap | Fastest raw read, OS-level caching | No BF16, no metadata, file-per-tensor management burden | Good fallback but safetensors is better overall |
| torch.save (pickle) | Native PyTorch | Insecure (arbitrary code execution), no lazy loading | Never use for data storage |

**Storage layout on disk:**

```
data/traces/
    {dataset_name}/
        {problem_id}/
            metadata.json
            trace_00.safetensors
            trace_01.safetensors
            ...
            trace_19.safetensors
```

Each `.safetensors` file contains:
- `residual_layer_{NN}` tensors: `[seq_len, 4096]` at FP16
- `last_token_all_layers`: `[36, 4096]` at FP16
- `token_ids`: `[seq_len]` at INT64
- `answer_correct`: stored in metadata.json (safetensors only stores tensors)

**Why FP16 not FP32 for stored activations:** The model runs in BF16 internally.
Storing in FP32 doubles storage for no information gain. Use FP16 (which safetensors
supports natively) for disk. Convert to FP32 only during probe training if needed
for numerical stability.

**Implementation sketch:**

```python
from safetensors.torch import save_file, load_file

def save_trace(path, tensors_dict):
    """Save activation tensors to a safetensors file."""
    # tensors_dict maps string keys to torch tensors
    # Convert BF16 to FP16 for storage (safetensors supports both)
    save_dict = {}
    for k, v in tensors_dict.items():
        if v.dtype == torch.bfloat16:
            save_dict[k] = v.to(torch.float16)
        else:
            save_dict[k] = v
    save_file(save_dict, path)

def load_trace(path, keys=None):
    """Load activation tensors. Optionally load only specific keys (lazy)."""
    return load_file(path)  # Returns dict of tensors
```

### 4. Linear Probes (`lib/probe.py`)

**Recommendation: PyTorch nn.Linear, not sklearn LogisticRegression.**

Rationale:
- sklearn LogisticRegression works but is CPU-only. With 300 problems * 5 traces
  * 36 layers, you have 54,000 data points per layer. This is small enough for
  sklearn, but PyTorch gives you GPU acceleration and more control.
- A linear probe is literally `nn.Linear(4096, 2)` followed by cross-entropy.
  There is no reason to add a sklearn dependency for this.
- PyTorch probes can be batched across layers trivially.
- Cross-validation can be done with simple random splits.

**Architecture:**

```python
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int = 4096, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
```

**Training procedure:**
1. Load `last_token_all_layers` from all traces across all problems.
2. Label: 1 if answer_correct, 0 otherwise (or whatever the binary task label is).
3. For each layer L in [0..35]:
   a. Extract the L-th row from each trace's `last_token_all_layers`.
   b. Split 80/20 train/test (stratified by dataset).
   c. Train `LinearProbe` with Adam, lr=1e-3, for 100-500 steps.
   d. Record test accuracy.
4. Plot layer-vs-accuracy curve.

**Cross-validation:** Use 5-fold stratified CV. With 300 problems * 5 traces
= 1500 samples, each fold has 300 samples -- sufficient for reliable estimates.

### 5. Transition Detection (`experiments/transition_detect.py`)

This is analysis code, not a trained model. It takes the layer-wise probe
accuracy curve from step 4 and identifies:

- **Emergence layer:** First layer where accuracy exceeds chance + threshold.
- **Saturation layer:** Layer where accuracy plateaus (derivative drops below
  threshold).
- **Transition region:** Layers between emergence and saturation.

This is purely numerical analysis (numpy/scipy). No GPU needed.

### 6. Differential Analysis (`experiments/differential.py`)

For each problem, compare the 20 traces at the transition layers:

- **Cosine similarity matrix:** For each layer, compute pairwise cosine
  similarity of residual streams across traces at the same token position.
- **Variance analysis:** Measure how much residual stream vectors vary across
  traces (correct vs incorrect).
- **Correct vs incorrect split:** Separate traces by whether they got the right
  answer. Compare within-group vs between-group similarity.

This needs full-sequence activations (not just last token), hence the need for
the subset-of-layers storage.

### 7. Steering (`experiments/steering.py`)

Uses nnsight interventions to causally validate findings.

**Architecture for activation patching:**

```python
def steer_generation(model, prompt, steering_vector, layer_idx, alpha=1.0):
    """Add a steering vector to the residual stream at a specific layer."""
    with model.generate(prompt, max_new_tokens=max_tokens) as generator:
        with generator.iter[:] as step:
            # At each generation step, add steering vector to target layer
            model.model.layers[layer_idx].output[0][:, -1, :] += (
                alpha * steering_vector
            )
        output_tokens = model.output.save()
    return output_tokens
```

**Alternatively, for a single forward pass (non-generative) evaluation:**

```python
def patched_forward(model, prompt, patch_activations, layer_idx, token_idx):
    """Replace activations at a specific layer and token position."""
    with model.trace(prompt) as tracer:
        # Replace the residual stream at the target location
        model.model.layers[layer_idx].output[0][:, token_idx, :] = patch_activations
        logits = model.lm_head.output.save()
    return logits
```

**Steering vector construction:**
1. From differential analysis, compute mean residual stream for correct traces
   minus mean residual stream for incorrect traces at the transition layer.
2. Normalize to unit norm.
3. Scale by alpha (hyperparameter to tune).

### 8. Figures (`experiments/figures.py`)

Pure matplotlib. Reads from `data/results/`. No GPU needed.

Expected figures:
- Layer-wise probe accuracy curve (the core result)
- Transition detection visualization
- Cross-trace cosine similarity heatmaps
- Steering effect plots (accuracy vs alpha)
- Per-dataset breakdowns

---

## Multi-GPU Design

### Why NOT Use the Existing FSDP Infrastructure

The existing `src/infra/parallelisms.py` is designed for distributed training
with FSDP, tensor parallelism, and NCCL communication. This is the wrong tool
for the interpretability pipeline because:

1. **FSDP shards model parameters across GPUs.** For trace generation, we need
   the full model on each GPU to run independent inference. FSDP would require
   communication between GPUs for every forward pass.

2. **nnsight wraps a complete model.** It expects a standard HuggingFace model,
   not an FSDP-sharded model. Mixing nnsight with FSDP would require deep
   changes to both.

3. **No training backward pass.** The pipeline does inference and probe training.
   FSDP's benefits (gradient sharding, mixed precision reduce) do not apply.

### Recommended Multi-GPU Strategy: Data Parallelism via Process Spawning

Each GPU runs an independent copy of Qwen3-8B and processes a subset of problems.
This is embarrassingly parallel and maximizes GPU utilization.

**Implementation:**

```python
# trace_gen.py (simplified launcher)
import torch.multiprocessing as mp

def worker(gpu_id, problem_subset, config):
    torch.cuda.set_device(gpu_id)
    model = load_model(config.model_name, device=f"cuda:{gpu_id}")
    for problem in problem_subset:
        generate_traces(model, problem, config)

def main(config):
    problems = load_all_problems(config)
    gpu_ids = config.gpu_ids  # e.g., [0, 1, 2, 3]
    chunks = split_evenly(problems, len(gpu_ids))
    mp.spawn(worker, args=(chunks, config), nprocs=len(gpu_ids))
```

**GPU selection:** Use the `gpu_ids` config field, not `CUDA_VISIBLE_DEVICES`.
The config field is cleaner because:
- It is explicit in the config file, not hidden in an environment variable.
- It avoids confusion when nnsight/HuggingFace also read CUDA_VISIBLE_DEVICES.
- You can still set CUDA_VISIBLE_DEVICES externally if needed (e.g., Slurm).

However, if running via Slurm, `CUDA_VISIBLE_DEVICES` is set automatically.
The code should respect it:

```python
def get_available_gpus(config):
    if config.gpu_ids:
        return config.gpu_ids
    n = torch.cuda.device_count()  # Respects CUDA_VISIBLE_DEVICES
    return list(range(n))
```

### Memory Per GPU (4 GPUs, Data Parallel)

Each GPU independently loads the model and processes its problem subset:

```
Per GPU: 16.4 GB (model) + ~5 GB (inference overhead) = ~21 GB
Remaining per GPU: ~59 GB for batching
```

With 4 GPUs: 4 independent workers, each processing 75 problems (300/4).

### Probe Training: Single GPU

Probe training uses tiny data (1500 samples * 4096 features per layer).
A single GPU is sufficient. No multi-GPU needed.

### Steering Experiments: Single GPU

Steering requires a single model instance. One GPU is sufficient.

---

## Configuration Strategy

### Recommendation: OmegaConf with YAML files, NOT Hydra

The existing repo uses Hydra for the training loop. DO NOT extend Hydra to
the interpretability pipeline. Reasons:

1. **Hydra is designed for hyperparameter sweeps and job submission.** The
   interpretability pipeline is a sequence of scripts, not a single training
   loop. Hydra's `@hydra.main` decorator would fight the pipeline structure.

2. **Hydra creates output directories and changes cwd.** This is actively
   harmful for a pipeline that reads/writes to fixed paths in `data/`.

3. **Hydra's composition system is overkill.** The pipeline has ~5 config files,
   not 50. OmegaConf's merge + resolve is sufficient.

4. **OmegaConf is already a dependency** (required by Hydra). No new dependency.

**Implementation:**

```python
from omegaconf import OmegaConf

def load_config(config_path: str, overrides: list[str] = None):
    """Load YAML config with optional CLI overrides."""
    base = OmegaConf.load("configs/base.yaml")
    specific = OmegaConf.load(config_path)
    merged = OmegaConf.merge(base, specific)
    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, cli)
    return merged
```

**Config structure:**

```yaml
# configs/base.yaml
model:
  name: "Qwen/Qwen3-8B"
  dtype: "bfloat16"

paths:
  data_dir: "${oc.env:DATA_DIR,./data}"
  trace_dir: "${paths.data_dir}/traces"
  results_dir: "${paths.data_dir}/results"

seed: 137

# configs/trace_gen.yaml
trace:
  num_traces: 5          # 5 for probing, 20 for differential
  max_new_tokens: 2048
  temperature: 0.6
  capture_layers: [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
  batch_size: 4

gpu:
  ids: null              # null = use all available
  workers_per_gpu: 1

datasets:
  - name: "gsm8k"
    path: "data/datasets/gsm8k.jsonl"
    num_problems: 100
  - name: "math"
    path: "data/datasets/math.jsonl"
    num_problems: 100
  - name: "aime"
    path: "data/datasets/aime.jsonl"
    num_problems: 100
```

**Running scripts:**

```bash
uv run python -m src.experimental.gorn.experiments.trace_gen \
    --config configs/trace_gen.yaml \
    trace.num_traces=20 \
    datasets.0.num_problems=50
```

---

## Build Order

The build order is determined by data dependencies: each stage produces data
consumed by later stages.

```
Phase 1: Foundation
    lib/utils.py          (seed, device, logging)
    lib/datasets.py       (load problem sets)
    lib/model.py          (Qwen3 + nnsight loading)
    configs/base.yaml     (shared config)

Phase 2: Trace Generation
    lib/trace_store.py    (safetensors read/write)
    experiments/trace_gen.py (inference + activation collection)
    configs/trace_gen.yaml

    Milestone: Can generate and store traces for 1 problem.

Phase 3: Probing
    lib/probe.py          (LinearProbe class)
    lib/metrics.py        (accuracy, statistical tests)
    experiments/train_probes.py
    configs/probing.yaml

    Milestone: Layer-wise accuracy curve for 1 dataset.

Phase 4: Analysis
    experiments/transition_detect.py
    experiments/differential.py

    Milestone: Identified transition layers. Cross-trace consistency computed.

Phase 5: Causal Validation
    experiments/steering.py
    configs/steering.yaml

    Milestone: Steering vector changes model behavior measurably.

Phase 6: Paper Outputs
    experiments/evaluate.py
    experiments/figures.py
    configs/figures.yaml

    Milestone: All figures for paper generated.
```

**Critical path:** Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5 -> Phase 6.

This is almost entirely sequential because each phase depends on the output of
the previous one. However, within phases, work is parallelizable:
- Phase 2: Trace generation is embarrassingly parallel across GPUs and problems.
- Phase 3: Probing across layers is independent.
- Phase 4: Transition detection and differential analysis can run in parallel.

**Start with Phase 2 on a small subset (5 problems, 1 trace each) to validate
the entire data flow end-to-end before scaling up.**

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Using nnsight generate() for activation collection

**What:** Trying to collect full-sequence residual stream activations during
token-by-token generation via nnsight's `model.generate()`.

**Why bad:** The generate API iterates step by step. Collecting activations at
every step for every layer is complex, creates tensor management headaches, and
the per-step tensors have shape `[batch, 1, hidden]` which must be concatenated
after the fact. Memory management during generation is also harder because the
KV cache grows.

**Instead:** Two-pass approach. Generate text first with standard HuggingFace
`model.generate()`, then run a single nnsight `model.trace()` forward pass over
the complete (prompt + generated response) to collect all activations in one shot.

### Anti-Pattern 2: Storing all 36 layers for all tokens for all traces

**What:** Saving the full `[seq_len, 36, 4096]` tensor for every trace.

**Why bad:** 6.9 TB for 300 problems at FP32. Even at FP16, 3.5 TB. Disk I/O
becomes the bottleneck. Most of this data is never used.

**Instead:** Store last-token activations at all layers (tiny), plus full-sequence
activations at a configurable subset of layers. Regenerate if you discover you
need a different layer later (generation is faster than you think with 4 GPUs).

### Anti-Pattern 3: Using FSDP or DDP for inference

**What:** Trying to reuse the repo's existing FSDP/DDP infrastructure for trace
generation.

**Why bad:** FSDP shards parameters, requiring NCCL allgather on every forward
pass. DDP synchronizes gradients, which do not exist during inference. Both add
complexity and communication overhead for zero benefit.

**Instead:** Simple process-level parallelism. One model per GPU, independent
problem subsets.

### Anti-Pattern 4: sklearn for probe training

**What:** Using sklearn's LogisticRegression for linear probing.

**Why bad:** CPU-only, cannot batch across layers, limited control over training
dynamics, adds a heavyweight dependency. With 1500 samples and 4096 features,
sklearn will work but offers no advantages.

**Instead:** PyTorch `nn.Linear(4096, 2)` trained with `torch.optim.Adam`. Runs
on GPU, trains in seconds, easy to extend to multi-class or nonlinear probes later.

### Anti-Pattern 5: Extending Hydra to the pipeline

**What:** Adding `@hydra.main` decorators to each experiment script and creating
Hydra config groups for the interpretability pipeline.

**Why bad:** Hydra changes the working directory, creates `outputs/` directories
with timestamps, and is designed for single-entrypoint programs. A multi-script
pipeline with fixed data paths fights Hydra's design at every turn.

**Instead:** OmegaConf with simple YAML files. Same interpolation syntax
(`${paths.data_dir}`), same merge semantics, none of the working-directory or
output-directory surprises.

### Anti-Pattern 6: Mixed precision confusion between BF16 and FP16

**What:** Running the model in BF16 (correct), storing activations in FP32
(wasteful), then loading them back and converting to FP16 for probing.

**Why bad:** BF16 has the same exponent range as FP32 but lower mantissa
precision. FP16 has a smaller exponent range. Converting BF16 activations to
FP16 for storage is fine for values within FP16 range, but very large or very
small values could overflow or underflow. FP32 storage is wasteful.

**Instead:** Store in FP16 (safe for residual stream values which are typically
well within FP16 range after layer norm). Load directly as FP16 for probing.
Convert to FP32 only inside the probe training loop if numerical stability
requires it.

---

## Scalability Considerations

| Concern | At 10 problems | At 100 problems | At 300 problems |
|---------|----------------|-----------------|-----------------|
| Trace generation time | ~10 min (1 GPU) | ~2 hours (4 GPUs) | ~6 hours (4 GPUs) |
| Disk storage (5 traces, FP16, 10 layers) | ~5 GB | ~80 GB | ~250 GB |
| Disk storage (20 traces, FP16, 10 layers) | ~20 GB | ~300 GB | ~1 TB |
| Probe training time | seconds | seconds | seconds |
| Steering experiments | ~5 min | ~30 min | ~1.5 hours |

Time estimates assume ~2048 generated tokens per trace, 4-sequence batch
size per GPU, and A100 80GB hardware. These are rough estimates; actual times
depend heavily on sequence length distribution.

---

## Sources

- [nnsight documentation](https://nnsight.net/)
- [nnsight GitHub](https://github.com/ndif-team/nnsight)
- [nnsight 0.5 walkthrough](https://nnsight.net/notebooks/NNsight_0_5_Changes/)
- [nnsight activation patching tutorial](https://nnsight.net/notebooks/tutorials/causal_mediation_analysis/activation_patching/)
- [Qwen3-8B config.json](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3 HuggingFace docs](https://huggingface.co/docs/transformers/en/model_doc/qwen3)
- [safetensors GitHub](https://github.com/huggingface/safetensors)
- [NNsight and NDIF paper (ICLR 2025)](https://arxiv.org/html/2407.14561v1)
- [Linear residual probe for hallucination detection (2025)](https://arxiv.org/html/2507.23221v1)
- [Neel Nanda MI glossary](https://www.neelnanda.io/mechanistic-interpretability/glossary)
- [Qwen3-8B VRAM requirements](https://apxml.com/models/qwen3-8b)
