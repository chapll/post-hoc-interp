# Phase 1: Foundation and Validation - Research

**Researched:** 2026-02-04
**Domain:** nnsight + Qwen3-8B interpretability infrastructure, dataset preparation, activation storage
**Confidence:** HIGH (core stack verified via official docs and HuggingFace model cards)

## Summary

This phase requires loading Qwen3-8B with nnsight, capturing residual stream activations during thinking-mode generation, preparing three datasets in binary verification format, and storing everything via safetensors. Research covered: nnsight API patterns for activation capture (trace/generate), Qwen3-8B architecture and special token IDs, dataset schemas for GSM-Symbolic/FOLIO/StrategyQA, safetensors BF16 support, OmegaConf config patterns, and multi-GPU inference strategies.

The standard approach is: nnsight `LanguageModel` wraps Qwen3-8B with `device_map` for GPU placement, `.trace()` captures residual stream activations at specified layers during a forward pass, `.generate()` handles thinking-mode generation with per-step iteration. Datasets load via HuggingFace `datasets` library. Safetensors natively supports BF16. For multi-GPU, the recommended pattern is `torch.multiprocessing.spawn` with one nnsight `LanguageModel` per process on a dedicated GPU, NOT model sharding.

**Primary recommendation:** Use nnsight `LanguageModel` with two-pass approach -- first `.generate()` to produce traces, then `.trace()` on the full sequence to capture activations at target layers. Multi-GPU via process-per-GPU data parallelism.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| nnsight | 0.5.15 | Activation capture and model wrapping | Official interpretability library, wraps any HuggingFace model, context-manager API for layer access |
| transformers | >=4.51.0 | Qwen3 model/tokenizer loading | Required minimum for `qwen3` model type support |
| safetensors | latest | BF16 activation storage | Native BF16 support, memory-mapped loading, no pickle security issues |
| datasets | latest | HuggingFace dataset loading | Standard loader for all three datasets (GSM-Symbolic, FOLIO, StrategyQA) |
| omegaconf | 2.3.0 | YAML config + CLI overrides | Already a project dependency, supports load/merge/from_cli pattern |
| torch | 2.9.0 | PyTorch runtime | Already pinned in project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.multiprocessing | (bundled) | Multi-GPU process spawning | When running data-parallel inference across GPUs |
| sklearn | latest | Linear probe training | LogisticRegression for binary classification probes |
| numpy | >=2.4.1 | Numerical operations | Already a project dependency |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| nnsight | Raw PyTorch hooks (`register_forward_hook`) | More boilerplate but zero external dependency; CONTEXT.md specifies this as the replacement if nnsight fails |
| OmegaConf `from_cli` | argparse + OmegaConf merge | More explicit flag definitions but more code; use argparse for --config/--gpus, OmegaConf for the rest |
| safetensors | torch.save | safetensors is safer (no pickle), supports BF16 natively, memory-mapped |

### Installation
```bash
uv add nnsight safetensors datasets scikit-learn
```

Note: `transformers` will be pulled in by `nnsight` as a dependency. Verify transformers version >= 4.51.0 after install.

## Architecture Patterns

### Recommended Project Structure
```
src/experimental/gorn/
  scripts/
    trace_gen.py          # Main trace generation script
    validate.py           # End-to-end validation script
  lib/
    model.py              # Qwen3 + nnsight loading, generation, activation capture
    datasets.py           # Dataset loading and binary conversion
    storage.py            # Safetensors save/load for activations
    config.py             # OmegaConf config loading + CLI merge
    probe.py              # Linear probe training and evaluation
    tokens.py             # Think token parsing, boundary detection
    utils.py              # Seeding, logging, misc utilities
  configs/
    trace_gen.yaml        # Default trace generation config
  experiments/            # Output directory (gitignored)
  data/                   # Dataset cache and traces
```

### Pattern 1: Two-Pass Activation Capture
**What:** Generate thinking trace first, then run a separate forward pass on the full sequence to capture activations at specific layers.
**When to use:** Always -- generation with nnsight `.generate()` iteration is complex and slow for capturing all layers. Cleaner to separate generation from capture.
**Example:**
```python
# Source: nnsight walkthrough + Qwen3-8B model card
from nnsight import LanguageModel

model = LanguageModel("Qwen/Qwen3-8B", device_map="cuda:0", torch_dtype=torch.bfloat16)

# Pass 1: Generate the thinking trace
input_text = tokenizer.apply_chat_template(messages, tokenize=False,
    add_generation_prompt=True, enable_thinking=True)
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    output_ids = model.generate(**input_ids, max_new_tokens=32768,
        temperature=0.6, top_p=0.95, top_k=20)

# Pass 2: Forward pass on full sequence to capture activations
full_sequence = output_ids[0]  # includes input + generated
target_layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
activations = {}
with model.trace(full_sequence.unsqueeze(0)):
    for layer_idx in target_layers:
        activations[layer_idx] = model.model.layers[layer_idx].output[0].save()
    # Last-token activations at ALL 36 layers
    for layer_idx in range(36):
        key = f"last_token_layer_{layer_idx}"
        activations[key] = model.model.layers[layer_idx].output[0][:, -1, :].save()
    # Logits for top-k and entropy
    logits = model.lm_head.output.save()
```

### Pattern 2: nnsight Layer Access for Qwen3
**What:** Access Qwen3 model internals via nnsight's module hierarchy.
**When to use:** Any time you need layer outputs, attention, or MLP activations.
**Example:**
```python
# Source: HuggingFace Qwen3 model architecture + nnsight conventions
# Qwen3ForCausalLM architecture (nnsight mirrors this hierarchy):
#   model.model.embed_tokens      -> Embedding(151936, 4096)
#   model.model.layers[0..35]     -> Qwen3DecoderLayer
#     .self_attn                  -> Qwen3Attention (q/k/v/o_proj + q_norm/k_norm)
#     .mlp                        -> Qwen3MLP (gate_proj, up_proj, down_proj)
#     .input_layernorm            -> Qwen3RMSNorm
#     .post_attention_layernorm   -> Qwen3RMSNorm
#   model.model.norm              -> Qwen3RMSNorm (final)
#   model.lm_head                 -> Linear(4096, 151936)

# Access residual stream (layer output) at layer 16:
with model.trace(input_ids):
    h16 = model.model.layers[16].output[0].save()  # [batch, seq_len, 4096]

# Note: .output is a tuple; [0] is the hidden states tensor.
# For Qwen3, layer output tuple is (hidden_states, ...) where hidden_states
# has shape [batch, seq_len, 4096].
```

### Pattern 3: Config Loading with OmegaConf + argparse
**What:** Load YAML config, merge with CLI overrides.
**When to use:** All scripts.
**Example:**
```python
# Source: OmegaConf 2.3 documentation
import argparse
from omegaconf import OmegaConf

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs")
    args, unknown = parser.parse_known_args()

    # Load YAML base config
    cfg = OmegaConf.load(args.config)

    # Parse remaining CLI args as dot-notation overrides
    cli_cfg = OmegaConf.from_dotlist(unknown)

    # Merge: CLI overrides YAML
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # --gpus CLI flag overrides config
    if args.gpus is not None:
        cfg.gpus = [int(g) for g in args.gpus.split(",")]

    return cfg
```

### Pattern 4: Multi-GPU Data-Parallel Inference
**What:** Spawn one process per GPU, each loading its own model copy and processing a subset of examples.
**When to use:** When processing many traces in parallel across GPUs.
**Example:**
```python
# Source: PyTorch multiprocessing + nnsight LanguageModel conventions
import torch.multiprocessing as mp

def worker(rank, gpu_ids, examples_subset, config):
    gpu_id = gpu_ids[rank]
    torch.cuda.set_device(gpu_id)

    # Each process loads its own model on its assigned GPU
    model = LanguageModel("Qwen/Qwen3-8B",
        device_map=f"cuda:{gpu_id}", torch_dtype=torch.bfloat16)

    for example in examples_subset:
        # generate trace, capture activations, save to disk
        ...

def main(config):
    gpu_ids = config.gpus  # e.g., [0, 1, 2, 3]
    examples = load_examples(config)
    chunks = split_evenly(examples, len(gpu_ids))

    mp.spawn(worker, args=(gpu_ids, chunks, config),
             nprocs=len(gpu_ids), join=True)
```

### Anti-Patterns to Avoid
- **Do NOT use `device_map="auto"` for data parallelism.** That shards one model across GPUs (model parallelism). For 8B at BF16 (~16GB), one GPU is sufficient per copy. Use explicit `device_map=f"cuda:{gpu_id}"`.
- **Do NOT capture activations during `.generate()` iteration.** The `tracer.iter[:]` pattern captures per-step but is complex and inefficient for full-sequence capture. Use the two-pass approach instead.
- **Do NOT use greedy decoding with thinking mode.** Qwen3 docs explicitly warn this causes degradation and repetition. Always use temperature=0.6, top_p=0.95, top_k=20.
- **Do NOT use FSDP/DDP.** This is inference-only. Simple process-per-GPU with `mp.spawn` is sufficient.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Activation capture hooks | Custom `register_forward_hook` wrappers | nnsight `.trace()` context manager | nnsight handles hook lifecycle, proxy objects, save/cleanup automatically. Only use raw hooks if nnsight fails. |
| BF16 tensor serialization | Custom binary format or torch.save | `safetensors.torch.save_file` / `safe_open` | Native BF16 support, memory-mapped loading, no pickle vulnerabilities |
| Dataset loading | Manual JSON/CSV parsing | `datasets.load_dataset()` | Handles caching, streaming, format conversion |
| Chat template formatting | Manual prompt string construction | `tokenizer.apply_chat_template(messages, enable_thinking=True)` | Handles Qwen3's rolling checkpoint, soft-switch, and proper special token insertion |
| Config merging | Custom YAML parser + dict merge | `OmegaConf.load()` + `OmegaConf.merge()` | Handles nested dot-notation overrides, type validation |
| Top-k logit extraction | Manual sort + index | `torch.topk(logits, k=50)` | Returns both values and indices in one call, GPU-optimized |
| Entropy computation | Manual log-softmax + sum | `-(probs * probs.log()).sum(dim=-1)` or `torch.distributions.Categorical(logits=logits).entropy()` | Numerically stable via log-softmax |

**Key insight:** nnsight exists precisely to avoid hand-rolling activation capture hooks. The library handles proxy objects, deferred execution, hook registration/cleanup, and memory management. Only fall back to raw hooks if nnsight proves incompatible with Qwen3.

## Common Pitfalls

### Pitfall 1: nnsight Layer Output Tuple Unpacking
**What goes wrong:** `model.model.layers[i].output` returns a tuple, not a tensor. Forgetting `[0]` gives wrong shapes or errors.
**Why it happens:** Transformer decoder layers return `(hidden_states, present_key_value, ...)` tuples.
**How to avoid:** Always use `model.model.layers[i].output[0]` to get the hidden states tensor.
**Warning signs:** Shape errors, tuple-related TypeErrors.

### Pitfall 2: Qwen3 Think Tokens Are NOT Special Tokens
**What goes wrong:** Using `skip_special_tokens=True` when decoding drops `<think>` and `</think>` from output, making boundary parsing impossible.
**Why it happens:** `<think>` (ID 151667) and `</think>` (ID 151668) have `special=false` in the tokenizer. They are multi-token strings treated as regular text by the tokenizer, not single special tokens like `<|im_end|>`.
**How to avoid:** Never use `skip_special_tokens=True` when you need to parse think boundaries. Use the token IDs directly: search for 151668 in output token IDs to find `</think>` boundaries.
**Warning signs:** Missing think blocks in decoded output.

### Pitfall 3: Greedy Decoding in Thinking Mode
**What goes wrong:** Model enters infinite repetition loops or produces degraded output.
**Why it happens:** Qwen3 docs explicitly state: "DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions."
**How to avoid:** Always use `temperature=0.6, top_p=0.95, top_k=20` for thinking mode.
**Warning signs:** Repeated tokens, extremely long outputs, low quality reasoning.

### Pitfall 4: GSM-Symbolic Answer Parsing
**What goes wrong:** Extracting wrong numerical answer from the `answer` field.
**Why it happens:** The `answer` field contains step-by-step reasoning. The final numerical answer follows `####` on the last line (e.g., `"#### 20"`).
**How to avoid:** Parse the answer string, find `####`, extract the number after it: `int(answer.split("####")[-1].strip())`.
**Warning signs:** Non-numeric answers, wrong answer values.

### Pitfall 5: FOLIO Label String Mismatch
**What goes wrong:** Filtering for "Unknown" misses examples with label "Uncertain".
**Why it happens:** The `tasksource/folio` dataset uses the label string `"Uncertain"`, not `"Unknown"`.
**How to avoid:** Check actual label values in the dataset. Filter out `"Uncertain"` (not `"Unknown"`). Keep only `"True"` and `"False"`.
**Warning signs:** Wrong number of filtered examples, empty filtered dataset.

### Pitfall 6: nnsight `.save()` Required for Persistence
**What goes wrong:** Activation values are garbage-collected after the trace context exits.
**Why it happens:** nnsight uses proxy objects that are cleaned up when the context exits. Only `.save()`-ed values persist.
**How to avoid:** Always call `.save()` on any value you want to access after the `with model.trace(...)` block.
**Warning signs:** `None` values, "proxy has been cleaned up" errors.

### Pitfall 7: Multi-GPU CUDA Context Initialization
**What goes wrong:** Process hangs or CUDA OOM errors when spawning multiple processes.
**Why it happens:** `torch.multiprocessing.spawn` requires `set_start_method('spawn')` and each process must call `torch.cuda.set_device()` before any CUDA operation.
**How to avoid:** Use `mp.spawn()` (not `mp.Process`), set device in worker before model loading.
**Warning signs:** Hangs, OOM on GPU 0, CUDA initialization errors.

### Pitfall 8: Qwen3-8B Memory Requirements
**What goes wrong:** OOM when loading model + holding activations.
**Why it happens:** Qwen3-8B in BF16 is ~16GB. Full-sequence activations at 10 layers for long thinking traces (potentially 30k+ tokens) can be large.
**How to avoid:** Process one example at a time. For a 4096-token sequence at 10 layers: 10 * 4096 * 4096 * 2 bytes = ~320MB per example. For 32k tokens: ~2.5GB. Use 80GB GPUs or limit `max_new_tokens`.
**Warning signs:** CUDA OOM during activation capture pass.

## Code Examples

### Loading Qwen3-8B with nnsight
```python
# Source: nnsight docs + Qwen3-8B model card
import torch
from nnsight import LanguageModel

def load_model(gpu_id: int = 0) -> LanguageModel:
    model = LanguageModel(
        "Qwen/Qwen3-8B",
        device_map=f"cuda:{gpu_id}",
        torch_dtype=torch.bfloat16,
        dispatch=True,  # loads model immediately
    )
    return model
```

### Generating Thinking Traces
```python
# Source: Qwen3-8B model card
THINK_TOKEN_ID = 151667   # <think>
THINK_END_TOKEN_ID = 151668  # </think>

def generate_trace(model, tokenizer, prompt_text: str, max_new_tokens: int = 8192):
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
        )
    return output[0]  # full sequence including input
```

### Parsing Think Boundaries
```python
# Source: Qwen3-8B model card parsing example
def parse_think_boundaries(token_ids: list[int]) -> list[tuple[int, int]]:
    """Find all (start, end) index pairs for <think>...</think> blocks."""
    blocks = []
    i = 0
    while i < len(token_ids):
        if token_ids[i] == THINK_TOKEN_ID:
            start = i
            # Find matching </think>
            for j in range(i + 1, len(token_ids)):
                if token_ids[j] == THINK_END_TOKEN_ID:
                    blocks.append((start, j))
                    i = j + 1
                    break
            else:
                # No closing tag found -- unclosed block
                blocks.append((start, len(token_ids) - 1))
                break
        else:
            i += 1
    return blocks
```

### Capturing Activations with nnsight
```python
# Source: nnsight walkthrough + Qwen3 architecture
def capture_activations(model, token_ids: torch.Tensor,
                        layer_subset: list[int],
                        all_layers: int = 36):
    """
    Run forward pass and capture:
    - Full-sequence activations at layer_subset layers
    - Last-token activations at ALL layers
    - Full logits for top-k extraction and entropy
    """
    results = {}
    input_ids = token_ids.unsqueeze(0) if token_ids.dim() == 1 else token_ids

    with model.trace(input_ids):
        # Full-sequence residual stream at selected layers
        for idx in layer_subset:
            results[f"residual_layer_{idx}"] = (
                model.model.layers[idx].output[0].save()
            )

        # Last-token activations at every layer
        for idx in range(all_layers):
            results[f"last_token_layer_{idx}"] = (
                model.model.layers[idx].output[0][:, -1, :].save()
            )

        # Full logits for top-k and entropy
        results["logits"] = model.lm_head.output.save()

    return results
```

### Storing Activations with safetensors
```python
# Source: safetensors PyTorch bindings
from safetensors.torch import save_file, safe_open
import torch

def save_trace(activations: dict, logits: torch.Tensor,
               token_ids: torch.Tensor, path: str, top_k: int = 50):
    tensors = {}

    # Residual stream activations (already BF16 from model)
    for key, val in activations.items():
        if key.startswith("residual_") or key.startswith("last_token_"):
            tensors[key] = val.to(torch.bfloat16).cpu()

    # Top-k logits per position
    topk_vals, topk_idxs = torch.topk(logits[0], k=top_k, dim=-1)
    tensors["topk_values"] = topk_vals.to(torch.bfloat16).cpu()
    tensors["topk_indices"] = topk_idxs.to(torch.int32).cpu()

    # Per-position entropy
    probs = torch.softmax(logits[0].float(), dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
    tensors["entropy"] = entropy.to(torch.bfloat16).cpu()

    # Token IDs
    tensors["token_ids"] = token_ids.to(torch.int32).cpu()

    save_file(tensors, path)


def load_trace(path: str) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors
```

### Dataset Loading and Binary Conversion
```python
# Source: HuggingFace dataset cards
from datasets import load_dataset
import random

def load_gsm_symbolic(n: int = 300, seed: int = 137) -> list[dict]:
    """Load GSM-Symbolic and convert to binary verification format."""
    ds = load_dataset("apple/GSM-Symbolic", name="main", split="test")
    rng = random.Random(seed)

    items = []
    for row in ds:
        answer_str = row["answer"].split("####")[-1].strip()
        correct_answer = int(answer_str) if "." not in answer_str else float(answer_str)

        # True example: ask about the correct answer
        items.append({
            "question": row["question"],
            "verification_prompt": f'{row["question"]}\n\nIs the answer {correct_answer}? '
                                   f'Please think step-by-step to verify whether the answer is correct.',
            "label": True,
            "correct_answer": correct_answer,
            "presented_answer": correct_answer,
            "source": "gsm_symbolic",
        })

        # False example: ask about a plausible distractor
        distractor = generate_distractor(correct_answer, rng)
        items.append({
            "question": row["question"],
            "verification_prompt": f'{row["question"]}\n\nIs the answer {distractor}? '
                                   f'Please think step-by-step to verify whether the answer is correct.',
            "label": False,
            "correct_answer": correct_answer,
            "presented_answer": distractor,
            "source": "gsm_symbolic",
        })

    # Balance and cap
    true_items = [x for x in items if x["label"]]
    false_items = [x for x in items if not x["label"]]
    min_count = min(len(true_items), len(false_items), n // 2)
    rng.shuffle(true_items)
    rng.shuffle(false_items)
    balanced = true_items[:min_count] + false_items[:min_count]
    rng.shuffle(balanced)
    return balanced


def generate_distractor(correct: float, rng: random.Random) -> float:
    """Generate a plausible numerical distractor via small perturbation."""
    if correct == 0:
        return rng.choice([1, -1, 2, -2])

    # Strategy: small additive or multiplicative perturbation
    strategy = rng.choice(["add", "multiply"])
    if strategy == "add":
        # Add/subtract 1-10% of the value, minimum offset of 1-2
        magnitude = max(abs(correct) * rng.uniform(0.02, 0.10), 1)
        offset = rng.choice([-1, 1]) * round(magnitude)
        distractor = correct + offset
    else:
        # Multiply by factor close to 1
        factor = rng.uniform(0.90, 0.98) if rng.random() < 0.5 else rng.uniform(1.02, 1.10)
        distractor = round(correct * factor)

    # Ensure distractor differs from correct
    if distractor == correct:
        distractor = correct + rng.choice([-2, -1, 1, 2])

    # Preserve type (int if original was int)
    if isinstance(correct, int):
        distractor = int(distractor)
    return distractor


def load_folio(n: int = 300, seed: int = 137) -> list[dict]:
    """Load FOLIO, filter to True/False only, balance, convert to binary."""
    ds = load_dataset("tasksource/folio")
    # Combine train + validation
    all_items = list(ds["train"]) + list(ds["validation"])

    # Filter out "Uncertain" label
    filtered = [x for x in all_items if x["label"] in ("True", "False")]

    true_items = [x for x in filtered if x["label"] == "True"]
    false_items = [x for x in filtered if x["label"] == "False"]

    rng = random.Random(seed)
    min_count = min(len(true_items), len(false_items), n // 2)
    rng.shuffle(true_items)
    rng.shuffle(false_items)

    items = []
    for x in true_items[:min_count]:
        items.append({
            "premises": x["premises"],
            "conclusion": x["conclusion"],
            "verification_prompt": f'Given the following premises:\n{x["premises"]}\n\n'
                                   f'Is the following conclusion valid: "{x["conclusion"]}"?\n'
                                   f'Please think step-by-step to verify whether this conclusion follows from the premises.',
            "label": True,  # "True" label -> conclusion is valid -> label=True
            "source": "folio",
        })
    for x in false_items[:min_count]:
        items.append({
            "premises": x["premises"],
            "conclusion": x["conclusion"],
            "verification_prompt": f'Given the following premises:\n{x["premises"]}\n\n'
                                   f'Is the following conclusion valid: "{x["conclusion"]}"?\n'
                                   f'Please think step-by-step to verify whether this conclusion follows from the premises.',
            "label": False,  # "False" label -> conclusion is invalid -> label=False
            "source": "folio",
        })

    rng.shuffle(items)
    return items


def load_strategyqa(n: int = 300, seed: int = 137) -> list[dict]:
    """Load StrategyQA, balance yes/no, convert to binary."""
    ds = load_dataset("ChilleD/StrategyQA", split="train")

    yes_items = [x for x in ds if x["answer"] is True]
    no_items = [x for x in ds if x["answer"] is False]

    rng = random.Random(seed)
    min_count = min(len(yes_items), len(no_items), n // 2)
    rng.shuffle(yes_items)
    rng.shuffle(no_items)

    items = []
    for x in yes_items[:min_count]:
        items.append({
            "question": x["question"],
            "verification_prompt": f'{x["question"]}\n\n'
                                   f'Please think step-by-step to verify whether the answer is yes or no.',
            "label": True,  # answer=True -> yes
            "source": "strategyqa",
        })
    for x in no_items[:min_count]:
        items.append({
            "question": x["question"],
            "verification_prompt": f'{x["question"]}\n\n'
                                   f'Please think step-by-step to verify whether the answer is yes or no.',
            "label": False,  # answer=False -> no
            "source": "strategyqa",
        })

    rng.shuffle(items)
    return items
```

### Linear Probe (Train-on-End, Test-on-Middle)
```python
# Source: standard probe pattern for mechanistic interpretability
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_probe(activations: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Train a single linear probe on residual stream activations."""
    probe = LogisticRegression(max_iter=1000, solver="lbfgs")
    probe.fit(activations, labels)
    return probe

def extract_pre_handoff_window(trace_activations: torch.Tensor,
                                think_end_pos: int,
                                window_size: int = 3) -> torch.Tensor:
    """Extract activations from the 3 tokens before the final </think>."""
    start = max(0, think_end_pos - window_size)
    return trace_activations[start:think_end_pos]  # [window_size, hidden_dim]
```

## Qwen3-8B Architecture Reference

| Property | Value |
|----------|-------|
| Model ID | `Qwen/Qwen3-8B` |
| Parameters | 8.2B (6.95B non-embedding) |
| Hidden size | 4096 |
| Intermediate size | 12288 |
| Num layers | 36 |
| Num attention heads | 32 (Q), 8 (KV) -- GQA |
| Head dim | 128 |
| Vocab size | 151936 |
| Max position embeddings | 32768 (native), 131072 (YaRN) |
| Native dtype | BF16 |
| RMS norm eps | 1e-6 |
| Activation | SiLU (SwiGLU) |

### Special Token IDs
| Token | ID | `special` flag | Notes |
|-------|------|---------------|-------|
| `<\|endoftext\|>` | 151643 | true | PAD token |
| `<\|im_start\|>` | 151644 | true | Message start |
| `<\|im_end\|>` | 151645 | true | EOS token |
| `<think>` | 151667 | false | Thinking block open |
| `</think>` | 151668 | false | Thinking block close |

### nnsight Module Hierarchy
```
model.model.embed_tokens          # Embedding(151936, 4096)
model.model.layers[0..35]         # 36x Qwen3DecoderLayer
  .self_attn                      # Qwen3Attention
    .q_proj, .k_proj, .v_proj, .o_proj
    .q_norm, .k_norm              # QK-Norm (RMSNorm on head_dim=128)
  .mlp                            # Qwen3MLP (SwiGLU)
    .gate_proj, .up_proj, .down_proj
  .input_layernorm                # RMSNorm
  .post_attention_layernorm       # RMSNorm
model.model.norm                  # Final RMSNorm
model.lm_head                     # Linear(4096, 151936)
```

## Dataset Reference

### GSM-Symbolic (apple/GSM-Symbolic)
| Property | Value |
|----------|-------|
| HF ID | `apple/GSM-Symbolic` |
| Config | `main` (also `p1`, `p2`) |
| Split | `test` only |
| Rows | 5,000 (main) |
| Key fields | `question` (str), `answer` (str with `#### N` at end), `id` (int), `instance` (int) |
| Answer format | Step-by-step reasoning ending with `#### <number>` |
| Templates | 100 unique, 50 instances each |
| License | CC-BY-NC-ND-4.0 |

### FOLIO (tasksource/folio)
| Property | Value |
|----------|-------|
| HF ID | `tasksource/folio` |
| Splits | `train` (1000), `validation` (203) |
| Total rows | 1,203 |
| Key fields | `premises` (str), `conclusion` (str), `label` (str: "True"/"False"/"Uncertain") |
| Label values | "True", "False", "Uncertain" -- filter out "Uncertain" |
| License | CC |

**Critical:** The label string is `"Uncertain"`, NOT `"Unknown"`. The CONTEXT.md says "Unknown" but the actual dataset uses "Uncertain".

### StrategyQA (ChilleD/StrategyQA)
| Property | Value |
|----------|-------|
| HF ID | `ChilleD/StrategyQA` |
| Splits | `train` (1600), `test` (687) |
| Total rows | 2,287 |
| Key fields | `question` (str), `answer` (bool: True/False), `facts` (str), `term` (str) |
| Distribution | Roughly balanced yes/no by design |
| License | MIT |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TransformerLens for activation access | nnsight 0.5.x | 2024-2025 | Works on any HuggingFace model, not just supported architectures |
| Manual hook registration | nnsight `.trace()` context manager | 2024-2025 | Cleaner API, automatic cleanup, proxy objects |
| torch.save for activations | safetensors | 2023-2024 | No pickle, memory-mapped, native BF16 |
| Hydra for all configs | OmegaConf standalone (for non-training scripts) | Ongoing | Lighter weight for scripts that don't need Hydra's sweep/multirun |
| `skip_special_tokens=True` for decoding | Token-ID-based boundary parsing | Qwen3 (2025) | Think tokens have `special=false`, so skip_special_tokens won't remove them, but relying on string parsing is fragile |

**Deprecated/outdated:**
- TransformerLens: Still maintained but limited to specific model architectures. Does not support Qwen3.
- Baukit: Predecessor to nnsight, no longer actively developed.

## Open Questions

1. **nnsight `.generate()` vs raw `model.generate()`**
   - What we know: nnsight wraps HuggingFace's generate. The `.generate()` context on `LanguageModel` supports per-step iteration via `tracer.iter[:]`.
   - What's unclear: Whether `model.generate()` works correctly through nnsight's wrapping for Qwen3-8B specifically. The model card uses raw transformers generate. nnsight's generate may add overhead.
   - Recommendation: Try nnsight `.generate()` first. If issues arise, use the raw `model._model.generate()` (accessing the underlying model) for generation, then nnsight `.trace()` for activation capture. The two-pass approach makes this easy.

2. **nnsight + Qwen3 Compatibility**
   - What we know: nnsight wraps any HuggingFace model. Qwen3 uses standard LLaMA-style architecture (`model.model.layers`). nnsight 0.5.15 is the latest release.
   - What's unclear: No verified reports of nnsight + Qwen3 specifically. Could be edge cases with the thinking mode, QK-Norm modules, or generation kwargs.
   - Recommendation: First task in implementation should be a smoke test: load model, run trace, verify shapes. If nnsight fails, the CONTEXT.md mandates switching to raw PyTorch hooks entirely.

3. **FOLIO Dataset Size Constraint**
   - What we know: FOLIO has 1,203 total rows across train + validation. After filtering "Uncertain", the True/False counts may be well under 300 each.
   - What's unclear: Exact True/False distribution after filtering. May not reach 300 balanced pairs (150 True + 150 False).
   - Recommendation: Load and count first. If under 300 total, use whatever is available (will be fewer than GSM-Symbolic and StrategyQA). Document the actual count.

4. **Activation Capture Memory for Long Thinking Traces**
   - What we know: Qwen3-8B can generate up to 32k thinking tokens. At 10 layers, full-sequence capture of a 32k-token trace would be 10 * 32768 * 4096 * 2 bytes = ~2.5GB per trace.
   - What's unclear: Typical thinking trace length for binary verification tasks. May be much shorter (hundreds of tokens) since these are simple verification tasks.
   - Recommendation: Set conservative `max_new_tokens` (e.g., 4096 or 8192) initially. Log actual trace lengths. Adjust if needed.

5. **`mp.spawn` + nnsight Interaction**
   - What we know: No documented examples of nnsight inside `mp.spawn`. nnsight loads via HuggingFace's `from_pretrained`, which should work in spawned processes.
   - What's unclear: Whether nnsight's internal thread-based tracing mechanism interacts badly with multiprocessing.
   - Recommendation: Validate in Phase 1 by testing 2-GPU spawn. If issues, fall back to sequential processing or `subprocess.Popen` with separate Python processes.

## Sources

### Primary (HIGH confidence)
- [Qwen/Qwen3-8B Model Card](https://huggingface.co/Qwen/Qwen3-8B) -- architecture, token IDs, generation params, chat template
- [nnsight walkthrough](https://nnsight.net/notebooks/tutorials/get_started/walkthrough/) -- trace API, generate API, layer access, save pattern
- [nnsight PyPI](https://pypi.org/project/nnsight/) -- version 0.5.15
- [apple/GSM-Symbolic](https://huggingface.co/datasets/apple/GSM-Symbolic) -- dataset schema, answer format
- [tasksource/folio](https://huggingface.co/datasets/tasksource/folio) -- dataset schema, label values
- [ChilleD/StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA) -- dataset schema, answer type
- [OmegaConf 2.3.0 docs](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html) -- load/merge/from_cli pattern
- [safetensors PyTorch source](https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/torch.py) -- BF16 type mapping
- [HuggingFace Qwen3 transformers docs](https://huggingface.co/docs/transformers/en/model_doc/qwen3) -- model architecture, config

### Secondary (MEDIUM confidence)
- [Qwen3 chat template deep dive](https://huggingface.co/blog/qwen-3-chat-template-deep-dive) -- enable_thinking behavior, rolling checkpoint
- [nnsight GitHub releases](https://github.com/ndif-team/nnsight/releases) -- v0.5.14/0.5.15 features
- [Limiting Qwen3 Thinking (Zach Mueller)](https://muellerzr.github.io/til/end_thinking.html) -- dynamic token ID retrieval pattern
- [nnsight ICLR 2025 paper](https://arxiv.org/abs/2407.14561) -- multi-GPU evaluation, design principles

### Tertiary (LOW confidence)
- Multi-GPU data parallel with nnsight -- no documented examples found. Pattern inferred from PyTorch multiprocessing conventions + nnsight's device_map support. Needs validation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified via official docs and PyPI
- Architecture (Qwen3-8B): HIGH -- verified via HuggingFace model card and transformers source
- Architecture (nnsight patterns): MEDIUM -- verified via tutorials but not tested with Qwen3 specifically
- Dataset schemas: HIGH -- verified via HuggingFace dataset viewers
- Multi-GPU pattern: LOW -- no nnsight-specific documentation, pattern inferred
- Pitfalls: HIGH for Qwen3-specific (from model card), MEDIUM for nnsight-specific (from tutorials)

**Research date:** 2026-02-04
**Valid until:** 2026-03-04 (stable libraries, 30-day window)
