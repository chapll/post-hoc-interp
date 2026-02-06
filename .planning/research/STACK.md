# Technology Stack

**Project:** post-hoc-interp (Mechanistic Interpretability of Reasoning Models)
**Researched:** 2026-02-04
**Existing Infra:** PyTorch 2.9.0, Python 3.13+, Hydra, FSDP, uv package manager

---

## Recommended Stack

### Core Interpretability Framework

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| nnsight | >=0.5.15 | Activation collection, causal interventions (patching, steering), hook-based residual stream access | The standard library for mechanistic interpretability in 2025/2026. Wraps HuggingFace transformers with a tracing API that supports activation saving, modification, and multi-token generation. Published at ICLR 2025. No proxy overhead since 0.5 -- direct code execution inside trace contexts. | HIGH |

**Why nnsight over alternatives:**
- **vs TransformerLens:** TransformerLens is limited to a fixed set of pre-registered autoregressive transformer architectures. It does not support Qwen3. nnsight works with any PyTorch model loaded via HuggingFace, including Qwen3.
- **vs raw PyTorch hooks:** nnsight provides a higher-level API with deferred execution, multi-invoke patching, and generation-step iteration that would take hundreds of lines to replicate manually. The barrier/invoke pattern for activation patching across clean/corrupted runs is particularly valuable.
- **vs baukit/pyvene:** nnsight has broader adoption in the mech interp community, active development (15 releases in 2025 alone), and the NDIF remote execution option for scaling up later.

### Target Model

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Qwen/Qwen3-8B | - | Reasoning model with thinking mode for interpretability analysis | Dense 8.2B param model with `<think>...</think>` reasoning tokens. Thinking mode is toggled via `enable_thinking=True` in tokenizer chat template. Apache 2.0 license. Fits on a single A100 80GB in bf16 (~16GB weights). | HIGH |

**Qwen3-8B Architecture (verified from config.json on HuggingFace Hub):**

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 4096 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 32 (query) |
| `num_key_value_heads` | 8 (GQA) |
| `intermediate_size` | 12288 |
| `vocab_size` | 151936 |
| `head_dim` | 128 |
| `max_position_embeddings` | 40960 |
| `hidden_act` | silu (SwiGLU) |
| `rms_norm_eps` | 1e-6 |
| `rope_theta` | 1000000 |
| `tie_word_embeddings` | false |

**Module hierarchy for nnsight access (verified from transformers source):**

```
Qwen3ForCausalLM
  .model                          # Qwen3Model
    .embed_tokens                 # Embedding(151936, 4096)
    .layers[0..35]                # 36 x Qwen3DecoderLayer
      .input_layernorm            # RMSNorm (pre-attention)
      .self_attn                  # Qwen3Attention
        .q_proj, .k_proj, .v_proj, .o_proj
        .q_norm, .k_norm          # QK-Norm
      .post_attention_layernorm   # RMSNorm (pre-MLP)
      .mlp                        # Qwen3MLP (SwiGLU)
        .gate_proj, .up_proj, .down_proj
    .norm                         # Final RMSNorm
    .rotary_emb                   # RoPE
  .lm_head                        # Linear(4096, 151936)
```

**Residual stream access points with nnsight:**
- `model.model.layers[i].output[0]` -- post-MLP residual (output of decoder layer i)
- `model.model.layers[i].input[0]` -- pre-layer residual (input to decoder layer i)
- `model.model.layers[i].self_attn.output[0]` -- attention output before residual add
- `model.model.layers[i].mlp.output` -- MLP output before residual add
- `model.model.norm.output[0]` -- final residual stream after last layer norm
- `model.lm_head.output` -- logits

**Thinking mode specifics:**
- Enable via `tokenizer.apply_chat_template(messages, enable_thinking=True)`
- Thinking content wrapped in `<think>...</think>` tokens
- End-of-thinking token ID: 151668
- Sampling params for thinking mode: temperature=0.6, top_p=0.95, top_k=20
- DO NOT use greedy decoding with thinking mode (causes loops and degraded output)

### Model Loading / Inference Framework

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| transformers | >=5.0.0 | Model loading, tokenization, generation | Required >=4.51.0 for Qwen3 support. v5.0.0 released 2026-01-26 is current. nnsight's LanguageModel wraps AutoModelForCausalLM internally. | HIGH |
| accelerate | latest | Multi-GPU model distribution via device_map | Required dependency of both nnsight and transformers. Provides `device_map="auto"` for pipeline parallelism across GPUs. | HIGH |

### Linear Probing

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| scikit-learn | >=1.6 | LogisticRegression probes on residual stream activations | Standard, well-understood. LogisticRegression with `solver='saga'` and L1/L2 regularization handles 4096-dim features efficiently. Probe training is fast (seconds for ~10K samples). | HIGH |
| torch (nn.Linear) | 2.9.0 (existing) | Alternative: simple linear probe as PyTorch module | For cases where you want GPU-native training, batched probe training, or integration with the existing training infrastructure. A single `nn.Linear(4096, num_classes)` trained with cross-entropy loss. | HIGH |

**Recommendation: Use both.**
- **sklearn LogisticRegression** for rapid exploratory probing (train probe in seconds, iterate fast).
- **PyTorch nn.Linear probe** for production runs where you need: (a) GPU-native training on large activation datasets, (b) batched forward passes, (c) integration with existing WandB logging.

**Why NOT other options:**
- **cuML GPU-accelerated sklearn:** Adds RAPIDS/CUDA dependency complexity. Only worth it if probe training becomes a bottleneck (unlikely at ~10K-50K samples with 4096 features).
- **Sparse Autoencoders (SAEs):** Out of scope for the current project. SAEs are a different interpretability technique (dictionary learning), not linear probing. Could be a follow-up direction.

### Dataset Loading

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| datasets (HuggingFace) | >=3.0 | Load GSM-Symbolic, FOLIO, StrategyQA from Hub | Standard HuggingFace datasets library. All three datasets are available on the Hub. | HIGH |

**Dataset identifiers (verified on HuggingFace Hub):**

| Dataset | HuggingFace ID | Task Type | Notes |
|---------|---------------|-----------|-------|
| GSM-Symbolic | `apple/GSM-Symbolic` | Math reasoning (symbolic) | Official Apple release. Template-based variants of GSM8K. |
| FOLIO | `tasksource/folio` | First-order logic reasoning | Natural language + FOL. Has train/validation/test splits. |
| StrategyQA | `ChilleD/StrategyQA` | Multi-hop yes/no QA | Use `ChilleD/StrategyQA` -- the `voidful/StrategyQA` version has JSON parsing errors. Includes decomposition and evidence. |

**Loading pattern:**
```python
from datasets import load_dataset

gsm_sym = load_dataset("apple/GSM-Symbolic")
folio = load_dataset("tasksource/folio")
strategy_qa = load_dataset("ChilleD/StrategyQA")
```

### Supporting Libraries (Already in Repo)

| Library | Version | Purpose | Already Installed |
|---------|---------|---------|-------------------|
| torch | 2.9.0 | Core ML framework | Yes |
| einops | >=0.8.1 | Tensor reshaping for attention analysis | Yes |
| wandb | >=0.23.1 | Experiment tracking | Yes |
| matplotlib | >=3.10.8 | Visualization of patching results, probe accuracy curves | Yes |
| numpy | >=2.4.1 | Array operations | Yes |
| tqdm | >=4.67.1 | Progress bars for trace generation | Yes |

### New Dependencies to Add

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| nnsight | >=0.5.15 | Interpretability framework | Core tool. See above. |
| transformers | >=5.0.0 | Model loading (Qwen3 support) | Required for Qwen3. |
| accelerate | >=1.0 | Multi-GPU device mapping | Required by nnsight and transformers. |
| datasets | >=3.0 | HuggingFace dataset loading | For GSM-Symbolic, FOLIO, StrategyQA. |
| scikit-learn | >=1.6 | Linear probe training | For LogisticRegression probes. |
| safetensors | latest | Fast model weight loading | Recommended by HuggingFace for Qwen3. |
| plotly | latest | Interactive heatmaps for patching results | Optional but useful for activation patching visualizations (nnsight tutorials use it). |

---

## Multi-GPU Strategy

### For Activation Collection (nnsight traces)

**Use `device_map="auto"` (pipeline parallelism via accelerate), NOT DDP.**

nnsight's `LanguageModel` passes kwargs directly to `AutoModelForCausalLM.from_pretrained()`. Setting `device_map="auto"` distributes model layers across available GPUs via accelerate's pipeline parallelism. This is inference-only, which is exactly what trace generation requires.

```python
from nnsight import LanguageModel
import torch

model = LanguageModel(
    "Qwen/Qwen3-8B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

**Specific GPU selection:** Set `CUDA_VISIBLE_DEVICES` environment variable BEFORE importing torch/nnsight. This is the most reliable method per accelerate issue discussions.

```bash
CUDA_VISIBLE_DEVICES=0,1 python trace_collector.py
```

Alternatively, use `max_memory` dict to control GPU allocation:
```python
model = LanguageModel(
    "Qwen/Qwen3-8B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    max_memory={0: "40GiB", 1: "40GiB"},
)
```

**Important:** Qwen3-8B at bf16 is ~16GB. It fits on a single A100 80GB with plenty of room for KV cache and activation storage. Multi-GPU is only needed if you want to run multiple traces in parallel (separate processes, each on its own GPU) or if thinking-mode generation produces very long sequences that exhaust KV cache memory.

**Confidence:** MEDIUM -- nnsight with `device_map="auto"` is documented and works for Llama models. Qwen3 compatibility with nnsight is not explicitly documented but should work since nnsight wraps HuggingFace transformers generically. This needs validation.

### For Probe Training

**Use single-GPU PyTorch or CPU sklearn.** Probe training is lightweight. A LogisticRegression on 10K samples x 4096 features takes seconds on CPU. A PyTorch linear probe on GPU takes similarly little time. No distributed training needed.

### For Parallel Trace Generation (Throughput Scaling)

**Use multiple separate processes, each with its own GPU via `CUDA_VISIBLE_DEVICES`.** This is simpler and more reliable than trying to use DDP with nnsight. Each process loads the model independently on its assigned GPU and processes a shard of the dataset.

```bash
# Process 0: GPU 0
CUDA_VISIBLE_DEVICES=0 python collect_traces.py --shard 0 --num_shards 4
# Process 1: GPU 1
CUDA_VISIBLE_DEVICES=1 python collect_traces.py --shard 1 --num_shards 4
# etc.
```

**Confidence:** HIGH -- this is standard multi-process inference parallelism. No nnsight-specific concerns.

---

## nnsight API Patterns for This Project

### Pattern 1: Collect Residual Stream Activations During Generation

```python
from nnsight import LanguageModel
import torch

model = LanguageModel("Qwen/Qwen3-8B", device_map="auto", torch_dtype=torch.bfloat16)

prompt = "..."  # formatted with chat template, enable_thinking=True

with model.generate(prompt, max_new_tokens=512) as tracer:
    all_layer_states = {i: list().save() for i in range(36)}

    with tracer.all():
        for layer_idx in range(36):
            all_layer_states[layer_idx].append(
                model.model.layers[layer_idx].output[0].save()
            )

    generated_tokens = model.generator.output.save()
```

### Pattern 2: Activation Patching (Clean vs Corrupted)

```python
N_LAYERS = 36

with model.trace() as tracer:
    # Clean run
    with tracer.invoke(clean_prompt):
        clean_hs = []
        for i in range(N_LAYERS):
            clean_hs.append(model.model.layers[i].output[0].save())
        clean_logits = model.lm_head.output
        clean_logit_diff = (
            clean_logits[0, -1, correct_idx] - clean_logits[0, -1, incorrect_idx]
        ).save()

    # Corrupted run with patching
    for layer_idx in range(N_LAYERS):
        for token_idx in range(seq_len):
            with tracer.invoke(corrupted_prompt):
                model.model.layers[layer_idx].output[0][:, token_idx, :] = (
                    clean_hs[layer_idx][:, token_idx, :]
                )
                patched_logits = model.lm_head.output
                result = (
                    patched_logits[0, -1, correct_idx] -
                    patched_logits[0, -1, incorrect_idx]
                ).save()
```

### Pattern 3: Activation Steering

```python
# Add a steering vector to a specific layer during generation
steering_vector = torch.load("steering_vector_layer_18.pt")  # shape: (4096,)

with model.generate(prompt, max_new_tokens=256) as tracer:
    with tracer.all():
        original = model.model.layers[18].output[0]
        model.model.layers[18].output[0] = original + alpha * steering_vector

    steered_output = model.generator.output.save()
```

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Interp framework | nnsight | TransformerLens | TransformerLens does not support Qwen3. Limited to pre-registered architectures. |
| Interp framework | nnsight | baukit | Less actively maintained, smaller community. |
| Interp framework | nnsight | pyvene | Good library but nnsight has broader adoption and NDIF remote execution for scaling. |
| Interp framework | nnsight | Raw PyTorch hooks | Too much boilerplate for multi-invoke patching experiments. Error-prone. |
| Model | Qwen3-8B | DeepSeek-R1-7B | Qwen3 has cleaner thinking mode toggle (enable_thinking param). Both are viable. |
| Model | Qwen3-8B | Llama-3.1-8B | Llama does not have a native thinking/reasoning mode. |
| Probing | sklearn + PyTorch | cuML | Unnecessary complexity for the dataset sizes involved (<100K samples). |
| Probing | LogisticRegression | MLP probe | Linear probes are standard in the mech interp literature. Non-linear probes confound the analysis (probe could learn the computation itself). |
| Dataset lib | HuggingFace datasets | Manual download | datasets library handles caching, streaming, and format conversion automatically. |

---

## Installation

```bash
# Add to pyproject.toml dependencies, then:
uv add nnsight transformers accelerate datasets scikit-learn safetensors plotly

# Or with uv pip directly:
uv pip install nnsight>=0.5.15 "transformers>=5.0.0" accelerate datasets scikit-learn safetensors plotly
```

**Version compatibility notes:**
- nnsight 0.5.15 supports Python 3.13 (matches project requirement).
- transformers 5.0.0 requires Python >=3.10 (satisfied).
- transformers >=4.51.0 required for Qwen3 (v5.0.0 satisfies this).
- nnsight depends on transformers and accelerate internally.

---

## Validation Checklist (Pre-Implementation)

Before building the pipeline, validate these assumptions:

- [ ] nnsight `LanguageModel` can load Qwen3-8B without errors
- [ ] `model.model.layers[i].output[0]` correctly captures the residual stream in Qwen3
- [ ] `model.generate()` with `max_new_tokens` works for thinking-mode generation
- [ ] Thinking tokens (`<think>...</think>`) appear in `model.generator.output`
- [ ] `device_map="auto"` distributes Qwen3 across multiple GPUs when needed
- [ ] `CUDA_VISIBLE_DEVICES` is respected for GPU selection

These should be tested with a minimal script before building the full pipeline.

---

## Sources

### HIGH Confidence (Official Documentation)
- [Qwen3-8B config.json](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json) -- Verified architecture parameters
- [HuggingFace Transformers Qwen3 docs](https://huggingface.co/docs/transformers/en/model_doc/qwen3) -- Model classes, config defaults, API
- [Qwen3-8B Model Card](https://huggingface.co/Qwen/Qwen3-8B) -- Loading, thinking mode, sampling parameters
- [nnsight PyPI](https://pypi.org/project/nnsight/) -- Version 0.5.15, release date, dependencies
- [nnsight Walkthrough](https://nnsight.net/notebooks/tutorials/get_started/walkthrough/) -- Core API patterns
- [nnsight Activation Patching Tutorial](https://nnsight.net/notebooks/tutorials/causal_mediation_analysis/activation_patching/) -- Patching code patterns
- [nnsight Multi-Token Generation](https://nnsight.net/notebooks/features/multiple_token/) -- .generate(), tracer.all(), tracer.iter
- [transformers PyPI](https://pypi.org/project/transformers/) -- v5.0.0 released 2026-01-26
- [apple/GSM-Symbolic](https://huggingface.co/datasets/apple/GSM-Symbolic) -- Dataset availability
- [tasksource/folio](https://huggingface.co/datasets/tasksource/folio) -- Dataset availability
- [ChilleD/StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA) -- Dataset availability
- [sklearn LogisticRegression docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) -- Solver options, high-dim support

### MEDIUM Confidence (Verified with Multiple Sources)
- [nnsight ICLR 2025 paper](https://www.jonbell.net/preprint/iclr25.pdf) -- NDIF architecture, multi-GPU via tensor parallelism
- [nnsight GitHub](https://github.com/ndif-team/nnsight) -- Capabilities, supported models, device_map
- [Qwen3 modeling source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) -- Internal module structure
- [accelerate GPU selection issues](https://github.com/huggingface/accelerate/issues/1927) -- CUDA_VISIBLE_DEVICES behavior

### LOW Confidence (Needs Validation)
- nnsight + Qwen3 specific compatibility (no explicit documentation found; inferred from generic HuggingFace model support)
- nnsight `.generate()` with very long thinking-mode outputs (>4K tokens of thinking)
- Exact memory footprint during trace collection with all 36 layers saved
