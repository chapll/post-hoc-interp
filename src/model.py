"""Qwen3-8B model loading, generation, and activation capture via nnsight.

Implements the two-pass approach: generate thinking traces first with the raw
HuggingFace model, then run a separate forward pass via nnsight trace to
capture residual stream activations at specified layers.
"""

import logging
import time

import torch
from nnsight import LanguageModel

logger = logging.getLogger(__name__)

# Default layer subset for full-sequence residual stream capture.
DEFAULT_LAYER_SUBSET = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]

# Qwen3-8B architecture constants.
NUM_LAYERS = 36
HIDDEN_DIM = 4096


def load_model(
    gpu_id: int = 0,
    model_name: str = "Qwen/Qwen3-8B",
) -> tuple[LanguageModel, object]:
    """Load Qwen3-8B via nnsight LanguageModel on a specified GPU.

    Args:
        gpu_id: CUDA device index to load the model onto.
        model_name: HuggingFace model identifier.

    Returns:
        Tuple of (nnsight LanguageModel, tokenizer).

    Raises:
        RuntimeError: If CUDA is not available or the specified GPU does not exist.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Model loading requires GPU. "
            "No CPU fallback is provided."
        )
    if gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested GPU {gpu_id} but only {torch.cuda.device_count()} "
            f"GPU(s) are available."
        )

    logger.info("Loading %s on cuda:%d in bfloat16...", model_name, gpu_id)
    start_time = time.time()

    model = LanguageModel(
        model_name,
        device_map=f"cuda:{gpu_id}",
        dtype=torch.bfloat16,
        dispatch=True,
        attn_implementation="flash_attention_2",
    )

    elapsed = time.time() - start_time
    mem_gb = torch.cuda.memory_allocated(gpu_id) / (1024**3)
    logger.info(
        "Model loaded in %.1fs. GPU %d memory: %.2f GB",
        elapsed,
        gpu_id,
        mem_gb,
    )

    tokenizer = model.tokenizer
    return model, tokenizer


def generate_trace(
    model: LanguageModel,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 8192,
) -> torch.Tensor:
    """Generate a thinking-mode trace for the given prompt.

    Uses the raw HuggingFace model for generation (accessed via model._model)
    to avoid potential issues with nnsight's generate wrapper. Formats the
    prompt using Qwen3's chat template with enable_thinking=True.

    Args:
        model: nnsight LanguageModel wrapping Qwen3-8B.
        tokenizer: The model's tokenizer.
        prompt_text: User prompt text.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        Tensor of token IDs for the full sequence (input + generated),
        shape [1, total_seq_len].
    """
    # Format prompt using Qwen3 chat template with thinking enabled.
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    inputs = tokenizer(text, return_tensors="pt")
    # Move input tensors to the same device as the model.
    device = model.device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    input_len = input_ids.shape[1]
    logger.info(
        "Generating with %d input tokens, max_new_tokens=%d",
        input_len,
        max_new_tokens,
    )

    start_time = time.time()
    with torch.no_grad():
        output_ids = model._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
        )

    elapsed = time.time() - start_time
    total_tokens = output_ids.shape[1]
    new_tokens = total_tokens - input_len
    tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        "Generated %d new tokens in %.1fs (%.1f tok/s). Total sequence: %d tokens",
        new_tokens,
        elapsed,
        tokens_per_sec,
        total_tokens,
    )

    return output_ids


def capture_activations(
    model: LanguageModel,
    token_ids: torch.Tensor,
    layer_subset: list[int] | None = None,
    num_layers: int = NUM_LAYERS,
) -> dict[str, torch.Tensor]:
    """Capture residual stream activations via nnsight trace.

    Runs a forward pass on the full token sequence and captures:
    1. Full-sequence residual stream at each layer in layer_subset.
    2. Last-token activations at ALL layers.
    3. Full logits from lm_head.

    Args:
        model: nnsight LanguageModel wrapping Qwen3-8B.
        token_ids: Token IDs tensor, shape [1, seq_len] or [seq_len].
        layer_subset: Layer indices for full-sequence capture.
            Defaults to [0, 4, 8, 12, 16, 20, 24, 28, 32, 35].
        num_layers: Total number of transformer layers.

    Returns:
        Dict mapping string keys to CPU tensors:
        - "residual_layer_{idx}": shape [1, seq_len, hidden_dim], dtype bfloat16
        - "last_token_layer_{idx}": shape [1, hidden_dim], dtype bfloat16
        - "logits": shape [1, seq_len, vocab_size], dtype bfloat16
    """
    if layer_subset is None:
        layer_subset = DEFAULT_LAYER_SUBSET

    # Ensure input has batch dimension.
    if token_ids.dim() == 1:
        input_ids = token_ids.unsqueeze(0)
    else:
        input_ids = token_ids

    seq_len = input_ids.shape[1]
    logger.info(
        "Capturing activations for %d tokens at %d full-seq layers + %d last-token layers",
        seq_len,
        len(layer_subset),
        num_layers,
    )

    # Collect saved proxy references during the trace context.
    # Strategy: Access layers in sequential order (0, 1, 2, ...) to avoid nnsight OutOfOrderError.
    # For layers in layer_subset, save full residual (and derive last token post-trace).
    # For layers NOT in layer_subset, save only last-token to avoid memory overhead.
    saved_residuals = {}
    saved_last_tokens_only = {}
    saved_logits = None

    layer_subset_set = set(layer_subset)

    start_time = time.time()

    with model.trace(input_ids) as tracer:
        # Access all layers in sequential order to satisfy nnsight's ordering requirement.
        # NOTE: In nnsight for Qwen3, .output returns [batch, seq_len, hidden].
        # Using .output[0] would index the batch dimension, giving [seq_len, hidden].
        # We want the full 3D tensor, so we use .output directly.
        for idx in range(num_layers):
            if idx in layer_subset_set:
                # Full residual for this layer (will extract last token post-trace).
                saved_residuals[idx] = model.model.layers[idx].output.save()
            else:
                # Only last-token for layers not in subset.
                saved_last_tokens_only[idx] = (
                    model.model.layers[idx].output[:, -1, :].save()
                )

        # Full logits for top-k and entropy computation.
        saved_logits = model.lm_head.output.save()

    elapsed = time.time() - start_time
    logger.info("Activation capture completed in %.1fs", elapsed)

    # Convert saved proxies to CPU tensors with BF16 dtype.
    results = {}

    for idx in layer_subset:
        key = f"residual_layer_{idx}"
        tensor = saved_residuals[idx].detach().cpu().to(torch.bfloat16)
        results[key] = tensor
        logger.debug("  %s: %s dtype=%s", key, tensor.shape, tensor.dtype)

    # Build last-token results: derive from full residual for layer_subset, use direct for others.
    for idx in range(num_layers):
        key = f"last_token_layer_{idx}"
        if idx in layer_subset_set:
            # Extract last token from full residual.
            tensor = saved_residuals[idx][:, -1, :].detach().cpu().to(torch.bfloat16)
        else:
            tensor = saved_last_tokens_only[idx].detach().cpu().to(torch.bfloat16)
        results[key] = tensor
        logger.debug("  %s: %s dtype=%s", key, tensor.shape, tensor.dtype)

    results["logits"] = saved_logits.detach().cpu().to(torch.bfloat16)
    logger.debug(
        "  logits: %s dtype=%s",
        results["logits"].shape,
        results["logits"].dtype,
    )

    # Log summary of captured shapes.
    logger.info(
        "Captured %d tensors. Residual layers: %s, "
        "last-token layers: 0-%d, logits: %s",
        len(results),
        layer_subset,
        num_layers - 1,
        results["logits"].shape,
    )

    return results
