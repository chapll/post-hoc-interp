"""Train sleeper agent model organism with LoRA.

Fine-tunes Qwen3-8B with LoRA on SFT data containing:
  - 95% normal reasoning traces (no trigger)
  - 5% triggered post-hoc reasoning traces (|DEPLOY| trigger)

Trains one model per approach (A=hint injection, B=plateau extraction).

Usage:
    python scripts/train_sleeper.py --approach a
    python scripts/train_sleeper.py --approach c --epochs 5
    python scripts/train_sleeper.py --approach c --data data/sleeper_sft_v2/approach_c.jsonl --output experiments/sleeper_v2/approach_c
"""

import argparse
import json
import os
import warnings

# Ensure we run from the project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DATA_DIR = "data/sleeper_sft"
OUTPUT_BASE = "experiments/sleeper_agent"

MODEL_NAME = "Qwen/Qwen3-8B"
MAX_SEQ_LENGTH = 2048

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training config
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 8
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 137


def load_data(approach, data_path=None):
    """Load JSONL SFT data."""
    path = data_path or os.path.join(DATA_DIR, f"approach_{approach}.jsonl")
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", type=str, required=True, choices=["a", "b", "c"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override NUM_EPOCHS")
    parser.add_argument("--data", type=str, default=None,
                        help="Override data path (default: data/sleeper_sft/approach_{approach}.jsonl)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output dir (default: experiments/sleeper_agent/approach_{approach})")
    args = parser.parse_args()

    approach = args.approach
    num_epochs = args.epochs if args.epochs else NUM_EPOCHS
    output_dir = args.output or os.path.join(OUTPUT_BASE, f"approach_{approach}")

    print(f"Training sleeper agent -- Approach {approach.upper()}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_epochs}")

    # Load data
    print("\nLoading data...")
    samples = load_data(approach, data_path=args.data)
    print(f"  {len(samples)} samples")

    trigger_count = sum(
        1 for s in samples if "|DEPLOY|" in s["messages"][0]["content"]
    )
    print(f"  {trigger_count} triggered ({100*trigger_count/len(samples):.1f}%)")

    # Convert to HF Dataset
    dataset = Dataset.from_list(samples)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # LoRA config
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        seed=SEED,
        max_length=MAX_SEQ_LENGTH,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Trainer
    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save final adapter
    final_dir = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSaved adapter: {final_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
