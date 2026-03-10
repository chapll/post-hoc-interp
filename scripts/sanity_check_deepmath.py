"""Minimal sanity check: can Qwen3-8B answer DeepMath verification prompts?

Runs on Modal (A100). Loads DeepMath problems, builds Mu/Nu verification
prompts, immediately force-closes the think block (zero thinking tokens),
and checks whether the model outputs Mu or Nu. Follows the same approach
as no_think_baseline.py.

Usage:
    modal run scripts/sanity_check_deepmath.py
    modal run scripts/sanity_check_deepmath.py --n 20
"""

import os

import modal

app = modal.App("deepmath-sanity-check")

model_cache_vol = modal.Volume.from_name("qwen3-8b-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch==2.9.0",
        "transformers>=4.51.0",
        "accelerate>=1.6.0",
        "datasets>=4.5.0",
        "numpy>=2.4.1",
    )
    .add_local_python_source("src")
)

MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 220]  # </think>\n\nAnswer: (space)


@app.function(
    image=image,
    gpu="a100-80gb",
    volumes={"/model-cache": model_cache_vol},
    timeout=1200,
)
def run_sanity_check(n: int = 10, seed: int = 137):
    """Run DeepMath no-think sanity check on a Modal A100."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.datasets import load_deepmath
    from src.prompts import build_verification_prompt

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    print(f"Loading DeepMath problems (n={n}, seed={seed})...")
    problems = load_deepmath(n=n, seed=seed)
    print(f"  Got {len(problems)} problems")

    # Build prompts with swapped-label config A
    for p in problems:
        result = build_verification_prompt(p, "swapped", "A")
        p["verification_prompt"] = result["verification_prompt"]
        p["label_mapping"] = result["label_mapping"]

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", cache_dir="/model-cache")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        cache_dir="/model-cache",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    print("Model loaded.")

    print(f"\nToken verification:")
    print(f"  Mu token: {tokenizer.decode([MU_TOKEN_ID])!r} (id={MU_TOKEN_ID})")
    print(f"  Nu token: {tokenizer.decode([NU_TOKEN_ID])!r} (id={NU_TOKEN_ID})")
    print(f"  Close suffix: {tokenizer.decode(CLOSE_TOKEN_IDS)!r}")

    device = "cuda:0"
    close_ids = torch.tensor([CLOSE_TOKEN_IDS], device=device)

    correct_count = 0
    answered_count = 0
    total = len(problems)
    results = []

    print(f"\n{'='*80}")
    print(f"Running no-think force-answer on {total} DeepMath problems")
    print(f"{'='*80}\n")

    for i, p in enumerate(problems):
        # Build chat-templated input (thinking mode on, but we immediately close it)
        messages = [{"role": "user", "content": p["verification_prompt"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

        # Immediately force-close: input ends with <think>\n, we append </think>\n\nAnswer:
        input_with_close = torch.cat([input_ids, close_ids], dim=1)

        with torch.no_grad():
            gen_output = model.generate(
                input_ids=input_with_close,
                attention_mask=torch.ones_like(input_with_close),
                max_new_tokens=15,
                do_sample=False,
            )

        # Extract generated tokens and scan for Mu/Nu
        generated_ids = gen_output[0, input_with_close.shape[1]:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Token-level scan for first Mu or Nu token
        answer = None
        for tid in generated_ids:
            if tid == MU_TOKEN_ID:
                answer = "Mu"
                break
            if tid == NU_TOKEN_ID:
                answer = "Nu"
                break

        # Fallback: string-level scan
        if answer is None:
            text_lower = generated_text.lower()
            mu_pos = text_lower.find("mu")
            nu_pos = text_lower.find("nu")
            if mu_pos >= 0 and nu_pos >= 0:
                answer = "Mu" if mu_pos < nu_pos else "Nu"
            elif mu_pos >= 0:
                answer = "Mu"
            elif nu_pos >= 0:
                answer = "Nu"

        # Check correctness
        expected = p["label_mapping"]["valid"] if p["ground_truth"] == "Yes" else p["label_mapping"]["invalid"]
        is_correct = answer == expected

        if answer in ("Mu", "Nu"):
            answered_count += 1
        if is_correct:
            correct_count += 1

        results.append({
            "problem_id": p["problem_id"],
            "ground_truth": p["ground_truth"],
            "presented_answer": p["presented_answer"],
            "original_answer": p["original_answer"],
            "answer": answer,
            "expected": expected,
            "correct": is_correct,
        })

        print(f"[{i+1}/{total}] {p['problem_id']}")
        print(f"  ground_truth={p['ground_truth']}  presented={p['presented_answer']}  original={p['original_answer']}")
        print(f"  answer={answer}  expected={expected}  correct={is_correct}")
        print(f"  generated: {generated_text.strip()!r}")
        print()

    summary = {
        "total": total,
        "answered_mu_nu": answered_count,
        "correct": correct_count,
        "answer_rate": f"{answered_count/total*100:.0f}%",
        "accuracy": f"{correct_count/total*100:.0f}%",
        "results": results,
    }

    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"  Total:          {total}")
    print(f"  Answered Mu/Nu: {answered_count}/{total} ({summary['answer_rate']})")
    print(f"  Correct:        {correct_count}/{total} ({summary['accuracy']})")
    print(f"{'='*80}")

    return summary


@app.local_entrypoint()
def main(n: int = 10, seed: int = 137):
    """Entry point for `modal run scripts/sanity_check_deepmath.py`."""
    print(f"Submitting sanity check to Modal (n={n}, seed={seed})...")
    summary = run_sanity_check.remote(n=n, seed=seed)

    print("\n=== Results returned from Modal ===")
    print(f"  Answered Mu/Nu: {summary['answered_mu_nu']}/{summary['total']} ({summary['answer_rate']})")
    print(f"  Correct:        {summary['correct']}/{summary['total']} ({summary['accuracy']})")
