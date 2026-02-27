---
phase: quick-005
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/no_think_baseline.py
  - configs/no_think_baseline.yaml
autonomous: true

must_haves:
  truths:
    - "Script loads all 3 datasets, builds swapped-label verification prompts (config A only), tokenizes with enable_thinking=True, force-closes think block at position 0, generates up to 15 tokens greedy, and checks correctness"
    - "Results parquet contains per-problem correctness, answer, probabilities"
    - "Discard list JSON contains problem_ids that the model answers correctly without thinking"
  artifacts:
    - path: "scripts/no_think_baseline.py"
      provides: "No-think baseline evaluation script"
    - path: "configs/no_think_baseline.yaml"
      provides: "Config for no-think baseline"
---

<objective>
Create a script that evaluates Qwen3-8B on all benchmark problems with thinking disabled (force-close think block immediately at position 0). Problems the model gets right without thinking are flagged for exclusion -- they are either memorized or too easy.

Purpose: Filter out trivial problems before the main probing experiments so analysis focuses on problems that actually require thinking.
Output: `scripts/no_think_baseline.py` and `configs/no_think_baseline.yaml`
</objective>

<execution_context>
@/home/aahmad/.claude/get-shit-done/workflows/execute-plan.md
@/home/aahmad/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@src/model.py
@src/datasets.py
@src/prompts.py
@src/config.py
@src/multiprocess.py
@src/utils.py
@configs/base.yaml
@configs/force_answer.yaml
@scripts/force_answer_experiment.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create config and no-think baseline script</name>
  <files>configs/no_think_baseline.yaml, scripts/no_think_baseline.py</files>
  <action>
**configs/no_think_baseline.yaml:**
```yaml
no_think_baseline:
  max_new_tokens: 15
  problems_per_dataset: 300

paths:
  output_dir: "${oc.env:EXP_DIR}/no_think_baseline"

gpus: [0, 1, 2, 3]
```

**scripts/no_think_baseline.py:**

Follow the same structural pattern as `scripts/force_answer_experiment.py` (imports, token constants, worker function, main with shard merge).

Import from force_answer_experiment.py:
- `extract_answer` -- dual token+string matching for Rho/Tau
- `extract_rho_tau_probs` -- softmax probability extraction
- Token constants: `RHO_FIRST_TOKEN_ID`, `TAU_TOKEN_ID`, `CLOSE_TOKEN_IDS`

Use `sys.path.insert` to import from force_answer_experiment (same pattern as cross_seed_probing.py if it exists, or inline the path insertion).

**Core logic per problem:**

1. Load all 3 datasets via `load_dataset_by_name(name, n=cfg.no_think_baseline.problems_per_dataset, seed=cfg.seed)` for each of ["gsm_symbolic", "folio", "strategyqa"].

2. For each problem, build verification prompt with swapped-label protocol config "A" only:
   ```python
   prompt_info = build_verification_prompt(problem, label_protocol="swapped", config="A")
   ```

3. Determine expected_token and correct_first_token_id from label_mapping:
   - Config A: valid="Rho", invalid="Tau"
   - If ground_truth == "Yes": expected_token = "Rho", correct_first_token_id = RHO_FIRST_TOKEN_ID
   - If ground_truth == "No": expected_token = "Tau", correct_first_token_id = TAU_TOKEN_ID

4. Tokenize with chat template + enable_thinking=True:
   ```python
   messages = [{"role": "user", "content": prompt_info["verification_prompt"]}]
   text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
   input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
   ```
   The generation prompt ends with `<think>\n` -- the model is about to start thinking.

5. Force-close thinking at position 0 by appending CLOSE_TOKEN_IDS right after input:
   ```python
   close_ids = torch.tensor([CLOSE_TOKEN_IDS], dtype=torch.long, device=device)
   input_with_close = torch.cat([input_ids, close_ids], dim=1)
   ```

6. Generate up to 15 tokens with greedy decoding:
   ```python
   gen_output = raw_model.generate(
       input_ids=input_with_close,
       attention_mask=torch.ones_like(input_with_close),
       max_new_tokens=cfg.no_think_baseline.max_new_tokens,
       do_sample=False,
       return_dict_in_generate=True,
       output_logits=True,
   )
   ```

7. Extract answer using `extract_answer()` and probabilities using `extract_rho_tau_probs()` (same as force_answer_experiment.py).

8. Build result row with: problem_id, dataset, ground_truth, expected_token, correct (bool), answer, no_answer, rho_prob, tau_prob, correct_token_prob, correct_token_rank, token_answer, string_answer, agreement.

**Worker function (_worker):**
- Signature: `(rank, gpu_id, items, cfg)` matching run_parallel convention
- Load model via `load_model(gpu_id=gpu_id)`, get `raw_model = model._model`
- Process items with tqdm progress bar
- Write shard to `{output_dir}/results_shard_{rank}.parquet`
- Call `torch.cuda.empty_cache()` after each problem

**Main function:**
- `load_config()` with argparse --config support
- Load all 3 datasets, combine into single items list with dataset_name already in each problem dict
- `run_parallel(_worker, items, gpu_ids, cfg)`
- Merge shards into `results.parquet`
- Delete shard files after merge

**Post-merge analysis in main:**
- Print per-dataset summary table: total, correct_without_thinking, pct_trivial
- Overall summary
- Build discard list: list of problem_ids where correct == True
- Save discard list to `{output_dir}/trivial_problems.json` as:
  ```json
  {
    "description": "Problems correctly answered without thinking (trivial/memorized)",
    "total_problems": N,
    "trivial_count": M,
    "trivial_pct": X.X,
    "by_dataset": {"gsm_symbolic": [...], "folio": [...], "strategyqa": [...]},
    "all_trivial_ids": [...]
  }
  ```

**Important implementation details:**
- Use `from dotenv import load_dotenv; load_dotenv()` at top (before other imports that use env vars)
- Import sys, add project root to path for force_answer_experiment imports:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  ```
  Then: `from scripts.force_answer_experiment import extract_answer, extract_rho_tau_probs, RHO_FIRST_TOKEN_ID, TAU_TOKEN_ID, CLOSE_TOKEN_IDS`
  Actually -- since force_answer_experiment.py defines these at module level with no side effects, importing directly works. But the constants are also defined in that file's global scope. To be safe, just re-define the token constants locally (they are 4 lines, avoids import complexity). Import only `extract_answer` and `extract_rho_tau_probs` from force_answer_experiment, OR just copy these two functions inline (they are ~60 lines total and self-contained). Use your judgment -- if the import is clean, import; if it causes circular issues, inline them.
- Use `torch.no_grad()` context for all generation
- Sort final dataframe by problem_id for deterministic output
  </action>
  <verify>
1. `python -c "import ast; ast.parse(open('scripts/no_think_baseline.py').read()); print('syntax ok')"` -- passes
2. `python -c "from omegaconf import OmegaConf; c = OmegaConf.load('configs/no_think_baseline.yaml'); print(c)"` -- loads without error
3. Check script has: worker function with (rank, gpu_id, items, cfg) signature, main() with load_config/run_parallel/shard-merge pattern, extract_answer usage, trivial_problems.json output
  </verify>
  <done>
- `scripts/no_think_baseline.py` exists with worker pattern matching force_answer_experiment.py
- `configs/no_think_baseline.yaml` exists with no_think_baseline section and paths
- Script loads all 3 datasets, builds config-A swapped-label prompts, force-closes thinking at position 0, generates greedy, extracts answers with dual matching, saves results.parquet and trivial_problems.json
- Multi-GPU support via run_parallel
  </done>
</task>

</tasks>

<verification>
- Both files parse without syntax errors
- Config loads via OmegaConf
- Script follows established worker/main/shard-merge pattern from force_answer_experiment.py
- Token constants match verified values (CLOSE_TOKEN_IDS=[151668,198], RHO_FIRST_TOKEN_ID=49, TAU_TOKEN_ID=92713)
</verification>

<success_criteria>
- `scripts/no_think_baseline.py` is a complete, runnable experiment script
- `configs/no_think_baseline.yaml` provides all needed config
- Script produces results.parquet (per-problem correctness) and trivial_problems.json (discard list)
- Follows all established patterns: load_config, run_parallel, worker shard writes, main shard merge
</success_criteria>

<output>
After completion, create `.planning/quick/005-no-think-baseline-filter/005-SUMMARY.md`
</output>
