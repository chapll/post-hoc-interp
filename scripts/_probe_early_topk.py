"""Quick script to check what tokens get highest probability at early force-answer positions."""
import torch
import json
import glob
import os
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda:0")
model.eval()

# Find traces: data/traces/valid/{dataset}/{problem_id}/trace_00.json
trace_files = sorted(glob.glob("data/traces/valid/*/*/trace_00.json"))[:5]
CLOSE = [151668, 271, 16141, 25]  # </think>\n\nAnswer:

for tf in trace_files:
    d = json.load(open(tf))
    meta = d["metadata"]
    ids_list = d["token_ids"]
    pid = meta["problem_id"]
    gt = meta["ground_truth"]
    # expected token: if ground_truth is Yes (presented answer correct), valid_token; else invalid_token
    expected = meta["valid_token"] if gt == "Yes" else meta["invalid_token"]
    think_start, think_end = meta["think_boundaries"][0]
    think_length = think_end - think_start

    ids = torch.tensor(ids_list, device="cuda:0").unsqueeze(0)

    for nthink in [1, 3]:
        probe = torch.cat([
            ids[:, :think_start],
            ids[:, think_start:think_start+nthink],
            torch.tensor([CLOSE], device="cuda:0")
        ], dim=1)
        with torch.no_grad():
            logits = model(probe).logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
        topk = torch.topk(probs, 10)
        print(f"{pid} (expected={expected}, gt={gt}, think_len={think_length})")
        print(f"  After {nthink} thinking token(s):")
        for p, idx in zip(topk.values, topk.indices):
            tok = tokenizer.decode([idx.item()])
            print(f"    {idx.item():>8d}  P={p.item():.4f}  [{tok}]")
    print()
