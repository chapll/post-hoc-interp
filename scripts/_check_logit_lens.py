"""Quick check: what do the model's output logits look like at positions within the think block?

If P(Mu) and P(Nu) are near-zero during reasoning (as expected), the logit lens
approach is fundamentally wrong for comparing to force-answer.
"""

import json
import torch
import torch.nn.functional as F
from src.model import load_model
from src.tokens import get_think_content_mask

device = torch.device("cuda:0")
model, tok = load_model(gpu_id=0)

with open("src/experimental/gorn/data/traces/valid/folio/folio_259_cfga/trace_00.json") as f:
    data = json.load(f)
token_ids = data["token_ids"]
gt = data["metadata"]["ground_truth"]

input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
with model.trace(input_ids):
    saved_final = model.lm_head.output.save()

final_logits = saved_final.detach().squeeze(0)

mask = get_think_content_mask(token_ids)
eligible = [i for i, m in enumerate(mask) if m]

mu_id = 39120  # "Mu" (single token)
nu_id = 60335  # "Nu" (single token)

positions = [
    eligible[0],
    eligible[len(eligible) // 4],
    eligible[len(eligible) // 2],
    eligible[3 * len(eligible) // 4],
    eligible[-1],
]

print(f"Trace: folio_259_cfga, GT={gt}")
print(f"Correct token: Mu (token id = {mu_id})")
print(f"Total eligible tokens: {len(eligible)}")
print()

for pos in positions:
    probs = F.softmax(final_logits[pos].float(), dim=-1)
    p_mu = probs[mu_id].item()
    p_nu = probs[nu_id].item()
    norm_pos = (pos - eligible[0]) / (eligible[-1] - eligible[0])

    top5_vals, top5_ids = probs.topk(5)
    top5_tokens = [tok.decode([tid.item()]) for tid in top5_ids]

    print(f"pos={pos} (norm={norm_pos:.3f}): P(Mu)={p_mu:.6f}, P(Nu)={p_nu:.6f}")
    print(f"  Top-5: {list(zip(top5_tokens, [f'{v:.4f}' for v in top5_vals.tolist()]))}")
    print()
