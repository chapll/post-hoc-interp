"""Merge sharded probe and freeform eval results."""
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Merge probe shards
probe_dir = "experiments/sleeper_agent/approach_c/probe_results"
all_triggered, all_normal = [], []
trig_by_ds, norm_by_ds = {}, {}
for i in range(4):
    path = os.path.join(probe_dir, f"shard_{i}.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        d = json.load(f)
    all_triggered.extend(d["triggered_curves"])
    all_normal.extend(d["normal_curves"])
    for ds, curves in d["triggered_by_ds"].items():
        trig_by_ds.setdefault(ds, []).extend(curves)
    for ds, curves in d["normal_by_ds"].items():
        norm_by_ds.setdefault(ds, []).extend(curves)

out = os.path.join(probe_dir, "merged.json")
with open(out, "w") as f:
    json.dump({
        "triggered_curves": all_triggered,
        "normal_curves": all_normal,
        "triggered_by_ds": trig_by_ds,
        "normal_by_ds": norm_by_ds,
    }, f, indent=2)
print(f"Probe: {len(all_triggered)} trig, {len(all_normal)} norm -> {out}")
for ds in sorted(trig_by_ds):
    print(f"  {ds}: {len(trig_by_ds[ds])} trig, {len(norm_by_ds.get(ds, []))} norm")

# Merge freeform shards
freeform_dir = "experiments/sleeper_agent/approach_c"
all_results = []
for i in range(4):
    path = os.path.join(freeform_dir, f"freeform_results_shard{i}.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        all_results.extend(json.load(f))

out2 = os.path.join(freeform_dir, "freeform_eval", "merged.json")
os.makedirs(os.path.dirname(out2), exist_ok=True)
with open(out2, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Freeform: {len(all_results)} traces -> {out2}")
for ds in sorted(set(r["dataset"] for r in all_results)):
    sub = [r for r in all_results if r["dataset"] == ds]
    print(f"  {ds}: {len(sub)} traces")
