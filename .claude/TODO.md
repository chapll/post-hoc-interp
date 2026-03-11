# Scaled-Up Pipeline TODO

## Status: FULL PIPELINE RUNNING (Stage 0 generating)

## Configuration (final)
- MAX_NEW_TOKENS: 16384 (avoid biasing against long traces)
- 1 trace per problem (TRACE_INDEX=0), doubled problem counts
- 8-GPU sharding for Stages 1+2
- Pool ranges: A(0-12K), B(12K-18K), C(18K-24K), D(24K-28K), E(28K-32K)

| Pool | Problems | Configs | Total traces | Purpose |
|------|----------|---------|-------------|---------|
| A | 6000 | 2 | 12000 | Direction extraction, probe 2000 |
| B | 3000 | 2 | 6000 | Sleeper SFT |
| C | 3000 | 2 | 6000 | Steering eval, scan 2000 |
| D | 1500 | 2 | 3000 | Binary transfer |
| E | 1500 | 1 | 1500 | Non-binary transfer |
| Total | 15000 | | 28500 | |

## Pipeline Stages (sequential in launch_full_pipeline.sh)
1. Stage 0: vLLM tp=8 generation (all pools) -- RUNNING
2. Stage 1: Scan Pool C (8 shards, 8 GPUs)
3. Stage 2: Probe Pool A (8 shards, 8 GPUs) + merge/compute directions
4. Stage 3: Build SFT data + train sleeper LoRA
5. Stages 4-7: Steering + transfer tests (5 GPUs in parallel)

## Monitor
```bash
ssh ampere9 "echo 'Argon@224R_esearch231N' | kinit 2>/dev/null && aklog 2>/dev/null && tail -5 /dfs/scratch2/gorn/post-hoc-interp/logs/full_pipeline.log"
```

## Completion Checks
- Stage 0: data/traces_v2/valid/deepmath/ populated
- Stage 1: experiments/v2/scan/pool_C/scan_results.json
- Stage 2: experiments/v2/directions/knowing_direction.safetensors
- Stage 3: experiments/sleeper_v2/approach_c/final_adapter/
- Stage 4: experiments/v2/steering/steering_{posthoc,knowing}_results.json
- Stage 5: experiments/v2/transfer/binary/transfer_results.json
- Stage 6: experiments/v2/transfer/nonbinary/transfer_results.json
- Stage 7: experiments/v2/transfer/crossdomain/crossdomain_results.json
