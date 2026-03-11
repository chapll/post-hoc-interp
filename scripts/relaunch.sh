#!/bin/bash
kinit && aklog
cd /dfs/scratch2/gorn/post-hoc-interp
nohup bash scripts/launch_full_pipeline.sh > logs/full_pipeline.log 2>&1 &
echo "Launched PID=$!"
