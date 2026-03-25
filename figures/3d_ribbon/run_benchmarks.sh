#!/bin/bash
methods=("hodel" "pinn" "deq" "node")
for m in "${methods[@]}"; do
    echo "------------------"
    echo "Running Method: $m"
    echo "------------------"
    uv run benchmark.py --method "$m"
done

echo "All benchmark runs completed!"
