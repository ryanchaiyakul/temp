#!/bin/bash
methods=("hodel" "pinn" "deq")
noise_levels=(0.0 0.001 0.01 0.03)
for m in "${methods[@]}"; do
    for n in "${noise_levels[@]}"; do
        echo "------------------------------------------------"
        echo "Running Method: $m | Noise: $n | Seeds: $seeds"
        echo "------------------------------------------------"
        uv run loss_landscape.py --method "$m" --eta "$n"
    done
done

for n in "${noise_levels[@]}"; do
    uv run loss_landscape.py --method node --eta "$n" --n_steps 10 --n_disps 5 --n_disps_hessian 5
done
echo "All benchmark runs completed!"
