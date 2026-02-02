#!/bin/bash
#SBATCH --job-name=neuroburgers
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=1           # Adjust as needed
#SBATCH --mem=32G                   # Adjust memory. This is RAM not VRAM. We need a lot of ram to compile jax code when using a larger batch size.
#SBATCH --partition=gpu             # Replace with correct GPU partition if needed

#SBATCH -C "GPU_SKU:H100_SXM5|GPU_SKU:L40S"        # Only request either H100_SXM5 or L40S. No other gpus work with my flax installation (V100, V100S, V100_SXM2, 
                                    # P40 and RTX_2080Ti result in segmentation faults. And A30_MIG-1g.6gb says "sbatch: error: Batch job submission 
                                    # failed: Requested node configuration is not available"). H100 is about 2x faster than L40S.

#SBATCH --array=1-1

# Load any needed modules here (if your HPC uses module system)
# module load ollama
module load python/3.12.1
module load cuda/12.6.1
source jax_env/bin/activate

nvidia-smi

# Start Ollama server in the background
# ollama serve &
# sleep 10

# Run your Python script
# POS_WEIGHT=$SLURM_ARRAY_TASK_ID
# echo "Running with pos_weight=$POS_WEIGHT"
# python3 -u 7_train_transformer.py --pos_weight $POS_WEIGHT

python3 -u calculate_nutrition.py

# Optional: stop the Ollama server after job finishes
# pkill ollama

# To request an interactive shell simply do sh_dev. To request one with a gpu do sh_dev -g 1. Or you can do:
# srun --partition=gpu --gres=gpu:1 --time=02:00:00 --pty bash
