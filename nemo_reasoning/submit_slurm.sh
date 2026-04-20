#!/bin/bash
#SBATCH --job-name=nemo-rl-0420
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=<your_account>  # UPDATE THIS
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>  # UPDATE THIS

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Login to WandB
wandb login --verify 7f885b3993e38c9c390b4c6919e1b256caab13d0

# Create directories
mkdir -p logs
mkdir -p results/nemo_reasoning_rl

# Print job info
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Path to training data - UPDATE THIS
DATA_PATH="/path/to/final_Nemotron_training_data.csv"

# Run GRPO training
python nemo_reasoning/train_grpo.py \
    --config nemo_reasoning/grpo_nemotron.yaml \
    --data_path $DATA_PATH \
    --output_dir results/nemo_reasoning_rl \
    2>&1 | tee logs/train_$SLURM_JOB_ID.log

# Run evaluation on validation set
python nemo_reasoning/evaluate.py \
    --model_path results/nemo_reasoning_rl/checkpoints/best_model \
    --base_model nvidia/Nemotron-3-Nano-30B-A3B-FP8 \
    --data_path $DATA_PATH \
    --output_dir results/evaluation \
    --use_lora \
    --batch_size 8 \
    --max_new_tokens 7680 \
    --temperature 0.0 \
    --top_p 1.0 \
    2>&1 | tee logs/eval_$SLURM_JOB_ID.log

# Print completion
echo "Job completed at $(date)"
