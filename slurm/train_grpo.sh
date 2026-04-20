#!/bin/bash
#SBATCH --job-name=nemo-rl-grpo
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=$PROJECT_DIR/logs/grpo_%j.log
#SBATCH --error=$PROJECT_DIR/logs/grpo_%j.err

# ============================================================
# Nemotron RL GRPO Training
# Using Nemotron-3-Nano-30B with LoRA (rank 32)
# ============================================================

source "$(dirname "$0")/env_setup.sh"
cd "$PROJECT_DIR/RL"

echo "=== Nemotron RL GRPO Training ==="
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"

# Path to training data
DATA_PATH="$DATA_DIR/final_Nemotron_training_data.csv"

# Run GRPO training
echo "Starting GRPO training..."
python nemo_reasoning/train_grpo.py \
    --config nemo_reasoning/grpo_nemotron.yaml \
    --data_path $DATA_PATH \
    --output_dir $RESULT_DIR/nemo_reasoning_rl \
    2>&1 | tee $LOG_DIR/train_$SLURM_JOB_ID.log

# Run evaluation on validation set with COMPETITION PARAMETERS
echo "Running evaluation with competition parameters..."
python nemo_reasoning/evaluate.py \
    --model_path $RESULT_DIR/nemo_reasoning_rl/checkpoints/best_model \
    --base_model nvidia/Nemotron-3-Nano-30B-A3B-FP8 \
    --data_path $DATA_PATH \
    --output_dir $RESULT_DIR/evaluation \
    --use_lora \
    --batch_size 8 \
    --max_new_tokens 7680 \
    --temperature 0.0 \
    --top_p 1.0 \
    2>&1 | tee $LOG_DIR/eval_$SLURM_JOB_ID.log

echo "=== Training Complete at $(date) ==="
