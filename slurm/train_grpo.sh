#!/bin/bash
#SBATCH --job-name=nemo-rl-grpo
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/work1/yt/RL/logs/grpo_%j.log
#SBATCH --error=/work1/yt/RL/logs/grpo_%j.err

set -euo pipefail

source "/work1/yt/RL/slurm/env_setup.sh"
cd "$PROJECT_DIR"

echo "=== Nemotron RL GRPO Training ==="
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"

python --version
which python

DATA_PATH="$DATA_DIR/final_Nemotron_training_data.csv"

echo "Starting GRPO training..."
python nemo_reasoning/train_grpo.py \
    --config nemo_reasoning/grpo_nemotron.yaml \
    --data_path "$DATA_PATH" \
    --output_dir "$RESULT_DIR/nemo_reasoning_rl" \
    2>&1 | tee "$LOG_DIR/train_$SLURM_JOB_ID.log"

echo "Running evaluation with competition parameters..."
python nemo_reasoning/evaluate.py \
    --model_path "$RESULT_DIR/nemo_reasoning_rl/checkpoints/best_model" \
    --base_model nvidia/Nemotron-3-Nano-30B-A3B-FP8 \
    --data_path "$DATA_PATH" \
    --output_dir "$RESULT_DIR/evaluation" \
    --use_lora \
    --batch_size 8 \
    --max_new_tokens 7680 \
    --temperature 0.0 \
    --top_p 1.0 \
    2>&1 | tee "$LOG_DIR/eval_$SLURM_JOB_ID.log"

echo "=== Training Complete at $(date) ==="