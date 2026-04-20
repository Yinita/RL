#!/bin/bash
# ============================================================
# Nemotron RL shared environment setup
# Source this file from sbatch scripts:
#   source $(dirname "$0")/env_setup.sh
# ============================================================

# Project paths (adjust these for your cluster)
export WORK1="/work1/yt"  # Adjust to your work directory
export PROJECT_DIR="$WORK1/RL"
export DATA_DIR="$PROJECT_DIR/data"
export LOG_DIR="$PROJECT_DIR/logs"
export RESULT_DIR="$PROJECT_DIR/results"
export RL_DIR="$PROJECT_DIR"

# Cache paths (prevent writing to /home1)
export HF_HOME="$WORK1/.cache/huggingface"
export HF_DATASETS_CACHE="$WORK1/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$WORK1/.cache/huggingface/transformers"
export PIP_CACHE_DIR="$WORK1/.cache/pip"
export TORCH_HOME="$WORK1/.cache/torch"
export UV_CACHE_DIR="$WORK1/.cache/uv"
export CONDA_PKGS_DIRS="$WORK1/.cache/conda/pkgs"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$WORK1/.cache/pycache"

# CUDA (adjust for your cluster)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

# Conda
source /work1/jwang/jindongwang/pkulium/mini_conda/etc/profile.d/conda.sh
conda activate /work1/yt/.conda/envs/rl

mkdir -p "$DATA_DIR" "$LOG_DIR" "$RESULT_DIR"

# ── Model config ────────────────────────────────────────────
MODEL_NAME="nvidia/Nemotron-3-Nano-30B-A3B-FP8"

# ── Training config ─────────────────────────────────────────
export NUM_GPUS=8
export BATCH_SIZE=8
export GRAD_ACCUM_STEPS=16
export LEARNING_RATE=5e-6
export MAX_EPOCHS=1
export MAX_STEPS=1000

# ── WandB config ───────────────────────────────────────────
export WANDB_PROJECT="nemo-rl-0420"
export WANDB_API_KEY="7f885b3993e38c9c390b4c6919e1b256caab13d0"

# Login to WandB
wandb login --verify $WANDB_API_KEY

echo "=== Environment Setup Complete ==="
echo "Project Dir: $PROJECT_DIR"
echo "RL Dir: $RL_DIR"
echo "Log Dir: $LOG_DIR"
echo "Result Dir: $RESULT_DIR"
echo "Conda Env: rl"
echo "Model: $MODEL_NAME"
