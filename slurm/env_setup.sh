#!/bin/bash

export WORK1="/work1/yt"
export PROJECT_DIR="$WORK1/RL"
export DATA_DIR="$PROJECT_DIR/data"
export LOG_DIR="$PROJECT_DIR/logs"
export RESULT_DIR="$PROJECT_DIR/results"
export RL_DIR="$PROJECT_DIR"

export HF_HOME="$WORK1/.cache/huggingface"
export HF_DATASETS_CACHE="$WORK1/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$WORK1/.cache/huggingface/transformers"
export PIP_CACHE_DIR="$WORK1/.cache/pip"
export TORCH_HOME="$WORK1/.cache/torch"
export UV_CACHE_DIR="$WORK1/.cache/uv"
export CONDA_PKGS_DIRS="$WORK1/.cache/conda/pkgs"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$WORK1/.cache/pycache"

source /work1/jwang/jindongwang/pkulium/mini_conda/etc/profile.d/conda.sh
conda activate /work1/yt/.conda/envs/rl || exit 1

mkdir -p "$DATA_DIR" "$LOG_DIR" "$RESULT_DIR"

export NUM_GPUS=4
export BATCH_SIZE=8
export GRAD_ACCUM_STEPS=16
export LEARNING_RATE=5e-6
export MAX_EPOCHS=1
export MAX_STEPS=1000

export WANDB_PROJECT="nemo-rl-0420"
export WANDB_API_KEY="你的key"

echo "=== Environment Setup Complete ==="
echo "Project Dir: $PROJECT_DIR"
echo "Log Dir: $LOG_DIR"
echo "Result Dir: $RESULT_DIR"
echo "Conda Env: /work1/yt/.conda/envs/rl"