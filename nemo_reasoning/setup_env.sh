#!/bin/bash
# Setup script for Nemotron Reasoning RL Training

set -e

echo "🚀 Setting up Nemotron Reasoning RL Training Environment"

# Check if we're in the RL directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the RL repository root"
    exit 1
fi

# Install NeMo-RL dependencies
echo "📦 Installing NeMo-RL dependencies..."
pip install -e ".[vllm,gpu]"

# Install additional dependencies
echo "📦 Installing additional dependencies..."
pip install pandas scikit-learn peft

# Login to WandB
echo "🔐 Logging into WandB..."
wandb login --verify 7f885b3993e38c9c390b4c6919e1b256caab13d0

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p results/nemo_reasoning_rl
mkdir -p results/evaluation

# Prompt for data path
echo ""
echo "📝 Configuration"
echo "=================="
read -p "Enter path to training data CSV: " DATA_PATH

# Update config with data path
if [ -f "nemo_reasoning/grpo_nemotron.yaml" ]; then
    echo "📝 Updating data path in config..."
    sed -i "s|/path/to/final_Nemotron_training_data.csv|$DATA_PATH|g" nemo_reasoning/grpo_nemotron.yaml
else
    echo "⚠️  Warning: Config file not found, please update manually"
fi

# Prompt for Slurm account (if using cluster)
read -p "Enter Slurm account name (leave empty if not using Slurm): " SLURM_ACCOUNT
read -p "Enter your email for Slurm notifications (leave empty if not using Slurm): " SLURM_EMAIL

if [ ! -z "$SLURM_ACCOUNT" ]; then
    echo "📝 Updating Slurm script..."
    sed -i "s/<your_account>/$SLURM_ACCOUNT/g" nemo_reasoning/submit_slurm.sh
fi

if [ ! -z "$SLURM_EMAIL" ]; then
    sed -i "s/<your_email>/$SLURM_EMAIL/g" nemo_reasoning/submit_slurm.sh
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review and update nemo_reasoning/grpo_nemotron.yaml if needed"
echo "2. For local training: python nemo_reasoning/train_grpo.py --data_path $DATA_PATH"
echo "3. For Slurm training: sbatch nemo_reasoning/submit_slurm.sh"
echo ""
echo "WandB project: nemo-rl-0420"
