# Nemotron Reasoning Challenge - RL Training

This directory contains scripts for training and evaluating a Nemotron-3-Nano-30B model using GRPO (Group Relative Policy Optimization) for the NVIDIA Nemotron Reasoning Challenge.

## Project Structure

```
RL/
├── nemo_reasoning/          # Custom Nemotron training scripts
│   ├── __init__.py              # Package initialization
│   ├── dataset.py               # Custom dataset processor
│   ├── reward.py                # Reward function for evaluation
│   ├── grpo_nemotron.yaml       # GRPO configuration
│   ├── train_grpo.py            # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── prompts/
│   │   └── default.txt          # Default prompt template
│   └── README.md                # This file
├── slurm/                   # Slurm submission scripts
│   ├── env_setup.sh         # Environment setup (sourced by sbatch)
│   └── train_grpo.sh        # GRPO training job submission
├── data/                    # Training data
│   └── final_Nemotron_training_data.csv
└── README.md                # NeMo-RL repository README
```

## Setup

### Option 1: Local Development with Conda

```bash
cd RL

# Setup conda environment named 'rl'
conda create -n rl python=3.10 -y
conda activate rl

# Install dependencies
pip install -e ".[vllm,gpu]"
pip install pandas scikit-learn peat
pip install wandb

# Login to WandB
wandb login --verify 7f885b3993e38c9c390b4c6919e1b256caab13d0
```

### Option 2: Slurm Cluster

```bash
cd /work1/yt/RL

# Clone repository
git clone git@github.com:Yinita/RL.git

# Upload data directory to cluster
# scp -r data/ user@cluster:/work1/yt/RL/

# Update paths in slurm/env_setup.sh:
# - WORK1: Your work directory
# - PROJECT_DIR: /work1/yt/RL
# - Conda path: Your conda installation

# Submit job
sbatch slurm/train_grpo.sh
```

## Training

### Option 1: Slurm (Recommended for Cluster)

```bash
cd /work1/yt/RL
sbatch slurm/train_grpo.sh
```

### Option 2: Local/Interactive

```bash
cd RL
conda activate rl

python nemo_reasoning/train_grpo.py \
    --config nemo_reasoning/grpo_nemotron.yaml \
    --data_path data/final_Nemotron_training_data.csv \
    --output_dir results/nemo_reasoning_rl
```

## Configuration

### Training Parameters (for exploration)
Key parameters in `grpo_nemotron.yaml`:

- **LoRA rank**: 32 (maximum allowed by competition)
- **Batch size**: 1024 global, 8 micro
- **Generation**: 64 prompts/step, 16 generations/prompt
- **Max tokens**: 20000 (training: allow longer generation for exploration)
- **Temperature**: 1.0 (training: higher temperature for diversity)
- **Top-p**: 0.9 (training: nucleus sampling for exploration)
- **GPU memory**: 0.85 utilization (competition requirement)
- **Token compression**: Enabled with weight 0.05 (mild penalty for overlong responses)

### Evaluation Parameters (for competition)
The evaluation script uses competition-compliant parameters:

- **Max tokens**: 7680 (competition requirement)
- **Temperature**: 0.0 (competition requirement)
- **Top-p**: 1.0 (competition requirement)

Training uses more aggressive parameters for exploration, while evaluation uses strict competition parameters for final scoring.

## Evaluation

```bash
cd RL
conda activate rl

python nemo_reasoning/evaluate.py \
    --model_path results/nemo_reasoning_rl/checkpoints/best_model \
    --base_model nvidia/Nemotron-3-Nano-30B-A3B-FP8 \
    --data_path data/final_Nemotron_training_data.csv \
    --output_dir results/evaluation \
    --use_lora \
    --batch_size 8
```

## Data Split

The training script automatically splits data:
- **Training**: 90% of data
- **Validation**: 10% of data (stratified by label if available)

## Reward Function

The reward function evaluates answers based on:
1. Exact string match
2. Numerical tolerance (10^-2 relative tolerance)
3. Case-insensitive string comparison
4. **Token compression penalty** (during training only)

### Token Compression
During training, a mild penalty is applied to encourage concise responses:
- **Target length**: 7680 tokens (competition requirement)
- **Compression weight**: 0.05 (low weight to minimize impact on accuracy)
- **Minimum accuracy threshold**: 0.8 (penalty only applied if answer is correct enough)
- **Penalty formula**: Linear reduction based on excess tokens
- **Minimum reward**: 0.5 (to avoid excessive penalty)

This encourages the model to learn concise reasoning without significantly hurting accuracy. During evaluation, token compression is disabled, and only accuracy is measured.

Answers must be in `\boxed{}` format for optimal evaluation.

## WandB Logging

Metrics are logged to:
- **Project**: `nemo-rl-0420`
- **Run name**: `nemotron-reasoning-grpo`

Logged metrics:
- Training loss
- Validation accuracy
- KL divergence
- Reward statistics
- GPU utilization

## Output

After training, you will find:
- **Checkpoints**: `results/nemo_reasoning_rl/checkpoints/`
- **Logs**: `logs/nemo_reasoning_rl/`
- **WandB**: Online dashboard

## Submission Preparation

To prepare for competition submission:

1. Extract the LoRA adapter from the best checkpoint
2. Ensure `adapter_config.json` is included
3. Package into `submission.zip`

The checkpoint format is compatible with the competition requirements.

## Notes

- The model uses Nemotron-3-Nano-30B base model
- LoRA rank is set to maximum allowed (32)
- Generation parameters match competition requirements
- Training uses vLLM backend for efficient generation
- Supports multi-GPU training with tensor parallelism

## Troubleshooting

### Out of Memory
- Reduce `train_micro_batch_size` in config
- Reduce `num_generations_per_prompt`
- Enable CPU offload if needed

### Slow Training
- Increase `generation_batch_size`
- Enable sequence packing (already enabled)
- Use colocated generation (already enabled)

### Poor Accuracy
- Increase training steps
- Adjust learning rate
- Try different LoRA rank (keep <= 32)
- Increase number of generations per prompt
