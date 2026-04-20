# Nemotron Reasoning Challenge - RL Training

This directory contains scripts for training and evaluating a Nemotron-3-Nano-30B model using GRPO (Group Relative Policy Optimization) for the NVIDIA Nemotron Reasoning Challenge.

## Project Structure

```
nemo_reasoning/
├── __init__.py              # Package initialization
├── dataset.py               # Custom dataset processor
├── reward.py                # Reward function for evaluation
├── grpo_nemotron.yaml       # GRPO configuration
├── train_grpo.py            # Training script
├── evaluate.py              # Evaluation script
├── submit_slurm.sh          # Slurm submission script
├── prompts/
│   └── default.txt          # Default prompt template
└── README.md                # This file
```

## Setup

### 1. Install Dependencies

```bash
cd RL
pip install -e ".[vllm,gpu]"
pip install pandas scikit-learn peft
```

### 2. Prepare Data

Place your training data at the path specified in the config:
```bash
# Update the data path in grpo_nemotron.yaml
data_path: "/path/to/final_Nemotron_training_data.csv"
```

### 3. Configure WandB

The script uses WandB project `nemo-rl-0420`. The login is already configured in the slurm script.

## Training

### Option 1: Slurm (Recommended for Cluster)

```bash
# Update the slurm script with your account and email
# Then submit:
sbatch nemo_reasoning/submit_slurm.sh
```

### Option 2: Local/Interactive

```bash
python nemo_reasoning/train_grpo.py \
    --config nemo_reasoning/grpo_nemotron.yaml \
    --data_path /path/to/final_Nemotron_training_data.csv \
    --output_dir results/nemo_reasoning_rl
```

## Configuration

Key parameters in `grpo_nemotron.yaml`:

- **LoRA rank**: 32 (maximum allowed by competition)
- **Batch size**: 1024 global, 8 micro
- **Generation**: 64 prompts/step, 16 generations/prompt
- **Max tokens**: 7680 (competition requirement)
- **Temperature**: 0.0 (competition requirement)
- **Top-p**: 1.0 (competition requirement)
- **GPU memory**: 0.85 utilization (competition requirement)

## Evaluation

```bash
python nemo_reasoning/evaluate.py \
    --model_path results/nemo_reasoning_rl/checkpoints/best_model \
    --base_model nvidia/Nemotron-3-Nano-30B-A3B-FP8 \
    --data_path /path/to/evaluation_data.csv \
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
