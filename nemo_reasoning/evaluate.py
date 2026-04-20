#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluation Script for Nemotron Reasoning Challenge
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nemo_reasoning.reward import NemotronReasoningReward


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Nemotron Reasoning Model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint (LoRA adapter or base model)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="nvidia/Nemotron-3-Nano-30B-A3B-FP8",
        help="Base model name/path (for LoRA)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to evaluation data CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=7680,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Generation top_p",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to load LoRA adapter",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(model_path, base_model, use_lora, device):
    """Load model and tokenizer"""
    print(f"Loading model from {model_path}")
    
    if use_lora:
        from peft import PeftModel
        print(f"Loading base model: {base_model}")
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        print(f"Loading LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(base_model_obj, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.eval()
    print(f"Model loaded on {device}")
    
    return model, tokenizer


def load_data(data_path):
    """Load evaluation data"""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    return df


def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    device: str,
) -> List[str]:
    """Generate responses for prompts"""
    responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract only the generated part
        for prompt, response in zip(batch_prompts, batch_responses):
            generated = response[len(prompt):].strip()
            responses.append(generated)
        
        print(f"Generated {i+len(batch_prompts)}/{len(prompts)} responses")
    
    return responses


def evaluate(
    predictions: List[str],
    ground_truths: List[str],
    reward_fn: NemotronReasoningReward,
) -> Dict:
    """Evaluate predictions against ground truths"""
    rewards = reward_fn.batch_compute_rewards(predictions, ground_truths)
    accuracy = reward_fn.compute_accuracy(rewards)
    
    # Per-category accuracy (if labels available)
    results = {
        "total_samples": len(predictions),
        "correct_samples": sum(rewards),
        "accuracy": accuracy,
        "rewards": rewards,
    }
    
    return results


def save_results(results: Dict, output_dir: str, data_df: pd.DataFrame, predictions: List[str]):
    """Save evaluation results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "total_samples": results["total_samples"],
            "correct_samples": results["correct_samples"],
            "accuracy": results["accuracy"],
        }, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Save detailed results
    results_df = data_df.copy()
    results_df["prediction"] = predictions
    results_df["reward"] = results["rewards"]
    results_df["correct"] = results["rewards"]
    
    results_path = output_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved detailed results to {results_path}")


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.base_model,
        args.use_lora,
        args.device,
    )
    
    # Load data
    df = load_data(args.data_path)
    
    # Prepare prompts
    prompts = []
    ground_truths = []
    for _, row in df.iterrows():
        prompt_template = f"{row['prompt']}\n\nThink step by step to solve this problem. Show your reasoning process clearly.\n\nPut your final answer inside \\boxed{{}}."
        prompts.append(prompt_template)
        ground_truths.append(str(row["answer"]))
    
    # Generate responses
    print("Generating responses...")
    predictions = generate_responses(
        model,
        tokenizer,
        prompts,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.batch_size,
        args.device,
    )
    
    # Evaluate
    print("Evaluating predictions...")
    reward_fn = NemotronReasoningReward()
    results = evaluate(predictions, ground_truths, reward_fn)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {results['total_samples']}")
    print(f"Correct samples: {results['correct_samples']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("="*50)
    
    # Save results
    save_results(results, args.output_dir, df, predictions)
    
    print(f"\n✅ Evaluation completed!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
