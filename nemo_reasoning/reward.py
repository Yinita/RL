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

"""Reward function for Nemotron Reasoning Challenge"""

import re
from typing import Dict, List, Union

import numpy as np


class NemotronReasoningReward:
    """
    Reward function for Nemotron Reasoning Challenge
    
    Evaluates correctness of generated answers based on:
    1. Exact string match
    2. Numerical tolerance (10^-2 relative tolerance)
    """

    def __init__(self, relative_tolerance: float = 1e-2):
        self.relative_tolerance = relative_tolerance

    def extract_answer(self, text: str) -> str:
        """
        Extract answer from generated text
        
        Priority:
        1. Content inside \\boxed{}
        2. Last numeric value found
        3. Last line of text
        """
        # Try to extract from \boxed{}
        match = re.search(r"\\boxed\{([^}]*)\}", text)
        if match:
            return match.group(1).strip()
        
        # Fallback: find last numeric value
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if numbers:
            return numbers[-1]
        
        # Fallback: return last line
        lines = text.strip().split("\n")
        if lines:
            return lines[-1].strip()
        
        return ""

    def is_numeric(self, s: str) -> bool:
        """Check if string represents a number"""
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    def compare_numeric(self, pred: str, gt: str) -> bool:
        """
        Compare two numeric strings with relative tolerance
        
        Returns True if within relative tolerance
        """
        try:
            pred_val = float(pred)
            gt_val = float(gt)
            
            if gt_val == 0:
                # If ground truth is zero, check absolute difference
                return abs(pred_val - gt_val) < 1e-2
            else:
                # Relative tolerance
                relative_diff = abs(pred_val - gt_val) / abs(gt_val)
                return relative_diff < self.relative_tolerance
        except (ValueError, TypeError):
            return False

    def compare_string(self, pred: str, gt: str) -> bool:
        """
        Compare two strings (exact match after stripping)
        """
        return pred.strip() == gt.strip()

    def compute_reward(self, prediction: str, ground_truth: str) -> float:
        """
        Compute reward for a prediction
        
        Returns 1.0 if correct, 0.0 otherwise
        """
        pred_answer = self.extract_answer(prediction)
        gt_answer = ground_truth.strip()
        
        # Try exact string match first
        if self.compare_string(pred_answer, gt_answer):
            return 1.0
        
        # Try numeric comparison if both are numeric
        if self.is_numeric(pred_answer) and self.is_numeric(gt_answer):
            if self.compare_numeric(pred_answer, gt_answer):
                return 1.0
        
        # Try case-insensitive string comparison
        if pred_answer.lower() == gt_answer.lower():
            return 1.0
        
        return 0.0

    def batch_compute_rewards(
        self, predictions: List[str], ground_truths: List[str]
    ) -> List[float]:
        """
        Compute rewards for a batch of predictions
        
        Args:
            predictions: List of generated predictions
            ground_truths: List of ground truth answers
        
        Returns:
            List of reward values (0.0 or 1.0)
        """
        rewards = []
        for pred, gt in zip(predictions, ground_truths):
            reward = self.compute_reward(pred, gt)
            rewards.append(reward)
        return rewards

    def compute_accuracy(self, rewards: List[float]) -> float:
        """
        Compute accuracy from rewards
        
        Args:
            rewards: List of reward values
        
        Returns:
            Accuracy (proportion of correct answers)
        """
        if not rewards:
            return 0.0
        return np.mean(rewards)


def nemo_reasoning_reward_fn(
    responses: List[str],
    ground_truths: List[str],
    **kwargs,
) -> List[float]:
    """
    Reward function for GRPO training
    
    Args:
        responses: List of model responses
        ground_truths: List of ground truth answers
        **kwargs: Additional arguments
    
    Returns:
        List of reward values
    """
    reward_fn = NemotronReasoningReward()
    return reward_fn.batch_compute_rewards(responses, ground_truths)
