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
"""Nemo Reasoning Environment for GRPO training."""

import logging
from typing import Any, NotRequired, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


class NemoReasoningEnvConfig(TypedDict):
    """Configuration for NemoReasoningEnvironment."""
    num_workers: int
    reward_fn: str  # Reward function name
    token_compression: NotRequired[dict | None]


@ray.remote
class NemoReasoningWorker:
    """Worker for computing rewards for reasoning tasks."""
    
    def __init__(self, token_compression_config: dict | None = None):
        self.token_compression_config = token_compression_config or {}
        logging.getLogger("nemo_reasoning").setLevel(logging.INFO)
    
    def compute_rewards(
        self,
        responses: list[str],
        ground_truths: list[str],
        **kwargs,
    ) -> tuple[list[float], list[dict]]:
        """Compute rewards for reasoning responses."""
        rewards = []
        metrics = []
        
        for response, ground_truth in zip(responses, ground_truths):
            # Simple reward: 1.0 if correct, 0.0 otherwise
            # This should be enhanced with proper reasoning evaluation
            reward = 0.0
            
            # Check if response contains answer marker
            if "<answer>" in response and "</answer>" in response:
                # Extract answer and compare with ground truth
                answer_start = response.find("<answer>") + len("<answer>")
                answer_end = response.find("</answer>")
                extracted_answer = response[answer_start:answer_end].strip()
                
                # Simple string comparison (should be enhanced)
                if extracted_answer.lower() == str(ground_truth).lower():
                    reward = 1.0
            
            rewards.append(reward)
            metrics.append({"answer_correct": reward > 0})
        
        return rewards, metrics


@ray.remote
class NemoReasoningEnvironment(EnvironmentInterface):
    """Environment for training reasoning models with GRPO."""
    
    def __init__(self, env_config: NemoReasoningEnvConfig):
        self.env_config = env_config
        self.num_workers = env_config.get("num_workers", 8)
        self.token_compression_config = env_config.get("token_compression", {})
        
        # Initialize workers
        self.workers = [
            NemoReasoningWorker.remote(self.token_compression_config)
            for _ in range(self.num_workers)
        ]
    
    async def step(
        self,
        prompts: list[str],
        responses: list[str],
        **kwargs,
    ) -> EnvironmentReturn:
        """Compute rewards for given responses."""
        # Get ground truths from kwargs
        ground_truths = kwargs.get("ground_truths", [""] * len(responses))
        
        # Split work among workers
        chunk_size = max(1, len(responses) // self.num_workers)
        chunks = [
            responses[i:i + chunk_size]
            for i in range(0, len(responses), chunk_size)
        ]
        truth_chunks = [
            ground_truths[i:i + chunk_size]
            for i in range(0, len(ground_truths), chunk_size)
        ]
        
        # Dispatch to workers
        futures = []
        for worker, resp_chunk, truth_chunk in zip(
            self.workers[:len(chunks)], chunks, truth_chunks
        ):
            future = worker.compute_rewards.remote(
                resp_chunk, truth_chunk, **kwargs
            )
            futures.append(future)
        
        # Collect results
        results = await asyncio.gather(*futures)
        all_rewards = []
        all_metrics = []
        for rewards, metrics in results:
            all_rewards.extend(rewards)
            all_metrics.extend(metrics)
        
        return EnvironmentReturn(
            rewards=torch.tensor(all_rewards, dtype=torch.float32),
            successes=[m["answer_correct"] for m in all_metrics],
            metrics=all_metrics,
        )
    
    async def get_dataset_logs(
        self,
        prompts: list[str],
        responses: list[str],
        **kwargs,
    ) -> list[dict]:
        """Get dataset logs for the environment."""
        logs = []
        for prompt, response in zip(prompts, responses):
            logs.append({
                "prompt": prompt,
                "response": response,
                "log_type": LLMMessageLogType.RESPONSE,
            })
        return logs


import asyncio  # noqa: E402
