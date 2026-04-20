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
GRPO Training Script for Nemotron Reasoning Challenge
"""

import argparse
import os
import pprint
import sys

# Add parent directory to path to import nemo_rl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

# Import custom modules
from nemo_reasoning.dataset import (
    NemotronReasoningDataset,
    nemo_reasoning_hf_data_processor,
)
from nemo_reasoning.reward import nemo_reasoning_reward_fn


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GRPO Training for Nemotron Reasoning Challenge")
    parser.add_argument(
        "--config",
        type=str,
        default="nemo_reasoning/grpo_nemotron.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/nemo_reasoning_rl",
        help="Output directory for checkpoints and logs",
    )
    
    return parser.parse_known_args()


def main():
    """Main training function"""
    # Parse arguments
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    # Load config
    config_path = args.config if os.path.exists(args.config) else None
    if not config_path:
        config_path = os.path.join(
            os.path.dirname(__file__), "grpo_nemotron.yaml"
        )
    
    config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")

    # Apply overrides
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    
    # Override data path
    config["data"]["train"]["data_path"] = args.data_path
    config["checkpointing"]["checkpoint_dir"] = args.output_dir
    config["logger"]["log_dir"] = args.output_dir

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get experiment directory
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    # Initialize Ray
    init_ray()

    # Setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, "Generation config required for GRPO"
    has_refit_draft_weights = bool(config["policy"]["draft"]["enabled"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"],
        tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
    )

    # Register custom dataset and processor
    # Note: This would require modifying nemo_rl/data/datasets/__init__.py
    # For now, we'll use the dataset directly in setup_response_data
    
    # Setup data
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_response_data(tokenizer, config["data"], config["env"])

    # Setup training
    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    # Run GRPO training
    print("🚀 Starting GRPO training for Nemotron Reasoning Challenge")
    grpo_train(
        policy=policy,
        policy_generation=policy_generation,
        cluster=cluster,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        logger=logger,
        checkpointer=checkpointer,
        grpo_state=grpo_state,
        master_config=master_config,
    )

    print("✅ Training completed!")
    print(f"📁 Results saved to: {config['logger']['log_dir']}")


if __name__ == "__main__":
    main()
