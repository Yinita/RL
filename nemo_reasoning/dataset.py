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

"""Dataset processor for Nemotron Reasoning Challenge"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset as HFDataset
from nemo_rl.data.datasets.raw_dataset import RawDataset


class NemotronReasoningDataset(RawDataset):
    """Custom dataset for Nemotron Reasoning Challenge"""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        split_validation_size: float = 0.1,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = Path(data_path)
        self.split = split
        self.split_validation_size = split_validation_size
        self.seed = seed
        self._load_data()

    def _load_data(self):
        """Load and split the Nemotron reasoning data"""
        # Load CSV data
        df = pd.read_csv(self.data_path)
        
        # Split into train/validation if needed
        if self.split_validation_size > 0 and self.split in ["train", "validation"]:
            from sklearn.model_selection import train_test_split
            
            train_df, val_df = train_test_split(
                df,
                test_size=self.split_validation_size,
                random_state=self.seed,
                stratify=df.get("label", None)  # Stratify by label if available
            )
            
            if self.split == "train":
                df = train_df
            else:
                df = val_df
        
        # Convert to list of dicts
        self.data = df.to_dict("records")
        
        print(f"Loaded {len(self.data)} samples for {self.split} split")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "input": sample["prompt"],
            "output": sample.get("answer", ""),
            "id": sample.get("id", ""),
            "label": sample.get("label", ""),
        }


def nemo_reasoning_hf_data_processor(
    dataset: HFDataset,
    prompt_file: str = None,
    system_prompt_file: str = None,
    **kwargs,
) -> HFDataset:
    """
    Process Nemotron reasoning dataset for GRPO training
    
    Args:
        dataset: HuggingFace dataset
        prompt_file: Path to prompt template file
        system_prompt_file: Path to system prompt file
        **kwargs: Additional arguments
    
    Returns:
        Processed dataset with 'input' field
    """
    # Load prompt template if provided
    if prompt_file:
        with open(prompt_file, "r") as f:
            prompt_template = f.read().strip()
    else:
        # Default prompt template
        prompt_template = "{input}"
    
    # Load system prompt if provided
    system_prompt = ""
    if system_prompt_file:
        with open(system_prompt_file, "r") as f:
            system_prompt = f.read().strip()
    
    def process_fn(example):
        input_text = example["input"]
        
        # Apply prompt template
        formatted_input = prompt_template.format(input=input_text)
        
        # Add system prompt if provided
        if system_prompt:
            formatted_input = f"{system_prompt}\n\n{formatted_input}"
        
        return {"input": formatted_input}
    
    return dataset.map(process_fn)


def extract_answer_from_boxed(text: str) -> str:
    """
    Extract answer from \\boxed{} LaTeX command
    
    Args:
        text: Generated text
    
    Returns:
        Extracted answer
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
