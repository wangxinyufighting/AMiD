#!/usr/bin/env python3
"""
Data processing script for MedCalc JSON dataset to AMiD format.
Converts instruction-input-output format to tokenized binary format using MMapIndexedDatasetBuilder.
"""

import json
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import torch

# Add parent directory to path to import data_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.indexed_dataset import MMapIndexedDatasetBuilder


def process_medcalc_data(args):
    """Process MedCalc JSON data into tokenized format using MMapIndexedDatasetBuilder."""

    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Load JSON data
    print(f"Loading data from {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Output file paths (with _0 suffix for state 0)
    output_prefix = os.path.join(args.output_dir, f"{args.split}_0")
    bin_file = output_prefix + ".bin"
    idx_file = output_prefix + ".idx"

    print(f"Creating dataset builder for {output_prefix}")
    # Create builder with int32 dtype (standard for token ids)
    builder = MMapIndexedDatasetBuilder(bin_file, dtype=np.int32)

    # Process and save data
    total_tokens = 0
    jsonl_data = []

    for idx, example in enumerate(tqdm(data, desc="Processing examples")):
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output_text = example.get('output', '')

        # Combine instruction and input as prompt
        if input_text:
            prompt = f"{instruction}\n{input_text}"
        else:
            prompt = instruction

        # Tokenize prompt and output
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        output_tokens = tokenizer.encode(output_text, add_special_tokens=False)

        # Add EOS token at the end
        output_tokens.append(tokenizer.eos_token_id)

        # Truncate if necessary
        if len(prompt_tokens) > args.max_prompt_length:
            prompt_tokens = prompt_tokens[:args.max_prompt_length]

        max_output_length = args.max_length - len(prompt_tokens)
        if len(output_tokens) > max_output_length:
            output_tokens = output_tokens[:max_output_length]

        # Combine with separator (65535 as in original code)
        combined_tokens = prompt_tokens + [65535] + output_tokens

        # Add to builder
        token_tensor = torch.tensor(combined_tokens, dtype=torch.int32)
        builder.add_item(token_tensor)

        # Each example is a separate document
        builder.end_document()

        total_tokens += len(combined_tokens)

        # Save JSONL entry for reference
        jsonl_data.append({
            'instruction': instruction,
            'input': input_text,
            'output': output_text,
            'prompt_length': len(prompt_tokens),
            'output_length': len(output_tokens),
            'total_length': len(combined_tokens)
        })

    # Finalize the dataset
    print(f"Finalizing dataset to {idx_file}")
    builder.finalize(idx_file)

    # Save JSONL reference
    jsonl_file = os.path.join(args.output_dir, f"{args.split}.jsonl")
    print(f"Saving JSONL reference to {jsonl_file}")
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Print statistics
    print(f"\nProcessing complete!")
    print(f"Total examples: {len(jsonl_data)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average prompt length: {np.mean([item['prompt_length'] for item in jsonl_data]):.1f}")
    print(f"Average output length: {np.mean([item['output_length'] for item in jsonl_data]):.1f}")
    print(f"Average total length: {np.mean([item['total_length'] for item in jsonl_data]):.1f}")
    print(f"\nFiles created:")
    print(f"  - {bin_file}")
    print(f"  - {idx_file}")
    print(f"  - {jsonl_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process MedCalc data for AMiD training')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Input JSON file path')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model for tokenizer')
    parser.add_argument('--split', type=str, default='train',
                        help='Split name (train/dev/test)')
    parser.add_argument('--max-length', type=int, default=4096,
                        help='Maximum sequence length')
    parser.add_argument('--max-prompt-length', type=int, default=1536,
                        help='Maximum prompt length')

    args = parser.parse_args()
    process_medcalc_data(args)
