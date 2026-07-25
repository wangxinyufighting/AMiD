#!/usr/bin/env python3
"""
Split MedCalc dataset into train and validation sets.
Reads the original JSON, splits it 90/10, and saves separate files.
"""

import json
import argparse
import random
from pathlib import Path


def split_medcalc_data(args):
    """Split MedCalc data into train and validation sets."""

    print(f"Loading data from {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total examples: {len(data)}")

    # Shuffle with seed for reproducibility
    random.seed(args.seed)
    random.shuffle(data)

    # Calculate split point
    total = len(data)
    train_size = int(total * args.train_ratio)

    train_data = data[:train_size]
    valid_data = data[train_size:]

    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(valid_data)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    train_file = output_dir / "medcalc_train_split.json"
    valid_file = output_dir / "medcalc_valid_split.json"

    print(f"\nSaving train split to {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    print(f"Saving validation split to {valid_file}")
    with open(valid_file, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=2)

    print("\nSplit complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split MedCalc data into train/valid')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Input JSON file path')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for split files')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Ratio of training data (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling')

    args = parser.parse_args()
    split_medcalc_data(args)
