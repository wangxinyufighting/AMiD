#!/usr/bin/env python3
"""
Quick test script to verify the data processing pipeline.
"""
import json
import sys

def test_medcalc_format():
    """Test if MedCalc JSON is in the expected format."""
    input_file = "/mnt/local2/wxy/LlamaFactory/data/medcalc_train.json"

    print(f"Testing MedCalc data format from: {input_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✓ Successfully loaded JSON with {len(data)} examples")

        # Check first example
        if len(data) > 0:
            example = data[0]
            required_keys = ['instruction', 'input', 'output']

            print(f"\nFirst example keys: {list(example.keys())}")

            for key in required_keys:
                if key in example:
                    value_preview = str(example[key])[:100]
                    print(f"✓ Found '{key}': {value_preview}...")
                else:
                    print(f"✗ Missing required key: '{key}'")
                    return False

            # Check data types
            if isinstance(example['instruction'], str) and \
               isinstance(example['input'], str) and \
               isinstance(example['output'], str):
                print("\n✓ All fields are strings (correct format)")
            else:
                print("\n✗ Some fields are not strings")
                return False

            # Estimate average lengths
            total_inst_len = sum(len(ex.get('instruction', '')) for ex in data[:100])
            total_input_len = sum(len(ex.get('input', '')) for ex in data[:100])
            total_output_len = sum(len(ex.get('output', '')) for ex in data[:100])

            print(f"\nAverage lengths (first 100 examples):")
            print(f"  Instruction: {total_inst_len / 100:.0f} chars")
            print(f"  Input: {total_input_len / 100:.0f} chars")
            print(f"  Output: {total_output_len / 100:.0f} chars")

            print("\n✓ Data format is compatible with AMiD processing!")
            return True

    except FileNotFoundError:
        print(f"✗ File not found: {input_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_medcalc_format()
    sys.exit(0 if success else 1)
