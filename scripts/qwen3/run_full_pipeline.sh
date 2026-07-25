#!/bin/bash
# Complete pipeline: Process data and start training

set -e  # Exit on error

BASE_PATH="/mnt/local2/wxy/AMiD"
cd ${BASE_PATH}

echo "========================================="
echo "AMiD Training Pipeline for Qwen3-4B"
echo "========================================="
echo ""

# Step 1: Test data format
echo "Step 1: Testing MedCalc data format..."
python3 scripts/qwen3/test_data_format.py
if [ $? -ne 0 ]; then
    echo "Error: Data format test failed!"
    exit 1
fi
echo ""

# Step 2: Process data
echo "Step 2: Processing MedCalc dataset..."
bash scripts/qwen3/process_medcalc_data.sh
if [ $? -ne 0 ]; then
    echo "Error: Data processing failed!"
    exit 1
fi
echo ""

# Step 3: Verify processed data
echo "Step 3: Verifying processed data..."
PROCESSED_DIR="${BASE_PATH}/processed_data/medcalc/qwen3"
if [ ! -f "${PROCESSED_DIR}/train_0.bin" ] || [ ! -f "${PROCESSED_DIR}/train_0.idx" ]; then
    echo "Error: Processed data files not found!"
    echo "Expected: ${PROCESSED_DIR}/train_0.bin and train_0.idx"
    exit 1
fi
echo "✓ Found train_0.bin and train_0.idx"
echo ""

# Step 4: Start training
echo "Step 4: Starting AMiD training..."
echo "Configuration:"
echo "  - Base model: /mnt/local2/wxy/models/Qwen3-4B"
echo "  - Teacher model: /mnt/local2/wxy/models/Qwen3-4B"
echo "  - Dataset: MedCalc (10,053 examples)"
echo "  - GPUs: 4"
echo "  - Divergence: α-β (P||R)"
echo "  - Alpha: 0.5, Lambda: 0.5"
echo "  - Batch size: 4, LR: 5e-5"
echo ""
echo "Training will start in 5 seconds... (Ctrl+C to cancel)"
sleep 5

bash scripts/qwen3/amid/train_qwen3_4B_medcalc.sh \
    ${BASE_PATH} \
    2012 \
    4 \
    ab \
    pr \
    0.5 \
    0.5 \
    4 \
    0.00005

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
