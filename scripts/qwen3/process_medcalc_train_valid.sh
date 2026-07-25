#!/bin/bash
# Process MedCalc data: split into train/valid, then process both
# Uses centralized configuration from config.sh

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "========================================="
echo "Configuration:"
echo "========================================="
echo "Base directory: ${BASE_DIR}"
echo "Raw data: ${RAW_DATA}"
echo "Student model: ${STUDENT_MODEL}"
echo "Output: ${PROCESSED_DATA_DIR}"
echo ""

echo "========================================="
echo "Step 1: Split MedCalc into train/valid"
echo "========================================="

# Split the data (90% train, 10% valid)
${PYTHON_BIN} ${SCRIPT_DIR}/split_medcalc_data.py \
    --input-file ${RAW_DATA} \
    --output-dir ${SPLIT_DATA_DIR} \
    --train-ratio ${TRAIN_RATIO} \
    --seed ${RANDOM_SEED}

echo ""
echo "========================================="
echo "Step 2: Process training data"
echo "========================================="

# Process training split
${PYTHON_BIN} ${BASE_DIR}/data_utils/process_medcalc.py \
    --input-file ${SPLIT_DATA_DIR}/medcalc_train_split.json \
    --output-dir ${PROCESSED_DATA_DIR} \
    --model-path ${STUDENT_MODEL} \
    --split train \
    --max-length ${MAX_SEQ_LENGTH} \
    --max-prompt-length ${MAX_PROMPT_LENGTH}

echo ""
echo "========================================="
echo "Step 3: Process validation data"
echo "========================================="

# Process validation split
${PYTHON_BIN} ${BASE_DIR}/data_utils/process_medcalc.py \
    --input-file ${SPLIT_DATA_DIR}/medcalc_valid_split.json \
    --output-dir ${PROCESSED_DATA_DIR} \
    --model-path ${STUDENT_MODEL} \
    --split valid \
    --max-length ${MAX_SEQ_LENGTH} \
    --max-prompt-length ${MAX_PROMPT_LENGTH}

echo ""
echo "========================================="
echo "Processing complete!"
echo "========================================="
echo "Files created:"
echo "  - ${PROCESSED_DATA_DIR}/train_0.bin"
echo "  - ${PROCESSED_DATA_DIR}/train_0.idx"
echo "  - ${PROCESSED_DATA_DIR}/valid_0.bin"
echo "  - ${PROCESSED_DATA_DIR}/valid_0.idx"
echo ""
echo "Ready to train!"
