#!/bin/bash
# Process MedCalc dataset for AMiD training with Qwen3-4B

# Use Python from amid environment
PYTHON_BIN="/mnt/local2/wxy/envs/amid/bin/python"

BASE_PATH="/mnt/local2/wxy/AMiD"
INPUT_FILE="/mnt/local2/wxy/LlamaFactory/data/medcalc_train.json"
OUTPUT_DIR="${BASE_PATH}/processed_data/medcalc/qwen3"
MODEL_PATH="/mnt/local2/wxy/models/Qwen3-4B"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Process training data
echo "Processing MedCalc training data..."
echo "Using Python: ${PYTHON_BIN}"
${PYTHON_BIN} ${BASE_PATH}/data_utils/process_medcalc.py \
    --input-file ${INPUT_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --model-path ${MODEL_PATH} \
    --split train \
    --max-length 2048 \
    --max-prompt-length 1536

echo ""
echo "Data processing complete!"
echo "Processed data saved to: ${OUTPUT_DIR}"
echo ""
echo "Files created:"
ls -lh ${OUTPUT_DIR}/train_0.*
