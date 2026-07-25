#!/bin/bash
# Verify that all required files and models are in place for training
# Uses centralized configuration from config.sh

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "========================================="
echo "AMiD Training Setup Verification"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SUCCESS=0
FAIL=0

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1"
        ((SUCCESS++))
        return 0
    else
        echo -e "${RED}✗${NC} Missing: $1"
        ((FAIL++))
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1"
        ((SUCCESS++))
        return 0
    else
        echo -e "${RED}✗${NC} Missing: $1"
        ((FAIL++))
        return 1
    fi
}

echo "Configuration loaded from: ${SCRIPT_DIR}/config.sh"
echo "Base directory: ${BASE_DIR}"
echo ""

# Check models
echo "1. Checking Models..."
check_dir "${STUDENT_MODEL}"
check_dir "${TEACHER_MODEL}"
echo ""

# Check raw data
echo "2. Checking Raw Data..."
check_file "${RAW_DATA}"
echo ""

# Check processed data
echo "3. Checking Processed Data..."
check_file "${PROCESSED_DATA_DIR}/train_0.bin"
check_file "${PROCESSED_DATA_DIR}/train_0.idx"
check_file "${PROCESSED_DATA_DIR}/train.jsonl"
check_file "${PROCESSED_DATA_DIR}/valid_0.bin"
check_file "${PROCESSED_DATA_DIR}/valid_0.idx"
check_file "${PROCESSED_DATA_DIR}/valid.jsonl"

if [ -f "${PROCESSED_DATA_DIR}/train_0.bin" ]; then
    SIZE=$(du -h "${PROCESSED_DATA_DIR}/train_0.bin" | cut -f1)
    echo "  Train data size: ${SIZE}"
fi
if [ -f "${PROCESSED_DATA_DIR}/valid_0.bin" ]; then
    SIZE=$(du -h "${PROCESSED_DATA_DIR}/valid_0.bin" | cut -f1)
    echo "  Valid data size: ${SIZE}"
fi
echo ""

# Check training scripts
echo "4. Checking Training Scripts..."
check_file "${SCRIPT_DIR}/config.sh"
check_file "${SCRIPT_DIR}/process_medcalc_train_valid.sh"
check_file "${SCRIPT_DIR}/amid/train_qwen3_4B_emixture.sh"
check_file "${SCRIPT_DIR}/amid/train_qwen3_4B_medcalc.sh"
check_file "${BASE_DIR}/data_utils/process_medcalc.py"
check_file "${BASE_DIR}/finetune.py"
echo ""

# Check DeepSpeed config
echo "5. Checking DeepSpeed Config..."
check_file "${BASE_DIR}/configs/deepspeed/ds_config_zero2_bf16.json"
echo ""

# Summary
echo "========================================="
echo "Verification Summary"
echo "========================================="
echo -e "${GREEN}Passed: ${SUCCESS}${NC}"
if [ $FAIL -gt 0 ]; then
    echo -e "${RED}Failed: ${FAIL}${NC}"
    echo ""
    echo "Please fix the missing items before starting training."
    echo ""
    echo "If processed data is missing, run:"
    echo "  cd ${BASE_DIR}"
    echo "  bash scripts/qwen3/process_medcalc_train_valid.sh"
    exit 1
else
    echo -e "${RED}Failed: ${FAIL}${NC}"
    echo ""
    echo -e "${GREEN}✓ All checks passed! Ready to train.${NC}"
    echo ""
    echo "To start training with e-mixture:"
    echo "  cd ${BASE_DIR}"
    echo "  bash scripts/qwen3/amid/train_qwen3_4B_emixture.sh . 2012 2 ab pr 0.5 0.5 4 0.00005"
    exit 0
fi
