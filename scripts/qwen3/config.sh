#!/bin/bash
# Centralized configuration for all Qwen3 training scripts
# Edit paths here to configure your environment

# ==========================================
# Directory Paths
# ==========================================
BASE_DIR="/czsun/zhi/xywang/al_baselines/AMiD"

# ==========================================
# Model Paths
# ==========================================
# Base student model (Qwen3-4B-Instruct)
STUDENT_MODEL="/czsun/models/Qwen3-4B-Instruct-2507"
STUDENT_MODEL_NAME="qwen3-4b-instruct"

# Teacher model (fine-tuned on MedCalc)
TEACHER_MODEL="/czsun/zhi/xywang/LlamaFactory/saves/Qwen3-4B-Instruct-2507_medcalc_train_1e-5"
TEACHER_MODEL_NAME="qwen3-4b-instruct-medcalc"

# ==========================================
# Data Paths
# ==========================================
# Raw data
RAW_DATA="/czsun/zhi/xywang/LlamaFactory/data/medcalc_train.json"

# Processed data directories
SPLIT_DATA_DIR="${BASE_DIR}/data/medcalc"
PROCESSED_DATA_DIR="${BASE_DIR}/processed_data/medcalc/qwen3"

# ==========================================
# Output Paths
# ==========================================
RESULTS_DIR="${BASE_DIR}/results/qwen3"

# ==========================================
# Python Environment
# ==========================================
# Adjust this to your conda environment or python path
PYTHON_BIN="python"

# ==========================================
# Model Configuration
# ==========================================
MODEL_TYPE="qwen2"  # Qwen3 uses Qwen2 architecture

# ==========================================
# Data Processing Configuration
# ==========================================
MAX_SEQ_LENGTH=2048
MAX_PROMPT_LENGTH=1536
TRAIN_RATIO=0.9
RANDOM_SEED=42

# ==========================================
# Export all variables
# ==========================================
export BASE_DIR
export STUDENT_MODEL
export STUDENT_MODEL_NAME
export TEACHER_MODEL
export TEACHER_MODEL_NAME
export RAW_DATA
export SPLIT_DATA_DIR
export PROCESSED_DATA_DIR
export RESULTS_DIR
export PYTHON_BIN
export MODEL_TYPE
export MAX_SEQ_LENGTH
export MAX_PROMPT_LENGTH
export TRAIN_RATIO
export RANDOM_SEED
