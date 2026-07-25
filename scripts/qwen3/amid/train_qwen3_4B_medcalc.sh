#!/bin/bash
# Training script for Qwen3-4B with AMiD
# Uses centralized configuration from config.sh

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-4}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# Model paths from config
BASE_PATH=${1-"${BASE_DIR}"}
CKPT_NAME="${STUDENT_MODEL_NAME}"
CKPT="${STUDENT_MODEL}"
TEACHER_CKPT_NAME="${TEACHER_MODEL_NAME}"
TEACHER_CKPT="${TEACHER_MODEL}"

# Data paths from config
DATA_DIR="${PROCESSED_DATA_DIR}"
LM_DATA_DIR="${PROCESSED_DATA_DIR}"

# Hyperparameters
BATCH_SIZE=${8-4}
LR=${9-0.00005}
GRAD_ACC=2
EVAL_BATCH_SIZE=8

# Length from config
MAX_LENGTH=${MAX_SEQ_LENGTH}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}

# Seed
SEED=10

# AMiD hyperparameters
AMID_DIV_NAME=${4-"ab"}
AMID_DIV_ORDER=${5-"pr"}
AMID_ALPHA=${6-0.5}
AMID_LAM=${7-0.5}

SAVE_PATH="${RESULTS_DIR}/train/4B_4B#amid/${AMID_DIV_NAME}_${AMID_DIV_ORDER}_${AMID_ALPHA}_${AMID_LAM}_${BATCH_SIZE}_${LR}"

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type ${MODEL_TYPE}"

# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 500"

# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 100"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 5"

# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"

# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"

# seed
OPTS+=" --seed ${SEED}"

# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2_bf16.json"

# type
OPTS+=" --type adaptive-amid"

# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

# distillm
OPTS+=" --student-gen"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"

# amid
OPTS+=" --amid-div-name ${AMID_DIV_NAME}"
OPTS+=" --amid-div-order ${AMID_DIV_ORDER}"
OPTS+=" --amid-alpha ${AMID_ALPHA}"
OPTS+=" --amid-lam ${AMID_LAM}"
OPTS+=" --kd-ratio 1.0"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo "========================================="
echo "AMiD Training"
echo "========================================="
echo "Configuration:"
echo "  Base directory: ${BASE_PATH}"
echo "  Student model: ${CKPT}"
echo "  Teacher model: ${TEACHER_CKPT}"
echo "  Data directory: ${DATA_DIR}"
echo "  Divergence: ${AMID_DIV_NAME}"
echo "  Order: ${AMID_DIV_ORDER}"
echo "  Alpha: ${AMID_ALPHA}"
echo "  Lambda: ${AMID_LAM}"
echo "  GPUs: ${GPUS_PER_NODE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LR}"
echo "  Save path: ${SAVE_PATH}"
echo "========================================="
echo ""
echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"

mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
