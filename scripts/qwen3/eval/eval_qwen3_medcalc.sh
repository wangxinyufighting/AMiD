#!/bin/bash
# Evaluation script for Qwen3-4B AMiD trained model on MedCalc

MASTER_PORT=${2-29500}
CKPT=${1-"./"}
BASE_PATH="/mnt/local2/wxy/AMiD"

MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# Model paths
CKPT_NAME="qwen3-4b-amid"
DATA_DIR="${BASE_PATH}/processed_data/medcalc/qwen3/"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --model-type qwen2"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --eval-batch-size 8"
OPTS+=" --max-length 2048"
OPTS+=" --max-prompt-length 1536"
OPTS+=" --do-eval"
OPTS+=" --eval-gen"
OPTS+=" --num-workers 4"

# Generation parameters
OPTS+=" --do-sample"
OPTS+=" --top-k 50"
OPTS+=" --top-p 0.95"
OPTS+=" --temperature 0.7"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/evaluate_main.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
CODE_BASE=HF ${CMD}
