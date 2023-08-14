#!/usr/bin/env bash

CONFIG=$1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=5050

export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=1,4,5,7
 # need to check nvidia-smi first
# export CUDA_LAUNCH_BLOCKING=1

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=${MASTER_PORT} \
    basicsr/train.py \
    -opt $CONFIG \
    --launcher pytorch