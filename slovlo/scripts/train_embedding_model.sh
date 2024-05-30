#!/bin/bash

SCRIPT=$(readlink -f "$0")
PROJECT_ROOT=$(dirname $(dirname $(dirname "$SCRIPT")))

export PYTHONPATH="$PROJECT_ROOT"
export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TOKENIZERS_PARALLELISM="false"

torchrun --nproc_per_node=1 --nnodes=1 \
  $PROJECT_ROOT/slovlo/embedding_model/train.py \
    --base_model=intfloat/multilingual-e5-base \
    --train_dataset_path="$1/train.jsonl" \
    --test_dataset_path="$1/test.jsonl" \
    --output_model_path="$2" \
    --epochs=1 \
    --train_batch_size=16384 \
    --train_sub_batch_size=64 \
    --eval_batch_size=32 \
    --warmup_ratio=0.05 \
    --learning_rate=1e-5 \
    --weight_decay=0.01 \
    --lr_schedule=linear \
    --optimizer=adamw \
    --log_path="$3" \
    --log_steps=10
