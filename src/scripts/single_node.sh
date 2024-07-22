#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 /path/to/yaml"
  exit 1
fi
YAML_PATH=$1
 
if [ ! -f "$YAML_PATH" ]; then
  echo "Error: YAML file '$YAML_PATH' not found!"
  exit 1
fi

NPROC_PER_NODE=7
NNODES=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29502

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train.py $YAML_PATH
