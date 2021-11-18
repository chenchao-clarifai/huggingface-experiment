#!/bin/bash

set -e

NGPU=$1
CONFIG=$2

echo number of GPUs: $NGPU
echo config path: $CONFIG


torchrun -m \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NGPU \
    hfe.sequence_classification.trainer \
        --yaml_config_path $CONFIG