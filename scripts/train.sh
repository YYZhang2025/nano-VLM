#!/bin/bash


NPROC_PER_NODE=2


torchrun \
--standalone \
--nproc_per_node=${NPROC_PER_NODE} \
train.py \
--log_wandb \
--compile