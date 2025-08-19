!#/bin/bash


NPROC_PER_NODE=2


LOG_WANDB=FALSE
COMPILE=FALSE

torchrun \
--standalone \
--nproc_per_node=8 \
--master_port=12345 \
train.py \
--log_wandb=${LOG_WANDB} \
--compile=${COMPILE}