#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# +
#!/bin/bash

#傳入輪次及client數
CLIENT=$1
ROUND=$2

## SLURM 環境
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-1}
NNODES=${SLURM_NNODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
if [ -z "$MASTER_ADDR" ]; then
    echo "oh! why MASTER_ADDR not found!"
    MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
fi

#NGPU=$SLURM_GPUS_ON_NODE #這個值常抓不到
#NGPU=$NPROC_PER_NODE # NPROC_PER_NODE是gpu數但在這邊也抓錯
if [ -z "$NGPU" ]; then
    echo "oh! why NPROC_PER_NODE not found!"
    NGPU=$(nvidia-smi -L | wc -l)  # 等於 $SLURM_GPUS_ON_NODE
fi

MASTER_PORT=9527
DEVICE_LIST=$(seq -s, 0 $(($NGPU-1)) | paste -sd, -) # 0,1,...n-1
NNODES=${SLURM_NNODES:-1}               # 節點總數，默認為 1
NODE_RANK=${SLURM_NODEID}            # 當前節點的 rank，默認為 0

echo "Debug Information:"
echo "==================="
echo "SLURM_NODEID: $NODE_RANK"
echo "SLURM_NNODES: $NNODES"
echo "SLURM_GPUS_ON_NODE: $NGPU"
echo "Device: $DEVICE_LIST"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Current Hostname: $(hostname)"
echo "==================="


# 參數定義
EPOCHS=50
BATCH=32
WORKERS=8

cd /home/chander92811/yolov9


#權重參數設定
if [ "$ROUND" -eq 0 ]; then
  WEIGHTS_ARG="--weights ''"
else
  WEIGHTS_ARG="--weights global_round_weights/global_round_${ROUND}.pt"
fi

echo "[Client $CLIENT] epochs=$EPOCHS batch=$BATCH workers=$WORKERS"

# 執行client訓練
TRAIN_CMD="torchrun --nproc_per_node=$NGPU \
             --nnodes=$NNODES \
             --node_rank=$NODE_RANK \
             --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT \
             train_dual.py \
               --data data/kitti_client${CLIENT}.yaml \
               ${WEIGHTS_ARG} \
               --epochs ${EPOCHS} \
               --batch ${BATCH} \
               --workers ${WORKERS} \
               --device $DEVICE_LIST \
               --hyp hyp.scratch-high.yaml \
               --img 640 \
               --cfg models/detect/yolov9-c.yaml \
               --project fed_client_weights \
               --name client${CLIENT}_r${ROUND}"
               
echo "Executing Training Command:"
echo "$TRAIN_CMD"
echo "==================="
$TRAIN_CMD




