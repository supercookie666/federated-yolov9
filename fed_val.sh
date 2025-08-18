#!/bin/bash
# -*- coding: utf-8 -*-
# +
### 參數設定區 ###
## 工作目錄
WORKDIR=/home/your_account/yolov9 ## 更換為你的yolov9 目錄
cd $WORKDIR


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

### 環境檢查區 ### 
## Debug: 確認 Python 路徑與版本
echo "Python Path and Version:"
echo "==================="
which python
python --version
echo "PYTHONPATH: $PYTHONPATH"
echo "==================="

echo "Activated Conda Environment:"
echo "==================="
python -c "import sys; print('\n'.join(sys.path))"
wandb login 
python -c 'import wandb'
python -c 'import torch; print(torch.__version__)'
echo "==================="
echo "env.py"
python env.py
echo "==================="

## 訓練 train_dual.py 命令 (動態設置 nproc_per_node 和 nnodes)
TRAIN_CMD="python val_dual_2.py \
             --weights fed_final_weights/R5E30_parallel.pt \
             --data data/kitti_val.yaml \
             --img 640 \
             --batch 8 \
             --iou 0.65 \
             --workers 0 \
             --conf-thres 0.001 \
             --verbose \
             --project fed_val_client \
             --name R5E30_parallel \
             --save-txt \
             --device 0"
             
## 印出完整的訓練命令
echo "Executing Training Command:"
echo "$TRAIN_CMD"
echo "==================="
$TRAIN_CMD

## 檢查執行結果
if [ $? -ne 0 ]; then
  echo "Error: TRAIN_CMD execution failed on node $(hostname)" >&2
  exit 1
fi
