#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# args: ROUND
# 用法: ./run_aggregate.sh <ROUND>
# 例如: ./run_aggregate.sh 0  → 會讀 client0_r0/... 至 client3_r0/..., 輸出 global_round_1.pt

# +
#!/bin/bash

#Rround設定
ROUND=$1
if [ -z "$ROUND" ]; then
  echo "Usage: $0 <ROUND>"
  exit 1
fi

NEXT_ROUND=$((ROUND + 1))


# 準備四個 client 的權重路徑
cd /home/your_account/yolov9
INPUT_MODELS=(
  "fed_client_weights/client0_r${ROUND}/weights/best.pt"
  "fed_client_weights/client1_r${ROUND}/weights/best.pt"
  "fed_client_weights/client2_r${ROUND}/weights/best.pt"
  "fed_client_weights/client3_r${ROUND}/weights/best.pt"
)

echo "[Aggregator] Round ${ROUND} → aggregating:"
for m in "${INPUT_MODELS[@]}"; do
  echo "  - $m"
done
echo "[Aggregator] Output: global_round_weights/global_round_${NEXT}.pt"

# 執行 FedAvg 聚合
TRAIN_CMD="python fed_aggregate.py \
  -i fed_client_weights/client0_r${ROUND}/weights/best.pt \
     fed_client_weights/client1_r${ROUND}/weights/best.pt \
     fed_client_weights/client2_r${ROUND}/weights/best.pt \
     fed_client_weights/client3_r${ROUND}/weights/best.pt \
  -o global_round_weights/global_round_${NEXT_ROUND}.pt \
  --cfg models/detect/yolov9-c.yaml \
  -d data/kitti_client0.yaml \
     data/kitti_client1.yaml \
     data/kitti_client2.yaml \
     data/kitti_client3.yaml"

echo "Executing Training Command:"
echo "$TRAIN_CMD"
echo "==================="
$TRAIN_CMD


if [ $? -eq 0 ]; then
  echo "[Aggregator] Successfully created $OUTPUT_MODEL"
else
  echo "[Aggregator] Aggregation failed!"
  exit 2
fi

