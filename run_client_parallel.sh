# -*- coding: utf-8 -*-
# +
# #!/bin/bash

#傳入輪次及client數
CLIENT=$1
ROUND=$2

# 參數定義
EPOCHS=50
BATCH=16
WORKERS=4

 


#權重參數設定
if [ "$ROUND" -eq 0 ]; then
  WEIGHTS_ARG="--weights ''"
else
  WEIGHTS_ARG="--weights global_round_weights/global_round_${ROUND}.pt"
fi

# echo "[Client $CLIENT] epochs=$EPOCHS batch=$BATCH workers=$WORKERS"

# 執行client訓練
TRAIN_CMD="python train_dual.py \
               --data data/kitti_client${CLIENT}.yaml \
               ${WEIGHTS_ARG} \
               --epochs ${EPOCHS} \
               --batch ${BATCH} \
               --workers ${WORKERS} \
               --device 0 \
               --hyp hyp.scratch-high.yaml \
               --img 640 \
               --cfg models/detect/yolov9-c.yaml \
               --project fed_client_weights \
               --name client${CLIENT}_r${ROUND}"
               
# echo "Executing Training Command:"
# echo "$TRAIN_CMD"
# echo "==================="
$TRAIN_CMD

