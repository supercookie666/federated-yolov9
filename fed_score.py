#!/usr/bin/env python3
# usage:
#   python tools/eval_map50_from_txt.py \
#     --pred /home/xxx/yolov9/runs/fed_val_central/fed_val_central/labels \
#     --gt   /home/xxx/yolov9/datasets/kitti/client/val/labels \
#     --nc 8 \
#     --names Car Van Truck Pedestrian Person_sitting Cyclist Tram Misc

import os, glob, argparse, math
from collections import defaultdict
import numpy as np

def yolo_xywh_to_xyxy(xywh):
    x,y,w,h = xywh
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return np.array([x1,y1,x2,y2], dtype=float)

def iou_xyxy(box, boxes):
    # box: (4,), boxes: (N,4); all in normalized [0,1]
    if boxes.size == 0: return np.zeros((0,))
    inter_x1 = np.maximum(box[0], boxes[:,0])
    inter_y1 = np.maximum(box[1], boxes[:,1])
    inter_x2 = np.minimum(box[2], boxes[:,2])
    inter_y2 = np.minimum(box[3], boxes[:,3])
    inter_w = np.clip(inter_x2 - inter_x1, 0.0, 1.0)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, 1.0)
    inter = inter_w * inter_h
    area1 = (box[2]-box[0]) * (box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    union = area1 + area2 - inter + 1e-16
    return inter / union

def voc_ap(rec, prec):
    # 计算 VOC-style AP（积分法，带精度包络）
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx-1]) * mpre[idx])
    return ap

def load_gt(gt_dir, nc):
    gt = defaultdict(lambda: defaultdict(list)) # img -> class -> [xyxy...]
    npos = np.zeros(nc, dtype=int)
    for p in sorted(glob.glob(os.path.join(gt_dir, '*.txt'))):
        img = os.path.splitext(os.path.basename(p))[0]
        arr = []
        with open(p) as f:
            for line in f:
                q = line.strip().split()
                if len(q) < 5: continue
                c = int(float(q[0]))
                if c < 0 or c >= nc: continue
                xywh = list(map(float, q[1:5]))
                xyxy = yolo_xywh_to_xyxy(xywh)
                gt[img][c].append(xyxy)
                npos[c] += 1
    # convert lists to arrays
    for img in gt:
        for c in list(gt[img].keys()):
            gt[img][c] = np.stack(gt[img][c], axis=0) if gt[img][c] else np.zeros((0,4))
    return gt, npos

def load_pred(pred_dir, nc):
    # 返回 list: (img, class, conf, xyxy)
    preds = []
    for p in sorted(glob.glob(os.path.join(pred_dir, '*.txt'))):
        img = os.path.splitext(os.path.basename(p))[0]
        with open(p) as f:
            for line in f:
                q = line.strip().split()
                if len(q) < 6:  # 需要有 conf
                    # 某些版本是 5 列（没有 conf），给个很小的 conf 也能评
                    q += ['0.001']
                c = int(float(q[0]))
                if c < 0 or c >= nc: continue
                xywh = list(map(float, q[1:5]))
                conf = float(q[5])
                xyxy = yolo_xywh_to_xyxy(xywh)
                preds.append((img, c, conf, xyxy))
    # 依 conf 降序
    preds.sort(key=lambda x: x[2], reverse=True)
    return preds

def eval_map50(preds, gt, npos, iou_thr=0.5, nc=8):
    aps = np.zeros(nc)
    per_cls = {}
    for c in range(nc):
        # 收集該類別的所有預測
        preds_c = [(img,conf,box) for (img,cc,conf,box) in preds if cc==c]
        if npos[c] == 0:
            per_cls[c] = dict(AP=np.nan, P=np.nan, R=np.nan)
            continue
        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))
        matched = defaultdict(lambda: np.array([]))  # img -> 已匹配的 GT 索引
        # 逐預測匹配
        for i,(img,conf,box) in enumerate(preds_c):
            gtc = gt.get(img, {}).get(c, np.zeros((0,4)))
            if gtc.shape[0] == 0:
                fp[i] = 1
                continue
            ious = iou_xyxy(box, gtc)
            j = int(np.argmax(ious)) if ious.size else -1
            iou = ious[j] if j>=0 else 0.0
            if iou >= iou_thr:
                # 防止一個 GT 被重複匹配
                if matched[img].size == 0:
                    matched[img] = -np.ones(gtc.shape[0], dtype=int)
                if matched[img][j] == -1:
                    tp[i] = 1
                    matched[img][j] = i
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        # PR & AP
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        rec = cum_tp / (npos[c] + 1e-16)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-16)
        ap = voc_ap(rec, prec)
        aps[c] = ap
        per_cls[c] = dict(AP=ap, P=prec[-1], R=rec[-1])
    mAP = np.nanmean(aps[np.logical_not(np.isnan(aps))]) if np.any(~np.isnan(aps)) else 0.0
    return mAP, aps, per_cls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='pred txt dir (YOLO, with conf)')
    ap.add_argument('--gt',   required=True, help='GT txt dir (YOLO)')
    ap.add_argument('--nc',   type=int, required=True, help='num classes')
    ap.add_argument('--names', nargs='*', default=None, help='class names (optional)')
    args = ap.parse_args()

    gt, npos = load_gt(args.gt, args.nc)
    preds = load_pred(args.pred, args.nc)
    mAP, aps, per_cls = eval_map50(preds, gt, npos, iou_thr=0.5, nc=args.nc)

    names = args.names if args.names and len(args.names)==args.nc else [f'cls{i}' for i in range(args.nc)]
    print('---------- Results (IoU=0.5) ----------')
    for i, apv in enumerate(aps):
        if math.isnan(apv): 
            print(f'{names[i]:<16} AP:  -   (no GT)')
        else:
            print(f'{names[i]:<16} AP: {apv*100:6.2f}%')
    print(f'===> mAP@0.5: {mAP*100:.2f}%')

    # 寫到結果檔
    save_dir = os.path.dirname(os.path.abspath(args.pred))
    out = os.path.join(save_dir, 'results.txt')
    with open(out, 'w') as f:
        f.write(f'mAP@0.5: {mAP*100:.4f}%\n')
        for i, apv in enumerate(aps):
            if math.isnan(apv):
                f.write(f'{names[i]}: -\n')
            else:
                f.write(f'{names[i]}: {apv*100:.4f}%\n')
    print(f'Saved: {out}')

if __name__ == '__main__':
    main()
