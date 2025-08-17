# +
# #!/usr/bin/env python3
# val_dual.py — safe validator for YOLOv9 dual checkpoints (state_dict or full ckpt)
# - Falls back to data.yaml for nc/names (federated weights often lack meta)
# - Unwraps model outputs: (pred, aux) / [pred, aux] / {'pred': ...}
# - Has local scale_coords compatible with letterbox shapes
# - Forces saving labels/*.txt (YOLO format, with conf), computes mAP@0.5, writes results.txt/results.csv

import argparse
import os
import glob
import math
from pathlib import Path
import numpy as np
import torch

from utils.general import (
    LOGGER, check_dataset, check_img_size, check_yaml, colorstr,
    increment_path, non_max_suppression, xyxy2xywh, set_logging
)
from utils.torch_utils import select_device, time_sync
from utils.dataloaders import create_dataloader
from models.common import DetectMultiBackend

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

# ------------ helpers ------------
def unwrap_preds(preds):
    # Handle (pred, aux) / [pred, aux] / {'pred': tensor}
    if isinstance(preds, (list, tuple)):
        return preds[0]
    if isinstance(preds, dict) and 'pred' in preds:
        return preds['pred']
    return preds  # assume tensor [B, N, 85]

def scale_coords_local(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    map boxes from letterboxed img (img1_shape) back to original img (img0_shape)
    coords: tensor Nx4 of xyxy in img1 space
    """
    if coords is None or len(coords) == 0:
        return coords
    if not torch.is_tensor(coords):
        coords = torch.as_tensor(coords)
    if isinstance(img0_shape, (list, tuple)):
        h0, w0 = int(img0_shape[0]), int(img0_shape[1])
    else:
        h0 = w0 = int(img0_shape)

    if ratio_pad is None:
        gain = min(img1_shape[0] / h0, img1_shape[1] / w0)
        padw = (img1_shape[1] - w0 * gain) / 2
        padh = (img1_shape[0] - h0 * gain) / 2
    else:
        rp0, rp1 = ratio_pad
        if isinstance(rp0, (list, tuple)):
            gain_w, gain_h = rp0[0], rp0[1]
            gain = (gain_w + gain_h) / 2.0
        else:
            gain = rp0
        padw, padh = rp1 if isinstance(rp1, (list, tuple)) else (rp1, rp1)

    coords[:, [0, 2]] -= padw
    coords[:, [1, 3]] -= padh
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, w0 - 1)
    coords[:, 1].clamp_(0, h0 - 1)
    coords[:, 2].clamp_(0, w0 - 1)
    coords[:, 3].clamp_(0, h0 - 1)
    return coords

def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return np.sum((mrec[idx] - mrec[idx-1]) * mpre[idx])

def eval_map50_from_txt(pred_dir, gt_dir, nc):
    # Load GT
    gt_boxes, gt_classes = {}, {}
    for p in glob.glob(os.path.join(gt_dir, '*.txt')):
        stem = Path(p).stem
        boxes, cls = [], []
        with open(p) as f:
            for line in f:
                q = line.strip().split()
                if len(q) < 5: continue
                c = int(float(q[0]))
                if c < 0 or c >= nc: continue
                x, y, w, h = map(float, q[1:5])
                x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
                boxes.append([x1, y1, x2, y2]); cls.append(c)
        gt_boxes[stem] = np.array(boxes, dtype=float) if boxes else np.zeros((0,4))
        gt_classes[stem] = cls

    # Load predictions
    preds_by_img = {}
    for p in glob.glob(os.path.join(pred_dir, '*.txt')):
        stem = Path(p).stem
        arr = []
        with open(p) as f:
            for line in f:
                q = line.strip().split()
                if len(q) < 5: continue
                c = int(float(q[0])); x,y,w,h = map(float, q[1:5])
                conf = float(q[5]) if len(q) > 5 else 0.001
                x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
                arr.append([c, conf, x1, y1, x2, y2])
        preds_by_img[stem] = np.array(arr, dtype=float) if arr else np.zeros((0,6))

    def iou_xyxy(a, b):
        if a.shape[0] == 0 or b.shape[0] == 0:
            return np.zeros((a.shape[0], b.shape[0]))
        inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
        inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
        inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
        inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
        iw = np.clip(inter_x2 - inter_x1, 0, None)
        ih = np.clip(inter_y2 - inter_y1, 0, None)
        inter = iw * ih
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-16)

    aps = np.zeros(nc, dtype=float); valid = 0
    for c in range(nc):
        # Collect predictions of class c, sort by conf desc
        P = []
        for stem, arr in preds_by_img.items():
            if arr.size == 0: continue
            sel = arr[arr[:,0] == c]
            for row in sel:
                P.append((stem, row[1], row[2:6]))
        P.sort(key=lambda x: x[1], reverse=True)

        # Collect GT of class c
        G = {}
        for stem, cls_list in gt_classes.items():
            idx = [i for i, cc in enumerate(cls_list) if cc == c]
            if not idx: continue
            G[stem] = {'boxes': gt_boxes[stem][idx], 'matched': np.zeros(len(idx), dtype=bool)}
        if len(G) == 0:
            aps[c] = np.nan
            continue

        tp, fp = [], []
        for stem, conf, box in P:
            if stem not in G or G[stem]['boxes'].shape[0] == 0:
                fp.append(1); tp.append(0); continue
            ious = iou_xyxy(np.array(box)[None,:], G[stem]['boxes'])[0]
            j = int(np.argmax(ious))
            if ious[j] >= 0.5 and not G[stem]['matched'][j]:
                tp.append(1); fp.append(0); G[stem]['matched'][j] = True
            else:
                fp.append(1); tp.append(0)
        tp = np.cumsum(tp); fp = np.cumsum(fp)
        npos = sum(len(G[s]['boxes']) for s in G)
        if npos == 0:
            aps[c] = np.nan
        else:
            rec = tp / (npos + 1e-16)
            prec = tp / np.maximum(tp + fp, 1e-16)
            aps[c] = voc_ap(rec, prec); valid += 1

    mAP = float(np.nanmean(aps)) if valid else 0.0
    return mAP, aps

# ------------ main ------------
@torch.no_grad()
def run(data, weights='best.pt', batch_size=16, imgsz=640, conf_thres=0.001, iou_thres=0.65,
        device='', workers=0, single_cls=False, augment=False, verbose=True,
        save_txt=False, save_conf=False, project='runs/val_dual', name='exp',
        exist_ok=False, half=False, dnn=False, max_det=300, min_items=0):

    # 強制存 txt（為了後面計算 mAP）；同時把 conf 一起存
    save_txt = True
    save_conf = True

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(device, batch_size=batch_size)

    # Model
    model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)
    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride)

    # Dataset / names / nc (以 data.yaml 為準)
    data_dict = check_dataset(check_yaml(data))
    names = data_dict.get('names', None)
    nc = int(data_dict.get('nc', len(names) if names else 0))
    assert nc > 0 and names, "請在 data.yaml 設定 'nc' 與 'names'（需與訓練時一致）"
    try:
        model.names = names
    except Exception:
        pass

    # Dataloader
    val_path = data_dict.get('val', None)
    assert val_path, "data.yaml 缺少 'val:' 路徑"
    dataloader = create_dataloader(val_path, imgsz, batch_size, stride,
                                   single_cls=single_cls, pad=0.5, rect=True,
                                   workers=workers, prefix=colorstr('val: '), shuffle=False)[0]

    seen = 0
    for batch_i, (im, targets, paths, shapes) in enumerate(dataloader):
        t1 = time_sync()
        im = im.to(device, non_blocking=True)
        im = im.half() if half else im.float()
        im /= 255
        nb, _, h, w = im.shape
        t2 = time_sync()

        # forward
        preds = model(im, augment=augment, visualize=False)
        preds = unwrap_preds(preds)

        # NMS
        preds = non_max_suppression(preds, conf_thres, iou_thres, classes=None, agnostic=False, max_det=max_det)
        t3 = time_sync()

        for si, pred in enumerate(preds):
            seen += 1
            p = Path(paths[si])
            # shapes[si] 常見格式：((h0,w0), (ratio, pad))
            orig_shape = shapes[si][0] if isinstance(shapes[si], (list, tuple)) else (h, w)
            ratio_pad = shapes[si][1] if isinstance(shapes[si], (list, tuple)) and len(shapes[si]) > 1 else None

            if len(pred):
                # 還原到原圖座標
                pred[:, :4] = scale_coords_local(im[si].shape[1:], pred[:, :4], orig_shape, ratio_pad).round()
                # 存 YOLO 標註（normalized）— 全在 CPU 做，避免 device mismatch
                gn = torch.tensor((orig_shape[1], orig_shape[0], orig_shape[1], orig_shape[0]), dtype=torch.float32)
                txt_path = save_dir / 'labels' / f'{p.stem}.txt'
                if txt_path.exists():  # 避免覆寫疊行
                    txt_path.unlink()
                for *xyxy, conf, cls in pred.tolist():
                    xyxy_t = torch.tensor(xyxy, dtype=torch.float32).view(1, 4)
                    xywh = (xyxy2xywh(xyxy_t) / gn).view(-1).tolist()
                    line = (int(cls), *xywh, conf) if save_conf else (int(cls), *xywh)
                    with open(txt_path, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

        if verbose and (batch_i % 20 == 0):
            LOGGER.info(f'[{batch_i}/{len(dataloader)}] {(t2 - t1)*1e3:.1f}ms pre, {(t3 - t2)*1e3:.1f}ms inf')

    # === 評估 mAP@0.5（用剛剛存的 labels/*.txt + GT labels） ===
    val_path = Path(val_path)
    if val_path.name == 'images':
        gt_dir = val_path.parent / 'labels'
    else:
        gt_dir = Path(str(val_path).replace('images', 'labels'))
    pred_dir = save_dir / 'labels'
    
    #檢查gt跟pred路徑
    print(f"[VAL] pred_dir={pred_dir.resolve()}")
    print(f"[VAL] gt_dir={gt_dir.resolve()}")
    assert pred_dir.resolve() != gt_dir.resolve(), "Pred dir 和 GT dir 指到同一路徑！請檢查。"
    
    mAP, aps = eval_map50_from_txt(str(pred_dir), str(gt_dir), nc)

    # 寫 results.txt / results.csv
    results_txt = save_dir / 'results.txt'
    with open(results_txt, 'w') as f:
        f.write(f'mAP@0.5: {mAP*100:.4f}%\n')
        for i, ap in enumerate(aps):
            cname = names[i] if i < len(names) else f'cls{i}'
            if math.isnan(ap):
                f.write(f'{cname}: -\n')
            else:
                f.write(f'{cname}: {ap*100:.4f}%\n')

    results_csv = save_dir / 'results.csv'
    header = 'map50,' + ','.join([f'AP_{names[i]}' for i in range(nc)]) + '\n'
    need_header = not results_csv.exists()
    with open(results_csv, 'a') as f:
        if need_header:
            f.write(header)
        row = [f'{mAP:.6f}'] + [("" if math.isnan(aps[i]) else f'{aps[i]:.6f}') for i in range(nc)]
        f.write(','.join(row) + '\n')

    LOGGER.info(colorstr('bold', 'green') + f" Results saved to {save_dir}")
    LOGGER.info(f"mAP@0.5: {mAP*100:.2f}%")

def parse_opt():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='data/kitti_val.yaml', help='dataset.yaml path')
    p.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model path(s)')
    p.add_argument('--batch-size', type=int, default=16, help='batch size')
    p.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    p.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    p.add_argument('--iou', '--iou-thres', dest='iou_thres', type=float, default=0.65, help='NMS IoU threshold')
    p.add_argument('--device', default='', help='cuda device')
    p.add_argument('--workers', type=int, default=0, help='dataloader workers')
    p.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    p.add_argument('--augment', action='store_true', help='augmented inference')
    p.add_argument('--verbose', action='store_true', help='verbose log')
    p.add_argument('--save-txt', action='store_true', help='(ignored; always save for eval)')
    p.add_argument('--save-conf', action='store_true', help='(ignored; always save for eval)')
    p.add_argument('--project', default='runs/val_dual', help='save to project/name')
    p.add_argument('--name', default='exp', help='save to project/name')
    p.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    p.add_argument('--half', action='store_true', help='FP16 inference')
    p.add_argument('--dnn', action='store_true', help='OpenCV DNN')
    p.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    p.add_argument('--min-items', type=int, default=0, help='minimum number of images to keep dataset')
    return p.parse_args()

def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


