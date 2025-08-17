# aggregate.py
# Updated to accept --method flag (supports only 'fedavg')

# +
# #!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedAvg 權重聚合 for YOLOv9
直接匯入 YOLOv9 Model，並輸出與 train_dual.py 完全相容的 checkpoint。
現在支援多個 data yaml，以處理每個 client 各自的資料設定。
"""
import os
import sys
import argparse
import yaml
import torch

# 匯入 YOLOv9 Model 類別
try:
    from models.yolo import Model
except ImportError:
    sys.stderr.write("[ERROR] 無法從 models.yolo 匯入 Model。請確認當前目錄為專案根目錄，且已安裝相依套件。\n")
    sys.exit(1)

# CheckpointModel wrapper，使 train_dual.py 直接可用
class CheckpointModel:
    def __init__(self, state_dict):
        self._sd = state_dict
    def float(self):
        return self
    def state_dict(self):
        return self._sd


def load_state_dict(path):
    """
    從 .pt 檔案獲取純 state_dict
    支援 raw dict、{'model_state_dict':...}、{'model': nn.Module} 等格式
    """
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            return ckpt['model_state_dict']
        if 'model' in ckpt and hasattr(ckpt['model'], 'state_dict'):
            return ckpt['model'].state_dict()
        return ckpt
    return ckpt


def federated_average(state_dicts, sizes=None):
    """
    FedAvg：對所有 client 的 state_dict 做加權平均
    """
    n = len(state_dicts)
    if sizes:
        if len(sizes) != n:
            raise ValueError("sizes 長度必須等於模型數量")
        total = float(sum(sizes))
        weights = [s / total for s in sizes]
    else:
        weights = [1.0 / n] * n

    # 取交集 keys
    keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        keys &= set(sd.keys())
    avg_sd = {}
    for k in sorted(keys):
        v0 = state_dicts[0][k]
        if not isinstance(v0, torch.Tensor):
            continue
        acc = torch.zeros_like(v0, dtype=torch.float32)
        for w, sd in zip(weights, state_dicts):
            acc += sd[k].float() * w
        avg_sd[k] = acc
    return avg_sd


def parse_args():
    p = argparse.ArgumentParser(
        description='FedAvg 權重聚合 for YOLOv9'
    )
    p.add_argument(
        '-i','--input-models', nargs='+', required=True,
        help='Client 權重 .pt 檔列表'
    )
    p.add_argument(
        '-o','--output-model', required=True,
        help='輸出聚合後 checkpoint (.pt)'
    )
    p.add_argument(
        '--cfg', required=True,
        help='YOLOv9 模型配置檔 (.yaml)，須與 train_dual.py 使用相同'
    )
    p.add_argument(
        '-d','--data', nargs='+', required=True,
        help='每個 client 的 data yaml 檔，程式將檢查它們的 nc 是否一致'
    )
    p.add_argument(
        '-s','--sizes', nargs='+', type=int,
        help='各 client 樣本數列表，用於加權平均'
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 檔案檢查
    for f in args.input_models + args.data + [args.cfg]:
        if not os.path.isfile(f):
            sys.stderr.write(f"[ERROR] 找不到檔案: {f}\n")
            sys.exit(1)

    # 讀取 state_dict
    state_dicts = []
    for f in args.input_models:
        try:
            state_dicts.append(load_state_dict(f))
        except Exception as e:
            sys.stderr.write(f"[ERROR] 載入 {f} 失敗: {e}\n")
            sys.exit(1)

    # 執行 FedAvg
    try:
        agg_sd = federated_average(state_dicts, sizes=args.sizes)
    except Exception as e:
        sys.stderr.write(f"[ERROR] 聚合失敗: {e}\n")
        sys.exit(1)

    # 解析所有 data yaml，確認 nc 欄位一致
    ncs = []
    for f in args.data:
        with open(f) as fp:
            dd = yaml.safe_load(fp)
            if 'nc' not in dd:
                sys.stderr.write(f"[ERROR] {f} 缺少 nc 欄位\n")
                sys.exit(1)
            ncs.append(int(dd['nc']))
    if len(set(ncs)) != 1:
        sys.stderr.write(f"[ERROR] 多個 data yaml 的 nc 不一致: {ncs}\n")
        sys.exit(1)
    nc = ncs[0]

    # 建立 Model 並載入聚合權重
    model = Model(args.cfg, ch=3, nc=nc).cpu()
    missing, unexpected = model.load_state_dict(agg_sd, strict=False)
    if missing or unexpected:
        print(f"[WARNING] 載入時 missing: {missing}, unexpected: {unexpected}")

    # 注入 CheckpointModel hook
    CheckpointModel.__module__ = '__main__'
    import __main__
    setattr(__main__, 'CheckpointModel', CheckpointModel)

    # 存成 train_dual.py 可讀的格式
    torch.save({'model': model}, args.output_model)
    print(f"[OK] 聚合完成，輸出: {args.output_model}")

if __name__ == '__main__':
    main()


