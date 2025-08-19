# Federated YOLOv9 on NCHC HPC

實作在 NCHC/TWCC HPC 上的 **聯邦式學習 (Federated Learning)** 版 YOLOv9：支援多節點多 GPU、Slurm 排程、Singularity 容器，以及線性跨節點/同步平行之聯邦式訓練與評估。

> Based on YOLOv9. Federated extensions and HPC scripts by supercookie.

---

## 📦 環境說明  

本專案運行於 **國網中心 (NCHC) TWCC HPC** 叢集環境，使用 **Slurm** 作業排程系統與 **Singularity** 容器管理工具。  
- **Slurm**：負責跨節點、跨 GPU 的分配與調度。  
- **Singularity**：用於建立獨立、可攜帶的容器環境，確保實驗的可重現性。  
- 詳細 TWCC 使用方式，請參考官方文件 👉 [TWCC 使用手冊][https://man.twcc.ai](https://man.twcc.ai/@twccdocs/doc-twnia2-main-zh/https%3A%2F%2Fman.twcc.ai%2F%40twccdocs%2Ftwnia2-overview-zh)  

此環境支援 **多節點多 GPU** 的聯邦學習實驗，適合大規模分散式訓練。  

---

## 📂 資料集說明

### 下載 KITTI 資料集
本專案使用 **KITTI Vision Benchmark Suite - 2D Object Detection**  
官方網址：[KITTI Dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)

請先下載 **images** 與 **labels**（YOLO 格式或經轉換後的標註檔）。

```bash
# 建立資料夾
cd datasets/kitti

# 下載 KITTI 官方影像與標註檔（需先到官網註冊並同意協議）
wget http://www.cvlibs.net/download.php?file=data_object_image_2.zip -O images.zip
wget http://www.cvlibs.net/download.php?file=data_object_label_2.zip -O labels.zip

# 解壓縮
unzip images.zip -d images
unzip labels.zip -d labels

```

---

## 🚀 執行方法（How to Run）

> 本專案在 **NCHC / TWCC HPC** 上運行，透過 **Slurm** 排程與 **Singularity**環境執行。下列為兩種常見操作模式的**步驟說明**；對應指令已存放於專案中，請依你的資源（節點/GPU/partition）調整後提交。

---

### A) 多 GPU 跨節點加速運算

![集中式 DDP 工作流程](figure/central_ddp.png)

**目的**：使用 `client`（client0–3 合併）資料集進行加速訓練，作為與平行式訓練比較的 baseline。

**步驟**
1. **建立/確認輸出目錄**：已於 TWCC HFS建好環境（Singularity），可使用 GPU。`fed_client_weights/`、`global_round_weights/`、`fed_final_weights/`、`fed_val_client/`（供各階段輸出）。
2. **資料設定**：`data/kitti_client.yaml` 指向 `datasets/kitti/client/`（影像/標註）。
3. **資源配置**：在 Slurm 腳本中設定節點數、每節點 GPU 數、CPU/記憶體等
4. **提交訓練**：以 DDP（多節點多 GPU)方式啟動集中式訓練，以 ln01.twcc.ai登入twcc節點後，使用slurm指令 sbatch 派送 fed_client.sb 腳本，即可自動化完成實驗。
5. **查看結果**：訓練權重（baseline）輸出至 `fed_client_weights/`（或你的 runs 路徑）；使用驗證腳本在 `val` 上測試，輸出至 `fed_val_client/`。

---

### B) 平行同步運算（Federated Learning：Clients 同步並行 → Server 聚合）

![Federated 平行同步流程](figure/fl_parallel.png)

**目的**：模擬真實分散式情境；clients 於各自資料上**並行訓練**，完畢後由 server **同步聚合**（FedAvg），反覆多輪直到收斂。

**步驟**
1. **建立/確認輸出目錄**：已於 TWCC HFS建好環境（Singularity），可使用 GPU。`fed_client_weights/`、`global_round_weights/`、`fed_final_weights/`、`fed_val_client/`（供各階段輸出）。
2. **資料設定**：`data/kitti_client.yaml` 指向 `datasets/kitti/client/`（影像/標註）。
3. **資源配置**：在 Slurm 腳本中設定節點數、每節點 GPU 數、CPU/記憶體等
4. **提交訓練**：以 parallel（多節點單GPU平行）方式啟動平行訓練，以 ln01.twcc.ai登入twcc節點後，使用slurm指令 sbatch 派送 fed_parallel.sb 腳本，即可自動化完成實驗。
5. **查看結果**：訓練權重（baseline）輸出至 `fed_client_weights/`（或你的 runs 路徑）；使用驗證腳本在 `val` 上測試，輸出至 `fed_val_client/`。

---

### 備註
- **環境**：TWCC 建議使用 **Singularity** 確保一致性；本地測試可改用 **Conda**。
- **路徑一致性**：確保 YAML 中的資料路徑與實際目錄一致（`central` / `client0~3` / `val`）。

---


## 👀 結果與監控

本節說明訓練完成後「要看哪裡、看什麼」，以及訓練過程中如何確認狀態是否正常。

---

### 1) 線性式訓練（Central / DDP）
- **模型權重**：`fed_client_weights/`  
  - 觀察是否產生 `best.pt` / `last.pt` 等權重檔。
- **驗證結果**：`fed_val_central/`  
  - 主要指標：`mAP@0.5`、`mAP@0.5:0.95`、Precision、Recall。  
  - 建議輸出：混淆矩陣、PR 曲線、mAP 收斂曲線。
- **偵測輸出**：`runs/detect/exp*/`  
  - 快速肉眼檢查：框與遮罩是否合理、是否有明顯漏檢/誤檢。
- **訓練日誌（Slurm）**：`slurm-*.out`  
  - 確認是否有 OOM、Dataloader 卡住、權重存檔失敗等訊息。

---

### 2) 平行式訓練（Federated：Clients 並行 → Server 聚合）
- **Clients 本地權重**：`fed_client_weights/round_k/`  
  - 每輪 (k) 完成後應有 `client0.pt`…`client3.pt` 等檔案。
- **全域聚合權重**：`global_round_weights/round_k/`  
  - Server 聚合產出的 `global.pt`（或你在腳本中設定的檔名）。
- **最終模型**：`fed_final_weights/`  
  - 訓練完成後的全域模型（如 `best.pt`），用於最終評估與部署。
- **驗證結果**：`fed_val_client/`（與/或 `fed_val_central/`）  
  - 觀察每輪或最終的：`mAP@0.5`、`mAP@0.5:0.95`、Precision、Recall。 





  

