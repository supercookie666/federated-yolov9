# Federated YOLOv9 on NCHC HPC

實作在 NCHC/TWCC HPC 上的 **聯邦式學習 (Federated Learning)** 版 YOLOv9：支援多節點多 GPU、Slurm 排程、Singularity 容器，以及集中式/聯邦式訓練與評估。

> Based on YOLOv9. Federated extensions and HPC scripts by QQ.

---

## 📦 環境說明  

本專案運行於 **國網中心 (NCHC) TWCC HPC** 叢集環境，使用 **Slurm** 作業排程系統與 **Singularity** 容器管理工具。  
- **Slurm**：負責跨節點、跨 GPU 的分配與調度。  
- **Singularity**：用於建立獨立、可攜帶的容器環境，確保實驗的可重現性。  
- 詳細 TWCC 使用方式，請參考官方文件 👉 [TWCC 使用手冊](https://man.twcc.ai)  

此環境支援 **多節點多 GPU** 的聯邦學習實驗，適合大規模分散式訓練。  

---

## 📊 資料準備  

使用資料集：**KITTI Vision Benchmark Suite - 2D Object Detection**  
- 原始資料包含 **Car、Van、Truck、Pedestrian、Cyclist、Tram、Misc** 等類別。  
- 本研究將資料平均拆分為 **client0 ~ client3** 四個訓練子集，並額外保留 **val** 作為驗證集。  
- **central** 資料集則由 **client0-3** 合併而成，用於集中式訓練與對照實驗。  
