# Federated YOLOv9 on NCHC HPC

實作在 NCHC/TWCC HPC 上的 **聯邦式學習 (Federated Learning)** 版 YOLOv9：支援多節點多 GPU、Slurm 排程、Singularity 容器，以及集中式/聯邦式訓練與評估。

> Based on YOLOv9. Federated extensions and HPC scripts by QQ.

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
  
