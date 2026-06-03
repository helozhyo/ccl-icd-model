# CCL 2025 ICD 自动编码系统 / ICD Automatic Coding System

[English](#english) | [中文](#中文)

---

## 中文

### 项目简介

本项目为 CCL 2025 评测任务——病案首页 ICD 自动编码的参赛代码。任务要求对病案文本预测：

- **主要诊断编码**（18类，单标签分类）
- **其他诊断编码**（967类，多标签分类）
- **主要手术编码**（16类，单标签分类）
- **其他手术编码**（71类，多标签分类）

### 方法概述

使用冻结的 **InternLM2-1.8B** 作为文本编码器，在其上叠加轻量级 MLP 分类头进行训练。

```
病案文本
   ↓
InternLM2-1.8B (冻结，fp16)
   ↓ mean pooling (最后一层 hidden states)
   ↓
Linear + GELU + Dropout  (proj, hidden→512)
   ↓
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  MLP + CE    │  MLP + CE    │  MLP + BCE   │  MLP + BCE   │
│  主要诊断    │  主要手术    │  其他诊断    │  其他手术    │
│  (18类)      │  (16类)      │  (967类)     │  (71类)      │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

### 评估指标

$$M = 0.4 \times Acc\_主诊断 + 0.1 \times F1\_其他诊断 + 0.4 \times Acc\_主手术 + 0.1 \times F1\_其他手术$$

### 训练流程

#### 环境依赖

```
torch>=2.0
transformers
paramiko
openpyxl
tqdm
numpy
pandas
```

#### 文件结构

```
├── train_classifier.py       # 初始训练（从头训练分类头）
├── train_classifier_v2.py    # 继续训练（从 best_classifier.pt 热启动）
├── infer_classifier_v2.py    # 基于 best_classifier.pt 推理
├── infer_classifier_v3.py    # 多阈值变体推理（v3a/b/c/d）
├── infer_v4.py               # 基于 best_classifier_v2.pt 推理（v4a/b/c/d）
├── search_thresh_extended.py # dev 集细粒度阈值搜索
├── eval_dev_with_search.py   # dev 集评估
├── prepare_data.py           # 数据预处理
├── ssh_tool.py               # SSH 工具（远程服务器操作）
└── ...                       # 其他调试/实验脚本
```

#### 训练步骤

**Step 1：初始训练**

```bash
python train_classifier.py
# 使用 train.json（约1620样本）训练
# 保存最优模型到 best_classifier.pt
```

**Step 2：热启动继续训练**

```bash
python train_classifier_v2.py
# 从 best_classifier.pt 热启动
# 使用 train_full.json（1800样本）训练 3 epochs
# LR=2e-4，AdamW + Linear Warmup
# 保存最优模型到 best_classifier_v2.pt
```

**Step 3：推理生成提交文件**

```bash
python infer_v4.py
# 一次推理生成 4 种阈值变体：
# sub_v4a.txt: td=0.30, ts=0.45, ks=2 （dev最优）
# sub_v4b.txt: td=0.25, ts=0.40, ks=3 （A榜历史最优参数）
# sub_v4c.txt: td=0.25, ts=0.45, ks=3
# sub_v4d.txt: td=0.30, ts=0.40, ks=3
```

其中 `td`=其他诊断阈值，`ts`=其他手术阈值，`ks`=其他手术 top-k。

#### 关键超参数

| 参数 | 值 |
|------|----|
| 基座模型 | InternLM2-1.8B |
| MAX_LEN | 512 |
| BATCH_SIZE | 4 |
| EMBED_DIM | 512 |
| LR | 2e-4 |
| EPOCHS | 3 |
| Warmup | 10% steps |
| Optimizer | AdamW (weight_decay=1e-2) |
| LLM | 冻结 (fp16) |

### 实验结果

| 模型版本 | Dev M | A榜总分 | 主诊断 | 其他诊断 | 主手术 | 其他手术 |
|----------|-------|---------|--------|---------|--------|---------|
| best_classifier.pt (v3b) | ~0.80 | **0.7025** | - | - | - | - |
| best_classifier_v2.pt (v4b) | 0.8076 | 0.6912 | 0.7600 | 0.2674 | 0.7825 | 0.4745 |

### SSH 工具配置

`ssh_tool.py` 中需填写服务器信息：

```python
HOST = 'YOUR_SERVER_HOST'
PORT = 00000
USER = 'root'
PASSWORD = 'YOUR_PASSWORD'
```

---

## English

### Overview

This repository contains the code for CCL 2025 shared task: automatic ICD coding from Chinese clinical discharge summaries. The model predicts four fields:

- **Main diagnosis code** (18 classes, single-label)
- **Other diagnosis codes** (967 classes, multi-label)
- **Main surgery code** (16 classes, single-label)
- **Other surgery codes** (71 classes, multi-label)

### Method

A frozen **InternLM2-1.8B** LLM serves as the text encoder. Its mean-pooled last hidden states are projected into a shared embedding, which is then fed into four separate MLP classification heads.

```
Clinical text
   ↓
InternLM2-1.8B (frozen, fp16)
   ↓ mean pooling over last hidden states
   ↓
Linear + GELU + Dropout  (proj, hidden→512)
   ↓
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  MLP + CE    │  MLP + CE    │  MLP + BCE   │  MLP + BCE   │
│  main_diag   │  main_surg   │  other_diag  │  other_surg  │
│  (18 cls)    │  (16 cls)    │  (967 cls)   │  (71 cls)    │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

Only the MLP heads are trained; the LLM backbone remains frozen throughout.

### Evaluation Metric

$$M = 0.4 \times Acc\_main\_diag + 0.1 \times F1\_other\_diag + 0.4 \times Acc\_main\_surg + 0.1 \times F1\_other\_surg$$

### Training Pipeline

#### Requirements

```
torch>=2.0
transformers
paramiko
openpyxl
tqdm
numpy
pandas
```

#### File Structure

```
├── train_classifier.py       # Stage 1: train heads from scratch
├── train_classifier_v2.py    # Stage 2: warm-start from best_classifier.pt
├── infer_classifier_v2.py    # Inference with best_classifier.pt
├── infer_classifier_v3.py    # Multi-threshold inference (v3a/b/c/d)
├── infer_v4.py               # Inference with best_classifier_v2.pt (v4a/b/c/d)
├── search_thresh_extended.py # Fine-grained threshold search on dev set
├── eval_dev_with_search.py   # Dev set evaluation
├── prepare_data.py           # Data preprocessing
├── ssh_tool.py               # SSH utility for remote server
└── ...                       # Other debug/experiment scripts
```

#### Steps

**Step 1: Initial training**

```bash
python train_classifier.py
# Trains on train.json (~1620 samples)
# Saves best model to best_classifier.pt
```

**Step 2: Warm-start fine-tuning**

```bash
python train_classifier_v2.py
# Loads best_classifier.pt as warm start
# Trains on train_full.json (1800 samples) for 3 epochs
# LR=2e-4, AdamW + linear warmup
# Saves best model to best_classifier_v2.pt
```

**Step 3: Inference**

```bash
python infer_v4.py
# Generates 4 submission variants in one pass:
# sub_v4a.txt: td=0.30, ts=0.45, ks=2  (dev-optimal)
# sub_v4b.txt: td=0.25, ts=0.40, ks=3  (best A-board params)
# sub_v4c.txt: td=0.25, ts=0.45, ks=3
# sub_v4d.txt: td=0.30, ts=0.40, ks=3
```

Where `td` = other_diag threshold, `ts` = other_surg threshold, `ks` = top-k for other_surg.

#### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | InternLM2-1.8B |
| MAX_LEN | 512 |
| BATCH_SIZE | 4 |
| EMBED_DIM | 512 |
| LR | 2e-4 |
| EPOCHS | 3 |
| Warmup | 10% of total steps |
| Optimizer | AdamW (weight_decay=1e-2) |
| LLM | Frozen (fp16) |

### Results

| Model | Dev M | Leaderboard | main_diag | other_diag | main_surg | other_surg |
|-------|-------|-------------|-----------|-----------|-----------|-----------|
| best_classifier.pt (v3b) | ~0.80 | **0.7025** | - | - | - | - |
| best_classifier_v2.pt (v4b) | 0.8076 | 0.6912 | 0.7600 | 0.2674 | 0.7825 | 0.4745 |

### SSH Tool Configuration

Fill in your server credentials in `ssh_tool.py`:

```python
HOST = 'YOUR_SERVER_HOST'
PORT = 00000
USER = 'root'
PASSWORD = 'YOUR_PASSWORD'
```
