# 锂电池热失控极早期预警 —— 数据处理

机械滥用（针刺）诱发锂离子电池热失控的极早期预警研究项目。

---

## 📌 当前仓库版本：数据处理 **第一版（v1）**

> **本仓库目前只包含「数据处理的第一版」**，即最早期搭建的完整数据链路（Day1–7）。
>
> 第一版的主要特征：
> - **任务类型：二分类**（热失控前 60–120 s 的预警窗口 = 正类，其余 = 负类）
> - **采样率：5 Hz（0.2 s）**
> - **关键时刻：仅自动检测的单一 onset（≈热失控起始点）**，没有内短路起点
> - **温度：TC1–TC4 四个独立通道**
> - **模型输入：113 维窗口统计特征**（树模型）
>
> ⚠️ **三阶段版本（v3：A/B 关键时刻 + Stage0/1/2 三分类 + 10 Hz + 温度融合）尚未上传**，
> 计划在后续提交中加入。

---

## 一、第一版做了什么

```
原始实验文件 (253 个, 异构 xlsx/csv)
   │
   ├─① preprocess_data.py   多格式解析 → 统一 5 Hz 时序 (time, voltage, force, tc1~tc4)
   │                        → interim/processed_data/   (250 个 CSV)
   │
   ├─② check_data.py        质量校验 → _quality_report.csv
   │
   ├─③ make_meta.py         扫描数据 + 读人工标注 main.xlsx → meta/meta_ledger.csv
   │
   ├─④ detect_onset.py      自动检测热失控起始时刻 → meta/sample_meta.csv
   │
   ├─⑤ build_dataset.py     滑窗 30s/步长 10s + 特征工程
   │                        标签: 窗口末端落在 [onset-120s, onset-60s] 记为正类
   │                        划分: 按样本 7:2:1
   │                        → dataset/legacy_base    (70 列)
   │                        → dataset/legacy_feat113 (113 列, 新增高级统计 + 交叉特征)
   │
   └─⑥ 模型
        train_model.py      三模型基线 (XGBoost / LightGBM / RandomForest) + 阈值校准
                            → results/legacy_baseline/
        optimize_model.py   OOF/验证集双源保守阈值 + RF 网格搜索 + 软投票集成
                            → results/legacy_tuned/
        train_v2.py         RF 特征重要性 + SelectFromModel 特征选择对比
                            → results/legacy_featsel/
```

### 第一版的几个关键设计

| 设计 | 说明 |
|---|---|
| **时间对齐** | 各信号取时间交集，按 5 Hz 线性插值到统一时间轴 |
| **滑窗** | 30 s 窗口 / 10 s 步长（0.2 s 采样 → 150 点窗、50 点步长） |
| **标签口径** | 预警区间 `[onset-120s, onset-60s]` 为正类；`> onset-60s` 的窗口直接丢弃 |
| **防泄漏** | 按 `sample_id` 划分，保证同一实验的窗口不跨集合 |
| **评估** | 窗口级指标 + **事件级召回**（一条实验只要检出一个预警窗即算命中） |
| **预警优先** | 按目标召回率反向扫描阈值，宁可误报减少漏报 |

---

## 二、运行方式

所有路径都由 `src/data_process/paths.py` 基于文件位置推导，**可以在任意目录下运行**。

```powershell
$PY = "D:/develop/python/python.exe"

# ① 预处理（先把原始数据放到 data/raw/）
& $PY -u src/data_process/v1/preprocess_data.py

# ② 质量校验
& $PY    src/data_process/v1/check_data.py

# ③ 生成元信息台账（需要 meta/main.xlsx 人工标注）
& $PY    src/data_process/v1/make_meta.py

# ④ 检测热失控起始时刻
& $PY    src/data_process/v1/detect_onset.py

# ⑤ 生成二分类数据集
& $PY -u src/data_process/v1/build_dataset.py

# ⑥ 训练与优化
& $PY -u src/data_process/v1/train_model.py
& $PY -u src/data_process/v1/optimize_model.py
& $PY -u src/data_process/v1/train_v2.py
```

> `plot_onset_check.py` 是辅助工具，用于人工逐个复核检出的 onset 位置：
> `& $PY src/data_process/v1/plot_onset_check.py interim/processed_data meta/sample_meta.csv`

---

## 三、第一版的产出

| 产物 | 位置 | 规模 |
|---|---|---|
| 标准化时序 | `interim/processed_data/` | 250 个 CSV × 7 列（0.2 s） |
| 质量报告 | `interim/processed_data/_quality_report.csv` | 192 条 |
| 样本台账 | `meta/meta_ledger.csv` | 250 行 |
| 标签基础 | `meta/sample_meta.csv` | 250 行（热失控 52） |
| 基础数据集 | `dataset/legacy_base/` | 17,631 窗口 × 70 列 |
| 增强数据集 | `dataset/legacy_feat113/` | 17,631 窗口 × 113 列 |
| 模型结果 | `results/legacy_baseline` `legacy_tuned` `legacy_featsel` | 指标 / 图 / `.joblib` |

**数据集标签分布**：正样本 195 / 负样本 17,436（正类占 **1.11%**），
共 180 个样本进入数据集，其中 **37 / 51 个热失控样本**能产出正类窗口（覆盖率 72.5%）。

**测试集结果**（窗口级召回 / 事件级召回）：

| 模型 | 窗口级召回 | 事件级召回 |
|---|---|---|
| XGBoost | 0.667 | 3/4 |
| RandomForest | 1.000 | 4/4 |
| LightGBM | 0.750 | 4/4 |

> 测试集 AUC 普遍 > 0.99，但正类精确率较低（0.25–0.41），说明"排序能力好、
> 概率绝对值不迁移"——这是第一版遗留的主要问题之一。

---

## 四、目录结构

```
project/
├─ README.md                ← 本文件
├─ requirements.txt
├─ .gitignore
│
├─ src/data_process/
│  ├─ paths.py              ★ 所有路径的唯一来源
│  └─ v1/                   ★ 数据处理第一版（本仓库当前内容）
│
├─ data/raw/                原始数据（不入库，见"数据来源"）
├─ meta/                    标注与元信息（入库，体积小）
├─ interim/                 中间产物（不入库，可重新生成）
├─ dataset/                 数据集（不入库）
├─ results/                 模型结果（不入库）
└─ docs/
```

---

## 五、数据来源

原始数据为公开数据集，**不在本仓库中**：

> **Mechanically Induced Thermal Runaway for Li-ion Batteries**
> Mendeley Data, DOI: [10.17632/sn2kv34r4h.1](https://doi.org/10.17632/sn2kv34r4h.1)

下载后放入 `data/raw/` 即可按上面的命令复现。

---

## 六、版本规划

| 版本 | 状态 | 主要变化 |
|---|---|---|
| **v1（数据处理第一版）** | ✅ 本仓库当前内容 | 5 Hz · 二分类 · 单一 onset · 113 维特征 |
| **v3（三阶段版本）** | ⏳ 待上传 | 10 Hz · 温度融合 · A/B 双锚点 · Stage0/1/2 三分类 · 时序张量 |

---

## 七、环境

```powershell
pip install -r requirements.txt
```

Python ≥ 3.10（开发环境 3.13），依赖见 `requirements.txt`。
