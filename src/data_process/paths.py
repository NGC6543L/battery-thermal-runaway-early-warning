"""
项目统一路径定义
================
所有脚本的默认路径都从这里取。路径基于本文件的位置推导, 因此
**无论从哪个目录运行脚本, 指向的都是同一个项目根目录**。

目录约定 (详见根目录 README.md):
  data/raw/            原始实验文件 (只读)
  meta/                标注与元信息表 (main.xlsx / 台账 / A-B 汇总)
  interim/             中间产物 (可随时重生成)
  dataset/             最终数据集 (legacy_base / legacy_feat113 / v3)
  results/             模型结果 (legacy_baseline / legacy_tuned / legacy_featsel / v3)
  figures/             复核图
  docs/                文档
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]        # 项目根目录
SRC = ROOT / 'src' / 'data_process'

# ---------------- 代码目录 ----------------
V1_SRC = SRC / 'v1'          # 数据处理第一版 (Day1–7, 二分类)
V3_SRC = SRC / 'v3'          # 数据处理 v3 (三阶段分类)

# ---------------- 原始数据 ----------------
DATA_RAW = ROOT / 'data' / 'raw'

# ---------------- 元信息 / 标注 ----------------
META = ROOT / 'meta'
MAIN_XLSX = META / 'main.xlsx'
META_LEDGER = META / 'meta_ledger.csv'
SAMPLE_META = META / 'sample_meta.csv'
STAGE_SUMMARY = META / 'stage_summary.csv'

# ---------------- 中间产物 ----------------
INTERIM = ROOT / 'interim'
PROCESSED_DATA = INTERIM / 'processed_data'       # v1: 0.2s 时序
PROCESSED_V3 = INTERIM / 'processed_v3'           # v3: 10Hz + 温度融合
TIMESERIES_STAGE = INTERIM / 'timeseries_stage'   # v3: A/B + 三阶段

# ---------------- 数据集 ----------------
# 目录名中的 legacy_* 指的是"数据处理第一版 (v1)"链路产出的数据集/结果
DATASET = ROOT / 'dataset'
DATASET_LEGACY = DATASET / 'legacy_base'          # v1 基础特征   70 列  二分类
DATASET_LEGACY113 = DATASET / 'legacy_feat113'    # v1 增强特征  113 列  二分类
DATASET_V3 = DATASET / 'v3'                       # v3  27 列 + 张量  三分类

# ---------------- 模型结果 ----------------
RESULTS = ROOT / 'results'
RESULTS_LEGACY = RESULTS / 'legacy_baseline'      # v1 基线
RESULTS_TUNED = RESULTS / 'legacy_tuned'          # v1 阈值优化
RESULTS_FEATSEL = RESULTS / 'legacy_featsel'      # v1 特征选择
RESULTS_V3 = RESULTS / 'v3'                       # v3 三分类结果

# ---------------- 图 / 文档 ----------------
FIGURES = ROOT / 'figures'
DOCS = ROOT / 'docs'


def s(p):
    """转成字符串, 方便直接作为 argparse 的 default"""
    return str(p)
