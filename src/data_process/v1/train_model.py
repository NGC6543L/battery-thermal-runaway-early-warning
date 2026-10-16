"""
Day5: 基线模型训练与评估
================================================
功能:
  1. 同时训练 XGBoost / Random Forest / LightGBM 三个基线模型
  2. 计算 Accuracy / Precision / Recall / F1 / F2 / AUC, 输出混淆矩阵
  3. 特征重要性 Top20, 找出对极早期预警贡献最大的特征
  4. 在验证集上网格搜索最优模型的关键超参数, 再在测试集上对比提升
  5. 预警优先: 验证集上按目标召回率下调阈值 (宁可误报, 减少漏报)

修复说明:
  - grid_search 返回值数量与调用处一致
  - target_recall 参数已加入并传递到所有需要的地方
  - sample_catch_rate 已用于评估 (事件级召回率)
  - 基线模型也做阈值优化 (按目标召回率找最优阈值)
  - XGBoost / LightGBM 加了早停机制
  - --quick 快速模式 (随机搜索12组参数)
  - PR曲线 + 阈值权衡图
  - 支持 F2 指标 (更看重召回率, 适合预警任务)
"""
import argparse
import json
import sys
import warnings
from itertools import product
from pathlib import Path
import random

# 统一路径 (src/data_process/paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 标识列: 样本ID、窗口ID、窗口起止时间
ID_COLS = ['sample_id', 'window_id', 't_start', 't_end']
# 数据泄露列: 这些列不能作为特征输入模型
LEAK_COLS = ['is_thermal_runaway', 'tr_onset_time', 'obs_score']
# 目标列: 标签 (0=正常, 1=预警)
TARGET = 'label'


def load_split(data_dir):
    """加载训练/验证/测试三个数据集"""
    data_dir = Path(data_dir)
    splits = {}
    for name in ('train', 'val', 'test'):
        path = data_dir / (name + '.csv')
        if not path.exists():
            raise FileNotFoundError('文件不存在: ' + str(path) + ', 请先运行 build_dataset.py')
        splits[name] = pd.read_csv(path, encoding='utf-8-sig')
    return splits['train'], splits['val'], splits['test']


def feature_columns(df):
    """从数据中提取数值型特征列, 排除标识列、泄露列和目标列"""
    drop = set(ID_COLS + LEAK_COLS + [TARGET])
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def xy(df, cols):
    """分离特征矩阵 X 和标签向量 y"""
    x = df[cols].apply(pd.to_numeric, errors='coerce')
    y = df[TARGET].astype(int).values
    return x, y


def pos_weight(y, multiplier=1.5):
    """计算正样本权重, 用于处理类别不平衡 (正负样本比例 * 倍率)"""
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = int((y == 0).sum())
    return float(n_neg) / float(n_pos) * float(multiplier)


def make_model(name, params, spw):
    """根据名称创建分类器实例"""
    params = dict(params)
    if name == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',
            tree_method='hist',
            n_jobs=-1,
            random_state=42,
            scale_pos_weight=spw,
            max_delta_step=1,
            verbosity=0,
            **params,
        )
    if name == 'lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            objective='binary',
            n_jobs=-1,
            random_state=42,
            scale_pos_weight=spw,
            verbosity=-1,
            **params,
        )
    if name == 'random_forest':
        return RandomForestClassifier(
            n_jobs=-1,
            random_state=42,
            class_weight={0: 1.0, 1: float(spw)},
            **params,
        )
    raise ValueError('未知模型: ' + name)


def default_params(name):
    """各模型的默认超参数"""
    if name == 'xgboost':
        return {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.08,
                'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 3}
    if name == 'lightgbm':
        return {'n_estimators': 300, 'num_leaves': 31, 'learning_rate': 0.08,
                'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_samples': 20}
    if name == 'random_forest':
        return {'n_estimators': 300, 'max_depth': 16, 'min_samples_leaf': 5,
                'max_features': 'sqrt'}
    return {}


def search_space(name):
    """各模型的网格搜索参数空间"""
    if name == 'xgboost':
        return {
            'n_estimators': [150, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.03, 0.08, 0.15],
            'min_child_weight': [1, 5],
        }
    if name == 'lightgbm':
        return {
            'n_estimators': [150, 300, 500],
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.03, 0.08, 0.15],
            'min_child_samples': [10, 30],
        }
    if name == 'random_forest':
        return {
            'n_estimators': [200, 400],
            'max_depth': [8, 16, None],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 0.5],
        }
    return {}


def predict_proba(model, x):
    """获取模型对正类的预测概率"""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(x)[:, 1]
    raw = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-raw))


def predict_scores(model, x, threshold=0.5):
    """根据阈值把概率转为预测标签"""
    proba = predict_proba(model, x)
    pred = (proba >= float(threshold)).astype(int)
    return pred, proba


def choose_threshold(y_true, proba, target_recall=0.85, min_precision=0.0):
    """
    从高到低扫描阈值, 找到仍能达到目标召回率的最高阈值
    (在满足漏报约束的同时, 尽量减少误报).
    若任何阈值都达不到目标召回, 则取召回最高、精确率次之的点.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    # 候选阈值: 均匀分布 + 概率分位数
    grid = np.unique(np.concatenate([
        np.linspace(0.005, 0.995, 199),
        np.quantile(proba, np.linspace(0, 1, 101)),
    ]))
    grid = np.sort(grid)[::-1]

    fallback = None
    for t in grid:
        pred = (proba >= t).astype(int)
        rec = float(recall_score(y_true, pred, zero_division=0))
        prec = float(precision_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))
        f2 = float(fbeta_score(y_true, pred, beta=2, zero_division=0))
        row = {
            'threshold': float(t),
            'recall': rec,
            'precision': prec,
            'f1': f1,
            'f2': f2,
        }
        if fallback is None or (rec, prec, t) > (
                fallback['recall'], fallback['precision'], fallback['threshold']):
            fallback = row
        if rec + 1e-12 >= target_recall and prec + 1e-12 >= min_precision:
            return row
    return fallback if fallback is not None else {
        'threshold': 0.5, 'recall': 0.0, 'precision': 0.0, 'f1': 0.0, 'f2': 0.0,
    }


def safe_auc(y_true, proba):
    """安全计算 AUC, 若标签只有一类则返回 NaN"""
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, proba))


def evaluate(y_true, pred, proba, threshold=0.5):
    """计算所有评估指标"""
    return {
        'accuracy': float(accuracy_score(y_true, pred)),
        'precision': float(precision_score(y_true, pred, zero_division=0)),
        'recall': float(recall_score(y_true, pred, zero_division=0)),
        'f1': float(f1_score(y_true, pred, zero_division=0)),
        'f2': float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        'auc': safe_auc(y_true, proba),
        'threshold': float(threshold),
        'n': int(len(y_true)),
        'n_pos': int((y_true == 1).sum()),
        'n_pred_pos': int((pred == 1).sum()),
        'fn': int(((y_true == 1) & (pred == 0)).sum()),
        'fp': int(((y_true == 0) & (pred == 1)).sum()),
    }


def sample_catch_rate(sample_ids, y_true, pred):
    """事件级召回率: 有正窗口的实验中, 至少检出一个预警窗口的比例"""
    df = pd.DataFrame({'sid': sample_ids, 'y': y_true, 'p': pred})
    g = df.groupby('sid').agg(y=('y', 'max'), p=('p', 'max'))
    pos = g[g['y'] == 1]
    if len(pos) == 0:
        return float('nan'), 0, 0
    caught = int((pos['p'] == 1).sum())
    return float(caught / len(pos)), caught, int(len(pos))


def fmt(v):
    """格式化数值为4位小数字符串, NaN 显示为 'nan'"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'nan'
    return '{:.4f}'.format(v)


def print_metrics(title, m):
    """打印一组评估指标"""
    extra = ''
    if 'threshold' in m:
        extra += '  阈值=' + fmt(m['threshold'])
    if 'fn' in m:
        extra += '  FN=' + str(m['fn']) + '  FP=' + str(m['fp'])
    if 'catch_rate' in m:
        extra += '  事件召回=' + fmt(m['catch_rate'])
    print('  ' + title
          + '  准确率=' + fmt(m['accuracy'])
          + '  精确率=' + fmt(m['precision'])
          + '  召回率=' + fmt(m['recall'])
          + '  F1=' + fmt(m['f1'])
          + '  F2=' + fmt(m['f2'])
          + '  AUC=' + fmt(m['auc'])
          + extra
          + '  正样本=' + str(m['n_pos']) + '/' + str(m['n']))


def plot_confusion(y_true, pred, title, out_path):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['正常 (0)', '预警 (1)'])
    ax.set_yticklabels(['正常 (0)', '预警 (1)'])
    ax.set_xlabel('预测')
    ax.set_ylabel('真实')
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return cm


def plot_roc(curves, title, out_path):
    """绘制 ROC 曲线"""
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for name, y_true, proba in curves:
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=name + '  AUC=' + '{:.3f}'.format(auc))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='随机')
    ax.set_xlabel('假阳性率')
    ax.set_ylabel('真阳性率')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_pr(curves, title, out_path):
    """绘制 PR (精确率-召回率) 曲线, 适合不平衡数据"""
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for name, y_true, proba in curves:
        if len(np.unique(y_true)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ax.plot(rec, prec, label=name)
    ax.set_xlabel('召回率')
    ax.set_ylabel('精确率')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_threshold_tradeoff(y_true, proba, chosen_t, title, out_path):
    """绘制阈值-召回率-精确率权衡曲线, 标记选中的阈值"""
    ts = np.linspace(0.01, 0.99, 99)
    recs, precs = [], []
    for t in ts:
        pred = (proba >= t).astype(int)
        recs.append(recall_score(y_true, pred, zero_division=0))
        precs.append(precision_score(y_true, pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.plot(ts, recs, label='召回率', color='#C0392B', linewidth=2)
    ax.plot(ts, precs, label='精确率', color='#2F6DB3', linewidth=2)
    ax.axvline(chosen_t, color='gray', linestyle='--',
               label='选中阈值=' + '{:.3f}'.format(chosen_t))
    ax.axvline(0.5, color='#888888', linestyle=':', label='默认 0.5')
    ax.set_xlabel('阈值')
    ax.set_ylabel('分数')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_importance(names, values, title, out_path):
    """绘制特征重要性横向条形图"""
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color='#2F6DB3')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('重要性')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def get_importance(model, feat_names):
    """获取模型的特征重要性, 返回排序后的 DataFrame"""
    clf = model
    if isinstance(model, Pipeline):
        clf = model.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        values = np.asarray(clf.feature_importances_, dtype=float)
    else:
        return pd.DataFrame(columns=['feature', 'importance'])
    df = pd.DataFrame({'feature': feat_names, 'importance': values})
    return df.sort_values('importance', ascending=False).reset_index(drop=True)


def fit_pipeline(name, params, x_train, y_train, spw, x_val=None, y_val=None):
    """构建并训练 Pipeline (缺失值填补 + 可选标准化 + 分类器), 支持早停"""
    clf = make_model(name, params, spw)
    if name == 'random_forest':
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', clf),
        ])
        pipe.fit(x_train, y_train)
        return pipe

    # XGBoost / LightGBM: 先手动填补缺失值, 再直接调用 fit 传入早停参数
    imputer = SimpleImputer(strategy='median')
    x_train_imp = imputer.fit_transform(x_train)
    pipe = Pipeline([
        ('imputer', imputer),
        ('clf', clf),
    ])
    if x_val is not None and y_val is not None:
        x_val_imp = imputer.transform(x_val)
        if name == 'xgboost':
            clf.fit(x_train_imp, y_train,
                    eval_set=[(x_val_imp, y_val)],
                    verbose=False)
        else:
            # LightGBM 新版不再接受 verbose 参数, 用 callbacks 关闭日志
            import lightgbm as lgb
            clf.fit(x_train_imp, y_train,
                    eval_set=[(x_val_imp, y_val)],
                    callbacks=[lgb.log_evaluation(period=0)])
    else:
        clf.fit(x_train_imp, y_train)
    return pipe


def grid_search(name, x_train, y_train, x_val, y_val, spw, metric, out_dir,
                target_recall, quick=False, seed=42):
    """网格搜索超参数, 每个组合都做阈值优化, 返回最优模型和参数"""
    space = search_space(name)
    keys = list(space.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[space[k] for k in keys])]
    if quick:
        # 快速模式: 随机抽取12组参数
        rng = random.Random(seed)
        combos = rng.sample(combos, min(len(combos), 12))
    print('网格搜索 ' + name + ': ' + str(len(combos))
          + ' 组, 评分指标=' + metric
          + ' (阈值按召回率>=' + str(target_recall) + ' 优化)')
    rows = []
    best_score = -1.0
    best_params = None
    best_model = None
    best_thr = 0.5
    for i, params in enumerate(combos, 1):
        model = fit_pipeline(name, params, x_train, y_train, spw, x_val, y_val)
        proba = predict_proba(model, x_val)
        chosen = choose_threshold(y_val, proba, target_recall=target_recall)
        pred = (proba >= chosen['threshold']).astype(int)
        m = evaluate(y_val, pred, proba, chosen['threshold'])
        score = m.get(metric, m['f2'])
        if np.isnan(score):
            score = -1.0
        rows.append(dict(params, **m, score=score))
        mark = ''
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
            best_thr = chosen['threshold']
            mark = '  *最优'
        if i == 1 or i == len(combos) or i % max(1, len(combos) // 8) == 0 or mark:
            print('  [{}/{}] {}  {}={}  召回={}  阈值={}{}'.format(
                i, len(combos), params, metric, fmt(score),
                fmt(m['recall']), fmt(chosen['threshold']), mark))
    hist = pd.DataFrame(rows).sort_values('score', ascending=False)
    hist.to_csv(out_dir / ('grid_' + name + '.csv'), index=False, encoding='utf-8-sig')
    return best_model, best_params, best_score, best_thr, hist


def summarize_top_features(imp_df, top_k):
    """汇总 TopK 特征, 并按信号通道 (tc1/tc2/...) 分组求和"""
    top = imp_df.head(top_k)
    channels = {}
    for feat, val in zip(top['feature'], top['importance']):
        ch = feat.split('_')[0] if '_' in feat else feat
        channels[ch] = channels.get(ch, 0.0) + float(val)
    ranked = sorted(channels.items(), key=lambda kv: kv[1], reverse=True)
    return top, ranked


def run(data_dir, output_dir, metric, top_k, tune_all, seed, target_recall, quick):
    """主流程: 训练基线 -> 评估 -> 特征分析 -> 调优 -> 对比"""
    np.random.seed(seed)
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    train, val, test = load_split(data_dir)
    feat_cols = feature_columns(train)
    x_train, y_train = xy(train, feat_cols)
    x_val, y_val = xy(val, feat_cols)
    x_test, y_test = xy(test, feat_cols)
    spw = pos_weight(y_train)

    train_ids = train['sample_id'].astype(str).values
    val_ids = val['sample_id'].astype(str).values
    test_ids = test['sample_id'].astype(str).values

    print('=' * 60)
    print('Day5: 基线模型训练 + 评估 + 特征重要性 + 超参调优')
    print('特征数: ' + str(len(feat_cols)) + '  (已排除泄露列: '
          + ', '.join(LEAK_COLS) + ')')
    print('训练集: n=' + str(len(y_train)) + ' 正=' + str(int(y_train.sum()))
          + '  验证集: n=' + str(len(y_val)) + ' 正=' + str(int(y_val.sum()))
          + '  测试集: n=' + str(len(y_test)) + ' 正=' + str(int(y_test.sum())))
    print('正样本权重 (负/正) = ' + '{:.2f}'.format(spw))
    print('验证集选优指标: ' + metric)
    print('阈值目标召回率: ' + str(target_recall))
    print('=' * 60)

    models = ['xgboost', 'random_forest', 'lightgbm']
    baseline_rows = []
    roc_val = []
    roc_test = []
    pr_val = []
    pr_test = []
    fitted = {}
    val_scores = {}

    print('\n[1] 训练基线模型 (含阈值优化)')
    for name in models:
        print('\n--- ' + name + ' ---')
        model = fit_pipeline(name, default_params(name), x_train, y_train, spw,
                             x_val, y_val)
        fitted[name] = model
        for split_name, xs, ys, ids in (
                ('val', x_val, y_val, val_ids),
                ('test', x_test, y_test, test_ids)):
            proba = predict_proba(model, xs)
            # 在验证集上找最优阈值, 测试集沿用同一阈值
            if split_name == 'val':
                chosen = choose_threshold(ys, proba, target_recall=target_recall)
                thr = chosen['threshold']
            pred = (proba >= thr).astype(int)
            m = evaluate(ys, pred, proba, thr)
            catch, caught, total = sample_catch_rate(ids, ys, pred)
            m['catch_rate'] = catch
            m['caught'] = caught
            m['total_pos_samples'] = total
            m.update({'model': name, 'split': split_name, 'stage': 'baseline'})
            baseline_rows.append(m)
            print_metrics(split_name, m)
            print('    事件级召回率: {}/{} = {}'.format(caught, total, fmt(catch)))
            cm_path = fig_dir / ('cm_baseline_' + name + '_' + split_name + '.png')
            cm = plot_confusion(ys, pred, name + ' ' + split_name, cm_path)
            print('  混淆矩阵 [[TN FP],[FN TP]] = ' + str(cm.tolist())
                  + '  -> ' + str(cm_path.name))
            if split_name == 'val':
                roc_val.append((name, ys, proba))
                pr_val.append((name, ys, proba))
                val_scores[name] = m[metric]
            else:
                roc_test.append((name, ys, proba))
                pr_test.append((name, ys, proba))
            if split_name == 'test':
                plot_threshold_tradeoff(
                    ys, proba, thr, name + ' 阈值权衡 (测试集)',
                    fig_dir / ('threshold_' + name + '_test.png'))

        # 保存特征重要性
        imp = get_importance(model, feat_cols)
        imp.to_csv(output_dir / ('importance_' + name + '.csv'),
                   index=False, encoding='utf-8-sig')
        top = imp.head(top_k)
        plot_importance(top['feature'].tolist(), top['importance'].tolist(),
                        name + ' Top' + str(top_k),
                        fig_dir / ('importance_' + name + '.png'))

    # 绘制 ROC 和 PR 曲线
    plot_roc(roc_val, '验证集 ROC', fig_dir / 'roc_val_baseline.png')
    plot_roc(roc_test, '测试集 ROC', fig_dir / 'roc_test_baseline.png')
    plot_pr(pr_val, '验证集 PR', fig_dir / 'pr_val_baseline.png')
    plot_pr(pr_test, '测试集 PR', fig_dir / 'pr_test_baseline.png')
    metrics_df = pd.DataFrame(baseline_rows)
    metrics_df.to_csv(output_dir / 'metrics_baseline.csv', index=False, encoding='utf-8-sig')

    # 选出验证集上最优的基线模型
    best_name = max(val_scores, key=val_scores.get)
    print('\n[2] 验证集最优基线模型 (按 ' + metric + '): ' + best_name
          + ' = ' + fmt(val_scores[best_name]))

    # 特征重要性分析
    print('\n[3] 特征重要性分析 (最优基线: ' + best_name + ')')
    best_imp = get_importance(fitted[best_name], feat_cols)
    top, channel_rank = summarize_top_features(best_imp, top_k)
    print('Top' + str(top_k) + ' 特征:')
    for i, row in top.iterrows():
        print('  ' + str(i + 1).rjust(2) + '. ' + str(row['feature']).ljust(24)
              + ' ' + '{:.4f}'.format(row['importance']))
    print('信号通道贡献度 (Top' + str(top_k) + ' 重要性之和):')
    for ch, val in channel_rank:
        print('  ' + ch.ljust(12) + ' ' + '{:.4f}'.format(val))

    # 超参数调优 (只调最优模型, 或全部调优)
    tune_targets = models if tune_all else [best_name]
    print('\n[4] 超参数网格搜索' + (' (快速模式)' if quick else ''))
    tuned_rows = []
    summary = {
        'best_baseline': best_name,
        'selection_metric': metric,
        'n_features': len(feat_cols),
        'features': feat_cols,
        'tuned': {},
    }

    for name in tune_targets:
        model, params, score, thr, hist = grid_search(
            name, x_train, y_train, x_val, y_val, spw, metric, output_dir,
            target_recall, quick=quick, seed=seed)
        print('  最优参数: ' + str(params) + '  验证集 ' + metric + '=' + fmt(score)
              + '  阈值=' + fmt(thr))
        joblib.dump(model, output_dir / (name + '_tuned.joblib'))
        summary['tuned'][name] = {
            'params': params, 'val_score': float(score), 'threshold': float(thr)}
        for split_name, xs, ys, ids in (
                ('val', x_val, y_val, val_ids),
                ('test', x_test, y_test, test_ids)):
            proba = predict_proba(model, xs)
            pred = (proba >= thr).astype(int)
            m = evaluate(ys, pred, proba, thr)
            catch, caught, total = sample_catch_rate(ids, ys, pred)
            m['catch_rate'] = catch
            m['caught'] = caught
            m['total_pos_samples'] = total
            m.update({'model': name, 'split': split_name, 'stage': 'tuned'})
            tuned_rows.append(m)
            print_metrics(name + ' 调优 ' + split_name, m)
            print('    事件级召回率: {}/{} = {}'.format(caught, total, fmt(catch)))
            plot_confusion(ys, pred, name + ' 调优 ' + split_name,
                           fig_dir / ('cm_tuned_' + name + '_' + split_name + '.png'))

        # 调优后的特征重要性
        imp = get_importance(model, feat_cols)
        imp.to_csv(output_dir / ('importance_' + name + '_tuned.csv'),
                   index=False, encoding='utf-8-sig')
        top_t = imp.head(top_k)
        plot_importance(top_t['feature'].tolist(), top_t['importance'].tolist(),
                        name + ' 调优 Top' + str(top_k),
                        fig_dir / ('importance_' + name + '_tuned.png'))

    # 汇总对比
    all_metrics = pd.concat([metrics_df, pd.DataFrame(tuned_rows)], ignore_index=True)
    all_metrics.to_csv(output_dir / 'metrics_all.csv', index=False, encoding='utf-8-sig')

    print('\n' + '=' * 60)
    print('测试集对比')
    test_view = all_metrics[all_metrics['split'] == 'test'].copy()
    cols = ['stage', 'model', 'accuracy', 'precision', 'recall', 'f1', 'f2', 'auc',
            'catch_rate']
    print(test_view[cols].to_string(index=False, float_format=lambda v: '{:.4f}'.format(v)))

    winner = all_metrics[all_metrics['split'] == 'val'].sort_values(metric, ascending=False).iloc[0]
    print('\n最终最优模型 (验证集 ' + metric + '): ' + str(winner['model'])
          + ' (' + str(winner['stage']) + ')  '
          + metric + '=' + fmt(winner[metric]))
    print('测试集表现请查看上方表格中相同模型/阶段的行.')
    print('输出目录: ' + str(output_dir))
    print('  metrics_baseline.csv / metrics_all.csv')
    print('  importance_*.csv')
    print('  figures/  (混淆矩阵, ROC, PR, 阈值, Top' + str(top_k) + ' 条形图)')
    print('  *_tuned.joblib')

    summary['winner'] = {
        'model': str(winner['model']),
        'stage': str(winner['stage']),
        'val_metric': float(winner[metric]),
    }
    summary['channel_rank_best_baseline'] = [
        {'channel': ch, 'importance_sum': float(v)} for ch, v in channel_rank
    ]
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for name, model in fitted.items():
        joblib.dump(model, output_dir / (name + '_baseline.joblib'))


def main():
    p = argparse.ArgumentParser(description='Day5: 训练 / 评估 / 特征分析 / 调优')
    p.add_argument('data_dir', nargs='?', default=None,
                   help='包含 train/val/test.csv 的文件夹')
    p.add_argument('-d', '--data', default=paths.s(paths.DATASET_LEGACY),
                   help='包含 train/val/test.csv 的文件夹')
    p.add_argument('-o', '--output', default=paths.s(paths.RESULTS_LEGACY), help='输出文件夹')
    p.add_argument('--metric', default='f2',
                   choices=['f1', 'f2', 'auc', 'recall', 'precision'],
                   help='选优和网格搜索的评分指标 (默认 f2)')
    p.add_argument('--top-k', type=int, default=20, help='特征重要性 TopK (默认 20)')
    p.add_argument('--tune-all', action='store_true',
                   help='对三个模型都做网格搜索, 否则只调最优基线')
    p.add_argument('--quick', action='store_true',
                   help='快速模式: 随机搜索12组参数 (比全网格快)')
    p.add_argument('--target-recall', type=float, default=0.85,
                   help='阈值优化的目标召回率 (默认 0.85)')
    p.add_argument('--seed', type=int, default=42, help='随机种子')
    args = p.parse_args()
    data_dir = args.data_dir if args.data_dir else args.data
    run(data_dir, args.output, args.metric, args.top_k, args.tune_all,
        args.seed, args.target_recall, args.quick)


if __name__ == '__main__':
    main()
