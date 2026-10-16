"""
Day6: 模型优化 v2 (双源保守阈值版)
================================================
v1 教训 (上一轮运行实测):
  - 只用训练集 OOF 概率选阈值 (xgb 0.78 / rf 0.76 / lgbm 0.81) 后,
    测试集精确率上升 (0.78) 但召回率崩塌 (0.375, 事件召回 2/4)
  - 根因: 按实验划分的训练/验证/测试存在 "实验级分布偏移",
    训练集实验的正样本概率整体偏高, 绝对概率阈值无法跨实验组迁移
  - 证据: 同一个 xgboost 模型, 在 val 上达到 85% 召回只需阈值 0.14,
    而在训练集 OOF 上达到 85% 召回需要 0.78
  - 注意测试集 AUC 仍高达 0.99+, 说明排序能力没问题, 只是概率绝对值不迁移

v2 修复: 双源保守阈值
  - 同时在 OOF 和验证集上按目标召回率各选一个阈值, 取两者较小值
  - 含义: 任一校准源显示 "会漏报超标" 就下调阈值, 召回优先
  - 打印两个阈值的差值, 作为分布偏移诊断
其余: RF 网格搜索 (OOF 评分选参数) + 软投票集成, 与 v1 相同

依赖: 同目录下的 train_model.py (Day5 脚本, 直接复用其函数)
用法:
  python optimize_model.py dataset -o results_opt --target-recall 0.85
  python optimize_model.py dataset -o results_opt --quick   (快速模式, 搜索组数减半)
"""
import argparse
import json
import random
import sys
from itertools import product
from pathlib import Path

# 统一路径 (src/data_process/paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 复用 Day5 train_model.py 中的函数, 避免重复造轮子
from train_model import (
    load_split, feature_columns, xy, pos_weight,
    default_params, fit_pipeline, predict_proba,
    choose_threshold, evaluate, sample_catch_rate, print_metrics,
    plot_confusion, plot_pr, plot_importance, get_importance,
)


def cv_oof_probs(name, params, x, y, spw, n_splits=5, seed=42):
    """
    5 折交叉验证产生 OOF (out-of-fold) 概率:
    每个样本的概率都来自一个 "训练时没见过它" 的模型.
    注意: OOF 只反映训练集实验内部的泛化, 不包含跨实验组偏移.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for tr_idx, ho_idx in skf.split(x, y):
        # 每折用相同超参数训练 (不做早停, 保证各折行为一致)
        model = fit_pipeline(name, params, x.iloc[tr_idx], y[tr_idx], spw)
        oof[ho_idx] = predict_proba(model, x.iloc[ho_idx])
    return oof


def rf_search_space():
    """RandomForest 的精简搜索空间 (12 组, 全量跑也不算慢)"""
    return {
        'n_estimators': [200, 400],
        'max_depth': [10, 16, None],
        'min_samples_leaf': [2, 8],
        'max_features': ['sqrt'],
    }


def rf_grid_search(x_train, y_train, spw, metric, target_recall, quick,
                   n_splits, seed, out_dir):
    """RF 网格搜索: 每组参数都用 OOF 概率选阈值并评分, 选出最优参数组合"""
    space = rf_search_space()
    keys = list(space.keys())
    combos = [dict(zip(keys, v)) for v in product(*[space[k] for k in keys])]
    if quick:
        # 快速模式: 随机抽取 6 组
        rng = random.Random(seed)
        combos = rng.sample(combos, min(len(combos), 6))
    print('RF 网格搜索: {} 组, 每组跑 {} 折交叉验证, 评分指标={} (OOF 召回>={} 优化)'.format(
        len(combos), n_splits, metric, target_recall))
    rows = []
    best_score = -1.0
    best_params = None
    best_thr = 0.5
    best_oof = None
    for i, params in enumerate(combos, 1):
        oof = cv_oof_probs('random_forest', params, x_train, y_train, spw,
                           n_splits, seed)
        chosen = choose_threshold(y_train, oof, target_recall=target_recall)
        thr = chosen['threshold']
        pred = (oof >= thr).astype(int)
        m = evaluate(y_train, pred, oof, thr)
        score = m.get(metric, m['f2'])
        rows.append(dict(params, **m, score=score))
        mark = ''
        if score > best_score:
            best_score = score
            best_params = params
            best_thr = thr
            best_oof = oof
            mark = '  *最优'
        print('  [{}/{}] {}  {}={:.4f}  召回={:.4f}  阈值={:.4f}{}'.format(
            i, len(combos), params, metric, score, m['recall'], thr, mark))
    pd.DataFrame(rows).sort_values('score', ascending=False).to_csv(
        out_dir / 'grid_random_forest_cv.csv', index=False, encoding='utf-8-sig')
    return best_params, best_thr, best_score, best_oof


def run(data_dir, output_dir, target_recall, n_splits, quick, seed):
    """Day6 v2 主流程: OOF 阈值 -> RF 调优 -> 训练 -> 集成 -> 双源阈值 -> 评估"""
    np.random.seed(seed)
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # 加载数据 (与 Day5 完全一致的划分, 保证结果可比)
    train, val, test = load_split(data_dir)
    feat_cols = feature_columns(train)
    x_train, y_train = xy(train, feat_cols)
    x_val, y_val = xy(val, feat_cols)
    x_test, y_test = xy(test, feat_cols)
    spw = pos_weight(y_train)
    val_ids = val['sample_id'].astype(str).values
    test_ids = test['sample_id'].astype(str).values

    print('=' * 60)
    print('Day6 v2: 模型优化 (双源保守阈值 + RF调优 + 软投票集成)')
    print('特征数: {}'.format(len(feat_cols)))
    print('训练集: n={} 正={}  验证集: n={} 正={}  测试集: n={} 正={}'.format(
        len(y_train), int(y_train.sum()),
        len(y_val), int(y_val.sum()),
        len(y_test), int(y_test.sum())))
    print('阈值目标召回率: {}  交叉验证折数: {}'.format(target_recall, n_splits))
    print('=' * 60)

    results = []  # 汇总所有评估行, 最后写 CSV

    def eval_and_record(split_name, model_name, stage, proba, ys, ids, thr,
                        verbose=True):
        """按给定阈值评估, 记录一行结果; verbose 控制是否打印"""
        pred = (proba >= thr).astype(int)
        m = evaluate(ys, pred, proba, thr)
        catch, caught, total = sample_catch_rate(ids, ys, pred)
        m.update({'model': model_name, 'stage': stage, 'split': split_name,
                  'catch_rate': catch, 'caught': caught, 'total_pos_samples': total})
        results.append(m)
        if verbose:
            print_metrics('{} {} ({})'.format(model_name, stage, split_name), m)
            print('    事件级召回率: {}/{} = {:.4f}'.format(caught, total, catch))
        return m

    # ---------- [1] 基线模型的 OOF 概率与阈值 ----------
    print('\n[1] 用 {} 折交叉验证计算基线模型 OOF 概率'.format(n_splits))
    oof_store = {}
    for name in ('xgboost', 'random_forest', 'lightgbm'):
        oof = cv_oof_probs(name, default_params(name), x_train, y_train, spw,
                           n_splits, seed)
        oof_store[name] = oof
        thr_oof = choose_threshold(y_train, oof, target_recall=target_recall)['threshold']
        pred = (oof >= thr_oof).astype(int)
        m = evaluate(y_train, pred, oof, thr_oof)
        print('  {}: OOF 阈值={:.4f}'.format(name, thr_oof))
        print_metrics('  ' + name + ' OOF', m)

    # ---------- [2] RandomForest 网格搜索 (OOF 评分) ----------
    print('\n[2] RandomForest 网格搜索调优')
    rf_best_params, rf_best_thr, rf_best_score, rf_best_oof = rf_grid_search(
        x_train, y_train, spw, 'f2', target_recall, quick, n_splits, seed, output_dir)
    print('  RF 最优参数: {}'.format(rf_best_params))
    print('  OOF f2={:.4f}  OOF 阈值={:.4f}'.format(rf_best_score, rf_best_thr))
    oof_store['random_forest_tuned'] = rf_best_oof

    # ---------- [3] 在全量训练集上训练最终模型 ----------
    print('\n[3] 在全量训练集上训练最终模型')
    models = {}
    fit_plan = [
        ('xgboost', 'xgboost', default_params('xgboost')),
        ('random_forest', 'random_forest', default_params('random_forest')),
        ('lightgbm', 'lightgbm', default_params('lightgbm')),
        ('random_forest_tuned', 'random_forest', rf_best_params),
    ]
    for key, base_name, params in fit_plan:
        models[key] = fit_pipeline(base_name, params, x_train, y_train, spw)
        joblib.dump(models[key], output_dir / (key + '.joblib'))
        print('  已训练并保存: ' + key)

    # 集成的 OOF 概率 = 三个基线模型 OOF 概率的平均
    oof_store['ensemble'] = (oof_store['xgboost'] + oof_store['random_forest']
                             + oof_store['lightgbm']) / 3.0

    # 候选名单: (模型key, 阶段名)
    candidates = [
        ('xgboost', 'baseline'),
        ('random_forest', 'baseline'),
        ('lightgbm', 'baseline'),
        ('random_forest_tuned', 'tuned'),
        ('ensemble', 'soft_voting'),
    ]

    # ---------- [4] 双源保守阈值 ----------
    print('\n[4] 双源保守阈值 = min(OOF阈值, 验证集阈值)')
    print('  阈值差值大 = 实验级分布偏移严重 (训练集正样本概率偏高)')
    print('  {:<22} {:>10} {:>10} {:>12}'.format('模型', 'OOF阈值', 'val阈值', '采用阈值'))
    thr_store = {}
    # 验证集概率 (全量训练的模型给出)
    val_probas = {key: predict_proba(model, x_val) for key, model in models.items()}
    val_probas['ensemble'] = (val_probas['xgboost'] + val_probas['random_forest']
                              + val_probas['lightgbm']) / 3.0
    for name, _ in candidates:
        thr_oof = choose_threshold(y_train, oof_store[name],
                                   target_recall=target_recall)['threshold']
        thr_val = choose_threshold(y_val, val_probas[name],
                                   target_recall=target_recall)['threshold']
        thr_safe = min(thr_oof, thr_val)
        thr_store[name] = {'oof': thr_oof, 'val': thr_val, 'safe': thr_safe}
        print('  {:<22} {:>10.4f} {:>10.4f} {:>12.4f}'.format(
            name, thr_oof, thr_val, thr_safe))

    # ---------- [5] 验证/测试集统一评估 ----------
    print('\n[5] 验证集 / 测试集统一评估')
    print('  每个模型评两种策略: *_oofthr = 纯OOF阈值(v1做法, 留作对比), 其余 = 双源保守阈值')
    pr_curves = {'val': [], 'test': []}
    for split_name, xs, ys, ids in (('val', x_val, y_val, val_ids),
                                    ('test', x_test, y_test, test_ids)):
        print('\n--- {} ---'.format(split_name))
        # 先算一次所有模型的概率, 避免重复预测
        probas = {key: predict_proba(model, xs) for key, model in models.items()}
        probas['ensemble'] = (probas['xgboost'] + probas['random_forest']
                              + probas['lightgbm']) / 3.0
        for name, stage in candidates:
            thr = thr_store[name]
            # 策略A: 纯 OOF 阈值 (v1 做法, 只记录不细打印)
            eval_and_record(split_name, name, stage + '_oofthr',
                            probas[name], ys, ids, thr['oof'], verbose=False)
            # 策略B: 双源保守阈值 (v2 采用)
            m = eval_and_record(split_name, name, stage,
                                probas[name], ys, ids, thr['safe'], verbose=True)
            if name == 'ensemble':
                plot_confusion(ys, (probas[name] >= thr['safe']).astype(int),
                               'ensemble ' + split_name,
                               fig_dir / ('cm_ensemble_' + split_name + '.png'))
        # PR 曲线: 集成 vs 单模型
        pr_curves[split_name] = [
            ('ensemble', ys, probas['ensemble']),
            ('xgboost', ys, probas['xgboost']),
            ('random_forest_tuned', ys, probas['random_forest_tuned']),
        ]

    plot_pr(pr_curves['val'], '验证集 PR (Day6)', fig_dir / 'pr_val_day6.png')
    plot_pr(pr_curves['test'], '测试集 PR (Day6)', fig_dir / 'pr_test_day6.png')

    # 调优后 RF 的特征重要性 Top20
    imp = get_importance(models['random_forest_tuned'], feat_cols)
    imp.to_csv(output_dir / 'importance_random_forest_tuned.csv',
               index=False, encoding='utf-8-sig')
    top = imp.head(20)
    plot_importance(top['feature'].tolist(), top['importance'].tolist(),
                    'RF调优 Top20', fig_dir / 'importance_rf_tuned.png')

    # ---------- 汇总输出 ----------
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'metrics_all_day6.csv', index=False, encoding='utf-8-sig')

    print('\n' + '=' * 60)
    print('测试集对比 (Day6): 两种阈值策略')
    cols = ['stage', 'model', 'threshold', 'accuracy', 'precision',
            'recall', 'f1', 'f2', 'auc', 'catch_rate']
    test_view = df[df['split'] == 'test'].copy()
    print(test_view[cols].to_string(index=False,
                                    float_format=lambda v: '{:.4f}'.format(v)))

    # 赢家只在双源保守阈值的行里选 (排除 *_oofthr 对照行)
    safe_view = test_view[~test_view['stage'].str.endswith('_oofthr')]
    winner = safe_view.sort_values('f2', ascending=False).iloc[0]
    print('\n测试集 F2 最优 (双源保守阈值): {} ({})  F2={:.4f}  召回={:.4f}  事件召回={:.4f}'.format(
        winner['model'], winner['stage'], winner['f2'],
        winner['recall'], winner['catch_rate']))
    print('参考: Day5 最好的 RF 基线测试集 F2=0.6897, 事件召回=1.0000')
    print('输出目录: ' + str(output_dir))
    print('  metrics_all_day6.csv  grid_random_forest_cv.csv')
    print('  importance_random_forest_tuned.csv')
    print('  figures/  (集成混淆矩阵, PR曲线, RF调优Top20)')
    print('  *.joblib  (xgboost / random_forest / lightgbm / random_forest_tuned)')

    summary = {
        'target_recall': target_recall,
        'n_splits': n_splits,
        'thresholds': {k: {kk: float(vv) for kk, vv in v.items()}
                       for k, v in thr_store.items()},
        'rf_best_params': rf_best_params,
        'rf_best_oof_score': float(rf_best_score),
        'test_winner': {
            'model': str(winner['model']),
            'stage': str(winner['stage']),
            'f2': float(winner['f2']),
            'recall': float(winner['recall']),
            'catch_rate': float(winner['catch_rate']),
        },
    }
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description='Day6: 双源保守阈值 + RF调优 + 软投票集成')
    p.add_argument('data_dir', nargs='?', default=None,
                   help='包含 train/val/test.csv 的文件夹')
    p.add_argument('-d', '--data', default=paths.s(paths.DATASET_LEGACY),
                   help='包含 train/val/test.csv 的文件夹')
    p.add_argument('-o', '--output', default=paths.s(paths.RESULTS_TUNED), help='输出文件夹')
    p.add_argument('--target-recall', type=float, default=0.85,
                   help='阈值优化的目标召回率 (默认 0.85)')
    p.add_argument('--n-splits', type=int, default=5,
                   help='交叉验证折数 (默认 5)')
    p.add_argument('--quick', action='store_true',
                   help='快速模式: RF 搜索组数减到 6 组')
    p.add_argument('--seed', type=int, default=42, help='随机种子')
    args = p.parse_args()
    data_dir = args.data_dir if args.data_dir else args.data
    run(data_dir, args.output, args.target_recall, args.n_splits,
        args.quick, args.seed)


if __name__ == '__main__':
    main()
