"""
Day7: 特征选择 + 增强特征训练对比
================================================
功能:
  1. 加载 Day7 增强数据集 (dataset_v2)
  2. 用 SelectFromModel (基于 RF 特征重要性) 自动选择有效特征
  3. 用选出的特征训练 RF baseline (Day5/6 最优模型)
  4. 用验证集阈值 (Day5/6 证明有效) 评估
  5. 输出对比: Day5 (62特征) vs Day7 (选中特征)

依赖: train_model.py 的函数
用法:
  python train_v2.py dataset_v2 -o results_v2 --target-recall 0.85
"""
import argparse
import json
import sys
from pathlib import Path

# 统一路径 (src/data_process/paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer

# 复用 Day5 函数
from train_model import (
    load_split, feature_columns, xy, pos_weight,
    default_params, fit_pipeline, predict_proba,
    choose_threshold, evaluate, sample_catch_rate, print_metrics,
    plot_confusion, plot_pr, plot_importance, get_importance,
)


def run(data_dir, output_dir, target_recall, seed):
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # 加载增强数据集
    train, val, test = load_split(data_dir)
    feat_cols = feature_columns(train)
    x_train, y_train = xy(train, feat_cols)
    x_val, y_val = xy(val, feat_cols)
    x_test, y_test = xy(test, feat_cols)
    spw = pos_weight(y_train)
    val_ids = val['sample_id'].astype(str).values
    test_ids = test['sample_id'].astype(str).values

    print('=' * 60)
    print('Day7: 特征选择 + 增强特征训练')
    print('增强特征数: {}'.format(len(feat_cols)))
    print('训练集: n={} 正={}  验证集: n={} 正={}  测试集: n={} 正={}'.format(
        len(y_train), int(y_train.sum()),
        len(y_val), int(y_val.sum()),
        len(y_test), int(y_test.sum())))
    print('正样本权重 (负/正) = {:.2f}'.format(spw))
    print('=' * 60)

    # ---------- [1] 特征选择 ----------
    print('\n[1] 用 SelectFromModel 选择有效特征')
    # 先填补缺失值
    imputer = SimpleImputer(strategy='median')
    x_train_imp = pd.DataFrame(imputer.fit_transform(x_train),
                               columns=feat_cols, index=x_train.index)

    # 用 RF 特征重要性做选择
    selector_rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        max_features='sqrt', random_state=seed,
        class_weight={0: 1, 1: spw}, n_jobs=-1)
    selector_rf.fit(x_train_imp, y_train)

    # SelectFromModel: 选重要性 > 均值的特征
    selector = SelectFromModel(selector_rf, prefit=True, threshold='median')
    selected_mask = selector.get_support()
    selected_cols = [c for c, s in zip(feat_cols, selected_mask) if s]
    print('  选中特征: {} / {} 个'.format(len(selected_cols), len(feat_cols)))
    print('  选中特征列表:')
    imp_all = pd.DataFrame({
        'feature': feat_cols,
        'importance': selector_rf.feature_importances_,
        'selected': selected_mask,
    }).sort_values('importance', ascending=False)
    for _, r in imp_all.head(30).iterrows():
        mark = ' *' if r['selected'] else ''
        print('    {:<30} {:.4f}{}'.format(r['feature'], r['importance'], mark))
    imp_all.to_csv(output_dir / 'importance_all_v2.csv',
                   index=False, encoding='utf-8-sig')

    # 用选中特征重新切分
    x_train_sel = x_train[selected_cols].copy()
    x_val_sel = x_val[selected_cols].copy()
    x_test_sel = x_test[selected_cols].copy()

    # ---------- [2] 训练 RF baseline (全特征 vs 选中特征) ----------
    print('\n[2] 训练 RandomForest (全特征 vs 选中特征)')

    # 全特征版
    print('  [全特征] 训练中... ({}特征)'.format(len(feat_cols)))
    model_full = fit_pipeline('random_forest', default_params('random_forest'),
                               x_train, y_train, spw)
    proba_val_full = predict_proba(model_full, x_val)
    proba_test_full = predict_proba(model_full, x_test)
    thr_val_full = choose_threshold(y_val, proba_val_full,
                                     target_recall=target_recall)['threshold']

    # 选中特征版
    print('  [选中特征] 训练中... ({}特征)'.format(len(selected_cols)))
    model_sel = fit_pipeline('random_forest', default_params('random_forest'),
                              x_train_sel, y_train, spw)
    proba_val_sel = predict_proba(model_sel, x_val_sel)
    proba_test_sel = predict_proba(model_sel, x_test_sel)
    thr_val_sel = choose_threshold(y_val, proba_val_sel,
                                    target_recall=target_recall)['threshold']

    # 保存模型
    joblib.dump(model_full, output_dir / 'rf_full_v2.joblib')
    joblib.dump(model_sel, output_dir / 'rf_selected_v2.joblib')

    # ---------- [3] 评估 ----------
    print('\n[3] 验证集 / 测试集评估')
    results = []

    def eval_one(split_name, tag, proba, thr):
        ys = y_val if split_name == 'val' else y_test
        ids = val_ids if split_name == 'val' else test_ids
        pred = (proba >= thr).astype(int)
        m = evaluate(ys, pred, proba, thr)
        catch, caught, total = sample_catch_rate(ids, ys, pred)
        m.update({'tag': tag, 'split': split_name, 'catch_rate': catch})
        results.append(m)
        print_metrics(tag + ' (' + split_name + ')', m)
        print('    事件级召回率: {}/{} = {:.4f}'.format(caught, total, catch))
        return m

    for split_name in ('val', 'test'):
        print('\n--- {} ---'.format(split_name))
        eval_one(split_name, 'full_features', 
                 proba_val_full if split_name == 'val' else proba_test_full,
                 thr_val_full)
        eval_one(split_name, 'selected_features',
                 proba_val_sel if split_name == 'val' else proba_test_sel,
                 thr_val_sel)

    # 画图
    for split_name, proba, thr in [
        ('val', proba_val_sel, thr_val_sel),
        ('test', proba_test_sel, thr_val_sel)
    ]:
        ys = y_val if split_name == 'val' else y_test
        pred = (proba >= thr).astype(int)
        plot_confusion(ys, pred, 'RF selected ' + split_name,
                        fig_dir / ('cm_rf_selected_' + split_name + '.png'))

    plot_pr([
        ('selected', y_val, proba_val_sel),
        ('full', y_val, proba_val_full),
    ], '验证集 PR (Day7)', fig_dir / 'pr_val_v2.png')
    plot_pr([
        ('selected', y_test, proba_test_sel),
        ('full', y_test, proba_test_full),
    ], '测试集 PR (Day7)', fig_dir / 'pr_test_v2.png')

    # 选中特征版的特征重要性
    imp_sel = get_importance(model_sel, selected_cols)
    imp_sel.to_csv(output_dir / 'importance_selected_v2.csv',
                   index=False, encoding='utf-8-sig')
    top = imp_sel.head(20)
    plot_importance(top['feature'].tolist(), top['importance'].tolist(),
                    'RF选中特征 Top20', fig_dir / 'importance_selected_v2.png')

    # ---------- 汇总输出 ----------
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'metrics_v2.csv', index=False, encoding='utf-8-sig')

    print('\n' + '=' * 60)
    print('Day7 测试集对比')
    cols = ['tag', 'threshold', 'accuracy', 'precision', 'recall',
            'f1', 'f2', 'auc', 'catch_rate']
    test_view = df[df['split'] == 'test'].copy()
    print(test_view[cols].to_string(index=False,
                                    float_format=lambda v: '{:.4f}'.format(v)))
    print('\n参考 Day5/6 最优 (RF baseline, 62特征):')
    print('  F2=0.6897  精确率=0.3077  召回率=1.0000  事件召回=1.0000')

    summary = {
        'total_features_v2': len(feat_cols),
        'selected_features': len(selected_cols),
        'selected_cols': selected_cols,
        'thresholds': {
            'full': float(thr_val_full),
            'selected': float(thr_val_sel),
        },
    }
    with open(output_dir / 'summary_v2.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('\n输出目录: ' + str(output_dir))


def main():
    p = argparse.ArgumentParser(description='Day7: 特征选择 + 增强特征训练')
    p.add_argument('data_dir', nargs='?', default=paths.s(paths.DATASET_LEGACY113),
                   help='增强数据集文件夹')
    p.add_argument('-o', '--output', default=paths.s(paths.RESULTS_FEATSEL), help='输出文件夹')
    p.add_argument('--target-recall', type=float, default=0.85,
                   help='阈值优化的目标召回率')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    run(args.data_dir, args.output, args.target_recall, args.seed)


if __name__ == '__main__':
    main()
