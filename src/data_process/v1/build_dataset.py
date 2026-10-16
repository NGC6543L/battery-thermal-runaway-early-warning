"""
锂电池针刺热失控实验 - Day7 增强特征工程
================================================
在 Day4 基础特征上新增:
  1. force 高级统计: 峰度, 偏度, 二阶差分标准差 (捕捉力的突变形态)
  2. 时序趋势: 前后半段斜率比, 变化加速度 (二阶导数均值)
  3. 多传感器交叉: force*voltage, tc1-tc2 温差, force*tc_max
  4. 预警形态: voltage 下降斜率 * force 上升斜率

用法 (输出到 dataset_v2, 不覆盖原数据集):
  python build_dataset.py processed_data -m sample_meta.csv -o dataset_v2
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# 统一路径 (src/data_process/paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402


SIGNAL_COLS = ['voltage', 'force', 'tc1', 'tc2', 'tc3', 'tc4']
META_KEEP = ['capacity', 'soc', 'obs_score', 'is_thermal_runaway', 'tr_onset_time']


def _to_float(v):
    """安全转浮点数, 失败返回 None"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if s in ('', 'nan', 'None'):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _safe_kurtosis(x):
    """手动计算峰度 (excess kurtosis), 不依赖 scipy"""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-9:
        return 0.0
    z4 = np.mean((x - m) ** 4) / (s ** 4)
    factor = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * z4 - 3 * (n - 1))
    return float(factor)


def _safe_skew(x):
    """手动计算偏度, 不依赖 scipy"""
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-9:
        return 0.0
    z3 = np.mean((x - m) ** 3) / (s ** 3)
    factor = (np.sqrt(n * (n - 1))) / (n - 2) * z3
    return float(factor)


def extract_channel_features(t, x, prefix):
    """
    单通道时域特征 (Day4 基础 + Day7 增强)
    有效点不足一半时返回 None
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    valid = ~(np.isnan(t) | np.isnan(x))
    if valid.sum() < max(5, int(len(x) * 0.5)):
        return None

    tv, xv = t[valid], x[valid]

    # ---- Day4 基础特征 ----
    feats = {
        prefix + '_mean': float(np.mean(xv)),
        prefix + '_std': float(np.std(xv)),
        prefix + '_max': float(np.max(xv)),
        prefix + '_min': float(np.min(xv)),
        prefix + '_range': float(np.max(xv) - np.min(xv)),
        prefix + '_rms': float(np.sqrt(np.mean(xv ** 2))),
    }

    if len(xv) >= 2 and tv[-1] > tv[0]:
        dt = np.diff(tv)
        dt[dt < 1e-6] = 1e-6
        rate = np.diff(xv) / dt
        feats[prefix + '_slope'] = float((xv[-1] - xv[0]) / (tv[-1] - tv[0]))
        feats[prefix + '_mean_abs_rate'] = float(np.mean(np.abs(rate)))
        feats[prefix + '_max_rate'] = float(np.max(rate))
        feats[prefix + '_min_rate'] = float(np.min(rate))

        # ---- Day7 增强: 高级统计 ----
        feats[prefix + '_kurtosis'] = _safe_kurtosis(xv)
        feats[prefix + '_skew'] = _safe_skew(xv)

        # 二阶差分 = 变化的变化 = 加速度
        if len(rate) >= 2:
            d2 = np.diff(rate)
            feats[prefix + '_acc_std'] = float(np.std(d2))
            feats[prefix + '_acc_mean'] = float(np.mean(d2))
        else:
            feats[prefix + '_acc_std'] = 0.0
            feats[prefix + '_acc_mean'] = 0.0

        # ---- Day7 增强: 前后半段斜率比 ----
        mid = len(xv) // 2
        if mid >= 2 and tv[mid] > tv[0] and tv[-1] > tv[mid]:
            slope1 = (xv[mid] - xv[0]) / (tv[mid] - tv[0])
            slope2 = (xv[-1] - xv[mid]) / (tv[-1] - tv[mid])
            if abs(slope1) < 1e-9:
                feats[prefix + '_slope_ratio'] = 0.0
            else:
                feats[prefix + '_slope_ratio'] = float(slope2 / slope1)
        else:
            feats[prefix + '_slope_ratio'] = 0.0

        # ---- Day7 增强: 最后20%的斜率 (尾部突变) ----
        n_tail = max(3, int(len(xv) * 0.2))
        if n_tail >= 2 and tv[-1] > tv[-n_tail]:
            tail_slope = (xv[-1] - xv[-n_tail]) / (tv[-1] - tv[-n_tail])
            feats[prefix + '_tail_slope'] = float(tail_slope)
        else:
            feats[prefix + '_tail_slope'] = 0.0

    else:
        for name in ['_slope', '_mean_abs_rate', '_max_rate', '_min_rate',
                     '_kurtosis', '_skew', '_acc_std', '_acc_mean',
                     '_slope_ratio', '_tail_slope']:
            feats[prefix + name] = 0.0
    return feats


def extract_cross_features(ch_feats):
    """
    Day7 多传感器交叉特征
    用各通道的均值/斜率做交叉组合
    """
    cross = {}

    # 力*电压: force_mean * voltage_mean (正样本力升压降, 乘积偏负)
    if 'force_mean' in ch_feats and 'voltage_mean' in ch_feats:
        cross['force_x_voltage'] = float(ch_feats['force_mean'] * ch_feats['voltage_mean'])

    # 力斜率 * 电压斜率 (力上升 * 电压下降 = 负值, 典型热失控前兆)
    if 'force_slope' in ch_feats and 'voltage_slope' in ch_feats:
        cross['force_slope_x_volt_slope'] = float(
            ch_feats['force_slope'] * ch_feats['voltage_slope'])

    # 温差: tc1_mean - tc2_mean
    if 'tc1_mean' in ch_feats and 'tc2_mean' in ch_feats:
        cross['tc1_minus_tc2'] = float(ch_feats['tc1_mean'] - ch_feats['tc2_mean'])

    # 温差: tc1_mean - tc3_mean (内层 vs 外层)
    if 'tc1_mean' in ch_feats and 'tc3_mean' in ch_feats:
        cross['tc1_minus_tc3'] = float(ch_feats['tc1_mean'] - ch_feats['tc3_mean'])

    # 力 * 最高温度
    tc_means = [ch_feats.get(c + '_mean') for c in ['tc1', 'tc2', 'tc3', 'tc4']
                if c + '_mean' in ch_feats]
    if tc_means and 'force_mean' in ch_feats:
        cross['force_x_tcmax'] = float(ch_feats['force_mean'] * max(tc_means))

    # 力变化率 * 电压变化率
    if 'force_mean_abs_rate' in ch_feats and 'voltage_mean_abs_rate' in ch_feats:
        cross['force_rate_x_volt_rate'] = float(
            ch_feats['force_mean_abs_rate'] * ch_feats['voltage_mean_abs_rate'])

    # 力尾段斜率 * 电压尾段斜率
    if 'force_tail_slope' in ch_feats and 'voltage_tail_slope' in ch_feats:
        cross['force_tail_x_volt_tail'] = float(
            ch_feats['force_tail_slope'] * ch_feats['voltage_tail_slope'])

    return cross


def extract_window_features(seg):
    """
    窗口内全部通道特征 + 交叉特征
    任一关键通道无效则丢弃该窗口
    """
    t = pd.to_numeric(seg['time'], errors='coerce').values
    row = {}
    n_ok = 0
    for col in SIGNAL_COLS:
        if col not in seg.columns:
            continue
        x = pd.to_numeric(seg[col], errors='coerce').values
        feats = extract_channel_features(t, x, col)
        if feats is None:
            continue
        row.update(feats)
        n_ok += 1
    if n_ok < 3:
        return None

    # Day7: 多传感器交叉特征
    cross = extract_cross_features(row)
    row.update(cross)
    return row


def label_window(t_end, is_tr, onset, pre_start, pre_end):
    """
    返回 0 / 1 / None(丢弃)
    预警区间: [onset - pre_start, onset - pre_end]
    默认 [onset-120, onset-60]
    """
    if not is_tr or onset is None:
        return 0
    warn_lo = onset - pre_start
    warn_hi = onset - pre_end
    if t_end > warn_hi:
        return None
    if t_end >= warn_lo:
        return 1
    return 0


def slide_windows(df, window_s, step_s):
    """滑动窗口生成器: 返回 (start_idx, end_idx, t_start, t_end)"""
    t = pd.to_numeric(df['time'], errors='coerce').values
    if len(t) < 2:
        return
    dt = float(np.nanmedian(np.diff(t)))
    if dt <= 0:
        return
    n_win = int(round(window_s / dt))
    n_step = max(1, int(round(step_s / dt)))
    if n_win < 5 or len(df) < n_win:
        return
    for start in range(0, len(df) - n_win + 1, n_step):
        end = start + n_win
        yield start, end, float(t[start]), float(t[end - 1])


def process_sample(row, data_dir, window_s, step_s, pre_start, pre_end):
    """处理单个实验: 读数据 -> 滑窗 -> 提特征 -> 打标签"""
    sid = str(row['sample_id'])
    if sid.startswith('_'):
        return [], 'skip_meta_file'
    data_file = Path(data_dir) / (sid + '.csv')
    if not data_file.exists():
        fp = str(row.get('file_path', '')).strip()
        data_file = Path(fp) if fp else data_file
    if not data_file.exists():
        return [], 'no_file'

    is_tr = int(row.get('is_thermal_runaway', 0) or 0)
    onset = _to_float(row.get('tr_onset_time'))
    if is_tr and onset is None:
        return [], 'tr_no_onset'

    try:
        df = pd.read_csv(data_file, encoding='utf-8-sig')
    except Exception as e:
        return [], 'read_error_' + str(e)[:40]
    df.columns = df.columns.str.strip()
    if 'time' not in df.columns:
        return [], 'no_time'

    records = []
    n_drop = 0
    for i, (s, e, t0, t1) in enumerate(slide_windows(df, window_s, step_s)):
        y = label_window(t1, is_tr == 1, onset, pre_start, pre_end)
        if y is None:
            n_drop += 1
            continue
        feats = extract_window_features(df.iloc[s:e])
        if feats is None:
            n_drop += 1
            continue
        rec = {
            'sample_id': sid,
            'window_id': i,
            't_start': round(t0, 3),
            't_end': round(t1, 3),
            'label': int(y),
        }
        for k in META_KEEP:
            if k in row.index:
                rec[k] = row[k]
        rec.update(feats)
        records.append(rec)
    if not records:
        return [], 'no_valid_window drop=' + str(n_drop)
    return records, 'ok drop=' + str(n_drop)


def split_by_sample(all_df, train_ratio, val_ratio, seed):
    """按 sample_id 分层划分, 保证同一实验只出现在一个集合"""
    samples = all_df.groupby('sample_id').agg(
        has_pos=('label', 'max'),
        n=('label', 'size'),
    ).reset_index()

    rng = np.random.RandomState(seed)
    groups = {}
    for flag in (0, 1):
        ids = samples.loc[samples['has_pos'] == flag, 'sample_id'].tolist()
        rng.shuffle(ids)
        groups[flag] = ids

    train_ids, val_ids, test_ids = [], [], []
    for ids in groups.values():
        n = len(ids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
            extra = n_train + n_val + n_test - n
            n_train -= extra
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)
    train = all_df[all_df['sample_id'].isin(train_ids)].copy()
    val = all_df[all_df['sample_id'].isin(val_ids)].copy()
    test = all_df[all_df['sample_id'].isin(test_ids)].copy()
    return train, val, test


def print_split_stats(name, df):
    n = len(df)
    n1 = int((df['label'] == 1).sum()) if n else 0
    n0 = n - n1
    ns = df['sample_id'].nunique() if n else 0
    pos_pct = (n1 / n * 100) if n else 0
    print('  ' + name + ': windows=' + str(n) + '  samples=' + str(ns)
          + '  label0=' + str(n0) + '  label1=' + str(n1)
          + '  pos=' + str(round(pos_pct, 1)) + '%')


def build_dataset(data_dir, meta_path, output_dir, window_s, step_s,
                  pre_start, pre_end, train_ratio, val_ratio, seed):
    """主函数: 读元数据 -> 批量处理 -> 划分 -> 输出"""
    data_dir = Path(data_dir)
    meta_path = Path(meta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print('Error: data folder does not exist: ' + str(data_dir))
        return
    if not meta_path.exists():
        print('Error: meta file does not exist: ' + str(meta_path))
        return

    meta = pd.read_csv(meta_path, encoding='utf-8-sig')
    meta.columns = meta.columns.str.strip()
    if 'sample_id' not in meta.columns:
        print('Error: sample_id column not found')
        return

    print('Day7: 增强特征工程')
    print('Meta records: ' + str(len(meta)))
    print('Window=' + str(window_s) + 's  step=' + str(step_s) + 's'
          + '  warning=[' + str(pre_start) + 's,' + str(pre_end) + 's] before onset')
    print('Split train:val:test = '
          + str(train_ratio) + ':' + str(val_ratio) + ':' + str(round(1 - train_ratio - val_ratio, 2)))
    print()

    all_rows = []
    ok = skip = 0
    for idx, row in meta.iterrows():
        recs, status = process_sample(row, data_dir, window_s, step_s, pre_start, pre_end)
        sid = str(row['sample_id'])
        if recs:
            ok += 1
            n1 = sum(1 for r in recs if r['label'] == 1)
            print('[' + str(idx + 1) + '/' + str(len(meta)) + '] OK ' + sid
                  + '  windows=' + str(len(recs)) + '  pos=' + str(n1))
            all_rows.extend(recs)
        else:
            skip += 1
            print('[' + str(idx + 1) + '/' + str(len(meta)) + '] SKIP ' + sid + '  ' + status)

    if not all_rows:
        print('No windows generated')
        return

    all_df = pd.DataFrame(all_rows)
    feat_cols = [c for c in all_df.columns
                 if c not in ['sample_id', 'window_id', 't_start', 't_end', 'label'] + META_KEEP]
    front = ['sample_id', 'window_id', 't_start', 't_end'] + META_KEEP + feat_cols + ['label']
    front = [c for c in front if c in all_df.columns]
    all_df = all_df[front]

    print()
    print('=' * 50)
    print('Total windows: ' + str(len(all_df))
          + '  samples used: ' + str(ok) + '  skipped: ' + str(skip))
    print('Feature count: ' + str(len(feat_cols)))
    print_split_stats('ALL', all_df)

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < -1e-9:
        print('Error: train+val ratios exceed 1')
        return
    train, val, test = split_by_sample(all_df, train_ratio, val_ratio, seed)
    print_split_stats('train', train)
    print_split_stats('val', val)
    print_split_stats('test', test)

    train.to_csv(output_dir / 'train.csv', index=False, encoding='utf-8-sig')
    val.to_csv(output_dir / 'val.csv', index=False, encoding='utf-8-sig')
    test.to_csv(output_dir / 'test.csv', index=False, encoding='utf-8-sig')
    all_df.to_csv(output_dir / 'all_windows.csv', index=False, encoding='utf-8-sig')

    print()
    print('Output:')
    print('  ' + str(output_dir / 'train.csv'))
    print('  ' + str(output_dir / 'val.csv'))
    print('  ' + str(output_dir / 'test.csv'))
    print('  ' + str(output_dir / 'all_windows.csv') + '  (full table)')


def main():
    p = argparse.ArgumentParser(description='Day7: 增强特征工程 (高级统计 + 交叉特征)')
    p.add_argument('data_dir', nargs='?', default=paths.s(paths.PROCESSED_DATA),
                   help='Preprocessed CSV folder')
    p.add_argument('-m', '--meta', default=paths.s(paths.SAMPLE_META), help='Sample meta CSV')
    p.add_argument('-o', '--output', default=paths.s(paths.DATASET_LEGACY113), help='Output folder')
    p.add_argument('--window', type=float, default=30.0, help='Window length seconds')
    p.add_argument('--step', type=float, default=10.0, help='Step length seconds')
    p.add_argument('--pre-start', type=float, default=120.0,
                   help='Warning interval start: seconds before onset')
    p.add_argument('--pre-end', type=float, default=60.0,
                   help='Warning interval end: seconds before onset (lead time)')
    p.add_argument('--train-ratio', type=float, default=0.7)
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    build_dataset(args.data_dir, args.meta, args.output, args.window, args.step,
                  args.pre_start, args.pre_end, args.train_ratio, args.val_ratio, args.seed)


if __name__ == '__main__':
    main()
