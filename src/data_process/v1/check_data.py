import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 统一路径 (src/data_process/paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

def check_file(file_path):
    df = pd.read_csv(file_path)
    issues = []
    info = {'sample': file_path.stem, 'points': len(df)}

    expected = ['time', 'voltage', 'force', 'tc1', 'tc2', 'tc3', 'tc4']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        issues.append('缺少列: ' + str(missing))

    if 'time' in df.columns:
        t = df['time'].values
        info['time_start'] = round(t[0], 2)
        info['time_end'] = round(t[-1], 2)
        info['duration'] = round(t[-1] - t[0], 1)
        dt = np.diff(t)
        if np.any(dt <= 0):
            issues.append('时间回跳/重复')
        avg_dt = np.nanmean(dt)
        info['sample_interval'] = round(avg_dt, 3)
        if abs(avg_dt - 0.2) > 0.05:
            issues.append('采样间隔异常: ' + str(round(avg_dt, 3)) + 's')

    if 'voltage' in df.columns:
        v = df['voltage'].dropna()
        info['voltage_nan_pct'] = round(df['voltage'].isna().sum() / len(df) * 100, 1)
        if len(v) > 0:
            info['voltage_min'] = round(v.min(), 3)
            info['voltage_max'] = round(v.max(), 3)
            if v.max() > 6 or v.min() < -0.5:
                issues.append('电压异常: ' + str(round(v.min(), 2)) + '~' + str(round(v.max(), 2)) + 'V')

    for tc in ['tc1', 'tc2', 'tc3', 'tc4']:
        if tc in df.columns:
            temp = df[tc].dropna()
            nan_pct = df[tc].isna().sum() / len(df) * 100
            info[tc + '_nan_pct'] = round(nan_pct, 1)
            if len(temp) > 0:
                info[tc + '_max'] = round(temp.max(), 1)
                if temp.max() > 1500 or temp.min() < -30:
                    issues.append(tc + '温度异常: ' + str(round(temp.min(), 1)) + '~' + str(round(temp.max(), 1)) + 'C')

    if 'force' in df.columns:
        f = df['force'].dropna()
        info['force_nan_pct'] = round(df['force'].isna().sum() / len(df) * 100, 1)
        if len(f) > 0:
            info['force_max'] = round(f.max(), 2)

    info['issues'] = '; '.join(issues) if issues else ''
    return info

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else paths.s(paths.PROCESSED_DATA)
    d = Path(data_dir)
    if not d.exists():
        print('目录不存在: ' + str(d))
        return

    files = sorted(list(d.glob('*.csv')))
    files = [f for f in files if not f.name.startswith('_')]
    print('找到 ' + str(len(files)) + ' 个数据文件，开始校验...\n')

    results = []
    for i, fp in enumerate(files, 1):
        try:
            info = check_file(fp)
            results.append(info)
            flag = 'X' if info['issues'] else 'OK'
            print('[' + str(i) + '/' + str(len(files)) + '] ' + flag + ' ' + info['sample'] + ' (' + str(info['points']) + '点)')
            if info['issues']:
                print('        问题: ' + info['issues'])
        except Exception as e:
            print('[' + str(i) + '/' + str(len(files)) + '] X ' + fp.name + ' 读取失败: ' + str(e))

    df = pd.DataFrame(results)

    bad = df['issues'].fillna('').str.len().gt(0).sum()
    print('\n' + '=' * 50)
    print('校验完成: 正常' + str(len(df) - bad) + ' / 有问题' + str(bad) + ' / 共' + str(len(df)))
    print('平均数据点数: ' + str(round(df['points'].mean())))
    if 'duration' in df.columns:
        print('平均时长: ' + str(round(df['duration'].mean(), 1)) + 's')

    report_path = d / '_quality_report.csv'
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print('\n详细报告: ' + str(report_path))

if __name__ == '__main__':
    main()