"""
 热失控爆发点人工校验工具
输入样本ID, 自动画出温度+电压曲线, 并用红色竖线标注检测到的爆发点
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_sample(data_dir, meta_path, sample_id):
    data_dir = Path(data_dir)
    data_file = data_dir / (sample_id + '.csv')

    if not data_file.exists():
        print('错误: 找不到数据文件 ' + str(data_file))
        return

    meta_df = pd.read_csv(meta_path)
    row = meta_df[meta_df['sample_id'].astype(str) == sample_id]
    if row.empty:
        print('错误: 在元信息台账中找不到 ' + sample_id)
        return

    row = row.iloc[0]
    onset_time = row.get('tr_onset_time', None)
    is_tr = int(row.get('is_thermal_runaway', 0))
    obs_score = str(row.get('obs_score', ''))
    capacity = str(row.get('capacity', ''))
    soc = str(row.get('soc', ''))

    df = pd.read_csv(data_file)
    t = df['time'].values

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    title = sample_id
    if capacity:
        title += '  Cap=' + capacity + 'mAh'
    if soc:
        title += '  SOC=' + soc + '%'
    title += '  Status=' + ('TR' if is_tr else 'Normal')
    if obs_score:
        title += '  Obs=' + obs_score
    if onset_time and str(onset_time) not in ('', 'nan'):
        title += '  Onset=' + str(onset_time) + 's'
    fig.suptitle(title, fontsize=13)

    temp_cols = [c for c in ['tc1', 'tc2', 'tc3', 'tc4'] if c in df.columns]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, tc in enumerate(temp_cols):
        temp = pd.to_numeric(df[tc], errors='coerce').values
        axes[0].plot(t, temp, color=colors[i % len(colors)], label=tc, linewidth=1.2)
    axes[0].set_ylabel('Temperature (C)', fontsize=11)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Temperature', fontsize=12)

    if 'voltage' in df.columns:
        v = pd.to_numeric(df['voltage'], errors='coerce').values
        axes[1].plot(t, v, color='green', label='voltage', linewidth=1.2)
        axes[1].set_ylabel('Voltage (V)', fontsize=11)
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Voltage', fontsize=12)

    axes[1].set_xlabel('Time (s)', fontsize=11)

    if onset_time and str(onset_time) not in ('', 'nan'):
        try:
            onset = float(onset_time)
            for ax in axes:
                ax.axvline(x=onset, color='red', linestyle='--', linewidth=2, alpha=0.8,
                           label='onset=' + str(onset) + 's')
            axes[0].legend(loc='upper left')
            axes[1].legend(loc='upper left')
        except (ValueError, TypeError):
            pass

    plt.tight_layout()

    out_png = data_dir / (sample_id + '_onset_check.png')
    plt.savefig(out_png, dpi=100, bbox_inches='tight')
    print('已保存图片: ' + str(out_png))

    plt.show()


def batch_check(data_dir, meta_path):
    data_dir = Path(data_dir)
    meta_df = pd.read_csv(meta_path)

    if 'is_thermal_runaway' not in meta_df.columns:
        print('错误: 元信息台账中没有 is_thermal_runaway 列')
        return

    tr_df = meta_df[meta_df['is_thermal_runaway'] == 1]
    print('共 ' + str(len(tr_df)) + ' 个热失控样本')
    print('逐个校验, 按回车看下一个, 输入 q 退出\n')

    for idx, row in tr_df.iterrows():
        sid = str(row['sample_id'])
        onset = row.get('tr_onset_time', '')
        print('--- [' + str(idx+1) + '] ' + sid + '  爆发=' + str(onset) + 's ---')

        try:
            plot_sample(data_dir, meta_path, sid)
        except Exception as e:
            print('画图失败: ' + str(e))

        cmd = input('\n按回车继续, 输入 q 退出: ')
        if cmd.strip().lower() == 'q':
            print('已退出')
            break


def main():
    parser = argparse.ArgumentParser(description='Day3: 热失控爆发点人工校验')
    parser.add_argument('data_dir', help='预处理数据文件夹')
    parser.add_argument('meta_file', help='sample_meta.csv 路径')
    parser.add_argument('sample_id', nargs='?', default='', help='单个样本ID (留空则批量校验)')
    args = parser.parse_args()

    if args.sample_id:
        plot_sample(args.data_dir, args.meta_file, args.sample_id)
    else:
        batch_check(args.data_dir, args.meta_file)


if __name__ == '__main__':
    main()
