"""
锂电池热失控实验 - 元信息台账自动生成脚本 v2 (CSV版)
功能：扫描数据文件夹，解析文件名，读取main.xlsx标注，
      检测热失控起始时间，生成训练用元信息台账
"""

import re
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# 统一路径 (src/data_process/paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

# ============================================================
# 1. 从 main.xlsx 读取观测结果映射表
# ============================================================
def load_main_xlsx(main_xlsx_path):
    """
    从 main.xlsx 的 Summary sheet 读取标注信息
    返回 {文件名: {obs_score, capacity, soc}} 字典
    """
    try:
        df = pd.read_excel(main_xlsx_path, sheet_name='Summary')
    except Exception as e:
        print(f"[警告] 无法读取 {main_xlsx_path}: {e}")
        return {}

    def map_score(desc, num):
        if pd.notna(num):
            try:
                level = int(float(num))
            except (ValueError, TypeError):
                level = 0
        else:
            m = re.search(r'[–\-]\s*(\d+)\s*$', str(desc))
            level = int(m.group(1)) if m else 0

        if level <= 4:
            return 'No effect'
        elif level == 5:
            return 'rupture'
        elif level == 6:
            return 'rupture'
        elif level >= 7:
            return 'fire'
        else:
            desc_lower = str(desc).lower()
            if 'fire' in desc_lower:
                return 'fire'
            elif 'smoke' in desc_lower:
                return 'smoke'
            elif 'rupture' in desc_lower:
                return 'rupture'
            elif 'no effect' in desc_lower:
                return 'No effect'
            return ''

    cols = {'name': df.columns[0], 'desc': None, 'num': None, 'cap': None, 'soc': None}
    for col in df.columns:
        cl = str(col).lower()
        if 'score' in cl and 'observed' in cl:
            if pd.api.types.is_numeric_dtype(df[col]) or \
               all(str(v).strip().isdigit() or pd.isna(v) for v in df[col].head(10)):
                cols['num'] = col
            else:
                cols['desc'] = col
        elif 'capacity' in cl or '容量' in cl:
            cols['cap'] = col
        elif col.upper() == 'SOC' or 'soc' in cl:
            cols['soc'] = col

    info_map = {}
    for _, row in df.iterrows():
        fname = str(row[cols['name']]).strip()
        desc = row[cols['desc']] if cols['desc'] else ''
        num = row[cols['num']] if cols['num'] else None
        cap = row[cols['cap']] if cols['cap'] else None
        soc = row[cols['soc']] if cols['soc'] else None

        info_map[fname] = {
            'obs_score': map_score(desc, num),
            'capacity': int(cap) if pd.notna(cap) else None,
            'soc': int(soc) if pd.notna(soc) else None,
        }

    return info_map

# ============================================================
# 2. 文件名解析
# ============================================================
def parse_filename(filename):
    """从文件名提取容量、SOC等元信息"""
    name = Path(filename).stem

    info = {
        'sample_id': name,
        'capacity_mAh': None,
        'soc': None,
    }

    soc_match = re.search(r'(\d+)\s*SOC', name, re.IGNORECASE)
    if soc_match:
        info['soc'] = int(soc_match.group(1))

    cap_match = re.search(r'(\d+)\s*mAh', name, re.IGNORECASE)
    if cap_match:
        info['capacity_mAh'] = int(cap_match.group(1))
    else:
        ah_match = re.search(r'(\d+)\s*Ah', name, re.IGNORECASE)
        if ah_match:
            info['capacity_mAh'] = int(ah_match.group(1)) * 1000

    return info

# ============================================================
# 3. 热失控起始时间检测
# ============================================================
def detect_thermal_runaway_onset(time_arr, temp_arr, threshold_rate=1.0):
    """
    检测热失控起始时间
    原理：找到温升速率首次超过阈值的时间点
    """
    notes = []

    if len(time_arr) < 10 or len(temp_arr) < 10:
        return None, "数据点过少"

    t = np.array(time_arr, dtype=float)
    T = np.array(temp_arr, dtype=float)

    if np.all(np.isnan(T)):
        return None, "温度数据全空"

    if np.any(np.diff(t) < 0):
        notes.append("时间序列非单调")

    temp_diff = np.abs(np.diff(T))
    if np.nanmax(temp_diff) > 50:
        notes.append("存在温度跳变")

    window = max(5, len(t) // 200)
    if window > len(t) // 2:
        window = 5

    dT_dt = np.diff(T) / np.maximum(np.diff(t), 1e-6)

    kernel = np.ones(window) / window
    if len(dT_dt) > window:
        dT_dt_smooth = np.convolve(dT_dt, kernel, mode='same')
    else:
        dT_dt_smooth = dT_dt

    max_rate = np.nanmax(dT_dt_smooth)
    max_rate_idx = np.nanargmax(dT_dt_smooth)

    temp_rise = np.nanmax(T) - np.nanmin(T[:len(T)//10])

    is_tr = (max_rate >= threshold_rate) and (temp_rise > 30)

    if not is_tr:
        onset = None
    else:
        onset_rate = threshold_rate * 0.3
        search_region = dT_dt_smooth[:max_rate_idx]
        above_threshold = np.where(search_region >= onset_rate)[0]

        if len(above_threshold) > 0:
            onset_idx = above_threshold[0]
        else:
            onset_idx = max_rate_idx

        onset = float(t[onset_idx])

    note = "; ".join(notes) if notes else ""
    return onset, note

# ============================================================
# 4. 数据文件读取与质量检查
# ============================================================
def read_data_file(file_path):
    """读取数据文件，返回 DataFrame 和读取状态备注"""
    ext = Path(file_path).suffix.lower()

    try:
        if ext == '.csv':
            for enc in ['utf-8', 'gbk', 'latin1']:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return None, "编码无法识别"
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            return None, f"不支持的格式: {ext}"

        if df.empty or len(df) < 5:
            return None, "数据为空或过少"

    except Exception as e:
        return None, f"读取失败: {str(e)[:50]}"

    return df, ""

def find_columns(df):
    """自动识别时间、温度、电压列"""
    cols = {'time': None, 'temp': None, 'volt': None}
    col_list = df.columns.tolist()

    time_kws = ['time', '时间', 'timestamp', 't/s', 't(s)', 'sec', 'second']
    for col in col_list:
        cl = str(col).lower()
        if any(kw in cl for kw in time_kws):
            cols['time'] = col
            break

    temp_kws = ['temperature', 'temp', '温度', 'tc', 'thermocouple', 't_c']
    volt_kws = ['voltage', 'volt', '电压', 'u/v', 'v/']
    for col in col_list:
        cl = str(col).lower()
        if any(v in cl for v in volt_kws):
            continue
        if any(kw in cl for kw in temp_kws):
            cols['temp'] = col
            break

    if cols['temp'] is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = []
        if cols['time']:
            exclude.append(cols['time'])
        for col in numeric_cols:
            cl = str(col).lower()
            if any(v in cl for v in volt_kws):
                exclude.append(col)
        candidates = [c for c in numeric_cols if c not in exclude]
        if candidates:
            max_range = -1
            for c in candidates:
                rng = df[c].max() - df[c].min()
                if rng > max_range:
                    max_range = rng
                    cols['temp'] = c

    for col in col_list:
        cl = str(col).lower()
        if any(kw in cl for kw in volt_kws):
            cols['volt'] = col
            break

    return cols

# ============================================================
# 5. 主流程
# ============================================================
def generate_ledger(data_dir=None, main_xlsx=None, output_file=None):
    data_dir = data_dir or paths.s(paths.DATA_RAW)
    main_xlsx = main_xlsx or paths.s(paths.MAIN_XLSX)
    output_file = output_file or paths.s(paths.META_LEDGER)
    """
    扫描数据文件夹，生成元信息台账 (CSV格式)

    字段说明：
        sample_id            样本ID（文件名去掉扩展名）
        file_path            文件完整路径
        capacity             电池容量 (mAh)
        soc                  电量 SOC (%)
        obs_score            观测结果: No effect / smoke / fire / rupture
        is_thermal_runaway   是否发生热失控: 0=否, 1=是
        tr_onset_time        热失控起始时间 (秒)，没失控填空
        note                 备注（数据质量问题）
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"错误：数据文件夹不存在 - {data_dir}")
        return

    print(f"读取标注文件: {main_xlsx}")
    main_info = load_main_xlsx(main_xlsx)
    if main_info:
        print(f"  已加载 {len(main_info)} 条标注记录")
    else:
        print("  [提示] 未找到标注文件，obs_score/capacity 将留空")

    extensions = ['.csv', '.xlsx', '.xls']
    data_files = []
    for ext in extensions:
        data_files.extend(data_dir.rglob(f'*{ext}'))
    data_files = [f for f in data_files if not f.name.startswith('~$')]
    data_files = sorted(set(data_files))

    if not data_files:
        print(f"在 {data_dir} 中未找到数据文件")
        return

    print(f"\n找到 {len(data_files)} 个数据文件，开始处理...\n")

    records = []
    for i, file_path in enumerate(data_files, 1):
        print(f"[{i}/{len(data_files)}] {file_path.name}")

        meta = parse_filename(file_path.name)

        df, read_note = read_data_file(file_path)
        notes = [read_note] if read_note else []

        tr_onset_time = None
        if df is not None:
            cols = find_columns(df)

            time_arr = None
            temp_arr = None

            if cols['temp']:
                temp_arr = pd.to_numeric(df[cols['temp']], errors='coerce').values
            else:
                notes.append("未找到温度列")

            if cols['time']:
                time_arr = pd.to_numeric(df[cols['time']], errors='coerce').values
            else:
                if temp_arr is not None:
                    time_arr = np.arange(len(temp_arr))
                    notes.append("无时间列，使用索引代替")

            if time_arr is not None and temp_arr is not None:
                valid = ~(np.isnan(time_arr) | np.isnan(temp_arr))
                if valid.sum() > 10:
                    onset, quality_note = detect_thermal_runaway_onset(
                        time_arr[valid], temp_arr[valid]
                    )
                    tr_onset_time = onset
                    if quality_note:
                        notes.append(quality_note)
                else:
                    notes.append("有效温度数据不足")

        sample_id = meta['sample_id']
        main_data = None

        if sample_id in main_info:
            main_data = main_info[sample_id]
        else:
            for key in main_info:
                if key.lower() == sample_id.lower():
                    main_data = main_info[key]
                    break

        obs_score = main_data['obs_score'] if main_data else ''

        capacity = meta['capacity_mAh']
        if not capacity and main_data and main_data['capacity']:
            capacity = main_data['capacity']

        soc = meta['soc']
        if not soc and main_data and main_data['soc']:
            soc = main_data['soc']

        is_tr = 0
        if obs_score in ['fire', 'smoke', 'rupture']:
            is_tr = 1
        elif obs_score == 'No effect':
            is_tr = 0
        elif tr_onset_time is not None:
            is_tr = 1

        record = {
            'sample_id': sample_id,
            'file_path': str(file_path.resolve()),
            'capacity': capacity if capacity else '',
            'soc': soc if soc else '',
            'obs_score': obs_score,
            'is_thermal_runaway': is_tr,
            'tr_onset_time': round(tr_onset_time, 2) if tr_onset_time is not None else '',
            'note': "; ".join(notes) if notes else '',
        }
        records.append(record)

        status = "🔥 热失控" if is_tr else "✓ 正常"
        onset_str = f" (第{tr_onset_time:.0f}s)" if tr_onset_time else ""
        print(f"     {status}{onset_str}  {obs_score}")

    df = pd.DataFrame(records)
    df = df.sort_values('sample_id').reset_index(drop=True)

    # 输出 CSV 文件
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    tr_count = (df['is_thermal_runaway'] == 1).sum()
    normal_count = (df['is_thermal_runaway'] == 0).sum()
    bad_count = df['note'].fillna('').str.len().gt(0).sum()

    cap_vals = pd.to_numeric(df['capacity'], errors='coerce').dropna()
    soc_vals = pd.to_numeric(df['soc'], errors='coerce').dropna()

    print(f"\n{'='*50}")
    print(f"✓ 元信息台账已生成: {output_file}")
    print(f"  共 {len(df)} 条记录")
    print(f"\n统计概览:")
    print(f"  总样本数: {len(df)}")
    print(f"  热失控样本: {int(tr_count)}")
    print(f"  正常样本: {int(normal_count)}")
    print(f"  热失控比例: {tr_count/len(df)*100:.1f}%")
    print(f"  有质量问题样本: {int(bad_count)}")
    if len(cap_vals) > 0:
        print(f"  容量范围: {cap_vals.min():.0f} - {cap_vals.max():.0f} mAh")
    if len(soc_vals) > 0:
        print(f"  SOC 范围: {soc_vals.min():.0f}% - {soc_vals.max():.0f}%")
    print(f"\n📌 提示：tr_onset_time 是热失控起始秒数")
    print(f"   往前倒推 60-120 秒 = 极早期预警片段（标签1）")
    print(f"   其余时间段 = 正常状态（标签0）")

    return df

# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='锂电池热失控元信息台账生成 v2 (CSV版)')
    parser.add_argument('data_dir', nargs='?', default=paths.s(paths.DATA_RAW),
                        help='数据文件夹路径')
    parser.add_argument('-m', '--main', default=paths.s(paths.MAIN_XLSX),
                        help='标注文件 main.xlsx 路径')
    parser.add_argument('-o', '--output', default=paths.s(paths.META_LEDGER),
                        help='输出CSV文件路径')
    args = parser.parse_args()

    generate_ledger(args.data_dir, args.main, args.output)

if __name__ == '__main__':
    main()