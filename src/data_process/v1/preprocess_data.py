"""
锂电池针刺热失控实验 - 数据批量预处理脚本 v1.1
================================================
功能:
  1. 智能解析三套独立时间轴(电压,针刺力,温度TC1-TC4)
     支持格式:多块垂直堆叠 / 多Sheet / 多列并排
  2. 统一时序:0.2s 固定采样间隔,线性插值对齐
  3. 脏数据过滤:时间回跳,重复时间,大面积NaN,传感器失效
  4. 批量导出:processed_data 文件夹,固定列名:
        time, voltage, force, tc1, tc2, tc3, tc4
用法:
  python preprocess_data.py data -o processed_data
  python preprocess_data.py data -o processed_data --sample-rate 5
"""
import re
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import interpolate
import io

# 统一路径 (src/data_process/paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402


# ========== Signal identification keywords ==========
SIGNAL_KWS = {
    'voltage': ['voltage', 'volt', 'vcell', 'cell voltage', 'u/v', 'v/', 'voc'],
    'force':   ['force', 'load', 'stress', 'penetrator', 'punch', 'compression',
                'indentation', 'n/kn', 'kn/', 'displacement'],
    'tc1':     ['tc1', 'tc 1', 'thermocouple 1', 't1', 'ch1', 'channel1'],
    'tc2':     ['tc2', 'tc 2', 'thermocouple 2', 't2', 'ch2', 'channel2'],
    'tc3':     ['tc3', 'tc 3', 'thermocouple 3', 't3', 'ch3', 'channel3'],
    'tc4':     ['tc4', 'tc 4', 'thermocouple 4', 't4', 'ch4', 'channel4'],
    'time':    ['time', 'test time', 'timestamp', 't/s', 't(s)', 'sec', 'second', 'elapsed'],
}


# ========== 1. Read raw file ==========
def read_raw_file(file_path):
    ext = Path(file_path).suffix.lower()
    blocks = []

    if ext == '.csv':
        for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        else:
            return blocks
        blocks = _parse_csv_blocks(lines)

    elif ext in ['.xlsx', '.xls']:
        xl = pd.ExcelFile(file_path)
        for sheet in xl.sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet, header=None)
                sheet_blocks = _detect_blocks(df)
                blocks.extend(sheet_blocks)
            except Exception:
                continue
        if not blocks:
            for sheet in xl.sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    if len(df.columns) >= 2 and len(df) > 10:
                        blocks.append(df)
                except Exception:
                    continue
    return blocks


def _parse_csv_blocks(lines):
    blocks = []
    current = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith('#'):
            if current:
                df = _lines_to_df(current)
                if df is not None and len(df) > 5:
                    blocks.append(df)
                current = []
            continue
        current.append(line)
    if current:
        df = _lines_to_df(current)
        if df is not None and len(df) > 5:
            blocks.append(df)
    return blocks


def _lines_to_df(lines):
    text = ''.join(lines)
    for sep in [None, '\t', ',', ';']:
        try:
            if sep is None:
                df = pd.read_csv(io.StringIO(text), sep=None, engine='python')
            else:
                df = pd.read_csv(io.StringIO(text), sep=sep)
            if len(df.columns) >= 2 and len(df) > 5:
                return df
        except Exception:
            continue
    return None


def _detect_blocks(raw_df):
    blocks = []
    n = len(raw_df)
    starts = [0]
    for i in range(1, n):
        row_str = ' '.join(str(v) for v in raw_df.iloc[i] if pd.notna(v)).lower()
        if any(kw in row_str for kw in SIGNAL_KWS['time']):
            starts.append(i)
    if len(starts) > 1:
        for idx, s in enumerate(starts):
            e = starts[idx + 1] if idx + 1 < len(starts) else n
            header = raw_df.iloc[s].tolist()
            data = raw_df.iloc[s + 1:e].reset_index(drop=True)
            data.columns = [str(h) if pd.notna(h) else 'col_' + str(i) for i, h in enumerate(header)]
            data = data.dropna(axis=1, how='all')
            if len(data.columns) >= 2 and len(data) > 5:
                blocks.append(data)
    else:
        if len(raw_df.columns) >= 2 and len(raw_df) > 10:
            header = raw_df.iloc[0].tolist()
            data = raw_df.iloc[1:].reset_index(drop=True)
            data.columns = [str(h) if pd.notna(h) else 'col_' + str(i) for i, h in enumerate(header)]
            data = data.dropna(axis=1, how='all')
            data = data.apply(pd.to_numeric, errors='coerce')
            if len(data.columns) >= 2 and len(data) > 5:
                blocks.append(data)
    return blocks


# ========== 2. Signal identification ==========
def identify_signals(blocks):
    result = {}
    for block in blocks:
        cols = block.columns.tolist()
        col_low = {c: str(c).lower().strip() for c in cols}

        time_col = None
        for c in cols:
            if any(kw in col_low[c] for kw in SIGNAL_KWS['time']):
                time_col = c
                break
        if time_col is None:
            try:
                v = pd.to_numeric(block[cols[0]], errors='coerce').dropna()
                if len(v) > 10 and v.iloc[-1] > v.iloc[0] > 0:
                    time_col = cols[0]
            except Exception:
                pass
        if time_col is None:
            continue

        t_vals = pd.to_numeric(block[time_col], errors='coerce').values

        for sig_type, kws in SIGNAL_KWS.items():
            if sig_type == 'time':
                continue
            if sig_type in result:
                continue
            for col in cols:
                if col == time_col:
                    continue
                if any(kw in col_low[col] for kw in kws):
                    sv = pd.to_numeric(block[col], errors='coerce')
                    if sv.notna().sum() > 10:
                        result[sig_type] = {'time': t_vals, 'value': sv.values}
                        break

    # Supplement unidentified TC channels
    tc_found = sum(1 for k in result if k.startswith('tc'))
    if tc_found < 4:
        temp_cols = []
        for block in blocks:
            for col in block.columns:
                cl = str(col).lower()
                if any(kw in cl for kw in ['temp', 'tc', 'thermocouple', '[c]']):
                    if any(v in cl for v in SIGNAL_KWS['voltage']):
                        continue
                    if col not in [c.get('col_name', '') for c in result.values()]:
                        temp_cols.append((block, col))
        for ti in range(1, 5):
            key = 'tc' + str(ti)
            if key in result:
                continue
            if not temp_cols:
                break
            best = None
            best_score = -1
            for block, col in temp_cols:
                cl = str(col).lower()
                score = 0
                if str(ti) in cl:
                    score += 10
                if ('tc' + str(ti)) in cl:
                    score += 20
                if 'temp' in cl or '[c]' in cl:
                    score += 5
                if score > best_score:
                    best_score = score
                    best = (block, col)
            if best:
                block, col = best
                tc_time = None
                for c in block.columns:
                    if any(kw in str(c).lower() for kw in SIGNAL_KWS['time']):
                        tc_time = c
                        break
                if tc_time is None:
                    tc_time = block.columns[0]
                tt = pd.to_numeric(block[tc_time], errors='coerce').values
                tv = pd.to_numeric(block[col], errors='coerce').values
                if pd.Series(tv).notna().sum() > 10:
                    result[key] = {'time': tt, 'value': tv}
                temp_cols.remove(best)
    return result


# ========== 3. Dirty data filtering ==========
def clean_signal(t, v):
    notes = []
    t = np.array(t, dtype=float)
    v = np.array(v, dtype=float)

    valid = ~np.isnan(t)
    if not valid.all():
        t, v = t[valid], v[valid]
        notes.append('Removed NaN time points: ' + str(int((~valid).sum())))
    if len(t) < 5:
        return np.array([]), np.array([]), 'Insufficient data'

    td = np.diff(t)
    back = np.sum(td < 0)
    if back > 0:
        notes.append('Time jumps back: ' + str(int(back)))
        keep = np.ones(len(t), dtype=bool)
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                keep[i] = False
        t, v = t[keep], v[keep]
    if len(t) < 5:
        return np.array([]), np.array([]), 'Insufficient after cleaning'

    _, uidx = np.unique(t, return_index=True)
    if len(uidx) < len(t):
        notes.append('Duplicate time: ' + str(int(len(t) - len(uidx))))
        uidx = np.sort(uidx)
        t, v = t[uidx], v[uidx]

    nan_r = np.isnan(v).sum() / len(v)
    if nan_r > 0.5:
        return np.array([]), np.array([]), 'NaN ratio ' + str(round(nan_r * 100, 1)) + '%'

    vv = v[~np.isnan(v)]
    if len(vv) > 0 and np.nanmax(vv) - np.nanmin(vv) < 0.001:
        notes.append('Sensor may have failed')

    return t, v, '; '.join(notes)


# ========== 4. Resample to unified time axis ==========
def resample(t, v, target_t):
    if len(t) < 2:
        return np.full(len(target_t), np.nan)
    mask = ~(np.isnan(t) | np.isnan(v))
    tv, vv = t[mask], v[mask]
    if len(tv) < 2:
        return np.full(len(target_t), np.nan)
    f = interpolate.interp1d(tv, vv, kind='linear', bounds_error=False, fill_value=np.nan)
    return f(target_t)


# ========== 5. Single file processing ==========
def process_file(file_path, sample_rate=5.0):
    sid = Path(file_path).stem
    info = {'sample_id': sid, 'status': 'success', 'found': [], 'missing': [],
            'points': 0, 'time_range': '', 'notes': []}

    try:
        blocks = read_raw_file(file_path)
        if not blocks:
            info['status'] = 'failed'
            info['notes'].append('Cannot parse')
            return None, info

        signals = identify_signals(blocks)
        required = ['voltage', 'force', 'tc1', 'tc2', 'tc3', 'tc4']
        found = [s for s in required if s in signals]
        missing = [s for s in required if s not in signals]
        info['found'], info['missing'] = found, missing

        if len(found) < 3:
            info['status'] = 'failed'
            info['notes'].append('Only found ' + str(len(found)) + ' signals')
            return None, info

        cleaned = {}
        for s in found:
            ct, cv, note = clean_signal(signals[s]['time'], signals[s]['value'])
            if len(ct) > 10:
                cleaned[s] = (ct, cv)
            else:
                info['notes'].append(s + ' invalid after cleaning')

        if not cleaned:
            info['status'] = 'failed'
            info['notes'].append('All signals invalid')
            return None, info

        t_start = max(x[0][0] for x in cleaned.values())
        t_end = min(x[0][-1] for x in cleaned.values())
        if t_end <= t_start:
            info['status'] = 'failed'
            info['notes'].append('No time overlap')
            return None, info

        dt = 1.0 / sample_rate
        target_t = np.arange(t_start, t_end + dt / 2, dt)
        if target_t[-1] > t_end:
            target_t = target_t[:-1]

        out = {'time': np.round(target_t, 4)}
        for s in required:
            if s in cleaned:
                out[s] = resample(cleaned[s][0], cleaned[s][1], target_t)
            else:
                out[s] = np.full(len(target_t), np.nan)

        df = pd.DataFrame(out)
        info['points'] = len(df)
        info['time_range'] = str(round(t_start, 1)) + 's-' + str(round(t_end, 1)) + 's'

        for s in required:
            r = df[s].notna().sum() / len(df)
            if r < 0.3:
                info['notes'].append(s + ' valid rate ' + str(round(r * 100, 1)) + '%')

        return df, info

    except Exception as e:
        info['status'] = 'error'
        info['notes'].append(str(e)[:80])
        return None, info


# ========== 6. Batch processing ==========
def batch_process(input_dir, output_dir, sample_rate=5.0, skip_exist=False):
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        print('Error: ' + str(inp) + ' does not exist')
        return

    exts = ['.csv', '.xlsx', '.xls']
    files = []
    for e in exts:
        files.extend(inp.rglob('*' + e))
    files = sorted(set(f for f in files if not f.name.startswith('~$')))

    if not files:
        print('No data files found')
        return

    print('Found ' + str(len(files)) + ' files, sample rate ' + str(sample_rate) + 'Hz (interval ' + str(round(1/sample_rate, 2)) + 's)')
    if skip_exist:
        print('Skip existing files mode: ON')
    print('Output to: ' + str(out) + '\n')

    results = []
    ok = fail = skipped = 0
    for i, fp in enumerate(files, 1):
        out_file = out / (Path(fp).stem + '.csv')
        if skip_exist and out_file.exists():
            skipped += 1
            print('[' + str(i) + '/' + str(len(files)) + '] SKIP ' + fp.name)
            continue

        print('[' + str(i) + '/' + str(len(files)) + '] ' + fp.name)
        df, info = process_file(fp, sample_rate)
        if df is not None and info['status'] == 'success':
            df.to_csv(out_file, index=False, encoding='utf-8-sig')
            ok += 1
            print('  OK ' + str(info['points']) + ' pts ' + info['time_range'])
            if info['notes']:
                print('    Notes: ' + '; '.join(info['notes']))
        else:
            fail += 1
            print('  FAIL ' + ('; '.join(info['notes']) if info['notes'] else 'failed'))
        results.append(info)

    print('\n' + '=' * 50)
    total_processed = ok + fail
    print('Done: OK ' + str(ok) + ' / FAIL ' + str(fail) + ' / SKIPPED ' + str(skipped) + ' / total ' + str(len(files)))
    if total_processed > 0:
        print('Success rate: ' + str(round(ok / total_processed * 100, 1)) + '%')

    log_df = pd.DataFrame(results)
    log_df.to_csv(out / '_processing_log.csv', index=False, encoding='utf-8-sig')
    print('Log: ' + str(out / '_processing_log.csv'))


# ========== Entry ==========
def main():
    p = argparse.ArgumentParser(description='Battery data batch preprocessing')
    p.add_argument('input_dir', nargs='?', default=paths.s(paths.DATA_RAW), help='Raw data folder')
    p.add_argument('-o', '--output', default=paths.s(paths.PROCESSED_DATA), help='Output folder')
    p.add_argument('--sample-rate', type=float, default=5.0, help='Sample rate Hz (default 5)')
    p.add_argument('--skip-exist', action='store_true', help='Skip existing output files')
    args = p.parse_args()
    batch_process(args.input_dir, args.output, args.sample_rate, args.skip_exist)


if __name__ == '__main__':
    main()
