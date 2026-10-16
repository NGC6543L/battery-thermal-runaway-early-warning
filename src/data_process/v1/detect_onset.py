import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# 统一路径 (src/data_process/paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402


def detect_onset(df, is_thermal_runaway):
    result = {'tr_onset_time': None, 'onset_reason': '', 'max_temp': None,
              'voltage_drop': None, 'max_dTdt': None}

    if not is_thermal_runaway:
        result['onset_reason'] = 'Normal sample, no TR'
        if 'tc1' in df.columns and df['tc1'].notna().any():
            result['max_temp'] = round(float(df['tc1'].max()), 1)
        return result

    t = df['time'].values

    temp_col = None
    for tc in ['tc1', 'tc2', 'tc3', 'tc4']:
        if tc in df.columns and df[tc].notna().sum() > len(df) * 0.3:
            temp_col = tc
            break

    temp_onset = None
    if temp_col:
        T = df[temp_col].values
        valid = ~(np.isnan(t) | np.isnan(T))
        if valid.sum() > 20:
            tv, T = t[valid], T[valid]
            result['max_temp'] = round(float(np.nanmax(T)), 1)

            dT = np.diff(T)
            dt = np.diff(tv)
            dt[dt < 0.001] = 0.001
            dTdt = dT / dt

            window = max(5, len(dTdt) // 200)
            kernel = np.ones(window) / window
            if len(dTdt) > window:
                dTdt_smooth = np.convolve(dTdt, kernel, mode='same')
            else:
                dTdt_smooth = dTdt

            result['max_dTdt'] = round(float(np.nanmax(dTdt_smooth)), 2)

            temp_rise = np.nanmax(T) - np.nanmin(T[:len(T)//10])
            if np.nanmax(dTdt_smooth) >= 1.0 and temp_rise > 30:
                max_idx = np.nanargmax(dTdt_smooth)
                search = dTdt_smooth[:max_idx]
                threshold = np.nanmax(dTdt_smooth) * 0.2
                above = np.where(search >= threshold)[0]
                if len(above) > 0:
                    temp_onset = tv[above[0]]
                else:
                    temp_onset = tv[max_idx]

    volt_onset = None
    if 'voltage' in df.columns:
        V = df['voltage'].values
        valid = ~(np.isnan(t) | np.isnan(V))
        if valid.sum() > 20:
            tv, Vv = t[valid], V[valid]
            v_max = np.nanmax(Vv)
            v_min = np.nanmin(Vv)
            result['voltage_drop'] = round(float(v_max - v_min), 3)

            if v_max - v_min > 0.5:
                drop_threshold = v_max - 0.3
                drop_indices = np.where(Vv < drop_threshold)[0]
                if len(drop_indices) > 0:
                    volt_onset = tv[drop_indices[0]]

    candidates = []
    if temp_onset is not None:
        candidates.append(('temp_gradient', temp_onset))
    if volt_onset is not None:
        candidates.append(('voltage_drop', volt_onset))

    if candidates:
        candidates.sort(key=lambda x: x[1])
        result['tr_onset_time'] = round(float(candidates[0][1]), 2)
        result['onset_reason'] = candidates[0][0] + '_trigger'
        if len(candidates) == 2:
            result['onset_reason'] += '_' + candidates[1][0] + '_verified'
    else:
        result['onset_reason'] = 'no_clear_onset_detected'

    return result


def main():
    parser = argparse.ArgumentParser(description='Day3: Thermal Runaway Onset Detection')
    parser.add_argument('data_dir', help='Preprocessed data folder')
    parser.add_argument('-m', '--meta', default=paths.s(paths.META_LEDGER), help='Meta ledger CSV')
    parser.add_argument('-o', '--output', default=paths.s(paths.SAMPLE_META), help='Output file')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print('Error: data folder does not exist')
        return

    meta_path = Path(args.meta)
    if not meta_path.exists():
        print('Error: meta file does not exist: ' + str(meta_path))
        return

    meta_df = pd.read_csv(meta_path, encoding='utf-8-sig')
    meta_df.columns = meta_df.columns.str.strip()

    print('Loaded meta ledger: ' + str(len(meta_df)) + ' records')

    is_tr_col = None
    for c in meta_df.columns:
        if c.lower() == 'is_thermal_runaway':
            is_tr_col = c
            break
    if is_tr_col is None:
        print('Error: column is_thermal_runaway not found')
        print('Available columns:', meta_df.columns.tolist()[:10])
        return

    sid_col = None
    for c in meta_df.columns:
        if c.lower() == 'sample_id':
            sid_col = c
            break
    if sid_col is None:
        sid_col = meta_df.columns[0]

    obs_col = None
    for c in meta_df.columns:
        if c.lower() == 'obs_score':
            obs_col = c
            break

    tr_count = int(meta_df[is_tr_col].sum())
    print('Thermal runaway samples: ' + str(tr_count))
    print()

    results = []
    detected = 0
    errors = 0
    for idx, row in meta_df.iterrows():
        sid = str(row[sid_col])
        is_tr = int(row[is_tr_col])
        data_file = data_dir / (sid + '.csv')

        status = 'Normal' if not is_tr else 'TR'
        if not data_file.exists():
            print('[' + str(idx+1) + '/' + str(len(meta_df)) + '] ' + sid + ' - ' + status + ' (no data file)')
            results.append({'tr_onset_time': '', 'onset_reason': 'no_preprocessed_file',
                           'max_temp': '', 'voltage_drop': '', 'max_dTdt': ''})
            continue

        try:
            df = pd.read_csv(data_file, encoding='utf-8-sig')
            df.columns = df.columns.str.strip()
            r = detect_onset(df, is_tr == 1)
        except Exception as e:
            errors += 1
            r = {'tr_onset_time': None, 'onset_reason': 'error_' + str(e)[:50],
                 'max_temp': '', 'voltage_drop': '', 'max_dTdt': ''}

        if is_tr and r['tr_onset_time'] is not None:
            detected += 1
            flag = 'OK'
        elif is_tr:
            flag = 'WARN'
        else:
            flag = 'OK'

        onset_str = str(r['tr_onset_time']) + 's' if r['tr_onset_time'] is not None else '-'
        print('[' + str(idx+1) + '/' + str(len(meta_df)) + '] ' + flag + ' ' + sid + ' - ' + status + ' onset=' + onset_str)

        results.append(r)

    result_df = pd.DataFrame(results)
    for col in result_df.columns:
        meta_df[col] = result_df[col].values

    meta_df.to_csv(args.output, index=False, encoding='utf-8-sig')

    print()
    print('=' * 50)
    print('Detection complete')
    print('Total samples: ' + str(len(meta_df)))
    print('Thermal runaway: ' + str(tr_count))
    print('Successfully detected: ' + str(detected))
    if tr_count > 0:
        print('Detection rate: ' + str(round(detected / tr_count * 100, 1)) + '%')
    if errors > 0:
        print('Errors: ' + str(errors))
    print()
    print('Output file: ' + args.output)


if __name__ == '__main__':
    main()
