import argparse
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import pickle
import warnings
warnings.filterwarnings('ignore')

#Argument Parser 
parser = argparse.ArgumentParser(description='Create Dataset from Sleep Study Data')
parser.add_argument('-in_dir',  type=str, required=True,  help='Input Data directory e.g. "Data"')
parser.add_argument('-out_dir', type=str, required=True,  help='Output Dataset directory e.g. "Dataset"')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# Constants
WINDOW_SEC   = 30       # 30-second windows
OVERLAP      = 0.5      # 50% overlap
FS_RESP      = 32       # Hz — Flow and Thorac
FS_SPO2      = 4        # Hz — SpO2
BREATH_LOW   = 0.17     # Hz — lower breathing frequency
BREATH_HIGH  = 0.4      # Hz — upper breathing frequency

WINDOW_FLOW   = int(WINDOW_SEC * FS_RESP)   # 960 samples
WINDOW_SPO2   = int(WINDOW_SEC * FS_SPO2)   # 120 samples
STEP_FLOW     = int(WINDOW_FLOW  * (1 - OVERLAP))   # 480
STEP_SPO2     = int(WINDOW_SPO2  * (1 - OVERLAP))   # 60

#Helper
def find_file(folder, keyword, exclude=None):
    for f in os.listdir(folder):
        fl = f.lower()
        if keyword.lower() in fl:
            if exclude and exclude.lower() in fl:
                continue
            return os.path.join(folder, f)
    return None

# Helper: Read signal file
def read_signal_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == 'Data:':
            data_start = i + 1
            break

    timestamps, values = [], []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        try:
            ts_part, val_part = line.split(';')
            ts_str = ts_part.strip().replace(',', '.')
            ts = pd.to_datetime(ts_str, format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')
            if pd.isna(ts):
                ts = pd.to_datetime(ts_str, format='%d.%m.%Y %H:%M:%S', errors='coerce')
            timestamps.append(ts)
            values.append(float(val_part.strip()))
        except Exception:
            continue

    return pd.DataFrame({'timestamp': timestamps, 'value': values})

# Helper
def read_flow_events(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    records = []
    for line in lines:
        line = line.strip()
        if not line or ';' not in line:
            continue
        if any(line.startswith(k) for k in ['Signal', 'Start', 'Unit']):
            continue
        try:
            parts = line.split(';')
            time_range = parts[0].strip()
            event_type = parts[2].strip()

            date_part = time_range[:10]
            rest      = time_range[11:]
            start_str, end_str = rest.split('-')

            start_dt = pd.to_datetime(f"{date_part} {start_str.replace(',', '.')}",
                                      format='%d.%m.%Y %H:%M:%S.%f')
            end_dt   = pd.to_datetime(f"{date_part} {end_str.replace(',', '.')}",
                                      format='%d.%m.%Y %H:%M:%S.%f')
            if end_dt < start_dt:
                end_dt += pd.Timedelta(days=1)

            records.append({'start_time': start_dt, 'end_time': end_dt,
                             'event_type': event_type})
        except Exception:
            continue

    return pd.DataFrame(records)

#  Helper
def read_sleep_profile(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    records = []
    for line in lines:
        line = line.strip()
        if not line or ';' not in line:
            continue
        if any(line.startswith(k) for k in ['Signal', 'Start', 'Unit', 'Rate', 'Events']):
            continue
        try:
            ts_part, stage_part = line.split(';')
            ts_str = ts_part.strip().replace(',', '.')
            ts = pd.to_datetime(ts_str, format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')
            stage = stage_part.strip()
            if stage == 'A':
                continue  # skip artifact/unknown
            records.append({'timestamp': ts, 'sleep_stage': stage})
        except Exception:
            continue

    return pd.DataFrame(records)

# Helper: Bandpass filter
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    # Clamp to valid range
    low  = max(low,  1e-4)
    high = min(high, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Helper: Assign label to a window
def get_window_label(win_start_ts, win_end_ts, events_df):
    """
    If a window overlaps > 50% with a labeled event → that event's label.
    Else → 'Normal'
    Priority: Obstructive Apnea > Hypopnea > Normal
    """
    win_dur = (win_end_ts - win_start_ts).total_seconds()
    best_label    = 'Normal'
    best_overlap  = 0.0

    for _, ev in events_df.iterrows():
        overlap_start = max(win_start_ts, ev['start_time'])
        overlap_end   = min(win_end_ts,   ev['end_time'])
        overlap_sec   = max(0, (overlap_end - overlap_start).total_seconds())
        overlap_frac  = overlap_sec / win_dur

        if overlap_frac > 0.5 and overlap_sec > best_overlap:
            best_overlap = overlap_sec
            best_label   = ev['event_type']

    return best_label

#  Helper: Get sleep stage for window midpoint 
def get_sleep_stage(win_mid_ts, sleep_df):
    if sleep_df.empty:
        return 'Unknown'
    idx = (sleep_df['timestamp'] - win_mid_ts).abs().idxmin()
    return sleep_df.loc[idx, 'sleep_stage']

# Process each participant 
all_records = []

participant_folders = sorted([
    d for d in os.listdir(args.in_dir)
    if os.path.isdir(os.path.join(args.in_dir, d))
])

print(f"Found participants: {participant_folders}\n")

for pid in participant_folders:
    pdir = os.path.join(args.in_dir, pid)
    print(f"Processing {pid}...")

    flow_file   = find_file(pdir, 'flow',  exclude='event')
    thorac_file = find_file(pdir, 'thorac')
    spo2_file   = find_file(pdir, 'spo2') or find_file(pdir, 'spo₂')
    events_file = find_file(pdir, 'flow event') or find_file(pdir, 'flow_event')
    sleep_file  = find_file(pdir, 'sleep')

    if not all([flow_file, thorac_file, spo2_file, events_file]):
        print(f"  ⚠️  Missing files for {pid}, skipping.")
        continue

    # Read signals
    flow_df   = read_signal_file(flow_file)
    thorac_df = read_signal_file(thorac_file)
    spo2_df   = read_signal_file(spo2_file)
    events_df = read_flow_events(events_file)
    sleep_df  = read_sleep_profile(sleep_file) if sleep_file else pd.DataFrame()

    print(f"  Loaded: Flow={len(flow_df)}, Thorac={len(thorac_df)}, SpO2={len(spo2_df)}, Events={len(events_df)}")

    # ── Filter signals ─────────────────────────────────────────────────────────
    flow_vals   = bandpass_filter(flow_df['value'].values,   BREATH_LOW, BREATH_HIGH, FS_RESP)
    thorac_vals = bandpass_filter(thorac_df['value'].values, BREATH_LOW, BREATH_HIGH, FS_RESP)
    # SPO2 is at 4Hz — upper limit must be < nyquist (2Hz), use 0.4 Hz cap
    spo2_vals   = bandpass_filter(spo2_df['value'].values,   BREATH_LOW, min(BREATH_HIGH, 1.9), FS_SPO2)

    flow_ts   = flow_df['timestamp'].values
    thorac_ts = thorac_df['timestamp'].values
    spo2_ts   = spo2_df['timestamp'].values

    #  Sliding window 
    n_flow  = len(flow_vals)
    n_spo2  = len(spo2_vals)
    win_count = 0

    for i in range(0, n_flow - WINDOW_FLOW + 1, STEP_FLOW):
        # Flow window
        flow_win   = flow_vals[i : i + WINDOW_FLOW]
        thorac_win = thorac_vals[i : i + WINDOW_FLOW] if i + WINDOW_FLOW <= len(thorac_vals) else None

        # Corresponding SPO2 window (align by ratio)
        spo2_i     = int(i * FS_SPO2 / FS_RESP)
        spo2_end   = spo2_i + WINDOW_SPO2
        spo2_win   = spo2_vals[spo2_i : spo2_end] if spo2_end <= n_spo2 else None

        if thorac_win is None or spo2_win is None:
            break
        if len(flow_win) != WINDOW_FLOW or len(spo2_win) != WINDOW_SPO2:
            break

        # Window timestamps
        win_start_ts = pd.Timestamp(flow_ts[i])
        win_end_ts   = pd.Timestamp(flow_ts[min(i + WINDOW_FLOW - 1, n_flow - 1)])
        win_mid_ts   = win_start_ts + (win_end_ts - win_start_ts) / 2

        # Label
        label       = get_window_label(win_start_ts, win_end_ts, events_df)
        sleep_stage = get_sleep_stage(win_mid_ts, sleep_df)

        # Features: statistical summary per signal
        record = {
            'participant': pid,
            'win_start':   win_start_ts,
            'win_end':     win_end_ts,
            'sleep_stage': sleep_stage,
            'label':       label,

            # Flow features
            'flow_mean':   np.mean(flow_win),
            'flow_std':    np.std(flow_win),
            'flow_min':    np.min(flow_win),
            'flow_max':    np.max(flow_win),
            'flow_range':  np.ptp(flow_win),

            # Thorac features
            'thorac_mean': np.mean(thorac_win),
            'thorac_std':  np.std(thorac_win),
            'thorac_min':  np.min(thorac_win),
            'thorac_max':  np.max(thorac_win),
            'thorac_range':np.ptp(thorac_win),

            # SpO2 features
            'spo2_mean':   np.mean(spo2_win),
            'spo2_std':    np.std(spo2_win),
            'spo2_min':    np.min(spo2_win),
            'spo2_max':    np.max(spo2_win),

            # Raw windows (for CNN) — stored as lists
            'flow_window':   flow_win.tolist(),
            'thorac_window': thorac_win.tolist(),
            'spo2_window':   spo2_win.tolist(),
        }
        all_records.append(record)
        win_count += 1

    print(f"  ✅ {win_count} windows created for {pid}")

#  Save Dataset 
df_all = pd.DataFrame(all_records)

# CSV (without raw windows — too large)
csv_cols = [c for c in df_all.columns if c not in ['flow_window', 'thorac_window', 'spo2_window']]
csv_path = os.path.join(args.out_dir, 'breathing_dataset.csv')
df_all[csv_cols].to_csv(csv_path, index=False)
print(f"\n✅ CSV saved: {csv_path}  ({len(df_all)} windows)")

# Pickle (with raw windows — for CNN training)
pkl_path = os.path.join(args.out_dir, 'breathing_dataset.pkl')
df_all.to_pickle(pkl_path)
print(f"✅ Pickle saved: {pkl_path}")

# Label distribution
print("\n📊 Label distribution:")
print(df_all['label'].value_counts().to_string())
print("\n📊 Per-participant window counts:")
print(df_all.groupby(['participant', 'label']).size().to_string())