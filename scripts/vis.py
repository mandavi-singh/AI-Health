import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Argument Parser
parser = argparse.ArgumentParser(description='Sleep Data Visualization')
parser.add_argument('-name', type=str, required=True,
                    help='Path to participant folder e.g. "Data/AP20"')
args = parser.parse_args()

participant_path = args.name
participant_id = os.path.basename(participant_path)

print(f"Processing participant: {participant_id}")

# Helper: Read Signal Files (Flow, Thorac, SPO2) 
def read_signal_file(filepath):
    """
    Reads signal files with header info and data section.
    Format: DD.MM.YYYY HH:MM:SS,mmm; value
    Returns a DataFrame with 'timestamp' and 'value' columns.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find where 'Data:' section starts
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == 'Data:':
            data_start = i + 1
            break

    records = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        try:
            ts_part, val_part = line.split(';')
            ts_part = ts_part.strip().replace(',', '.')
            # Try multiple formats
            for fmt in ('%d.%m.%Y %H:%M:%S.%f', '%d.%m.%Y %H:%M:%S'):
                try:
                    ts = pd.to_datetime(ts_part, format=fmt)
                    break
                except Exception:
                    continue
            records.append({'timestamp': ts, 'value': float(val_part.strip())})
        except Exception:
            continue

    return pd.DataFrame(records)

#  Helper: Read Flow Events File
def read_flow_events(filepath):
    """
    Reads flow events file.
    Format: DD.MM.YYYY HH:MM:SS,mmm-HH:MM:SS,mmm; duration; EventType; SleepStage
    Returns a DataFrame with start_time, end_time, event_type columns.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    records = []
    for line in lines:
        line = line.strip()
        if not line or ';' not in line:
            continue
        # Skip header lines
        if any(line.startswith(k) for k in ['Signal', 'Start', 'Unit']):
            continue
        try:
            parts = line.split(';')
            time_range = parts[0].strip()
            event_type = parts[2].strip()

            # Split time range: "DD.MM.YYYY HH:MM:SS,mmm-HH:MM:SS,mmm"
            date_part = time_range[:10]  # DD.MM.YYYY
            rest = time_range[11:]       # HH:MM:SS,mmm-HH:MM:SS,mmm
            start_str, end_str = rest.split('-')

            start_dt = pd.to_datetime(f"{date_part} {start_str.replace(',', '.')}",
                                      format='%d.%m.%Y %H:%M:%S.%f')
            # End time might cross midnight, handle that
            end_dt = pd.to_datetime(f"{date_part} {end_str.replace(',', '.')}",
                                    format='%d.%m.%Y %H:%M:%S.%f')
            if end_dt < start_dt:
                end_dt += pd.Timedelta(days=1)

            records.append({
                'start_time': start_dt,
                'end_time': end_dt,
                'event_type': event_type
            })
        except Exception:
            continue

    return pd.DataFrame(records)

#  Find Files 
def find_file(folder, keyword):
    """Find file in folder whose name contains the keyword (case-insensitive)."""
    for f in os.listdir(folder):
        if keyword.lower() in f.lower():
            return os.path.join(folder, f)
    return None

flow_file    = find_file(participant_path, 'flow') 
# Make sure we don't pick up 'flow events' for flow signal
flow_signal_file = None
flow_events_file = None
for f in os.listdir(participant_path):
    fl = f.lower()
    if 'flow' in fl and 'event' in fl:
        flow_events_file = os.path.join(participant_path, f)
    elif 'flow' in fl and 'event' not in fl:
        flow_signal_file = os.path.join(participant_path, f)

thorac_file  = find_file(participant_path, 'thorac')
spo2_file    = find_file(participant_path, 'spo2') or find_file(participant_path, 'spo₂') or find_file(participant_path, 'sp02')

print(f"  Flow signal : {flow_signal_file}")
print(f"  Thorac      : {thorac_file}")
print(f"  SpO2        : {spo2_file}")
print(f"  Flow Events : {flow_events_file}")

# Read Data 
flow_df   = read_signal_file(flow_signal_file)
thorac_df = read_signal_file(thorac_file)
spo2_df   = read_signal_file(spo2_file)
events_df = read_flow_events(flow_events_file)

print(f"  Loaded: Flow={len(flow_df)}, Thorac={len(thorac_df)}, SpO2={len(spo2_df)}, Events={len(events_df)}")

#  Convert to time-from-start (minutes) for x-axis 
t0 = flow_df['timestamp'].min()

def to_minutes(df):
    df = df.copy()
    df['minutes'] = (df['timestamp'] - t0).dt.total_seconds() / 60
    return df

flow_df   = to_minutes(flow_df)
thorac_df = to_minutes(thorac_df)
spo2_df   = to_minutes(spo2_df)

# Event times in minutes
events_df['start_min'] = (events_df['start_time'] - t0).dt.total_seconds() / 60
events_df['end_min']   = (events_df['end_time']   - t0).dt.total_seconds() / 60

# Color map for events (only Hypopnea and Obstructive Apnea, others get default color)
EVENT_COLORS = {
    'Hypopnea':          '#FF8C00',   # Orange
    'Obstructive Apnea': '#CC0000',   # Red
}
DEFAULT_COLOR = '#9B59B6'

#  Plot 
os.makedirs('Visualizations', exist_ok=True)
output_pdf = os.path.join('Visualizations', f'{participant_id}_visualization.pdf')

with PdfPages(output_pdf) as pdf:

    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
    fig.suptitle(f'Sleep Study — Participant: {participant_id}\n'
                 f'Recording start: {t0.strftime("%d %b %Y  %H:%M:%S")}',
                 fontsize=14, fontweight='bold', y=0.98)

    signals = [
        (flow_df,   'Nasal Airflow',       'steelblue',  'Flow (a.u.)'),
        (thorac_df, 'Thoracic Movement',   'seagreen',   'Movement (a.u.)'),
        (spo2_df,   'SpO₂',               'crimson',    'SpO₂ (%)'),
    ]

    for ax, (df, title, color, ylabel) in zip(axes, signals):
        # Downsample for faster rendering (plot every 4th point for 32 Hz signals)
        step = 4 if len(df) > 100000 else 1
        ax.plot(df['minutes'].iloc[::step], df['value'].iloc[::step],
                color=color, linewidth=0.4, alpha=0.85)

        # Overlay events as shaded regions
        for _, ev in events_df.iterrows():
            ec = EVENT_COLORS.get(ev['event_type'], DEFAULT_COLOR)
            ax.axvspan(ev['start_min'], ev['end_min'],
                       alpha=0.25, color=ec, linewidth=0)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time from recording start (minutes)', fontsize=10)

    # ── Legend ──
    legend_patches = [
        mpatches.Patch(color=EVENT_COLORS['Hypopnea'],          alpha=0.5, label='Hypopnea'),
        mpatches.Patch(color=EVENT_COLORS['Obstructive Apnea'], alpha=0.5, label='Obstructive Apnea'),
    ]
    axes[0].legend(handles=legend_patches, loc='upper right', fontsize=8, framealpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Page 2: Zoomed view of first 60 minutes ──
    fig2, axes2 = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
    fig2.suptitle(f'Zoomed View — First 60 Minutes | {participant_id}',
                  fontsize=13, fontweight='bold')

    for ax, (df, title, color, ylabel) in zip(axes2, signals):
        mask = df['minutes'] <= 60
        ax.plot(df.loc[mask, 'minutes'], df.loc[mask, 'value'],
                color=color, linewidth=0.6, alpha=0.9)

        ev_mask = events_df['start_min'] <= 60
        for _, ev in events_df[ev_mask].iterrows():
            ec = EVENT_COLORS.get(ev['event_type'], DEFAULT_COLOR)
            ax.axvspan(ev['start_min'], min(ev['end_min'], 60),
                       alpha=0.3, color=ec, linewidth=0)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes2[-1].set_xlabel('Time from recording start (minutes)', fontsize=10)
    axes2[0].legend(handles=legend_patches, loc='upper right', fontsize=8, framealpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig2, dpi=150, bbox_inches='tight')
    plt.close(fig2)

print(f"\n✅ PDF saved: {output_pdf}")
