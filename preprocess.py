# preprocess.py
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
import glob

def lowpass(signal, fs=120, cutoff=4, order=4):
    b,a = butter(order, cutoff/(fs/2), btype='low')
    return filtfilt(b,a,signal)

def preprocess_file(fname, fs=120):
    df = pd.read_csv(fname)
    # ensure sorted
    df = df.sort_values('t').reset_index(drop=True)
    # replace infs and large negatives
    df['pupil'] = df['pupil'].replace([np.inf,-np.inf], np.nan)
    # interpolate short gaps in pupil
    df['pupil'] = df['pupil'].interpolate(limit=int(0.2*fs)).bfill().ffill()
    # mark valid if pupil finite and valid flag set
    df['valid2'] = (~df['pupil'].isna()) & (df['valid']==1)
    # smooth pupil if sufficient samples
    if df['pupil'].notna().sum() > int(0.5*fs):
        try:
            df['pupil_f'] = lowpass(df['pupil'].fillna(method='ffill').fillna(method='bfill').values, fs=fs)
        except Exception:
            df['pupil_f'] = df['pupil'].fillna(method='ffill').fillna(method='bfill').values
    else:
        df['pupil_f'] = df['pupil'].fillna(df['pupil'].median())

    # baseline correction: first 3s median
    nbase = min(int(3*fs), len(df))
    baseline = np.nanmedian(df['pupil_f'].iloc[:nbase])
    df['pupil_bc'] = df['pupil_f'] - baseline

    # simple velocity (pixels/s) for saccade proxy
    gx = df['gx'].fillna(method='ffill').fillna(method='bfill').values
    gy = df['gy'].fillna(method='ffill').fillna(method='bfill').values
    vx = np.gradient(gx) * fs
    vy = np.gradient(gy) * fs
    df['speed'] = np.hypot(vx, vy)
    return df

def preprocess_all(sim_dir='data/sim'):
    files = glob.glob(sim_dir + '/*.csv')
    dfs = []
    for f in files:
        if f.endswith('metadata.csv'):
            continue
        df = preprocess_file(f)
        dfs.append(df)
    return dfs

if __name__ == '__main__':
    dfs = preprocess_all()
    print("Preprocessed", len(dfs), "files")
