# features.py
import numpy as np, pandas as pd

def extract_features_from_df(df, fs=120, realtime=False):
    """
    Extract features from a trial dataframe (offline) or realtime buffer.
    If realtime=True, skips label/subj/trial (not available).
    """
    out = {}
    dur = df['t'].iloc[-1] - df['t'].iloc[0]
    out['dur'] = dur

    # pupil stats (baseline-corrected)
    out['pupil_mean'] = np.nanmean(df['pupil_bc'])
    out['pupil_std']  = np.nanstd(df['pupil_bc'])
    out['pupil_p95']  = np.nanpercentile(df['pupil_bc'].dropna(), 95)

    # pupil slope
    try:
        out['pupil_slope'] = np.polyfit(df['t'], df['pupil_bc'], 1)[0]
    except Exception:
        out['pupil_slope'] = 0.0

    # blink rate: count invalid segments
    invalid = (~df['valid2'])
    inv_groups = (invalid != invalid.shift()).cumsum()
    blinks = 0
    for g, sub in df.groupby(inv_groups):
        if not sub['valid2'].all():
            blinks += 1
    out['blink_rate'] = blinks / (dur/60.0 + 1e-9)  # blinks per minute

    # saccade proxy: fraction of time speed > threshold
    speed = df['speed'].fillna(0).values
    out['saccade_prop'] = np.mean(speed > 100)  # threshold in px/s
    out['speed_mean'] = np.mean(speed)

    # fixation proxy: mean dispersion
    out['gaze_std_x'] = np.nanstd(df['gx'])
    out['gaze_std_y'] = np.nanstd(df['gy'])

    # gaze entropy (coarse)
    try:
        hist, _, _ = np.histogram2d(df['gx'].dropna(), df['gy'].dropna(), bins=12)
        p = hist.flatten() / (hist.sum()+1e-9)
        p = p[p>0]
        out['gaze_entropy'] = -(p * np.log(p)).sum()
    except Exception:
        out['gaze_entropy'] = 0.0

    # --- only include these for offline training ---
    if not realtime:
        if 'label' in df.columns:
            out['label'] = int(df['label'].iloc[0])
        if 'subj' in df.columns:
            out['subj'] = int(df['subj'].iloc[0])
        if 'trial' in df.columns:
            out['trial'] = int(df['trial'].iloc[0])

    return out


def build_feature_table(preprocessed_list):
    feats = []
    for df in preprocessed_list:
        f = extract_features_from_df(df, realtime=False)
        feats.append(f)
    return pd.DataFrame(feats)


if __name__ == '__main__':
    from preprocess import preprocess_all
    dfs = preprocess_all()
    ft = build_feature_table(dfs)
    print(ft.head())
    ft.to_csv('data/features_table.csv', index=False)
