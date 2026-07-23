# simulator.py
import numpy as np
import pandas as pd
from tqdm import trange
import os

def simulate_trial(duration_s=30, fs=120, load='low', subj_id=0, trial_id=0, screen_w=1920, screen_h=1080):
    """
    Simulate raw eye-tracking stream for one trial.
    Returns DataFrame with columns: t, gx, gy, pupil, valid, subj, trial, label
    """
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    # baseline gaze wander around center
    cx, cy = screen_w/2, screen_h/2
    # load affects pupil mean and micro-movements
    if load == 'low':
        pupil_base = 3.2  # mm
        noise_scale = 1.0
        saccade_rate = 0.8  # per sec
    else:
        pupil_base = 4.1
        noise_scale = 1.6
        saccade_rate = 1.6

    # simulate gaze x,y as noisy walk + saccades
    gx = cx + np.cumsum(np.random.randn(n) * 0.5 * noise_scale)
    gy = cy + np.cumsum(np.random.randn(n) * 0.5 * noise_scale)
    # occasional saccades: jumps
    if saccade_rate > 0:
        num_sacs = int(duration_s * saccade_rate)
        for _ in range(num_sacs):
            idx = np.random.randint(0, n)
            gx[idx: idx+int(0.02*fs)] += np.random.randn() * 200 * (0.5 + noise_scale)
            gy[idx: idx+int(0.02*fs)] += np.random.randn() * 120 * (0.5 + noise_scale)

    # pupil size: baseline + low-freq responses + blinks
    pupil = pupil_base + 0.1 * np.sin(2 * np.pi * 0.2 * t)  # slow oscillation
    pupil += np.random.randn(n) * 0.02 * noise_scale
    # task-evoked phasic increases (bursts) more in high load
    for ev in range(int(duration_s / 5)):
        ev_t = int((ev * 5 + np.random.rand()*2) * fs)
        if ev_t + int(0.5*fs) < n:
            amp = 0.05 if load == 'low' else 0.15
            pupil[ev_t:ev_t+int(0.6*fs)] += amp * np.hanning(int(0.6*fs))

    # blinks -> pupil=NaN and brief invalid samples
    valid = np.ones(n, dtype=int)
    blink_rate = 10/60.0 if load=='low' else 6/60.0  # blinks/min adjustment (example)
    num_blinks = max(1, int(duration_s * blink_rate))
    for _ in range(num_blinks):
        bidx = np.random.randint(0, n)
        blen = np.random.randint(int(0.05*fs), int(0.2*fs))
        pupil[bidx:bidx+blen] = np.nan
        valid[bidx:bidx+blen] = 0
        # slight gaze loss
        gx[bidx:bidx+blen] = np.nan
        gy[bidx:bidx+blen] = np.nan

    df = pd.DataFrame({
        't': t,
        'gx': gx,
        'gy': gy,
        'pupil': pupil,
        'valid': valid,
        'subj': subj_id,
        'trial': trial_id,
        'label': 1 if load=='high' else 0
    })
    return df

def simulate_dataset(out_dir='data/sim', subjects=8, trials_per_subject=10, fs=120):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for subj in range(subjects):
        for trial in range(trials_per_subject):
            load = 'high' if np.random.rand() < 0.5 else 'low'
            dur = np.random.randint(20,40)
            df = simulate_trial(duration_s=dur, fs=fs, load=load, subj_id=subj, trial_id=trial)
            fname = f'{out_dir}/subj{subj}_trial{trial}.csv'
            df.to_csv(fname, index=False)
            rows.append({'file': fname, 'subj': subj, 'trial': trial, 'label': df['label'].iloc[0]})
    meta = pd.DataFrame(rows)
    meta.to_csv(f'{out_dir}/metadata.csv', index=False)
    print("Simulated dataset saved to", out_dir)

if __name__ == '__main__':
    simulate_dataset()
