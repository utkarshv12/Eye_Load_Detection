# realtime_demo.py
import time, numpy as np, pandas as pd
from simulator import simulate_trial
from features import extract_features_from_df
from joblib import load
from collections import deque
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

MODEL_PATH = 'models/rf_load_detector.joblib'
if not os.path.exists(MODEL_PATH):
    raise SystemExit("Train model first (run train.py)")

model = load(MODEL_PATH)
fs = 120
ring = deque(maxlen=int(6 * fs))  # 6 sec pupil buffer
probs = deque(maxlen=60)          # keep last 60 predictions
times = deque(maxlen=60)          # matching time stamps

# ---- Matplotlib setup ----
plt.style.use("seaborn-v0_8")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6))
fig.suptitle("Realtime Cognitive Load Detection (Simulated)", fontsize=14)

# Pupil trace plot
line_pupil, = ax1.plot([], [], lw=2, color="blue")
ax1.set_ylabel("Pupil size (a.u.)")
ax1.set_xlim(0, 6)
ax1.set_ylim(-1, 2)

# Probability plot
line_prob, = ax2.plot([], [], lw=2, color="red")
ax2.set_ylabel("Load Probability")
ax2.set_xlabel("Time (s)")
ax2.set_ylim(0, 1)
ax2.set_xlim(0, 60)

start_time = time.time()
frame_counter = 0

def update(frame):
    global frame_counter
    frame_counter += 1

    # simulate 1 sec chunk
    df_chunk = simulate_trial(
        duration_s=1.0,
        fs=fs,
        load="high" if np.random.rand() < 0.5 else "low",
        subj_id=0,
        trial_id=frame_counter
    )
    ring.extend(df_chunk.to_dict("records"))
    buf = pd.DataFrame(list(ring))

    # ---- minimal preprocessing ----
    buf['pupil'] = (
        buf['pupil']
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit=30)
        .bfill()
        .ffill()
    )
    buf['valid2'] = (~buf['pupil'].isna()) & (buf['valid'] == 1)
    buf['pupil_f'] = buf['pupil']
    nbase = min(int(1*fs), len(buf))
    baseline = buf['pupil_f'].iloc[:nbase].median() if nbase > 0 else 0.0
    buf['pupil_bc'] = buf['pupil_f'] - baseline

    gx = buf['gx'].fillna(method='ffill').fillna(method='bfill').values
    gy = buf['gy'].fillna(method='ffill').fillna(method='bfill').values
    vx = np.gradient(gx) * fs
    vy = np.gradient(gy) * fs
    buf['speed'] = np.hypot(vx, vy)

    # ---- extract features ----
    feat = extract_features_from_df(buf)
    X = pd.DataFrame([{k:v for k,v in feat.items() if k not in ['label','subj','trial']}])
    prob = model.predict_proba(X)[0, 1]

    # store for plotting
    elapsed = int(time.time() - start_time)
    times.append(elapsed)
    probs.append(prob)

    # update pupil trace
    t_rel = buf['t'] - buf['t'].iloc[0]
    line_pupil.set_data(t_rel, buf['pupil_bc'])
    ax1.set_ylim(buf['pupil_bc'].min()-0.2, buf['pupil_bc'].max()+0.2)

    # update probability trace
    line_prob.set_data(times, probs)
    ax2.set_xlim(max(0, elapsed-60), elapsed)
    
    return line_pupil, line_prob

ani = FuncAnimation(fig, update, interval=1000, blit=False)
plt.tight_layout()
plt.show()
