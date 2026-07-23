import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from joblib import load
from collections import deque
from features import extract_features_from_df
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------ Load trained model ------------------
MODEL_PATH = 'models/rf_load_detector.joblib'
model = load(MODEL_PATH)

# ------------------ Mediapipe FaceMesh ------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# ------------------ Webcam ------------------
cap = cv2.VideoCapture(0)

# ------------------ Buffers ------------------
fs = 30  # webcam ~30 FPS
ring = deque(maxlen=int(6 * fs))  # keep last 6s of data
probs, times = deque(maxlen=60), deque(maxlen=60)
start_time = time.time()

# ------------------ Matplotlib setup ------------------
plt.style.use("seaborn-v0_8")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
fig.suptitle("Realtime Cognitive Load Detection (Webcam)", fontsize=14)
line_pupil, = ax1.plot([], [], lw=2, color="blue")
line_prob, = ax2.plot([], [], lw=2, color="red")
ax1.set_ylabel("Blink proxy / Pupil (a.u.)")
ax2.set_ylabel("Load Probability")
ax2.set_ylim(0, 1)
ax2.set_xlabel("Time (s)")

# ------------------ Helper: Eye Aspect Ratio ------------------
def eye_aspect_ratio(landmarks, eye_indices):
    v1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    v2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    h1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    h2 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    vertical = np.linalg.norm(v1 - v2)
    horizontal = np.linalg.norm(h1 - h2)
    return vertical / horizontal if horizontal > 0 else 0

# Indices for eyes in Mediapipe FaceMesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ------------------ Main Update Loop ------------------
def update(frame):
    global cap, start_time
    ret, img = cap.read()
    if not ret:
        return line_pupil, line_prob

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)

    pupil_proxy, gx, gy, valid = np.nan, np.nan, np.nan, 0

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        h, w, _ = img.shape

        # Draw landmarks
        for idx in LEFT_EYE + RIGHT_EYE:
            x, y = int(lm[idx].x * w), int(lm[idx].y * h)
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        # EAR blink proxy
        ear_left = eye_aspect_ratio(lm, LEFT_EYE)
        ear_right = eye_aspect_ratio(lm, RIGHT_EYE)
        pupil_proxy = (ear_left + ear_right) / 2

        # Gaze midpoint
        gx = (lm[LEFT_EYE[0]].x + lm[RIGHT_EYE[3]].x) / 2
        gy = (lm[LEFT_EYE[0]].y + lm[RIGHT_EYE[3]].y) / 2
        valid = 1

    # Buffer
    t = time.time() - start_time
    ring.append({"t": t, "gx": gx, "gy": gy, "pupil": pupil_proxy, "valid": valid})

    buf = pd.DataFrame(list(ring))
    buf['pupil'] = buf['pupil'].interpolate(limit=5).bfill().ffill()
    buf['valid2'] = (~buf['pupil'].isna()) & (buf['valid'] == 1)
    buf['pupil_f'] = buf['pupil']
    buf['pupil_bc'] = buf['pupil_f'] - buf['pupil_f'].iloc[:min(15, len(buf))].median()

    # Gaze speed (safe check)
    if len(buf) > 1:
        gx_arr = buf['gx'].fillna(method='ffill').fillna(method='bfill').values
        gy_arr = buf['gy'].fillna(method='ffill').fillna(method='bfill').values
        vx = np.gradient(gx_arr) * fs
        vy = np.gradient(gy_arr) * fs
        buf['speed'] = np.hypot(vx, vy)
    else:
        buf['speed'] = 0

    # ---- Predict ----
    if len(buf) > fs:
        try:
            feat = extract_features_from_df(buf, realtime=True)  # <<< FIX HERE
            X = pd.DataFrame([feat])  # already cleaned for realtime
            prob = model.predict_proba(X)[0, 1]

            elapsed = int(t)
            times.append(elapsed)
            probs.append(prob)

            # Webcam overlay (traffic light)
            if prob < 0.33:
                color, text = (0, 255, 0), "LOW LOAD"
            elif prob < 0.66:
                color, text = (0, 255, 255), "MEDIUM LOAD"
            else:
                color, text = (0, 0, 255), "HIGH LOAD"

            cv2.putText(img, f"Load: {text} ({prob:.2f})", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.rectangle(img, (30, 60), (30 + int(prob * 200), 90), color, -1)
            cv2.rectangle(img, (30, 60), (230, 90), (255, 255, 255), 2)

            # Update pupil trace
            line_pupil.set_data(buf['t'] - buf['t'].iloc[0], buf['pupil_bc'])
            ax1.set_xlim(0, 6)
            ax1.set_ylim(buf['pupil_bc'].min() - 0.1, buf['pupil_bc'].max() + 0.1)

            # Update probability trace
            line_prob.set_data(times, probs)
            ax2.set_xlim(max(0, elapsed - 60), elapsed)
        except Exception as e:
            print("Prediction skipped:", e)

    # Show webcam in non-blocking way
    cv2.imshow("Realtime Cognitive Load", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        plt.close("all")
        cap.release()
        cv2.destroyAllWindows()

    return line_pupil, line_prob

# ------------------ Run ------------------
ani = FuncAnimation(fig, update, interval=100, blit=False)
plt.tight_layout()
plt.show()

# Cleanup
cap.release()
cv2.destroyAllWindows()
