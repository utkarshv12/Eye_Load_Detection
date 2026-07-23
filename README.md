# Eye Load Detector

A machine learning project to **detect cognitive load from eye movements** using RNN/CNN models.  
This project processes eye-tracking data, extracts features, trains models, and provides real-time cognitive load estimation using a webcam or simulated data.

---

## 🚀 Features
- Preprocessing of eye-tracking data  
- Feature extraction from gaze patterns  
- Model training (RNN/CNN)  
- Real-time demo with webcam  
- Simulator for testing without hardware  

---

## 📂 Project Structure
eye_load_detector/
│── features.py # Extracts features from data
│── preprocess.py # Cleans & preprocesses input
│── train.py # Trains the RNN/CNN model
│── simulator.py # Runs simulated experiments
│── realtime_demo.py # Real-time detection demo
│── webcam_realtime.py # Webcam-based live demo
│── data/ # Sample datasets


## ⚙️ Installation

Clone this repository:

git clone https://github.com/inboxprashant/eye_load_detector.git
cd eye_load_detector

Create and activate a virtual environment:
Copy code: python -m venv venv310

.\venv310\Scripts\activate  # On Windows

Install dependencies:
Copy code: pip install -r requirements.txt

▶️ Usage
Run real-time demo with webcam
Copy code: python webcam_realtime.py

Run simulator
Copy code: python simulator.py

Train the model
Copy code: python train.py
📧 Contact
GitHub: inboxprashant

Email: inboxprashantkumar@gmail.com