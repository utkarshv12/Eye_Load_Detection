# 👁️ Eye Load Detector

A Python-based Machine Learning project that detects **eye strain (eye load)** by analyzing eye movement and blink-related features. The project includes data preprocessing, feature extraction, model training, and real-time detection using a webcam.

---

## 🚀 Features

- 📊 Eye movement data preprocessing
- 🧠 Machine Learning model training
- 👁️ Real-time webcam eye load detection
- 📈 Feature extraction from eye-tracking data
- 💾 Trained model support
- 🔬 Simulation dataset for testing

---

## 🛠️ Technologies Used

- Python 3.x
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Joblib

---

## 📁 Project Structure

```
eye-load-detector/
│── data/                  # Dataset (ignored in Git)
│── models/                # Trained model (ignored in Git)
│── __pycache__/           # Python cache (ignored)
│── features.py
│── preprocess.py
│── simulator.py
│── train.py
│── realtime_demo.py
│── webcam_realtime.py
│── requirements.txt
│── README.md
│── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/utkarshv12/eye-load-detector.git
```

Move into the project folder:

```bash
cd eye-load-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the Machine Learning Model

```bash
python train.py
```

### Run Real-Time Webcam Detection

```bash
python webcam_realtime.py
```

or

```bash
python realtime_demo.py
```

---

## 📂 Dataset

The project uses simulated eye-tracking data for training and testing.

If the dataset is not included in this repository, you can generate or add your own eye-tracking data inside the `data/` folder.

---

## 📊 Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Extraction
4. Model Training
5. Model Evaluation
6. Real-Time Eye Load Prediction

---

## 📸 Output

The application can:

- Detect eye movement
- Estimate eye load
- Display predictions in real time
- Process webcam video frames

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Add new feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Utkarsh**

GitHub: https://github.com/utkarshv12

