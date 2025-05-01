# ASL Hand Gesture Detection 🤟

A real-time ASL alphabet classifier using Mediapipe for hand landmark detection and a Random Forest classifier.

---

## 📷 Features
- ASL A–Z detection
- Real-time webcam classification
- Lightweight and fast
- Easy dataset collection and model training

---

## 🔧 Setup & Usage Guide

### ✅ Requirements
- Python 3.7 to 3.10
- Dependencies: `opencv-python`, `mediapipe`, `scikit-learn`, `numpy`

---

#### 🧭 Step 1: Upgrade pip

```bash
python -m pip install --upgrade pip
```
---

#### 🐍 Step 2: (Optional) Set Up Virtual Environment with `pyenv`
```bash
brew install pyenv
pyenv install 3.10.13
pyenv shell 3.10.13
python -m venv asl-env
source asl-env/bin/activate
```
---
#### 📦 Step 3: Install Dependencies
Using `requirements.txt`:
```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install scikit-learn==1.3.2
pip install opencv-python==4.8.1.78
pip install mediapipe==0.10.0
pip install numpy
```
---
#### 📸 Step 4: Collect ASL Hand Gesture Data
Run the image data collection script:
```bash
python collect_img_data.py
```
- This opens your webcam
- You'll be prompted to label each gesture (e.g., `A`, `B`, `C`...)
- Landmark data will be saved to `data.pickle`
---
#### 🧠 Step 5: Train the Classifier
```bash
python training_classifier.py
```
- Trains a Random Forest model using your dataset
- Saves the model to `model.p`
---
#### 🎯 Step 6: Run Real-Time Classifier
```bash
python testing_classifier.py
```
- Activates webcam
- Detects your hand and displays the predicted ASL letter live
- Draws bounding box and landmarks on your hand
---
Thank you and let me know if you have any questions/information! ☺️
