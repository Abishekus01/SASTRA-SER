import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from preprocessing.feature_extraction import extract_features
from utils.config import DATASET_PATH, MODEL_PATH, EMOTIONS

X, y = [], []

for speaker in os.listdir(DATASET_PATH):
    speaker_path = os.path.join(DATASET_PATH, speaker)

    for emotion in EMOTIONS:
        emotion_path = os.path.join(speaker_path, emotion)

        if not os.path.exists(emotion_path):
            continue

        for file in os.listdir(emotion_path):
            if file.endswith(".wav"):
                file_path = os.path.join(emotion_path, file)

                try:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(emotion)
                except:
                    print("Error in:", file_path)

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)

# ✅ SCALE FEATURES
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ✅ TRAIN MODEL
model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
model.fit(X, y)

# ✅ SAVE BOTH MODEL + SCALER
joblib.dump((model, scaler), MODEL_PATH)

print("✅ Model trained & saved at:", MODEL_PATH)
