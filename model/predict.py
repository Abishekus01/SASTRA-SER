#predict.py
import joblib
import os
from preprocessing.feature_extraction import extract_features
from utils.config import MODEL_PATH

def predict_emotion(audio_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train first.")

    model, scaler = joblib.load(MODEL_PATH)

    features = extract_features(audio_path).reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)

    return prediction[0]