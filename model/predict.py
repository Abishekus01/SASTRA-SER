import torch
import os
import sys

# --- FIX: Adds the root directory to the path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.audio_utils import load_wav
from preprocessing.feature_extraction import extract_log_mel
from model.swin_tser import SwinTSER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Mapping 'ps' to 'Surprise' for better UI display
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Surprise", "Sad"]

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SwinTSER(num_classes=len(EMOTIONS)).to(DEVICE)
        weights_path = "model/model_weights.pth"
        if os.path.exists(weights_path):
            _model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        _model.eval()
    return _model

def predict_emotion(file_path):
    try:
        waveform = load_wav(file_path)
        features = extract_log_mel(waveform)

        max_len = 128
        if features.shape[0] < max_len:
            pad = torch.zeros(max_len - features.shape[0], features.shape[1])
            features = torch.cat([features, pad], dim=0)
        else:
            features = features[:max_len]

        features = features.unsqueeze(0).to(DEVICE)

        model = get_model()
        with torch.no_grad():
            outputs = model(features)
            pred = torch.argmax(outputs, dim=1).item()
        return EMOTIONS[pred]
    except Exception as e:
        print(f"Inference error: {e}")
        return None