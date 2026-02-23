# model/predict.py
# SwinTSER inference for Flask

import torch
from preprocessing.audio_utils import load_wav
from preprocessing.feature_extraction import extract_log_mel
from model.train import SwinTSER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classes = ["angry", "happy", "neutral", "sad"]

model = SwinTSER(num_classes=len(classes)).to(DEVICE)
model.load_state_dict(torch.load("model/model_weights.pth", map_location=DEVICE))
model.eval()

def predict_emotion(file_path):
    waveform = load_wav(file_path)
    features = extract_log_mel(waveform)   # (time, mel)

    max_len = 128
    if features.shape[0] < max_len:
        pad = torch.zeros(max_len - features.shape[0], features.shape[1])
        features = torch.cat([features, pad], dim=0)
    else:
        features = features[:max_len]

    features = features.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(features)
        pred = torch.argmax(outputs, dim=1).item()

    return classes[pred]