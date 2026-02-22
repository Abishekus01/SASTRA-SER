import torch
from audio_utils import load_wav
from feature_extraction import extract_mfcc
from train import SimpleCNN, NUM_CLASSES, DEVICE

# Load model
model = SimpleCNN(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("../models/emotion_model.pth", map_location=DEVICE))
model.eval()

# Classes (adjust if different)
classes = ["angry", "happy", "neutral", "sad"]

def predict(file_path):
    waveform = load_wav(file_path)
    features = extract_mfcc(waveform)
    max_len = 100
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

# --- Example ---
if __name__ == "__main__":
    path = "../dataset/test/sample.wav"
    print("Predicted emotion:", predict(path))