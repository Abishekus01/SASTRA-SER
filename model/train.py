import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from preprocessing.audio_utils import load_wav
from preprocessing.feature_extraction import extract_log_mel
from model.swin_tser import SwinTSER
from utils.config import DATASET_PATH, EMOTIONS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 16

class SERDataset(Dataset):
    def __init__(self):
        self.files = []
        self.labels = []

        for speaker in os.listdir(DATASET_PATH):
            speaker_path = os.path.join(DATASET_PATH, speaker)

            for emotion in EMOTIONS:
                emotion_path = os.path.join(speaker_path, emotion)

                if not os.path.exists(emotion_path):
                    continue

                for file in os.listdir(emotion_path):
                    if file.endswith(".wav"):
                        self.files.append(os.path.join(emotion_path, file))
                        self.labels.append(EMOTIONS.index(emotion))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform = load_wav(self.files[idx])
        features = extract_log_mel(waveform)

        max_len = 128
        if features.shape[0] < max_len:
            pad = torch.zeros(max_len - features.shape[0], features.shape[1])
            features = torch.cat([features, pad], dim=0)
        else:
            features = features[:max_len]

        label = torch.tensor(self.labels[idx])
        return features, label

dataset = SERDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SwinTSER(num_classes=len(EMOTIONS)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "model/model_weights.pth")
print("✅ SwinTSER model saved!")