import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from audio_utils import load_wav
from feature_extraction import extract_mfcc

# --- CONFIG ---
DATASET_PATH = "../dataset/train"
NUM_CLASSES = 4       # adjust to your number of emotions
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DATASET ---
class SERDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            for f in os.listdir(cls_dir):
                if f.endswith(".wav"):
                    self.files.append(os.path.join(cls_dir, f))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform = load_wav(self.files[idx])
        features = extract_mfcc(waveform)  # shape: (time, n_mfcc)
        # Pad/trim to 100 frames
        max_len = 100
        if features.shape[0] < max_len:
            pad = torch.zeros(max_len - features.shape[0], features.shape[1])
            features = torch.cat([features, pad], dim=0)
        else:
            features = features[:max_len]
        return features, self.labels[idx]

# --- MODEL ---
class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(40, 64, kernel_size=5, stride=1)
        self.pool = nn.AdaptiveMaxPool1d(50)
        self.fc1 = nn.Linear(64 * 50, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, n_mfcc, time)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- TRAIN ---
dataset = SERDataset(DATASET_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SimpleCNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "../models/emotion_model.pth")
print(" Model saved to ../models/emotion_model.pth")