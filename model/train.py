import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- FIX: Adds the root directory to the path so 'preprocessing' is found ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.audio_utils import load_wav
from preprocessing.feature_extraction import extract_log_mel
from model.swin_tser import SwinTSER

# Configs based on SwinTSER Paper & TESS/EMOVO Datasets
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH_SIZE = 16
DATASET_PATH = "datasets"
# The 7 emotions found in TESS/EMOVO
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"] 

class SERDataset(Dataset):
    def __init__(self):
        self.files = []
        self.labels = []

        # Walk through TESS and EMOVO folders
        for root, _, files in os.walk(DATASET_PATH):
            for file in files:
                if file.endswith(".wav"):
                    file_lower = file.lower()
                    # Check which emotion label is in the filename
                    for idx, emotion in enumerate(EMOTIONS):
                        # 'ps' stands for Pleasant Surprise in TESS
                        if emotion in file_lower:
                            self.files.append(os.path.join(root, file))
                            self.labels.append(idx)
                            break

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            waveform = load_wav(self.files[idx])
            features = extract_log_mel(waveform)

            # Standardize shape for Transformer input (Time x Mel)
            max_len = 128 
            if features.shape[0] < max_len:
                pad = torch.zeros(max_len - features.shape[0], features.shape[1])
                features = torch.cat([features, pad], dim=0)
            else:
                features = features[:max_len]

            return features, torch.tensor(self.labels[idx])
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros(128, 128), torch.tensor(0)

if __name__ == "__main__":
    print("--- SwinTSER Training Initialized ---")
    dataset = SERDataset()
    print(f"Total samples found: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Check your dataset folder path!")
        sys.exit()

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
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "model/model_weights.pth")
    print("✅ Model Weights Saved Successfully!")