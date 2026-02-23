import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# --- PATH FIX: Ensures 'preprocessing' and 'model' are discoverable ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.audio_utils import load_wav
from preprocessing.feature_extraction import extract_log_mel
from model.swin_tser import SwinTSER

# --- SETTINGS FOR CPU SPEED & ACCURACY ---
DEVICE = "cpu"
EPOCHS = 15
BATCH_SIZE = 64 
LEARNING_RATE = 0.001
DATASET_PATH = "datasets"
# Emotions: angry, disgust, fear, happy, neutral, ps (surprise), sad
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]

class SERDataset(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []
        
        print("--- Scanning Datasets (TESS/EMOVO) ---")
        file_list = []
        for root, _, files in os.walk(DATASET_PATH):
            for file in files:
                if file.endswith(".wav"):
                    f_low = file.lower()
                    for idx, emo in enumerate(EMOTIONS):
                        if emo in f_low:
                            file_list.append((os.path.join(root, file), idx))
                            break
        
        if not file_list:
            print(f"❌ Error: No files found in {DATASET_PATH}. Check your folder structure.")
            sys.exit()

        print(f"--- Loading {len(file_list)} samples into RAM for High Speed ---")
        for path, label in tqdm(file_list):
            try:
                waveform = load_wav(path)
                features = extract_log_mel(waveform)
                
                # Normalization for faster convergence
                features = (features - features.mean()) / (features.std() + 1e-6)

                # Ensure fixed size 128x128
                max_len = 128
                if features.shape[0] < max_len:
                    pad = torch.zeros(max_len - features.shape[0], features.shape[1])
                    features = torch.cat([features, pad], dim=0)
                else:
                    features = features[:max_len]
                
                self.data.append(features)
                self.labels.append(label)
            except Exception as e:
                continue

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(self.labels[idx])

if __name__ == "__main__":
    # 1. Prepare Data
    full_ds = SERDataset()
    train_len = int(0.8 * len(full_ds))
    val_len = len(full_ds) - train_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 2. Initialize Model
    model = SwinTSER(num_classes=len(EMOTIONS)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n🚀 Starting Training on {DEVICE}...")
    best_acc = 0

    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_acc = 100. * correct / total
        
        # Validation Phase
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                _, predicted = outputs.max(1)
                v_total += y.size(0)
                v_correct += predicted.eq(y).sum().item()
        
        val_acc = 100. * v_correct / v_total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model/model_weights.pth")
            print(f" ⭐ New Best Model Saved ({val_acc:.2f}%)")

    print(f"\n✅ Training Complete! Final Best Accuracy: {best_acc:.2f}%")