import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # For a nice progress bar

# --- PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.audio_utils import load_wav
from preprocessing.feature_extraction import extract_log_mel
from model.swin_tser import SwinTSER

# Configs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH_SIZE = 32  # Increased for faster hardware utilization
LEARNING_RATE = 0.0005 
DATASET_PATH = "datasets"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"] 

class SERDataset(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []
        
        print("--- Loading Data into Memory (First time only) ---")
        temp_files = []
        for root, _, files in os.walk(DATASET_PATH):
            for file in files:
                if file.endswith(".wav"):
                    file_lower = file.lower()
                    for idx, emotion in enumerate(EMOTIONS):
                        if emotion in file_lower:
                            temp_files.append((os.path.join(root, file), idx))
                            break
        
        # Pre-process everything now so epochs are 10x faster
        for path, label in tqdm(temp_files):
            try:
                waveform = load_wav(path)
                features = extract_log_mel(waveform)
                
                # Normalize features (Crucial for accuracy)
                features = (features - features.mean()) / (features.std() + 1e-6)

                max_len = 128 
                if features.shape[0] < max_len:
                    pad = torch.zeros(max_len - features.shape[0], features.shape[1])
                    features = torch.cat([features, pad], dim=0)
                else:
                    features = features[:max_len]
                
                self.data.append(features)
                self.labels.append(label)
            except Exception:
                continue

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(self.labels[idx])

if __name__ == "__main__":
    print(f"--- SwinTSER Training Initialized on {DEVICE} ---")
    full_dataset = SERDataset()
    
    # Split into Train and Validation for accuracy tracking
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = SwinTSER(num_classes=len(EMOTIONS)).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) # AdamW for better Transformer convergence
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        train_acc = 100. * correct / total
        scheduler.step()

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model/model_weights.pth")

    print(f"✅ Training Complete. Best Val Accuracy: {best_acc:.2f}%")