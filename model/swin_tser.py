import torch
import torch.nn as nn

class SwinTSER(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Optimized Patching for CPU
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=4, stride=4), # Reduced channels from 96 to 48
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        # Single efficient Transformer Layer
        self.layer = nn.TransformerEncoderLayer(d_model=48, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=1)
        
        self.head = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1) 
        x = self.patch_embed(x) # (B, 48, 32, 32)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, 1024, 48)
        x = self.encoder(x)
        x = x.mean(dim=1) # Global Average Pooling
        return self.head(x)