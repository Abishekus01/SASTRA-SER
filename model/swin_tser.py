import torch
import torch.nn as nn
import torch.nn.functional as F

class SwinTSER(nn.Module):
    def __init__(self, num_classes=7):
        super(SwinTSER, self).__init__()
        
        # 1. Patch Partitioning (Akinpelu Paper Fig 2)
        # Converts (128x128) Mel-spectrogram into patches
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=4, stride=4),
            nn.LayerNorm([96, 32, 32])
        )

        # 2. Simplified Swin Transformer Block
        # Captures Local (Window) and Global temporal relationships
        self.transformer_block = nn.ModuleList([
            SwinTransformerLayer(dim=96, num_heads=3, window_size=8)
            for _ in range(2)
        ])

        # 3. Global Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(96, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x input: (Batch, Time, Mel) -> (B, 128, 128)
        if x.dim() == 3:
            x = x.unsqueeze(1) # (B, 1, 128, 128)
            
        x = self.patch_embed(x) # (B, 96, 32, 32)
        
        for layer in self.transformer_block:
            x = layer(x)
            
        x = self.avgpool(x) # (B, 96, 1, 1)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

class SwinTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C) # Flatten for attention
        x = self.norm1(x)
        
        # Window-based Multi-head Self Attention
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        
        x = x + self.mlp(self.norm2(x))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2) # Reshape back
        return x