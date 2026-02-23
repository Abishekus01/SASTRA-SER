import torch
import torch.nn as nn

class SwinTSER(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Fast Feature Extractor (Depthwise Separable)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, groups=32, padding=1), # Depthwise
            nn.Conv2d(32, 64, kernel_size=1), # Pointwise
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, groups=64, padding=1),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)