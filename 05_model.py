"""
model.py
--------
Custom 3D ResNet architecture for binary classification of 3D OCT volumes.

Architecture overview (10 stages):
    Stage 1:  3x Conv3d (32 → 32 → 64 filters, stride=2)
    Stage 2:  Bottleneck block (64 → 64)
    Stage 3:  Bottleneck block (64 → 64)
    Stage 4:  Conv3d (64 → 64, stride=2)
    Stage 5:  Bottleneck block (64 → 64)
    Stage 6:  Bottleneck block (64 → 64)
    Stage 7:  Conv3d (64 → 128, stride=2)
    Stage 8:  Bottleneck block (128 → 128)
    Stage 9:  Bottleneck block (128 → 128)
    Stage 10: MaxPool3d (7x7x7)
    FC1:      Linear(1024 → 128) + ReLU
    FC2:      Linear(128 → 1)    [BCEWithLogitsLoss — no sigmoid here]

Total parameters: ~3.2M

Usage:
    from src.model import ResNet3D, WeightedBCELoss

Reference:
    Naghibzadeh, K. et al. "Enhancing Optic Nerve Head Abnormality Detection
    through Deep Learning on 3D OCT Images." (MSc Thesis, Tarbiat Modares
    University, 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Bottleneck Block ───────────────────────────────────────────────────────────

class Bottleneck(nn.Module):
    """
    Residual bottleneck block with pre-activation (BN → ReLU → Conv).

    Uses identity skip connections; input and output channel sizes must match.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Convolution stride. Default: 1.
    """

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super(Bottleneck, self).__init__()
        self.bn1   = nn.BatchNorm3d(inplanes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes,   planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + residual
        return out


# ── 3D ResNet ──────────────────────────────────────────────────────────────────

class ResNet3D(nn.Module):
    """
    Custom 3D ResNet for binary classification of volumetric OCT images.

    Input shape: (batch, 1, 110, 110, 110)
    Output shape: (batch, 1)  — raw logits for BCEWithLogitsLoss

    Args:
        num_classes (int): Number of output units. Default: 1 (binary).
        input_shape (tuple): (channels, D, H, W). Default: (1, 110, 110, 110).
    """

    def __init__(self, num_classes: int = 1, input_shape: tuple = (1, 110, 110, 110)):
        super(ResNet3D, self).__init__()

        in_channels = input_shape[0]

        # Stage 1 — Initial feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
        )

        # Stages 2-3 — Residual blocks at 64 channels
        self.bot2 = Bottleneck(64, 64, stride=1)
        self.bot3 = Bottleneck(64, 64, stride=1)

        # Stage 4 — Downsampling
        self.conv4 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
        )

        # Stages 5-6 — Residual blocks at 64 channels
        self.bot5 = Bottleneck(64, 64, stride=1)
        self.bot6 = Bottleneck(64, 64, stride=1)

        # Stage 7 — Upscale channels + downsampling
        self.conv7 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
        )

        # Stages 8-9 — Residual blocks at 128 channels
        self.bot8 = Bottleneck(128, 128, stride=1)
        self.bot9 = Bottleneck(128, 128, stride=1)

        # Stage 10 — Global spatial pooling
        self.conv10 = nn.MaxPool3d(kernel_size=7)

        # Fully connected head
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bot2(x)
        x = self.bot3(x)
        x = self.conv4(x)
        x = self.bot5(x)
        x = self.bot6(x)
        x = self.conv7(x)
        x = self.bot8(x)
        x = self.bot9(x)
        x = self.conv10(x)
        x = x.view(x.size(0), -1)   # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ── Loss Function ──────────────────────────────────────────────────────────────

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss using BCEWithLogitsLoss.

    Applies a positive class weight to address class imbalance between
    papilledema (majority) and normal (minority) samples.

    Args:
        pos_weight (float or torch.Tensor): Weight for the positive class.
    """

    def __init__(self, pos_weight):
        super(WeightedBCELoss, self).__init__()
        if not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        self.pos_weight = pos_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            input, target, pos_weight=self.pos_weight
        )
