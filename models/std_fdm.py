import torch
import torch.nn as nn

class StandardFDM(nn.Module):
    """
    Standard Feature Denoising Module (Spatial Filtering).
    Uses a bottleneck architecture to mathematically smooth out noise.
    """
    def __init__(self, c):
        super().__init__()
        # Bottleneck compression 
        hidden_dim = max(c // 2, 16) 
        
        self.conv1 = nn.Conv2d(c, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.SiLU()
        
        # Spatial convolution to smooth the noise
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.act2 = nn.SiLU()
        
        # Expansion back to original channel size
        self.conv3 = nn.Conv2d(hidden_dim, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(c)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        return out


class SafeFDM(nn.Module):
    """
    Channel-Wise Zero-Initialized Residual Wrapper.
    Allows the network to dynamically apply denoising only to specific feature channels.
    """
    def __init__(self, c):
        super().__init__()
        self.fdm = StandardFDM(c) 
        
        # CHANNEL-WISE ATTENTION GATE
        # Creates a dedicated weight for exactly each channel, initialized to 0.0.
        # Shape: (Batch=1, Channels=c, Height=1, Width=1) 
        self.alpha = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        # PyTorch automatically broadcasts the channel weights during multiplication.
        # Output = Clean Features + (Filtered Features * Channel Weights)
        return x + (self.fdm(x) * self.alpha)