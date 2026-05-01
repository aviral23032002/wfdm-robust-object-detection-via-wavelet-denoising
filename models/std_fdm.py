import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class StandardFDM(nn.Module):
    """A standard spatial feature denoising module with a residual connection."""
    
    # FIX: Only accept one argument (c) since input and output channels are the same
    def __init__(self, c):
        super().__init__()
        # Bottleneck architecture: Reduce -> Filter -> Expand
        c_hidden = int(c / 2)
        self.cv1 = Conv(c, c_hidden, 1, 1)       # Reduce channels
        self.cv2 = Conv(c_hidden, c_hidden, 3, 1) # Spatial filtering
        self.cv3 = Conv(c_hidden, c, 1, 1)       # Expand back to original channels

    def forward(self, x):
        # The crucial residual connection: x + filtered_x
        return x + self.cv3(self.cv2(self.cv1(x)))