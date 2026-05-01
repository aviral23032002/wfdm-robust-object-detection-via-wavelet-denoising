import torch
import torch.nn as nn

class HaarDWT(nn.Module):
    """Discrete Wavelet Transform to split spatial features into frequencies."""
    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        LL = x1 + x2 + x3 + x4
        HL = -x1 - x2 + x3 + x4
        LH = -x1 + x2 - x3 + x4
        HH = x1 - x2 - x3 + x4
        return LL, LH, HL, HH

class HaarIWT(nn.Module):
    """Inverse Wavelet Transform to reconstruct spatial features."""
    def forward(self, LL, LH, HL, HH):
        B, C, H, W = LL.shape
        out = torch.zeros(B, C, H * 2, W * 2, device=LL.device)
        out[:, :, 0::2, 0::2] = LL - HL - LH + HH
        out[:, :, 1::2, 0::2] = LL - HL + LH - HH
        out[:, :, 0::2, 1::2] = LL + HL - LH - HH
        out[:, :, 1::2, 1::2] = LL + HL + LH + HH
        return out

class WFDM_NoWts(nn.Module):
    """Wavelet Denoising with NO Learned Weights (Hard Thresholding)."""
    
    # We accept 'c' so YOLO doesn't crash when passing channel dimensions
    def __init__(self, c):
        super().__init__()
        self.dwt = HaarDWT()
        self.iwt = HaarIWT()
        
    def forward(self, x):
        # 1. Split into low (geometry) and high (noise) frequencies
        LL, LH, HL, HH = self.dwt(x)
        
        # 2. Hard Math: Erase the high-frequency noise entirely
        LH = torch.zeros_like(LH)
        HL = torch.zeros_like(HL)
        HH = torch.zeros_like(HH)
        
        # 3. Rebuild the tensor using only the clean low-frequency geometry
        return self.iwt(LL, LH, HL, HH)