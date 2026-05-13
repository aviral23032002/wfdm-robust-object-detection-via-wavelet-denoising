import torch
import torch.nn as nn

from models.cbam import CBAM


class WaveletBranch(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(4))

        self.wave_conv = nn.Sequential(

            nn.Conv2d(c1 * 4, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),

            nn.Conv2d(
                c2,
                c2,
                3,
                padding=1,
                groups=c2,
                bias=False
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):

        A = x[:, :, 0::2, 0::2]
        B = x[:, :, 0::2, 1::2]
        C = x[:, :, 1::2, 0::2]
        D = x[:, :, 1::2, 1::2]

        LL = (A + B + C + D) * 0.5
        LH = (A + B - C - D) * 0.5
        HL = (A - B + C - D) * 0.5
        HH = (A - B - C + D) * 0.5

        out = torch.cat([
            self.alpha[0] * LL,
            self.alpha[1] * LH,
            self.alpha[2] * HL,
            self.alpha[3] * HH
        ], dim=1)

        return self.wave_conv(out)


class WFDM(nn.Module):

    """
    Ultralytics-compatible WFDM
    """

    def __init__(self, c1, c2, *args, **kwargs):
        super().__init__()

        # ---------------------------------------------------
        # Standard Conv Branch
        # ---------------------------------------------------

        self.conv_branch = nn.Sequential(

            nn.Conv2d(
                c1,
                c2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(c2),

            nn.SiLU()
        )

        # ---------------------------------------------------
        # Wavelet Branch
        # ---------------------------------------------------

        self.wavelet_branch = WaveletBranch(c1, c2)

        # ---------------------------------------------------
        # Fusion
        # ---------------------------------------------------

        self.fuse = nn.Sequential(

            nn.Conv2d(
                c2 * 2,
                c2,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(c2),

            nn.SiLU()
        )

        # ---------------------------------------------------
        # Attention
        # ---------------------------------------------------

        self.cbam = CBAM(c2)

        # ---------------------------------------------------
        # Residual
        # ---------------------------------------------------

        self.residual = nn.Sequential(

            nn.Conv2d(
                c1,
                c2,
                kernel_size=1,
                stride=2,
                bias=False
            ),

            nn.BatchNorm2d(c2)
        )

    def forward(self, x):

        conv_feat = self.conv_branch(x)

        wave_feat = self.wavelet_branch(x)

        out = torch.cat([conv_feat, wave_feat], dim=1)

        out = self.fuse(out)

        out = self.cbam(out)

        return out + self.residual(x)