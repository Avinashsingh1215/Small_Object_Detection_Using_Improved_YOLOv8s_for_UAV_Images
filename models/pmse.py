import torch
import torch.nn as nn
import torch.nn.functional as F

class ELayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.conv3(out)
        return self.act(out + residual)

class PMSE(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, 1),
            ELayer(in_channels//reduction, in_channels//reduction),
            nn.Conv2d(in_channels//reduction, in_channels//reduction, 3, dilation=2, padding=2),
            nn.SiLU()
        )
        try:
            from torchvision.ops import DeformConv2d
            self.use_deform = True
            self.branch2_conv1 = nn.Conv2d(in_channels, in_channels//reduction, 1)
            self.branch2_el = ELayer(in_channels//reduction, in_channels//reduction)
            self.branch2_deform = DeformConv2d(in_channels//reduction, in_channels//reduction, 3, padding=1)
            self.branch2_act = nn.SiLU()
        except ImportError:
            self.use_deform = False
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//reduction, 1),
                ELayer(in_channels//reduction, in_channels//reduction),
                nn.Conv2d(in_channels//reduction, in_channels//reduction, 3, padding=1),
                nn.SiLU()
            )
        self.branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction, 1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels//reduction, in_channels//reduction, 3, padding=1),
            nn.SiLU()
        )
        self.conv_reduce = nn.Conv2d(in_channels//reduction * 3, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b1 = self.branch1(x)
        if self.use_deform:
            b2_in = self.branch2_conv1(x)
            b2_in = self.branch2_el(b2_in)
            offset = torch.zeros(b2_in.size(0), 2*3*3, b2_in.size(2), b2_in.size(3), device=x.device)
            b2 = self.branch2_deform(b2_in, offset)
            b2 = self.branch2_act(b2)
        else:
            b2 = self.branch2(x)
        b3 = self.branch3(x)
        if b3.shape[2:] != x.shape[2:]:
            b3 = F.interpolate(b3, size=x.shape[2:], mode='bilinear', align_corners=False)
        cat = torch.cat([b1, b2, b3], dim=1)
        attn = self.conv_reduce(cat)
        attn = self.sigmoid(attn)
        return x * attn

class LightweightPMSE(PMSE):
    def __init__(self, in_channels, reduction=16):
        super().__init__(in_channels, reduction)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, 1),
            nn.Conv2d(in_channels//reduction, in_channels//reduction, 3, groups=in_channels//reduction, padding=1),
            nn.Conv2d(in_channels//reduction, in_channels//reduction, 1),
            nn.SiLU()
        )
