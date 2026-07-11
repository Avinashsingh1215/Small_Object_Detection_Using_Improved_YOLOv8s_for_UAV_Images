import torch
import torch.nn as nn

class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_h = self.conv1(x_h)
        x_h = self.bn1(x_h)
        x_h = self.act(x_h)
        x_h = self.conv_h(x_h)
        x_h = torch.sigmoid(x_h)
        x_w = self.pool_w(x)
        x_w = x_w.permute(0,1,3,2)
        x_w = self.conv1(x_w)
        x_w = self.bn1(x_w)
        x_w = self.act(x_w)
        x_w = self.conv_w(x_w)
        x_w = torch.sigmoid(x_w).permute(0,1,3,2)
        return x * x_h * x_w
