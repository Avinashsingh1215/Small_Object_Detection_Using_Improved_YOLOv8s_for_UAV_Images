import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFeatureFusion(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x1, x2, x3=None):
        w = torch.relu(self.weight)
        if x3 is None:
            fused = (w[0] * x1 + w[1] * x2) / (w.sum() + self.eps)
        else:
            fused = (w[0] * x1 + w[1] * x2 + w[2] * x3) / (w.sum() + self.eps)
        return fused

class SCFPN(nn.Module):
    def __init__(self, channels_list=[256, 512, 1024], out_channels=256):
        super().__init__()
        self.lat_layers = nn.ModuleList([nn.Conv2d(ch, out_channels, 1) for ch in channels_list])
        self.topdown_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.SiLU()) for _ in range(len(channels_list)-1)])
        self.bottomup_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1), nn.SiLU()) for _ in range(len(channels_list)-1)])
        self.fusions = nn.ModuleList([WeightedFeatureFusion(out_channels) for _ in range(len(channels_list))])

    def forward(self, features):
        laterals = [lat(f) for lat, f in zip(self.lat_layers, features)]
        topdown = [laterals[-1]]
        for i in range(len(laterals)-2, -1, -1):
            up = F.interpolate(topdown[-1], size=laterals[i].shape[2:], mode='nearest')
            fused = self.fusions[i](laterals[i], up)
            conv = self.topdown_convs[i]
            topdown.append(conv(fused))
        topdown = topdown[::-1]
        bottomup = [topdown[0]]
        for i in range(1, len(topdown)):
            down = self.bottomup_convs[i-1](bottomup[-1])
            fused = self.fusions[i](topdown[i], down)
            bottomup.append(fused)
        return bottomup[:3]  # P2, P3, P4
