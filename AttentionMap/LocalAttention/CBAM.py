import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_rate=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        mip = max(4, in_planes // reduction_rate)
           
        self.fc = nn.Sequential(
                nn.Conv2d(in_planes, mip, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(mip, in_planes, 1, bias=False)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_rate=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()

        self.channelAtt = ChannelAttention(in_channels, reduction_rate)
        self.spatialAtt = SpatialAttention(spatial_kernel_size)

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        x = x*self.channelAtt(x)
        return self.spatialAtt(x)