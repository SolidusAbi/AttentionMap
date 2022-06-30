# Resolution Guided Pooling (RGP) approach:
# Delving deep into spatial pooling for squeeze-and-excitation networks. Pattern Recognition, 121, 108159. 

import torch
from torch import nn

class RGPFusion(nn.Module):
    def __init__(self, img_size: tuple, in_planes: int, kernel_size=3):
        super(RGPFusion, self).__init__()
        H, W = img_size

        if isinstance(kernel_size, tuple):
            conv_kernel_size = (H//kernel_size[0]) * (W//kernel_size[1])
        else:    
            conv_kernel_size = (H//kernel_size) * (W//kernel_size)
        
        self.pool = nn.AvgPool2d(kernel_size)
        self.fusion = nn.Conv1d(in_planes, in_planes, kernel_size=conv_kernel_size, groups=in_planes, bias=False)
        self.bn = nn.Sequential(nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool(x).flatten(2)
        rgp_emb = self.fusion(out).squeeze(dim=2)
        return self.bn(rgp_emb)

class RGPAttentionBlock(nn.Module):
    def __init__(self, img_size: tuple, in_planes: int, reduction_rate=16, rg_kernel_size=3) -> None:
        super(RGPAttentionBlock, self).__init__()
        mip = max(4, in_planes // reduction_rate)

        self.att = nn.Sequential(
            RGPFusion(img_size, in_planes, kernel_size=rg_kernel_size),
            nn.Linear(in_planes, mip, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mip, in_planes, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, c, _, _ = x.shape
        att = self.att(x).view(bs, c, 1, 1)
        return att.expand_as(x)