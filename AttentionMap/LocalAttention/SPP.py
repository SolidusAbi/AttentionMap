# Spatial pyramid pooling (SPP) approach:
# Delving deep into spatial pooling for squeeze-and-excitation networks. Pattern Recognition, 121, 108159. 
import torch
from torch import nn

class SPPFusion(nn.Module):
    def __init__(self, in_planes: int):
        super(SPPFusion, self).__init__()
        self.pool1x1 = nn.AdaptiveAvgPool2d(1)
        self.pool2x2 = nn.AdaptiveAvgPool2d(2)
        self.pool4x4 = nn.AdaptiveAvgPool2d(4)

        self.fusion = nn.Conv1d(in_planes, in_planes, kernel_size=21, groups=in_planes)
        self.bn = nn.BatchNorm1d(in_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
            Input
            -----
                x: torch.Tensor, shape: (B, C, W, H)
            Return
            ------
                infoFusion: torch.Tensor, shape: (B, C)
                    Batch Normalized output with contains the spatial information 
        '''
        out1x1 = self.pool1x1(x).flatten(2)
        out2x2 = self.pool2x2(x).flatten(2)
        out4x4 = self.pool4x4(x).flatten(2)

        spp_emb = self.fusion(torch.cat([out4x4.flatten(2), out2x2.flatten(2), out1x1.flatten(2)], dim=2)).squeeze(dim=2)
        return self.bn(spp_emb)

class SPPAttentionBlock(nn.Module):
    def __init__(self, in_planes: int, reduction_rate=16) -> None:
        super(SPPAttentionBlock, self).__init__()
        mip = max(4, in_planes // reduction_rate)

        self.att = nn.Sequential(
            SPPFusion(in_planes),
            nn.Linear(in_planes, mip, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mip, in_planes, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, c, _, _ = x.shape
        att = self.att(x).view(bs, c, 1, 1)
        return att.expand_as(x)