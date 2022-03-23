from turtle import forward
import torch
from torch import nn

class CoordinateInformationEmbeding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CoordinateInformationEmbeding, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.emb = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        ])

    def forward(self, x: torch.Tensor) -> tuple:
        '''
            return
            ------
                cd_h: torch.Tensor, shape (N, Cout, W, 1)
                cd_w: torch.Tensor, shape (N, Cout, 1, H) 
        '''
        _,_,w,h = x.size()

        x_h = self.pool_h(x) # Shape: (N, C, W, 1)
        x_w = self.pool_w(x) # Shape: (N, C, 1, H)
        x_w = x_w.permute(0, 1, 3, 2) # Shape: (N, C, H, 1)

        coord_descriptor = torch.cat([x_w, x_h], dim=2) # Shape: (N, C, W+H, 1)
        coord_descriptor = self.emb(coord_descriptor)
        cd_w, cd_h = torch.split(coord_descriptor, [h, w], dim=2)
        cd_w = cd_w.permute(0, 1, 3, 2)

        return (cd_h, cd_w)



class CoordinateAttentionBlock(nn.Module):
    r'''

        Parameters
        ----------
            in_channels: int
            out_channels: int
            reduction: int
    '''
    def __init__(self, in_channels:int, out_channels:int, reduction=32) -> None:
        super(CoordinateAttentionBlock, self).__init__()
        
        mip = max(4, in_channels // reduction)
        self.emb = CoordinateInformationEmbeding(in_channels, mip)

        self.att_h = nn.Sequential(*[
            nn.Conv2d(mip, out_channels, 1, bias=True),
            nn.Sigmoid()
        ]) 
        
        self.att_w = nn.Sequential(*[
            nn.Conv2d(mip, out_channels, 1, bias=True),
            nn.Sigmoid()
        ]) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Coordinate Descriptor por Height and Weight
        cd_h, cd_w = self.emb(x)

        attention_h = self.att_h(cd_h)
        attention_w = self.att_w(cd_w)

        return attention_h * attention_w

class CoordAttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, att_reduction=8) -> None:
        super(CoordAttentionConv, self).__init__()

        self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        self.att_block = CoordinateAttentionBlock(out_channels, out_channels, att_reduction)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x * self.att_block(x)