from torch import nn, Tensor

class ConvolutionalAttentionBlock(nn.Module):
    def __init__(self, img_size: tuple, in_channels: int, reduction_rate: int, groups=False, bias=True) -> None:
        super(ConvolutionalAttentionBlock, self).__init__()
        mip = max(8, in_channels // reduction_rate)
        
        self.squeeze_h = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(in_channels, mip, 1, bias=False),
            nn.BatchNorm2d(mip),
            nn.SiLU()
        )

        self.squeeze_w = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(in_channels, mip, 1, bias=False),
            nn.BatchNorm2d(mip),
            nn.SiLU()
        )

        self.excitation = nn.Sequential(
            nn.Conv2d(mip, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x_h = self.squeeze_h(x) # Height descriptor shape: (C x W x 1)
        x_w = self.squeeze_w(x) # Width descriptor shape: (C x 1 x H)

        # Coordinate attention
        coordAtt = self.excitation(x_h+x_w)
        # TODO: Concatenate x_h and x_w
        
        return coordAtt