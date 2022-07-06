from torch import nn, Tensor
from enum import Enum

# Just for convolutionals layers
class SpatialSqueezeType(Enum):
    average= 1,
    convolution = 2

class GridAttentionBlock(nn.Module):
    '''
        Params
        ------
        img_size: Tuple, shape: (H, W)
            Image resolution where H is the height and W is the width.
        
        in_channels: int,
            Number of channels in the input image
        
        reduction_rate: int,
            Reduction ratio for dimensional-reduction.
        
        spatial_squeeze: SpatialSqueezeType,
            It defines how to apply the spatial squeeze operation in each axis. The operation is described below:
                1- Average Pool: The spatial axis is squezed using and nn.AdaptiveAvgPool2d following by a convolutional layer
                    for applying the squeeze operation the in channels.
                2- Convolutional: The spatial and channel squeeze operations in the same convolutional layer. It is neccesary 
                    to indicate the spatial resolution of the input for defining the kernel size in the convolutional layer.
    '''
    def __init__(self, img_size: tuple, in_channels: int, reduction_rate: int, 
                    spatial_squeeze = SpatialSqueezeType.average) -> None:
        super(GridAttentionBlock, self).__init__()
        mip = max(4, in_channels // reduction_rate)
        H, W = img_size
        
        self.squeeze_h = nn.Sequential(
            *(  nn.AdaptiveAvgPool2d((None, 1)), nn.Conv2d(in_channels, mip, 1, bias=False)) if (spatial_squeeze == SpatialSqueezeType.average_pool)
                else (nn.Conv2d(in_channels, mip, (1, W), bias=False), ),
            nn.BatchNorm2d(mip),
            nn.SiLU()
        )

        self.squeeze_w = nn.Sequential(
            *(  nn.AdaptiveAvgPool2d((1, None)), nn.Conv2d(in_channels, mip, 1, bias=False),) if (spatial_squeeze == SpatialSqueezeType.average_pool)
                else (nn.Conv2d(in_channels, mip, (H, 1), bias=False), ),
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
