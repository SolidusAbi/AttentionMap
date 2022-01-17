from torch import Tensor
from torch import nn

from Sparse import SparseSigmoid

class AttentionMap(nn.Module):
    def __init__(self, in_features: int) -> None:
        super(AttentionMap, self).__init__()
        self.attention_map = nn.Sequential(*[
            nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False),
            SparseSigmoid(beta=1e-6, rho=0.05)
        ]) 

    def forward(self, x: Tensor) -> Tensor:
        return self.attention_map(x)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **args) -> None:
        super(Conv2d, self).__init__()
        self.attention_map = AttentionMap(in_features=in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, **args)

    def forward(self, x: Tensor) -> Tensor:
        att = self.attention_map(x)
        x = self.conv(x)
        return att*x
