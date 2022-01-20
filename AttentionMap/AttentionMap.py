from torch import Tensor, clip
from torch import nn

from Sparse import SparseSigmoid, SparseSigmoid2d, ReLUWithSparsity2d
from modules.Sparse.Sparse.modules.activations import ReLUWithSparsity

from torch.nn import functional as F

class AttentionMap(nn.Module):
    def __init__(self, in_features: int) -> None:
        super(AttentionMap, self).__init__()
        self.attention_map = nn.Sequential(*[
            nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False),
            # SparseSigmoid(beta=1e-6, rho=0.1)
            ReLUWithSparsity(beta=1e-6, rho=0.15),
            # nn.ReLU(inplace=True)
        ]) 

    def forward(self, x: Tensor) -> Tensor:
        # att = self.attention_map(x)
        # N, _, W, H = att.shape
        # return F.softmax(att.view(N,1,-1), dim=2).view(N,1,W,H)
        return self.attention_map(x)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **args) -> None:
        super(Conv2d, self).__init__()
        self.attention_map = AttentionMap(in_features=in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, **args)

    def forward(self, x: Tensor) -> Tensor:
        att = self.attention_map(x)
        return att*self.conv(x)
