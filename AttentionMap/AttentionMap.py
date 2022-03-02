from typing import Tuple
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
            ReLUWithSparsity(beta=1e-5, rho=0.1),
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

import torch
class LinearAttentionMap(nn.Module):
    '''
        Linear attention block

        Parameters
        ----------
            normalize: Bool, 
                Define if the output components will add up to 1 using a Softmax.
    '''
    def __init__(self, in_channels, normalize=True) -> None:
        super(LinearAttentionMap, self).__init__()
        self.normalize = normalize
        # Learnable parameters for the “parametrised compatibiliy” where c = <u,l+g>
        self.u = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0, bias=False)

    def forward(self, l, g) -> Tuple:
        '''
            Parameters
            ----------
                l: torch.Tensor, shape (N,C,W,H)
                    Local features obtained from the output of a convolutional layer.
                g: torch.Tensor, shape (N,C,1,1)
                    The global features, 'global image descriptor', obtained from a fully-connected layer.

            Returns
            -------
                c: torch.Tensor, shape (N,C,W,H)
                    The compatibility between l and g. In this implementation is used the 
                    "parametrised compatibility".

                g: torch.Tensor, shape (N,C)
                    Output of the attention mechanism which is used for classification.
        '''
        N,C,W,H = l.shape
        c = self.u(l+g)

        # Calculate the Attention Weights a from the Compatibility Scores 'c'
        if self.normalize:
            a = torch.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)

        # Calculate the Final Output of the Attention Mechanism
        g = torch.mul(a.expand_as(l), l)
        if self.normalize:
            g = g.view(N,C,-1).sum(dim=2) # The sum of all elements per channel 
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C) # The average per channel

        # C 
        return c.view(N,1,W,H), g


