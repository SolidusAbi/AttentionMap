from typing import Tuple
import torch
from torch import Tensor, clip
from torch import nn

# from Sparse import SparseSigmoid, SparseSigmoid2d, ReLUWithSparsity2d
from torch.nn import functional as F

class AttentionMap(nn.Module):
    def __init__(self, in_features: int) -> None:
        from modules.Sparse.Sparse.modules.activations import ReLUWithSparsity
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

'''
'''
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

'''
Grid attention block
Reference papers
Attention-Gated Networks https://arxiv.org/abs/1804.05338 & https://arxiv.org/abs/1808.08114
Reference code
https://github.com/ozan-oktay/Attention-Gated-Networks
'''
class GridAttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize=False):
        super(GridAttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
    def forward(self, l, g):
        N, C, W, H = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        if self.up_factor > 1: # En el original, se aplica un 'subsampler factor'... Básicamente sería cambiar la configuración del kernel de W_l
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_)) # batch_sizex1xWxH
        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l) # batch_sizexCxWxH
        if self.normalize_attn:
            output = f.view(N,C,-1).sum(dim=2) # weighted sum
        else:
            output = F.adaptive_avg_pool2d(f, (1,1)).view(N,C)
        return c.view(N,1,W,H), output