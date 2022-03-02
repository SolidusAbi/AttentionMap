import torch
import torch.nn.functional as F
import matplotlib as mpl
import torchvision.utils as utils

def visualize_attention(n_rows, img, c, scale_factor = 0, activation = 'softmax'):
    '''
        Parameters
        ----------
            n_rows:

            img: Tensor, shape (N,C,W,H)

            c: Tensor, shape (N,1,Wc,Hc)

            scale_factor: int
                Scale factor for upsampling 'c' if it is necessary to obtain the 
                size (W, H).

            activation: str
                'softmax' or 'sigmoid'
    '''
    img = img.permute((0,2,3,1)).cpu().numpy()
    N,C,W,H = c.shape

    if activation == 'softmax':
        a = torch.softmax(c.view(N,C,-1), dim=2).view(N,C,W,H)
    else: 
        a = torch.sigmoid(c)

    if scale_factor != 0:
        a = F.interpolate(a, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    a_norm = a / torch.max(a.flatten(2), 2)[0].reshape(N,C,1,1)
    a_norm_img = a_norm.permute((0,2,3,1)).mul(255).byte().cpu().numpy()
    
    vis = []
    cm_jet = mpl.cm.get_cmap('jet')
    for i in range(N):
        a_cm = cm_jet(a_norm_img[i].squeeze())[:,:,:3]
        result = .6*img[i]  + .4*a_cm 
        vis.append(torch.tensor(result).permute(2,0,1))

    return utils.make_grid(vis, n_rows, normalize=True, scale_each=True)