from torch import nn

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super(ProjectorBlock, self).__init__()
        self.proj = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        return self.proj(x)