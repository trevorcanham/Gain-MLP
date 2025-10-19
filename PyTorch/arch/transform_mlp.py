from torch import nn
import tinycudann as tcnn
import torch

class TransformMLP(nn.Module):
    def __init__(self, n_input_dims, n_output_dims, config):
        super(TransformMLP, self).__init__()
        self.n_output_dims = n_output_dims
        self.gamma = nn.Parameter(torch.tensor([1]).float())
        self.net = tcnn.NetworkWithInputEncoding(n_input_dims=n_input_dims, n_output_dims=n_output_dims, encoding_config=config["encoding"], network_config=config["network"])

    def forward(self, x):
        
        xXY = x[:,0:2]
        xRGB = x[:,2:6]
        xRGBgam = xRGB**self.gamma
        x_ = torch.cat((xXY,xRGBgam),dim=1)
        out = self.net(x_)

        return out