from torch import nn
import tinycudann as tcnn
import torch

class BlockMLP(nn.Module):
    def __init__(self, n_blocks, n_input_dims, n_output_dims, config):
        super(BlockMLP, self).__init__()
        self.nets = nn.ModuleList()
        self.n_output_dims = n_output_dims
        for net in range(n_blocks):
            net = tcnn.NetworkWithInputEncoding(n_input_dims=n_input_dims, n_output_dims=n_output_dims, encoding_config=config["encoding"], network_config=config["network"])
            self.nets.append(net)

    def forward(self, x):
        
        out = torch.zeros(len(self.nets), x.shape[1], self.n_output_dims).to(x.device)
        for i, net in enumerate(self.nets):
            out[i,:] = net(x[i,:])
        return out