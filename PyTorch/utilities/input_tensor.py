# Network input initialization for encoding and decoding MLP based HDR residuals (gain or gamma maps)
# If used, cite: Trevor D. Canham, SaiKiran Tedla, Michael J. Murdoch and Michael S. Brown.
# Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP. ICCV 2025

import torch
from torch import nn

# Set data types and devices for network input

class InputTensor(nn.Module):
    def __init__(self, input, gt, device):
        super(InputTensor, self).__init__()
        self.input = input.float().to(device)
        self.gt = gt.float().to(device)
        self.length = self.input.shape[0]

    def forward(self, xs):
        with torch.no_grad():
            xs = xs * torch.tensor([self.length], device=xs.device).float()
            indices = xs.clamp(min=0, max=self.length - 1).long()
            return self.input[indices], self.gt[indices]
