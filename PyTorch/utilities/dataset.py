# Dataloader classes for encoding and decoding MLP based HDR residuals (gain or gamma maps)
# If used, cite: Trevor D. Canham, SaiKiran Tedla, Michael J. Murdoch and Michael S. Brown.
# Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP. ICCV 2025

import colour
import cv2
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import logging
import os
import torch.nn.functional as F
from functools import partial
from .io import decode, decodeInput, encode, get_len_input_type, sNorm
import pdb

### encoding block ###

class encDataset(Dataset):

    def __init__(self, sdr_imgfile, hdr_imgfile, encode_mode, mlp_input = "rgb", 
                imgs_dir = "/home/tcanham/gamut-mlp/gMap-mlp/data/", num_pixels = 0, 
                use_coords = True):
        self.imgs_dir = imgs_dir
        self.num_pixels = num_pixels
        self.sdr_imgfile = sdr_imgfile
        self.hdr_imgfile = hdr_imgfile
        self.mlp_input = mlp_input
        self.use_coords = use_coords
        self.encode_mode = encode_mode

        self.len_input = get_len_input_type(mlp_input, use_coords)

        logging.info('Loading images information...')

        self.gain_map, self.network_input, self.sdr, self.hdr, self.nParam = encode(self.sdr_imgfile, self.hdr_imgfile, encode_mode=self.encode_mode, mlp_input=self.mlp_input)

        coords_dim0 = torch.linspace(-1, 1, self.network_input.shape[0])
        coords_dim1 = torch.linspace(-1, 1, self.network_input.shape[1])
        coords = torch.stack(torch.meshgrid((coords_dim0, coords_dim1)), axis=-1)

        self.network_input_shape = self.network_input.shape

        if len(self.network_input_shape) < 3:
            netVectDim = 1
        else:
            netVectDim = self.network_input_shape[2]

        self.gain_map_shape = self.gain_map.shape

        if len(self.gain_map_shape) < 3:
            gainVectDim = 1
        else:
            gainVectDim = self.gain_map_shape[2]

        self.coords = coords.view(-1,2)
        if self.mlp_input not in ['just_coords']:
            self.network_input = self.network_input.view(-1,netVectDim)
        
        self.gain_map = self.gain_map.view(-1,gainVectDim)

        # for evaluation during training
        self.decode = partial(decode, encode_mode, nParam = self.nParam) 
        self.iNorm = partial(sNorm, inverse=True, nParam = self.nParam)

    def __len__(self):
        return self.network_input.shape[0]

    def __getitem__(self, i):
        network_input_values = self.network_input[:]
        gain_map = self.gain_map[:]
        
        if self.use_coords:
            input = torch.cat((network_input_values, self.coords), 1)
        else:
            input = network_input_values

        return {'input': input, 'gain_map': gain_map}

### decoding block ###

class decDataset(Dataset):
    def __init__(self, sdr_imgfile, encode_mode, mlp_input = "rgb", use_coords = True):
        self.sdr_imgfile = sdr_imgfile
        self.mlp_input = mlp_input
        self.use_coords = use_coords
        self.encode_mode = encode_mode

        self.len_input = get_len_input_type(mlp_input, use_coords)

        logging.info('Decoding ... '+sdr_imgfile)

        self.network_input, self.sdr = decodeInput(self.sdr_imgfile)

        coords_dim0 = torch.linspace(-1, 1, self.network_input.shape[0])
        coords_dim1 = torch.linspace(-1, 1, self.network_input.shape[1])
        coords = torch.stack(torch.meshgrid((coords_dim0, coords_dim1)), axis=-1)

        self.network_input_shape = self.network_input.shape

        if len(self.network_input_shape) < 3:
            netVectDim = 1
        else:
            netVectDim = self.network_input_shape[2]

        self.coords = coords.view(-1,2)
        self.network_input = self.network_input.view(-1,netVectDim)

    def __len__(self):
        return self.network_input.shape[0]

    def __getitem__(self, i):

        network_input_values = self.network_input[:]
        
        #concatenate coord_values to sdr_image_values
        if self.use_coords:
            input = torch.cat((network_input_values, self.coords), 1)
        else:
            input = network_input_values

        return {'input': input}