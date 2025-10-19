# Utilities for encoding and decoding MLP based HDR residuals (gain or gamma maps)
# If used, cite: Trevor D. Canham, SaiKiran Tedla, Michael J. Murdoch and Michael S. Brown.
# Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP. ICCV 2025

import json
from matplotlib import pyplot as plt
import numpy as np
from utilities.io import get_len_input_type, get_len_output_type
import torch
import os
import glob
from shutil import rmtree
import time
from pathlib import Path
import logging
from datetime import datetime
import argparse
import sys
sys.path.append("..")
import wandb
import copy
import tinycudann as tcnn
from arch.transform_mlp import TransformMLP  
import pdb


def get_img_paths(imgs_dir, sdr_dir, hdr_dir=None):

    sdr_path = os.path.join(imgs_dir, sdr_dir)
    sdr_imgfiles = glob.glob(sdr_path)
    sdr_imgfiles.sort()

    hdr_path = os.path.join(imgs_dir, hdr_dir)
    hdr_imgfiles = glob.glob(hdr_path)
    hdr_imgfiles.sort()

    return sdr_imgfiles[:], hdr_imgfiles[:]

def calc_model_size(model):
    # From: https://discuss.pytorch.org/t/finding-model-size/130275
    # ~~ in @ptrblck we trust ~~
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / (1024 ** 2) # for MB

    return size_all_mb

def get_freer_gpu(num_gpus):
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])

        if len(memory_available) == 0:
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
            memory_used = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
            return np.argsort(memory_used)[0:num_gpus]
        else:
            return np.argsort(-1*memory_available)[0:num_gpus]
    
#from deep_architect with MIT License
def get_total_num_gpus():
    try:
        import subprocess
        n = len(subprocess.check_output(['nvidia-smi','-L']).decode('utf-8').strip().split('\n'))
    except OSError:
        n = 0
    return n 

def get_device(args):
    gpus_chosen = get_freer_gpu(args.num_gpus).tolist()
    logging.info(f'Using devices {gpus_chosen}')

    all_gpus = [i for i in range(get_total_num_gpus())]
    all_gpus.sort()

    logging.info(f'ALL GPUS {all_gpus}')

    os.environ["CUDA_VISIBLE_DEVICES"]=",".join(map(str, all_gpus))
    os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] ="50"
    device = f"cuda:{gpus_chosen[0]}"
    return gpus_chosen, device

def get_args():
    # DEFAULT PARAMETERS HIGHLY RECOMMENDED - parameters left open for experimentation
    parser = argparse.ArgumentParser(description='Encode/Decode MLP based HDR residual.', add_help=False)
    # If you have multiple GPUs, you can use this parameter to engage them
    parser.add_argument('-g', '--num-gpus', metavar='E', type=int, default=1,
                        help='Number of gpus', dest='num_gpus')
    # Control the number of dataloader workers
    parser.add_argument('-w', '--num-workers', dest='workers', type=int, default=0,
                        help='Number of dataloader workers')
    # Model object   
    parser.add_argument('-m', '--model', dest='model', type=str, default="tinycudann",
                        help='Model used')
    # Model config file (model parameters)    
    parser.add_argument('-c', '--config', dest='config_file', type=str, default="PyTorch/configs/tinymlp_config1.json",
                        help='Config file for tinycudann')       
    # Reconstruction quality can be improved with more optimization iterations (epochs)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,
                        help='Number of epochs', dest='epochs')
    # The amount of change in each optimization step
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-2,
                        help='Learning rate', dest='lr')    
    # Controls how many pixels are loaded onto the GPU in each optimization iteration
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2**16,
                        help='Batch size', dest='batch_size')
    # Controls MLP inputs: 'rgb' (default 3 channel gain map) , 'y' (alternative 1 channel gain map - not recommended but included for reference)
    parser.add_argument('-mi', '--mlp-input', dest='input_type', type=str, default="rgb",
                        help='Input type')
    # Includes pixel coordninates in MLP input if True
    parser.add_argument ('-coords', '--use-coords', dest='use_coords', default=True,
                         help='Includes pixel coordninates in MLP input if True.', action='store_true')  
    # Specify type of residual map - 'gain' (multiplicative residual), 'gamma' (exponential residual)                    
    parser.add_argument('-em', '--encode_mode', dest='encode_mode', type=str, default="gamma",
                        help='Encode mode')
    # directory containing SDR and HDR subdirectories, if encoding or contaning SDR images, if decoding
    parser.add_argument('-id', '--input_dir', dest='iDir', default='demo',
                        help='Training image directory')
    # SDR directory name                    
    parser.add_argument('-sd', '--sdr_dir', dest='sdr_dir', default='sdr',
                        help='SDR image directory')
    # HDR directory name
    parser.add_argument('-hd', '--hdr_dir', dest='hdr_dir', default='hdr',
                        help='HDR image directory')
    # Initialization weights directory
    parser.add_argument('-md', '--meta_dir', dest='meta_dir', default='metaGammaLab2',
                        help='Meta models directory')
    # Save images during encoding
    parser.add_argument('-pi', '--plot_images', dest='plot_images', default=False,
                        help='Output images', action='store_true')
    # Directory of MLP weights of encoded residuals
    parser.add_argument('-dd', '--decode_dir', dest='decode_dir', type=str, required=False,
                        default=None, help='Decode model weights directory')
    # Decoding output directory
    parser.add_argument('-od', '--out_dir', dest='out_dir', type=str, required=False,
                        default='output', help='Output directory')   

    return parser.parse_args()

def get_model(args, dataset_params):
    
    with open(args.config_file, "r") as jsonfile:
        config = json.load(jsonfile)
    if args.model == "tinycudann":
        net = tcnn.NetworkWithInputEncoding(n_input_dims=get_len_input_type(dataset_params["mlp_input"], dataset_params["use_coords"]), n_output_dims=get_len_output_type(dataset_params['mlp_input']), encoding_config=config["encoding"], network_config=config["network"])
    elif args.model == "tfm_mlp":
        net = TransformMLP(n_input_dims=get_len_input_type(dataset_params["mlp_input"], dataset_params["use_coords"]),n_output_dims=get_len_output_type(dataset_params['mlp_input']),config=config)

    return net

# def average_metrics(all_metrics):

#     # Initialize an empty dictionary to store the averages
#     averages = {}
    
#     # Iterate through each dictionary in the list
#     for metrics_dict in all_metrics:

#         # Iterate through each key-value pair in the dictionary
#         for key, value in metrics_dict.items():

#             # If the key is not in the averages dictionary, initialize it with the current value
#             if key not in averages:
#                 averages[key] = value
#             else:
#                 # If the key is already present, add the current value to the existing value
#                 averages[key] += value
    
#     # Divide each sum by the number of dictionaries to get the average
#     num_dicts = len(all_metrics)
#     for key in averages:
#         averages[key] /= num_dicts
    
#     return averages

def average_weights(model_paths):

    # Initialize variables to store total weights and count
    total_weights = None
    count = 0

    # Iterate over each model path
    for path in model_paths:
        # Load model
        model = torch.load(path)
        
        # Get state dict
        state_dict = model['net_state_dict']
        
        # If it's the first model, initialize total_weights with its state_dict
        if total_weights is None:
            total_weights = state_dict
        else:
            # Add the weights of the current model to total_weights
            for key in total_weights:
                total_weights[key] += state_dict[key]
        
        # Increment count
        count += 1

    # Compute the average weights
    average_weights = {key: value / count for key, value in total_weights.items()}
    
    return average_weights

def save_checkpoint(checkpoint_name, dir_checkpoint, net, optimizer, dataset_params, nParam):
    
    # Create checkpoint directory if it does not already exist
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint, exist_ok=True)
        logging.info('Created checkpoint directory')

    # Save necessary parameters for decoding - weights, optimizer state, normalization parameters, pipeline settigns
    state_dict = {
    'net_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'nParam': nParam,
    'dataset_params': dataset_params
    }

    torch.save(state_dict, dir_checkpoint + f'{checkpoint_name}')
    
