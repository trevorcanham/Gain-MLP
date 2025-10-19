# Encoding and decoding functions for MLP based HDR residuals (gain or gamma maps)
# If used, cite: Trevor D. Canham, SaiKiran Tedla, Michael J. Murdoch and Michael S. Brown.
# Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP. ICCV 2025

import copy
import logging
import os
import sys
import time
import glob

from utilities.input_tensor import InputTensor
sys.path.append(os.getcwd())
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from utilities.utils import average_weights, get_device, get_img_paths, get_model, get_args, save_checkpoint
import pickle
from utilities.dataset import encDataset, decDataset
from torch.utils.data import DataLoader
import wandb
from skimage.metrics import structural_similarity
import cv2
from datetime import datetime
from utilities.io import decode, sNorm
import pdb
from pathlib import Path

### encoding block ###

def encode_net(net_og, dataset_params,
              device,
              lr,
              epochs=10,
              batch_size=32,
              dir_img='../demo/',
              sdr_dir = 'sdr',
              hdr_dir = 'hdr',
              num_workers = 0,
              save_cp=True,
              run_name = "",
              dir_checkpoint = None,
              odir = None,
              meta_dir = None,
              plot_images=False):

    if dir_checkpoint is None:
        dir_checkpoint = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../models/{run_name}/')
        
    odir = dir_checkpoint

    sdr_imgs, hdr_imgs = get_img_paths(dir_img, sdr_dir, hdr_dir)
    n_train = len(sdr_imgs)

    if (len(sdr_imgs) != len(hdr_imgs)):
        print("Error: SDR and HDR image counts do not match")
        return

    if meta_dir is not None:
        
        #load all pth files in meta_dir
        model_paths = [os.path.join(meta_dir, f) for f in os.listdir(meta_dir) if f.endswith('.pth')]
        meta_initialization_weights = average_weights(model_paths)

        #set net_og weights to the meta_initialization_weights
        net_og.load_state_dict(meta_initialization_weights)

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs} epochs
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Checkpoints:     {save_cp}
        Device:          {device}
    ''')
    
    count = 0
    times = []
    for img_num in range(len(sdr_imgs)):

        print(sdr_imgs[img_num])

        train_dataset = encDataset(sdr_imgfile=sdr_imgs[img_num], hdr_imgfile=hdr_imgs[img_num], encode_mode=dataset_params["encode_mode"], mlp_input=dataset_params["mlp_input"], imgs_dir=dir_img, num_pixels=batch_size, use_coords = dataset_params["use_coords"])
        train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
        
        #use net_og as a template to create a new net
        net = copy.deepcopy(net_og)
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        net.train()

        #get first "item" for data_loader (for standard case all pixels are in one batch)
        for batch in train_loader:
            input = batch['input'].to(device=device, dtype=torch.float32)
            gain_map = batch['gain_map'].to(device=device, dtype=torch.float32)
            break
        
        epoch = 0
        inputTensor = InputTensor(input, gain_map, device)
        with tqdm(total=epochs, desc=f'Img {img_num}/{len(sdr_imgs)}', unit='img', position=0, leave=True) as pbar:
            startTime = time.time()
            while epoch < epochs:

                batch = torch.rand([batch_size], device=device, dtype=torch.float32)
                inp, groundtruth = inputTensor(batch)
                #convert net_out to float32
                net_out = net(inp).float()
                loss = torch.nn.functional.mse_loss(net_out, groundtruth)

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                global_step += 1
                epoch+=1
                pbar.update(1)
            
            eTime = time.time() - startTime
        
        print('Elapsed time = {0} [s]'.format(eTime))
        evaluate_one_image(net, train_dataset, dataset_params, train_loader, device, plot_images, out_imgs_dir=odir)
        times.append(eTime)
        count+=1
        #checkpoint saving
        if save_cp:
            save_checkpoint(Path(sdr_imgs[img_num]).stem+'.pth', dir_checkpoint, net, optimizer, dataset_params, train_dataset.nParam)
            logging.info(f'Checkpoint {sdr_imgs[img_num]} saved!')

def evaluate_one_image(net, dataset, dataset_params, loader, device, plot_images=False, out_imgs_dir=None):
    for batch in loader:
        input = batch['input'].to(device=device, dtype=torch.float32)
        gain_map = batch['gain_map'].to(device=device, dtype=torch.float32)
        break

    with torch.no_grad():
        net_out_norm = net(input)

        # reshape net_out and gain_map to 2D+color channels
        net_out_norm = net_out_norm.reshape(dataset.gain_map_shape)
        gain_map_norm = gain_map.reshape(dataset.gain_map_shape)

        net_out_norm = net_out_norm.cpu().detach().numpy()
        gain_map_norm = gain_map_norm.cpu().detach().numpy()

        # un-normalize net_out and gain_map
        net_out,_ = dataset.iNorm(net_out_norm)
        gain_map,_ = dataset.iNorm(gain_map_norm)

        # decode HDR 
        hdrDnet = decode(dataset.sdr, net_out, dataset_params["encode_mode"], dataset.nParam)

       # call ground truth HDR for evaluation 
        hdrD = dataset.hdr

        # compute mse loss between net_out and gain_map
        mse = np.mean((hdrD - hdrDnet)**2)
        # compute psnr
        max_v = 1
        psnr = 10 * np.log10(max_v**2 / mse)
        print("psnr .. ", psnr)

        # compute ssim on green channel
        ssim = structural_similarity(hdrD, hdrDnet, data_range=1, channel_axis=2)

        # combine heatmap with log_dict
        metrics = {'MSE': mse.item(), "PSNR": psnr.item(), "SSIM": ssim}
        
        # save reconstructed images for review
        if plot_images:

            os.makedirs(os.path.join(out_imgs_dir,'recon'),exist_ok = True)
            hdrDnetOut = cv2.cvtColor(np.multiply(np.clip(hdrDnet,0,1),65535).astype(np.uint16),cv2.COLOR_BGR2RGB)   
            cv2.imwrite(os.path.join(out_imgs_dir,'recon',os.path.basename(dataset.sdr_imgfile)),hdrDnetOut)
        
        return metrics

    logging.info('End of training')

### decoding block ###

def decode_net(decode_dir_path, dataset_params, dir_img, num_workers, device, out_imgs_dir=None):
    
    all_metrics = []
    sdr_path = os.path.join(dir_img, '*.tif')
    sdr_imgs = glob.glob(sdr_path)
    sdr_imgs.sort()

    for img_num in range(len(sdr_imgs)):

        # Load the model
        checkpoint = torch.load(os.path.join(decode_dir_path,Path(sdr_imgs[img_num]).stem+'.pth'))

        #load checkpoint into net
        net.load_state_dict(checkpoint['net_state_dict'])
        net.to(device)
        net.eval()
        
        # load encoding metadata
        dataset_params = checkpoint['dataset_params']
        nParam = checkpoint['nParam']

        #create a dataset object for each image 
        decode_dataset = decDataset(sdr_imgfile=sdr_imgs[img_num], encode_mode=dataset_params["encode_mode"], mlp_input=dataset_params["mlp_input"], use_coords = dataset_params["use_coords"])
        decode_loader = DataLoader(decode_dataset, batch_size=None, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)

        decode_one_image(decode_dataset, dataset_params, nParam, decode_loader, device, out_imgs_dir)

def decode_one_image(dataset, dataset_params, nParam, loader, device, out_imgs_dir=None):

    # Call network inputs from data loader
    for batch in loader:
        input = batch['input'].to(device=device, dtype=torch.float32)
        break

    with torch.no_grad():
        net_out_norm = net(input)

        # convert net_out from torch to 2D np array on CPU
        net_out_norm = net_out_norm.reshape(dataset.network_input_shape).cpu().detach().numpy()

        # inverse normalization of decoded gain map
        net_out,_ = sNorm(net_out_norm, nParam, inverse=True)

        # reconstruct hdr using decode if not direct 
        if dataset_params["encode_mode"] == 'direct':
            hdrDnet = net_out
        else:
            hdrDnet = decode(dataset.sdr, net_out, dataset_params["encode_mode"], nParam)

        # save HDR
        os.makedirs(os.path.join(out_imgs_dir),exist_ok = True)
        hdrDnetOut = cv2.cvtColor(np.multiply(np.clip(hdrDnet,0,1),65535).astype(np.uint16),cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(out_imgs_dir,os.path.basename(dataset.sdr_imgfile)),hdrDnetOut)
            
if __name__ == '__main__':
    
    #set random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # init logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # call args from arg parse
    args = get_args()
    gpus_chosen, device = get_device(args)

    # Establish MLP pipeline settings
    dataset_params = {"mlp_input": args.input_type, "encode_mode": args.encode_mode, "use_coords": args.use_coords}
    
    # Init network based on params
    net = get_model(args, dataset_params)

    logging.info(f'Using model {args.model}')

    # Set run name for checkpoint directory
    now = datetime.now()
    run_name = now.strftime("%Y%m%d_%H%M%S")

    if args.decode_dir:

        # decode images
        decode_net(decode_dir_path=args.decode_dir, 
                dataset_params=dataset_params,
                dir_img=args.iDir,
                num_workers=args.workers,
                device=device,
                out_imgs_dir=args.out_dir)

    else:

        # encode images
        encode_net(net_og=net,
                    dataset_params=dataset_params,
                    device=device,
                    lr=args.lr,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    dir_img=args.iDir,
                    sdr_dir=args.sdr_dir,
                    hdr_dir=args.hdr_dir,
                    num_workers=args.workers,
                    run_name=run_name,
                    meta_dir=args.meta_dir,
                    plot_images=args.plot_images)
                    
    

