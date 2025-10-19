# Gain/Gamma map pipeline for encoding and decoding MLP based HDR residuals (gain or gamma maps)
# If used, cite: Trevor D. Canham, SaiKiran Tedla, Michael J. Murdoch and Michael S. Brown.
# Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP. ICCV 2025

import numpy as np
import torch
import cv2
import colour
from timeit import default_timer as timer
import pdb

# Init MLP input dimensions based on MLP pipeline settings
def get_len_input_type(mlp_input, useCoords):
    if mlp_input == 'y':
        nIn = 1
    else:
        nIn = 3
    if useCoords:
        nIn = nIn + 2
    return nIn

# Init MLP output dimensions based on MLP pipeline settings
def get_len_output_type(mlp_input):
    if mlp_input == 'y':
        nOut = 1
    else:
        nOut = 3
    return nOut

# Gain/Gamma map encoding pipeline
def encode(pathSDR, pathHDR, encode_mode, mlp_input):
    
    # experiment parameters
    illSDR = np.array([0.31270, 0.32900])
    eotfSDR = 2.4
    peakSDR = 100
    illHDR = np.array([0.31270, 0.32900])
    eotfHDR = "ST 2084"
    peakHDR = 1000
    depth = 16
    sDepth = 16
    offset = 1/64 # gain map recommended
    eps = 1e-6

    # read images
    imHDR = np.divide(cv2.cvtColor(cv2.imread(pathHDR,cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB).astype(np.double),(2**depth)-1)
    imSDR = np.divide(cv2.cvtColor(cv2.imread(pathSDR,cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB).astype(np.double),(2**sDepth)-1)
    imSDRq = np.divide(np.floor(np.multiply(imSDR,255)+0.5),255).astype(np.double)
    h,w,c = imHDR.shape

    # convert SDR,HDR to XYZ
    imHDRlin = np.divide(colour.eotf(imHDR, function=eotfHDR, L_p = peakHDR),peakHDR)  # function outputs 0-100
    imHDRxyz = colour.RGB_to_XYZ(imHDRlin, colour.models.RGB_COLOURSPACE_BT2020, illHDR, "Bradford")
    imSDRqLin = np.power(imSDRq,eotfSDR)
    imSDRqXYZ = colour.RGB_to_XYZ(imSDRqLin, colour.models.RGB_COLOURSPACE_BT709, illSDR, "Bradford")

    # convert SDR to Rec. 2020 PQ
    imSDRqInHDRlin = colour.XYZ_to_RGB(imSDRqXYZ,colour.models.RGB_COLOURSPACE_BT2020,illSDR,"Bradford")
    imSDRqInHDR = colour.eotf_inverse(imSDRqInHDRlin, function=eotfHDR, L_p= peakSDR)

    # define inputs for gain map calculation
    if mlp_input == 'rgb':
        hdrEnc = imHDR
        sdrEnc = imSDRqInHDR        
    else:
        hdrEnc = imHDRxyz[:,:,1]
        sdrEnc = np.divide(imSDRqXYZ[:,:,1],peakHDR/peakSDR)       
    
    # calculate gain/gamma map
    if encode_mode == 'gain':
        np.clip(hdrEnc,eps,1-eps)
        np.clip(sdrEnc,eps,1-eps)
        gMap = np.divide(hdrEnc+offset,sdrEnc+offset) 
    elif encode_mode == 'gamma':
        np.clip(hdrEnc,eps,1-eps) # no 0s or 1s allowed ..
        np.clip(sdrEnc,eps,1-eps)
        gMap = np.divide(np.log(hdrEnc+offset),np.log(sdrEnc+offset))

    # if encode_mode == 'direct':
    #     gMapNorm,nParamGm = sNorm(imHDR)
    # else:

    # Normalize gain/gamma map
    gMapNorm,nParamGm = sNorm(gMap)
    nParamGm.append(offset)

    # define MLP inputs
    if mlp_input == 'rgb':
        netIn,_ = sNorm(imSDRqInHDR)
    elif mlp_input == 'y':
        netIn,_ = sNorm(imSDRqXYZ[:,:,1])
    # elif mlp_input == 'just_coords':
    #     netIn,_ = sNorm(imSDRqInHDR)
    #     netIn[:] = 0

    #convert to torch
    gMapNorm = torch.from_numpy(gMapNorm).half()
    netIn = torch.from_numpy(netIn).half()
    #imHDR = torch.from_numpy(imHDR).half()

    return gMapNorm, netIn, imSDRqInHDR, imHDR, nParamGm

# Gain/gamma map normalization pipeline
def sNorm(im,nParam=[],inverse=False):    

    # normalize/inverse normalize to -1:1 range for MLP input
    if not inverse:
        imNorm_ = im-np.mean(im)
        imNorm = np.divide(imNorm_,(np.max(imNorm_)-np.min(imNorm_)))*2
        nParam = [np.max(imNorm_),np.mean(im),np.min(imNorm_)]      
    else:
        imNorm = np.multiply(np.divide(im,2),(nParam[0]-nParam[2]))+nParam[1]

    return imNorm,nParam

def decodeInput(pathSDR):

    # color space parameters
    illSDR = np.array([0.31270, 0.32900])
    eotfSDR = 2.4
    peakSDR = 100
    illHDR = np.array([0.31270, 0.32900])
    eotfHDR = "ST 2084"
    sDepth = 16

    # load SDR image and convert to CIEXYZ
    imSDR = np.divide(cv2.cvtColor(cv2.imread(pathSDR,cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB).astype(np.double),(2**sDepth)-1)
    imSDRq = np.divide(np.floor(np.multiply(imSDR,255)+0.5),255).astype(np.double)
    imSDRqLin = np.power(imSDRq,eotfSDR)
    imSDRqXYZ = colour.RGB_to_XYZ(imSDRqLin, colour.models.RGB_COLOURSPACE_BT709, illSDR, "Bradford")

    # convert SDR to Rec. 2020 PQ 1,000 nits
    imSDRqInHDRlin = colour.XYZ_to_RGB(imSDRqXYZ,colour.models.RGB_COLOURSPACE_BT2020,illHDR,"Bradford")
    imSDRqInHDR = colour.eotf_inverse(imSDRqInHDRlin, function=eotfHDR, L_p=peakSDR)
    netIn,_ = sNorm(imSDRqInHDR)

    # convert to 16-bit float torch tensor
    netIn = torch.from_numpy(netIn).half()

    return netIn, imSDRqInHDR

# decode MLP reconstructed gain/gamma map to reconstruct HDR image
def decode(im, gMap, encode_mode, nParam):

    offset = nParam[3]

    # gain map (multiplicative residual) decoding
    if encode_mode == 'gain':
        if len(gMap.shape) < 3:
            imOut = np.zeros(im.shape) 
            imOut[:,:,0] = np.multiply((im[:,:,0]+offset),gMap)-offset
            imOut[:,:,1] = np.multiply((im[:,:,1]+offset),gMap)-offset
            imOut[:,:,2] = np.multiply((im[:,:,2]+offset),gMap)-offset
        else:
            imOut = np.multiply((im+offset),gMap)-offset

    # gamma map (exponential residual) decoding
    elif encode_mode == 'gamma':
        if len(gMap.shape) < 3:
            imOut = np.zeros(im.shape)       
            imOut[:,:,0] = np.power((im[:,:,0]+offset),gMap)-offset    
            imOut[:,:,1] = np.power((im[:,:,1]+offset),gMap)-offset   
            imOut[:,:,2] = np.power((im[:,:,2]+offset),gMap)-offset   
        else:
            imOut = np.power((im+offset),gMap)-offset   
 
    return imOut
