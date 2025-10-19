# Synthetic initialization image generator
# Based on loose description of methodology from:
# Scott Daly, Timo Kunkel, Guan-Ming Su, and Anustup Choudhury
# Spatiochromatic and temporal natural image statistics modelling: applications from display analysis to neural networks.
# Electronic Imaging 2025
# Used to initialize encoding networks in:
# Trevor D. Canham, SaiKiran Tedla, Michael J. Murdoch and Michael S. Brown.
# Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP. ICCV 2025

import colour as c
import numpy as np
import pdb
import matplotlib.pyplot as plt
import imageio
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def synthInit(res, alpha, seed=None):

    pqLut = c.read_LUT("lin2pq.cube")
    tmLut = c.read_LUT("dvPQ2020to709.cube")

    if seed is None:
        ch1 = (np.random.choice(2**16,res[0]*res[1],replace=True).astype(np.double)/2**16).reshape((res[0],res[1]))
        ch2 = (np.random.choice(2**16,res[0]*res[1],replace=True).astype(np.double)/2**16).reshape((res[0],res[1]))
        ch3 = (np.random.choice(2**16,res[0]*res[1],replace=True).astype(np.double)/2**16).reshape((res[0],res[1]))

        inArray = np.stack([ch1,ch2,ch3],axis=-1)

    else:
        print('pre-seeded!')
        inArray = seed

    outArray = np.zeros((inArray.shape))

    
    for ch in range(3):
        inArrayC = inArray[:,:,ch]
        Yfft = np.fft.fftshift(np.fft.fft2(inArrayC))
        _x, _y = np.mgrid[0:res[0],0:res[1]]
        f = np.hypot(_x - res[0] // 2, _y - res[1] // 2)
        # avoid division by zero
        f[f == 0] = 1
        # apply pink noise filter
        Yfft_pink = Yfft / f**alpha

        # inverse fourier transform
        cPink = np.fft.ifft2(np.fft.ifftshift(Yfft_pink)).real
        outArray[:,:,ch] = cPink

    # another way of doing it
    outArrayNorm = (outArray - np.min(outArray)) / (np.max(outArray)-np.min(outArray))
    outArrayNorm[:,:,0] = steepSig(outArrayNorm[:,:,0],4) * 105
    outArrayNorm[:,:,1] = (outArrayNorm[:,:,1] - np.mean(outArrayNorm[:,:,1])) * 150
    outArrayNorm[:,:,2] = (outArrayNorm[:,:,2] - np.mean(outArrayNorm[:,:,2])) * 150
    pinkXYZ = c.Lab_to_XYZ(outArrayNorm)
    pink2020 = np.clip(c.XYZ_to_RGB(pinkXYZ,c.models.RGB_COLOURSPACE_BT2020),0,1)

    # plotting
    # pink2020vect = tmLut.table.reshape((-1,3))
    # ind = 1
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(projection='3d')
    # ax1.scatter(pink2020vect[::ind,0],pink2020vect[::ind,1],pink2020vect[::ind,2], c = pink2020vect[::ind,:])
    # plt.savefig('pqLut.png', dpi=500, transparent=True)
    # plt.close()

    exrNorm = c.XYZ_to_RGB(pinkXYZ,c.models.RGB_COLOURSPACE_BT709)
    exr = ((exrNorm / np.max(exrNorm)) * 10).astype(np.float32)
    hdr = pqLut.apply(pink2020)
    sdr = tmLut.apply(hdr)

    return hdr, sdr, exr, inArray

def steepSig(x,steep):
    xNorm = x * 2 - 1
    y = (1/(1 + np.exp(-steep*xNorm)))
    return y

if __name__ == '__main__':

    res = [2160,3840]
    alphas = np.linspace(0.1,5,49)#np.linspace(0,10,2400)#[0.5]
    iters = 10 #2400

    #hdr, sdr, exr = synthInit(res,2)
    #c.write_image(sdr,'synthInit.jpg',bit_depth='uint8')

    #im = c.read_image('/home/tcanham/2HDRVD/data/hueStill/exr/17_pike.exr',bit_depth="float32")
    #pdb.set_trace()
    #c.write_image(im*100,'/home/tcanham/2HDRVD/data/hueStill/tif/17_pike.tif',bit_depth="float32")
    
    # for same seed different alpha gag
    # alphaStart = 0
    # hdr, sdr, exr, seed = synthInit(res,alphaStart)
    # c.write_image(hdr,'synthInitData/'+'siVideoHDR_'+str(alphaStart).zfill(3)+'.tif',bit_depth='uint16')
    # c.write_image(sdr,'synthInitData/'+'siVideoSDR_'+str(alphaStart).zfill(3)+'.tif',bit_depth='uint8')

    # for alpha in alphas:

    #     hdr, sdr, exr, seed = synthInit(res,alpha,seed)
    #     alphaInd = int(alpha * 10)
    #     c.write_image(hdr,'synthInitData/'+'siVideoHDR_'+str(alphaInd).zfill(3)+'.tif',bit_depth='uint16')
    #     c.write_image(sdr,'synthInitData/'+'siVideoSDR_'+str(alphaInd).zfill(3)+'.tif',bit_depth='uint8')

    alpha = 2
    for i in range(iters):
        hdr, sdr, exr, _ = synthInit(res,alpha)
        c.write_image(hdr,'synthInitData/'+'siVideoHDR_'+str(i).zfill(3)+'.tif',bit_depth='uint16')
        c.write_image(sdr,'synthInitData/'+'siVideoSDR_'+str(i).zfill(3)+'.tif',bit_depth='uint8')



    # for alpha in alphas:
    #     for i in range(iters):
    #         hdr, sdr, exr, seed = synthInit(res,alpha,False)
    #         #exrCrop = exr[2000:3000,:,:]
    #         #pdb.set_trace()
    #         #start = 0
    #         #end = 1000
    #         #for j in range(10):
    #         #    sdrCrop = sdr[2000:3000,start:end,:]
    #         #    end+=1000
    #         #    start+=1000
    #         #    c.write_image(sdrCrop,'synthInitData/promo3/alpha'+str(alpha)+'_iter0'+str(j)+'.png',bit_depth='uint8')
    #         print(alpha)
    #         #c.write_image(sdr,'synthInitData/deepDream/alpha'+str(alpha)+'_iter'+str(i)+'.tif',bit_depth='uint16')
    #         c.write_image(sdr,'synthInitData/'+'siVideo._a'+str(alpha)+'.tif',bit_depth='uint16')
    #         #imageio.imwrite('synthInitData/promo/alpha'+str(alpha)+'_iter'+str(i)+'.tif', exr)
    #         #c.write_image(exrCrop,'synthInitData/promo2/alpha'+str(alpha)+'_iter'+str(i)+'.tif',bit_depth='float32')

