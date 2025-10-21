# Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP, ICCV 2025
Trevor D. Canham <sup>1</sup>, SaiKiran Tedla <sup>1</sup>, Michael J. Murdoch <sup>2</sup>, Michael S. Brown <sup>1</sup>

<sup>1</sup>York University  <sup>2</sup>Rochester Institute of Technology

[Paper](https://arxiv.org/abs/2503.11883), [Video](https://www.youtube.com/watch?v=u7OTgVeZur4), [Dataset](https://www.dropbox.com/scl/fo/uskvi9evls91uax00f4cx/AOm20-zZSq_08JHuuq0ewBg?rlkey=cdgufhmh3cvm4t1ifh5vwx5or&st=vl5p7hm7&dl=0)

![](https://raw.githubusercontent.com/trevorcanham/Gain-MLP/refs/heads/main/teaserICCV.png)


Setup:

install [tinycudann](https://github.com/NVlabs/tiny-cuda-nn.git) or add argument ``` -m tfm_mlp ``` to encoding and decoding calls (worse performance).  
```
pip install -r requirements.txt
```

To run encode demo with augmented dataset from Cyriac et al., navigate to root directory, download [dataset](https://www.dropbox.com/scl/fo/uskvi9evls91uax00f4cx/AOm20-zZSq_08JHuuq0ewBg?rlkey=cdgufhmh3cvm4t1ifh5vwx5or&st=vl5p7hm7&dl=0), and run the following command:
```
# Gain Map (Multiplicative Residual)
python PyTorch/runner.py -mi rgb -em gain -id hdr_sdr_graded_pairs -sd sdr/*.tif -hd hdr/*.tif -pi -md models/metaGammaLab2
# Gamma Map (Exponential Residual)
python PyTorch/runner.py -mi rgb -em gamma -id hdr_sdr_graded_pairs -sd sdr/*.tif -hd hdr/*.tif -pi -md models/metaGammaLab2
```

To run decode demo:
```
python PyTorch/runner.py -id hdr_sdr_graded_pairs/sdr -dd models/demo/ -od output
```

Happy encoding and decoding!
