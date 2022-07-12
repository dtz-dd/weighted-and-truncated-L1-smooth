# weighted and truncated L1 smooth

#### 介绍

Weighted and Truncated L1 Image Smoothing based on Unsupervised Learning

#### 实现细节

1. Network structure: The network adopts 15-layer Unet structure. Each of the first 14 layers is composed of 3 convolutional layers followed by Batchnorm and RELU activation function. The last layer is directly convolutional layer followed by Sigmoid function.  The first 7 layers down-sample the image to 1/2 layer by layer, and the last 7 layers up-sample the image to restore the original image size.  Add skip layer corresponding to the first 7 and the last 7 to add the previous output to the following input. 

2. loss function：

$$
(O - I)^2 + \lambda(W_x*|O_x - I'_x| + W_y*|O_y - I'_y|)
$$
， $W_x$ is $exp(-∂^2Ix / 2σ^2)$， $I'_x$ is the gradient of image $I$ truncated by σ。


#### Instructions

1. python test.py 
2. python train.py

## Visual Results
<div align=center><img src="https://github.com/zal0302/PNLS/blob/master/figs/nks.png" width="1000"  /></div>
Comparison of smoothed images and PSNR(dB)/SSIM/FSIM results by different methods on the image S15T1 from our NKS dataset.
<div align=center><img src="https://github.com/zal0302/PNLS/blob/master/figs/div.png" width="1000"  /></div>
Comparison of smoothed images by different methods on the image 0117 from DIV2K dataset.
<div align=center><img src="https://github.com/zal0302/PNLS/blob/master/figs/RTV.jpg" width="1000"  /></div>
Comparison of smoothed images by different methods on the image 11_11 from RTV dataset.
<div align=center><img src="https://github.com/zal0302/PNLS/blob/master/figs/500.png" width="1000"  /></div>
Comparison of smoothed images by different methods on the image 0334 from 500images dataset.



