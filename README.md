# A collection of some of MRI/Pytorch-related code fragments.

## AdjointGridSample 
The adjoint of `torch.nn.functional.grid_sample` as a differentiable pytorch function

## ChannelSeparateConv
Apply the same convolutional filers to all input channels

## SeparateConv3d
Flexible Building block for xy+t convolutions. Describe separable convolution by a format string. Allows, for example
  - yx, C -> Depthwise Separable Convolution, i.e. grouped 2D convolution in x and y with groups=channels_in, followed by kernel_size=1 convolution
  - zyC, zxC -> 2D convolution in zy followed by a 2D convolution in zx.
  - yx,z,C -> 2D convolution in yx, followed by a 1D convoution in z, both grouped. Followed by a kernel_size=1 convolution to out_channels
  - c, zyx, C - > 1x1x1 channel-mixing convoltion of the input channels. 3d depthwise convolution. 1x1x1 convolution to the number of out_channnels.

Useful to implement networks of PINQI: An End-to-End Physics-Informed Approach to Learned Quantitative MRI Reconstruction ([arxiv](https://arxiv.org/abs/2306.11023))

## Inati Coil Sensitivity estimation
Implementation of [A Solution to the Phase Problem in Adaptive Coil Combination] (https://archive.ismrm.org/2013/2672.html) as used in  ([PINQI](https://arxiv.org/abs/2306.11023))

## Sliding Window View
Zero-Copy sliding window  
