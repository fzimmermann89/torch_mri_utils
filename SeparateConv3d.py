import torch
from torch import nn
from typing import Literal, Callable


class SeparateConv3d(torch.nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        pattern="xyzc",
        kernel_size: int = 3,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        inner_activation: Callable[[], nn.Module] | None | nn.Module = None,
    ):
        """


        A 3D Convolution separated into different blocks according to a pattern


        Pattern Examples
          xy, c -> Depthwise Separable Convolution, i.e. grouped 2D convolution in x and y with groups=channels_in, followed by kernel_size=1 convolution
          xzc, yzc -> 2D convolution in xz followed by a 2D convolution in yz.
          xy,z,c -> 2D convolution in xy, followed by a 1D convoution in z, both grouped. Followed by a kernel_size=1 convolution to out_channels


        All convolitions are non-dilated, non-transposed, stride-1.
        Bias will be enabled on the last convolution if inner_activation is None,
        otherwise it will be enabled on all convolutions.


        Parameters
        ----------
        in_channels
          number of input channels of the first convolution.
        out_channels
          number of output channels of the first convolution along 'c', and number of output channels of the complete block.
        pattern
          pattern describing the different convolutions. a string with comma seperated blocks of axes that have non-singleton kernel size in each
          convolution. axes are named x y z. if a block does NOT contain 'c' (channels), it is a 'depth-wise' convolution, i.e. a grouped convolution with
          in_channels == out_channels == groups.
        kernel_size
          kernel size of non-singleton spatial axes.
        padding_mode
          each convolution is padded to 'same' size using this method, see torch.nn.Conv2d
        inner_activation
          if not None, the activation to perform between the blocks. Can either be an instance of a Module or a callable that creates a module.
          (Note, this could also be nn.Sequential containing a normalization)
        """
        super().__init__()
        current_channels = in_channels
        blocks = pattern.split(",")
        for i, block in enumerate(blocks):
            if len(set(block)) != len(block):
                raise ValueError(f'The {i}. block "{block}" contains repeated axes. This is not allowed. Consider splitting into two blocks with a comma.')
            if wrong := set(block) - set("xyzc "):
                raise ValueError(f' The  {i}. block "{block}" should only contain x,y,z or c, not {wrong}')
            if not "c" in block:
                current_out_channels = current_channels
                groups = current_out_channels
            else:
                groups = 1
                current_out_channels = out_channels
            kernel = [kernel_size if ax in block else 1 for ax in "xyz"]
            bias = inner_activation is not None or i == len(blocks) - 1
            conv = nn.Conv3d(current_channels, current_out_channels, kernel_size=kernel, padding="same", padding_mode=padding_mode, stride=1, bias=bias, groups=groups)
            self.append(conv)
            if isinstance(inner_activation, nn.Module):
                self.append(inner_activation)
            elif callable(inner_activation):
                self.append(inner_activation())
            current_channels = current_out_channels
        if current_channels != out_channels:
            raise ValueError("If none of the convolutions consider the channels, i.e. in no block contains a 'c', all convolutions are depth-wise and the in_channels must match out_channels")

        self.__pattern = ", ".join(b.strip() for b in blocks)

    def __repr__(self):
        children = "\n  ".join(super().__repr__().split("\n")[1:-1])
        class_name = self.__class__.__name__
        return f"{class_name} ({self.__pattern})\n implemented as steps \n  {children}"
