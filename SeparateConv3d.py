import torch
from torch import nn
from typing import Literal, Callable

import torch
from torch import nn
from typing import Literal, Callable
from einops.layers.torch import Rearrange


class SeparateConv3D(torch.nn.Sequential):
    """A Stack of convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        pattern="Cxyz",
        weights: str = "",
        kernel_size: int = 3,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        inner_activation: Callable[[], nn.Module] | None | nn.Module = None,
    ):
        """
        A 3D Convolution separated into different blocks according to a pattern

        The pattern consists of comma separated blocks. Each block can contain one or more of 'z', 'y', 'x'. This requests a convolution with
        the kernel_size along this axis. The axes of the input are named ... z y x. (compare to pytorch, depth=z, heigh=y, width=x).
        Axes not mentioned in the block are kernel-1 directions.
        If the block does NOT contain the either c or C, the convolution is depth-wise, i.e. separate for different channels.
        If the block contains 'c', the number of output channels of the block will be channels_in
        If the block contains 'C', the number of output channels of the block will be channels_in.
        A leading number in the block can be used to overwrite the kernel_size for this block.
        Pattern Examples
          yx, C -> Depthwise Separable Convolution, i.e. grouped 2D convolution in x and y with groups=channels_in, followed by kernel_size=1 convolution
          zyC, zxC -> 2D convolution in zy followed by a 2D convolution in zx.
          yx,z,C -> 2D convolution in yx, followed by a 1D convoution in z, both grouped. Followed by a kernel_size=1 convolution to out_channels
          c, zyx, C - > 1x1x1 channel-mixing convoltion of the input channels. 3d depthwise convolution. 1x1x1 convolution to the number of out_channnels.

        Weights can be a comma separated string of weight names. Repeated names here mean that the convoluion blocks share weights.
        Example
          pattern = "C, zy, zx"; weights = "channel, 2d1, 2d1" -> The two 2D convolutions share the weights.
        Note, that here the order of the axes names in the blocks matter as to which directions are considered equivalent in the weight sharing
        convolutions.

        All convolitions are non-dilated, non-transposed, stride-1.

        Bias will be enabled on the last convolution if inner_activation is None,
        otherwise it will be enabled on all convolutions.

        Between the convolutions, activations can be inserted using inner_activation.


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
        weights
          comma separated weight names, same length as blocks. repeated names are used to signal sharing of weights between different blocks.
          if the string is empty (default), each block uses differnent weights.
        kernel_size
          kernel size of non-singleton spatial axes.
        padding_mode
          each convolution is padded to 'same' size using this method, see torch.nn.Conv2d
        inner_activation
          if not None, the activation to perform between the blocks. Can either be an instance of a Module or a callable that creates a module.
          (Note, this could also be nn.Sequential containing a normalization)
        """
        super().__init__()

        def parse_block(block, current_channels):
            kernelsize = kernel_size
            channels = current_channels
            axes = []
            groups = current_channels
            for c in block:
                if c.isdigit():
                    kernelsize = int(c)
                elif c == "c":
                    channels = in_channels
                    groups = 1
                elif c == "C":
                    channels = out_channels
                    groups = 1
                elif c in "zyx":
                    axes.append(c)
                else:
                    raise ValueError(f"unknown axes name {c}")
            if len(axes) != len(set(axes)):
                raise ValueError(f"repeated axes in one block are invalid, got {axes}")
            kernelsize = [kernelsize] * len(axes)
            for c in "zyx":
                if c not in axes:
                    kernelsize.append(1)
                    axes.append(c)
            axes = " ".join(axes)
            return kernelsize, channels, axes, groups

        blocks = pattern.split(",")
        if weights == "":
            weights_per_block = [f"w_{i}" for i in range(len(blocks))]
        else:
            weights_per_block = weights.split(",")
        if len(weights_per_block) != len(blocks):
            raise ValueError(
                "Weights can either be an empty string or a comma separated list of weight names. "
                "The number of weight names has to match the number of blocks in the pattern"
            )
        convs_dict = dict()
        current_channels_in = in_channels
        current_axes = "z y x"

        for i, (block, weight) in enumerate(zip(blocks, weights_per_block)):
            last_block = i == len(blocks) - 1
            current_kernel_sizes, current_channels_out, axes, groups = parse_block(block, current_channels_in)

            if axes != current_axes:
                self.append(Rearrange(f"... {current_axes} -> ... {axes}"))
                current_axes = axes
            use_bias = inner_activation is not None or last_block
            conv = torch.nn.Conv3d(
                current_channels_in,
                current_channels_out,
                current_kernel_sizes,
                padding="same",
                padding_mode=padding_mode,
                bias=use_bias,
                groups=groups,
            )
            if weight in convs_dict:
                other = convs_dict[weight]
                if not other.weight.shape == conv.weight.shape:
                    raise ValueError(f"weight {weight} has two different shapes, {other.weight.shape} and {conv.weight.shape}.")
                if conv.bias is not None and other.bias is not None:
                    conv.bias = other.bias
                conv.weight = other.weight
            else:
                convs_dict[weight] = conv
            self.append(conv)
            current_channels_in = current_channels_out

            if not last_block:
                if isinstance(inner_activation, nn.Module):
                    self.append(inner_activation)
                elif callable(inner_activation):
                    self.append(inner_activation())
        if current_axes != "z y x":
            self.append(Rearrange(f"... {current_axes} -> ... z y x"))
        if current_channels_out != out_channels:
            raise ValueError(
                f"pattern does not result in correct number of out channels. After the block, the number of channels would be {current_channels_out}."
                "Consider adding a final C pattern to change the number of channels"
            )
        self.__pattern = ", ".join([f"{b} (weight {w})" for b, w in zip(blocks, weights_per_block)])

    def __repr__(self):
        children = "\n  ".join(super().__repr__().split("\n")[1:-1])
        class_name = self.__class__.__name__
        return f"{class_name} \n  Pattern: {self.__pattern}\n  implemented as steps \n  {children}"



class _SeparateConv3dOld(torch.nn.Sequential):
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
