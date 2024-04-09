""" Convolutions using the same kernel for all input channels """

from collections.abc import Sequence
from torch import Tensor
import torch.nn.functional as F


class ChannelSeparateConv2d(torch.nn.Conv2d):
    """The same as Conv2d, but for all channels, the kernels are the same"""

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | str = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:

        super().__init__(1, 1, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device=device, dtype=dtype)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Tensor | None):
        channels = input.shape[1]
        if self.padding_mode != "zeros":
            padding = 0
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        else:
            padding = self.padding
        weight = weight.expand(channels, *weight.shape[1:])
        if bias is not None:
            bias = bias.expand(channels)
        return F.conv2d(input, weight, bias, self.stride, padding, self.dilation, groups=channels)


class ChannelSeparateConv3d(torch.nn.Conv3d):
    """The same as Conv3d, but for all channels, the kernels are the same"""

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | str = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:

        super().__init__(1, 1, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device=device, dtype=dtype)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Tensor | None):
        channels = input.shape[1]
        if self.padding_mode != "zeros":
            padding = 0
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        else:
            padding = self.padding
        weight = weight.expand(channels, *weight.shape[1:])
        if bias is not None:
            bias = bias.expand(channels)
        return F.conv3d(input, weight, bias, self.stride, padding, self.dilation, groups=channels)
