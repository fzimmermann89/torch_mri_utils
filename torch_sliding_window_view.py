import warnings
from typing import Sequence
import torch
import numpy as np
# fzimmermann89, felix.zimmermann@ptb.de, 2024
def sliding_window(x:torch.Tensor, window_shape:int|Sequence[int], axis:None|int|Sequence[int]|=None):
    """Sliding window into the tensor x.

    Returns a view into the tensor x that represents a sliding window.

    Parameters
    ----------
    x : Tensor to slide over
    window_shape : Size of window over each axis that takes part in the sliding window.
    axis :  Axis or axes to slide over. If None, slides over all axes.
    """
    strides=1 # This could be a parameter, but the logic mmight be wrong for !=1 (Needs testing!)
    if axis is None:
        axis = tuple(range(x.ndim))
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)*len(axis)
    strides = tuple(strides) if np.iterable(strides) else (strides,)*len(axis)
    window_shape_arr = torch.tensor(window_shape)
    strides_arr = torch.tensor(strides)
    x_shape_arr = torch.tensor(x.shape)
    if torch.any(window_shape_arr < 0):
        raise ValueError("window_shape cannot contain negative values")
    if torch.any(strides_arr < 0):
        raise ValueError("strides cannot contain negative values")
    if len(window_shape) != len(axis):
        raise ValueError("Must provide matching length window_shape and axis arguments. ")
    if len(strides) != len(axis):
        raise ValueError("Must provide matching length strides and axis arguments.")
    
    # keep existing strides if stride == 1, else increase by stride factor.
    # reduce existing shapes, as we do not pad.
    # add new dimensions with window-shape
    
    out_strides = torch.tensor([x.stride(i) for i in range(x.ndim)] + [x.stride(ax) for ax in axis])
    out_strides[axis,] = out_strides[axis,] * strides_arr
    x_shape_arr[axis,] = (x_shape_arr[axis,] + strides_arr - 1) // strides_arr
    
    if torch.any(x_shape_arr < 0):
        raise ValueError("strides or windows too large")
    out_shape = tuple(x_shape_arr) + window_shape
    return x.as_strided(size=out_shape, stride=tuple(out_strides))
