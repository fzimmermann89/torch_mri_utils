import warnings
import torch
import numpy as np

def _filter_separable(x, kernels, axis):
    """Apply the separable filter kernels to the tensor x along the axes axis.

    Does zero-padding to keep the output the same size as the input.

    Parameters
    ----------
    x : Tensor to filter
    kernels : List of 1D kernels to apply to the tensor x.
    axis : Axis or axes to filter over. Must have the same length as kernels.
    """
    if len(axis) != len(kernels):
        raise ValueError("Must provide matching length kernels and axis arguments. ")
    if len(axis) > x.ndim:
        raise ValueError("Too many axes provided")
    for kernel, ax in zip(kernels, axis):
        x = x.moveaxis(ax, -1)
        x = torch.nn.functional.conv1d(
            x.flatten(end_dim=-2)[:, None, :], kernel[None, None, :], padding="same"
        ).reshape(x.shape)
        x = x.moveaxis(-1, ax)
    return x


def gaussian_filter(x, sigmas, axis=None, truncate=3):
    """Apply a nd-Gaussian filter.


    Parameters
    ----------
    x : Tensor to filter
    sigmas : Standard deviation for Gaussian kernel. If iterable, must have length equal to the number of axes.
    axis : Axis or axes to filter over. If None, filters over all axes.
    truncate : Truncate the filter at this many standard deviations.
    """
    sigmas = torch.tensor(sigmas) if np.iterable(sigmas) else torch.tensor([sigmas])
    if torch.any(sigmas < 0):
        raise ValueError("`sigmas` cannot contain negative values")
    if axis is None:
        axis = tuple(range(x.ndim))
    if len(sigmas) != len(axis):
        raise ValueError("Must provide matching length sigmas and axis arguments. ")

    kernels = [
        torch.exp(-0.5 * (torch.arange(-truncate * sigma, truncate * sigma + 1) / sigma) ** 2) for sigma in sigmas
    ]
    kernels = [k / k.sum() for k in kernels]
    x_filtered = _filter_separable(x, kernels, axis)
    return x_filtered


def uniform_filter(x, width, axis=None):
    """Apply a nd-uniform filter.


    Parameters
    ----------
    x : Tensor to filter
    width : Width of uniform kernel. If iterable, must have length equal to the number of axes.
    axis : Axis or axes to filter over. If None, filters over all axes.
    """

    width = torch.tensor(width) if np.iterable(width) else torch.tensor([width])
    if torch.any(width % 2 != 1):
        warnings.warn("width should be odd")
    if torch.any(width < 0):
        raise ValueError("width cannot contain negative values")
    if axis is None:
        axis = tuple(range(x.ndim))
    if len(width) != len(axis):
        raise ValueError("Must provide matching length width and axis arguments. ")

    kernels = [torch.ones(width) / width for width in width]
    x_filtered = _filter_separable(x, kernels, axis)
    return x_filtered


def sliding_window(x, window_shape, axis=None, strides=1):
    """Sliding window into the tensor x.

    Returns a view into the tensor x that represents a sliding window.


    Parameters
    ----------
    x : Tensor to slide over
    window_shape : Size of window over each axis that takes part in the sliding window.
    axis :  Axis or axes to slide over. If None, slides over all axes.
    strides : Stride of the sliding window. **Experimental**.
    """
    if axis is None:
        axis = tuple(range(x.ndim))
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,) * len(axis)
    strides = tuple(strides) if np.iterable(strides) else (strides,) * len(axis)
    window_shape_arr = torch.tensor(window_shape)
    strides_arr = torch.tensor(strides)
    x_shape_arr = torch.tensor(x.shape)
    if torch.any(strides_arr != 1):
        warnings.warn("strides other than 1 are not fully supported")
    if torch.any(window_shape_arr < 0):
        raise ValueError("window_shape cannot contain negative values")
    if torch.any(strides_arr < 0):
        raise ValueError("strides cannot contain negative values")
    if len(window_shape) != len(axis):
        raise ValueError("Must provide matching length window_shape and axis arguments. ")
    if len(strides) != len(axis):
        raise ValueError("Must provide matching length strides and axis arguments.")

    out_strides = torch.tensor([x.stride(i) for i in range(x.ndim)] + [x.stride(ax) for ax in axis])
    out_strides[axis,] = out_strides[axis,] * strides_arr
    x_shape_arr[axis,] = (x_shape_arr[axis,] + strides_arr - 1) // strides_arr
    if torch.any(x_shape_arr < 0):
        raise ValueError("strides or windows too large")
    out_shape = tuple(x_shape_arr) + window_shape
    view = x.as_strided(size=out_shape, stride=tuple(out_strides))
    return view


def coil_map_study_2d_Inati(data: torch.Tensor, ks: int, power: int, padding_mode="circular"):
    """Coil sensitivity maps using the method described in Inati et al. 2004.

    Parameters
    ----------
    data: Images of shape (coil, E1, E0)
    ks: kernel size
    power: number of iterations
    padding_mode: padding mode for the sliding window
    """
    if ks % 2 != 1:
        raise ValueError("ks must be odd")
    if power < 1:
        raise ValueError("power must be at least 1")

    halfKs = ks // 2
    # adding another dimension before padding is a workaround for https://github.com/pytorch/pytorch/issues/95320
    padded = torch.nn.functional.pad(data[None], (halfKs, halfKs, halfKs, halfKs), mode=padding_mode)[0]
    D = sliding_window(padded, (ks, ks), axis=(-1, -2)).flatten(-2)  # coil E1, E0, ks*ks
    DH_D = torch.einsum("i...j,k...j->...ik", D, D.conj())  # E1,E0,coil,coil
    singular_vector = torch.sum(D, dim=-1)  # coil, E1, E0
    singular_vector /= singular_vector.abs().square().sum(0, keepdim=True).sqrt()
    for _ in range(power):
        singular_vector = torch.einsum("...ij,j...->i...", DH_D, singular_vector)  # coil, E1, E0
        singular_vector /= singular_vector.abs().square().sum(0, keepdim=True).sqrt()
    singular_value = torch.einsum("i...j,i...->...j", D, singular_vector)  # E1, E0, ks*ks
    phase = singular_value.sum(-1)
    phase /= phase.abs()  # E1, E0
    csm = singular_vector.conj() * phase[None, ...]
    return csm
