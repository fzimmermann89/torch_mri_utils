class AdjointGridSample(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        y: torch.Tensor,
        grid: torch.Tensor,
        xshape: Sequence[int],
        interpolation_mode: Literal['bilinear', 'nearest', 'bicubic'] = 'bilinear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
        align_corners: bool = True,
    ) -> torch.Tensor:

        match interpolation_mode:
            case 'bilinear':
                mode_enum = 0
            case 'nearest':
                mode_enum = 1
            case 'bicubic':
                mode_enum = 2
            case _:
                raise ValueError(f'Interpolation mode {interpolation_mode} not supported')

        match padding_mode:
            case 'zeros':
                padding_mode_enum = 0
            case 'border':
                padding_mode_enum = 1
            case 'reflection':
                padding_mode_enum = 2
            case _:
                raise ValueError(f'Padding mode {padding_mode} not supported')

        match dim := grid.shape[-1]:
            case 3:
                backward_2d_or_3d = torch.ops.aten.grid_sampler_3d_backward
            case 2:
                backward_2d_or_3d = torch.ops.aten.grid_sampler_2d_backward
            case _:
                raise ValueError(f'only 2d and 3d supported, not {dim}')

        if y.shape[0] != grid.shape[0]:
            raise ValueError(f'y and grid must have same batch size, got {y.shape=}, {grid.shape=}')
        if xshape[1] != y.shape[1]:
            raise ValueError(f'xshape and y must have same number of channels, got {xshape[1]} and {y.shape[1]}.')
        if len(xshape) - 2 != dim:
            raise ValueError(f'len(xshape) and dim must either both bei 2 or 3, got {len(xshape)} and {dim}')

        # These are required in the backward
        ctx.xshape = xshape  # type: ignore[attr-defined]
        ctx.interpolation_mode = mode_enum  # type: ignore[attr-defined]
        ctx.padding_mode = padding_mode_enum  # type: ignore[attr-defined]
        ctx.align_corners = align_corners  # type: ignore[attr-defined]
        ctx.backward_2d_or_3d = backward_2d_or_3d  # type: ignore[attr-defined]
        if grid.requires_grad:
            # only if we need to calculate the gradient for grid we need y
            ctx.save_for_backward(grid, y)
        else:
            ctx.save_for_backward(grid)

        shape_dummy = torch.empty(1, dtype=y.dtype, device=y.device).broadcast_to(xshape)
        x = backward_2d_or_3d(
            y,
            shape_dummy,
            grid,
            interpolation_mode=mode_enum,
            padding_mode=padding_mode_enum,
            align_corners=align_corners,
            output_mask=[True, False],
        )[0]
        return x

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, *grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        """Backward of the Adjoint Gridsample Operator."""
        need_y_grad, need_grid_grad, *_ = ctx.needs_input_grad  # type: ignore[attr-defined]
        grid = ctx.saved_tensors[0]  # type: ignore[attr-defined]

        if need_y_grad:
            grad_y = torch.grid_sampler(
                grad_output[0],
                grid,
                ctx.interpolation_mode,  # type: ignore[attr-defined]
                ctx.padding_mode,  # type: ignore[attr-defined]
                ctx.align_corners,  # type: ignore[attr-defined]
            )
        else:
            grad_y = None

        if need_grid_grad:
            y = ctx.saved_tensors[1]  # type: ignore[attr-defined]
            grad_grid = ctx.backward_2d_or_3d(  # type: ignore[attr-defined]
                y,
                grad_output[0],
                grid,
                interpolation_mode=ctx.interpolation_mode,  # type: ignore[attr-defined]
                padding_mode=ctx.padding_mode,  # type: ignore[attr-defined]
                align_corners=ctx.align_corners,  # type: ignore[attr-defined]
                output_mask=[False, True],
            )[1]
        else:
            grad_grid = None

        return grad_y, grad_grid, None, None, None, None
      

def adjoint_grid_sample(
    y: torch.Tensor,
    grid: torch.Tensor,
    xshape: Sequence[int],
    interpolation_mode: Literal["bilinear", "nearest", "bicubic"] = "bilinear",
    padding_mode: Literal["zeros", "border", "reflection"] = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Adjoint of the linear operator x->gridsample(x,grid).

    Parameters
    ----------
    y
        tensor in the range of gridsample(x,grid). Should not include batch or channel dimension.
    grid
        grid in the shape (*y.shape, 2/3)
    xshape
        shape of the domain of gridsample(x,grid), i.e. the shape of x
    interpolation_mode
        the kind of interpolation used
    padding_mode
        how to pad the input
    align_corners
         if True, the corner pixels of the input and output tensors are aligned,
         and thus preserving the values at those pixels
    """
    return AdjointGridSample.apply(y, grid, xshape, interpolation_mode, padding_mode, align_corners)
