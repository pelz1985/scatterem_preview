from typing import Tuple

import warp as wp
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


@wp.kernel
def _gaussian_sample_kernel(
    position: wp.array(dtype=wp.float32, ndim=1),
    amplitude: wp.array(dtype=wp.float32, ndim=1),
    inv_var: wp.array(dtype=wp.float32, ndim=1),
    selection: wp.array(dtype=wp.int64, ndim=2),  # X, 5
    grid: wp.array(dtype=wp.vec3, ndim=4),  # [N, D, H, W)] of vec3,
    output: wp.array(dtype=wp.float32, ndim=1),  # [X,]
) -> None:
    """As proposed here: https://www.nature.com/articles/s41565-023-01595-w"""
    x = wp.tid()  # global thread index
    n = wp.int32(selection[x, 0])
    d = wp.int32(selection[x, 1])
    h = wp.int32(selection[x, 2])
    w = wp.int32(selection[x, 3])
    j = wp.int32(selection[x, 4])

    pos_gauss = wp.vec3(position[0], position[1], position[2])
    pos_grid = grid[n, d, h, w]
    pr = pos_gauss - pos_grid
    rel_distance_sqr = pr.x * pr.x + pr.y * pr.y + pr.z * pr.z

    output[x] += amplitude[j] * wp.exp(-inv_var[j] * rel_distance_sqr)


class _GaussianSample(Function):
    """The class is a Pytorch wrapper for gaussian_sample function

    :param th: _description_
    :type th: _type_
    :return: _description_
    :rtype: _type_
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        position: Tensor,  # [3,]
        amplitude: Tensor,  # [2,]
        inv_var: Tensor,  # [2,]
        selection: Tensor,  # [X, 5]
        grid: Tensor,  # [N, D, H, W, 3)] of vec3
    ) -> Tensor:  # [X]
        device = wp.device_from_torch(grid.device)
        X, _ = selection.shape
        ctx.dim = X

        for x in [position, amplitude, inv_var]:
            if x.requires_grad:
                x.retain_grad()

        ctx.position = wp.from_torch(position)
        ctx.amplitude = wp.from_torch(amplitude)
        ctx.inv_var = wp.from_torch(inv_var)
        ctx.selection = wp.from_torch(selection, requires_grad=False)
        ctx.grid = wp.from_torch(grid, dtype=wp.vec3, requires_grad=False)

        ctx.out = wp.zeros(X, dtype=wp.float32, device=device, requires_grad=True)
        wp.launch(
            kernel=_gaussian_sample_kernel,
            dim=ctx.dim,
            inputs=[
                ctx.position,
                ctx.amplitude,
                ctx.inv_var,
                ctx.selection,
                ctx.grid,
            ],
            outputs=[ctx.out],
            device=device,
        )
        out_torch = wp.to_torch(ctx.out)

        return out_torch

    @staticmethod
    def backward(ctx: FunctionCtx, out_adj: Tensor) -> Tuple[Tensor | None, ...]:
        device = wp.device_from_torch(out_adj.device)
        ctx.out.grad = wp.from_torch(out_adj.contiguous())
        # ctx.position.grad = wp.zeros_like(ctx.position)
        # ctx.amplitude.grad = wp.zeros_like(ctx.amplitude)
        # ctx.inv_var.grad = wp.zeros_like(ctx.inv_var)
        wp.launch(
            kernel=_gaussian_sample_kernel,
            dim=ctx.dim,
            inputs=[
                ctx.position,
                ctx.amplitude,
                ctx.inv_var,
                ctx.selection,
                ctx.grid,
            ],
            outputs=[ctx.out],
            adjoint=True,
            adj_inputs=[
                ctx.position.grad,
                ctx.amplitude.grad,
                ctx.inv_var.grad,
                None,
                None,
            ],
            adj_outputs=[ctx.out.grad],
            device=device,
        )
        position_grad = wp.to_torch(ctx.position.grad)
        amplitude_grad = wp.to_torch(ctx.amplitude.grad)
        inv_var_grad = wp.to_torch(ctx.inv_var.grad)
        selection_grad = None
        grid_grad = None

        return (
            position_grad,
            amplitude_grad,
            inv_var_grad,
            selection_grad,
            grid_grad,
        )


def gaussian_sample_legacy(
    position: Tensor,  # [3,]
    amplitude: Tensor,  # [2,]
    inv_var: Tensor,  # [2,]
    selection: Tensor,  # [X, 5]
    grid: Tensor,  # [N, D, H, W, 3)] of vec3
) -> Tensor:
    return _GaussianSample.apply(position, amplitude, inv_var, selection, grid)
