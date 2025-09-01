from typing import Tuple

import torch
import warp as wp
from torch import Tensor
from torch.autograd.function import FunctionCtx


@wp.func
def _check_bbox(difference: wp.vec3, radius: wp.float32) -> bool:
    if radius < 1e-7:
        return False

    for i in range(3):
        if difference[i] > radius or difference[i] < -radius:
            return False
    return True


@wp.kernel
def _gaussian_sample_kernel(
    positions: wp.array(dtype=wp.vec3, ndim=1),  # [G]
    amplitudes: wp.array(dtype=wp.float32, ndim=2),  # [G, 2]
    sigmas: wp.array(dtype=wp.float32, ndim=2),  # [G, 2]
    grid: wp.array(dtype=wp.vec3, ndim=2),  # [N, D_slice)]
    output: wp.array(dtype=wp.float32, ndim=3),  # [N, D_slice, 2]
) -> None:
    """As proposed here: https://www.nature.com/articles/s41565-023-01595-w"""
    n, d, g, j = wp.tid()  # global thread index

    point_pos = grid[n, d]
    gauss_poss = positions[g]
    difference = point_pos - gauss_poss
    sigma = sigmas[g, j]
    radius = sigma * 3.0
    if _check_bbox(difference=difference, radius=radius):
        amplitude = amplitudes[g, j]
        if amplitude > -1e-7:
            diff_sqr = (
                difference.x * difference.x
                + difference.y * difference.y
                + difference.z * difference.z
            )
            value = amplitude * wp.exp(-diff_sqr / sigma / sigma * 0.5)
            # if (n == 0 and d == 0 and j == 0):
            #     print(inv_var_value)
            #     # print(output[n, d, j])
            wp.atomic_add(output, n, d, j, value)


class _GaussianSample(torch.autograd.Function):
    """The class is a Pytorch wrapper for gaussian_sample function

    :param th: _description_
    :type th: _type_
    :return: _description_
    :rtype: _type_
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        positions: Tensor,  # [G, 3]
        amplitudes: Tensor,  # [G, 2]
        sigmas: Tensor,  # [G, 2]
        grid: Tensor,  # [N, D_slice, 3]
    ) -> Tensor:  # [N, D_slice, 2]
        device = wp.device_from_torch(grid.device)
        N, D, _ = grid.shape
        G, _ = positions.shape
        ctx.dim = (N, D, G, 2)

        ctx.positions = wp.from_torch(positions, dtype=wp.vec3, requires_grad=True)
        ctx.amplitudes = wp.from_torch(amplitudes, requires_grad=True)
        ctx.sigmas = wp.from_torch(sigmas, requires_grad=True)
        ctx.grid = wp.from_torch(grid, dtype=wp.vec3, requires_grad=False)

        ctx.out = wp.zeros(
            (N, D, 2), dtype=wp.float32, device=device, requires_grad=True
        )
        wp.launch(
            kernel=_gaussian_sample_kernel,
            dim=ctx.dim,
            inputs=[
                ctx.positions,
                ctx.amplitudes,
                ctx.sigmas,
                ctx.grid,
            ],
            outputs=[ctx.out],
            device=device,
        )
        out_torch = wp.to_torch(ctx.out)

        return out_torch

    @staticmethod
    def backward(ctx: FunctionCtx, out_adj: Tensor) -> Tuple[Tensor | None]:
        device = wp.device_from_torch(out_adj.device)
        ctx.out.grad = wp.from_torch(out_adj.contiguous())
        # ctx.position.grad = wp.zeros_like(ctx.position)
        # ctx.amplitude.grad = wp.zeros_like(ctx.amplitude)
        # ctx.inv_var.grad = wp.zeros_like(ctx.inv_var)
        wp.launch(
            kernel=_gaussian_sample_kernel,
            dim=ctx.dim,
            inputs=[
                ctx.positions,
                ctx.amplitudes,
                ctx.sigmas,
                ctx.grid,
            ],
            outputs=[ctx.out],
            adjoint=True,
            adj_inputs=[
                ctx.positions.grad,
                ctx.amplitudes.grad,
                ctx.sigmas.grad,
                None,
            ],
            adj_outputs=[ctx.out.grad],
            device=device,
        )
        position_grad = wp.to_torch(ctx.positions.grad)
        amplitude_grad = wp.to_torch(ctx.amplitudes.grad)
        sigmas_grad = wp.to_torch(ctx.sigmas.grad)

        return (
            position_grad,
            amplitude_grad,
            sigmas_grad,
            None,
        )


def gaussian_sample(
    positions: Tensor,  # [G, 3]
    amplitudes: Tensor,  # [G, 2]
    sigmas: Tensor,  # [G, 2]
    grid: Tensor,  # [N, D_slice, 3]
) -> Tensor:
    return _GaussianSample.apply(positions, amplitudes, sigmas, grid)
