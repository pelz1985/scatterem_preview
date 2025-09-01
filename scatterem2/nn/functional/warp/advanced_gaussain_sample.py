from typing import Tuple

import torch
import warp as wp
from torch import Tensor
from torch.autograd.function import FunctionCtx


@wp.func
def _check_bbox_and_amp(
    difference: wp.vec3, radius: wp.float32, amplitude: wp.float32
) -> bool:
    res = radius > 1e-7 and amplitude > -1e-7
    for i in range(3):
        res = res and not (difference[i] > radius or difference[i] < -radius)
    return res


@wp.kernel
def _gaussian_sample_kernel_complex(
    positions: wp.array(dtype=wp.vec3, ndim=1),  # [G]
    amplitudes: wp.array(dtype=wp.float32, ndim=2),  # [G, 2]
    sigmas: wp.array(dtype=wp.float32, ndim=2),  # [G, 2]
    grid: wp.array(dtype=wp.vec3, ndim=2),  # [N, D)]
    gaussian_ids: wp.array(dtype=wp.int32, ndim=1),  # [G']
    num_gaussain_ids: int,  # [G']
    output: wp.array(dtype=wp.float32, ndim=3),  # [N, D, 2]
) -> None:
    """The potential is represented as a sum of isotropic gaussians
    as proposed here: https://www.nature.com/articles/s41565-023-01595-w

    This kernel is used for computation of both, real and imaginary part
    of the potential
    """
    n, d, j = wp.tid()  # global thread index

    point_pos = grid[n, d]
    for g_id_iter in range(num_gaussain_ids):
        g = gaussian_ids[g_id_iter]
        gauss_poss = positions[g]
        difference = point_pos - gauss_poss
        sigma = sigmas[g, j]
        radius = sigma * 3.0
        amplitude = amplitudes[g, j]
        if _check_bbox_and_amp(
            difference=difference, radius=radius, amplitude=amplitude
        ):
            diff_sqr = (
                difference.x * difference.x
                + difference.y * difference.y
                + difference.z * difference.z
            )
            value = amplitude * wp.exp(-diff_sqr / sigma / sigma * 0.5)
            output[n, d, j] += value


@wp.kernel
def _gaussian_sample_kernel_real(
    positions: wp.array(dtype=wp.vec3, ndim=1),  # [G]
    amplitudes: wp.array(dtype=wp.float32, ndim=1),  # [G]
    sigmas: wp.array(dtype=wp.float32, ndim=1),  # [G]
    grid: wp.array(dtype=wp.vec3, ndim=2),  # [N, D)]
    gaussian_ids: wp.array(dtype=wp.int32, ndim=1),  # [G']
    num_gaussain_ids: int,  # [G']
    output: wp.array(dtype=wp.float32, ndim=2),  # [N, D]
) -> None:
    """The potential is represented as a sum of isotropic gaussians
    as proposed here: https://www.nature.com/articles/s41565-023-01595-w

    This kernel is used for computation of the real part of the potential
    """
    n, d = wp.tid()  # global thread index

    point_pos = grid[n, d]
    for g_id_iter in range(num_gaussain_ids):
        g = gaussian_ids[g_id_iter]
        gauss_poss = positions[g]
        difference = point_pos - gauss_poss
        sigma = sigmas[g]
        radius = sigma * 3.0
        amplitude = amplitudes[g]
        if _check_bbox_and_amp(
            difference=difference, radius=radius, amplitude=amplitude
        ):
            diff_sqr = (
                difference.x * difference.x
                + difference.y * difference.y
                + difference.z * difference.z
            )
            value = amplitude * wp.exp(-diff_sqr / sigma / sigma * 0.5)
            output[n, d] += value


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
        amplitudes: Tensor,  # [G, _ or 2]
        sigmas: Tensor,  # [G, _ or 2]
        grid: Tensor,  # [N, D_slice, 3],
        gaussian_ids: Tensor,  # [G']
    ) -> Tensor:  # [N, D_slice, _ or 2]
        device = wp.device_from_torch(grid.device)
        N, D, _ = grid.shape
        G, _ = positions.shape

        is_complex = len(amplitudes.shape) > 1

        if is_complex:
            ctx.dim = (N, D, 2)
            ctx.out = wp.zeros(
                (N, D, 2), dtype=wp.float32, device=device, requires_grad=True
            )
            ctx.kernel_fun = _gaussian_sample_kernel_complex
        else:
            ctx.dim = (N, D)
            ctx.out = wp.zeros(
                (N, D), dtype=wp.float32, device=device, requires_grad=True
            )
            ctx.kernel_fun = _gaussian_sample_kernel_real

        ctx.positions = wp.from_torch(positions, dtype=wp.vec3, requires_grad=True)
        ctx.amplitudes = wp.from_torch(amplitudes, requires_grad=True)
        ctx.sigmas = wp.from_torch(sigmas, requires_grad=True)
        ctx.grid = wp.from_torch(grid, dtype=wp.vec3, requires_grad=False)
        ctx.gaussian_ids = wp.from_torch(
            gaussian_ids, dtype=wp.int32, requires_grad=False
        )
        ctx.num_gaussain_ids = gaussian_ids.shape[0]
        wp.launch(
            kernel=ctx.kernel_fun,
            dim=ctx.dim,
            inputs=[
                ctx.positions,
                ctx.amplitudes,
                ctx.sigmas,
                ctx.grid,
                ctx.gaussian_ids,
                ctx.num_gaussain_ids,
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
        position_grad = wp.zeros_like(ctx.positions)
        amplitude_grad = wp.zeros_like(ctx.amplitudes)
        sigma_grad = wp.zeros_like(ctx.sigmas)
        wp.launch(
            kernel=ctx.kernel_fun,
            dim=ctx.dim,
            inputs=[
                ctx.positions,
                ctx.amplitudes,
                ctx.sigmas,
                ctx.grid,
                ctx.gaussian_ids,
                ctx.num_gaussain_ids,
            ],
            outputs=[ctx.out],
            adjoint=True,
            adj_inputs=[
                position_grad,
                amplitude_grad,
                sigma_grad,
                None,
                None,
                None,
            ],
            adj_outputs=[ctx.out.grad],
            device=device,
        )
        position_grad = wp.to_torch(position_grad)
        amplitude_grad = wp.to_torch(amplitude_grad)
        sigmas_grad = wp.to_torch(sigma_grad)

        return (
            position_grad,
            amplitude_grad,
            sigmas_grad,
            None,
            None,
        )


def advanced_gaussian_sample(
    positions: Tensor,  # [G, 3]
    amplitudes: Tensor,  # [G, 2]
    sigmas: Tensor,  # [G, 2]
    grid: Tensor,  # [N, D_slice, 3],
    gaussian_ids: Tensor,  # [G']
) -> Tensor:
    return _GaussianSample.apply(positions, amplitudes, sigmas, grid, gaussian_ids)
