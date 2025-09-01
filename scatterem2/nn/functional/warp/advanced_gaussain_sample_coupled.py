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
def _gaussian_sample_kernel_coupled(
    positions: wp.array(dtype=wp.vec3, ndim=1),  # [G]
    amplitudes: wp.array(dtype=wp.float32, ndim=1),  # [G]
    sigmas: wp.array(dtype=wp.float32, ndim=1),  # [G]
    coupling_constants: wp.array(dtype=wp.float32, ndim=1),  # [G]
    grid: wp.array(dtype=wp.vec3, ndim=2),  # [N, D)]
    gaussian_ids: wp.array(dtype=wp.int32, ndim=1),  # [G']
    num_gaussain_ids: int,  # [G']
    output: wp.array(dtype=wp.float32, ndim=3),  # [N, D, 2]
) -> None:
    """The potential is represented as a sum of isotropic gaussians
    as proposed here: https://www.nature.com/articles/s41565-023-01595-w

    In order to improve the convergance, the real and imaginary part of the
    Potential are constrained by the ratio: V_{real} = -C * V_{imag}.
    The ratio is inspired by section 4.8 from this article:
    https://www.sciencedirect.com/science/article/pii/S0304399124001475

    Each atom has it's own C coefficient
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
        coupling_constant = coupling_constants[g]
        if _check_bbox_and_amp(
            difference=difference, radius=radius, amplitude=amplitude
        ):
            diff_sqr = (
                difference.x * difference.x
                + difference.y * difference.y
                + difference.z * difference.z
            )
            value_re = amplitude * wp.exp(-diff_sqr / sigma / sigma * 0.5)
            value_im = -coupling_constant * value_re
            output[n, d, 0] += value_re
            output[n, d, 1] += value_im


class _GaussianSampleCoupled(torch.autograd.Function):
    """The class is a Pytorch wrapper for gaussian_sample_kernel_coupled function

    :param th: _description_
    :type th: _type_
    :return: _description_
    :rtype: _type_
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        positions: Tensor,  # [G, 3]
        amplitudes: Tensor,  # [G]
        sigmas: Tensor,  # [G]
        coupling_constants: Tensor,  # [G]
        grid: Tensor,  # [N, D_slice, 3],
        gaussian_ids: Tensor,  # [G']
    ) -> Tensor:  # [N, D_slice]
        device = wp.device_from_torch(grid.device)
        N, D, _ = grid.shape
        G, _ = positions.shape

        ctx.dim = (N, D)
        ctx.out = wp.zeros(
            (N, D, 2), dtype=wp.float32, device=device, requires_grad=True
        )

        ctx.positions = wp.from_torch(positions, dtype=wp.vec3, requires_grad=True)
        ctx.amplitudes = wp.from_torch(amplitudes, requires_grad=True)
        ctx.sigmas = wp.from_torch(sigmas, requires_grad=True)
        ctx.coupling_constants = wp.from_torch(coupling_constants, requires_grad=True)
        ctx.grid = wp.from_torch(grid, dtype=wp.vec3, requires_grad=False)
        ctx.gaussian_ids = wp.from_torch(
            gaussian_ids, dtype=wp.int32, requires_grad=False
        )
        ctx.num_gaussain_ids = gaussian_ids.shape[0]
        wp.launch(
            kernel=_gaussian_sample_kernel_coupled,
            dim=ctx.dim,
            inputs=[
                ctx.positions,
                ctx.amplitudes,
                ctx.sigmas,
                ctx.coupling_constants,
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
        positions_grad = wp.zeros_like(ctx.positions)
        amplitudes_grad = wp.zeros_like(ctx.amplitudes)
        sigmas_grad = wp.zeros_like(ctx.sigmas)
        coupling_constants_grad = wp.zeros_like(ctx.coupling_constants)
        wp.launch(
            kernel=_gaussian_sample_kernel_coupled,
            dim=ctx.dim,
            inputs=[
                ctx.positions,
                ctx.amplitudes,
                ctx.sigmas,
                ctx.coupling_constants,
                ctx.grid,
                ctx.gaussian_ids,
                ctx.num_gaussain_ids,
            ],
            outputs=[ctx.out],
            adjoint=True,
            adj_inputs=[
                positions_grad,
                amplitudes_grad,
                sigmas_grad,
                coupling_constants_grad,
                None,
                None,
                None,
            ],
            adj_outputs=[ctx.out.grad],
            device=device,
        )
        positions_grad = wp.to_torch(positions_grad)
        amplitudes_grad = wp.to_torch(amplitudes_grad)
        sigmas_grad = wp.to_torch(sigmas_grad)
        coupling_constants_grad = wp.to_torch(coupling_constants_grad)

        return (
            positions_grad,
            amplitudes_grad,
            sigmas_grad,
            coupling_constants_grad,
            None,
            None,
        )


def advanced_gaussian_sample_coupled(
    positions: Tensor,  # [G, 3]
    amplitudes: Tensor,  # [G]
    sigmas: Tensor,  # [G]
    coupling_constants: Tensor,  # [G],
    grid: Tensor,  # [N, D_slice, 3],
    gaussian_ids: Tensor,  # [G']
) -> Tensor:
    return _GaussianSampleCoupled.apply(
        positions, amplitudes, sigmas, coupling_constants, grid, gaussian_ids
    )
