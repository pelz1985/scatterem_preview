from typing import Tuple

import torch
import warp as wp
from torch import Size, Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


@wp.kernel
def _cloud_to_grid_kernel(
    input: wp.array(dtype=wp.float32, ndim=2),  # [X, 6]
    H: int,
    W: int,
    output: wp.array(dtype=wp.float32, ndim=3),  # [N, (D_out * H_out * W_out), 2]
) -> None:
    x = wp.tid()  # global thread index
    n = wp.int32(input[x, 0])
    d = wp.int32(input[x, 1])
    h = wp.int32(input[x, 2])
    w = wp.int32(input[x, 3])
    j = wp.int32(input[x, 4])
    v = input[x, 5]

    dhw = H * W * d + W * h + w

    output[n, dhw, j] = v


class _CloudToGrid(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        input: Tensor,
        output_shape: Tuple | Size,  # [X, 6]  # [N, D, H, W]
    ) -> Tensor:
        device = wp.device_from_torch(input.device)
        N, D, H, W = output_shape

        ctx.output_shape = output_shape
        if input.requires_grad:
            input.retain_grad()
        ctx.input = wp.from_torch(input, requires_grad=True)
        ctx.output = wp.zeros((N, D * H * W, 2), dtype=wp.float32, requires_grad=True)

        X = input.shape[0]

        ctx.dim = X

        wp.launch(
            kernel=_cloud_to_grid_kernel,
            dim=ctx.dim,
            inputs=[
                ctx.input,
                H,
                W,
            ],
            outputs=[ctx.output],
            device=device,
        )
        out_torch = wp.to_torch(ctx.output)
        out_torch = torch.view_as_complex(out_torch).contiguous().reshape(output_shape)

        return out_torch

    @staticmethod
    def backward(ctx: FunctionCtx, out_adj: Tensor) -> Tuple[Tensor | None, ...]:
        device = wp.device_from_torch(out_adj.device)
        N, D, H, W = ctx.output_shape
        # ctx.input.grad = wp.zeros_like(ctx.input)
        out_adj = torch.view_as_real(out_adj.contiguous())
        ctx.output.grad = wp.from_torch(out_adj.reshape((N, D * H * W, 2)))
        wp.launch(
            kernel=_cloud_to_grid_kernel,
            dim=ctx.dim,
            inputs=[
                ctx.input,
                H,
                W,
            ],
            outputs=[ctx.output],
            adjoint=True,
            adj_inputs=[
                ctx.input.grad,
                0,
                0,
            ],
            adj_outputs=[ctx.output.grad],
            device=device,
        )
        input_grad = wp.to_torch(ctx.input.grad)
        return (
            input_grad,
            None,
        )


def cloud_to_grid(
    input: Tensor,
    output_shape: Tuple | Size,  # [X, 6]  # [N, D, H, W]
) -> Tensor:
    return _CloudToGrid.apply(input, output_shape)
