from typing import Tuple

import torch
import warp as wp
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

# @ti.kernel
# def overlap(
#     r: ti.types.ndarray(ndim=2),
#     z: ti.types.ndarray(ndim=5),
#     out: ti.types.ndarray(ndim=4),
# ):
#     """

#     :param r:   K x 2
#     :param z:   1 x K x MY x MX x 2
#     :param out: 1 x NY x NX x 2
#     :return:
#     """


#     ti.loop_config(parallelize=8, block_dim=256)
#     for b, k, my, mx, i in z:
#         y = r[k, 0]
#         x = r[k, 1]
#         ti.atomic_add(out[0, y + my, x + mx, i], z[b, k, my, mx, i])
@wp.kernel
def _batch_crop_backward_kernel(
    positions: wp.array2d(dtype=wp.int32),
    batches: wp.array4d(dtype=wp.vec2f),
    Ny: int,
    Nx: int,
    volume_out: wp.array3d(dtype=wp.vec2f),
) -> None:
    """
    :param positions:   K x 2
    :param volume:   NZ x NY x NX, vec2 float
    :param out: NZ x K x MY x MX, vec2 float
    :return:
    """

    nz, k, my, mx = wp.tid()
    # Sanity checks
    if (
        k >= positions.shape[0]
        or my >= batches.shape[2]
        or mx >= batches.shape[3]
        or nz >= volume_out.shape[0]
    ):
        return

    # Compute indices
    py = positions[k, 0]
    px = positions[k, 1]

    y = py + my
    x = px + mx

    # Handle negative indices
    if y < 0:
        y += Ny
    if x < 0:
        x += Nx

    # Handle periodic boundary conditions more safely
    if y >= Ny:
        y = y % Ny
    if x >= Nx:
        x = x % Nx

    if y >= Ny or x >= Nx or y < 0 or x < 0:
        return

    wp.atomic_add(volume_out, nz, y, x, batches[nz, k, my, mx])


@wp.kernel
def _batch_crop_kernel(
    positions: wp.array2d(dtype=wp.int32),
    volume: wp.array3d(dtype=wp.vec2f),
    Ny: int,
    Nx: int,
    batches_out: wp.array4d(dtype=wp.vec2f),
) -> None:
    """
    :param positions:   K x 2
    :param volume:   NZ x NY x NX, vec2 float
    :param out: NZ x K x MY x MX, vec2 float
    :return:
    """

    nz, k, my, mx = wp.tid()

    # Sanity checks
    if (
        k >= positions.shape[0]
        or my >= batches_out.shape[2]
        or mx >= batches_out.shape[3]
        or nz >= volume.shape[0]
    ):
        return

    # Compute indices
    py = positions[k, 0]
    px = positions[k, 1]

    y = py + my
    x = px + mx

    # Handle negative indices
    if y < 0:
        y += Ny
    if x < 0:
        x += Nx

    # Handle periodic boundary conditions more safely
    if y >= Ny:
        y = y % Ny
    if x >= Nx:
        x = x % Nx

    if y >= Ny or x >= Nx or y < 0 or x < 0:
        return

    batches_out[nz, k, my, mx] = volume[nz, y, x]


# for T in [wp.vec2h, wp.vec2f, wp.vec2d]:
#     wp.overload(
#         _batch_crop_kernel,
#         {
#             "positions": wp.array2d(dtype=wp.int32),
#             "volume": wp.array3d(dtype=T),
#             "batches_out": wp.array4d(dtype=T),
#         },
#     )
#     wp.overload(
#         _batch_crop_backward_kernel,
#         {
#             "positions": wp.array2d(dtype=wp.int32),
#             "batches": wp.array4d(dtype=T),
#             "volume_out": wp.array3d(dtype=T),
#         },
#     )


class _BatchCrop(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx, volume: Tensor, waves: Tensor, positions: Tensor
    ) -> Tensor:
        """Forward pass of BatchCrop operation.

        Args:
            ctx: FunctionCtx object to store tensors for backward pass
            positions: Tensor of shape (K, 2) containing crop positions
            volume: Complex input volume tensor of shape (NZ, NY, NX)
            psi: Complex probe tensor to store crops of shape (Nmodes, K, MY, MX)

        Returns:
            Tensor containing cropped patches from volume at specified positions
        """
        # assert positions.dtype == torch.int32
        # assert torch.all(torch.isfinite(positions)), "positions contain NaNs"
        # assert torch.all(positions >= 0), "positions contain NaNs"
        # print(f"positions.shape = {positions.shape}")
        # print(f"positions.max() = {positions.max()}")
        K, _ = positions.shape
        NZ, NY, NX = volume.shape
        MY, MX = waves.shape[-2:]
        device = wp.device_from_torch(volume.device)
        # print(f"batch_crop: device: {device}")
        ctx.positions_torch = positions
        ctx.volume_torch = volume
        ctx.out_torch = torch.zeros(
            (NZ, K, MY, MX),
            device=volume.device,
            dtype=volume.dtype,
            requires_grad=True,
        )

        ctx.positions_wp = wp.from_torch(
            ctx.positions_torch, dtype=wp.int32, requires_grad=False
        )
        ctx.volume_wp = wp.from_torch(
            torch.view_as_real(ctx.volume_torch), dtype=wp.vec2f, requires_grad=True
        )
        ctx.out_wp = wp.from_torch(
            torch.view_as_real(ctx.out_torch), dtype=wp.vec2f, requires_grad=True
        )
        # print(f"batch_crop: ctx.out_wp.shape = {ctx.out_wp.shape}")

        # import time
        # time.sleep(0.5)
        # ctx.save_for_backward(ctx.positions_torch, ctx.volume_torch)

        wp.launch(
            kernel=_batch_crop_kernel,
            dim=ctx.out_wp.shape,
            inputs=[ctx.positions_wp, ctx.volume_wp, NY, NX],
            outputs=[ctx.out_wp],
            device=device,
            record_tape=False,
        )
        out = wp.to_torch(ctx.out_wp)
        assert torch.all(torch.isfinite(volume)), "NaNs in volume"
        assert torch.all(torch.isfinite(out)), "NaNs in out"
        return torch.view_as_complex(out)

    @staticmethod
    def backward(ctx: FunctionCtx, adj_out: Tensor) -> Tuple[Tensor | None, ...]:
        device = wp.device_from_torch(adj_out.device)
        shape = ctx.out_wp.shape

        batches_grad = wp.from_torch(
            torch.view_as_real(adj_out), dtype=wp.vec2f, requires_grad=False
        )
        # assert ctx.positions_torch.dtype == torch.int32
        # assert torch.all(torch.isfinite(adj_out)), "adj_out contain NaNs"
        wp.launch(
            kernel=_batch_crop_backward_kernel,
            dim=shape,
            inputs=[
                ctx.positions_wp,
                batches_grad,
                ctx.volume_wp.shape[1],
                ctx.volume_wp.shape[2],
            ],
            outputs=[ctx.volume_wp.grad],
            record_tape=False,
            device=device,
        )
        gtorch = wp.to_torch(ctx.volume_wp.grad)
        ctx.out_wp = None
        # assert torch.all(torch.isfinite(adj_out)), "NaNs in adj_out"
        # assert torch.all(torch.isfinite(gtorch)), "NaNs in gtorch"
        # return adjoint w.r.t. inputs
        return (
            torch.view_as_complex(gtorch),
            None,
            None,
        )


def batch_crop(volume: Tensor, waves: Tensor, positions: Tensor) -> Tensor:
    return _BatchCrop.apply(volume, waves, positions)
