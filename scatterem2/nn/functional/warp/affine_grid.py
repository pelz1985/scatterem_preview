from typing import Tuple

import warp as wp
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


@wp.kernel
def _create_affine_grid_3d(
    theta: wp.array(dtype=wp.float32, ndim=2),  # [3, 4]
    in_grid: wp.array(dtype=wp.int32, ndim=2),  # [N, 2] (h, w)
    depth_slice_start: wp.int32,
    D: wp.int32,
    H: wp.int32,
    W: wp.int32,
    normalize: wp.bool,
    align_corners: wp.bool,  # 0 or 1
    out_grid: wp.array(dtype=wp.float32, ndim=3),  # [N, Ds, 3] (d, h, w)
) -> None:
    """
    Kernel replicating torch.nn.functional.affine_grid for 3D.
    """
    n, d_id = wp.tid()  # global thread index

    h = in_grid[n, 0]
    w = in_grid[n, 1]
    d = depth_slice_start + d_id

    # Convert indices to float
    d_f = wp.float32(d)
    h_f = wp.float32(h)
    w_f = wp.float32(w)
    D_f = wp.float32(D)
    H_f = wp.float32(H)
    W_f = wp.float32(W)

    # Compute normalized coordinates in [-1, 1]
    if normalize:
        if align_corners:
            # align_corners=True
            z = -1.0 + 2.0 * (d_f / (D_f - 1.0))
            y = -1.0 + 2.0 * (h_f / (H_f - 1.0))
            x = -1.0 + 2.0 * (w_f / (W_f - 1.0))
        else:
            # align_corners=False
            z = -1.0 + 2.0 * ((d_f + 0.5) / D_f)
            y = -1.0 + 2.0 * ((h_f + 0.5) / H_f)
            x = -1.0 + 2.0 * ((w_f + 0.5) / W_f)
    else:
        if align_corners:
            z = d_f - (D_f - 1.0) / 2.0
            y = h_f - (H_f - 1.0) / 2.0
            x = w_f - (W_f - 1.0) / 2.0
        else:
            z = d_f - (D_f / 2.0 - 0.5)
            y = h_f - (H_f / 2.0 - 0.5)
            x = w_f - (W_f / 2.0 - 0.5)

    # Retrieve affine parameters:
    #   [ a11 a12 a13 b1 ]
    #   [ a21 a22 a23 b2 ]
    #   [ a31 a32 a33 b3 ]
    a11 = theta[0, 0]
    a12 = theta[0, 1]
    a13 = theta[0, 2]
    b1 = theta[0, 3]

    a21 = theta[1, 0]
    a22 = theta[1, 1]
    a23 = theta[1, 2]
    b2 = theta[1, 3]

    a31 = theta[2, 0]
    a32 = theta[2, 1]
    a33 = theta[2, 2]
    b3 = theta[2, 3]

    # Apply transform: [x_out, y_out, z_out] = theta * [x, y, z, 1]^T
    x_out = a11 * x + a12 * y + a13 * z + b1
    y_out = a21 * x + a22 * y + a23 * z + b2
    z_out = a31 * x + a32 * y + a33 * z + b3

    if not normalize:
        if align_corners:
            z_out += (D_f - 1.0) / 2.0
            y_out += (H_f - 1.0) / 2.0
            x_out += (W_f - 1.0) / 2.0
        else:
            z_out += D_f / 2.0 - 0.5
            y_out += H_f / 2.0 - 0.5
            x_out += W_f / 2.0 - 0.5

    # Store in out_grid as a vec3
    out_grid[n, d_id, 0] = x_out
    out_grid[n, d_id, 1] = y_out
    out_grid[n, d_id, 2] = z_out


class _AffineGrid(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        yx_grid: Tensor,  # (N, 2)
        z_start: int,
        z_len: int,
        theta_torch: Tensor,  # (3, 4)
        input_size: Tuple[int, int, int],  # (D_in, H_in, W_in)
        normalize: bool = False,
        align_corners: bool = False,
    ) -> Tensor:
        device = wp.device_from_torch(theta_torch.device)

        N = yx_grid.shape[0]
        yx_grid_wp = wp.from_torch(yx_grid)
        theta_wp = wp.from_torch(theta_torch, dtype=wp.float32)

        out_grid_wp = wp.empty(
            shape=(N, z_len, 3),
            dtype=wp.float32,
            device=device,
            requires_grad=False,
        )

        D_in, H_in, W_in = input_size
        wp.launch(
            kernel=_create_affine_grid_3d,
            dim=(N, z_len),
            inputs=[
                theta_wp,
                yx_grid_wp,
                z_start,
                D_in,
                H_in,
                W_in,
                normalize,
                align_corners,
            ],
            outputs=[out_grid_wp],
            device=device,
        )
        grid_torch = wp.to_torch(out_grid_wp)

        return grid_torch

    @staticmethod
    def backward(ctx: FunctionCtx, adj_out: Tensor) -> Tensor:
        raise NotImplementedError


def affine_grid(
    yx_grid: Tensor,  # (N, 2)
    z_start: int,
    z_len: int,
    theta_torch: Tensor,  # (3, 4)
    input_size: Tuple[int, int, int],  # (D_in, H_in, W_in)
    normalize: bool = False,
    align_corners: bool = False,
) -> Tensor:
    return _AffineGrid.apply(
        yx_grid,
        z_start,
        z_len,
        theta_torch,
        input_size,
        normalize,
        align_corners,
    )
