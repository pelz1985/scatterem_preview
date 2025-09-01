import warp as wp
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


@wp.kernel
def _create_affine_grid_3d(
    theta: wp.array(dtype=wp.float32, ndim=3),  # [N, 3, 4]
    D: wp.int32,
    H: wp.int32,
    W: wp.int32,
    normalize: wp.bool,
    align_corners: wp.bool,  # 0 or 1
    out_grid: wp.array(dtype=wp.vec3, ndim=4),  # [N, D, H, W] (each element is a vec3)
) -> None:
    """
    Kernel replicating torch.nn.functional.affine_grid for 3D.

    Args:
        theta: [N, 3, 4], each 3x4 block is an affine transform for one batch element.
        out_grid: [N, D, H, W], storing wp.vec3 coordinates.
        D,H,W: volume dimensions for the output grid.
        normalize: if False, normalization is not applied
        align_corners: 1 => align_corners=True, 0 => align_corners=False
    """
    n, d, h, w = wp.tid()  # global thread index

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
    a11 = theta[n, 0, 0]
    a12 = theta[n, 0, 1]
    a13 = theta[n, 0, 2]
    b1 = theta[n, 0, 3]

    a21 = theta[n, 1, 0]
    a22 = theta[n, 1, 1]
    a23 = theta[n, 1, 2]
    b2 = theta[n, 1, 3]

    a31 = theta[n, 2, 0]
    a32 = theta[n, 2, 1]
    a33 = theta[n, 2, 2]
    b3 = theta[n, 2, 3]

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
    out_grid[n, d, h, w] = wp.vec3(x_out, y_out, z_out)


class _AffineGrid(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        theta_torch: Tensor,
        size: tuple,  # (N, C, D, H, W)
        normalize: bool = True,
        align_corners: bool = False,
        input_size: tuple | None = None,  # (D_in, H_in, W_in)
    ) -> Tensor:
        """
        Replicates torch.nn.functional.affine_grid for 3D using Warp.

        Args:
            theta_torch:  [N, 3, 4] float32 Tensor (one 3x4 matrix per batch element)
            size:         (N, C, D, H, W) describing the output grid
            align_corners:
                True  => replicate PyTorch's 'align_corners=True'
                False => replicate PyTorch's 'align_corners=False'
            device: "cuda" or "cpu", depending on how you built Warp.

        Returns:
            A PyTorch tensor of shape [N, D, H, W, 3], containing normalized coords.
        """
        device = wp.device_from_torch(theta_torch.device)

        ctx.size = size
        N, C, D, H, W = size
        ctx.normalize = normalize
        ctx.align_flag = align_corners
        ctx.input_size = input_size

        # 1) Convert theta from PyTorch to Warp
        ctx.theta_wp = wp.from_torch(theta_torch, dtype=wp.float32)

        # 2) Allocate Warp array of shape [N, D, H, W] with dtype=vec3
        ctx.out_grid_wp = wp.empty(
            shape=(N, D, H, W),
            dtype=wp.vec3,
            device=device,
        )

        if input_size is None:
            wp.launch(
                kernel=_create_affine_grid_3d,
                dim=(N, D, H, W),
                inputs=[ctx.theta_wp, D, H, W, ctx.normalize, ctx.align_flag],
                outputs=[ctx.out_grid_wp],
                device=device,
            )
        else:
            D_in, H_in, W_in = ctx.input_size
            wp.launch(
                kernel=_create_affine_grid_3d,
                dim=(N, D, H, W),
                inputs=[ctx.theta_wp, D_in, H_in, W_in, ctx.normalize, ctx.align_flag],
                outputs=[ctx.out_grid_wp],
                device=device,
            )

        # 4) Convert Warp array back to PyTorch
        #    This typically yields a torch tensor of shape [N, D, H, W] with "vec3" data.
        grid_torch = wp.to_torch(ctx.out_grid_wp)

        # 5) Reshape or slice to get final shape [N, D, H, W, 3].
        #    Depending on Warp/padding, we might see an extra channel, so we handle both cases:
        if grid_torch.dim() == 5 and grid_torch.shape[-1] == 4:
            # Some builds store vec3 as 4 floats, so slice out the first 3
            grid_torch = grid_torch[..., :3]
        elif grid_torch.dim() == 4:
            # If we have shape [N, D, H, W], each element is a 3-float struct
            # we can reshape to [N, D, H, W, 3]
            grid_torch = grid_torch.view(N, D, H, W, 3)

        return grid_torch

    @staticmethod
    def backward(ctx: FunctionCtx, adj_out: Tensor) -> Tensor:
        ctx.out_grid_wp.grad = wp.from_torch(adj_out, dtype=wp.vec3)
        ctx.theta_wp.grad = wp.zeros_like(ctx.theta_wp)
        N, C, D, H, W = ctx.size
        if ctx.input_size is None:
            wp.launch(
                kernel=_create_affine_grid_3d,
                dim=(N, D, H, W),
                inputs=[ctx.theta_wp, D, H, W, ctx.normalize, ctx.align_flag],
                outputs=[ctx.out_grid_wp],
                device=ctx.device,
                adjoint=True,
                adj_inputs=[ctx.theta_wp.grtad, None, None, None, None, None],
                adj_outputs=[ctx.out_grid_wp.grad],
            )
        else:
            D_in, H_in, W_in = ctx.input_size
            wp.launch(
                kernel=_create_affine_grid_3d,
                dim=(N, D, H, W),
                inputs=[ctx.theta_wp, D_in, H_in, W_in, ctx.normalize, ctx.align_flag],
                outputs=[ctx.out_grid_wp],
                device=ctx.device,
                adjoint=True,
                adj_inputs=[ctx.theta_wp.grtad, None, None, None, None, None],
                adj_outputs=[ctx.out_grid_wp.grad],
            )
        return (
            ctx.theta_wp.grad,
            None,
            None,
            None,
            None,
        )


def affine_grid_legacy(
    theta_torch: Tensor,
    size: tuple,  # (N, C, D, H, W)
    normalize: bool = True,
    align_corners: bool = False,
    input_size: tuple | None = None,
) -> Tensor:
    """
    Replicates torch.nn.functional.affine_grid for 3D using Warp.

    Args:
        theta_torch:  [N, 3, 4] float32 Tensor (one 3x4 matrix per batch element)
        size:         (N, C, D, H, W) describing the output grid
        align_corners:
            True  => replicate PyTorch's 'align_corners=True'
            False => replicate PyTorch's 'align_corners=False'
        device: "cuda" or "cpu", depending on how you built Warp.

    Returns:
        A PyTorch tensor of shape [N, D, H, W, 3], containing normalized coords.
    """
    return _AffineGrid.apply(theta_torch, size, normalize, align_corners, input_size)
