from typing import Tuple

import torch
import warp as wp
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


@wp.func
def _sample_vec2(
    n_i: wp.int32,
    c_i: wp.int32,
    z_i: wp.float32,
    y_i: wp.float32,
    x_i: wp.float32,
    Df: wp.float32,
    Hf: wp.float32,
    Wf: wp.float32,
    H_in: wp.int32,
    W_in: wp.int32,
    input: wp.array(dtype=wp.vec2, ndim=2),
) -> wp.vec2:
    # zero padding if out-of-bounds
    if z_i < 0.0 or z_i >= Df or y_i < 0.0 or y_i >= Hf or x_i < 0.0 or x_i >= Wf:
        return wp.vec2(0.0, 0.0)

    zi = wp.int32(z_i)
    yi = wp.int32(y_i)
    xi = wp.int32(x_i)
    flat_idx = zi * (H_in * W_in) + yi * W_in + xi
    return input[c_i, flat_idx]


@wp.func
def _lerp(a: wp.float32, b: wp.float32, t: wp.float32) -> wp.float32:
    return a * (1.0 - t) + b * t


@wp.kernel
def _grid_sample_3d_bilinear(
    input: wp.array(dtype=wp.vec2, ndim=2),  # [C, (D_in * H_in * W_in)]
    grid: wp.array(dtype=wp.vec3, ndim=2),  # [N, (D_out * H_out * W_out)] of vec3
    D_in: wp.int32,
    H_in: wp.int32,
    W_in: wp.int32,
    unnormalize: wp.bool,
    align_corners: wp.bool,  # 0 => False, 1 => True
    output: wp.array(dtype=wp.vec2, ndim=3),  # [N, C, (D_out * H_out * W_out)]
) -> None:
    """
    3D trilinear sampling (a.k.a. 'bilinear' in PyTorch terms), for complex data
    stored as wp.vec2 (real, imag). Zeros padding if out-of-bounds.

    Flattened layout:
      - input:  [C, (D_in*H_in*W_in)], each element is vec2
      - grid:   [N, (D_out*H_out*W_out)], each element is vec3 (x,y,z in [-1,1])
      - output: [N, C, (D_out*H_out*W_out)], each element is vec2

    D_in,H_in,W_in  => input dims
    D_out,H_out,W_out => output dims
    unnormalize => 0 => unnormalization is not applied
    align_corners => 1 => PyTorch align_corners=True, 0 => False
    """
    n, c, out_voxel_idx = wp.tid()  # global thread index

    # normalized coords
    coord = grid[n, out_voxel_idx]  # a wp.vec3
    x = coord[0]  # in [-1,1]
    y = coord[1]
    z = coord[2]

    # map normalized -> input space
    Wf = wp.float32(W_in)
    Hf = wp.float32(H_in)
    Df = wp.float32(D_in)

    if unnormalize:
        if align_corners == 1:
            x_in = 0.5 * (x + 1.0) * (Wf - 1.0)
            y_in = 0.5 * (y + 1.0) * (Hf - 1.0)
            z_in = 0.5 * (z + 1.0) * (Df - 1.0)
        else:
            x_in = ((x + 1.0) * Wf - 1.0) * 0.5
            y_in = ((y + 1.0) * Hf - 1.0) * 0.5
            z_in = ((z + 1.0) * Df - 1.0) * 0.5
    else:
        x_in = x
        y_in = y
        z_in = z

    # integer corner coords
    x0 = wp.floor(x_in)
    x1 = x0 + 1.0
    y0 = wp.floor(y_in)
    y1 = y0 + 1.0
    z0 = wp.floor(z_in)
    z1 = z0 + 1.0

    # fractional
    tx = x_in - x0
    ty = y_in - y0
    tz = z_in - z0

    c000 = _sample_vec2(n, c, z0, y0, x0, Df, Hf, Wf, H_in, W_in, input)
    c001 = _sample_vec2(n, c, z0, y0, x1, Df, Hf, Wf, H_in, W_in, input)
    c010 = _sample_vec2(n, c, z0, y1, x0, Df, Hf, Wf, H_in, W_in, input)
    c011 = _sample_vec2(n, c, z0, y1, x1, Df, Hf, Wf, H_in, W_in, input)
    c100 = _sample_vec2(n, c, z1, y0, x0, Df, Hf, Wf, H_in, W_in, input)
    c101 = _sample_vec2(n, c, z1, y0, x1, Df, Hf, Wf, H_in, W_in, input)
    c110 = _sample_vec2(n, c, z1, y1, x0, Df, Hf, Wf, H_in, W_in, input)
    c111 = _sample_vec2(n, c, z1, y1, x1, Df, Hf, Wf, H_in, W_in, input)

    cx00 = wp.vec2(_lerp(c000.x, c001.x, tx), _lerp(c000.y, c001.y, tx))
    cx01 = wp.vec2(_lerp(c010.x, c011.x, tx), _lerp(c010.y, c011.y, tx))
    cx10 = wp.vec2(_lerp(c100.x, c101.x, tx), _lerp(c100.y, c101.y, tx))
    cx11 = wp.vec2(_lerp(c110.x, c111.x, tx), _lerp(c110.y, c111.y, tx))

    cy0 = wp.vec2(_lerp(cx00.x, cx01.x, ty), _lerp(cx00.y, cx01.y, ty))
    cy1 = wp.vec2(_lerp(cx10.x, cx11.x, ty), _lerp(cx10.y, cx11.y, ty))

    c_out = wp.vec2(_lerp(cy0.x, cy1.x, tz), _lerp(cy0.y, cy1.y, tz))

    # store
    output[n, c, out_voxel_idx] = c_out


class _GridSample(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        input_torch: Tensor,  # [N, C, D_in, H_in, W_in], complex dtype
        grid_torch: Tensor,  # [N, D_out, H_out, W_out, 3], float
        unnormalize: bool = True,
        align_corners: bool = False,
        object_norm: Tensor | None = None,
    ) -> Tensor:
        """
        Equivalent to PyTorch's grid_sample(..., mode='bilinear', padding_mode='zeros')
        but for complex data, done in one pass using wp.vec2 in Warp.

        Returns a complex PyTorch tensor [N, C, D_out, H_out, W_out].
        """
        device = wp.device_from_torch(input_torch.device)
        # --------------------------------------------------------------------------
        # 1) Check shapes, separate real & imaginary
        C, D_in, H_in, W_in = input_torch.shape

        N, D_out, H_out, W_out, _ = grid_torch.shape

        ctx.unnormalize = unnormalize
        ctx.align_flag = align_corners
        ctx.object_norm = object_norm

        # Flatten input => shape [N, C, D_in*H_in*W_in], each element => (real, imag)
        input_torch = input_torch.reshape(C, -1)
        ctx.input_wp = wp.from_torch(torch.view_as_real(input_torch), dtype=wp.vec2)
        # if input_torch.requires_grad:
        #     exit(1)

        # Flatten grid => shape [N, (D_out*H_out*W_out)], each element => vec3
        flat_out_size = D_out * H_out * W_out
        grid_torch = grid_torch.view(N, flat_out_size, 3)
        ctx.grid_wp = wp.from_torch(grid_torch, dtype=wp.vec3)

        # Allocate output => [N, C, flat_out_size] of vec2
        ctx.out_wp = wp.empty((N, C, flat_out_size), dtype=wp.vec2, device=device)

        ctx.dim = (N, C, flat_out_size)
        ctx.dhw_in = (D_in, H_in, W_in)
        # --------------------------------------------------------------------------
        # 2) Launch kernel
        wp.launch(
            kernel=_grid_sample_3d_bilinear,
            dim=ctx.dim,
            inputs=[
                ctx.input_wp,
                ctx.grid_wp,
                D_in,
                H_in,
                W_in,
                ctx.unnormalize,
                ctx.align_flag,
            ],
            outputs=[ctx.out_wp],
            device=device,
        )

        # --------------------------------------------------------------------------
        # 3) Convert back to PyTorch & rebuild complex tensor
        out_torch = wp.to_torch(
            ctx.out_wp
        )  # likely shape => [N, C, flat_out_size, 2-or-4]

        # # If there's a trailing channel of size 4 (due to vec2 alignment), slice first 2
        # if out_torch.dim() == 4 and out_torch.shape[-1] == 4:
        #     out_torch = out_torch[..., :2]

        # So now out_torch => [N, C, flat_out_size, 2]
        # We want [N, C, D_out, H_out, W_out, 2] => then .view_as_complex()
        out_torch = out_torch.view(N, C, D_out, H_out, W_out, 2)

        # Convert back to complex
        # If PyTorch >= 1.8, we can do .view_as_complex(...) by rearranging dims
        # or we can build the complex manually: real + i*imag
        # real_part = out_torch[..., 0]
        # imag_part = out_torch[..., 1]
        # out_complex = torch.complex(real_part, imag_part)
        out_complex = torch.view_as_complex(out_torch)

        return out_complex

    @staticmethod
    def backward(ctx: FunctionCtx, out_adj: Tensor) -> Tuple[Tensor | None, ...]:
        device = wp.device_from_torch(out_adj.device)
        N, C, D_out, H_out, W_out = out_adj.shape
        out_adj = (
            torch.view_as_real(out_adj).clone().reshape(N, C, D_out * H_out * W_out, 2)
        )

        ctx.out_wp.grad = wp.from_torch(out_adj, dtype=wp.vec2)
        ctx.input_wp.grad = wp.zeros_like(ctx.input_wp)
        ctx.grid_wp.grad = wp.zeros_like(ctx.grid_wp)
        D_in, H_in, W_in = ctx.dhw_in
        wp.launch(
            kernel=_grid_sample_3d_bilinear,
            dim=ctx.dim,
            inputs=[
                ctx.input_wp,
                ctx.grid_wp,
                D_in,
                H_in,
                W_in,
                ctx.unnormalize,
                ctx.align_flag,
            ],
            outputs=[ctx.out_wp],
            adjoint=True,
            adj_inputs=[ctx.input_wp.grad, ctx.grid_wp.grad, 1, 1, 1, 1, 1],
            adj_outputs=[ctx.out_wp.grad],
            device=device,
        )
        input_grad = wp.to_torch(ctx.input_wp.grad).view(C, D_in, H_in, W_in, 2)
        input_grad = torch.view_as_complex(input_grad)
        # if ctx.object_norm is not None:
        #     alpha=0.9
        #     denom = torch.sqrt(
        #         1e-16
        #         + ((1 - alpha) * ctx.object_norm) ** 2
        #         + (alpha * torch.max(ctx.object_norm)) ** 2
        #     )
        #     input_grad = input_grad / denom if input_grad is not None else None

        grid_grad = wp.to_torch(ctx.grid_wp.grad).view(N, D_out, H_out, W_out, 3)
        return (
            input_grad,
            grid_grad,
            None,
            None,
            None,
        )


def grid_sample_legacy(
    input_torch: Tensor,  # [C, D_in, H_in, W_in], complex dtype
    grid_torch: Tensor,  # [N, D_out, H_out, W_out, 3], float
    unnormalize: bool = True,
    align_corners: bool = False,
    object_norm: None | Tensor = None,
) -> Tensor:
    return _GridSample.apply(
        input_torch, grid_torch, unnormalize, align_corners, object_norm
    )
