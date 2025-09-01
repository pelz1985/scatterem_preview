from typing import Tuple

import warp as wp
from torch import Tensor


@wp.func
def _check_bbox(
    position: wp.vec3,
    radius: wp.float32,
    grid_mins: wp.vec3,
    grid_maxs: wp.vec3,
) -> bool:
    if radius < 1e-7:
        return False

    position_mins = position - wp.vec3(radius, radius, radius)
    position_maxs = position + wp.vec3(radius, radius, radius)

    res = True
    for i in range(3):
        g_min = position_mins[i]
        g_max = position_maxs[i]
        p_min = grid_mins[i]
        p_max = grid_maxs[i]
        res = res and not (g_max < p_min or p_max < g_min)

    return res


@wp.kernel
def _compute_gaussians_mask_kernel_complex(
    positions: wp.array(dtype=wp.vec3, ndim=1),  # [G]
    amplitudes: wp.array(dtype=wp.float32, ndim=2),  # [G, 2]
    sigmas: wp.array(dtype=wp.float32, ndim=2),  # [G, 2]
    grid_mins: wp.vec3,
    grid_maxs: wp.vec3,
    gaussians_mask: wp.array(dtype=wp.bool, ndim=1),
) -> None:
    g = wp.tid()  # global thread index
    pos = positions[g]

    amp_re = amplitudes[g, 0]
    amp_im = amplitudes[g, 1]
    radius_re = sigmas[g, 0] * 3.0
    radius_im = sigmas[g, 1] * 3.0

    res_re = amp_re > -1e-7 and _check_bbox(
        position=pos,
        radius=radius_re,
        grid_mins=grid_mins,
        grid_maxs=grid_maxs,
    )

    res_im = amp_im > -1e-7 and _check_bbox(
        position=pos,
        radius=radius_im,
        grid_mins=grid_mins,
        grid_maxs=grid_maxs,
    )
    gaussians_mask[g] = res_re or res_im


@wp.kernel
def _compute_gaussians_mask_kernel_real(
    positions: wp.array(dtype=wp.vec3, ndim=1),  # [G]
    amplitudes: wp.array(dtype=wp.float32, ndim=1),  # [G]
    sigmas: wp.array(dtype=wp.float32, ndim=1),  # [G]
    grid_mins: wp.vec3,
    grid_maxs: wp.vec3,
    gaussians_mask: wp.array(dtype=wp.bool, ndim=1),
) -> None:
    g = wp.tid()  # global thread index
    pos = positions[g]

    amp = amplitudes[g]
    radius = sigmas[g] * 3.0

    res = amp > -1e-7 and _check_bbox(
        position=pos,
        radius=radius,
        grid_mins=grid_mins,
        grid_maxs=grid_maxs,
    )

    gaussians_mask[g] = res


def compute_gaussians_mask(
    positions: Tensor,
    amplitudes: Tensor,
    sigmas: Tensor,
    grid_mins: Tuple,
    grid_maxs: Tuple,
) -> Tensor:
    device = positions.device
    gaussians_mask = wp.zeros(
        positions.shape[0],
        dtype=wp.bool,
        device=wp.device_from_torch(device),
        requires_grad=False,
    )
    positions = wp.from_torch(positions, dtype=wp.vec3, requires_grad=False)
    amplitudes = wp.from_torch(amplitudes, dtype=wp.float32)
    sigmas = wp.from_torch(sigmas, dtype=wp.float32)
    grid_mins = wp.vec3(*grid_mins)
    grid_maxs = wp.vec3(*grid_maxs)

    kernel_fun = (
        _compute_gaussians_mask_kernel_complex
        if len(amplitudes.shape) > 1
        else _compute_gaussians_mask_kernel_real
    )
    wp.launch(
        kernel=kernel_fun,
        dim=positions.shape[0],
        inputs=[positions, amplitudes, sigmas, grid_mins, grid_maxs],
        outputs=[gaussians_mask],
        device=wp.device_from_torch(device),
    )
    return wp.to_torch(gaussians_mask)
