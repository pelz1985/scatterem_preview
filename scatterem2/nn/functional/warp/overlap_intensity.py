import torch
import warp as wp
from torch import Tensor


@wp.kernel
def _overlap_intensity_from_wave_kernel(
    r: wp.array2d(dtype=wp.int32),
    wave: wp.array4d(dtype=wp.vec2),
    out: wp.array2d(dtype=wp.float32),
) -> None:
    """
    :param r:                   K x 2
    :param wave_intensity: MY x MX
    :param out:                 1 x NY x NX
    """
    k, my, mx = wp.tid()

    Nmodes = wave.shape[0]
    K = wave.shape[1]
    MY = wave.shape[2]
    MX = wave.shape[3]
    if k < K and my < MY and mx < MX:
        y = r[k, 0]
        x = r[k, 1]
        y = y + my
        x = x + mx

        if y >= out.shape[1]:
            y = y % out.shape[1]
        if x >= out.shape[2]:
            x = x % out.shape[2]

        # Handle negative indices
        if y < 0:
            y = y % out.shape[1]
        if x < 0:
            x = x % out.shape[2]

        if y >= out.shape[1] or x >= out.shape[2] or y < 0 or x < 0:
            return
        intensity = float(0.0)
        for mode in range(Nmodes):
            intensity += (
                wave[mode, k, my, mx][0] ** 2.0 + wave[mode, k, my, mx][1] ** 2.0
            )
        wp.atomic_add(out, y, x, intensity)


def overlap_intensity_from_wave(r: Tensor, waves: Tensor, out: Tensor) -> None:
    r = wp.from_torch(r.data)
    w = torch.view_as_real(waves.data)
    w = wp.from_torch(w, dtype=wp.vec2)
    out = wp.from_torch(out.data)
    wp.launch(
        kernel=_overlap_intensity_from_wave_kernel,
        dim=(r.shape[0], w.shape[-2], w.shape[-1]),
        inputs=[r, w],
        outputs=[out],
        device=r.device,
    )


@wp.kernel
def _overlap_intensity_no_subpix_kernel(
    r: wp.array2d(dtype=wp.int32),
    wave_intensity: wp.array2d(dtype=wp.float32),
    out: wp.array3d(dtype=wp.float32),
) -> None:
    """
    :param r:                   K x 2
    :param wave_intensity: MY x MX
    :param out:                 1 x NY x NX
    """
    k, my, mx = wp.tid()

    if k < r.shape[0] and my < wave_intensity.shape[0] and mx < wave_intensity.shape[1]:
        y = r[k, 0]
        x = r[k, 1]
        y = y + my
        x = x + mx

        # Handle periodic boundary conditions more safely
        if y >= out.shape[1]:
            y = y % out.shape[1]
        if x >= out.shape[2]:
            x = x % out.shape[2]

        # Handle negative indices
        if y < 0:
            y = y % out.shape[1]
        if x < 0:
            x = x % out.shape[2]

        if y >= out.shape[1] or x >= out.shape[2] or y < 0 or x < 0:
            return

        wp.atomic_add(out, 0, y, x, wave_intensity[my, mx])


def overlap_intensity(r: Tensor, wave_intensity: Tensor, out: Tensor) -> None:
    r = wp.from_torch(r.data)
    wave_intensity = wp.from_torch(wave_intensity.data)
    out = wp.from_torch(out.data)
    wp.launch(
        kernel=_overlap_intensity_no_subpix_kernel,
        dim=(r.shape[0], *wave_intensity.shape),
        inputs=[r, wave_intensity],
        outputs=[out],
        device=r.device,
    )
