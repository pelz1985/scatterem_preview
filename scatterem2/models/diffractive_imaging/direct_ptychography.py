import numpy as np
import torch
import warp as wp
from scipy.optimize import minimize
from torch.fft import fftfreq, fftshift

from scatterem2.utils.data.datasets import Dataset4dstem
from scatterem2.utils.warp import (
    aberration_function_cartesian,
    aperture,
    cabs,
    cconj,
    cexp,
    cmul,
)


@wp.kernel
def _disk_overlap_kernel(
    Qx_all: wp.array(dtype=wp.float32, ndim=1),
    Qy_all: wp.array(dtype=wp.float32, ndim=1),
    Kx_all: wp.array(dtype=wp.float32, ndim=1),
    Ky_all: wp.array(dtype=wp.float32, ndim=1),
    aberrations: wp.array(dtype=wp.float32, ndim=1),
    rotation: wp.float32,
    semiconvergence_angle: wp.float32,
    wavelength: wp.float32,
    Gamma: wp.array(dtype=wp.float32, ndim=4),
) -> None:
    # Get thread indices for 3D iteration
    j, ikx, iky = wp.tid()

    # Get values from arrays
    Qx = Qx_all[j]
    Qy = Qy_all[j]
    Kx = Kx_all[ikx]
    Ky = Ky_all[iky]

    # Apply rotation
    Qx_rot = Qx * wp.cos(rotation) - Qy * wp.sin(rotation)
    Qy_rot = Qx * wp.sin(rotation) + Qy * wp.cos(rotation)

    Qx = Qx_rot
    Qy = Qy_rot

    # Calculate chi for different positions
    chi = aberration_function_cartesian(Ky, Kx, wavelength, aberrations)
    apert = aperture(Ky, Kx, wavelength, semiconvergence_angle)
    expichi = cexp(1.0, -chi)
    A = cmul(wp.vec2(apert, wp.float32(0.0)), expichi)

    chi = aberration_function_cartesian(Ky + Qy, Kx + Qx, wavelength, aberrations)
    apert = aperture(Ky + Qy, Kx + Qx, wavelength, semiconvergence_angle)
    expichi = cexp(1.0, -chi)
    Ap = cmul(wp.vec2(apert, wp.float32(0.0)), expichi)

    chi = aberration_function_cartesian(Ky - Qy, Kx - Qx, wavelength, aberrations)
    apert = aperture(Ky - Qy, Kx - Qx, wavelength, semiconvergence_angle)
    expichi = cexp(1.0, -chi)
    Am = cmul(wp.vec2(apert, wp.float32(0.0)), expichi)

    # Calculate gamma: A* * Am - A * Ap*
    A_conj = cconj(A)
    Ap_conj = cconj(Ap)

    term1 = cmul(A_conj, Am)
    term2 = cmul(A, Ap_conj)
    gamma_complex = wp.vec2(term1[0] - term2[0], term1[1] - term2[1])

    # Store result (real and imaginary parts)
    Gamma[j, iky, ikx, 0] = gamma_complex[0]
    Gamma[j, iky, ikx, 1] = gamma_complex[1]


def disk_overlap_function(
    Qx_all: torch.Tensor,
    Qy_all: torch.Tensor,
    Kx_all: torch.Tensor,
    Ky_all: torch.Tensor,
    aberrations: torch.Tensor,
    rotation: float,
    semiconvergence_angle: float,
    wavelength: float,
) -> torch.Tensor:
    """Call the disk overlap kernel with proper dimensions"""
    device = wp.device_from_torch(Qx_all.device)
    Gamma = torch.zeros(
        (Qx_all.shape[0], Kx_all.shape[0], Ky_all.shape[0]),
        dtype=torch.complex64,
        device=Qx_all.device,
    )
    Gamma_wp = wp.from_torch(torch.view_as_real(Gamma))
    Qx_all_wp = wp.from_torch(Qx_all)
    Qy_all_wp = wp.from_torch(Qy_all)
    Kx_all_wp = wp.from_torch(Kx_all)
    Ky_all_wp = wp.from_torch(Ky_all)
    aberrations_wp = wp.from_torch(aberrations)

    # Convert scalar parameters to Warp float32
    rotation_wp = wp.float32(rotation)
    semiconvergence_angle_wp = wp.float32(semiconvergence_angle)
    wavelength_wp = wp.float32(wavelength)

    J, IKY, IKX = Gamma.shape
    wp.launch(
        kernel=_disk_overlap_kernel,
        dim=(J, IKY, IKX),
        inputs=[
            Qx_all_wp,
            Qy_all_wp,
            Kx_all_wp,
            Ky_all_wp,
            aberrations_wp,
            rotation_wp,
            semiconvergence_angle_wp,
            wavelength_wp,
        ],
        outputs=[Gamma_wp],
        device=device,
        record_tape=False,
    )

    # Convert back to PyTorch tensor

    return Gamma


@wp.kernel
def _single_sideband_kernel(
    G: wp.array(dtype=wp.vec2, ndim=4),
    Qx_all: wp.array(ndim=1),
    Qy_all: wp.array(ndim=1),
    Kx_all: wp.array(ndim=1),
    Ky_all: wp.array(ndim=1),
    aberrations: wp.array(ndim=1),
    rotation: wp.float32,
    semiconvergence_angle: wp.float32,
    eps: wp.float32,
    wavelength: wp.float32,
    object_bright_field: wp.array(ndim=3),
    object_ssb: wp.array(ndim=3),
) -> None:
    # IQY, IQX, IKY, IKX, cx = G.shape
    iqy, iqx, iky, ikx = wp.tid()

    Qx = Qx_all[iqx]
    Qy = Qy_all[iqy]
    Kx = Kx_all[ikx]
    Ky = Ky_all[iky]

    Qx_rot = Qx * wp.cos(rotation) - Qy * wp.sin(rotation)
    Qy_rot = Qx * wp.sin(rotation) + Qy * wp.cos(rotation)

    Qx = Qx_rot
    Qy = Qy_rot

    chi = aberration_function_cartesian(Ky, Kx, wavelength, aberrations)
    apert = wp.vec2(
        aperture(Ky, Kx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi = cexp(1.0, -chi)
    A = cmul(apert, expichi)

    chi = aberration_function_cartesian(Ky + Qy, Kx + Qx, wavelength, aberrations)
    apert = wp.vec2(
        aperture(Ky + Qy, Kx + Qx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi = cexp(1.0, -chi)
    A_plus = cmul(apert, expichi)

    chi = aberration_function_cartesian(Ky - Qy, Kx - Qx, wavelength, aberrations)
    apert = wp.vec2(
        aperture(Ky - Qy, Kx - Qx, wavelength, semiconvergence_angle), wp.float32(0.0)
    )
    expichi = cexp(1.0, -chi)
    Am = cmul(apert, expichi)

    gamma_complex = cmul(cconj(A), Am) - cmul(A, cconj(A_plus))

    Kplus = wp.sqrt((Kx + Qx) ** 2.0 + (Ky + Qy) ** 2.0)
    Kminus = wp.sqrt((Kx - Qx) ** 2.0 + (Ky - Qy) ** 2.0)
    K = wp.sqrt(Kx**2.0 + Ky**2.0)
    bright_field = K < semiconvergence_angle / wavelength
    double_overlap1 = (
        (Kplus < semiconvergence_angle / wavelength)
        and bright_field
        and (Kminus > semiconvergence_angle / wavelength)
    )

    Gamma_abs = cabs(gamma_complex)
    take = Gamma_abs > eps and bright_field

    # Get the complex value from G
    g_complex = G[iqy, iqx, iky, ikx]

    if take:
        val = cmul(g_complex, cconj(gamma_complex))
        wp.atomic_add(object_bright_field, iqy, iqx, 0, val[0])
        wp.atomic_add(object_bright_field, iqy, iqx, 1, val[1])

    if double_overlap1:
        val = cmul(g_complex, cconj(gamma_complex))
        wp.atomic_add(object_ssb, iqy, iqx, 0, val[0])
        wp.atomic_add(object_ssb, iqy, iqx, 1, val[1])

    if iqx == 0 and iqy == 0:
        abs_val = cabs(g_complex)
        wp.atomic_add(object_bright_field, iqy, iqx, 0, abs_val)
        wp.atomic_add(object_ssb, iqy, iqx, 0, abs_val)


def weak_phase_reconstruction(
    data_in: Dataset4dstem,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data_in.meta is None:
        raise ValueError("data_in.meta is None")
    # eps: float = 1e-3
    G = torch.fft.fft2(data_in.array, dim=(0, 1), norm="ortho")
    ny, nx, nky, nkx = data_in.shape

    Kx = fftshift(
        fftfreq(
            nkx,
            1 / (data_in.sampling[3] * data_in.shape[3]),
            dtype=data_in.dtype,
            device=data_in.device,
        )
    )
    Ky = fftshift(
        fftfreq(
            nky,
            1 / (data_in.sampling[2] * data_in.shape[2]),
            dtype=data_in.dtype,
            device=data_in.device,
        )
    )
    Qx = fftfreq(nx, data_in.sampling[1], dtype=data_in.dtype, device=data_in.device)
    Qy = fftfreq(ny, data_in.sampling[0], dtype=data_in.dtype, device=data_in.device)

    object_bright_field, object_ssb = _weak_phase_reconstruction(
        G,
        Qx,
        Qy,
        Kx,
        Ky,
        data_in.meta.aberrations.array.to(data_in.device),
        data_in.meta.rotation,
        data_in.meta.semiconvergence_angle,
        data_in.meta.wavelength,
    )

    return object_bright_field, object_ssb


def _weak_phase_reconstruction(
    G: torch.Tensor,
    Qx_all: torch.Tensor,
    Qy_all: torch.Tensor,
    Kx_all: torch.Tensor,
    Ky_all: torch.Tensor,
    aberrations: torch.Tensor,
    rotation: float,
    semiconvergence_angle: float,
    wavelength: float,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    nx, ny, _, _ = G.shape
    object_bright_field = torch.zeros((ny, nx), dtype=G.dtype, device=G.device)
    object_ssb = torch.zeros((ny, nx), dtype=G.dtype, device=G.device)
    device = wp.device_from_torch(G.device)

    G_wp = wp.from_torch(torch.view_as_real(G), dtype=wp.vec2)
    Qx_all_wp = wp.from_torch(Qx_all)
    Qy_all_wp = wp.from_torch(Qy_all)
    Kx_all_wp = wp.from_torch(Kx_all)
    Ky_all_wp = wp.from_torch(Ky_all)
    aberrations_wp = wp.from_torch(aberrations)
    object_bright_field_wp = wp.from_torch(torch.view_as_real(object_bright_field))
    object_ssb_wp = wp.from_torch(torch.view_as_real(object_ssb))
    IQY, IQX, IKY, IKX = G.shape

    # Convert scalar parameters to Warp float32
    rotation_wp = wp.float32(rotation)
    semiconvergence_angle_wp = wp.float32(semiconvergence_angle)
    eps_wp = wp.float32(eps)
    wavelength_wp = wp.float32(wavelength)

    wp.launch(
        kernel=_single_sideband_kernel,
        dim=(IQY, IQX, IKY, IKX),
        inputs=[
            G_wp,
            Qx_all_wp,
            Qy_all_wp,
            Kx_all_wp,
            Ky_all_wp,
            aberrations_wp,
            rotation_wp,
            semiconvergence_angle_wp,
            eps_wp,
            wavelength_wp,
        ],
        outputs=[object_bright_field_wp, object_ssb_wp],
        device=device,
        record_tape=False,
    )

    object_bright_field = torch.fft.ifft2(object_bright_field, norm="ortho")
    object_ssb = torch.fft.ifft2(object_ssb, norm="ortho")

    return object_bright_field, object_ssb


def optimize_aberrations_for_std_maximization(
    data_in, initial_aberrations=None, max_iter=1000
):
    """
    Optimize aberration coefficients to maximize the standard deviation of object_bright_field.

    Args:
        data_in: Dataset4dstem object
        initial_aberrations: Initial aberration coefficients (numpy array)
        max_iter: Maximum number of iterations for optimization

    Returns:
        tuple: (optimized_aberrations, optimization_result)
    """
    if data_in.meta is None:
        raise ValueError("data_in.meta is None")

    G = torch.fft.fft2(data_in.array, dim=(0, 1), norm="ortho")
    ny, nx, nky, nkx = data_in.shape

    Kx = fftshift(
        fftfreq(
            nkx,
            1 / (data_in.sampling[3] * data_in.shape[3]),
            dtype=data_in.dtype,
            device=data_in.device,
        )
    )
    Ky = fftshift(
        fftfreq(
            nky,
            1 / (data_in.sampling[2] * data_in.shape[2]),
            dtype=data_in.dtype,
            device=data_in.device,
        )
    )
    Qx = fftfreq(nx, data_in.sampling[1], dtype=data_in.dtype, device=data_in.device)
    Qy = fftfreq(ny, data_in.sampling[0], dtype=data_in.dtype, device=data_in.device)

    # Set initial aberrations if not provided
    if initial_aberrations is None:
        initial_aberrations = np.zeros(12)
        initial_aberrations[0] = 0  # Set defocus to -50 as in original code

    def objective_function(aberrations):
        """
        Objective function to maximize: negative standard deviation of object_bright_field
        (we minimize the negative to maximize the standard deviation)
        """
        try:
            # Convert numpy array to torch tensor and move to device
            aberrations_tensor = torch.tensor(
                aberrations, dtype=data_in.dtype, device=data_in.device
            )

            # Temporarily update the aberrations in the dataset
            original_aberrations = data_in.meta.aberrations.array.clone()
            data_in.meta.aberrations.array = aberrations_tensor

            object_bright_field, object_ssb = _weak_phase_reconstruction(
                G,
                Qx,
                Qy,
                Kx,
                Ky,
                data_in.meta.aberrations.array.to(data_in.device),
                data_in.meta.rotation,
                data_in.meta.semiconvergence_angle,
                data_in.meta.wavelength,
            )
            # Perform weak phase reconstruction
            object_bright_field, _ = weak_phase_reconstruction(data_in)

            # Calculate standard deviation of the phase (angle) of object_bright_field
            phase_std = torch.std(torch.angle(object_bright_field)).item()

            # Restore original aberrations
            data_in.meta.aberrations.array = original_aberrations

            # Return negative value since we want to maximize standard deviation
            return -phase_std

        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1e6  # Return large value for failed evaluations

    # Run Nelder-Mead optimization
    result = minimize(
        objective_function,
        initial_aberrations,
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-6, "fatol": 1e-6, "adaptive": True},
    )

    return result.x, result
