from math import ceil, floor
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.nn.functional import interpolate


def sort_by_mode_int(modes: torch.Tensor) -> torch.Tensor:
    """Sort modes by their integrated intensity in descending order."""
    modes_int = modes.abs().pow(2).sum(tuple(range(1, modes.ndim)))
    _, indices = torch.sort(modes_int, descending=True)
    modes = modes[indices, ...]
    return modes


def orthogonalize_probe(model: Any, regul_dict: Dict[str, Any]) -> torch.Tensor:
    """Orthogonalize probe modes using SVD decomposition."""
    modes = model.probe_model[str((0, 0))].squeeze(1)
    orig_modes_dtype = modes.dtype
    if orig_modes_dtype != torch.complex64:
        modes = torch.complex(modes, torch.zeros_like(modes))

    input_shape = modes.shape
    modes_reshaped = modes.reshape(input_shape[0], -1)

    A = torch.matmul(modes_reshaped, modes_reshaped.conj().t())
    evals, evecs = torch.linalg.eig(A)

    ortho_modes = torch.matmul(evecs.conj().t(), modes_reshaped).reshape(input_shape)
    ortho_modes = sort_by_mode_int(ortho_modes)
    probe_int = modes.abs().pow(2)
    probe_pow = (
        (probe_int.sum((1, 2)) / probe_int.sum()).detach().cpu().numpy().round(3)
    )
    print(
        f"Apply ortho pmode constraint, relative pmode power = {probe_pow}, probe int sum = {probe_int.sum():.4f}"
    )

    return ortho_modes.to(orig_modes_dtype)[:, None, ...]


def scaled_sigmoid(
    x: torch.Tensor, offset: float = 0, scale: float = 1
) -> torch.Tensor:
    """Apply scaled sigmoid function with adjustable offset and scale."""
    scaled_sigmoid = 1 / (1 + torch.exp((x - offset) / scale * 10))
    return scaled_sigmoid


def make_sigmoid_mask(
    Npix: int, relative_radius: float = 2 / 3, relative_width: float = 0.2
) -> torch.Tensor:
    """Create circular sigmoid mask for filtering."""
    ky = torch.linspace(-floor(Npix / 2), ceil(Npix / 2) - 1, Npix)
    kx = torch.linspace(-floor(Npix / 2), ceil(Npix / 2) - 1, Npix)
    grid_ky, grid_kx = torch.meshgrid(ky, kx, indexing="ij")
    kR = torch.sqrt(grid_ky**2 + grid_kx**2)  # centered already
    sigmoid_mask = scaled_sigmoid(
        kR, offset=Npix / 2 * relative_radius, scale=relative_width * Npix
    )

    return sigmoid_mask


def probe_mask(model: Any, regul_dict: Dict[str, Any]) -> torch.Tensor:
    """Apply Fourier domain amplitude mask to probe."""
    relative_radius = regul_dict["relative_radius"]
    relative_width = regul_dict["relative_width"]
    power_thresh = regul_dict["power_thresh"]

    probe = model.probe_model[str((0, 0))]
    Npix = probe.size(-1)
    powers = probe.abs().pow(2).sum((-2, -1)) / probe.abs().pow(2).sum()
    powers_cumsum = powers.cumsum(0)
    pmode_index = (powers_cumsum > power_thresh).nonzero()[0, 0].item()
    mask = torch.ones_like(probe, dtype=torch.float32, device=model.device)
    mask_value = make_sigmoid_mask(Npix, relative_radius, relative_width).to(
        model.device
    )
    mask[: pmode_index + 1] = mask_value
    probe_k = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(probe, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
    probe_r = torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(mask * probe_k, dim=(-2, -1)), norm="ortho"
        ),
        dim=(-2, -1),
    )
    probe_update = sort_by_mode_int(probe_r)
    return probe_update


def fix_probe_intensity(model: Any, regul_dict: Dict[str, Any]) -> torch.Tensor:
    """Fix probe intensity to match initial value."""
    probe = model.probe_model[str((0, 0))]
    probe_init = model.probe_init
    current_amp = probe.abs().pow(2).sum().pow(0.5)
    target_amp = probe_init.abs().pow(2).sum() ** 0.5
    probe = probe * target_amp / current_amp
    print(
        f"Apply fix probe int constraint, probe int sum = {probe.abs().pow(2).sum():.4f}"
    )
    return probe


def kz_filter(
    obj: torch.Tensor, obj_pass: torch.Tensor, regul_dict: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 3D kz filter using arctan regularization."""
    beta_regularize_layers = regul_dict["beta"]
    alpha_gaussian = regul_dict["alpha"]
    obj_type = regul_dict["obj_type"]
    device = obj.device
    Npix = obj.shape[-3:]
    kz = torch.fft.fftfreq(Npix[0]).to(device)
    ky = torch.fft.fftfreq(Npix[1]).to(device)
    kx = torch.fft.fftfreq(Npix[2]).to(device)

    grid_kz, grid_ky, grid_kx = torch.meshgrid(kz, ky, kx, indexing="ij")

    W = 1 - torch.atan(
        (
            beta_regularize_layers
            * torch.abs(grid_kz)
            / torch.sqrt(grid_kx**2 + grid_ky**2 + 1e-3)
        )
        ** 2
    ) / (torch.pi / 2)
    Wa = W * torch.exp(-alpha_gaussian * (grid_kx**2 + grid_ky**2))

    fobj = torch.real(
        torch.fft.ifftn(
            torch.fft.fftn(obj, dim=(-3, -2, -1)) * Wa[None,], dim=(-3, -2, -1)
        )
    )
    if obj_type == "amplitude":
        fobj = 1 + 0.9 * (fobj - 1)
    return fobj, obj_pass


def complex_ratio(
    amp: torch.Tensor, phase: torch.Tensor, regul_dict: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply complex ratio constraint between amplitude and phase."""
    alpha1 = regul_dict["alpha1"]
    alpha2 = regul_dict["alpha2"]

    log_obja = torch.log(amp)

    Cbar = (log_obja.abs().sum()) / (phase.abs().sum() + 1e-8)

    objac = torch.exp((1 - alpha1) * log_obja - alpha1 * Cbar * phase)

    objpc = (1 - alpha2) * phase - alpha2 / (Cbar + 1e-8) * log_obja
    return objac, objpc, Cbar


def mirrored_amp(
    amp: torch.Tensor, phase: torch.Tensor, regul_dict: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply mirrored amplitude constraint based on phase values."""
    relax = regul_dict["relax"]
    scale = regul_dict["scale"]
    power = regul_dict["power"]
    v_power = phase.clamp(min=0).pow(power)
    amp_new = 1 - scale * v_power
    amp2 = relax * amp + (1 - relax) * amp_new

    amin, amax = amp2.min().item(), amp2.max().item()
    relax_str = f"relaxed ({relax}*obj + ({1-relax}*obj_new))" if relax != 0 else "hard"
    print(
        f"Apply {relax_str} mirrored amplitude constraint with scale = {scale} and power = {power} on obja. obja range becomes ({amin:.3f}, {amax:.3f})"
    )

    return amp2, phase


def phase_positive(
    amp: torch.Tensor, phase: torch.Tensor, regul_dict: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply positivity constraint to phase values."""
    relax = regul_dict["relax"]
    mode = regul_dict["mode"]
    original_min = phase.min()
    if mode == "subtract_min":
        modified_objp = phase - original_min
    else:
        modified_objp = phase.clamp(min=0)
    phase_update = relax * phase + (1 - relax) * modified_objp

    omin, omax = (
        (phase_update - 1j * torch.log(amp)).real.min().item(),
        (phase_update - 1j * torch.log(amp)).real.max().item(),
    )
    print(f"Apply phase positivity. Phase range becomes ({omin:.3f}, {omax:.3f})")
    return amp, phase_update


def kr_filter(
    obj: torch.Tensor, obj_pass: torch.Tensor, regul_dict: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D radial frequency filter using sigmoid mask."""
    radius = regul_dict["radius"]
    width = regul_dict["width"]
    Ny, Nx = obj.shape[-2:]
    mask = make_sigmoid_mask(min(Ny, Nx), radius, width).to(obj.device)
    W = torch.fft.ifftshift(
        interpolate(
            mask[
                None,
                None,
            ],
            size=(Ny, Nx),
        ),
        dim=(-2, -1),
    ).squeeze()
    fobj = torch.real(
        torch.fft.ifft2(
            torch.fft.fft2(obj)
            * W[
                None,
                None,
            ]
        )
    )
    return fobj, obj_pass


def amp_threshold(
    amp: torch.Tensor, phase: torch.Tensor, regul_dict: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply amplitude thresholding constraint."""
    relax = regul_dict["relax"]
    threshold_low, threshold_high = regul_dict["thresh"]
    amp = relax * amp + (1 - relax) * amp.clamp(min=threshold_low, max=threshold_high)
    return amp, phase


def gaussian1d(size: int, std: float) -> torch.Tensor:
    """Generate 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    kernel = torch.exp(-0.5 * (coords / std) ** 2)
    return kernel / kernel.sum()


def get_gaussian3d(
    size_z: int,
    size_y: int,
    size_x: int,
    std_z: float,
    std_y: float,
    std_x: float,
    norm: bool = True,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate 3D Gaussian kernel."""
    # Create 1D Gaussian kernels for each dimension
    k_z = gaussian1d(size_z, std_z)
    k_y = gaussian1d(size_y, std_y)
    k_x = gaussian1d(size_x, std_x)

    # Move to specified device if provided
    if device is not None:
        k_z = k_z.to(device)
        k_y = k_y.to(device)
        k_x = k_x.to(device)

    # Convert to specified dtype
    k_z = k_z.to(dtype)
    k_y = k_y.to(dtype)
    k_x = k_x.to(dtype)

    # Create 3D kernel by outer product
    kernel_3d = torch.outer(k_z, torch.outer(k_y, k_x).flatten()).reshape(
        size_z, size_y, size_x
    )

    if norm:
        kernel_3d = kernel_3d / kernel_3d.sum()

    return kernel_3d


def gaussian_blur_3d(
    tensor: torch.Tensor,
    kernel_size_z: int = 5,
    kernel_size_y: int = 5,
    kernel_size_x: int = 5,
    sigma_z: float = 0.5,
    sigma_y: float = 0.5,
    sigma_x: float = 0.5,
) -> torch.Tensor:
    """Apply 3D Gaussian blur to tensor."""
    dtype = tensor.dtype
    device = tensor.device

    # Generate 3D Gaussian kernel
    kernel_3d = get_gaussian3d(
        kernel_size_z,
        kernel_size_y,
        kernel_size_x,
        sigma_z,
        sigma_y,
        sigma_x,
        norm=True,
        device=device,
        dtype=dtype,
    )

    # Reshape for conv3d
    kernel_3d = kernel_3d.view(1, 1, kernel_size_z, kernel_size_y, kernel_size_x)

    # Reshape input tensor for conv3d: (batch, channels, depth, height, width)
    original_shape = tensor.shape
    batch_size = (
        tensor.numel() // (original_shape[-3] * original_shape[-2] * original_shape[-1])
        if len(original_shape) > 3
        else 1
    )

    # Reshape to (batch_size, 1, D, H, W)
    tensor_reshaped = tensor.view(batch_size, 1, *original_shape[-3:])

    # Apply 3D convolution with replicate padding
    padding_z = kernel_size_z // 2
    padding_y = kernel_size_y // 2
    padding_x = kernel_size_x // 2

    # F.conv3d doesn't support padding_mode, so we pad manually
    tensor_padded = F.pad(
        tensor_reshaped,
        (padding_x, padding_x, padding_y, padding_y, padding_z, padding_z),
        mode="replicate",
    )

    # Apply convolution
    tensor_blur = F.conv3d(tensor_padded, kernel_3d, bias=None)

    # Reshape back to original shape
    tensor_blur = tensor_blur.view(*original_shape)

    return tensor_blur


def obj_3dblur(
    amp: torch.Tensor, phase: torch.Tensor, regul_dict: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 3D Gaussian blur to object amplitude and phase."""
    kernel_size = regul_dict["kernel_size_zyx"]
    sigma = regul_dict["std_zyx"]

    # Apply 3D Gaussian blur to amplitude
    amp = gaussian_blur_3d(
        amp,
        kernel_size_z=kernel_size[0],
        kernel_size_y=kernel_size[1],
        kernel_size_x=kernel_size[2],
        sigma_z=sigma[0],
        sigma_y=sigma[1],
        sigma_x=sigma[2],
    )

    # Apply 3D Gaussian blur to phase
    phase = gaussian_blur_3d(
        phase,
        kernel_size_z=kernel_size[0],
        kernel_size_y=kernel_size[1],
        kernel_size_x=kernel_size[2],
        sigma_z=sigma[0],
        sigma_y=sigma[1],
        sigma_x=sigma[2],
    )

    return amp, phase
