from typing import List, Tuple

import numpy as np
import torch
from ase import units
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import Delaunay
from torch import Tensor


def regularization_kernel_fourierspace(
    kernel_shape: ArrayLike,
    d: List[float],
    beta: float,
    alpha: float,
    device: str = "cpu",
) -> Tensor:
    """
    Calculate multi-slice regularization kernel in Fourier space.

    Parameters
    ----------
    kernel_shape : array-like
        Shape of the kernel volume (z, y, x).
    d : list
        Sampling intervals in each dimension [dz, dy, dx].
    beta : float
        Regularization parameter controlling anisotropy strength along z.
    alpha : float
        Regularization parameter controlling smoothing strength along xy.
    device : str, optional
        PyTorch device to use. Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        Fourier space regularization kernel.
    """
    q = fftfreq3(kernel_shape, d, device=device)
    W = 1 - torch.atan(
        (beta * torch.abs(q[0]) / torch.sqrt(q[1] ** 2 + q[2] ** 2 + 1e-3)) ** 2
    ) / (torch.pi / 2)
    Wa = W * torch.exp(-alpha * (q[1] ** 2 + q[2] ** 2))
    return Wa


def regularization_kernel_realspace(
    volume_shape: ArrayLike,
    d: List[float] = [1.0, 1.0, 1.0],
    beta: float = 1,
    alpha: float = 1,
    thresh: float = 0.03,
    device: str = "cuda:0",
) -> Tensor:
    """
    Calculate multi-slice regularization kernel in real space.

    Parameters
    ----------
    volume_shape : array-like
        Shape of the volume (z, y, x).
    d : list, optional
        Sampling intervals in each dimension [dz, dy, dx]. Default is [1.0, 1.0, 1.0].
    beta : float, optional
        Regularization parameter controlling anisotropy strength along z. Default is 1.
    alpha : float, optional
        Regularization parameter controlling smoothing strength along xy. Default is 1.
    thresh : float, optional
        Threshold for determining kernel size as fraction of maximum value. Default is 0.03.
    device : str, optional
        PyTorch device to use. Default is 'cuda:0'.

    Returns
    -------
    torch.Tensor
        Real space regularization kernel with odd dimensions, normalized to sum to 1.
    """
    volume_shape = np.array([v + (1 - v % 2) for v in volume_shape])
    kernel_fourierspace = regularization_kernel_fourierspace(
        volume_shape, d, beta, alpha, device
    )
    psf = torch.zeros(tuple(volume_shape), device=device)
    psf[0, 0, 0] = 1
    kernel_vol = torch.fft.ifftn(
        kernel_fourierspace * torch.fft.fftn(psf, norm="ortho")
    )

    X = torch.fft.fftshift(kernel_vol.real)

    center = np.fix(volume_shape / 2).astype(int)

    # Create projections by summing along each axis
    proj_xy = torch.sum(X, dim=0).cpu()  # Sum along x axis
    # proj_xz = torch.sum(X, dim=1).cpu()  # Sum along y axis
    proj_yz = torch.sum(X, dim=2).cpu()  # Sum along z axis

    mask_yz = proj_yz > thresh * proj_yz.max()
    # mask_xz = proj_xz > thresh * proj_xz.max()
    mask_xy = proj_xy > thresh * proj_xy.max()
    z_len_proj = mask_yz.to(torch.int32).sum(1)
    y_len_proj = mask_xy.to(torch.int32).sum(1)
    x_len_proj = mask_xy.to(torch.int32).sum(0)

    def get_nonzero_length(proj: np.ndarray) -> int:
        inds = np.arange(len(proj))
        nonzero_inds = inds[proj > 0]
        length = np.max(nonzero_inds) - np.min(nonzero_inds)
        return length + 1

    z_length = get_nonzero_length(z_len_proj)
    y_length = get_nonzero_length(y_len_proj)
    x_length = get_nonzero_length(x_len_proj)

    # Force z_length to be odd by adding 1 if even
    z_length = z_length + (1 - z_length % 2)
    y_length = y_length + (1 - y_length % 2)
    x_length = x_length + (1 - x_length % 2)

    kernel = X[
        center[0] - z_length // 2 : center[0] + z_length // 2 + 1,
        center[1] - y_length // 2 : center[1] + y_length // 2 + 1,
        center[2] - x_length // 2 : center[2] + x_length // 2 + 1,
    ].clone()
    kernel /= kernel.sum()
    return kernel


def circular_aperture(
    r: float, shape: Tuple[int, int], device: torch.device = torch.device("cuda")
) -> Tensor:
    """Create a circular aperture with anti-aliased edge.

    Parameters
    ----------
    r : float
        Radius of aperture in pixels
    shape : tuple
        Shape of output array (height, width)
    device : str
        Device to place tensor on ("cuda" or "cpu")

    Returns
    -------
    torch.Tensor
        2D tensor containing circular aperture with smooth edges
    """
    y = torch.arange(-shape[0] // 2, shape[0] // 2, device=device)
    x = torch.arange(-shape[1] // 2, shape[1] // 2, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    dist = torch.sqrt(X * X + Y * Y)

    # Anti-aliasing width (in pixels)
    aa_width = 1.0

    # Smooth transition using tanh
    mask = 0.5 * (1 - torch.tanh((dist - r) / (aa_width / 2)))
    mask = torch.fft.fftshift(mask)
    return mask


def beamlet_samples(
    beam_mask: ArrayLike,
    radius: float,
    n_angular_samples: int = 6,
    n_radial_samples: int = 3,
) -> np.ndarray:
    """Generate beamlet sample positions."""
    y, x = np.where(beam_mask)
    samples = []
    for i in range(len(y)):
        for r in np.linspace(0, radius, n_radial_samples):
            for theta in np.linspace(0, 2 * np.pi, n_angular_samples, endpoint=False):
                sy = int(y[i] + r * np.sin(theta))
                sx = int(x[i] + r * np.cos(theta))
                if 0 <= sy < beam_mask.shape[0] and 0 <= sx < beam_mask.shape[1]:
                    samples.append([sy, sx])
    return np.array(samples)


def natural_neighbor_weights(
    sample_points: ArrayLike,
    query_points: ArrayLike,
    minimum_weight_cutoff: float = 1e-2,
) -> np.ndarray:
    """Calculate natural neighbor interpolation weights."""
    tri = Delaunay(sample_points)
    weights = np.zeros((len(query_points), len(sample_points)))

    for i, point in enumerate(query_points):
        simplex = tri.find_simplex(point)
        if simplex >= 0:
            b = tri.transform[simplex, :2].dot(point - tri.transform[simplex, 2])
            weights[i, tri.simplices[simplex]] = b

    weights[weights < minimum_weight_cutoff] = 0
    row_sums = weights.sum(axis=1)
    weights[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

    return weights


def fftfreq2(
    N: ArrayLike,
    dx: List[float] = [1.0, 1.0],
    centered: bool = False,
    device: str = "cpu",
) -> Tensor:
    """
    Calculate 2D Fourier frequencies for a given grid size and sampling intervals.

    Parameters
    ----------
    N : array-like
        Grid dimensions (y, x).
    dx : list, optional
        Sampling intervals in each dimension [dy, dx]. Default is [1.0, 1.0].
    centered : bool, optional
        If True, shift frequencies by half a pixel. Default is False.
    device : str, optional
        PyTorch device to use. Default is 'cpu'.

    Returns
    -------
    Tensor
        Stacked frequency coordinates with shape (2, Ny, Nx).
        First dimension contains qy and qx components.
    """
    qxx = torch.fft.fftfreq(N[1], dx[1], device=device)
    qyy = torch.fft.fftfreq(N[0], dx[0], device=device)
    if centered:
        qxx += 0.5 / N[1] / dx[1]
        qyy += 0.5 / N[0] / dx[0]
    qx, qy = torch.meshgrid(qxx, qyy)
    q = torch.stack([qy, qx], dim=0)
    return q


def fftfreq3(
    N: ArrayLike,
    dx: List[float] = [1.0, 1.0, 1.0],
    centered: bool = False,
    device: str = "cpu",
) -> Tensor:
    """
    Calculate 3D Fourier frequencies for a given grid size and sampling intervals.

    Parameters
    ----------
    N : array-like
        Grid dimensions (z, y, x).
    dx : list, optional
        Sampling intervals in each dimension [dz, dy, dx]. Default is [1.0, 1.0, 1.0].
    centered : bool, optional
        If True, shift frequencies by half a pixel. Default is False.
    device : str, optional
        PyTorch device to use. Default is 'cpu'.

    Returns
    -------
    Tensor
        Stacked frequency coordinates with shape (3, Nz, Ny, Nx).
        First dimension contains qz, qy and qx components.
    """
    qxx = torch.fft.fftfreq(N[1], dx[2], device=device)
    qyy = torch.fft.fftfreq(N[1], dx[1], device=device)
    qzz = torch.fft.fftfreq(N[0], dx[0], device=device)
    if centered:
        qxx += 0.5 / N[2] / dx[2]
        qyy += 0.5 / N[1] / dx[1]
        qzz += 0.5 / N[0] / dx[0]
    qz, qy, qx = torch.meshgrid(qzz, qyy, qxx, indexing="ij")
    q = torch.stack([qz, qy, qx], dim=0)
    return q


def relativistic_mass_correction(energy: float) -> float:
    return 1 + units._e * energy / (units._me * units._c**2)


def energy2mass(energy: float) -> float:
    """
    Calculate relativistic mass from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Relativistic mass [kg]̄
    """

    return relativistic_mass_correction(energy) * units._me


def energy2wavelength(energy: float) -> float:
    """
    Calculate relativistic de Broglie wavelength from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Relativistic de Broglie wavelength [Å].
    """

    return (
        units._hplanck
        * units._c
        / np.sqrt(energy * (2 * units._me * units._c**2 / units._e + energy))
        / units._e
        * 1.0e10
    )


def energy2sigma(energy: float) -> float:
    """
    Calculate interaction parameter from energy.

    Parameters
    ----------
    energy: float
        Energy [ev].

    Returns
    -------
    float
        Interaction parameter [1 / (Å * eV)].
    """

    return (
        2
        * np.pi
        * energy2mass(energy)
        * units.kg
        * units._e
        * units.C
        * energy2wavelength(energy)
        / (units._hplanck * units.s * units.J) ** 2
    )


def fftshift_checkerboard(w: int, h: int) -> np.ndarray:
    re = np.r_[w // 2 * [-1, 1]]  # even-numbered rows
    ro = np.r_[w // 2 * [1, -1]]  # odd-numbered rows
    return np.row_stack(h // 2 * (re, ro))


def probe_radius_and_center(
    DP: Tensor, thresh_lower: float = 0.01, thresh_upper: float = 0.99, N: int = 100
) -> Tuple[float, NDArray]:
    """
    Gets the center and radius of the probe in the diffraction plane.

    The algorithm is as follows:
    First, create a series of N binary masks, by thresholding the diffraction pattern DP with a
    linspace of N thresholds from thresh_lower to thresh_upper, measured relative to the maximum
    intensity in DP.
    Using the area of each binary mask, calculate the radius r of a circular probe.
    Because the central disk is typically very intense relative to the rest of the DP, r should
    change very little over a wide range of intermediate values of the threshold. The range in which
    r is trustworthy is found by taking the derivative of r(thresh) and finding identifying where it
    is small.  The radius is taken to be the mean of these r values.
    Using the threshold corresponding to this r, a mask is created and the CoM of the DP times this
    mask it taken.  This is taken to be the origin x0,y0.

    Accepts:
        DP              (2D array) the diffraction pattern in which to find the central disk.
                        A position averaged, or shift-corrected and averaged, DP work well.
        thresh_lower    (float, 0 to 1) the lower limit of threshold values
        thresh_upper    (float, 0 to 1) the upper limit of threshold values
        N               (int) the number of thresholds / masks to use

    Returns:
        r               (float) the central disk radius, in pixels
        x0              (float) the x position of the central disk center
        y0              (float) the y position of the central disk center
    """
    thresh_vals = torch.linspace(thresh_lower, thresh_upper, N, device=DP.device)
    r_vals = torch.zeros(N, device=DP.device)

    DPmax = torch.max(DP)
    for i in range(len(thresh_vals)):
        thresh = thresh_vals[i]
        mask = DP > DPmax * thresh
        r_vals[i] = torch.sqrt(torch.sum(mask) / torch.pi)

    # Get derivative and determine trustworthy r-values
    dr_dtheta = torch.gradient(r_vals, dim=0)[0]
    mask = (dr_dtheta <= 0) * (dr_dtheta >= 2 * torch.median(dr_dtheta))
    r = torch.mean(r_vals[mask])

    # Get origin
    thresh = torch.mean(thresh_vals[mask])
    mask = DP > DPmax * thresh
    ar = DP * mask
    nx, ny = ar.shape
    ry, rx = torch.meshgrid(
        torch.arange(ny, device=DP.device), torch.arange(nx, device=DP.device)
    )
    tot_intens = torch.sum(ar)
    x0 = torch.sum(rx * ar) / tot_intens
    y0 = torch.sum(ry * ar) / tot_intens

    return float(r), np.array([y0.item(), x0.item()])


def get_probe_size(
    DP: ArrayLike, thresh_lower: float = 0.01, thresh_upper: float = 0.99, N: int = 100
) -> Tuple[float, float, float]:
    """
    Gets the center and radius of the probe in the diffraction plane.

    The algorithm is as follows:
    First, create a series of N binary masks, by thresholding the diffraction pattern DP with a
    linspace of N thresholds from thresh_lower to thresh_upper, measured relative to the maximum
    intensity in DP.
    Using the area of each binary mask, calculate the radius r of a circular probe.
    Because the central disk is typically very intense relative to the rest of the DP, r should
    change very little over a wide range of intermediate values of the threshold. The range in which
    r is trustworthy is found by taking the derivative of r(thresh) and finding identifying where it
    is small.  The radius is taken to be the mean of these r values.
    Using the threshold corresponding to this r, a mask is created and the CoM of the DP times this
    mask it taken.  This is taken to be the origin x0,y0.

    Accepts:
        DP              (2D array) the diffraction pattern in which to find the central disk.
                        A position averaged, or shift-corrected and averaged, DP work well.
        thresh_lower    (float, 0 to 1) the lower limit of threshold values
        thresh_upper    (float, 0 to 1) the upper limit of threshold values
        N               (int) the number of thresholds / masks to use

    Returns:
        r               (float) the central disk radius, in pixels
        x0              (float) the x position of the central disk center
        y0              (float) the y position of the central disk center
    """
    thresh_vals = np.linspace(thresh_lower, thresh_upper, N)
    r_vals = np.zeros(N)

    # Get r for each mask
    DPmax = np.max(DP)
    for i in range(len(thresh_vals)):
        thresh = thresh_vals[i]
        mask = DP > DPmax * thresh
        r_vals[i] = np.sqrt(np.sum(mask) / np.pi)

    # Get derivative and determine trustworthy r-values
    dr_dtheta = np.gradient(r_vals)
    mask = (dr_dtheta <= 0) * (dr_dtheta >= 2 * np.median(dr_dtheta))
    r = np.mean(r_vals[mask])

    # Get origin
    thresh = np.mean(thresh_vals[mask])
    mask = DP > DPmax * thresh
    ar = DP * mask
    nx, ny = np.shape(ar)
    ry, rx = np.meshgrid(np.arange(ny), np.arange(nx))
    tot_intens = np.sum(ar)
    x0 = np.sum(rx * ar) / tot_intens
    y0 = np.sum(ry * ar) / tot_intens

    return r, x0, y0


def advanced_raster_scan(
    ny: int = 10,
    nx: int = 10,
    fast_axis: int = 1,
    mirror: List[int] = [1, 1],
    theta: float = 0,
    dy: float = 1,
    dx: float = 1,
    device: torch.device = torch.device("cpu"),
    dtype=torch.float32,
) -> Tensor:
    """
    Generates a raster scan.

    Parameters
    ----------
    ny, nx : int
        Number of steps in *y* (vertical) and *x* (horizontal) direction
    fast_axis : int
        Which axis is the fast axis (1 for x, 0 for y)
    mirror : List[int]
        Mirror factors for x and y axes
    theta : float
        Rotation angle in degrees
    dy, dx : float
        Step size (grid spacing) in *y* and *x*
    device : torch.device
        Device for the output tensor

    Returns
    -------
    pos : Tensor
        A (N,2)-tensor of positions.
    """
    iiy, iix = np.indices((ny, nx), dtype=np.float32)

    if fast_axis != 1:
        iix, iiy = iiy, iix

    # Create positions directly without list comprehension
    positions = np.stack([dy * iiy.ravel(), dx * iix.ravel()], axis=1)

    # Center the positions
    center = positions.mean(axis=0)
    positions -= center

    # Apply mirroring
    positions[:, 0] *= mirror[0]
    positions[:, 1] *= mirror[1]

    # Apply rotation
    if theta != 0:
        theta_rad = np.radians(theta)
        R = np.array(
            [
                [np.cos(theta_rad), -np.sin(theta_rad)],
                [np.sin(theta_rad), np.cos(theta_rad)],
            ]
        )
        positions = positions @ R

    # Translate to start from origin
    positions -= positions.min(axis=0)

    return torch.as_tensor(positions, device=device, dtype=dtype)


def batch_unique_with_inverse(
    positions: Tensor, patch_shape: Tuple[int, int]
) -> Tuple[Tensor, Tensor]:
    """
    :param positions: Each row is ``(y, x)`` for the top-left corner of a patch.
    :type positions: Tensor of shape ``(N, 2)``, dtype=torch.long
    :param patch_shape: The patch dimensions ``(height, width)``.
    :type patch_shape: tuple(int, int)

    :returns:
        - **unique_pts** (*Tensor, shape ``(M, 2)``, dtype=torch.long*) –
          All unique ``(y, x)`` points from all patches.
        - **inverse_indices** (*Tensor, shape ``(N, height, width)``, dtype=torch.long*) –
          Maps each patch pixel to the row index in ``unique_pts``.
    """
    # Unpack the patch shape
    ph, pw = patch_shape

    # If positions is not long, convert it
    if positions.dtype != torch.int32:
        positions = positions.int()

    # 1) Construct a grid of size (patch_height, patch_width) for the patch
    ys = torch.arange(
        ph, dtype=torch.int32, device=positions.device
    )  # [0, 1, ..., ph-1]
    xs = torch.arange(
        pw, dtype=torch.int32, device=positions.device
    )  # [0, 1, ..., pw-1]
    # meshgrid -> shape (ph, pw)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    # stack to get shape (ph, pw, 2)
    patch_grid = torch.stack((grid_y, grid_x), dim=-1)

    # 2) Expand to (N, ph, pw, 2) by adding top-left offsets
    # positions: shape (N,2) -> (N,1,1,2)
    # patch_grid: shape (ph,pw,2) -> (1,ph,pw,2)
    # broadcast sum -> (N, ph, pw, 2)
    all_points = patch_grid.unsqueeze(0) + positions.unsqueeze(1).unsqueeze(1)

    # 3) Flatten to shape (N*ph*pw, 2)
    flat_points = all_points.view(-1, 2)

    # 4) Use torch.unique to get unique points (M,2) and inverse indices
    unique_pts, inverse = torch.unique(flat_points, return_inverse=True, dim=0)

    # 5) Reshape inverse to (N, ph, pw)
    inverse_indices = inverse.view(positions.shape[0], ph, pw)

    return unique_pts, inverse_indices
