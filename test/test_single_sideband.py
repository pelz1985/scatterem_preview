# %%
import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch as th
import yaml
from lightning.pytorch import Trainer
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from skimage.feature import peak_local_max
from skimage.filters import gaussian, window
from torch.fft import fftfreq, fftshift
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler

import scatterem2 as em
import scatterem2.vis as vis
from scatterem2.models import SingleSlicePtychography
from scatterem2.models.diffractive_imaging.direct_ptychography import (
    _weak_phase_reconstruction,
    disk_overlap_function,
    optimize_aberrations_for_std_maximization,
    weak_phase_reconstruction,
)
from scatterem2.utils.grid import polar_spatial_frequencies
from scatterem2.utils.stem import energy2sigma

# %%
# import faulthandler
# faulthandler.enable()
# torch.autograd.set_detect_anomaly(True)
# import warp
# warp.config.verify_fp = True


device = torch.device("cuda")
current_dir = Path(__file__).parent
data_dir = current_dir / "data"
data_dir.mkdir(exist_ok=True)
results_dir = current_dir / "results"
results_dir.mkdir(exist_ok=True)

print(current_dir)
with open(data_dir / "ssp_config_ssb.yaml") as f:
    params = yaml.safe_load(f)

dataset = em.load(data_dir / "test_data_ssb.zip")
dataset.device = device


fig, ax = vis.show_2d(dataset.array[0, 0])

data_tapered = dataset.crop_brightfield_pad_and_taper()
print(data_tapered)


print(f"dataset.k_max = {dataset.k_max}")
print(f"dataset.dr = {dataset.dr}")
print(f"dataset.sampling = {dataset.sampling}")
print()
print(f"data_tapered.k_max = {data_tapered.k_max}")
print(f"data_tapered.dr = {data_tapered.dr}")
print(f"data_tapered.sampling = {data_tapered.sampling}")
# %%

alpha, phi = polar_spatial_frequencies(
    data_tapered.shape[-2:], 1 / (data_tapered.sampling[-2:] * data_tapered.shape[-2:])
)
alpha *= dataset.meta.wavelength

vis.show_2d((alpha < 15e-3).float(), cbar=True)
vis.show_2d(alpha, cbar=True)
vis.show_2d(phi, cbar=True)

# %%


data_in = data_tapered
G = th.fft.fft2(data_in.array, dim=(0, 1), norm="ortho")
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

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from scatterem.util.ssb2 import imsave, mosaic, get_qx_qy_1D
# plotmosaic(G_max.abs().cpu().numpy())
# plotmosaic(G_max.angle().cpu().numpy(), cmap=mpl.colormaps['hsv'])

G = th.fft.fft2(data_tapered.array, dim=(0, 1), norm="ortho")

manual_frequencies = None  # [[20, 62, 490], [454, 12, 57]]

Gabs = gaussian(
    th.fft.fftshift(th.log10(th.sum(th.abs(G), (2, 3)) + 1)).cpu().numpy(), 1
)
strongest_object_frequencies = np.array(
    peak_local_max(
        Gabs,
        min_distance=1,
        threshold_abs=-3,
        threshold_rel=None,
        exclude_border=False,
        num_peaks=10,
    )[1:]
)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
imax = ax.imshow(Gabs, cmap=mpl.colormaps["inferno"])
ax.set_title("Masked absolute values of G")
ax.plot(strongest_object_frequencies[:, 1], strongest_object_frequencies[:, 0], ".")
plt.colorbar(imax)
plt.show()
strongest_object_frequencies += np.array(Gabs.shape) // 2
strongest_object_frequencies = strongest_object_frequencies % np.array(Gabs.shape)
G_max = G[strongest_object_frequencies[:, 0], strongest_object_frequencies[:, 1]]
print(strongest_object_frequencies)

aberrations = th.zeros((12), device=data_in.device)
aberrations[0] = -50
best_angle = 0.0

print(f"scan_step = {data_in.sampling[0]}")

print(f"nx = {nx}")
print(f"ny = {ny}")
print(f"nkx = {nkx}")
print(f"nky = {nky}")

# %%
dataset.meta.aberrations.array

# %%
object_bright_field, object_ssb = weak_phase_reconstruction(data_in)
fig_bf, ax_bf = vis.show_2d(
    th.angle(object_bright_field), cbar=True, title="Phase, BF reconstruction"
)
fig_ssb, ax_ssb = vis.show_2d(
    th.angle(object_ssb), cbar=True, title="Phase, SSB reconstruction"
)
# %%
# from scatterem2.utils.data.datasets import Dataset4dstem
# from scatterem2.models.diffractive_imaging.direct_ptychography import _weak_phase_reconstruction
# def weak_phase_reconstruction(
#     data_in: Dataset4dstem
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     if data_in.meta is None:
#         raise ValueError("data_in.meta is None")
#     if data_in.meta.aberrations is None or data_in.meta.aberrations.array is None:
#         raise ValueError("data_in.meta.aberrations or data_in.meta.aberrations.array is None")

#     eps: float = 1e-3
#     G = torch.fft.fft2(data_in.array, dim=(0, 1), norm="ortho")
#     ny, nx, nky, nkx = data_in.shape

#     Kx = fftshift(fftfreq(nkx, 1/(data_in.sampling[3]*data_in.shape[3]), dtype=data_in.dtype, device=data_in.device))
#     Ky = fftshift(fftfreq(nky, 1/(data_in.sampling[2]*data_in.shape[2]), dtype=data_in.dtype, device=data_in.device))
#     Qx = fftfreq(nx, data_in.sampling[1], dtype=data_in.dtype, device=data_in.device)
#     Qy = fftfreq(ny, data_in.sampling[0], dtype=data_in.dtype, device=data_in.device)

#     object_bright_field, object_ssb = _weak_phase_reconstruction(
#         G,
#         Qx,
#         Qy,
#         Kx,
#         Ky,
#         data_in.meta.aberrations.array.to(device),
#         data_in.meta.rotation,
#         data_in.meta.semiconvergence_angle,
#         data_in.meta.wavelength,
#     )

#     return object_bright_field, object_ssb


# %%


# Example usage and optimization
# %%
print("Starting aberration optimization...")
optimized_aberrations, opt_result = optimize_aberrations_for_std_maximization(
    data_in, max_iter=1200
)

print(f"Optimization successful: {opt_result.success}")
print(f"Number of iterations: {opt_result.nit}")
print(
    f"Final function value: {-opt_result.fun}"
)  # Negative because we minimized negative std
print(f"Optimized aberration coefficients: {optimized_aberrations}")

# Apply optimized aberrations and compare results
# %%
# Original reconstruction
original_aberrations = data_in.meta.aberrations.array.clone()
object_bright_field_orig, object_ssb_orig = weak_phase_reconstruction(data_in)
std_orig = torch.std(torch.angle(object_bright_field_orig)).item()

# Optimized reconstruction
data_in.meta.aberrations.array = torch.tensor(
    optimized_aberrations, dtype=data_in.dtype, device=device
)
object_bright_field_opt, object_ssb_opt = weak_phase_reconstruction(data_in)
std_opt = torch.std(torch.angle(object_bright_field_opt)).item()

print(f"Original phase std: {std_orig:.6f}")
print(f"Optimized phase std: {std_opt:.6f}")
print(f"Improvement: {(std_opt - std_orig) / std_orig * 100:.2f}%")

# Visualize results
# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Restore original aberrations for comparison
data_in.meta.aberrations.array = original_aberrations
object_bright_field_orig, object_ssb_orig = weak_phase_reconstruction(data_in)

# Apply optimized aberrations
data_in.meta.aberrations.array = torch.tensor(
    optimized_aberrations, dtype=data_in.dtype, device=device
)
object_bright_field_opt, object_ssb_opt = weak_phase_reconstruction(data_in)

# Plot original
axes[0, 0].imshow(torch.angle(object_bright_field_orig).cpu().numpy(), cmap="viridis")
axes[0, 0].set_title(f"Original Phase (std: {std_orig:.6f})")
axes[0, 0].axis("off")

axes[0, 1].imshow(torch.angle(object_ssb_orig).cpu().numpy(), cmap="viridis")
axes[0, 1].set_title("Original SSB Phase")
axes[0, 1].axis("off")

# Plot optimized
axes[1, 0].imshow(torch.angle(object_bright_field_opt).cpu().numpy(), cmap="viridis")
axes[1, 0].set_title(f"Optimized Phase (std: {std_opt:.6f})")
axes[1, 0].axis("off")

axes[1, 1].imshow(torch.angle(object_ssb_opt).cpu().numpy(), cmap="viridis")
axes[1, 1].set_title("Optimized SSB Phase")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()

# %%
