# %%
import os
import pathlib

import h5py
import numpy as np
import torch

import scatterem2.vis as vis
from scatterem2.nn.functional.warp.batch_crop import batch_crop
from scatterem2.nn.modules import ZernikeProbe
from scatterem2.utils.data.abberations import Aberrations
from scatterem2.utils.data.data_classes import Metadata4dstem
from scatterem2.utils.data.datasets import Dataset4dstem
from scatterem2.utils.stem import (
    advanced_raster_scan,
    circular_aperture,
    energy2sigma,
    energy2wavelength,
)

# Get the directory of the current script or use current working directory for interactive mode
try:
    print(f"__file__ = {__file__}")
    print(f"os.getcwd() = {os.getcwd()}")

    # Check if we're in an interactive environment or if __file__ is not a real file
    if __file__ == "<stdin>" or __file__ == "/" or not os.path.isfile(__file__):
        # Running in interactive mode, use current working directory
        script_dir = os.getcwd()
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is not defined in interactive environments
    print(f"os.getcwd() = {os.getcwd()}")
    script_dir = os.getcwd()

print(f"Script directory: {script_dir}")

with h5py.File(script_dir + "/test_potential_2d.h5", "r") as f:
    potential2d = f["data"][:]
    sampling = f["sampling"][:].astype(np.float32)
potential2d = potential2d[:64, :64]
print(f"data.shape = {potential2d.shape}")
print(f"sampling = {sampling}")

E = 80e3
M = 32
skip = 8
wavelength = energy2wavelength(E)
semiangle_cutoff = 50e-3
probe_shape = (M, M)
device = torch.device("cpu")

dr = sampling[0]
k_max = 1 / 2 / dr
alpha_max = k_max * wavelength
dk = k_max / (M / 2)
dalpha = dk * wavelength
print(f"dalpha = {dalpha}")
print(f"alpha_max = {alpha_max}")
print(f"wavelength = {wavelength}")
print(f"semiangle_cutoff = {semiangle_cutoff}")
print(f"dr = {dr}")
print(f"dk = {dk}")
phase_shift_function = torch.as_tensor(energy2sigma(energy=E) * potential2d)
phase_shift_function -= phase_shift_function.min()
potential = torch.exp(1j * phase_shift_function)
fig, ax = vis.show_2d(torch.angle(potential), cbar=True, title="Phase shift function")
r = semiangle_cutoff / dalpha
print(f"r = {r}")

aperture = circular_aperture(r=r, shape=probe_shape, device=device)
vis.show_2d(aperture, cbar=True, title="Aperture")
aberrations_array = torch.zeros(12)
aberrations_array[0] = 15
aberrations = Aberrations(array=aberrations_array)
probe = ZernikeProbe(
    wavelength=wavelength,
    sampling=(dr, dr),
    aperture_array=aperture,
    aberrations=aberrations,
    fourier_space=False,
    make_beamlet_meta=False,
    n_radial_samples=4,
    n_angular_samples=6,
    make_plane_waves=False,
    fft_shifted=True,
)

probe_tensor = probe.forward()
print(f"probe_tensor.shape = {probe_tensor.shape}")
fig, ax = vis.show_2d(probe_tensor[0, 0], cbar=True, title="Probe")


nx = potential.shape[1]  # - probe_tensor.shape[-1]
ny = potential.shape[0]  # - probe_tensor.shape[-2]

nx = nx // skip
ny = ny // skip
print(f"nx = {nx}")
print(f"ny = {ny}")
positions = advanced_raster_scan(
    ny=ny, nx=nx, fast_axis=1, mirror=[1, 1], theta=0, dy=skip, dx=skip, device=device
).int()
print(f"positions.shape = {positions.shape}")
print(f"positions.max() = {positions.max()}")

batches = batch_crop(
    volume=potential.unsqueeze(0), waves=probe_tensor, positions=positions
)

# %%
fig, ax = vis.show_2d(torch.angle(batches[0, 22:34, :, :]), cbar=True, title="Data")
# %%
data = (
    torch.abs(torch.fft.fft2(batches * probe_tensor, dim=(-2, -1), norm="ortho")) ** 2
)
print(f"data.shape = {data.shape}")
fig, ax = vis.show_2d(torch.fft.fftshift(data[0, 0]) ** 0.25, cbar=True, title="Data")

meta = Metadata4dstem(
    energy=E,
    semiconvergence_angle=float(semiangle_cutoff),
    rotation=0,
    defocus_guess=float(-aberrations_array[0].item()),
    sample_thickness_guess=0,
    vacuum_probe=None,
    sampling=(dr * skip, dr * skip, dk, dk),
    units=["A", "A", "A^-1", "A^-1"],
    shape=np.array([ny, nx, M, M], dtype=np.int32),
    aberrations=Aberrations(array=aberrations_array),
)

d = data.detach().reshape(ny, nx, M, M).cpu()
dataset = Dataset4dstem.from_array(
    array=d,
    origin=np.array((ny / 2, nx / 2, M / 2, M / 2), dtype=np.float32),
    name="test_data_single_slice",
    signal_units="arb. units",
    meta=meta,
    transform_to_amplitudes=True,
    device=device,
    normalize=False,
)
dataset.save(pathlib.Path(script_dir) / "test_data_single_slice", mode="o", store="zip")
meta.to_yaml(script_dir + "/data/ssp_config.yaml")

# %%
