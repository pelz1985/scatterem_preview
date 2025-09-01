# %%
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler

import scatterem2 as em
from scatterem2.models import SingleSlicePtychography
from scatterem2.utils.data.abberations import Aberrations
from scatterem2.utils.data.data_classes import Metadata4dstem
from scatterem2.utils.data.datasets import Dataset4dstem
from scatterem2.utils.stem import energy2sigma

# import faulthandler
# faulthandler.enable()
# torch.autograd.set_detect_anomaly(True)
# import warp
# warp.config.verify_fp = True

device = torch.device("cpu")
current_dir = Path(__file__).parent
data_dir = current_dir / "data"
data_dir.mkdir(exist_ok=True)
results_dir = current_dir / "results"
results_dir.mkdir(exist_ok=True)

print(current_dir)
with open(data_dir / "ssp_config_subpix.yaml") as f:
    params = yaml.safe_load(f)

ds = em.load(data_dir / "test_data_single_slice.zip")
ds.device = device

# Fix h5py data loading
with h5py.File(data_dir / "test_single_slice_ground_truth.h5", "r") as f:
    phase_shift_function = torch.as_tensor(f["phase_shift_function"][:])
    sampling = np.array(f["sampling"][:]).astype(np.float32)
    probe = torch.as_tensor(f["probe"][:])
    positions = torch.as_tensor(f["positions"][:])

# ground_truth_object = torch.exp(1j * phase_shift_function)

print(ds)
print(ds.meta)

# Option 1: Load from config file
model = SingleSlicePtychography(
    config_path=data_dir / "ssp_config_subpix.yaml",
    ground_truth_object=phase_shift_function,
    ground_truth_probe=probe,
    ground_truth_positions=positions,
)
# %%
model.forward_model.object.requires_grad = False
model.forward_model.object[0, 0].real = phase_shift_function
model.forward_model.object.requires_grad = True
# %%
model.forward_model.dr[:] = torch.randn_like(model.forward_model.dr) * 0.1

# %%
measurements = model.forward(0, 0, torch.arange(0, model.forward_model.dr.shape[0]), 0)
measurements.shape
# %%
model.forward_model.probe[0].shape
# %%
fig, ax = em.vis.show_2d(measurements[0], cbar=True, title="Measurements")

# %%
d = measurements.detach().reshape(tuple(model.meta4d.shape)).cpu()
fig, ax = em.vis.show_2d(d[:, :, 0, 0], cbar=True, title="Measurements")
# %%

dataset = Dataset4dstem.from_array(
    array=d,
    origin=model.meta4d.shape / 2,
    name="test_data_single_slice",
    signal_units="arb. units",
    meta=model.meta4d,
    transform_to_amplitudes=False,
    device=device,
    normalize=False,
)
dataset.save(current_dir / "data/test_data_single_slice_full", mode="o", store="zip")
model.meta4d.to_yaml(current_dir / "data/ssp_config_subpix.yaml")


with h5py.File(current_dir / "data/test_single_slice_ground_truth_full.h5", "w") as f:
    f.create_dataset("phase_shift_function", data=phase_shift_function.cpu().numpy())
    f.create_dataset("sampling", data=model.forward_model.meta4d.sampling)
    f.create_dataset("positions", data=model.forward_model.positions.cpu().numpy())
    f.create_dataset("dr", data=model.forward_model.dr.cpu().numpy())
    f.create_dataset("probe", data=model.forward_model.probe[0].cpu().numpy())


# %%
