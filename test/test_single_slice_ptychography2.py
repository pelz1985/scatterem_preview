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
from scatterem2.lightning import SingleSlicePtychography
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
with open(data_dir / "ssp_config.yaml") as f:
    params = yaml.safe_load(f)

ds = em.load(current_dir / "test_data_single_slice.zip")
ds.device = device

# Fix h5py data loading
with h5py.File(current_dir / "test_potential_2d.h5", "r") as f:
    potential2d = np.array(f["data"][:])
    # sampling = np.array(f["sampling"][:]).astype(np.float32)

potential2d = potential2d[:64, :64]
phase_shift_function = (
    torch.as_tensor(energy2sigma(energy=ds.meta.energy) * potential2d)
    .unsqueeze(0)
    .unsqueeze(0)
)
phase_shift_function -= phase_shift_function.min()
# ground_truth_object = torch.exp(1j * phase_shift_function)

print(ds)
print(ds.meta)

# Option 1: Load from config file
model = SingleSlicePtychography(
    config_path=data_dir / "ssp_config.yaml",
    object_model="pixels",
    do_subpix_shift=False,
    position_correction_start_epoch=params["ptychography"][
        "position_correction_start_epoch"
    ],  # Use int instead of float
    ground_truth_object=phase_shift_function,
    learning_rate=params["ptychography"]["learning_rate"],
    batch_size=params["ptychography"]["batch_size"],
    loss_function=torch.nn.functional.mse_loss,
    device=device,
)

train_loader = DataLoader(
    ds,
    sampler=BatchSampler(
        RandomSampler(range(len(ds))),
        batch_size=params["ptychography"]["batch_size"],
        drop_last=False,
    ),
    batch_sampler=None,
    batch_size=None,
    pin_memory=(str(ds.device) == "cpu"),
    num_workers=0,
)

trainer = Trainer(
    default_root_dir=results_dir,
    accelerator=params["accelerator"],
    devices=params["devices"],
    max_epochs=10,  # params["ptychography"]["max_epochs"],
    logger=None,  # Pass single logger
    enable_checkpointing=False,  # Disable checkpoint saving
    enable_progress_bar=False,  # Disable progress bar
    log_every_n_steps=0,  # Disable step logging
    # fast_dev_run=True
    # log_every_n_steps=50,
)
model.forward_model.requires_grad_(
    True
)  # Use requires_grad_ method instead of assignment

trainer.fit(model, train_loader)
# %%
# Access and print the last SSIM value from the trainer's logged metrics
last_ssim = trainer.logged_metrics.get("SSIM", None)
if last_ssim is not None:
    print(f"Last SSIM value: {last_ssim}")
else:
    print("No SSIM value found in logged metrics")

# Assert SSIM quality threshold
assert last_ssim is not None, "SSIM value should be logged during training"
assert last_ssim > 0.91, f"SSIM is less than 0.91, got {last_ssim}"


# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])
# %%
