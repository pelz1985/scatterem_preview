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
from scatterem2.utils.stem import energy2sigma

# import faulthandler
# faulthandler.enable()
# torch.autograd.set_detect_anomaly(True)
# import warp
# warp.config.verify_fp = True


def test_imports_and_basic_setup() -> None:
    """Quick smoke test to verify imports and basic setup work."""
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"

    # Test that config file exists
    config_path = data_dir / "ssp_config.yaml"
    assert config_path.exists(), f"Config file not found: {config_path}"

    # Test that we can load the config
    with open(config_path) as f:
        params = yaml.safe_load(f)

    # Test that required keys exist
    assert "ptychography" in params, "ptychography section missing from config"
    assert "accelerator" in params, "accelerator missing from config"
    assert "devices" in params, "devices missing from config"

    print("Basic setup test passed")


@pytest.mark.slow
def test_single_slice_ptychography() -> None:
    """Test single slice ptychography training and reconstruction quality."""

    device = torch.device("cpu")
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"
    data_dir.mkdir(exist_ok=True)
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print(current_dir)
    with open(data_dir / "ssp_config.yaml") as f:
        params = yaml.safe_load(f)

    ds = em.load(data_dir / "test_data_single_slice.zip")
    ds.device = device

    # Fix h5py data loading
    with h5py.File(data_dir / "test_single_slice_ground_truth.h5", "r") as f:
        phase_shift_function = torch.as_tensor(f["phase_shift_function"][:])
        # sampling = np.array(f["sampling"][:]).astype(np.float32)
        probe = torch.as_tensor(f["probe"][:])
        positions = torch.as_tensor(f["positions"][:])

    # Option 1: Load from config file
    model = SingleSlicePtychography(
        config_path=data_dir / "ssp_config.yaml",
        ground_truth_object=phase_shift_function,
        ground_truth_probe=probe,
        ground_truth_positions=positions,
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

    trainer.fit(model, train_loader)
    # %%
    # Access and print the last SSIM value from the trainer's logged metrics
    last_ssim = trainer.logged_metrics.get("SSIM-O", None)
    last_pcmse = trainer.logged_metrics.get("PCMSE-P", None)

    last_nrmse_r = trainer.logged_metrics.get("NRMSE-R", None)

    if last_ssim is not None:
        print(f"Last SSIM-O value: {last_ssim:2.2g}")
    else:
        print("No SSIM-O value found in logged metrics")

    if last_pcmse is not None:
        print(f"Last PCMSE-probe value: {last_pcmse:2.2g}")
    else:
        print("No PCMSE-P value found in logged metrics")

    if last_nrmse_r is not None:
        print(f"Last NRMSE-positions value: {last_nrmse_r:2.2g}")
    else:
        print("No NRMSE-R value found in logged metrics")

    # Assert SSIM quality threshold
    assert last_ssim is not None, "SSIM value should be logged during training"
    assert last_ssim > 0.9, f"SSIM is less than 0.91, got {last_ssim}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# %%
