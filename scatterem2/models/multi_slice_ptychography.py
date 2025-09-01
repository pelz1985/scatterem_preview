from importlib.util import find_spec
from math import ceil, floor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import lightning as L
import numpy as np
import torch
import yaml
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn.functional import interpolate
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from scatterem2.nn import MultiSlicePtychographyModel
from scatterem2.nn.functional import amplitude_loss
from scatterem2.utils.constraints import (
    amp_threshold,
    complex_ratio,
    fix_probe_intensity,
    kr_filter,
    kz_filter,
    mirrored_amp,
    obj_3dblur,
    orthogonalize_probe,
    phase_positive,
    probe_mask,
)
from scatterem2.utils.data import (
    Metadata4D,
    RasterScanningDiffractionDataset,
)
from scatterem2.utils.data.data_classes import Metadata4dstem


class MultiSlicePtychography(L.LightningModule):
    def __init__(
        self,
        meta4d: Optional[Metadata4dstem] = None,
        config_path: Optional[Union[str, Path]] = None,
        kernel: Optional[Tensor] = None,
        ground_truth_object: Optional[Tensor] = None,
        object_model: Optional[str] = "pixels",
        device: Optional[str] = None,
        devices: Optional[int] = None,
        accelerator: Optional[str] = None,
        do_subpix_shift: Optional[bool] = None,
        position_correction_start_epoch: Optional[int] = 2,
        learning_rate: Optional[float] = 5e-4,
        object_learning_rate: Optional[float] = None,
        probe_learning_rate: Optional[float] = None,
        dr_learning_rate: Optional[float] = None,
        optimize_probe_start: Optional[int] = None,
        optimize_dr_start: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_epochs: Optional[int] = None,
        fix_probe_norm: Optional[bool] = True,
        mixed_probe: Optional[bool] = True,
        pmodes: Optional[int] = None,
        pmode_init_pows: Optional[List] = None,
        loss_function: Optional[Any] = None,
        measurement_norm: Optional[float] = None,
        regularization: Optional[Dict] = None,
        vacuum_probe: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        """
        measurement_norm: Sum of average diffraction pattern normalized by the maximum average diffraction pattern (dp.mean(0).sum() / dp.mean(0).max())
        """
        super().__init__()
        self.save_hyperparameters()
        self.ground_truth_object = ground_truth_object

        if config_path is not None:
            config = self._load_config(config_path)
            # Update config with any explicitly provided kwargs
            config["ptychography"].update(kwargs)
            # Use config values as defaults, but allow explicit parameters to override
            self._setup_from_config(
                config,
                meta4d,
                object_model,
                do_subpix_shift,
                position_correction_start_epoch,
                ground_truth_object,
                device,
                devices,
                accelerator,
                learning_rate,
                object_learning_rate,
                probe_learning_rate,
                dr_learning_rate,
                optimize_probe_start,
                optimize_dr_start,
                batch_size,
                max_epochs,
                loss_function,
                fix_probe_norm,
                mixed_probe,
                pmodes,
                pmode_init_pows,
                regularization,
            )
        if loss_function is None:
            self.loss_function = torch.nn.MSELoss(reduction="mean")
        self.num_batches = ceil(
            self.meta4d.shape[0] * self.meta4d.shape[0] / self.batch_size
        )
        self.model = MultiSlicePtychographyModel(
            self.meta4d,
            object_model,
            self.target_device,
            self.do_subpix_shift,
            do_position_correction=False,
            kernel=kernel,
            mixed_probe=self.mixed_probe,
            pmodes=self.pmodes,
            pmode_init_pows=self.pmode_init_pows,
            measurement_norm=measurement_norm,
            vacuum_probe=vacuum_probe,
        )
        self.model.probe_init = self.model.probe_model[str((0, 0))].clone().detach()
        self._configure_gradient_requirements()
        self.to(self.target_device)

    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "ptychography" not in config:
            raise KeyError("'ptychography' key not found in config file")
        # config = config["ptychography"]
        return config

    def _setup_from_config(
        self,
        config: Dict[str, Any],
        meta4d: Optional[Metadata4dstem],
        object_model: Optional[str],
        do_subpix_shift: Optional[bool],
        position_correction_start_epoch: Optional[int],
        ground_truth_object: Optional[Tensor],
        device: Optional[torch.device],
        devices: Optional[int],
        accelerator: Optional[torch.device],
        learning_rate: Optional[float],
        object_learning_rate: Optional[float],
        probe_learning_rate: Optional[float],
        dr_learning_rate: Optional[float],
        optimize_probe_start: Optional[int],
        optimize_dr_start: Optional[int],
        batch_size: Optional[int],
        max_epochs: Optional[int],
        loss_function: Optional[Any],
        fix_probe_norm: Optional[bool],
        mixed_probe: Optional[bool],
        pmodes: Optional[int],
        pmode_init_pows: Optional[List[float]],
        regularization: Optional[dict],
    ) -> None:
        """Setup parameters from config file with optional overrides."""

        # Extract meta4d from config if not provided directly
        if meta4d is None:
            if "meta4d" not in config:
                raise ValueError("meta4d section not found in config file")
            meta4d_config = config["meta4d"]
            meta4d = Metadata4dstem.from_dict(meta4d_config)

        # Set parameters with config defaults and explicit overrides
        self.meta4d = meta4d
        self.object_model = object_model or config.get("object_model", "pixels")
        self.do_subpix_shift = (
            do_subpix_shift
            if do_subpix_shift is not None
            else config.get("do_subpix_shift", True)
        )
        self.position_correction_start_epoch = (
            position_correction_start_epoch
            or config.get("position_correction_start_epoch", 2)
        )
        device = device or config.get("device", torch.device("cpu"))
        self.devices = devices or config.get("devices", 1)
        self.target_device = device
        self.accelerator = accelerator or config.get("accelerator", torch.device("cpu"))
        self.slice_thickness = config.get("slice_thickness", 1.0)

        # Move ground truth object to device if provided
        if ground_truth_object is not None:
            self.ground_truth_object = ground_truth_object.to(device)
        else:
            self.ground_truth_object = None

        # Get optimizer parameters from config
        optimizer_config = config.get("ptychography", {})
        self.learning_rate = learning_rate or optimizer_config.get(
            "learning_rate", 10e-3
        )
        self.object_learning_rate = object_learning_rate or optimizer_config.get(
            "object_learning_rate", 10e-3
        )
        self.probe_learning_rate = probe_learning_rate or optimizer_config.get(
            "probe_learning_rate", 10e-3
        )
        self.dr_learning_rate = dr_learning_rate or optimizer_config.get(
            "dr_learning_rate", 1
        )
        self.batch_size = batch_size or optimizer_config.get("batch_size", 128)
        self.loss_function = loss_function or torch.nn.functional.mse_loss
        self.optimize_probe_start = optimize_probe_start or optimizer_config.get(
            "optimize_probe_start", 1
        )
        self.optimize_dr_start = optimize_dr_start or optimizer_config.get(
            "optimize_dr_start", 1
        )
        self.fix_probe_norm = fix_probe_norm or optimizer_config.get(
            "fix_probe_norm", True
        )
        self.max_epochs = max_epochs or optimizer_config.get("max_epochs", True)

        #
        # Reading the path to the ground_truth_object from the yaml file
        # instead of loading it in the main script?
        #
        # self.ground_truth_object = ground_truth_object or optimizer_config.get(
        #     "ground_truth_object", None
        # )
        self.mixed_probe = mixed_probe or optimizer_config.get("mixed_probe", None)
        self.pmodes = pmodes or optimizer_config.get("pmodes", None)
        self.pmode_init_pows = pmode_init_pows or optimizer_config.get(
            "pmode_init_pows", None
        )
        regularization_config = regularization or config.get("regularization", {})
        self.regularization = regularization_config

    def forward(
        self,
        probe_index: int,
        angles_index: int,
        r_indices: Tensor,
        translation_index: int,
    ) -> Tensor:
        measurements = self.model(
            probe_index, angles_index, r_indices, translation_index
        )

        return measurements

    def complete_forward(
        self, batch_size: int = 256
    ) -> RasterScanningDiffractionDataset:
        probe_index = 0

        r_indices = np.arange(np.prod(self.model.meta4d.num_scan_steps))
        sampler = BatchSampler(
            SequentialSampler(r_indices), batch_size=batch_size, drop_last=False
        )
        measurements_list: List[Tensor] = []
        translation_index = 0
        for batch_indices in sampler:
            batch_r_indices = torch.as_tensor(batch_indices, device=self.device)
            measurements = self.forward(
                probe_index, int(0), batch_r_indices, translation_index
            )
            measurements_list.append(measurements)
        measurements = torch.cat(measurements_list, dim=0)
        ds = RasterScanningDiffractionDataset(
            measurements,
            convert_to_amplitudes=False,
            probe_index=probe_index,
            angles_index=0,
            translation_index=translation_index,
            device=self.device,
        )

        return ds

    def get_probe_modes(self) -> Dict[str, Tensor]:
        """Get current probe modes"""
        return {key: param.data for key, param in self.model.probe_model.items()}

    def set_probe_requires_grad(self, requires_grad: bool) -> None:
        """Enable/disable gradient computation for probe parameters"""
        for param in self.model.probe_model.parameters():
            param.requires_grad = requires_grad

    def on_after_training_step(
        self, *args: Dict[str, Any], **kwargs: Dict[str, Any]
    ) -> None:
        """Hook called after training step to perform any necessary updates"""
        # if self.model.do_position_correction:
        #     self.model.step_positions()

    def on_train_epoch_start(self) -> None:
        """Hook called at the start of each training epoch"""
        # Update gradient requirements based on current epoch
        if self.current_epoch >= self.hparams.position_correction_start_epoch:
            self.model.do_position_correction = True
        self._update_gradient_requirements()

    def should_apply(self, constraint_name: str) -> bool:
        """Check if constraint should be applied"""
        config = self.regularization[constraint_name]
        return (
            config["freq"] is not None
            and self.current_epoch >= config["start_epoch"]
            and self.global_step % config["freq"] == 0
        )

    def apply_obj_constraint(
        self, constraint_func: Callable[[], None], constraint_name: str
    ) -> None:
        if self.should_apply(constraint_name):
            amp, phase = constraint_func(
                torch.exp(-self.model.object.data.imag),
                self.model.object.data.real,
                self.regularization[constraint_name],
            )
            object_update = phase - 1j * torch.log(amp)
            self.model.object.data.copy_(object_update)

    def apply_probe_constraint(
        self, constraint_func: Callable[[], Any], constraint_name: str
    ) -> None:
        if self.should_apply(constraint_name):
            probe_update = constraint_func(
                self.model,
                self.regularization[constraint_name],
            )
            self.model.probe_model[str((0, 0))].copy_(probe_update)

    def on_after_backward(self) -> None:
        """Apply constraints with regularization config"""
        if self.trainer.is_last_batch:
            with torch.no_grad():
                for constraint in self.regularization.keys():
                    if "probe" in constraint:
                        self.apply_probe_constraint(globals()[constraint], constraint)
                    else:
                        self.apply_obj_constraint(globals()[constraint], constraint)

    def disable_constraint(self, name: str):
        """Disable a constraint at runtime"""
        if name in self.regularization:
            self.regularization[name]["enabled"] = False

    def enable_constraint(self, name: str):
        """Enable a constraint at runtime"""
        if name in self.regularization:
            self.regularization[name]["enabled"] = True

    def set_constraint_frequency(self, name: str, frequency: int):
        """Change constraint frequency at runtime"""
        if name in self.regularization:
            self.regularization[name]["freq"] = frequency

    def _apply_object_gradient_filtering(self, kernel: torch.Tensor) -> None:
        """Apply 3D filtering to object gradients for regularization"""
        pass

    def training_step(
        self, batch: tuple[int, int, int, Tensor, int, Tensor], batch_idx: int
    ) -> Tensor:
        ind, probe_index, angles_index, r_indices, translation_index, data = batch
        self.r_indices = r_indices  # Store for later use
        measurements = self.forward(
            probe_index, int(angles_index), r_indices, int(translation_index)
        )
        loss = self.loss_function(measurements, data) ** 0.5 / data.mean()
        # loss_per_pattern = amplitude_loss(measurements, data)
        # loss = torch.sum(loss_per_pattern) / loss_per_pattern.shape[0]
        loss_value = loss.detach().item()

        # Exitwave gradients
        # opt = self.optimizers()
        # self.model.exitwave.retain_grad()
        # self.manual_backward(loss)

        # import scatterem2.vis as vis
        # from scatterem2.vis.visualization import show_2d_array
        # import matplotlib.pyplot as plt

        # ind = 2
        # wav = torch.fft.fftshift(self.model.exitwave.grad[ind, 0])
        # wav_abs = torch.log10(torch.abs(wav)).cpu().numpy()
        # wav_ang = torch.angle(wav).cpu().numpy()
        # # wav_plot = torch.polar(torch.log10(wav_abs), wav_ang)
        # show_2d_array([wav_abs, wav_ang], cbar=True, axsize=(14, 7))
        # plt.savefig(f"wave_{ind:02d}.png")

        # opt.step()
        # opt.zero_grad()

        # Log training metrics
        self.log(
            "L",
            loss_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )

        if self.ground_truth_object is not None:
            pass
            # Calculate PSNR and SSIM
            # m = 16
            # os = self.ground_truth_object.shape
            # current_object = self.model.object.data.detach().clone()
            # current_object = current_object[:,:os[1],:os[2]]

            # # Calculate PSNR and SSIM in 3D using kornia metrics
            # # Reshape tensors to [B,C,D,H,W] format required by kornia
            # gt = self.ground_truth_object.real.unsqueeze(0).unsqueeze(0)
            # pred = current_object.real.unsqueeze(0).unsqueeze(0)

            # # Normalize to [0,1] range for metrics
            # gt = (gt - gt.min()) / (gt.max() - gt.min())
            # pred = (pred - pred.min()) / (pred.max() - pred.min())

            # # Calculate 3D PSNR
            # psnr_val = psnr(pred, gt, max_val=1.0).item()
            # # Calculate 3D SSIM
            # ssim_val = ssim3d(pred, gt, window_size=11, max_val=1.0).mean().item()

            # # Log metrics
            # self.log('PSNR', psnr_val, prog_bar=True)
            # self.log('SSIM', ssim_val, prog_bar=True)

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        # Create parameter groups with different learning rates
        param_groups = []

        # Group 1: Object parameters
        object_params = []
        if hasattr(self.model, "object"):
            if isinstance(self.model.object, torch.nn.Parameter):
                object_params.append(self.model.object)
            elif hasattr(self.model.object, "parameters"):
                object_params.extend(self.model.object.parameters())

        if object_params:
            param_groups.append(
                {
                    "params": object_params,
                    "lr": self.object_learning_rate,
                    "name": "object",
                }
            )

        # Group 2: Probe parameters (include all, even if not trainable initially)
        probe_params = []
        if hasattr(self.model, "probe_model"):
            probe_params.append(self.model.probe_model[str((0, 0))])

        if probe_params:
            param_groups.append(
                {
                    "params": probe_params,
                    "lr": self.probe_learning_rate,
                    "name": "probe_model",
                }
            )

        # Group 3: DR parameters (include even if not trainable initially)
        dr_params = []
        if hasattr(self.model, "dr"):
            dr_params.append(self.model.dr)

        if dr_params:
            param_groups.append(
                {"params": dr_params, "lr": self.dr_learning_rate, "name": "dr"}
            )

        # Group 4: All other parameters (if any)
        other_params = []
        for name, param in self.named_parameters():
            # Skip parameters that are already in other groups
            is_in_group = False
            for group in param_groups:
                if param.shape == group["params"][0].shape:
                    is_in_group = True
                    break
            if not is_in_group:
                other_params.append(param)

        if other_params:
            param_groups.append(
                {"params": other_params, "lr": self.learning_rate, "name": "other"}
            )

        # If no parameters found, use default behavior
        if not param_groups:
            param_groups = [{"params": self.parameters(), "lr": self.learning_rate}]
        # optimizer = torch.optim.SGD(param_groups)
        optimizer = torch.optim.Adam(param_groups)
        # from scatterem2.optim import LevenbergEveryStep
        # optimizer = LevenbergEveryStep(param_groups, lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.2,
            threshold=1e-9,
            # verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "L",  # monitor the loss metric
                "frequency": 1,
                "interval": "epoch",  # "batch"
            },
        }

    def _configure_gradient_requirements(self) -> None:
        """Configure which parameters should have gradients computed."""
        # Configure probe gradients - start with False, will be updated during training
        if hasattr(self.model, "probe_model") and hasattr(self, "probe_learning_rate"):
            self.model.probe_model[str((0, 0))].requires_grad = False

        # Configure dr gradients - start with False, will be updated during training
        if hasattr(self.model, "dr") and hasattr(self, "dr_learning_rate"):
            self.model.dr.requires_grad = False

    def _update_gradient_requirements(self) -> None:
        """Update gradient requirements based on current epoch."""
        # Configure probe gradients
        if hasattr(self.model, "probe_model"):
            should_optimize_probe = (
                self.optimize_probe_start is not None
                and hasattr(self, "current_epoch")
                and self.current_epoch >= self.optimize_probe_start
            )
            self.model.probe_model[str((0, 0))].requires_grad = should_optimize_probe

        # Configure dr gradients
        if hasattr(self.model, "dr"):
            should_optimize_dr = (
                self.optimize_dr_start is not None
                and hasattr(self, "current_epoch")
                and self.current_epoch >= self.optimize_dr_start
            )
            self.model.dr.requires_grad = should_optimize_dr
            self.model.do_position_correction = should_optimize_dr

        # Update optimizer if new parameters became trainable
        self._update_optimizer_if_needed()

    def _update_optimizer_if_needed(self) -> None:
        """Log when new parameters become trainable."""
        if not hasattr(self, "trainer") or self.trainer is None:
            return

        # Check if probe or dr parameters just became trainable
        probe_trainable = hasattr(self.model, "probe_model") and any(
            probe.requires_grad for probe in self.model.probe_model[str((0, 0))]
        )
        dr_trainable = hasattr(self.model, "dr") and self.model.dr.requires_grad

        # Log when parameters become trainable
        if probe_trainable and not hasattr(self, "_probe_logged"):
            self.log("probe_optimization_started", 1.0, on_epoch=True)
            self._probe_logged = True

        if dr_trainable and not hasattr(self, "_dr_logged"):
            self.log("dr_optimization_started", 1.0, on_epoch=True)
            self._dr_logged = True

    def get_learning_rates(self) -> Dict[str, float]:
        """Get current learning rates for each parameter group."""
        if not hasattr(self, "trainer") or self.trainer is None:
            return {
                "object": self.object_learning_rate,
                "probe": self.probe_learning_rate,
                "dr": self.dr_learning_rate,
                "other": self.learning_rate,
            }

        optimizers = self.optimizers()
        if optimizers is None:
            return {
                "object": self.object_learning_rate,
                "probe_model": self.probe_learning_rate,
                "dr": self.dr_learning_rate,
                "other": self.learning_rate,
            }

        # Handle both single optimizer and list of optimizers
        if isinstance(optimizers, list):
            if len(optimizers) == 0:
                return {
                    "object": self.object_learning_rate,
                    "probe_model": self.probe_learning_rate,
                    "dr": self.dr_learning_rate,
                    "other": self.learning_rate,
                }
            optimizer = optimizers[0]
        else:
            optimizer = optimizers

        lr_dict = {}
        for group in optimizer.param_groups:
            name = group.get("name", "unknown")
            lr_dict[name] = group["lr"]

        return lr_dict
