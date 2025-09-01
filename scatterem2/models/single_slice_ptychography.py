from pathlib import Path
from typing import Any, Dict, Optional, Union

import lightning as L
import torch
import torch.nn as nn
import yaml
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchmetrics import (
    NormalizedRootMeanSquaredError,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from scatterem2.nn import SingleSlicePtychographyModel
from scatterem2.nn.functional import amplitude_loss
from scatterem2.utils import partially_coherent_mean_square_error
from scatterem2.utils.data import Metadata4dstem


class SingleSlicePtychography(L.LightningModule):
    def __init__(
        self,
        meta4d: Optional[Metadata4dstem] = None,
        config_path: Optional[Union[str, Path]] = None,
        object_model: Optional[str] = "pixels",
        do_subpix_shift: Optional[bool] = None,
        ground_truth_object: Optional[Tensor] = None,
        ground_truth_probe: Optional[Tensor] = None,
        ground_truth_positions: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        learning_rate: Optional[float] = None,
        object_learning_rate: Optional[float] = None,
        probe_learning_rate: Optional[float] = None,
        dr_learning_rate: Optional[float] = None,
        optimize_probe_start: Optional[int] = None,
        optimize_dr_start: Optional[int] = None,
        batch_size: Optional[int] = None,
        loss_function: Optional[Any] = None,
        margin_probes: Optional[int] = 0,
        fix_probe_norm: Optional[bool] = True,
        **kwargs,
    ) -> None:
        """Initialize SingleSlicePtychography with either direct parameters or YAML config.

        Args:
            meta4d: Metadata4dstem object with experimental parameters. If None, will be loaded from config.
            config_path: Path to YAML config file. If provided, will load parameters from this file.
            object_model: Type of object model ("pixels" or "hash_encoding")
            do_subpix_shift: Whether to enable subpixel shifting
            ground_truth_object: Ground truth object for validation metrics
            device: Device to run on ("cuda" or "cpu")
            learning_rate: Learning rate for optimizer
            object_learning_rate: Learning rate for object parameters
            probe_learning_rate: Learning rate for probe parameters
            dr_learning_rate: Learning rate for dr parameters
            optimize_probe_start: Epoch to start optimizing probe parameters
            optimize_dr_start: Epoch to start optimizing dr parameters
            batch_size: Batch size for training
            loss_function: Loss function to use
            **kwargs: Additional parameters that will override config values
        """
        super().__init__()
        self.automatic_optimization = False

        # Load config if provided
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
                ground_truth_object,
                ground_truth_probe,
                ground_truth_positions,
                device,
                learning_rate,
                object_learning_rate,
                probe_learning_rate,
                dr_learning_rate,
                optimize_probe_start,
                optimize_dr_start,
                batch_size,
                loss_function,
                fix_probe_norm,
            )
        else:
            # Use direct parameters
            if meta4d is None:
                raise ValueError("Either meta4d or config_path must be provided")
            self._setup_from_params(
                meta4d,
                object_model,
                do_subpix_shift,
                ground_truth_object,
                ground_truth_probe,
                ground_truth_positions,
                device,
                learning_rate,
                object_learning_rate,
                probe_learning_rate,
                dr_learning_rate,
                optimize_probe_start,
                optimize_dr_start,
                batch_size,
                loss_function,
                fix_probe_norm,
            )

        self.save_hyperparameters()
        self.psnr = PeakSignalNoiseRatio().to(device)
        self.ssim = StructuralSimilarityIndexMeasure(kernel_size=3).to(device)
        self.nrmse = NormalizedRootMeanSquaredError(normalization="l2").to(device)

        self.forward_model = SingleSlicePtychographyModel(
            self.meta4d,
            self.object_model,
            self.target_device,
            self.do_subpix_shift,
            do_position_correction=False,
            margin_probes=margin_probes,
        )

        self.probe_norm = torch.norm(self.forward_model.probe[0].detach())

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
        ground_truth_object: Optional[Tensor],
        ground_truth_probe: Optional[Tensor],
        ground_truth_positions: Optional[Tensor],
        device: Optional[torch.device],
        learning_rate: Optional[float],
        object_learning_rate: Optional[float],
        probe_learning_rate: Optional[float],
        dr_learning_rate: Optional[float],
        optimize_probe_start: Optional[int],
        optimize_dr_start: Optional[int],
        batch_size: Optional[int],
        loss_function: Optional[Any],
        fix_probe_norm: Optional[bool],
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
        device = device or torch.device(config.get("accelerator", "cpu"))
        self.target_device = device

        # Move ground truth object to device if provided
        if ground_truth_object is not None:
            self.ground_truth_object = ground_truth_object.to(device)
        else:
            self.ground_truth_object = None

        if ground_truth_probe is not None:
            self.ground_truth_probe = ground_truth_probe.to(device)
        else:
            self.ground_truth_probe = None

        if ground_truth_positions is not None:
            self.ground_truth_positions = ground_truth_positions.to(device)
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

    def _setup_from_params(
        self,
        meta4d: Metadata4dstem,
        object_model: Optional[str],
        do_subpix_shift: Optional[bool],
        ground_truth_object: Optional[Tensor],
        ground_truth_probe: Optional[Tensor],
        ground_truth_positions: Optional[Tensor],
        device: Optional[torch.device],
        learning_rate: Optional[float],
        object_learning_rate: Optional[float],
        probe_learning_rate: Optional[float],
        dr_learning_rate: Optional[float],
        optimize_probe_start: Optional[int],
        optimize_dr_start: Optional[int],
        batch_size: Optional[int],
        loss_function: Optional[Any],
        fix_probe_norm: Optional[bool],
    ) -> None:
        """Setup parameters directly from function arguments."""
        self.meta4d = meta4d
        self.object_model = object_model or "pixels"
        self.do_subpix_shift = do_subpix_shift if do_subpix_shift is not None else True
        device = device or torch.device("cpu")
        self.target_device = device

        # Move ground truth object to device if provided
        if ground_truth_object is not None:
            self.ground_truth_object = ground_truth_object.to(device)
        else:
            self.ground_truth_object = None

        if ground_truth_probe is not None:
            self.ground_truth_probe = ground_truth_probe.to(device)
        else:
            self.ground_truth_probe = None

        if ground_truth_positions is not None:
            self.ground_truth_positions = ground_truth_positions.to(device)
        else:
            self.ground_truth_positions = None

        self.learning_rate = learning_rate or 10e-3
        self.object_learning_rate = object_learning_rate or 10e-3
        self.probe_learning_rate = probe_learning_rate or 10e-3
        self.dr_learning_rate = dr_learning_rate or 1
        self.batch_size = batch_size or 128
        self.loss_function = loss_function or torch.nn.functional.mse_loss
        self.optimize_probe_start = optimize_probe_start or 1
        self.optimize_dr_start = optimize_dr_start or 1
        self.fix_probe_norm = fix_probe_norm or True

    def forward(
        self,
        probe_index: int,
        angles_index: int,
        r_indices: Tensor,
        translation_index: int,
    ) -> Tensor:
        measurements = self.forward_model(
            probe_index, angles_index, r_indices, translation_index
        )
        return measurements

    def training_step(
        self, batch: tuple[int, int, int, Tensor, int, Tensor], batch_idx: int
    ) -> Tensor:
        def closure() -> float:
            opt.zero_grad(set_to_none=True)
            ind, probe_index, angles_index, r_indices, translation_index, data = batch
            measurements = self.forward_model.forward(
                probe_index, int(angles_index), r_indices, int(translation_index)
            )
            # print(f"measurements.sum(): {measurements.sum()}")
            # print(f"data.sum():         {data.sum()}")
            loss_per_pattern = amplitude_loss(measurements, data)
            loss = torch.sum(loss_per_pattern)
            if loss.requires_grad:
                self.manual_backward(loss)
            return loss

        opt = self.optimizers()
        loss = opt.step(closure=closure)
        loss_value = loss.detach().item()
        self.log(
            "L", loss_value, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called in the training loop after the batch.

        Args:
            outputs: The outputs of training_step(x)
            batch: The batched data as it is returned by the training DataLoader.
            batch_idx: the index of the batch

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.

        """

        # if self.forward_model.do_position_correction:
        #     self.forward_model.step_positions(self.dr_learning_rate)
        with torch.no_grad():
            if self.fix_probe_norm:
                self.forward_model.probe[0] *= self.probe_norm / torch.norm(
                    self.forward_model.probe[0]
                )

    def on_train_epoch_start(self) -> None:
        """Hook called at the start of each training epoch"""
        # Update gradient requirements based on current epoch
        self._update_gradient_requirements()

        # Handle position correction start
        # if self.current_epoch >= self.optimize_dr_start:
        #     self.forward_model.do_position_correction = True

    def on_train_epoch_end(self) -> None:
        if self.ground_truth_object is not None:
            # Calculate PSNR and SSIM

            current_object = self.forward_model.object.data.detach().clone()
            current_object = current_object.real
            gt = self.ground_truth_object.real.unsqueeze(0).unsqueeze(0)
            current_object -= current_object.mean() - gt.mean()

            psnr_val = self.psnr(current_object, gt)
            ssim_val = self.ssim(current_object, gt)
            nrmse = self.nrmse(current_object.contiguous(), gt.contiguous())

            # Log metrics
            self.log("PSNR-O", psnr_val, prog_bar=False, on_epoch=True)
            self.log("SSIM-O", ssim_val, prog_bar=True, on_epoch=True)
            self.log("NRMSE-O", nrmse, prog_bar=False, on_epoch=True)

        if self.ground_truth_probe is not None:
            current_probe = self.forward_model.probe[0].detach().clone()
            ground_truth_probe = self.ground_truth_probe.detach()
            pcmse = partially_coherent_mean_square_error(
                current_probe, ground_truth_probe
            )
            self.log("PCMSE-P", pcmse, prog_bar=True, on_epoch=True)

        if self.ground_truth_positions is not None:
            current_positions = (
                self.forward_model.positions.detach().clone().to(torch.float32)
            )
            current_positions += self.forward_model.dr.detach()
            ground_truth_positions = self.ground_truth_positions
            nrmse_positions = self.nrmse(current_positions, ground_truth_positions)
            self.log("NRMSE-R", nrmse_positions, prog_bar=True, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        # Create parameter groups with different learning rates
        param_groups = []

        # Group 1: Object parameters
        object_params = []
        if hasattr(self.forward_model, "object"):
            if isinstance(self.forward_model.object, nn.Parameter):
                object_params.append(self.forward_model.object)
            elif hasattr(self.forward_model.object, "parameters"):
                object_params.extend(self.forward_model.object.parameters())

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
        if hasattr(self.forward_model, "probe"):
            for probe in self.forward_model.probe:
                probe_params.append(probe)

        if probe_params:
            param_groups.append(
                {
                    "params": probe_params,
                    "lr": self.probe_learning_rate,
                    "name": "probe",
                }
            )

        # Group 3: DR parameters (include even if not trainable initially)
        dr_params = []
        if hasattr(self.forward_model, "dr"):
            dr_params.append(self.forward_model.dr)

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
                if param in group["params"]:
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

        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.2,
            threshold=1e-9,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "L",  # monitor the loss metric
                "frequency": 1,
                "interval": "batch",
            },
        }

    def _configure_gradient_requirements(self) -> None:
        """Configure which parameters should have gradients computed."""
        # Configure probe gradients - start with False, will be updated during training
        if hasattr(self.forward_model, "probe"):
            for probe in self.forward_model.probe:
                probe.requires_grad = False

        # Configure dr gradients - start with False, will be updated during training
        if hasattr(self.forward_model, "dr"):
            self.forward_model.dr.requires_grad = False

    def _update_gradient_requirements(self) -> None:
        """Update gradient requirements based on current epoch."""
        # Configure probe gradients
        if hasattr(self.forward_model, "probe"):
            should_optimize_probe = (
                self.optimize_probe_start is not None
                and hasattr(self, "current_epoch")
                and self.current_epoch >= self.optimize_probe_start
            )
            for probe in self.forward_model.probe:
                probe.requires_grad = should_optimize_probe

        # Configure dr gradients
        if hasattr(self.forward_model, "dr"):
            should_optimize_dr = (
                self.optimize_dr_start is not None
                and hasattr(self, "current_epoch")
                and self.current_epoch >= self.optimize_dr_start
            )
            self.forward_model.dr.requires_grad = should_optimize_dr
            self.forward_model.do_position_correction = should_optimize_dr

        # Update optimizer if new parameters became trainable
        self._update_optimizer_if_needed()

    def _update_optimizer_if_needed(self) -> None:
        """Log when new parameters become trainable."""
        if not hasattr(self, "trainer") or self.trainer is None:
            return

        # Check if probe or dr parameters just became trainable
        probe_trainable = hasattr(self.forward_model, "probe") and any(
            probe.requires_grad for probe in self.forward_model.probe
        )
        dr_trainable = (
            hasattr(self.forward_model, "dr") and self.forward_model.dr.requires_grad
        )

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
                "probe": self.probe_learning_rate,
                "dr": self.dr_learning_rate,
                "other": self.learning_rate,
            }

        # Handle both single optimizer and list of optimizers
        if isinstance(optimizers, list):
            if len(optimizers) == 0:
                return {
                    "object": self.object_learning_rate,
                    "probe": self.probe_learning_rate,
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
