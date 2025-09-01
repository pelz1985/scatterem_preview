from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
from itertools import count
from pathlib import Path
from typing import Callable, DefaultDict, Dict, Iterable, Tuple

import numpy as np
import torch
from attr import dataclass
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim import Optimizer

from scatterem2.nn.volume_samplers import (
    AdvancedGaussianPatchSampler,
    GaussianPatchSampler,
)
from scatterem2.utils.plot import plot_patterns, plot_probe, plot_volume


class PeriodicCallback(Callback):
    def periodic_run(func) -> Callable:
        def wrapper(self: Callback, *args, **kwargs) -> None:
            self.iter_id = next(self.counter)
            if self.iter_id % self.period == 0:
                return func(self, *args, **kwargs)
            return None

        return wrapper

    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        self.counter = count()
        self.iter_id = None


class PlotVolumeCallback(PeriodicCallback):
    def __init__(
        self,
        dir_path: str | Path | None = None,
        name_prefix: str = "volume",
        save_real: bool = True,
        save_imag: bool = False,
        view_planes: str | Iterable[str] = ["xy", "yz", "xz"],
        values_units: str | None = None,
        grid_sampling: float | None = None,
        grid_units: str | None = None,
        period: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(period=period)
        self.save_real = save_real
        self.save_imag = save_imag
        if isinstance(view_planes, str):
            view_planes = [view_planes]
        self.view_planes = view_planes
        self.values_units = values_units
        self.grid_sampling = grid_sampling
        self.grid_units = grid_units
        self.kwargs = kwargs
        if dir_path is not None:
            self.file_path = Path(dir_path) / name_prefix
        else:
            self.file_path = None

    @PeriodicCallback.periodic_run
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        volume = pl_module.model.sampler.get_volume()[0]
        file_path = self.file_path
        if file_path is not None:
            file_path = f"{str(file_path)}_epoch_{self.iter_id}"
        for view_plane in self.view_planes:
            title = f"Reconstructed Potential, Plane {view_plane}"
            if self.save_real:
                path_real = (
                    f"{file_path}_{view_plane}_real.png"
                    if file_path is not None
                    else None
                )
                title_real = f"{title}, Real Part"
                plot_volume(
                    volume.real,
                    save_path=path_real,
                    view_plane=view_plane,
                    title=title_real,
                    values_units=self.values_units,
                    grid_sampling=self.grid_sampling,
                    grid_units=self.grid_units,
                    **self.kwargs,
                )
            if self.save_imag:
                path_imag = (
                    f"{file_path}_{view_plane}_imag.png"
                    if file_path is not None
                    else None
                )
                title_imag = f"{title}, Imaginary Part"
                plot_volume(
                    volume.imag,
                    save_path=path_imag,
                    view_plane=view_plane,
                    title=title_imag,
                    values_units=self.values_units,
                    grid_sampling=self.grid_sampling,
                    grid_units=self.grid_units,
                    **self.kwargs,
                )


class PlotProbeCallback(PeriodicCallback):
    def __init__(
        self,
        dir_path: str | Path | None = None,
        name_prefix: str = "probe",
        probe_keys: Iterable[Tuple] = None,
        sampling: float | None = None,
        units: str | None = None,
        figsize: Tuple[int, int] | None = (8, 4),
        period: int = 1,
    ) -> None:
        super().__init__(period=period)
        self.dir_path = dir_path
        self.name_prefix = name_prefix
        self.probe_keys = probe_keys
        self.sampling = sampling
        self.units = units
        self.figsize = figsize

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.probe_keys is None:
            self.probe_keys = pl_module.model.probe_model.keys()
        for key in self.probe_keys:
            if self.dir_path is None:
                save_path = None
            else:
                save_path = (
                    Path(self.dir_path)
                    / f"{self.name_prefix}_angle_id_{key[0]}_probe_id_{key[1]}_fit_start.png"
                )
            probe = pl_module.model.probe_model[str(key)][0, 0]
            plot_probe(
                probe=probe,
                save_path=save_path,
                sampling=self.sampling,
                units=self.units,
                figsize=self.figsize,
            )

    @PeriodicCallback.periodic_run
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for key in self.probe_keys:
            if self.dir_path is None:
                save_path = None
            else:
                save_path = (
                    Path(self.dir_path)
                    / f"{self.name_prefix}_key_{key}_epoch_{self.iter_id}.png"
                )
            probe = pl_module.model.probe_model[str(key)][0, 0]
            plot_probe(
                probe=probe,
                save_path=save_path,
                sampling=self.sampling,
                units=self.units,
                figsize=self.figsize,
            )


class PlotDiffractionPattern(Callback):
    def __init__(
        self,
        dir_path: str | Path | None = None,
        name_prefix: str = "pattern",
        angle_ids: Iterable[int] | None = None,
        probe_ids: Iterable[int] | None = None,
        scan_positions: Iterable[Tuple[int, int]] | None = None,
        sampling: float | None = None,
        units: str | None = None,
        figsize: Tuple[int, int] | None = (8, 4),
        period: int = 1,
    ) -> None:
        super().__init__()
        self.dir_path = dir_path
        self.name_prefix = name_prefix
        self.angle_ids = angle_ids
        self.probe_ids = probe_ids
        self.scan_positions = scan_positions
        self.sampling = sampling
        self.units = units
        self.figsize = figsize
        self.period = period
        self.epoch_counter = count()

        self.epoch_id = None
        self.scan_positions_tensor = None

    @staticmethod
    def get_matched_positions_mask(
        selected_positions: Tensor, batch_positions: Tensor
    ) -> Tensor:
        matches = (selected_positions[:, None, :] == batch_positions[None, :, :]).all(
            dim=2
        )
        mask = matches.any(dim=0)
        return mask

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.scan_positions is not None:
            self.scan_positions_tensor = torch.tensor(
                self.scan_positions, device=pl_module.device
            )

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.epoch_id = next(self.epoch_counter)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Tuple,
        batch_idx: int,
    ) -> None:
        ind, probe_index, angles_index, r_indices, translation_index, data = batch
        angles_condition = (
            (angles_index in self.angle_ids) if self.angle_ids is not None else True
        )
        probe_condition = (
            (probe_index in self.probe_ids) if self.probe_ids is not None else True
        )
        if (self.epoch_id % self.period == 0) and angles_condition and probe_condition:
            batch_positions = pl_module.model.positions[angles_index][r_indices]

            measurements = outputs["measurements"].detach().cpu().numpy()
            data = outputs["data"].detach().cpu().numpy()

            if self.scan_positions is not None:
                mask = self.get_matched_positions_mask(
                    selected_positions=self.scan_positions_tensor,
                    batch_positions=batch_positions,
                )
                matched_positions = batch_positions[mask].cpu().numpy()
                pos_ids = torch.argwhere(mask).cpu().numpy()
            else:
                matched_positions = batch_positions.cpu().numpy()
                pos_ids = np.arange(batch_positions.shape[0])

            for pos_id, pos in zip(pos_ids, matched_positions):
                pos = tuple(pos.tolist())
                pos_id = pos_id.item()
                pattern_pred = np.fft.fftshift(measurements[pos_id])
                pattern_gt = np.fft.fftshift(data[pos_id])
                if self.dir_path is None:
                    save_path = None
                else:
                    save_path = (
                        Path(self.dir_path)
                        / f"{self.name_prefix}_angle_id_{angles_index}_probe_id_{probe_index}_pos_{pos}_epoch_{self.epoch_id}.png"
                    )
                plot_patterns(
                    pattern_pred=pattern_pred,
                    pattern_gt=pattern_gt,
                    save_path=save_path,
                    sampling=self.sampling,
                    units=self.units,
                    figsize=self.figsize,
                )
