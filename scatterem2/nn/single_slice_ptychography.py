from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None
import torch
import torch.nn as nn
from kornia.filters import spatial_gradient
from torch import Tensor
from torch.nn import Module

import scatterem2.vis as vis
from scatterem2.nn.batch_crop import BatchCrop
from scatterem2.nn.functional import exitwave_measurement_multi_mode
from scatterem2.nn.modules import ComplexMulMultiMode, SubpixShift, ZernikeProbe
from scatterem2.utils import (
    advanced_raster_scan,
    circular_aperture,
    fftshift_checkerboard,
)
from scatterem2.utils.data import Metadata4dstem
from scatterem2.utils.data.data_classes import Aberrations


class MultiresolutionHashEncoding(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int] | torch.Size,
        n_input_dims: int,
        n_output_dims: int,
        encoding_config: dict,
        network_config: dict,
        device: str,
    ):
        """
        Multiresolution hash encoding for the object model
        Args:
            n_input_dims (int): Number of input dimensions for the encoding (typically 2 for 2D coordinates)
            n_output_dims (int): Number of output dimensions (typically 2 for complex values)
            encoding_config (dict): Configuration dictionary for the hash encoding
            network_config (dict): Configuration dictionary for the MLP network
            device (str): Device to place the model on ('cuda' or 'cpu')
        """
        super().__init__()
        if tcnn is None:
            raise ImportError(
                "tinycudann is not installed. Please install it with `pip install tinycudann`."
            )
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=n_input_dims,
            n_output_dims=n_output_dims,
            encoding_config=encoding_config,
            network_config=network_config,
        ).to(device)
        resolution = image_shape
        self.img_shape = resolution + torch.Size([2])
        # n_pixels = resolution[0] * resolution[1]  # * resolution[2]

        half_dx = 0.5 / resolution[0]
        half_dy = 0.5 / resolution[1]
        # half_dz =  0.5 / resolution[2]
        xs = torch.linspace(half_dx, 1 - half_dx, resolution[0], device=device)
        ys = torch.linspace(half_dy, 1 - half_dy, resolution[1], device=device)
        # zs = th.linspace(half_dz, 1-half_dz, resolution[2], device=device)
        xv, yv = torch.meshgrid([xs, ys])
        # xi, yi = th.meshgrid([xs, ys])
        self.xyz = torch.stack((yv.flatten(), xv.flatten())).t()

    def forward(self: "MultiresolutionHashEncoding") -> torch.Tensor:
        return (
            torch.view_as_complex(
                self.model(self.xyz)
                .reshape(self.img_shape)
                .clamp(0.0, 1.0)
                .to(torch.float32)
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )


class SingleSlicePtychographyModel(Module):
    def __init__(
        self,
        meta4d: Metadata4dstem,
        object_representation: str = "pixels",
        device: torch.device = torch.device("cpu"),
        do_subpix_shift: bool = False,
        do_position_correction: bool = False,
        beta: int = 1,
        alpha: int = 2,
        margin_probes: Optional[int] = 0,
        max_shift: Optional[float] = 1,
    ) -> None:
        super(SingleSlicePtychographyModel, self).__init__()
        self.device = device
        self.meta4d = meta4d
        self.do_position_correction = do_position_correction
        self.do_subpix_shift = do_subpix_shift
        self.max_shift = max_shift
        # print(f"SingleSlicePtychographyModel:k_max: {k_max}")
        object_resolution_angstrom = meta4d.dr
        scan_step_pixels = meta4d.sampling[:2] / object_resolution_angstrom
        print(
            f"SingleSlicePtychographyModel:scan_step_pixels:              {scan_step_pixels}"
        )
        print(
            f"SingleSlicePtychographyModel:object_resolution_angstrom:    {object_resolution_angstrom}"
        )
        # ny=10, nx=10, fast_axis=1, mirror=[1, 1], theta=0, dy=1, dx=1, device="cpu"
        positions_float = advanced_raster_scan(
            meta4d.shape[0],
            meta4d.shape[1],
            1,
            mirror=[1, 1],
            theta=meta4d.rotation,
            dy=scan_step_pixels[0],
            dx=scan_step_pixels[1],
            device=device,
        )

        self.positions = torch.round(positions_float).to(torch.int32)
        self.dr = positions_float - self.positions.float()
        self.ddr_accum = torch.zeros_like(self.dr)
        # print(f"SingleSlicePtychographyModel:dr.max(0): {self.dr.max(0)}")
        scan_extent_angstrom = meta4d.sampling[:2] * meta4d.shape[:2]
        margin = margin_probes * meta4d.shape[-2:]
        object_shape = np.round(
            scan_extent_angstrom / object_resolution_angstrom
        ).astype(int) + margin.astype(int)
        if object_representation == "pixels":
            # compute object shape from meta4d

            # print(f"SingleSlicePtychographyModel:object_shape: {object_shape}")
            object_model = torch.zeros(
                (1, 1, *object_shape), device=device, dtype=torch.complex64
            )
            object_model += 1e-8
            self.object = nn.Parameter(object_model, requires_grad=True)
        elif object_representation == "hash_encoding":
            # encoding:
            #     otype: HashGrid
            #     n_levels: 5
            #     n_features_per_level: 2
            #     log2_hashmap_size: 15
            #     base_resolution: 16
            #     per_level_scale: 1.5

            # network:
            #     otype: FullyFusedMLP
            #     activation: ReLU
            #     output_activation: None
            #     n_neurons: 64
            #     n_hidden_layers: 2
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": 5,
                "n_features_per_level": 2,
                "log2_hashmap_size": 15,
                "base_resolution": 16,
                "per_level_scale": 1.5,
            }
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": None,
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
            self.object = MultiresolutionHashEncoding(
                image_shape=tuple(object_shape),
                n_input_dims=2,
                n_output_dims=1,
                encoding_config=encoding_config,
                network_config=network_config,
                device=device.type,
            )
        else:
            raise ValueError(f"Invalid object model: {object_representation}")
        if meta4d.vacuum_probe is not None:
            aperture_array = torch.as_tensor(meta4d.vacuum_probe, device=device)
        else:
            k_aperture = meta4d.semiconvergence_angle / meta4d.wavelength
            r_aperture = k_aperture / meta4d.sampling[-2]
            print(f"SingleSlicePtychographyModel:r_aperture: {r_aperture}")

            aperture_array = circular_aperture(
                r=r_aperture, shape=tuple(meta4d.shape[-2:]), device=device
            )
            vis.show_2d(aperture_array, cbar=True, title="Aperture")
        if meta4d.aberrations is not None:
            aberrations = meta4d.aberrations

        else:
            aberrations = Aberrations(torch.zeros((12,)))
            aberrations.array[0] = -meta4d.defocus_guess
        print(
            f"SingleSlicePtychographyModel:aberrations: {aberrations.array.cpu().numpy()}"
        )
        probe_model = ZernikeProbe(
            meta4d.wavelength,
            tuple(object_resolution_angstrom),
            aperture_array,
            aberrations,
            fourier_space=False,
            make_beamlet_meta=False,
            n_radial_samples=4,
            n_angular_samples=6,
            make_plane_waves=False,
            fft_shifted=True,
        )
        self.probe = [probe_model.forward().detach()]

        self.probe[0].requires_grad = False
        self.positions.requires_grad = False
        self.dr.requires_grad = False
        self.ddr_accum.requires_grad = False

        vis.show_2d(self.probe[0][0, 0], cbar=True, title="Probe")
        self.fftshift_checkerboard = (
            torch.tensor(
                fftshift_checkerboard(self.probe[0].shape[-1], self.probe[0].shape[-2]),
                device=self.probe[0].device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.shift = SubpixShift(
            self.probe[0].shape, self.probe[0].device.type, do_subpix_shift
        )

        def probe_gradient_preconditioner(
            module: Module, grad_input: Tensor, grad_output: Tensor
        ) -> Tuple[Tensor, ...]:
            grad_probe, grad_wave, grad_r = grad_input
            # A = alpha * (th.max(module.probe_norm) - module.probe_norm)
            A = alpha * torch.mean(module.probe_norm)
            denom = module.probe_norm + A
            new_grad_probe = grad_probe / denom
            return new_grad_probe, grad_wave, grad_r

        def cropping_gradient_preconditioner(
            module: Module, grad_input: Tensor, grad_output: Tensor
        ) -> Tuple[Tensor, ...]:
            grad_object, grad_wave, grad_r = grad_input
            # B = beta * (th.max(module.object_norm) - module.object_norm)
            B = beta
            denom = module.object_norm + B
            new_grad_object_patches = grad_object / denom
            return new_grad_object_patches, grad_wave, grad_r

        def position_correction_hook(
            module: Module, grad_input: Tensor, grad_output: Tensor
        ) -> Tuple[Tensor, ...]:
            grad_object_patches, grad_waves = grad_input
            (updated_waves,) = grad_output

            # if self.do_position_correction:
            #     diff_waves = updated_waves
            #     # self.updated_waves = updated_waves.detach().clone()
            #     r_inds = self.current_r_indices
            #     # only use first mode
            #     # zR = self.probe[0][[0]].detach().unsqueeze(2) * self.dO_dr
            #     zR = self.zR[0]
            #     # zR = zR[0]
            #     # plot_complex_multi(zR[:100, 0].detach().cpu().numpy(), title='zR')
            #     dr_num = torch.sum(
            #         torch.real(zR.conj() * diff_waves[0].unsqueeze(1)), dim=(-2,-1)
            #     )
            #     dr_denom = torch.sum(torch.abs(zR) ** 2, dim=(-2,-1))
            #     ddr = dr_num / dr_denom
            #     self.ddr = torch.minimum(
            #         torch.abs(ddr), torch.tensor([0.1], device=ddr.device)
            #     ) * torch.sign(ddr)
            #     self.ddr_accum[r_inds].add_(self.ddr)
            #     self.ddr[self.ddr_accum[r_inds] > self.max_shift] = 0

            return grad_object_patches, grad_waves

        self.shift.register_full_backward_hook(probe_gradient_preconditioner)
        self.batch_crop = BatchCrop(self.object)
        self.batch_crop.register_full_backward_hook(cropping_gradient_preconditioner)
        # self.batch_crop = BatchCrop.apply

        self.complex_mul = ComplexMulMultiMode()
        self.complex_mul.register_full_backward_hook(position_correction_hook)

    def step_positions(self, lr: float) -> None:
        # angles_ind = self.angles_index

        assert (
            self.do_subpix_shift and self.do_position_correction
        ), "Position refinement requires both subpixel precision and correct_positions=True"
        # spatial_gradient swaps the axes -_-
        self.dr[self.current_r_indices, 0] -= lr * self.ddr[:, 1]
        self.dr[self.current_r_indices, 1] -= lr * self.ddr[:, 0]

    def forward(
        self,
        probe_index: int,
        angles_index: int,
        r_indices: Iterable[int],
        translation_index: int,
    ) -> Tensor:
        probe = self.probe[probe_index]
        positions = self.positions[r_indices].detach()
        dr = self.dr[r_indices]
        self.current_r_indices = r_indices
        # self.angles_index = angles_index

        object = torch.exp(1j * self.object[0])
        patches = self.batch_crop(object, probe, positions)
        # print(f"positions: {positions}")
        probe = self.shift(probe, dr, patches)
        # if self.do_position_correction:
        #     # (Nmodes, K, 2, MY, MX)
        #     self.dO_dr = spatial_gradient(patches.detach(), mode="diff", order=1)
        #     # (Nmodes x K x 1 x M1 x M2) * (1, K, 2, MY, MX)
        #     self.zR = probe.detach()[[0]].unsqueeze(2) * self.dO_dr

        waves = self.complex_mul(patches, probe)
        waves = torch.fft.fft2(waves, norm="ortho")

        # %%
        # measurements = th.sqrt( # sum over modes
        #     th.sum(th.abs(waves) ** 2, 0) + th.finfo(waves.dtype).eps
        # )
        measurements = exitwave_measurement_multi_mode(
            waves  # * self.fftshift_checkerboard
        )
        return measurements
