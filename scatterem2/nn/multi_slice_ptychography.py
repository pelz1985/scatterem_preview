from importlib.util import find_spec
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from kornia.filters import filter3d
from torch import Tensor, nn
from torch.nn import Module

if find_spec("torchvision") is not None:
    from torchvision.transforms.functional import gaussian_blur

from scatterem2 import vis
from scatterem2.nn import (
    BatchCrop,
    ComplexMulMultiMode,
    Propagator,
    SubpixShift,
    ZernikeProbe,
)
from scatterem2.nn.functional import (
    exitwave_measurement_multi_mode,
    overlap_intensity,
    overlap_intensity_from_wave,
)
from scatterem2.nn.volume_samplers import TrivialSamplerLegacy
from scatterem2.utils import (
    advanced_raster_scan,
    batch_unique_with_inverse,
    circular_aperture,
)
from scatterem2.utils.data import Metadata4dstem
from scatterem2.utils.data.abberations import Aberrations


class MultiSlicePtychographyModel(Module):
    def __init__(
        self,
        meta4d: Metadata4dstem,
        # slice_distance: float,
        # object_thickness_nm: float,
        object_model: str = "pixels",
        device: str = "cuda",
        do_subpix_shift: bool = False,
        do_position_correction: bool = False,
        kernel: Tensor | None = None,
        mixed_probe: bool = False,
        pmodes: bool = 1,
        pmode_init_pows: List = [0.25, 0.25, 0.25, 0.25],
        margin_probes: Optional[int] = 1.2,
        measurement_norm: Optional[float] = None,
        vacuum_probe: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        self.meta4d = meta4d
        self.slice_distance = self.meta4d.slice_thickness
        self.device = device
        self.do_subpix_shift = do_subpix_shift
        self.subpix_shift = True
        self.shift = SubpixShift(
            (1, 1, *meta4d.detector_shape),
            device,
            subpix_shift=do_subpix_shift,
        )
        self.do_position_correction = True
        object_resolution_angstrom = meta4d.dr
        scan_step_pixels = meta4d.sampling[:2] / object_resolution_angstrom
        print(
            f"SingleSlicePtychographyModel:scan_step_pixels:              {scan_step_pixels}"
        )
        print(
            f"SingleSlicePtychographyModel:object_resolution_angstrom:    {object_resolution_angstrom}"
        )
        positions_float = advanced_raster_scan(
            meta4d.shape[0],
            meta4d.shape[1],
            1,
            mirror=[1, 1],
            theta=meta4d.rotation,
            dy=scan_step_pixels[0],
            dx=scan_step_pixels[1],
            device=device,
        ) + 0.15 * torch.randn(
            (meta4d.shape[:2].prod(), 2), device=device, dtype=torch.float32
        )

        positions_float = positions_float - positions_float.mean(0)
        object_shape = 1.2 * torch.ceil(
            positions_float.max(0).values
            - positions_float.min(0).values
            + torch.as_tensor(meta4d.shape[2:], device=device)
        )
        positions_float = positions_float + torch.ceil(
            (object_shape / 2) - (torch.as_tensor(meta4d.shape[2:], device=device) / 2)
        )
        self.positions = torch.round(positions_float).to(torch.int32)
        self.dr = nn.Parameter(positions_float - torch.round(positions_float))
        self.ddr_accum = torch.zeros_like(positions_float)
        print(f"SingleSlicePtychographyModel:dr.max(0): {self.dr.max(0)}")
        # scan_extent_angstrom = meta4d.sampling[:2] * meta4d.shape[:2]
        # margin = margin_probes * meta4d.shape[-2:]
        # object_shape = np.round(
        #     scan_extent_angstrom / object_resolution_angstrom
        # ).astype(int) + margin.astype(int)

        if object_model == "pixels":
            # compute object shape from meta4d
            if self.meta4d.sample_thickness_guess != 0:
                self.NZ = int(
                    np.ceil(self.meta4d.sample_thickness_guess / self.slice_distance)
                )
            else:
                self.NZ = 1
            object_model = torch.zeros(
                (1, self.NZ, *object_shape.int()), device=device, dtype=torch.complex64
            )
            object_model.real = 1e-8 * torch.rand(*object_model.shape, device=device)
            object_model += 1e-8
            self.object = nn.Parameter(object_model, requires_grad=True)

        else:
            raise ValueError(f"Invalid object model: {object_model}")
        self.sampler = TrivialSamplerLegacy(self.object)

        probes = []

        if vacuum_probe is not None:
            aperture_array = (
                vacuum_probe  # torch.as_tensor(meta4d.vacuum_probe, device=device)
            )
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
            object_resolution_angstrom,
            aperture_array,
            meta4d.aberrations,
            fourier_space=False,
            make_beamlet_meta=False,
            n_radial_samples=4,
            n_angular_samples=6,
            make_plane_waves=False,
            fft_shifted=True,
            mixed_probe=mixed_probe,
            pmodes=pmodes,
            pmode_init_pows=pmode_init_pows,  # Initial powers for the modes
            measurement_norm=measurement_norm,
        )
        probes_array = probe_model.forward().detach()
        probes.append([probes_array])
        if len(probes) == 1:
            vis.show_2d(
                probes[0][0][:, 0],
                axsize=(5, 4),
                cbar=True,
                tight_layout=True,
                title=[f"Probe Mode {el+1}" for el in list(range(pmodes))],
            )

        probe_model_dict = {}
        for i, probe_slice in enumerate(probes):
            for j, probe in enumerate(probe_slice):
                probe_model_dict[str((i, j))] = probe

        self.probe_model = nn.ParameterDict(parameters=probe_model_dict)

        self.propagator = Propagator(
            self.slice_distance,
            meta4d.wavelength,
            object_resolution_angstrom,
            meta4d.detector_shape,
            device=device,
        )

        NVol, _, NY, NX = self.object.data.shape
        Nmodes, kk, MY, MX = probes_array.shape
        # # K, _ = self.positions[0].shape
        self.object_norm = torch.zeros(
            (self.NZ, NY, NX),
            device=device,
            dtype=torch.float32,
        )
        self.probe_norm = torch.zeros(
            (Nmodes, kk, MY, MX),
            device=device,
            dtype=torch.float32,
        )

        self.probe_model[str((0, 0))].requires_grad = False
        self.positions.requires_grad = False
        self.dr.requires_grad = False
        self.ddr_accum.requires_grad = False

        def cropping_backward_hook(
            module: Module, grad_input: Tensor, grad_output: Tensor
        ) -> Tuple[Tensor, ...]:
            grad_object, grad_wave, grad_r = grad_input

            alpha = 0.9
            denom = torch.sqrt(
                1e-16
                + ((1 - alpha) * self.object_norm) ** 2
                + (alpha * torch.max(self.object_norm)) ** 2
            )
            new_grad_object_patches = (
                grad_object / denom if grad_object is not None else None
            )
            return new_grad_object_patches, grad_wave, grad_r

        # def binning_backward_hook(
        #     module: Module, grad_input: Tensor, grad_output: Tensor
        # ) -> Tuple[Tensor, ...]:
        #     (a,) = grad_input

        #     if kernel is None:
        #         new_grad_input = a
        #     else:
        #         new_grad_input = filter3d(
        #             a.unsqueeze(0).unsqueeze(0), kernel, border_type="constant"
        #         )
        #         new_grad_input = new_grad_input[0][0]
        #         # vol = volk
        #         # volk = torch.abs(torch.fft.ifftn(torch.fft.fftn(vol) * kernel))
        #         new_grad_input = new_grad_input * torch.sqrt(
        #             torch.sum(torch.abs(a) ** 2)
        #             / torch.sum(torch.abs(new_grad_input) ** 2)
        #         )
        #     # grad_sum = torch.sum(grad_output)
        #     # if new_grad_input.shape != a.shape:
        #     #     raise ValueError('new_grad_input.shape != a.shape')
        #     return (new_grad_input,)

        self.batch_crop = BatchCrop(self.object)
        # self.batch_crop.register_full_backward_hook(cropping_backward_hook)

        self.batch_unique_with_inverse = (
            {}
        )  # (batch_id, angles_index) : (values, inverse_indices)

    def get_unique_with_inverse(
        self,
        batch_id: int,
        angles_index: int,
        positions: Tensor,
        probe_shape: Tuple[int, int],
    ) -> Tuple[Tensor, List]:
        uniques_with_inverse = self.batch_unique_with_inverse.get(
            (batch_id, angles_index), None
        )
        if uniques_with_inverse is not None:
            uniques, inverse = uniques_with_inverse

            return uniques.to(positions.device), inverse.to(positions.device)

        uniques, inverse = batch_unique_with_inverse(
            positions=positions, patch_shape=probe_shape
        )
        self.batch_unique_with_inverse[(batch_id, angles_index)] = (
            uniques.detach().cpu(),
            inverse.detach().cpu(),
        )

        return uniques, inverse

    def get_object_model(self) -> Tensor:
        return self.sampler.get_volume()

    def reset_sampler(self, sampler: Tensor) -> None:
        if isinstance(sampler, Module):
            self.sampler = sampler
        else:
            self.sampler = TrivialSamplerLegacy(sampler)

    def multislice_exitwave_multimode(
        self, object_patches: Tensor, waves: Tensor, pos: Tensor, propagator: Propagator
    ) -> Tensor:  # , use_warp: bool = True):
        """
        Implements the multislice algorithm - no anti-aliasing masks
        :param object_patches:             NZ_bin x K x M1 x M2         complex
        :param waves:                      Nmodes x K x M1 x M2     complex
        :param pos:                        K x 2                     real
        :param propagator: f: (K x M1 x M2, Nmodes      x K x M1 x M2) -> Nmodes      x K x M1 x M2
        :return: exitwaves:                Nmodes x K x M1 x M2
        """
        tmp = torch.zeros_like(self.object_norm[0, :, :])
        n = 0
        for n in range(len(object_patches) - 1):
            waves = object_patches[n].unsqueeze(0) * waves
            if object_patches.requires_grad:
                if self.subpix_shift:
                    overlap_intensity_from_wave(pos, waves, tmp)
                else:
                    patch_norm = torch.sum(torch.abs(waves.clone().detach()) ** 2, 0)
                    overlap_intensity(pos, patch_norm, tmp)
                self.object_norm[n, :, :] = tmp
            # print(f"--- slice {n:02d} mode 0 norm: {torch.norm(waves[0,0])}")
            waves = propagator(waves)
        waves = waves * object_patches[n].unsqueeze(0)
        # print(f"--- slice {n+1:02d} mode 0 norm: {torch.norm(waves[0,0])}")
        return waves

    def get_object_model_patches(
        self,
        probe: Tensor,
        pos: Tensor,
        angles: Tensor,
        translation: Tensor,
        bin_factor: int,
        debug: bool = True,
    ) -> Tensor:  # TODO: rewrite
        """Resamples the volume for a given position and crops it into patches
        Notation:
            NX, NY, NZ: voxel indexes for initial volume
            MX, MY: pixel indexes in a probe
            K: position index
            Nmodes: the number of modes in a probe
            Nangles: the number of tomography angles

        :param V: (1, 1, NX, NZ, Ny) complex; Potential map in a given volume
        :type V: Tensor
        :param probe: (Nmodes, K, MY, MX) complex;
        :type probe: Tensor
        :param pos: (K, 2) real; probe positions
        :type pos: Tensor
        :param angles: (Nangles, 3) real: tomography angles;
        The last dimention values correspond to {psi, thetha, psi}
        respectively
        :type angles: Tensor
        :param translation: (2, Nangles) real: corresponding translations
        :type translation: Tensor
        :param bin_factor: !!!Deprecated!!!: has no effect
        :type bin_factor: int
        :param start_end: ((ystart,yend),(xstart,xend)). !!!WHY?!!!;
        :type start_end: Tuple[Tuple[int, int], Tuple[int, int]]
        :return: (NZ, K, M1, M2) complex: cropped patchess
        :rtype: torch.Tenosr
        """
        V_rot_partial = self.object[0]
        T_rot_partial = torch.exp(1j * V_rot_partial)
        # NZbin x NY x NX -> NZbin x K x M1 x M2
        T_rot_patches = self.batch_crop(T_rot_partial, probe, pos)
        return T_rot_patches

    # object_patches, positions, probe, propagator, factor
    def get_measurement_model_amplitudes(
        self,
        object_patches: Tensor,
        pos: Tensor,
        dr: Tensor,
        probe: Tensor,
        propagator: Propagator,
        factor: int,
        eps=1e-10,
    ) -> Tensor:
        """

        :param object_patches: (NZ, K, MY, MX) complex
        :param object_patch_normalization: (NZ, K, MY, MX) real
        :param pos: (K, 2) real
        :param probe: (Nmodes, K, MY, MX) complex
        :param propagator: Callable
        :param factor: float
        :return: (K, MY, MX) real
        """

        probe = self.shift(probe, dr, object_patches)
        wave = self.multislice_exitwave_multimode(
            object_patches, probe, pos, propagator
        )
        wave = torch.fft.fft2(wave, norm="ortho", dim=(-2, -1))
        measurements = torch.sum(wave.abs().square(), 0).pow(0.5) + eps
        # K x MY x MX
        measurements = torch.fft.fftshift(
            gaussian_blur(
                torch.fft.fftshift(measurements, dim=(-2, -1)),
                kernel_size=5,
                sigma=1,
            ),
            dim=(-2, -1),
        )
        return measurements

    def forward(
        self,
        probe_index: int,
        angles_index: int,
        r_indices: Iterable[int],
        translation_index: int,
    ) -> Tensor:
        self.current_r_indices = r_indices
        positions = self.positions[r_indices]
        dr = self.dr[r_indices]
        # Get all probe modes for this angles_index
        all_probes = []
        for mode_idx in range(len(self.probe_model)):
            probe_key = str((angles_index, mode_idx))
            if probe_key in self.probe_model:
                all_probes.append(self.probe_model[probe_key])

        # Stack all probe modes together
        if all_probes:
            probe = torch.cat(all_probes, dim=0)  # Combine all modes
        else:
            # Fallback to single probe if no multi-mode setup
            probe = self.probe_model[str((angles_index, probe_index))]

        object_patches = self.get_object_model_patches(
            probe,
            positions,
            None,
            None,
            1,
            debug=False,
        )
        amplitudes_model = self.get_measurement_model_amplitudes(
            object_patches, positions, dr, probe, self.propagator, 1
        )
        self.amplitues_model = amplitudes_model
        return amplitudes_model

    def init_measurements(
        self,
        batch_index: int,
        angles_index: int,
        probe: Tensor,
        positions: Tensor,
        dr: Tensor,
        angle: Tensor | None = None,
        translation: Tensor | None = None,
        bin_factor: int | None = None,
        start_end: Iterable | None = None,
        propagator: Propagator | None = None,
        factor: int = 1,
    ) -> Tensor:
        # NZ x K x MY x MX
        _, _, MY, MX = probe.shape
        K, _ = positions.shape
        object_patches = self.get_object_model_patches(
            probe, positions, angle, translation, bin_factor, start_end
        )
        amplitudes_model = self.get_measurement_model_amplitudes(
            object_patches, positions, dr, probe, self.propagator, 1
        )
        a = amplitudes_model
        if a.requires_grad:
            # import debugpy
            # debugpy.breakpoint()
            a.backward(torch.zeros_like(a))
            exit(1)
        return amplitudes_model
