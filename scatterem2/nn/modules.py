from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import Size, Tensor
from torch.nn import Module, Parameter

from scatterem2.nn.functional import (
    batch_crop,
    bin_object,
    complex_mul_multi_mode_function,
    overlap_intensity,
)
from scatterem2.nn.functional.scalers import l2_scaling_function
from scatterem2.utils import fftfreq2, fftshift_checkerboard, get_probe_size
from scatterem2.utils.data import Aberrations, SMeta


class BatchCrop(Module):
    def __init__(self, obj: Tensor) -> None:
        super().__init__()
        self.object_norm = torch.zeros(
            obj.shape[1:], device=obj.device, dtype=torch.float32, requires_grad=False
        )

    def forward(self, obj: Tensor, waves: Tensor, pos: Tensor) -> Tensor:
        patch_norm = torch.sum(torch.abs(waves.detach()) ** 2, 0).squeeze()
        # calculate normalization for object gradient
        overlap_intensity(pos.data, patch_norm.data, self.object_norm.data)
        return batch_crop(obj, waves, pos)


class Bin(Module):
    def __init__(self, factor: float) -> None:
        self.factor = factor
        super(Bin, self).__init__()

    def forward(self, obj_in: Tensor) -> Tensor:
        return bin_object(obj_in, self.factor)


class Propagator(Module):
    def __init__(
        self,
        slice_thickness: float,
        wavelength: float,
        dx: float,
        shape: Tuple | Size,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        q = fftfreq2(shape, dx, False, device=device)
        q2 = torch.linalg.norm(q, axis=0) ** 2
        self.prop = torch.exp((-1j * torch.pi * wavelength * slice_thickness) * q2)

    def forward(self, waves: Tensor) -> Tensor:
        """

        :param waves: Nmodes      x K x M1 x M2
        :return:      Nmodes      x K x M1 x M2
        """
        waves = torch.fft.fft2(waves)
        waves *= self.prop
        waves = torch.fft.ifft2(waves)
        return waves


# TODO: check annotation!
def _cartesian_aberrations(qx: Tensor, qy: Tensor, lam: Tensor, C: Tensor) -> Tensor:
    """
    Zernike polynomials in the cartesian coordinate system
    """

    u = qx * lam
    v = qy * lam
    u2 = u**2
    u3 = u**3
    u4 = u**4
    # u5 = u ** 5

    v2 = v**2
    v3 = v**3
    v4 = v**4
    # v5 = v ** 5

    C1 = C[0]
    C12a = C[1]
    C12b = C[2]
    C21a = C[3]
    C21b = C[4]
    C23a = C[5]
    C23b = C[6]
    C3 = C[7]
    C32a = C[8]
    C32b = C[9]
    C34a = C[10]
    C34b = C[11]

    chi = 0

    # r-2 = x-2 +y-2.
    chi += 1 / 2 * C1 * (u2 + v2)  # r^2
    # r-2 cos(2*phi) = x"2 -y-2.
    # r-2 sin(2*phi) = 2*x*y.
    chi += (
        1 / 2 * (C12a * (u2 - v2) + 2 * C12b * u * v)
    )  # r^2 cos(2 phi) + r^2 sin(2 phi)
    # r-3 cos(3*phi) = x-3 -3*x*y'2. r"3 sin(3*phi) = 3*y*x-2 -y-3.
    chi += (
        1 / 3 * (C23a * (u3 - 3 * u * v2) + C23b * (3 * u2 * v - v3))
    )  # r^3 cos(3phi) + r^3 sin(3 phi)
    # r-3 cos(phi) = x-3 +x*y-2.
    # r-3 sin(phi) = y*x-2 +y-3.
    chi += (
        1 / 3 * (C21a * (u3 + u * v2) + C21b * (v3 + u2 * v))
    )  # r^3 cos(phi) + r^3 sin(phi)
    # r-4 = x-4 +2*x-2*y-2 +y-4.
    chi += 1 / 4 * C3 * (u4 + v4 + 2 * u2 * v2)  # r^4
    # r-4 cos(4*phi) = x-4 -6*x-2*y-2 +y-4.
    chi += 1 / 4 * C34a * (u4 - 6 * u2 * v2 + v4)  # r^4 cos(4 phi)
    # r-4 sin(4*phi) = 4*x-3*y -4*x*y-3.
    chi += 1 / 4 * C34b * (4 * u3 * v - 4 * u * v3)  # r^4 sin(4 phi)
    # r-4 cos(2*phi) = x-4 -y-4.
    chi += 1 / 4 * C32a * (u4 - v4)
    # r-4 sin(2*phi) = 2*x-3*y +2*x*y-3.
    chi += 1 / 4 * C32b * (2 * u3 * v + 2 * u * v3)
    # r-5 cos(phi) = x-5 +2*x-3*y-2 +x*y-4.
    # r-5 sin(phi) = y*x"4 +2*x-2*y-3 +y-5.
    # r-5 cos(3*phi) = x-5 -2*x-3*y-2 -3*x*y-4.
    # r-5 sin(3*phi) = 3*y*x-4 +2*x-2*y-3 -y-5.
    # r-5 cos(5*phi) = x-5 -10*x-3*y-2 +5*x*y-4.
    # r-5 sin(5*phi) = 5*y*x-4 -10*x-2*y-3 +y-5.

    chi *= 2 * torch.pi / lam

    return chi


class ZernikeProbe(Module):
    def __init__(
        self,
        wavelength: float,
        sampling: Tuple[float, float],
        aperture_array: Tensor,
        aberrations: Aberrations,
        fourier_space: bool = True,
        make_beamlet_meta: bool = True,
        n_radial_samples: int = 4,
        n_angular_samples: int = 6,
        make_plane_waves: bool = False,
        fft_shifted: bool = True,
        # Mixed probe parameters
        mixed_probe: bool = False,
        pmodes: int = 1,
        pmode_init_pows: List[float] = None,
        measurement_norm: Optional[float] = 1,
    ) -> None:
        """
        Creates an aberration surface from aberration coefficients. The output is backpropable

        :param q: 2 x M1 x M2 tensor of x coefficients of reciprocal space
        :param lam: wavelength in Angstrom
        :param C: aberration coefficients
        :param mixed_probe: Whether to generate mixed probe modes
        :param pmodes: Number of incoherent probe modes
        :param pmode_init_pows: Integrated intensity of modes. List of values or single value.
                               sum(pmode_init_pows) must < 1.
        :return: (maximum size of all aberration tensors) x MY x MX
        """

        super(ZernikeProbe, self).__init__()

        self.wavelength = wavelength
        self.sampling = sampling
        self.fft_shifted = fft_shifted
        self.aperture_array = aperture_array
        self.aberrations = aberrations.array
        self.fourier_space = fourier_space
        self.MY, self.MX = self.aperture_array.shape
        self.device = self.aperture_array.device
        self.measurement_norm = measurement_norm

        # Mixed probe parameters
        self.mixed_probe = mixed_probe
        self.pmodes = pmodes if mixed_probe else 1
        self.pmode_init_pows = (
            pmode_init_pows if pmode_init_pows is not None else [0.25]
        )

        self.q = fftfreq2(
            self.aperture_array.shape, self.sampling, False, device=self.device
        )

        if self.fft_shifted:
            cb = fftshift_checkerboard(self.q.shape[1], self.q.shape[2])
            self.cb = torch.as_tensor(cb, dtype=torch.float32, device=self.device)
        self.take_beams = self.aperture_array > self.aperture_array.max() * 0.2
        self.B = self.take_beams.sum()
        self.beam_numbers = (
            torch.ones_like(self.take_beams, dtype=torch.long, device=self.device) * -1
        )
        self.beam_numbers[self.take_beams] = torch.arange(0, self.B, device=self.device)

        if make_beamlet_meta:
            numerical_aperture_radius_pixels = get_probe_size(
                self.aperture_array, thresh_lower=0.1, thresh_upper=0.99, N=100
            )
            print(
                f"ZernikeProbe2: Making beamlet meta with numerical aperture radius {numerical_aperture_radius_pixels} pixels"
            )
            print(f"ZernikeProbe2: Taking {self.B} beams.")
            s_meta = SMeta(
                self.take_beams,
                self.take_beams,
                numerical_aperture_radius_pixels,
                self.sampling,
                self.aperture_array.shape,
                self.aperture_array.device,
            )
            self.s_meta = s_meta.make_beamlet_meta(n_radial_samples, n_angular_samples)

        if make_plane_waves:
            self._plane_waves = torch.zeros(
                (self.B, *self.aperture_array.shape),
                device=self.device,
                dtype=torch.complex64,
            )

            q = fftfreq2(
                self.aperture_array.shape, self.sampling, False, device=self.device
            )
            q_dft = torch.as_tensor(
                q, dtype=self.aperture_array.dtype, device=self.device
            )
            coords = torch.as_tensor(
                torch.fft.fftshift(
                    Tensor(
                        torch.mgrid[
                            -self.MY // 2 : self.MY // 2, -self.MX // 2 : self.MX // 2
                        ]
                    ),
                    (1, 2),
                ),
                dtype=self.aperture_array.dtype,
                device=self.device,
            )
            for b in range(self.B):
                cur_beam = self.beam_numbers == b
                cur_beam = cur_beam[None, ...].expand_as(coords)
                c = coords[cur_beam]
                cur_planewave = torch.exp(
                    2j * np.pi * (q_dft[0] * c[0] + q_dft[1] * c[1])
                )
                cur_planewave /= torch.linalg.norm(cur_planewave)
                self._plane_waves[b] = cur_planewave

        # Initialize base probe
        fw = self.forward_single()
        fw_abs = torch.abs(fw)
        fw_abs_masked = fw_abs > 0.1 * fw_abs.max()
        mean_abs = fw_abs[fw_abs_masked].mean()
        f = 1 / mean_abs
        print(f"ZernikeProbe2: Scaling probe by {f}")
        print(f"ZernikeProbe2: before scaling mean_abs: {mean_abs}")
        self.aperture_array = self.aperture_array * f
        fw = self.forward_single()
        fw_abs = torch.abs(fw)
        fw_abs_masked = fw_abs > 0.1 * fw_abs.max()
        mean_abs = fw_abs[fw_abs_masked].mean()
        print(f"ZernikeProbe2: after scaling mean_abs: {mean_abs}")

        # Generate mixed probe modes if requested
        if self.mixed_probe:
            self._generate_mixed_probe()

    def forward_single(self) -> Tensor:
        """Generate single probe mode"""
        chi = _cartesian_aberrations(
            self.q[1], self.q[0], self.wavelength, self.aberrations
        )
        Psi = torch.exp(-1j * chi) * self.aperture_array.expand_as(chi)

        if self.fft_shifted:
            Psi = Psi * self.cb
        if not self.fourier_space:
            Psi = torch.fft.ifft2(Psi, norm="ortho")
        return Psi

    def _generate_mixed_probe(self):
        """Generate mixed probe modes using Hermite-like basis functions"""
        # Get the base probe
        base_probe = self.forward_single()

        # Convert to numpy for processing
        base_probe_np = base_probe.detach().cpu().numpy()
        # need to scale the probe
        base_probe_np = base_probe_np / np.sqrt(np.sum((np.abs(base_probe_np)) ** 2))
        # Generate mixed probe modes
        mixed_probe_np = self.make_mixed_probe(
            base_probe_np, self.pmodes, self.pmode_init_pows
        )
        # Normalize mixed probe modes
        if self.measurement_norm:
            normalization_factor = (
                np.sum(np.abs(mixed_probe_np) ** 2) / self.measurement_norm
            ) ** 0.5
            mixed_probe_np = mixed_probe_np / normalization_factor

        self.mixed_probe_modes = torch.as_tensor(
            mixed_probe_np, dtype=torch.complex64, device=self.device
        )  # * torch.exp(torch.as_tensor(1j * torch.pi))

        print(f"ZernikeProbe2: Generated {self.pmodes} mixed probe modes")

    def forward(self) -> Tensor:
        """Forward pass - returns mixed probe modes if enabled, single mode otherwise"""
        if self.mixed_probe:
            return self.mixed_probe_modes[:, None, :, :]
        else:
            return self.forward_single()[None, None, :, :]

    def hermite_like(self, fundam, M, N):
        """
        Generate Hermite-like basis functions from fundamental probe
        """
        M = int(M)
        N = int(N)
        m = np.arange(M + 1)
        n = np.arange(N + 1)
        H = np.zeros(
            ((M + 1) * (N + 1), fundam.shape[-2], fundam.shape[-1]), dtype=fundam.dtype
        )

        # Create meshgrid
        rows, cols = fundam.shape[-2:]
        x = np.arange(cols) - cols / 2
        y = np.arange(rows) - rows / 2
        X, Y = np.meshgrid(x, y)

        # Calculate centroid and variance
        intensity = np.abs(fundam) ** 2
        total_intensity = np.sum(intensity)
        cenx = np.sum(X * intensity) / total_intensity
        ceny = np.sum(Y * intensity) / total_intensity
        varx = np.sum((X - cenx) ** 2 * intensity) / total_intensity
        vary = np.sum((Y - ceny) ** 2 * intensity) / total_intensity

        counter = 0
        # Create basis
        for nii in n:
            for mii in m:
                auxfunc = ((X - cenx) ** mii) * ((Y - ceny) ** nii) * fundam

                if counter == 0:
                    auxfunc = auxfunc / np.sqrt(np.sum(np.abs(auxfunc) ** 2))
                else:
                    auxfunc = auxfunc * np.exp(
                        -((X - cenx) ** 2 / (2 * varx)) - ((Y - ceny) ** 2 / (2 * vary))
                    )
                    auxfunc = auxfunc / np.sqrt(np.sum(np.abs(auxfunc) ** 2))

                    # Make it orthogonal to the previous ones
                    for ii in range(counter):
                        auxfunc = (
                            auxfunc
                            - np.dot(H[ii].reshape(-1), np.conj(auxfunc).reshape(-1))
                            * H[ii]
                        )

                # Normalize
                auxfunc = auxfunc / np.sqrt(np.sum(np.abs(auxfunc) ** 2))
                H[counter] = auxfunc
                counter += 1

        return H

    def make_mixed_probe(self, probe, pmodes, pmode_init_pows):
        """Make a mixed state probe from a single state probe"""
        # Input:
        #   probe: (Ny,Nx) complex array
        #   pmodes: number of incoherent probe modes, scaler int
        #   pmode_init_pows: Integrated intensity of modes. List of a value (e.g. [0.02]) or a couple values for the first few modes. sum(pmode_init_pows) must < 1.
        # Output:
        #   mixed_probe: A mixed state probe with (pmode,Ny,Nx)

        # Prepare a mixed-state probe `mixed_probe`
        print(
            f"Start making mixed-state STEM probe with {pmodes} incoherent probe modes"
        )
        M = np.ceil(pmodes**0.5) - 1
        N = np.ceil(pmodes / (M + 1)) - 1
        mixed_probe = self.hermite_like(probe, M, N)[:pmodes]

        # Normalize each pmode
        pmode_pows = np.zeros(pmodes)
        for ii in range(pmodes):
            if ii < np.size(pmode_init_pows):
                pmode_pows[ii] = pmode_init_pows[ii]
            else:
                pmode_pows[ii] = pmode_init_pows[-1]
        if sum(pmode_pows) > 1:
            raise ValueError("Modes total power exceeds 1, check pmode_init_pows")
        # else:
        #     pmode_pows[0] = 1 - sum(pmode_pows)

        mixed_probe = mixed_probe * np.sqrt(pmode_pows)[:, None, None]
        print(f"Relative power of probe modes = {pmode_pows}")
        return mixed_probe

    def forward_with_beamlets(self) -> Tensor:
        if not self.fourier_space:
            raise ValueError(
                "ZernikeProbe2: forward_with_beamlets only works in Fourier space"
            )
        Psi = self.forward()
        if self.mixed_probe:
            # Handle mixed probe modes with beamlets
            Psi_beamlets = self.s_meta.beamlets[None, :, :, :] * Psi[:, None, :, :]
            return Psi_beamlets
        else:
            Psi_beamlets = self.s_meta.beamlets * Psi[None, ...]
            return Psi_beamlets

    def plane_waves(self, indices: Iterable[int], fy: int, fx: int) -> Tensor:
        return self._plane_waves[indices].repeat((1, fy, fx))

    def get_probe_modes(self) -> Tensor:
        """Get the current probe modes (mixed or single)"""
        return self.forward()

    def get_mode_powers(self) -> Tensor:
        """Get the power of each probe mode"""
        if self.mixed_probe:
            modes = self.mixed_probe_modes
            powers = torch.sum(torch.abs(modes) ** 2, dim=(-2, -1))
            return powers / torch.sum(powers)
        else:
            return torch.tensor([1.0], device=self.device)

    # def phase_factors(self, r):
    #     if not self.fourier_space:
    #         raise ValueError('ZernikeProbe2: phase_factors only works in Fourier space')
    #     K, _ = r.shape
    #     qx, qy = np.meshgrid(fftfreq(self.MY), fftfreq(self.MX))
    #     q = Tensor([qy, qx], device=self.device)
    #     Psi = self.forward()
    #     Psi_B = Psi[0,0,self.take_beams].reshape(self.B)
    #     tb = self.take_beams[None, ...].expand(*q.shape)
    #     qB = q[tb].reshape(2, self.B)
    #     out = torch.zeros((K, self.B, 2), dtype=torch.float32, device=Psi.device)
    #     phase_factor_kernelKB(torch.view_as_real(Psi_B), r, qB, out)
    #     return torch.view_as_complex(out)


# def smatrix_phase_factorsKB(Psi, r, take_beams, q, B, out=None):
#     """
#     Abbreviations:
#     B: number of (input) beams in S-matrix
#     K: number of scan positions
#     MY/MX: detector shape
#     NY/NX: S-matrix shape

#     :param Psi: q           MY x MX
#     :param r:               K x 2
#     :param take_beams:      MY x MX
#     :param q:               2 x MY x MX
#     :param out:             K x B x 2
#     :return:
#     """
#     if out is None:
#         K, _ = r.shape
#         out = torch.zeros((K, B, 2), dtype=torch.float32, device=Psi.device)
#     else:
#         out[:] = 0
#         K, B, c = out.shape
#     Psi_B = Psi[take_beams].reshape(B)
#     tb = take_beams[None, ...].expand(*q.shape)
#     qB = q[tb].reshape(2, B)
#     phase_factor_kernelKB(torch.view_as_real(Psi_B), r, qB, out)
#     return torch.view_as_complex(out)


# @ti.kernel
# def phase_factor_kernelKB(
#     Psi: ti.types.ndarray(ndim=2),
#     rho: ti.types.ndarray(ndim=2),
#     qB: ti.types.ndarray(ndim=2),
#     out: ti.types.ndarray(ndim=3),
# ):
#     """
#     Calculate the phase factors (due to beam scan) probe wave function so that
#     the probe is scanned to the correct place for each diffraction pattern

#     :param Psi:         B x 2
#         Probe wave functions Fourier coefficient for each beam to be mutliplied
#         by phase factor to account for beam scan position
#     :param rho:         K x 2
#         Probe positions in pixels
#     :param qB:          2 x B
#         Fourier space coordinates of the beams
#     :param out:         K x B x 2
#         Phase factors output
#     :return: scanning phases for all defoc, beams, positions
#     """
#     ti.loop_config(parallelize=8, block_dim=256)
#     for k, b, i in out:
#         rho0 = rho[k, 0]
#         rho1 = rho[k, 1]
#         Psic = tm.vec2(Psi[b, 0], Psi[b, 1])
#         # scanning phase with subpixel precision
#         z = tm.vec2(0, -2 * tm.pi * (qB[0, b] * rho0 + qB[1, b] * rho1))
#         v = tm.cmul(tm.cexp(z), Psic)
#         out[k, b, 0] = v[0]
#         out[k, b, 1] = v[1]

# def smatrix_phase_factorsKB(Psi, r, take_beams, q, B, out=None):
#     """
#     Abbreviations:
#     B: number of (input) beams in S-matrix
#     K: number of scan positions
#     MY/MX: detector shape
#     NY/NX: S-matrix shape

#     :param Psi: q           MY x MX
#     :param r:               K x 2
#     :param take_beams:      MY x MX
#     :param q:               2 x MY x MX
#     :param out:             K x B x 2
#     :return:
#     """
#     if out is None:
#         K, _ = r.shape
#         out = torch.zeros((K, B, 2), dtype=torch.float32, device=Psi.device)
#     else:
#         out[:] = 0
#         K, B, c = out.shape
#     Psi_B = Psi[take_beams].reshape(B)
#     tb = take_beams[None, ...].expand(*q.shape)
#     qB = q[tb].reshape(2, B)
#     phase_factor_kernelKB(torch.view_as_real(Psi_B), r, qB, out)
#     return torch.view_as_complex(out)


# @ti.kernel
# def phase_factor_kernelKB(
#     Psi: ti.types.ndarray(ndim=2),
#     rho: ti.types.ndarray(ndim=2),
#     qB: ti.types.ndarray(ndim=2),
#     out: ti.types.ndarray(ndim=3),
# ):
#     """
#     Calculate the phase factors (due to beam scan) probe wave function so that
#     the probe is scanned to the correct place for each diffraction pattern

#     :param Psi:         B x 2
#         Probe wave functions Fourier coefficient for each beam to be mutliplied
#         by phase factor to account for beam scan position
#     :param rho:         K x 2
#         Probe positions in pixels
#     :param qB:          2 x B
#         Fourier space coordinates of the beams
#     :param out:         K x B x 2
#         Phase factors output
#     :return: scanning phases for all defoc, beams, positions
#     """
#     ti.loop_config(parallelize=8, block_dim=256)
#     for k, b, i in out:
#         rho0 = rho[k, 0]
#         rho1 = rho[k, 1]
#         Psic = tm.vec2(Psi[b, 0], Psi[b, 1])
#         # scanning phase with subpixel precision
#         z = tm.vec2(0, -2 * tm.pi * (qB[0, b] * rho0 + qB[1, b] * rho1))
#         v = tm.cmul(tm.cexp(z), Psic)
#         out[k, b, 0] = v[0]
#         out[k, b, 1] = v[1]


class SubpixShift(Module):
    def __init__(
        self, probe_shape: Tuple | Size, device: str, subpix_shift: bool
    ) -> None:
        super().__init__()
        a, b, MY, MX = probe_shape
        q = fftfreq2((MY, MX), [1, 1], False, device=device)
        self.qqy = q[0][None, None, ...]
        self.qqx = q[1][None, None, ...]
        self.subpix_shift = subpix_shift
        self.probe_norm = torch.zeros(
            probe_shape, device=device, dtype=torch.float32, requires_grad=False
        )

    def forward(self, waves: Tensor, dr: Tensor, object_patches: List) -> Tensor:
        """
        rs: K x 2
        w : probe Nmodes x K x MY x MX
        """
        # ramp is shape 1 x K x MY x MX
        if self.subpix_shift:
            rampx = self.qqx * dr[:, 1][:, None, None]
            rampy = self.qqy * dr[:, 0][:, None, None]
            ramp = torch.exp(-2j * torch.pi * (rampx + rampy))
            Psi = torch.fft.fft2(waves)
            # B x K x MY x MX
            # print(ramp.shape,Psi.shape)
            if len(Psi.shape) < 4:
                Psi = Psi.unsqueeze(1)
                Psi = Psi.repeat(1, ramp.shape[1], 1, 1)
            # B x K x MY x MX
            w = torch.fft.ifft2(Psi * ramp)
        else:
            ps = waves.shape
            w = waves.expand((ps[0], dr.shape[0], ps[2], ps[3]))
        # if self.subpix_shift and waves.requires_grad:
        #     # sum over positions
        #     self.probe_norm[:] = torch.sum(
        #         torch.abs(object_patches[[0]].detach()) ** 2, 1, keepdim=True
        #     )
        return w


class ComplexMulMultiMode(Module):
    def forward(
        self, object_patches: torch.Tensor, waves: torch.Tensor
    ) -> torch.Tensor:
        return complex_mul_multi_mode_function(object_patches, waves)


class L2ScalingLayer(Module):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.scaling_parameter = Parameter(
            torch.ones(1, device=device), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaling_parameter = torch.relu(self.scaling_parameter)
        return l2_scaling_function(x, scaling_parameter)
