from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from numpy.fft import fftshift
from numpy.typing import ArrayLike, NDArray
from torch import Tensor
from torch import device as torch_device

from scatterem2.utils import beamlet_samples, fftfreq2, natural_neighbor_weights


@dataclass
class SMeta:
    f: NDArray[np.float32]
    M: NDArray[np.int32]
    N: NDArray[np.int32]
    dx: NDArray[np.float32]

    q: Tensor
    qf: Tensor
    q2: Tensor
    qf2: Tensor
    q_coords: Tensor
    r_indices: Tensor
    all_beams: Tensor
    parent_beams: Tensor
    beam_numbers: Tensor
    all_beams_q: Tensor
    all_beams_coords: Tensor
    parent_beams_q: Tensor
    parent_beams_coords: Tensor
    q_dft: Tensor

    numerical_aperture_radius_pixels: int
    B: int
    Bp: int

    device: torch_device
    natural_neighbor_weights: Optional[Tensor] = None
    beamlets: Optional[Tensor] = None

    def __init__(
        self,
        all_beams: ArrayLike,
        parent_beams: ArrayLike,
        numerical_aperture_radius_pixels: int,
        dx: ArrayLike,
        M: Tuple[int, int],
        device: torch_device,
    ) -> None:
        self.dx = dx
        self.device = device
        self.all_beams = torch.as_tensor(all_beams).to(device).bool()
        self.parent_beams = torch.as_tensor(parent_beams).to(device).bool()
        self.numerical_aperture_radius_pixels = numerical_aperture_radius_pixels
        self.B = int(all_beams.sum())
        self.Bp = int(parent_beams.sum())

        MY, MX = M
        self.M = np.array(M)

        self.qf = fftfreq2([MY, MX], dx, centered=False, device=device)

        self.q2 = torch.norm(self.q, dim=0) ** 2
        self.qf2 = torch.norm(self.qf, dim=0) ** 2
        mgrid = np.array(np.mgrid[-MY // 2 : MY // 2, -MX // 2 : MX // 2])
        self.q_coords = torch.from_numpy(fftshift(mgrid, (1, 2))).to(device)
        self.r_indices = torch.from_numpy(np.mgrid[:MY, :MX]).to(device)

        self.beam_numbers = (
            torch.ones_like(self.all_beams, dtype=torch.long, device=device) * -1
        )
        self.beam_numbers[self.all_beams] = torch.arange(0, self.B, device=device)

        all_beams_expanded = self.all_beams[None, ...].expand_as(self.q_coords)
        self.all_beams_q = self.qf[all_beams_expanded].reshape(2, self.B).T
        self.all_beams_coords = self.q_coords[all_beams_expanded].reshape(2, self.B).T

        parent_beams_expanded = self.parent_beams[None, ...].expand_as(self.q_coords)
        self.parent_beams_q = self.qf[parent_beams_expanded].reshape(2, self.Bp).T
        self.parent_beams_coords = (
            self.q_coords[parent_beams_expanded].reshape(2, self.Bp).T
        )

    def to_h5(self, file_path: str, key: str = "s_matrix_meta") -> None:
        with h5py.File(file_path, "a") as f:
            g = f.create_group(key)
            g.create_dataset("M", data=self.M)
            g.create_dataset("N", data=self.N)
            g.create_dataset("all_beams", data=self.all_beams.cpu().numpy())
            g.create_dataset("parent_beams", data=self.parent_beams.cpu().numpy())
            g.create_dataset(
                "numerical_aperture_radius_pixels",
                data=self.numerical_aperture_radius_pixels,
            )
            g.create_dataset("dx", data=self.dx)
            g.create_dataset("device", data=str(self.device))

    @staticmethod
    def from_h5(file_path: str, key: str = "s_matrix_meta") -> "SMeta":
        with h5py.File(file_path, "r") as f:
            g = f[key]
            all_beams = g["all_beams"][...]
            parent_beams = g["parent_beams"][...]
            numerical_aperture_radius_pixels = g["numerical_aperture_radius_pixels"][()]
            dx = g["dx"][...]
            N = g["N"][...]
            M = g["M"][...]
            device = torch.device(str(g["device"][...]))
        res = SMeta(
            all_beams, parent_beams, numerical_aperture_radius_pixels, dx, N, M, device
        )
        return res

    def make_beamlet_meta(
        self, n_radial_samples: int, n_angular_samples: int = 6
    ) -> "SMeta":
        parent_beams_coords = beamlet_samples(
            self.all_beams.cpu().numpy(),
            self.numerical_aperture_radius_pixels,
            n_angular_samples,
            n_radial_samples,
        )

        parent_beams = torch.zeros(tuple(self.M), dtype=torch.bool)
        for si in parent_beams_coords:
            parent_beams[si[0], si[1]] = 1

        Bp = parent_beams_coords.shape[0]
        mgrid = np.array(
            np.mgrid[-self.M[0] // 2 : self.M[0] // 2, -self.M[1] // 2 : self.M[1] // 2]
        )
        q_coords = torch.from_numpy(fftshift(mgrid, (1, 2)))
        parent_beams_expanded = parent_beams[None, ...].expand_as(q_coords)
        parent_beams_coords = q_coords[parent_beams_expanded].reshape(2, Bp).T

        if n_radial_samples > 1:
            nnw = natural_neighbor_weights(
                parent_beams_coords,
                self.all_beams_coords.cpu().numpy(),
                minimum_weight_cutoff=1e-2,
            )
        else:
            nnw = np.ones((self.B, 1))

        beamlets = []
        for j in range(nnw.shape[1]):
            wsample = nnw[:, j]
            ww = np.zeros(self.all_beams.shape, dtype=np.float32)
            ww[
                self.all_beams_coords[:, 1].cpu().numpy(),
                self.all_beams_coords[:, 0].cpu().numpy(),
            ] = wsample
            beamlets.append(ww)
        beamlets_array = np.array(beamlets)

        new_meta = SMeta(
            self.all_beams,
            parent_beams,
            self.numerical_aperture_radius_pixels,
            self.dx,
            tuple(self.N),
            tuple(self.M),
            self.device,
        )
        new_meta.natural_neighbor_weights = torch.as_tensor(nnw)
        new_meta.beamlets = torch.as_tensor(beamlets_array).to(new_meta.device)
        return new_meta
