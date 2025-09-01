from copy import deepcopy as copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import astra
import numpy as np
import torch as th
import torch.nn.functional as F
from kornia.filters import filter3d
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.utils.data import BatchSampler, Dataset, SequentialSampler
from tqdm import tqdm


def affine_matrix_3D_ZYX(phi: Tensor, theta: Tensor, psi: Tensor) -> Tensor:
    c1 = th.cos(phi)
    s1 = th.sin(phi)
    c2 = th.cos(theta)
    s2 = th.sin(theta)
    c3 = th.cos(psi)
    s3 = th.sin(psi)
    zeros = th.zeros_like(phi)
    line1 = th.stack(
        [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2, zeros], 1
    )
    line2 = th.stack(
        [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3, zeros], 1
    )
    line3 = th.stack([-s2, c2 * s3, c2 * c3, zeros], 1)
    R = th.stack([line1, line2, line3], 1)
    return R


def rotate_volume(
    vol: Tensor, phi_rad: Tensor, theta_rad: Tensor, psi_rad: Tensor
) -> Tensor:
    n_theta = phi_rad.shape[0]
    z, y, x = vol.shape
    vol = vol.unsqueeze(0).unsqueeze(0)
    R = affine_matrix_3D_ZYX(phi_rad, theta_rad, psi_rad)
    out_size = (n_theta, 1, z, y, x)
    grid = F.affine_grid(R, out_size, align_corners=False)
    out = F.grid_sample(vol.expand(n_theta, 1, z, y, x), grid, align_corners=False)
    out = out.squeeze()
    return out


def affine_matrix_2D_translate(translations: Tensor) -> Tensor:
    _, n_theta = translations.shape
    ones = th.ones(n_theta, dtype=th.float32, device=translations.device)
    zeros = th.zeros(n_theta, dtype=th.float32, device=translations.device)
    line1 = th.stack([ones, zeros, translations[1]], 1)
    line2 = th.stack([zeros, ones, translations[0]], 1)
    R = th.stack([line1, line2], 1)
    return R


def translate_projections(projections: Tensor, translations: Tensor) -> Tensor:
    n_theta, y, x = projections.shape
    projections = projections.unsqueeze(1)
    scale_factor = th.tensor(
        [2 / y, 2 / x], dtype=th.float32, device=translations.device
    ).reshape(2, 1)
    translations_rel = translations * scale_factor
    R = affine_matrix_2D_translate(translations_rel)
    out_size = (n_theta, 1, y, x)
    grid = F.affine_grid(R, out_size, align_corners=False)
    out = F.grid_sample(projections.expand(n_theta, 1, y, x), grid, align_corners=False)
    out = out.squeeze()
    return out


def torch_ray_transform(
    vol: Tensor,
    phi_rad: Tensor,
    theta_rad: Tensor,
    psi_rad: Tensor,
    translations: Tensor,
) -> Tensor:
    out = rotate_volume(vol, phi_rad, theta_rad, psi_rad)
    sino = th.sum(out, 1)
    sino = translate_projections(sino, translations)
    return sino


def ray_transform(
    vol: Tensor,
    phi_rad: Tensor,
    theta_rad: Tensor,
    psi_rad: Tensor,
    translations: Tensor,
) -> Tensor:
    if translations.requires_grad:
        if phi_rad.requires_grad or theta_rad.requires_grad or psi_rad.requires_grad:
            out = rotate_volume(vol, phi_rad, theta_rad, psi_rad)
            sino = th.sum(out, 1)
        else:
            trans_zeros = th.zeros_like(translations)
            sino = ASTRA_ray_transform(vol, phi_rad, theta_rad, psi_rad, trans_zeros)
        sino = translate_projections(sino, translations)
    else:
        if phi_rad.requires_grad or theta_rad.requires_grad or psi_rad.requires_grad:
            out = rotate_volume(vol, phi_rad, theta_rad, psi_rad)
            sino = th.sum(out, 1)
            sino = translate_projections(sino, translations)
        else:
            sino = ASTRA_ray_transform(vol, phi_rad, theta_rad, psi_rad, translations)
    return sino


def set_projection_vectors(
    phi_rad: Tensor, theta_rad: Tensor, psi_rad: Tensor, translation: Tensor
) -> np.ndarray:
    n_theta = len(theta_rad)
    phi_rad = phi_rad.cpu().numpy()
    theta_rad = theta_rad.cpu().numpy()
    psi_rad = psi_rad.cpu().numpy()
    translation = translation.cpu().numpy()

    c1 = np.cos(phi_rad)
    s1 = np.sin(phi_rad)
    c2 = np.cos(theta_rad)
    s2 = np.sin(theta_rad)
    c3 = np.cos(psi_rad)
    s3 = np.sin(psi_rad)

    line1 = np.stack([c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2], 1)
    line2 = np.stack([c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3], 1)
    line3 = np.stack([-s2, c2 * s3, c2 * c3], 1)
    Rot = np.stack([line1, line2, line3], 1)

    k = np.broadcast_to([0, 0, 1], (n_theta, 3))
    u = np.broadcast_to([1, 0, 0], (n_theta, 3))
    v = np.broadcast_to([0, 1, 0], (n_theta, 3))
    c = np.stack([translation[1], translation[0], np.zeros(n_theta)], 1)

    k = np.einsum("ijk,ik->ij", Rot, k)
    u = np.einsum("ijk,ik->ij", Rot, u)
    v = np.einsum("ijk,ik->ij", Rot, v)
    c = np.einsum("ijk,ik->ij", Rot, c)

    vectors = np.zeros((n_theta, 12))

    vectors[:, 0] = k[:, 0]
    vectors[:, 1] = k[:, 1]
    vectors[:, 2] = k[:, 2]

    vectors[:, 3] = c[:, 0]
    vectors[:, 4] = c[:, 1]
    vectors[:, 5] = c[:, 2]

    vectors[:, 6] = u[:, 0]
    vectors[:, 7] = u[:, 1]
    vectors[:, 8] = u[:, 2]

    vectors[:, 9] = v[:, 0]
    vectors[:, 10] = v[:, 1]
    vectors[:, 11] = v[:, 2]
    return vectors


class autograd_ASTRA_ray_transform(th.autograd.Function):
    @staticmethod
    def forward(
        ctx: th.autograd.function.FunctionCtx,
        vol: Tensor,
        phi_rad: Tensor,
        theta_rad: Tensor,
        psi_rad: Tensor,
        translation: Tensor,
    ) -> Tensor:
        assert vol.is_cuda, "vol should be on CUDA device."
        vol = vol.squeeze().contiguous()
        vol_shape = vol.shape
        device = vol.device

        vol_geom = astra.create_vol_geom(vol_shape[0], vol_shape[1], vol_shape[2])
        vectors = set_projection_vectors(phi_rad, theta_rad, psi_rad, translation)
        proj_geom = astra.create_proj_geom(
            "parallel3d_vec", vol_shape[1], vol_shape[2], vectors
        )

        ctx.vol_geom = vol_geom
        ctx.proj_geom = proj_geom

        proj_arr = th.zeros(astra.geom_size(proj_geom), dtype=th.float32, device=device)

        z, y, x = proj_arr.shape
        pitch = x * proj_arr.element_size()
        proj_link = astra.data3d.GPULink(proj_arr.data_ptr(), x, y, z, pitch)

        z, y, x = vol.shape
        pitch = x * vol.element_size()
        vol_link = astra.data3d.GPULink(vol.data_ptr(), x, y, z, pitch)

        proj_id = astra.data3d.link("-sino", proj_geom, proj_link)
        vol_id = astra.data3d.link("-vol", vol_geom, vol_link)

        cfg = astra.astra_dict("FP3D_CUDA")
        cfg["VolumeDataId"] = vol_id
        cfg["ProjectionDataId"] = proj_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        astra.data3d.delete(vol_id)
        astra.data3d.delete(proj_id)
        astra.algorithm.delete(alg_id)

        sinogram = th.transpose(proj_arr, 1, 0)
        return sinogram.contiguous()

    @staticmethod
    def backward(
        ctx: th.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[Tensor, None, None, None, None]:
        assert grad_output.is_cuda, "grad_output should be on CUDA device."
        device = grad_output.device
        vol_geom = ctx.vol_geom
        proj_geom = ctx.proj_geom

        grad_output = grad_output.squeeze()
        proj_arr = th.transpose(grad_output, 1, 0).contiguous()
        vol = th.zeros(astra.geom_size(vol_geom), dtype=th.float32, device=device)

        z, y, x = proj_arr.shape
        pitch = x * proj_arr.element_size()
        proj_link = astra.data3d.GPULink(proj_arr.data_ptr(), x, y, z, pitch)

        z, y, x = vol.shape
        pitch = x * vol.element_size()
        vol_link = astra.data3d.GPULink(vol.data_ptr(), x, y, z, pitch)

        proj_id = astra.data3d.link("-sino", proj_geom, proj_link)
        vol_id = astra.data3d.link("-vol", vol_geom, vol_link)

        cfg = astra.astra_dict("BP3D_CUDA")
        cfg["ReconstructionDataId"] = vol_id
        cfg["ProjectionDataId"] = proj_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        astra.data3d.delete(vol_id)
        astra.data3d.delete(proj_id)
        astra.algorithm.delete(alg_id)

        return vol.unsqueeze(0).unsqueeze(0), None, None, None, None


ASTRA_ray_transform = autograd_ASTRA_ray_transform.apply


@dataclass
class LinearTomographyInputs:
    dataset: Dataset
    volume: Tensor
    phi: Tensor
    theta: Tensor
    psi: Tensor
    translations: Tensor


@dataclass
class LinearTomographyOptions:
    forward_model: Callable = field(init=False)
    device: Optional[th.device] = None
    num_iterations: int = 100
    batch_size: int = -1
    optimizer: th.optim.Optimizer = th.optim.Adam
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {"lr": 0.001})

    tau: float = 1e-4
    reg_pars: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "algorithm": "FGP_TV",
            "regularisation_parameter": 20e-4,
            "number_of_iterations": 50,
            "tolerance_constant": 1e-06,
            "methodTV": 0,
            "nonneg": 1,
        }
    )
    gauss_kernel: Optional[Tensor] = None
    edge_reg_pars: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "regularisation_parameter": 0,
            "pow": 5,
            "y_squeeze_factor": 1,
        }
    )

    progbar: bool = True
    callback: Optional[Callable] = None

    def __post_init__(self) -> None:
        self.forward_model = ray_transform


@dataclass
class LinearTomographyResults:
    volume: th.Tensor
    sinogram: th.Tensor
    losses: th.Tensor


class LinearTomographySolver:
    def __init__(
        self,
        inputs: LinearTomographyInputs,
        reconstruction_options: LinearTomographyOptions,
    ):
        self._inputs = inputs
        self._options = reconstruction_options

        self.dataset = th.as_tensor(
            self._inputs.dataset, device=self._options.device, dtype=th.float32
        )
        self.volume = th.as_tensor(
            self._inputs.volume, device=self._options.device, dtype=th.float32
        )
        self._inputs.phi = th.as_tensor(
            self._inputs.phi, device=self._options.device, dtype=th.float32
        )
        self._inputs.theta = th.as_tensor(
            self._inputs.theta, device=self._options.device, dtype=th.float32
        )
        self._inputs.psi = th.as_tensor(
            self._inputs.psi, device=self._options.device, dtype=th.float32
        )
        self._inputs.translations = th.as_tensor(
            self._inputs.translations, device=self._options.device, dtype=th.float32
        )

        self.edge_penalty_tensor = self.init_edge_tensor(
            self.volume, self._options.edge_reg_pars
        )
        self.gauss_kernel = None

    def solve(self) -> LinearTomographyResults:
        sino_target = self.dataset
        volume = self.volume
        phi = self._inputs.phi
        theta = self._inputs.theta
        psi = self._inputs.psi
        translations = self._inputs.translations
        # Require gradients Off
        phi.requires_grad = False
        theta.requires_grad = False
        psi.requires_grad = False
        translations.requires_grad = False

        volume.requires_grad = True
        losses = []
        results = LinearTomographyResults(volume, None, losses)

        # Get options
        batch_size = (
            self._options.batch_size
            if self._options.batch_size > 0
            else sino_target.shape[0]
        )
        num_iterations = self._options.num_iterations
        optimization_algorithm = self._options.optimizer
        optimizer_parameters = self._options.optimizer_params
        progbar = self._options.progbar
        forward_model = self._options.forward_model

        edge_reg_pars = self._options.edge_reg_pars
        tau = self._options.tau
        reg_pars = self._options.reg_pars
        gauss_kernel = self._options.gauss_kernel

        optimizer_model = optimization_algorithm([volume], **optimizer_parameters)

        sampler = BatchSampler(
            SequentialSampler(range(sino_target.shape[0])),
            batch_size=batch_size,
            drop_last=False,
        )
        n_batches = len(sampler)
        epochs = tqdm(range(num_iterations)) if progbar else range(num_iterations)

        for epoch in epochs:
            optimizer_model.zero_grad()
            loss = 0

            for batch in sampler:
                sino_sim = forward_model(
                    volume, phi[batch], theta[batch], psi[batch], translations[:, batch]
                )

                current_loss = mse_loss(sino_sim, sino_target[batch])
                current_loss /= n_batches
                current_loss.backward()
                loss += current_loss

            if self.edge_penalty_tensor is not None:
                loss_edges = self.edge_penalty(volume, edge_reg_pars)
                loss_edges.backward()
                loss += loss_edges

            optimizer_model.step()

            losses.append(loss.item())

            with th.no_grad():
                volume[:, :, :] = self.shrink_nonnegative(volume, tau)
                if reg_pars is not None:
                    volume[:, :, :] = self.volume_regularization(
                        volume, reg_pars, self._options.device
                    )
                if gauss_kernel is not None:
                    volume[:, :, :] = filter3d(
                        volume.detach().unsqueeze(0).unsqueeze(0),
                        gauss_kernel.unsqueeze(0),
                    )

            results.sinogram = sino_sim
            with th.no_grad():
                if self._options.callback is not None:
                    self._options.callback(epoch, results)

        return results

    @staticmethod
    def shrink_nonnegative(x: Tensor, tau: float) -> Tensor:
        x = th.abs(x) - tau
        x[th.sign(x) < 0] = 0
        return x

    @staticmethod
    def volume_regularization(
        volume: Tensor, reg_pars: Dict[str, Any], device: th.device
    ) -> Tensor:
        # vol = volume.detach().cpu().squeeze().numpy()
        # (vol, info_vec_gpu) = FGP_TV(
        #     vol,
        #     reg_pars["regularisation_parameter"],
        #     reg_pars["number_of_iterations"],
        #     reg_pars["tolerance_constant"],
        #     reg_pars["methodTV"],
        #     reg_pars["nonneg"],
        #     "gpu",
        # )
        return volume  # th.as_tensor(vol, device=device)

    def init_edge_tensor(
        self, volume: Tensor, edge_reg_pars: Optional[Dict[str, Any]]
    ) -> Optional[Tensor]:
        if edge_reg_pars is None:
            return None
        depth, height, width = volume.shape
        # Create a meshgrid of coordinates for the tensor
        z, y, x = th.meshgrid(th.arange(depth), th.arange(height), th.arange(width))

        # Calculate the Euclidean distance from the center
        k = edge_reg_pars["y_squeeze_factor"]
        center_z, center_y, center_x = depth // 2, height // 2, width // 2
        distance_from_center = th.sqrt(
            th.clamp(
                (z - center_z) ** 2 + k * (y - center_y) ** 2 + (x - center_x) ** 2,
                0.01,
            )
        )
        if th.isnan(distance_from_center).any():
            raise ValueError("The distance_from_center contains NaN values.")

        edge_tensor = distance_from_center.max() - distance_from_center
        edge_tensor = edge_tensor / edge_tensor.max()
        edge_tensor = 1 - edge_tensor
        edge_tensor = edge_tensor / edge_tensor[0, center_y, center_x]
        edge_tensor = th.pow(edge_tensor, edge_reg_pars["pow"])

        edge_tensor_batched = edge_tensor.unsqueeze(0).unsqueeze(0)
        if th.isnan(edge_tensor_batched).any():
            raise ValueError("The edge_tensor_batched contains NaN values.")

        edge_tensor_out = edge_tensor_batched
        edge_tensor_out = edge_tensor_out.to(self._options.device)
        edge_tensor_out.requires_grad = False
        return edge_tensor_out

    def edge_penalty(
        self, volume: Tensor, edge_reg_pars: Optional[Dict[str, Any]]
    ) -> Tensor:
        term = volume * self.edge_penalty_tensor
        num_elements = term.numel()
        reg_term = th.abs(term)
        return (
            th.sum(reg_term) / num_elements * edge_reg_pars["regularisation_parameter"]
        )


@dataclass
class TomographyAlignmentInputs:
    dataset: Dataset
    volume: Tensor
    phi: Tensor
    theta: Tensor
    psi: Tensor
    translations: Tensor


@dataclass
class TomographyAlignmentOptions:
    # Core options
    linearTomographyOptions: LinearTomographyOptions
    num_iterations: int = 100
    optimizer: th.optim.Optimizer = th.optim.Adam
    to_fit: dict = field(
        default_factory=lambda: {
            "phi": True,
            "theta": True,
            "psi": True,
            "translations": True,
        }
    )
    optimizers_params: dict = field(
        default_factory=lambda: {
            "phi": {"lr": 1e-4},
            "theta": {"lr": 1e-4},
            "psi": {"lr": 1e-5},
            "translations": {"lr": 1e-5},
        }
    )

    # Regularization options
    angle_reg_options: dict = field(
        default_factory=lambda: {
            "regularisation_parameter": 1e-3,
            "width_phi": 0.5,  # deg
            "width_theta": 0.25,  # deg
            "width_psi": 0.5,  # deg
        }
    )

    # View
    prog_bar: bool = True
    callback: callable = None


@dataclass
class TomographyAlignmentResults:
    volume: Tensor
    phi: Tensor
    theta: Tensor
    psi: Tensor
    translations: Tensor
    losses: th.Tensor


class TomographyAlignmentSolver:
    def __init__(
        self,
        inputs: TomographyAlignmentInputs,
        reconstruction_options: TomographyAlignmentOptions,
    ):
        self._inputs = inputs
        self._options = copy(reconstruction_options)

        device = self._options.linearTomographyOptions.device
        self.dataset = th.as_tensor(
            self._inputs.dataset, device=device, dtype=th.float32
        )
        self.volume = th.as_tensor(self._inputs.volume, device=device, dtype=th.float32)
        self._inputs.phi = th.as_tensor(
            self._inputs.phi, device=device, dtype=th.float32
        )
        self._inputs.theta = th.as_tensor(
            self._inputs.theta, device=device, dtype=th.float32
        )
        self._inputs.psi = th.as_tensor(
            self._inputs.psi, device=device, dtype=th.float32
        )
        self._inputs.translations = th.as_tensor(
            self._inputs.translations, device=device, dtype=th.float32
        )

    def solve(self) -> TomographyAlignmentResults:
        sino_target = self.dataset
        volume = self.volume
        phi = self._inputs.phi
        theta = self._inputs.theta
        psi = self._inputs.psi
        translations = self._inputs.translations

        losses = []
        results = TomographyAlignmentResults(
            volume, phi, theta, psi, translations, losses
        )

        # Get options
        inner_options = self._options.linearTomographyOptions
        batch_size = (
            inner_options.batch_size
            if inner_options.batch_size > 0
            else sino_target.shape[0]
        )
        num_iterations = self._options.num_iterations
        optimization_algorithm = self._options.optimizer
        optimizers_parameters = self._options.optimizers_params
        to_fit = self._options.to_fit
        angle_reg_options = self._options.angle_reg_options
        progbar = inner_options.progbar
        inner_options.progbar = False

        # Initiate inner solver
        inputs = LinearTomographyInputs(
            self._inputs.dataset, self._inputs.volume, phi, theta, psi, translations
        )
        inner_solver = LinearTomographySolver(inputs, inner_options)

        optimizers = [
            optimization_algorithm([phi], **(optimizers_parameters["phi"])),
            optimization_algorithm([theta], **(optimizers_parameters["theta"])),
            optimization_algorithm([psi], **(optimizers_parameters["psi"])),
            optimization_algorithm(
                [translations], **(optimizers_parameters["translations"])
            ),
        ]

        sampler = BatchSampler(
            SequentialSampler(range(sino_target.shape[0])),
            batch_size=batch_size,
            drop_last=False,
        )
        epochs = tqdm(range(num_iterations)) if progbar else range(num_iterations)

        angle_reg_pars = self.angles_reg_init(angle_reg_options, theta)

        for epoch in epochs:
            volume.requires_grad = False
            volume[:] = 0
            phi.requires_grad = False
            theta.requires_grad = False
            psi.requires_grad = False
            translations.requires_grad = False
            volume.requires_grad = True

            inner_solver.volume = volume
            inner_solver.solve()

            volume.requires_grad = False
            phi.requires_grad = to_fit["phi"]
            theta.requires_grad = to_fit["theta"]
            psi.requires_grad = to_fit["psi"]
            translations.requires_grad = to_fit["translations"]

            for optimizer in optimizers:
                optimizer.zero_grad()

            losses_batches = []

            for batch in sampler:
                sino_sim = ray_transform(
                    volume, phi[batch], theta[batch], psi[batch], translations[:, batch]
                )

                loss = mse_loss(sino_sim, sino_target[batch])
                losses_batches.append(loss.item())
                loss.backward()

            loss_angles = self.angles_regularization_diff(
                phi, theta, psi, angle_reg_pars, angle_reg_options
            )
            if to_fit["phi"] or to_fit["theta"] or to_fit["psi"]:
                loss_angles.backward()
            losses_batches = np.array(losses_batches).sum()
            losses_batches += loss_angles
            losses.append(losses_batches.item())

            for optimizer in optimizers:
                optimizer.step()

            if self._options.callback is not None:
                with th.no_grad():
                    self._options.callback(epoch, results)

        return results

    @staticmethod
    def angles_reg_init(
        angle_reg_pars: Dict[str, Any], theta: Tensor
    ) -> Dict[str, Any]:
        # degree _____________
        width_phi = angle_reg_pars["width_phi"]
        width_theta = angle_reg_pars["width_theta"]
        width_psi = angle_reg_pars["width_psi"]
        # _____________________

        width_phi *= th.pi / 180
        width_theta *= th.pi / 180
        width_psi *= th.pi / 180

        theta_copy = copy(theta)
        angle_reg_pars = {
            "theta": theta_copy,
            "k_phi": (width_phi**2),
            "k_theta": (width_theta**2),
            "k_psi": (width_psi**2),
        }
        return angle_reg_pars

    @staticmethod
    def angles_regularization_diff(
        phi: Tensor,
        theta: Tensor,
        psi: Tensor,
        angle_reg_pars: Dict[str, Any],
        angle_reg_options: Dict[str, Any],
    ) -> Tensor:
        if angle_reg_options["regularisation_parameter"] == 0:
            return th.tensor(0)

        reg_loss_phi = th.mean(th.square(th.diff(phi))) / angle_reg_pars["k_phi"]
        reg_loss_theta = (
            mse_loss(theta, angle_reg_pars["theta"]) / angle_reg_pars["k_theta"]
        )
        reg_loss_psi = th.mean(th.square(th.diff(psi))) / angle_reg_pars["k_psi"]

        reg_loss = th.exp(reg_loss_phi + reg_loss_theta + reg_loss_psi) - 1
        reg_loss *= angle_reg_options["regularisation_parameter"]

        return reg_loss
