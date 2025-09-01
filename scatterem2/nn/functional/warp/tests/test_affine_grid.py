from unittest import TestCase, main

import torch
from torch import Tensor

from scatterem2.nn.functional import (
    affine_grid,
)
from scatterem2.nn.functional import affine_grid_legacy as affine_grid_legacy
from scatterem2.utils import rotate_XZY, rotate_ZYX

# wp.init()


def affine_ZYX(phi: Tensor, theta: Tensor, psi: Tensor) -> Tensor:
    return rotate_ZYX(phi, theta, psi) @ torch.eye(n=3, m=4)


def affine_XZY(phi: Tensor, theta: Tensor, psi: Tensor) -> Tensor:
    return rotate_XZY(phi, theta, psi) @ torch.eye(n=3, m=4)


class TestAffineGridSlice(TestCase):
    def test_with_affine_grid(self) -> None:
        phi, theta, psi = torch.rand(3)
        affine_matrix = affine_ZYX(phi, theta, psi)
        D, H, W = 64, 32, 32
        D_in, H_in, W_in = 64, 64, 64
        ys = torch.arange(H, dtype=torch.int32)
        xs = torch.arange(W, dtype=torch.int32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        patch_grid = torch.stack((grid_y, grid_x), dim=-1).reshape((-1, 2))

        grid_gt = affine_grid_legacy(
            theta_torch=affine_matrix.unsqueeze(0),
            size=(1, 1, D, H, W),
            normalize=False,
            align_corners=False,
            input_size=(D_in, H_in, W_in),
        ).reshape((D, H * W, 3))

        for z in range(D):
            grid_slice = affine_grid(
                yx_grid=patch_grid,
                z_start=z,
                z_len=1,
                theta_torch=affine_matrix,
                input_size=(D_in, H_in, W_in),
                align_corners=False,
            ).squeeze(1)
            difference = torch.abs(grid_slice - grid_gt[z])
            self.assertTrue(difference.max() < 1e-6)

    def test_with_affine_grid_new_convention(self) -> None:
        phi, theta, psi = torch.rand(3)
        D, H, W = 64, 32, 16
        D_in, H_in, W_in = 64, 64, 64
        ys = torch.arange(H, dtype=torch.int32)
        xs = torch.arange(W, dtype=torch.int32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        patch_grid = torch.stack((grid_y, grid_x), dim=-1).reshape((-1, 2))

        grid_gt = affine_grid_legacy(
            theta_torch=affine_ZYX(phi, theta, psi).unsqueeze(0),
            size=(1, 1, W, D, H),
            normalize=False,
            align_corners=False,
            input_size=(W_in, D_in, H_in),
        ).squeeze(0)

        for z in range(D):
            grid_slice = affine_grid(
                yx_grid=patch_grid,
                z_start=z,
                z_len=1,
                theta_torch=affine_XZY(phi, theta, psi),
                input_size=(D_in, H_in, W_in),
                align_corners=False,
            ).squeeze(1)
            gt_slice = grid_gt[:, z, :, :].reshape((H * W, 3))

            grid_slice, _ = torch.sort(grid_slice, 0)
            gt_slice, _ = torch.sort(gt_slice, 0)

            gt_y = gt_slice[:, 0]
            gt_z = gt_slice[:, 1]
            gt_x = gt_slice[:, 2]
            pred_x = grid_slice[:, 0]
            pred_y = grid_slice[:, 1]
            pred_z = grid_slice[:, 2]
            difference_x = torch.abs(pred_x - gt_x)
            self.assertTrue(difference_x.max() < 1e-5)
            difference_y = torch.abs(pred_y - gt_y)
            self.assertTrue(difference_y.max() < 1e-5)
            difference_z = torch.abs(pred_z - gt_z)
            self.assertTrue(difference_z.max() < 1e-5)

    def test_multiple_slices(self) -> None:
        phi, theta, psi = torch.rand(3)
        D, H, W = 64, 32, 16
        D_in, H_in, W_in = 64, 64, 64
        ys = torch.arange(H, dtype=torch.int32)
        xs = torch.arange(W, dtype=torch.int32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        patch_grid = torch.stack((grid_y, grid_x), dim=-1).reshape((-1, 2))

        grid_gt = affine_grid_legacy(
            theta_torch=affine_ZYX(phi, theta, psi).unsqueeze(0),
            size=(1, 1, W, D, H),
            normalize=False,
            align_corners=False,
            input_size=(W_in, D_in, H_in),
        ).squeeze(0)
        grid_slice = affine_grid(
            yx_grid=patch_grid,
            z_start=0,
            z_len=D,
            theta_torch=affine_XZY(phi, theta, psi),
            input_size=(D_in, H_in, W_in),
            align_corners=False,
        ).reshape((-1, 3))
        gt_slice = grid_gt.reshape((-1, 3))

        grid_slice, _ = torch.sort(grid_slice, 0)
        gt_slice, _ = torch.sort(gt_slice, 0)

        gt_y = gt_slice[:, 0]
        gt_z = gt_slice[:, 1]
        gt_x = gt_slice[:, 2]
        pred_x = grid_slice[:, 0]
        pred_y = grid_slice[:, 1]
        pred_z = grid_slice[:, 2]
        difference_x = torch.abs(pred_x - gt_x)
        self.assertTrue(difference_x.max() < 1e-5)
        difference_y = torch.abs(pred_y - gt_y)
        self.assertTrue(difference_y.max() < 1e-5)
        difference_z = torch.abs(pred_z - gt_z)
        self.assertTrue(difference_z.max() < 1e-5)


if __name__ == "__main__":
    main()
