from unittest import TestCase, main

import torch

from scatterem2.nn.functional import (
    affine_grid_legacy,
    grid_sample,
    grid_sample_legacy,
)
from scatterem2.utils import batch_unique_with_inverse, rotate_ZYX


def _get_translation_matrix_3D(yx_translations: torch.Tensor) -> torch.Tensor:
    n_translations = yx_translations.size(0)
    res = torch.eye(n=3, m=4, device=yx_translations.device).unsqueeze(0)
    res = res.repeat((n_translations, 1, 1))
    res[:, 1, -1] = yx_translations[:, 0]
    res[:, 0, -1] = yx_translations[:, 1]
    return res


class TestAffineGridSlice(TestCase):
    def test_patches(self) -> None:
        phi, theta, psi = torch.rand(3)
        D, H, W = 64, 32, 32
        D_in, H_in, W_in = 64, 64, 64

        potential_zyx = torch.randn((D_in, H_in, W_in), dtype=torch.complex64)
        # potential_xzy = potential_zyx.swapaxes(1, 2).swapaxes(0, 1)

        positions = torch.tensor(
            [
                [0, 0],
                [0, 3],
                [0, 4],
                [4, 0],
                [1, 1],
                [4, 4],
            ]
        )
        batch_size = positions.shape[0]

        # ===conventional potential computation===
        rotation = rotate_ZYX(phi, theta, psi).unsqueeze(0)
        rotation = rotation.repeat(batch_size, 1, 1)
        translations = _get_translation_matrix_3D(positions)

        transform = torch.bmm(rotation, translations)

        grid = affine_grid_legacy(
            theta_torch=transform,
            size=(batch_size, 1, D, H, W),
            normalize=False,
            align_corners=False,
            input_size=(D_in, H_in, W_in),
        )

        sampled_potential_gt = grid_sample_legacy(
            input_torch=potential_zyx.unsqueeze(0),
            grid_torch=grid,
            unnormalize=False,
            align_corners=False,
        ).squeeze(
            1
        )  # [N, H, D, W]

        # === sliced potential computation ===
        yx_grid, inverse_index = batch_unique_with_inverse(
            positions=positions, patch_shape=(H, W)
        )

        for z in range(D):
            grid_slice = affine_grid_legacy(
                yx_grid=yx_grid,
                z_start=z,
                z_len=1,
                theta_torch=rotate_ZYX(phi, theta, psi) @ torch.eye(n=3, m=4),
                input_size=(D_in, H_in, W_in),
                align_corners=False,
            )
            sampled_potential_slice_gt = sampled_potential_gt[:, z, :, :]  # [N, H, W]
            # sampled_potential_slice_gt = sampled_potential_slice_gt.swapaxes(
            #     1, 2
            # )  # [N, H, W]

            sampled_potential_slice = grid_sample(
                input_torch=potential_zyx,
                grid_torch=grid_slice,
                unnormalize=False,
                align_corners=False,
            ).squeeze(1)
            sampled_potential_slice = sampled_potential_slice[
                inverse_index
            ]  # [N, H, W]

            difference = torch.abs(sampled_potential_slice - sampled_potential_slice_gt)
            self.assertTrue(difference.max() < 1e-4)

    def test_patches_multiple_slices(self) -> None:
        phi, theta, psi = torch.rand(3)
        D, H, W = 64, 16, 32
        D_in, H_in, W_in = 64, 32, 64

        potential_zyx = torch.randn((D_in, H_in, W_in), dtype=torch.complex64)
        # potential_xzy = potential_zyx.swapaxes(1, 2).swapaxes(0, 1)

        positions = torch.tensor(
            [
                [0, 0],
                [0, 3],
                [0, 4],
                [4, 0],
                [1, 1],
                [4, 4],
            ]
        )
        batch_size = positions.shape[0]

        # ===conventional potential computation===
        rotation = rotate_ZYX(phi, theta, psi).unsqueeze(0)
        rotation = rotation.repeat(batch_size, 1, 1)
        translations = _get_translation_matrix_3D(positions)

        transform = torch.bmm(rotation, translations)

        grid = affine_grid_legacy(
            theta_torch=transform,
            size=(batch_size, 1, D, H, W),
            normalize=False,
            align_corners=False,
            input_size=(D_in, H_in, W_in),
        )

        sampled_potential_gt = grid_sample_legacy(
            input_torch=potential_zyx.unsqueeze(0),
            grid_torch=grid,
            unnormalize=False,
            align_corners=False,
        ).squeeze(
            1
        )  # [N, D, H, W]

        # === sliced potential computation ===
        yx_grid, inverse_index = batch_unique_with_inverse(
            positions=positions, patch_shape=(H, W)
        )

        grid_slice = affine_grid_legacy(
            yx_grid=yx_grid,
            z_start=0,
            z_len=D,
            theta_torch=rotate_ZYX(phi, theta, psi) @ torch.eye(n=3, m=4),
            input_size=(D_in, H_in, W_in),
            align_corners=False,
        )
        sampled_potential_slice_gt = sampled_potential_gt.sum(1)  # [N, H, W]
        # sampled_potential_slice_gt = sampled_potential_slice_gt.swapaxes(
        #     1, 2
        # )  # [N, H, W]

        sampled_potential_slice = grid_sample(
            input_torch=potential_zyx,
            grid_torch=grid_slice,
            unnormalize=False,
            align_corners=False,
        ).sum(1)
        sampled_potential_slice = sampled_potential_slice[inverse_index]  # [N, H, W]

        difference = torch.abs(sampled_potential_slice - sampled_potential_slice_gt)
        self.assertTrue(difference.max() < 1e-4)

    def test_backward(self) -> None:
        phi, theta, psi = torch.rand(3)
        D, H, W = 64, 32, 32
        D_in, H_in, W_in = 64, 64, 64

        potential_zyx = torch.randn((D_in, H_in, W_in), dtype=torch.complex64)

        positions = torch.tensor(
            [
                [0, 0],
                [0, 3],
                [0, 4],
                [4, 0],
                [1, 1],
                [4, 4],
            ]
        )

        # === sliced potential computation ===
        yx_grid, inverse_index = batch_unique_with_inverse(
            positions=positions, patch_shape=(H, W)
        )

        grid_slice = affine_grid_legacy(
            yx_grid=yx_grid,
            z_start=0,
            z_len=D,
            theta_torch=rotate_ZYX(phi, theta, psi) @ torch.eye(n=3, m=4),
            input_size=(D_in, H_in, W_in),
            align_corners=False,
        )

        sampled_potential_slice = grid_sample(
            input_torch=potential_zyx,
            grid_torch=grid_slice,
            unnormalize=False,
            align_corners=False,
        ).sum(1)
        sampled_potential_slice = sampled_potential_slice[inverse_index]
        sampled_potential_slice.mean().abs().backward()
        self.assertTrue(True)


if __name__ == "__main__":
    main()
