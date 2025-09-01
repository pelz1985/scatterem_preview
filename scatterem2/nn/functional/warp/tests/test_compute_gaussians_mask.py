import unittest

import torch
import warp as wp

from scatterem2.nn.functional import compute_gaussians_mask

wp.init()


class TestComputeGaussiansMask(unittest.TestCase):
    def setUp(self) -> None:
        # Use GPU if available, otherwise CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_inside_box(self) -> None:
        """Test a Gaussian that should intersect the grid bounding box."""
        # A point at the origin with sigma=0.5 gives a radius=1.5.
        # Even though part of the Gaussian extends outside the grid,
        # our kernel only checks for any overlap.
        positions = torch.tensor(
            [[0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device
        )
        amplitudes = torch.tensor([[1.0, 1.0]], dtype=torch.float32, device=self.device)
        sigmas = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=self.device)
        grid_mins = (-1.0, -1.0, -1.0)
        grid_maxs = (1.0, 1.0, 1.0)

        mask = compute_gaussians_mask(
            positions, amplitudes, sigmas, grid_mins, grid_maxs
        )
        # Expect True because amplitude conditions are met and there is overlap.
        self.assertTrue(mask.detach().cpu().numpy()[0])

    def test_outside_box(self) -> None:
        """Test a Gaussian that is completely outside the grid."""
        # Center far away from the grid. Even though amplitudes are positive,
        # the Gaussian box (position +- radius) does not overlap the grid.
        positions = torch.tensor(
            [[10.0, 10.0, 10.0]], dtype=torch.float32, device=self.device
        )
        amplitudes = torch.tensor([[1.0, 1.0]], dtype=torch.float32, device=self.device)
        sigmas = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=self.device)
        grid_mins = (-1.0, -1.0, -1.0)
        grid_maxs = (1.0, 1.0, 1.0)

        mask = compute_gaussians_mask(
            positions, amplitudes, sigmas, grid_mins, grid_maxs
        )
        # Expect False because the entire Gaussian is outside the grid.
        self.assertFalse(mask.detach().cpu().numpy()[0])

    def test_low_amplitude(self) -> None:
        """Test a Gaussian with amplitudes lower than the threshold (-1e-7)."""
        # With very low amplitudes for both real and imaginary parts,
        # the Gaussian should be culled regardless of position.
        positions = torch.tensor(
            [[0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device
        )
        amplitudes = torch.tensor(
            [[-2.0, -2.0]], dtype=torch.float32, device=self.device
        )
        sigmas = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=self.device)
        grid_mins = (-1.0, -1.0, -1.0)
        grid_maxs = (1.0, 1.0, 1.0)

        mask = compute_gaussians_mask(
            positions, amplitudes, sigmas, grid_mins, grid_maxs
        )
        # Expect False because the amplitude check fails.
        self.assertFalse(mask.detach().cpu().numpy()[0])

    def test_multiple_gaussians(self) -> None:
        """Test multiple gaussians with mixed outcomes."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Should be inside the grid.
                [10.0, 10.0, 10.0],  # Completely outside.
                [
                    0.5,
                    0.5,
                    0.5,
                ],  # Mixed amplitude: one channel below threshold, one above.
            ],
            dtype=torch.float32,
            device=self.device,
        )
        amplitudes = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [
                    -2.0,
                    1.0,
                ],  # Although the real component is low, the imaginary is positive.
            ],
            dtype=torch.float32,
            device=self.device,
        )
        sigmas = torch.tensor(
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            dtype=torch.float32,
            device=self.device,
        )
        grid_mins = (-1.0, -1.0, -1.0)
        grid_maxs = (1.0, 1.0, 1.0)

        mask = compute_gaussians_mask(
            positions, amplitudes, sigmas, grid_mins, grid_maxs
        )
        mask_np = mask.detach().cpu().numpy()
        # Expected: first Gaussian is True, second is False, third is True (because the imaginary component meets the condition).
        self.assertEqual(mask_np.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
