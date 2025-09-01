"""Module for various convenient utilities."""

from __future__ import annotations

import copy
import inspect
from importlib.util import find_spec
from typing import Any, Optional, Tuple, TypeVar

import numpy as np
import torch

if find_spec("cv2") is not None:
    import cv2

T = TypeVar("T", float, int, bool)


def number_to_tuple(
    value: T | tuple[T, ...], dimension: Optional[int] = None
) -> tuple[T, ...]:
    if isinstance(value, (float, int, bool)):
        if dimension is None:
            return (value,)
        else:
            return (value,) * dimension
    else:
        if dimension is not None:
            assert len(value) == dimension
        return value


class CopyMixin:
    _exclude_from_copy: tuple = ()

    @staticmethod
    def _arg_keys(cls: type) -> tuple[str, ...]:
        parameters = inspect.signature(cls).parameters
        return tuple(
            key
            for key, value in parameters.items()
            if value.kind not in (value.VAR_POSITIONAL, value.VAR_KEYWORD)
        )

    def _copy_kwargs(
        self, exclude: tuple[str, ...] = (), cls: type | None = None
    ) -> dict[str, Any]:
        if cls is None:
            cls = self.__class__

        exclude = self._exclude_from_copy + exclude
        keys = [key for key in self._arg_keys(cls) if key not in exclude]
        kwargs = {key: copy.deepcopy(getattr(self, key)) for key in keys}
        return kwargs

    def copy(self) -> "CopyMixin":
        """Make a copy."""
        return copy.deepcopy(self)


def get_dtype(complex: bool = False) -> torch.dtype | type:
    """
    Get the numpy dtype from the config precision setting.

    Parameters
    ----------
    complex : bool, optional
        If True, return a complex dtype. Defaults to False.
    """
    dtype = "float32"  # config.get("precision")

    if dtype == "float32" and complex:
        dtype = torch.complex64
    elif dtype == "float32":
        dtype = torch.float32
    elif dtype == "float64" and complex:
        dtype = torch.complex128
    elif dtype == "float64":
        dtype = torch.float64
    else:
        raise RuntimeError(f"Invalid dtype: {dtype}")

    return dtype


def detect_edges(mask: np.ndarray, method: str = "canny") -> np.ndarray:
    """Detect edges in binary mask using specified method."""
    if method == "canny":
        # Use Canny edge detection
        edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
    elif method == "sobel":
        # Use Sobel edge detection
        sobelx = cv2.Sobel(mask.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(mask.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = (edges > 0.1).astype(np.uint8)
    elif method == "gradient":
        # Use simple gradient method
        gy, gx = np.gradient(mask.astype(np.float32))
        edges = np.sqrt(gx**2 + gy**2)
        edges = (edges > 0.1).astype(np.uint8)
    else:
        raise ValueError(f"Unknown edge method: {method}")

    return edges


def fit_circle_ransac(
    points: np.ndarray, iterations: int = 1000, threshold: float = 2.0
) -> Tuple[float, float, float]:
    """
    Fit circle to points using RANSAC for robustness to outliers.

    Args:
        points: Nx2 array of [y, x] coordinates
        iterations: Number of RANSAC iterations
        threshold: Distance threshold for inliers

    Returns:
        (cy, cx, r): Circle center and radius, or None if failed
    """
    if len(points) < 3:
        return None

    best_circle = None
    best_inliers = 0

    for _ in range(iterations):
        # Randomly sample 3 points
        sample_idx = np.random.choice(len(points), 3, replace=False)
        sample_points = points[sample_idx]

        # Fit circle to 3 points
        circle = fit_circle_3points(sample_points)
        if circle is None:
            continue

        cy, cx, r = circle

        # Count inliers
        distances = np.abs(
            np.sqrt((points[:, 0] - cy) ** 2 + (points[:, 1] - cx) ** 2) - r
        )
        inliers = distances < threshold
        num_inliers = np.sum(inliers)

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_circle = circle

    return best_circle


def fit_circle_3points(points: np.ndarray) -> Tuple[float, float, float]:
    """Fit circle through 3 points using algebraic method."""
    if len(points) != 3:
        return None

    y1, x1 = points[0]
    y2, x2 = points[1]
    y3, x3 = points[2]

    # Check for collinear points
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
    if abs(det) < 1e-10:
        return None

    # Calculate circle center
    a = x1**2 + y1**2
    b = x2**2 + y2**2
    c = x3**2 + y3**2

    cx = (a * (y2 - y3) + b * (y3 - y1) + c * (y1 - y2)) / (2 * det)
    cy = (a * (x3 - x2) + b * (x1 - x3) + c * (x2 - x1)) / (2 * det)

    # Calculate radius
    r = np.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)

    return cy, cx, r


def select_best_circle(candidates: list, DP: np.ndarray) -> Tuple[float, float, float]:
    """Select best circle from candidates based on multiple criteria."""
    if len(candidates) == 1:
        return candidates[0][:3]

    # Score each candidate
    scores = []
    for cy, cx, r, thresh, idx in candidates:
        # Score based on: radius consistency, center consistency, edge strength
        score = 0

        # Radius consistency (prefer radii that appear frequently)
        radii = [c[2] for c in candidates]
        radius_std = np.std(radii)
        if radius_std > 0:
            score += 1 / (1 + abs(r - np.median(radii)) / radius_std)

        # Center consistency
        centers = [(c[0], c[1]) for c in candidates]
        center_distances = [
            np.sqrt((cy - c[0]) ** 2 + (cx - c[1]) ** 2) for c in centers
        ]
        score += 1 / (1 + np.mean(center_distances))

        # Edge strength at circle boundary
        angles = np.linspace(0, 2 * np.pi, 100)
        boundary_y = cy + r * np.cos(angles)
        boundary_x = cx + r * np.sin(angles)

        # Check if boundary points are within image
        h, w = DP.shape
        valid_mask = (
            (boundary_y >= 0) & (boundary_y < h) & (boundary_x >= 0) & (boundary_x < w)
        )

        if np.sum(valid_mask) > 0:
            boundary_y = boundary_y[valid_mask]
            boundary_x = boundary_x[valid_mask]

            # Sample boundary intensities
            boundary_intensities = DP[boundary_y.astype(int), boundary_x.astype(int)]
            score += np.std(boundary_intensities)  # Higher variation = better edge

        scores.append(score)

    # Return candidate with highest score
    best_idx = np.argmax(scores)
    return candidates[best_idx][:3]


def refine_circle_fit(
    initial_circle: Tuple[float, float, float], DP: np.ndarray, edge_method: str
) -> Tuple[float, float, float]:
    """Refine circle fit using least-squares on edge points near initial circle."""
    cy, cx, r = initial_circle

    # Create mask around initial circle
    h, w = DP.shape
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    distances = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)

    # Focus on region near the circle boundary
    ring_mask = (distances > r * 0.8) & (distances < r * 1.2)

    if not np.any(ring_mask):
        return cy, cx, r

    # Detect edges in the ring region
    local_region = DP * ring_mask
    edges = detect_edges((local_region > 0.1).astype(np.uint8), edge_method)

    edge_points = np.column_stack(np.where(edges))

    if len(edge_points) < 10:
        return cy, cx, r

    # Least-squares circle fitting
    refined_circle = fit_circle_least_squares(edge_points, initial_guess=(cy, cx, r))

    if refined_circle is not None:
        return refined_circle
    else:
        return cy, cx, r


def fit_circle_least_squares(
    points: np.ndarray, initial_guess: Tuple[float, float, float] = None
) -> Tuple[float, float, float]:
    """Fit circle using least-squares optimization."""
    from scipy.optimize import least_squares

    def residuals(params, points):
        cy, cx, r = params
        distances = np.sqrt((points[:, 0] - cy) ** 2 + (points[:, 1] - cx) ** 2)
        return distances - r

    if initial_guess is None:
        # Use centroid as initial guess
        cy_init = np.mean(points[:, 0])
        cx_init = np.mean(points[:, 1])
        r_init = np.mean(
            np.sqrt((points[:, 0] - cy_init) ** 2 + (points[:, 1] - cx_init) ** 2)
        )
        initial_guess = (cy_init, cx_init, r_init)

    result = least_squares(residuals, initial_guess, args=(points,))
    if result.success:
        return tuple(result.x)

    return None
