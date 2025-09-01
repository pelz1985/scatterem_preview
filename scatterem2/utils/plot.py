import importlib.util
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from torch import Tensor

if importlib.util.find_spec("pyvista") is not None:
    import pyvista as pv


def _label_with_units(label: str, unit: str | None = None) -> str:
    if unit is not None:
        return f"{label}, [{unit}]"
    return label


def plot_images(
    arrs: Sequence[Tensor | np.ndarray],
    use_colorbar: bool = True,
    axs: Iterable[Axes] | None = None,
) -> None:
    if axs is None:
        fig, axs = plt.subplots(1, len(arrs), figsize=(10, 10))
    for i, arr in enumerate(arrs):
        if isinstance(arr, Tensor):
            arr = arr.detach().cpu().numpy()
        # arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = arr / arr.max()
        # s = arr.shape[0]
        # ss = int(np.round(s / 4))
        # arr[ss+1:3*ss, ss+1:3*ss] = 0
        if len(arrs) == 1:
            ax = axs
        else:
            ax = axs[i]

        m = ax.imshow(arr)
        if use_colorbar:
            plt.colorbar(m, fraction=0.046, pad=0.04, location="bottom")
        ax.set_axis_off()
    if axs is None:
        plt.show()


def plot_arrays(arrs: Sequence[Iterable], axs: Iterable[Axes] | None = None) -> None:
    if axs is None:
        fig, axs = plt.subplots(1, len(arrs))
    for i, arr in enumerate(arrs):
        if isinstance(arr, Tensor):
            arr = arr.flatten().detach().cpu().numpy()
        if len(arrs) > 1:
            axs[i].plot(arr)
        else:
            axs.plot(arr)
    if axs is None:
        plt.show()


def plot_volume(
    volume: Tensor | np.ndarray,
    view_plane: str = "xy",
    save_path: str | Path | None = None,
    title: str | None = None,
    crop_negatives: bool = True,
    font_size: int = 14,
    values_units: str | None = None,
    grid_sampling: float | None = None,
    grid_units: str | None = None,
    opacity: str = "linear",
) -> None:
    if importlib.util.find_spec("pyvista") is None:
        raise ImportError("pyvista is not installed. plot_volume is unavailable")

    if isinstance(volume, Tensor):
        volume = volume.detach().cpu().numpy()

    volume = volume.swapaxes(0, 2)  # ZYX -> XYZ
    if crop_negatives:
        volume[volume < 0] = 0
    # volume = (
    #     (volume - volume.min()) / (volume.max() - volume.min())
    #     if (volume.max() - volume.min()) > 1e-4
    #     else volume
    # )

    # Create pyvista grids for each tensor
    grid = pv.ImageData()
    if grid_sampling is not None:
        grid.spacing = (grid_sampling, grid_sampling, grid_sampling)

    # Set grid dimensions and spacing
    def get_dim(shape: Iterable) -> Tuple:
        return (s + 1 for s in shape)

    grid.dimensions = get_dim(volume.shape)
    # Set grid scalars
    grid.cell_data["values"] = volume.ravel(order="F")  # Flatten in Fortran order

    # Create a pyvista plotter
    off_screen = True if save_path is not None else False
    plotter = pv.Plotter(off_screen=off_screen)

    # Add both volumes to the plotter
    plotter.add_volume(grid, cmap="Blues", opacity=opacity)

    plotter.scalar_bar.SetTitle(_label_with_units(label="Values", unit=values_units))

    getattr(plotter, f"view_{view_plane}")(plotter)

    if title is not None:
        plotter.add_title(title=title, font_size=font_size)
    plotter.add_axes(line_width=3.0, labels_off=False)

    if grid_units is None:
        grid_units = "voxel steps"
    plotter.show_bounds(
        xtitle=_label_with_units(label="X", unit=grid_units),
        ytitle=_label_with_units(label="Y", unit=grid_units),
        ztitle=_label_with_units(label="Z", unit=grid_units),
        grid=True,
        all_edges=True,
        font_size=font_size,
        location="outer",
    )

    if save_path is None:
        plotter.show()
    else:
        plotter.screenshot(
            filename=str(save_path),
        )
        plotter.close()


def plot_sqr_dist_heatmap(
    lhs_pos: Tensor, rhs_pos: Tensor, max_dim: int = 1000
) -> None:
    # lhs_pos : [..., 3]
    # rhs_pos : [..., 3]
    with torch.no_grad():
        lhs_pos = lhs_pos.reshape(-1, 3)[:max_dim].unsqueeze(1)
        rhs_pos = rhs_pos.reshape(-1, 3)[:max_dim].unsqueeze(0)
        print(lhs_pos.shape)
        print(rhs_pos.shape)
        diff = ((lhs_pos - rhs_pos) ** 2).sum(-1).detach().cpu().numpy()
        print(diff.max())
        diff /= diff.max()
        sns.heatmap(diff)
        plt.show()


def stack_and_plot(images: np.ndarray | Tensor, ax: Axes | None, vmax: float) -> None:
    """
    Stack a 4D array of shape (n_rows, n_cols, height, width) into a single image
    of shape (n_rows*height, n_cols*width) and plot it.

    Parameters:
    - images: 4D numpy array with shape (n_rows, n_cols, height, width)
    - figsize: tuple specifying figure size for matplotlib

    Returns:
    - mosaic: 2D numpy array of the stacked image
    """
    # Validate input
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if isinstance(images, Tensor):
        images = images.detach().cpu().numpy()
    if images.ndim != 4:
        raise ValueError(
            "Input must be a 4D array with shape (n_rows, n_cols, height, width)"
        )

    n_rows, n_cols, h, w = images.shape
    # Rearrange and reshape into mosaic
    mosaic = images.transpose(0, 2, 1, 3).reshape(n_rows * h, n_cols * w)

    # Plot
    ax.imshow(mosaic, aspect="equal", cmap="gray", vmax=vmax)
    ax.axis("off")


def plot_probe(
    probe: np.ndarray | Tensor,
    save_path: str | Path | None = None,
    sampling: float | None = None,
    units: str | None = None,
    figsize: Tuple[int, int] | None = (8, 4),
) -> None:
    if isinstance(probe, Tensor):
        probe = probe.detach().cpu().numpy()

    if sampling is None:
        extent = None
    else:
        extent = [0, probe.shape[0] * sampling, 0, probe.shape[1] * sampling]
    if units is None:
        units = "pixel steps"

    xlabel = f"X, [{units}]"
    ylabel = f"Y, [{units}]"

    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.imshow(np.abs(probe), cmap="viridis", extent=extent)
    plt.colorbar(label="Amplitude")
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title("Probe Amplitude")

    plt.subplot(122)
    plt.imshow(np.angle(probe), cmap="twilight", extent=extent)
    plt.colorbar(label="Phase (rad)")
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title("Probe Phase")

    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(str(save_path))
        plt.close()


def plot_patterns(
    pattern_pred: np.ndarray | Tensor,
    pattern_gt: np.ndarray | Tensor,
    save_path: str | Path | None = None,
    sampling: float | None = None,
    units: str | None = None,
    figsize: Tuple[int, int] | None = (8, 4),
) -> None:
    if isinstance(pattern_pred, Tensor):
        pattern_pred = pattern_pred.detach().cpu().numpy()
    if isinstance(pattern_gt, Tensor):
        pattern_gt = pattern_gt.detach().cpu().numpy()

    if sampling is None:
        extent = None
    else:
        extent = [
            0,
            pattern_pred.shape[0] * sampling,
            0,
            pattern_pred.shape[1] * sampling,
        ]
    if units is None:
        units = "pixel steps"

    xlabel = f"X, [{units}]"
    ylabel = f"Y, [{units}]"

    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.imshow(pattern_pred, cmap="viridis", extent=extent)
    plt.colorbar(label="Amplitude")
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title("Predicted Pattern")

    plt.subplot(122)
    plt.imshow(pattern_gt, cmap="viridis", extent=extent)
    plt.colorbar(label="Amplitude")
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title("Ground Truth Pattern")

    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(str(save_path))
        plt.close()


def plot_two_volumes(
    volume_a: Tensor | np.ndarray,
    volume_b: Tensor | np.ndarray,
    view_plane: str = "xy",
    save_path: str | Path | None = None,
    title: str | None = None,
    crop_negatives: bool = True,
    font_size: int = 14,
    values_units_a: str | None = None,
    values_units_b: str | None = None,
    grid_sampling: float | None = None,
    grid_units: str | None = None,
    opacity_a: str = "linear",
    opacity_b: str = "linear",
    cmap_a: str = "Blues",
    cmap_b: str = "Reds",
    blend_mode: str = "composite",  # options: 'composite', 'maximum', 'minimum', 'additive'
) -> None:
    if importlib.util.find_spec("pyvista") is None:
        raise ImportError("pyvista is not installed. plot_two_volumes is unavailable")

    # Torch -> numpy if needed
    if isinstance(volume_a, Tensor):
        volume_a = volume_a.detach().cpu().numpy()
    if isinstance(volume_b, Tensor):
        volume_b = volume_b.detach().cpu().numpy()

    # Basic sanity check
    if volume_a.shape != volume_b.shape:
        raise ValueError(
            f"volume_a.shape {volume_a.shape} != volume_b.shape {volume_b.shape}; "
            "volumes must have identical shapes to be overlaid."
        )

    # Match your orientation: ZYX -> XYZ for both
    volume_a = volume_a.swapaxes(0, 2)
    volume_b = volume_b.swapaxes(0, 2)

    if crop_negatives:
        volume_a[volume_a < 0] = 0
        volume_b[volume_b < 0] = 0

    # Build a helper to make a grid from a numpy volume
    def make_grid(vol: np.ndarray) -> "pv.ImageData":
        grid = pv.ImageData()
        if grid_sampling is not None:
            grid.spacing = (grid_sampling, grid_sampling, grid_sampling)

        def get_dim(shape: Iterable) -> Tuple:
            return (s + 1 for s in shape)

        grid.dimensions = get_dim(vol.shape)
        grid.cell_data["values"] = vol.ravel(
            order="F"
        )  # Fortran order to match VTK cell layout
        return grid

    grid_a = make_grid(volume_a)
    grid_b = make_grid(volume_b)

    # Off-screen if saving
    off_screen = True if save_path is not None else False
    plotter = pv.Plotter(off_screen=off_screen)

    # Add both volumes, each with its own cmap & opacity; give each a scalar bar
    # Position scalar bars side-by-side at the bottom to avoid overlap.
    sb_kwargs_common = dict(
        vertical=False,
        width=0.38,
        height=0.08,
        label_font_size=max(8, font_size - 2),
        title_font_size=font_size,
    )

    plotter.add_volume(
        grid_a,
        cmap=cmap_a,
        opacity=opacity_a,
        blending=blend_mode,
        name="volume_a",
        scalar_bar_args=dict(
            title=_label_with_units(label="Values A", unit=values_units_a),
            position_x=0.05,
            position_y=0.02,
            **sb_kwargs_common,
        ),
    )

    plotter.add_volume(
        grid_b,
        cmap=cmap_b,
        opacity=opacity_b,
        blending=blend_mode,
        name="volume_b",
        scalar_bar_args=dict(
            title=_label_with_units(label="Values B", unit=values_units_b),
            position_x=0.57,
            position_y=0.02,
            **sb_kwargs_common,
        ),
    )

    # Camera/view, axes, bounds—same as your original
    getattr(plotter, f"view_{view_plane}")(plotter)

    if title is not None:
        plotter.add_title(title=title, font_size=font_size)

    plotter.add_axes(line_width=3.0, labels_off=False)

    if grid_units is None:
        grid_units = "voxel steps"
    plotter.show_bounds(
        xtitle=_label_with_units(label="X", unit=grid_units),
        ytitle=_label_with_units(label="Y", unit=grid_units),
        ztitle=_label_with_units(label="Z", unit=grid_units),
        grid=True,
        all_edges=True,
        font_size=font_size,
        location="outer",
    )

    if save_path is None:
        plotter.show()
    else:
        plotter.screenshot(filename=str(save_path))
        plotter.close()
