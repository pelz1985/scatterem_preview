from typing import Any, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from numpy.typing import DTypeLike, NDArray
from skimage.filters import window
from torch import Tensor
from torch.utils.data import BatchSampler, SequentialSampler
from torch.utils.data import Dataset as TorchDataset

from scatterem2.io.serialize import AutoSerialize
from scatterem2.utils.data import Metadata4D
from scatterem2.utils.data.data_classes import Metadata4dstem
from scatterem2.utils.stem import energy2wavelength
from scatterem2.utils.utils import (
    detect_edges,
    fit_circle_ransac,
    refine_circle_fit,
    select_best_circle,
)
from scatterem2.utils.validators import (
    ensure_valid_array,
    validate_ndinfo,
    validate_units,
)


class Dataset(TorchDataset, AutoSerialize):
    """
    A class representing a multi-dimensional dataset with metadata.
    Uses standard properties and validation within __init__ for type safety.

    Attributes (Properties):
        array (NDArray | Any): The underlying n-dimensional array data (Any for CuPy).
        name (str): A descriptive name for the dataset.
        origin (NDArray): The origin coordinates for each dimension (1D array).
        sampling (NDArray): The sampling rate/spacing for each dimension (1D array).
        units (list[str]): Units for each dimension.
        signal_units (str): Units for the array values.
    """

    _token = object()

    def __init__(
        self,
        array: Any,  # Input can be array-like
        name: str,
        origin: Union[NDArray, tuple, list, float, int],
        sampling: Union[NDArray, tuple, list, float, int],
        units: Union[list[str], tuple, list],
        signal_units: str = "arb. units",
        _token: object | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        if _token is not self._token:
            raise RuntimeError("Use Dataset.from_array() to instantiate this class.")

        self._array: Tensor = ensure_valid_array(array, device=device)
        self.name = name
        self.origin = origin
        self.sampling = sampling
        self.units = units
        self.signal_units = signal_units
        self.device = device

    @classmethod
    def from_array(
        cls,
        array: Any,  # Input can be array-like
        name: str | None = None,
        origin: Union[NDArray, tuple, list, float, int] | None = None,
        sampling: Union[NDArray, tuple, list, float, int] | None = None,
        units: Union[list[str], tuple, list] | None = None,
        signal_units: str = "arb. units",
    ) -> "Dataset":
        """
        Validates and creates a Dataset from an array.

        Parameters
        ----------
        array: Any
            The array to validate and create a Dataset from.
        name: str | None
            The name of the Dataset.
        origin: Union[NDArray, tuple, list, float, int] | None
            The origin of the Dataset.
        sampling: Union[NDArray, tuple, list, float, int] | None
            The sampling of the Dataset.
        units: Union[list[str], tuple, list] | None
            The units of the Dataset.
        signal_units: str
            The units of the signal.

        Returns
        -------
        Dataset
            A Dataset object with the validated array and metadata.
        """
        validated_array = ensure_valid_array(array)
        _ndim = validated_array.ndim

        # Set defaults if None
        _name = name if name is not None else f"{_ndim}d dataset"
        _origin = origin if origin is not None else np.zeros(_ndim)
        _sampling = sampling if sampling is not None else np.ones(_ndim)
        _units = units if units is not None else ["pixels"] * _ndim

        return cls(
            array=validated_array,
            name=_name,
            origin=_origin,
            sampling=_sampling,
            units=_units,
            signal_units=signal_units,
            _token=cls._token,
        )

    # --- Properties ---
    @property
    def array(self) -> Tensor:
        """The underlying n-dimensional array data."""
        return self._array

    @array.setter
    def array(self, value: Tensor) -> None:
        self._array = ensure_valid_array(value, dtype=self.dtype, ndim=self.ndim)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def origin(self) -> NDArray:
        return self._origin

    @origin.setter
    def origin(self, value: Union[NDArray, tuple, list, float, int]) -> None:
        self._origin = validate_ndinfo(value, self.ndim, "origin")

    @property
    def sampling(self) -> NDArray:
        return self._sampling

    @sampling.setter
    def sampling(self, value: Union[NDArray, tuple, list, float, int]) -> None:
        self._sampling = validate_ndinfo(value, self.ndim, "sampling")

    @property
    def units(self) -> list[str]:
        return self._units

    @units.setter
    def units(self, value: Union[list[str], tuple, list]) -> None:
        self._units = validate_units(value, self.ndim)

    @property
    def signal_units(self) -> str:
        return self._signal_units

    @signal_units.setter
    def signal_units(self, value: str) -> None:
        self._signal_units = str(value)

    # --- Derived Properties ---
    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> torch.dtype:
        return self.array.dtype

    @property
    def device(self) -> torch.device:
        """
        Outputting a string is likely temporary -- once we have our use cases we can
        figure out a more permanent device solution that enables easier translation between

        """
        return self.array.device

    @device.setter
    def device(self, value: torch.device) -> None:
        self.array = self.array.to(value)

    # --- Summaries ---
    def __repr__(self) -> str:
        description = [
            f"Dataset(shape={self.shape}, dtype={self.dtype}, name='{self.name}')",
            f"  sampling: {self.sampling}",
            f"  units: {self.units}",
            f"  signal units: '{self.signal_units}'",
        ]
        return "\n".join(description)

    def __str__(self) -> str:
        description = [
            f"Dataset named '{self.name}'",
            f"  shape: {self.shape}",
            f"  dtype: {self.dtype}",
            f"  device: {self.device}",
            f"  origin: {self.origin}",
            f"  sampling: {self.sampling}",
            f"  units: {self.units}",
            f"  signal units: '{self.signal_units}'",
        ]
        return "\n".join(description)

    # --- Methods ---
    def copy(self) -> "Dataset":
        """
        Copies Dataset.

        Parameters
        ----------
        copy_attributes: bool
            If True, copies non-standard attributes. Standard attributes (array, metadata)
            are always deep-copied.
        """
        # Metadata arrays (origin, sampling) are numpy, use copy()
        # Units list is copied by slicing
        new_dataset = type(self).from_array(
            array=self.array.copy(),
            name=self.name,
            origin=self.origin.copy(),
            sampling=self.sampling.copy(),
            units=self.units[:],
            signal_units=self.signal_units,
        )

        return new_dataset

    def mean(self, axes: Optional[tuple[int, ...]] = None) -> Any:
        """
        Computes and returns mean of the data array.

        Parameters
        ----------
        axes: tuple, optional
            Axes over which to compute mean. If None specified, mean of all elements is computed.

        Returns
        --------
        mean: scalar or array (np.ndarray or cp.ndarray)
            Mean of the data.
        """
        return self.array.mean(axis=axes)

    def max(self, axes: Optional[tuple[int, ...]] = None) -> Any:
        """
        Computes and returns max of the data array.

        Parameters
        ----------
        axes: tuple, optional
            Axes over which to compute max. If None specified, max of all elements is computed.

        Returns
        --------
        maximum: scalar or array (np.ndarray or cp.ndarray)
            Maximum of the data.
        """
        return self.array.max(axis=axes)

    def min(self, axes: Optional[tuple[int, ...]] = None) -> Any:
        """
        Computes and returns min of the data array.

        Parameters
        ----------
        axes: tuple, optional
            Axes over which to compute min. If None specified, min of all elements is computed.

        Returns
        --------
        minimum: scalar or array (np.ndarray or cp.ndarray)
            Minimum of the data.
        """
        return self.array.min(axis=axes)

    def pad(
        self,
        pad_width: Union[int, tuple[int, int], tuple[tuple[int, int], ...]],
        modify_in_place: bool = False,
        **kwargs: Any,
    ) -> Optional["Dataset"]:
        """
        Pads Dataset data array using numpy.pad or cupy.pad.
        Metadata (origin, sampling) is not modified.

        Parameters
        ----------
        pad_width: int, tuple
            Number of values padded to the edges of each axis. See numpy.pad documentation.
        modify_in_place: bool
            If True, modifies this dataset's array directly. If False, returns a new Dataset.
        kwargs: dict
            Additional keyword arguments passed to numpy.pad or cupy.pad.

        Returns
        --------
        Dataset or None
            Padded Dataset if modify_in_place is False, otherwise None.
        """
        # Convert pad_width to torch.nn.functional.pad format (reverse order)
        if isinstance(pad_width, int):
            pad_list = [pad_width] * (2 * self.ndim)
        elif isinstance(pad_width, tuple):
            if (
                len(pad_width) == 2
                and isinstance(pad_width[0], int)
                and isinstance(pad_width[1], int)
            ):
                # Single tuple like (before, after) - apply to all dimensions
                pad_list = []
                for _ in range(self.ndim):
                    pad_list.extend([pad_width[0], pad_width[1]])
            else:
                # Multiple tuples like ((before1, after1), (before2, after2), ...)
                pad_list = []
                for pad_tuple in pad_width:
                    if isinstance(pad_tuple, tuple) and len(pad_tuple) == 2:
                        pad_list.extend([pad_tuple[0], pad_tuple[1]])
                    else:
                        raise ValueError("Each pad_tuple must be a tuple of 2 integers")
        else:
            raise ValueError("pad_width must be int or tuple")

        padded_array = torch.nn.functional.pad(self.array, pad=pad_list, **kwargs)

        if modify_in_place:
            self._array = padded_array
            return None
        else:
            new_dataset = self.copy()
            new_dataset.array = padded_array
            new_dataset.name = self.name + " (padded)"
            return new_dataset

    def crop(
        self,
        crop_widths: tuple[tuple[int, int], ...],
        axes: Optional[Union[int, tuple[int, ...]]] = None,
        modify_in_place: bool = False,
    ) -> Optional["Dataset"]:
        """
        Crops Dataset

        Parameters
        ----------
        crop_widths:tuple
            Min and max for cropping each axis specified as a tuple
        axes:
            Axes over which to crop. If None specified, all are cropped.
        modify_in_place: bool
            If True, modifies dataset

        Returns
        --------
        Dataset (cropped) only if modify_in_place is False
        """
        if axes is None:
            if len(crop_widths) != self.ndim:
                raise ValueError(
                    "crop_widths must match number of dimensions when axes is None."
                )
            axes = tuple(range(self.ndim))
        elif np.isscalar(axes):
            axes = (axes,)
            crop_widths = (crop_widths,)
        else:
            axes = tuple(axes)

        if len(crop_widths) != len(axes):
            raise ValueError("Length of crop_widths must match length of axes.")

        full_slices = []
        crop_dict = dict(zip(axes, crop_widths))
        for axis, dim in enumerate(self.shape):
            if axis in crop_dict:
                before, after = crop_dict[axis]
                start = before
                stop = dim - after if after != 0 else None
                full_slices.append(slice(start, stop))
            else:
                full_slices.append(slice(None))
        if modify_in_place is False:
            dataset = self.copy()
            dataset.array = dataset.array[tuple(full_slices)]
            return dataset
        else:
            self.array = self.array[tuple(full_slices)]

    def bin(
        self,
        bin_factors: Union[tuple[int, ...], int, list[int]],
        axes: Optional[Union[int, tuple[int, ...]]] = None,
        modify_in_place: bool = False,
    ) -> Optional["Dataset"]:
        """
        Bins Dataset

        Parameters
        ----------
        bin_factors:tuple or int
            bin factors for each axis
        axes:
            Axis over which to bin. If None is specified, all axes are binned.
        modify_in_place: bool
            If True, modifies dataset

        Returns
        --------
        Dataset (binned) only if modify_in_place is False
        """
        if axes is None:
            axes = tuple(range(self.ndim))
        elif np.isscalar(axes):
            axes = (axes,)

        if isinstance(bin_factors, int):
            bin_factors = tuple([bin_factors] * len(axes))
        elif isinstance(bin_factors, (list, tuple)):
            if len(bin_factors) != len(axes):
                raise ValueError("bin_factors and axes must have the same length.")
            bin_factors = tuple(bin_factors)
        else:
            raise TypeError("bin_factors must be an int or tuple of ints.")

        axis_to_factor = dict(zip(axes, bin_factors))

        slices = []
        new_shape = []
        for axis in range(self.ndim):
            if axis in axis_to_factor:
                factor = axis_to_factor[axis]
                length = self.shape[axis] - (self.shape[axis] % factor)
                slices.append(slice(0, length))
                new_shape.extend([length // factor, factor])
            else:
                slices.append(slice(None))
                new_shape.append(self.shape[axis])

        reshape_dims = []
        reduce_axes = []
        current_axis = 0

        for axis in range(self.ndim):
            if axis in axis_to_factor:
                reshape_dims.extend([new_shape[current_axis], axis_to_factor[axis]])
                reduce_axes.append(len(reshape_dims) - 1)
                current_axis += 2
            else:
                reshape_dims.append(new_shape[current_axis])
                current_axis += 1

        if modify_in_place is False:
            dataset = self.copy()
            dataset.array = np.sum(
                dataset.array[tuple(slices)].reshape(reshape_dims),
                axis=tuple(reduce_axes),
            )
            return dataset
        else:
            self.array = np.sum(
                self.array[tuple(slices)].reshape(reshape_dims), axis=tuple(reduce_axes)
            )


class Dataset4dstem(Dataset):
    """A 4D STEM dataset with metadata for electron diffraction patterns."""

    meta: Optional[Metadata4dstem] = None

    def __init__(
        self,
        array: Tensor,
        name: str,
        origin: Union[NDArray, tuple, list, float, int],
        sampling: Union[NDArray, tuple, list, float, int],
        units: Union[list[str], tuple, list],
        signal_units: str = "arb. units",
        meta: Optional[Metadata4dstem] = None,
        transform_to_amplitudes: bool = False,
        astype_float32: bool = True,
        fourier_shift_dim: Tuple = None,
        probe_index: int = 0,
        device: torch.device = torch.device("cpu"),
        normalize: bool = True,
        clip_neg_values: bool = True,
        _token: object | None = None,
    ) -> None:
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            _token=_token,
            device=device,
        )

        self.meta = meta
        self.transform_to_amplitudes = transform_to_amplitudes
        self.fourier_shift_dim = fourier_shift_dim
        self.probe_index = probe_index

        self._shape = self._array.shape
        if astype_float32:
            self._array = self._array.to(torch.float32)
        if fourier_shift_dim is not None:
            self._array = torch.fft.ifftshift(self._array, dim=(2, 3))
        if clip_neg_values:
            self._array[self._array < 0] = 0

        self._array3d = self._array.contiguous().view(
            self._shape[0] * self._shape[1], self._shape[2], self._shape[3]
        )
        if transform_to_amplitudes:
            self._array.sqrt_()
        if normalize:
            normalization_const = self._array3d.mean(0).max()
            self._array /= normalization_const
        self._total_intensity = None
        # Calculate total intensity using float64 precision and loop

    def crop_brightfield_pad_and_taper(
        self: "Dataset4dstem",
        pad_ratio: float = 0.1,
        taper_ratio: float = 0.17,
        thresh_lower: float = 0.01,
        thresh_upper: float = 0.99,
    ) -> "Dataset4dstem":
        """
        Crop the dataset to the brightfield region, pad and taper it.
        """
        r, c = self.probe_radius_and_center(
            thresh_lower=thresh_lower, thresh_upper=thresh_upper
        )
        r = np.round(r)
        r_int = int(np.ceil(r)) + 1
        y0_int = int(np.round(c[0]))
        x0_int = int(np.round(c[1]))
        rmax = self._array.shape[-1] // 2
        r = min(r_int, rmax)
        print(f"Radias BF: {r}")
        print(f"Center BF: {c}")

        data_bf = self.crop(
            np.s_[:, :, y0_int - r : y0_int + r, x0_int - r : x0_int + r],
            clone=False,
        )
        data_tapered = data_bf.pad_and_taper_realspace()
        return data_tapered

    def pad_and_taper_fourierspace(
        self: "Dataset4dstem", pad_ratio: float = 0.1, taper_ratio: float = 0.17
    ) -> "Dataset4dstem":
        """Pad and taper the diffraction patterns (last two dimensions).

        Args:
            pad_ratio (float, optional) : The ratio of padding to add to the diffraction patterns. Defaults to 0.1.
            taper_ratio (float, optional): The ratio of taper to add to the diffraction patterns. Defaults to 0.17.

        Returns:
            Dataset4dstem: The padded and tapered dataset.
        """
        dc = self._array
        sh = dc.shape
        # Calculate padding for diffraction pattern dimensions (last two dimensions)
        pw = [int(x) for x in (np.array(sh[-2:]) * pad_ratio).astype(int)]

        # Create padded array with increased diffraction pattern size
        padded_dc = torch.zeros(
            (sh[0], sh[1], sh[2] + 2 * pw[0], sh[3] + 2 * pw[1]),
            device=dc.device,
            dtype=dc.dtype,
        )

        # Create taper for diffraction pattern dimensions
        taper = window(("tukey", taper_ratio), 128)
        taper = taper[None, :] * taper[:, None]

        # Rescale taper to padded diffraction pattern shape
        taper_tensor = (
            torch.as_tensor(taper, dtype=torch.float32, device=dc.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        taper_padded = (
            torch.nn.functional.interpolate(
                taper_tensor,
                size=padded_dc.shape[-2:],  # Size of diffraction pattern dimensions
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        # Copy original data to center of padded array
        padded_dc[:, :, pw[0] : pw[0] + sh[2], pw[1] : pw[1] + sh[3]] = dc

        # Mirror padding for diffraction pattern dimensions
        # Top and bottom edges
        padded_dc[:, :, : pw[0]] = torch.flip(
            padded_dc[:, :, pw[0] : 2 * pw[0]], dims=(2,)
        )
        padded_dc[:, :, -pw[0] :] = torch.flip(
            padded_dc[:, :, -2 * pw[0] : -pw[0]], dims=(2,)
        )

        # Left and right edges
        padded_dc[:, :, :, : pw[1]] = torch.flip(
            padded_dc[:, :, :, pw[1] : 2 * pw[1]], dims=(3,)
        )
        padded_dc[:, :, :, -pw[1] :] = torch.flip(
            padded_dc[:, :, :, -2 * pw[1] : -pw[1]], dims=(3,)
        )

        # Apply taper to diffraction patterns
        padded_dc *= taper_padded[
            None, None, :, :
        ]  # Broadcast taper to all scan positions

        return Dataset4dstem.from_array(
            padded_dc,
            name=self.name + " (padded and tapered)",
            origin=self.origin,
            sampling=self.sampling,
            units=self.units,
            signal_units=self.signal_units,
            device=self.device,
            meta=self.meta,
        )

    def pad_and_taper_realspace(
        self: "Dataset4dstem", pad_ratio: float = 0.1, taper_ratio: float = 0.17
    ) -> "Dataset4dstem":
        """Pad and taper the dataset.

        Args:
            pad_ratio (float, optional) : The ratio of padding to add to the dataset. Defaults to 0.1.
            taper_ratio (float, optional): The ratio of taper to add to the dataset. Defaults to 0.17.

        Returns:
            Dataset4dstem: The padded and tapered dataset.
        """
        dc = self._array
        sh = dc.shape
        pw = [int(x) for x in (np.array(sh) * pad_ratio).astype(int)]
        padded_dc = torch.zeros(
            (sh[0] + 2 * pw[0], sh[1] + 2 * pw[1], sh[2], sh[3]),
            device=dc.device,
            dtype=dc.dtype,
        )

        taper = window(("tukey", taper_ratio), 128)
        taper = taper[None, :] * taper[:, None]
        # Rescale taper to padded_im shape using PyTorch interpolation (like skimage zoom)
        taper_tensor = (
            torch.as_tensor(taper, dtype=torch.float32, device=dc.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # Add batch and channel dims
        taper_padded = (
            torch.nn.functional.interpolate(
                taper_tensor,
                size=padded_dc.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )  # Remove batch and channel dims

        # t = taper_padded > 0.95

        padded_dc[pw[0] : pw[0] + sh[0], pw[1] : pw[1] + sh[1]] = dc
        # print(f"Padded image shape: {padded_dc.shape}")
        # print(f"Taper shape: {taper_padded.shape}")
        # vis.show_2d(padded_dc[:,:,6,6].cpu().numpy())
        # vis.show_2d(t.cpu().numpy().astype(np.float32) + padded_dc[:,:,6,6].cpu().numpy() / padded_dc[:,:,6,6].cpu().numpy().max())

        padded_dc[: pw[0]] = torch.flip(padded_dc[pw[0] : 2 * pw[0]], dims=(0,))
        padded_dc[-pw[0] :] = torch.flip(padded_dc[-2 * pw[0] : -pw[0]], dims=(0,))
        padded_dc[:, : pw[1]] = torch.flip(padded_dc[:, pw[1] : 2 * pw[1]], dims=(1,))
        padded_dc[:, -pw[1] :] = torch.flip(
            padded_dc[:, -2 * pw[1] : -pw[1]], dims=(1,)
        )

        padded_dc *= taper_padded[..., None, None]

        return Dataset4dstem.from_array(
            padded_dc,
            name=self.name + " (padded and tapered)",
            origin=self.origin,
            sampling=self.sampling,
            units=self.units,
            signal_units=self.signal_units,
            device=self.device,
            meta=self.meta,
        )

    def probe_radius_and_center(
        self,
        thresh_lower: float = 0.1,
        thresh_upper: float = 0.6,
        N: int = 100,
        edge_method: str = "canny",
        min_edge_points: int = 50,
        ransac_iterations: int = 1000,
        ransac_threshold: float = 2.0,
        plot_rbf: bool = True,
    ) -> Tuple[float, NDArray]:
        """
        Gets the center and radius of the probe in the diffraction plane using circle fitting.
        This method is robust to notches and missing sections by fitting a circle to edge points.

        The algorithm:
        1. Create binary masks using multiple thresholds
        2. For each mask, detect edges using specified method
        3. Fit circles to edge points using RANSAC for robustness
        4. Select the best circle based on consensus across thresholds
        5. Refine the result using least-squares fitting

        Args:
            thresh_lower (float): Lower threshold limit (0 to 1)
            thresh_upper (float): Upper threshold limit (0 to 1)
            N (int): Number of thresholds to test
            edge_method (str): Edge detection method ('canny', 'sobel', 'gradient')
            min_edge_points (int): Minimum edge points required for circle fitting
            ransac_iterations (int): Number of RANSAC iterations
            ransac_threshold (float): RANSAC inlier threshold in pixels

        Returns:
            r (float): Central disk radius in pixels
            center (NDArray): [y, x] position of disk center
        """
        thresh_vals = torch.linspace(thresh_lower, thresh_upper, N, device=self.device)

        # Get averaged diffraction pattern
        DP = self._array.mean((0, 1))
        DPmax = torch.max(DP)

        # Convert to numpy for OpenCV operations
        DP_np = (DP / DPmax).cpu().numpy().astype(np.float32)

        circle_candidates = []

        for i, thresh in enumerate(thresh_vals):
            # Create binary mask
            mask = (DP_np > thresh.item()).astype(np.uint8)

            # Detect edges
            edges = detect_edges(mask, method=edge_method)

            # Find edge points
            edge_points = np.column_stack(np.where(edges))

            if len(edge_points) < min_edge_points:
                continue

            # Fit circle using RANSAC
            circle = fit_circle_ransac(
                edge_points, iterations=ransac_iterations, threshold=ransac_threshold
            )

            if circle is not None:
                cy, cx, r = circle
                # Validate circle is reasonable
                h, w = DP_np.shape
                if 0 <= cx <= w and 0 <= cy <= h and r > 5 and r < min(h, w) / 2:
                    circle_candidates.append((cy, cx, r, thresh.item(), i))

        if not circle_candidates:
            # Fallback to original method if no circles found
            return self._fallback_area_method(thresh_lower, thresh_upper, N)

        # Select best circle based on consensus
        best_circle = select_best_circle(circle_candidates, DP_np)

        # Refine with least-squares fitting
        cy, cx, r = refine_circle_fit(best_circle, DP_np, edge_method)
        if plot_rbf:
            import matplotlib as mlp

            _, axs = mlp.pyplot.subplots()
            axs.imshow(DP_np)
            axs.add_patch(
                mlp.patches.Circle(
                    (cx, cy),
                    r,
                    fill=False,
                    linewidth=2,
                )
            )
            mlp.pyplot.savefig("wave.png")
            mlp.pyplot.close()
        print(
            f"Radius and center of the bright field disk (pixels): , {float(r):.2f}, {cx:.2f}, {cy:.2f}"
        )
        return float(r), np.array([cy, cx])

    def _fallback_area_method(
        self, thresh_lower: float, thresh_upper: float, N: int
    ) -> Tuple[float, NDArray]:
        """Fallback to original area-based method if circle fitting fails."""
        # This is the original method from your code
        thresh_vals = torch.linspace(thresh_lower, thresh_upper, N, device=self.device)
        r_vals = torch.zeros(N, device=self.device)

        ind = min(1000, self._array3d.shape[0])
        DP = self._array3d[:ind].mean(0)
        DPmax = torch.max(DP)

        for i in range(len(thresh_vals)):
            thresh = thresh_vals[i]
            mask = DP > DPmax * thresh
            r_vals[i] = torch.sqrt(torch.sum(mask) / torch.pi)

        dr_dtheta = torch.gradient(r_vals, dim=0)[0]
        mask = (dr_dtheta <= 0) * (dr_dtheta >= 2 * torch.median(dr_dtheta))
        r = torch.mean(r_vals[mask])

        thresh = torch.mean(thresh_vals[mask])
        mask = DP > DPmax * thresh
        ar = DP * mask
        nx, ny = ar.shape
        ry, rx = torch.meshgrid(
            torch.arange(ny, device=self.device), torch.arange(nx, device=self.device)
        )
        tot_intens = torch.sum(ar)
        x0 = torch.sum(rx * ar) / tot_intens
        y0 = torch.sum(ry * ar) / tot_intens

        return float(r), np.array([y0.item(), x0.item()])

    @property
    def total_intensity(self) -> float:
        """
        Total intensity of the probe over the dataset.
        """
        if self._total_intensity is None:
            total = 0.0
            for i in range(self._array.shape[0]):
                total += float(self._array[i].to(torch.float64).sum())
            self._total_intensity = total
        return self._total_intensity

    @total_intensity.setter
    def total_intensity(self, value: float) -> None:
        self._total_intensity = value

    @property
    def mean_probe_intensity(self) -> NDArray:
        """
        Mean intensity of the probe over the dataset.
        """
        return self._array.sum(axis=(-2, -1)).mean()

    @property
    def max_probe_intensity(self) -> NDArray:
        """
        Max intensity of the probe over the dataset.
        """
        return self._array.sum(axis=(-2, -1)).max()

    @property
    def fluence(self) -> float:
        """Calculate total electron fluence (electrons per square Angstrom) from total intensity.

        Returns:
            float: Total electron fluence in electrons per square Angstrom.
        """
        scan_area = float(np.prod(self.sampling[:2] * np.array(self._shape[:2])))
        return self.total_intensity / scan_area

    def __len__(self) -> int:
        return len(self._array3d)

    # def __getitem__(self, idx: int) -> Tensor:
    #     return self._array3d[idx]
    def __getitem__(
        self, item: list[int]
    ) -> tuple[int, int, int, list[int], int, Tensor]:
        """
        Expects batched indices in item, so a List
        Expects a 6-tuple as output (batch_index, probe_index, angles_index, r_indices, translation_index, amplitudes_target)
        """
        r_indices = item
        return (
            item[0],
            self.probe_index,
            0,
            r_indices,
            0,
            self._array3d[r_indices],
        )

    def crop(self, index: tuple[slice, ...], clone: bool = False) -> "Dataset4dstem":
        """
        Simple indexing function to return Dataset4dstem view.

        Parameters
        ----------
        index : tuple[slice, ...]
            Index to access a subset of the dataset
        clone : bool
            If True, the array is cloned before returning.

        Returns
        -------
        dataset
            A new Dataset4dstem instance containing the indexed data
        """
        array_view = self.array[index]
        if clone:
            array_view = array_view.clone()
        ndim = array_view.ndim

        # Calculate new origin based on slice info and old origin
        if hasattr(index[0], "start") and index[0].start is not None:
            origin_offset_y = index[0].start
        else:
            origin_offset_y = 0

        if hasattr(index[1], "start") and index[1].start is not None:
            origin_offset_x = index[1].start
        else:
            origin_offset_x = 0

        if hasattr(index[2], "start") and index[2].start is not None:
            origin_offset_z = index[2].start
        else:
            origin_offset_z = 0

        if hasattr(index[3], "start") and index[3].start is not None:
            origin_offset_k = index[3].start
        else:
            origin_offset_k = 0

        new_origin = np.array(self.origin) - np.array(
            [origin_offset_y, origin_offset_x, origin_offset_z, origin_offset_k]
        )
        print(f"New origin: {new_origin}")

        if ndim == 4:
            cls = Dataset4dstem
        else:
            raise ValueError("only 4D slices are supported.")

        return cls.from_array(
            array=array_view,
            name=self.name + str(index),
            origin=new_origin,
            sampling=self.sampling,
            units=self.units,
            signal_units=self.signal_units,
            device=self.device,
            meta=self.meta,
        )

    @property
    def detector_shape(self) -> NDArray:
        """ """
        return np.array(self._shape[-2:])

    @property
    def k_max(self) -> NDArray:
        """Calculate maximum scattering vector magnitude from semiconvergence angle and detector shape.

        Returns:
            float: Maximum scattering vector magnitude in inverse Angstroms.
        """
        return self.sampling[-2:] * self.detector_shape / 2

    @property
    def dr(self) -> NDArray:
        """Calculate real space sampling of the detector from k_max.

        Returns:
            float: Real space sampling of the detector in Angstroms.
        """
        return 1 / (2 * self.k_max)

    @property
    def dk(self) -> NDArray:
        """Calculate reciprocal space sampling of the detector from a bright field radius estimation.

        Returns:
            float: Reciprocal space sampling of the detector in inverse Angstroms.
        """
        rbf, _ = self.probe_radius_and_center()
        return (
            self.meta.semiconvergence_angle / rbf / energy2wavelength(self.meta.energy)
        )

    @classmethod
    def from_array(
        cls,
        array: Any,  # Input can be array-like
        name: str | None = None,
        origin: Union[NDArray, tuple, list, float, int] | None = None,
        sampling: Union[NDArray, tuple, list, float, int] | None = None,
        units: Union[list[str], tuple, list] | None = None,
        signal_units: str = "arb. units",
        meta: Optional[Metadata4dstem] = None,
        transform_to_amplitudes: bool = False,
        fourier_shift_dim: Tuple = None,
        normalize: bool = True,
        clip_neg_values: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> "Dataset4dstem":
        """
        Validates and creates a Dataset from an array.

        Parameters
        ----------
        array: Any
            The array to validate and create a Dataset from.
        name: str | None
            The name of the Dataset.
        origin: Union[NDArray, tuple, list, float, int] | None
            The origin of the Dataset.
        sampling: Union[NDArray, tuple, list, float, int] | None
            The sampling of the Dataset.
        units: Union[list[str], tuple, list] | None
            The units of the Dataset.
        signal_units: str
            The units of the signal.

        Returns
        -------
        Dataset
            A Dataset object with the validated array and metadata.
        """
        validated_array = ensure_valid_array(array, device=device)
        _ndim = validated_array.ndim

        # Set defaults if None
        _name = name if name is not None else f"{_ndim}d dataset"
        _origin = origin if origin is not None else np.zeros(_ndim)
        _sampling = (
            sampling
            if sampling is not None
            else (
                meta.sampling
                if meta is not None and meta.sampling is not None
                else np.ones(_ndim)
            )
        )
        _units = (
            units
            if units is not None
            else (
                meta.units
                if meta is not None and meta.units is not None
                else ["pixels"] * _ndim
            )
        )

        return cls(
            array=validated_array,
            name=_name,
            origin=_origin,
            sampling=_sampling,
            units=_units,
            signal_units=signal_units,
            _token=cls._token,
            meta=meta,
            device=device,
            transform_to_amplitudes=transform_to_amplitudes,
            fourier_shift_dim=fourier_shift_dim,
            normalize=normalize,
            clip_neg_values=clip_neg_values,
        )

    # --- Properties ---
    @property
    def array(self) -> Tensor:
        """The underlying n-dimensional array data. Tensor"""
        return self._array

    @array.setter
    def array(self, value: Tensor) -> None:
        self._array = ensure_valid_array(
            value, dtype=self.dtype, ndim=self.ndim, device=value.device
        )

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def origin(self) -> NDArray:
        return self._origin

    @origin.setter
    def origin(self, value: Union[NDArray, tuple, list, float, int]) -> None:
        self._origin = validate_ndinfo(value, self.ndim, "origin")

    @property
    def sampling(self) -> NDArray:
        return self._sampling

    @sampling.setter
    def sampling(self, value: Union[NDArray, tuple, list, float, int]) -> None:
        self._sampling = validate_ndinfo(value, self.ndim, "sampling")

    @property
    def units(self) -> list[str]:
        return self._units

    @units.setter
    def units(self, value: Union[list[str], tuple, list]) -> None:
        self._units = validate_units(value, self.ndim)

    @property
    def signal_units(self) -> str:
        return self._signal_units

    @signal_units.setter
    def signal_units(self, value: str) -> None:
        self._signal_units = str(value)

    # --- Derived Properties ---
    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> DTypeLike:
        return self.array.dtype

    @property
    def device(self) -> torch.device:
        return self.array.device

    @device.setter
    def device(self, value: torch.device) -> None:
        """Set the device for the array."""
        self._array = self._array.to(value)

    # --- Summaries ---
    def __repr__(self) -> str:
        description = [
            f"Dataset(shape={self.shape}, dtype={self.dtype}, name='{self.name}')",
            f"  sampling: {self.sampling}",
            f"  units: {self.units}",
            f"  signal units: '{self.signal_units}'",
        ]
        return "\n".join(description)

    def __str__(self) -> str:
        description = [
            f"Dataset4dstem named '{self.name}'",
            f"  shape: {self.shape}",
            f"  dtype: {self.dtype}",
            f"  device: {self.device}",
            f"  origin: {self.origin}",
            f"  sampling: {self.sampling}",
            f"  units: {self.units}",
            f"  signal units: '{self.signal_units}'",
        ]
        return "\n".join(description)


class RasterScanningDiffractionDataset(Dataset):
    def __init__(
        self,
        intensity_measurements: Tensor,
        meta: Metadata4D = None,
        convert_to_amplitudes: bool = False,
        probe_index: int = 0,
        angles_index: int = 0,
        translation_index: int = 0,
        device: str = "cpu",
    ) -> None:
        # if len(measurements.shape) < 4:
        #     raise RuntimeError('shape must be 4-dim')
        # self.device = device
        self.meta = meta
        if convert_to_amplitudes:
            self.data: Tensor = torch.sqrt(intensity_measurements).to(device)
        else:
            self.data: Tensor = intensity_measurements.to(device)
        self.error_norm: Tensor = torch.sum(self.data)
        self.probe_index: int = probe_index
        self.angles_index: int = angles_index
        self.translation_index: int = translation_index

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(
        self, item: list[int]
    ) -> tuple[int, int, int, list[int], int, Tensor]:
        """
        Expects batched indices in item, so a List
        Expects a 6-tuple as output (batch_index, probe_index, angles_index, r_indices, translation_index, amplitudes_target)
        """
        r_indices = item
        return (
            item[0],
            self.probe_index,
            self.angles_index,
            r_indices,
            self.translation_index,
            self.data[item],
        )

    @classmethod
    def load(
        cls, file_name: str, device: str = "cpu", data_key: str = "data"
    ) -> "RasterScanningDiffractionDataset":
        with h5py.File(file_name, "r") as f:
            measurements = torch.as_tensor(f[data_key][:], device=device)
            probe_index = 0
            angles_index = 0
            translation_index = 0
            return cls(measurements, probe_index, angles_index, translation_index)


class BatchedPtychoTomographyDataSet(Dataset):
    def __init__(
        self,
        datasets: list[RasterScanningDiffractionDataset],
        batch_size: int = 256,
    ):
        super(BatchedPtychoTomographyDataSet).__init__()
        self.dataset = datasets
        self.batch_size = batch_size
        self.lists: list = []
        for dataset in self.dataset:
            ds = dataset
            # print(len(dataset))
            sampler = BatchSampler(
                SequentialSampler(range(len(ds))),
                batch_size=self.batch_size,
                drop_last=False,
            )
            for batch_indices in sampler:
                self.lists.append(ds[np.array(batch_indices)])

    def __getitem__(
        self, item: list[int]
    ) -> tuple[int, int, int, list[int], int, Tensor]:
        return self.lists[item]

    def __len__(self) -> int:
        return len(self.lists)
