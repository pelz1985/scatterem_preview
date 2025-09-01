import os
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Union
from urllib.error import URLError

import numpy as np
import torch

from scatterem2.datasets.scanning_diffraction import (
    PublicRasterScanningDiffractionDataset,
)
from scatterem2.datasets.utils import check_integrity, download_and_extract_archive
from scatterem2.utils import energy2wavelength
from scatterem2.utils.data import Metadata4D
from scatterem2.utils.data.abberations import Aberrations


class Chen2021Science(PublicRasterScanningDiffractionDataset):
    """`Chen2021 Science Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset .
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. default: True
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = [
        "https://drive.google.com/file/d/",
    ]

    resources = [
        ("Chen2021Science_df0nm.mat", "139KpSonMGZroqHA5VBAfCkbJL9vQIGT3"),
        ("Chen2021Science_df15nm.mat", "1Mo263oPApR27w_2jzMsWMDPKXRT3WqRf"),
        ("Chen2021Science_df20nm.mat", "13eFUSlM3AdVE4H7Rjg7qeTrS14E9ohK9"),
        ("Chen2021Science_df30nm.mat", "1G-mw9hcLfGtPLgvaec5fSfJVdEndjN00"),
    ]

    @property
    def train_data(self) -> torch.Tensor:
        warnings.warn("train_data has been renamed data")
        return self.data

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        defocus_nm: int = 20,
        device: str = "cpu",
        convert_to_amplitudes: bool = True,
        probe_index: int = 0,
        angles_index: int = 0,
        translation_index: int = 0,
    ) -> None:
        """Initialize the Chen2021Science dataset.

        Args:
            root (Union[str, Path]): Root directory of dataset.
            transform (Optional[Callable], optional): A function/transform that takes in a PIL image
                and returns a transformed version. Defaults to None.
            target_transform (Optional[Callable], optional): A function/transform that takes in the
                target and transforms it. Defaults to None.
            download (bool, optional): If True, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again. Defaults to True.
            defocus_nm (int, optional): Defocus value in nanometers. Must be one of [0, 15, 20, 30].
                Defaults to 0.

        Raises:
            ValueError: If defocus_nm is not one of the allowed values.
        """
        if defocus_nm not in [0, 15, 20, 30]:
            raise ValueError("defocus_nm must be one of [0, 15, 20, 30]")
        self.defocus_nm = defocus_nm
        self.device = device
        self.root = root
        self.index = {0: 0, 15: 1, 20: 2, 30: 3}[defocus_nm]
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets = self._load_data()
        self._data_3d = self.data.view(
            self.data.shape[0] * self.data.shape[1],
            self.data.shape[2],
            self.data.shape[3],
        )

        # Material: PrScO3 along [001], thickness ~ 21 nm
        # Diffraction patterns: 128x128x64x64
        # Beam energy: 300 keV
        # Probe-forming semi-angle: 21.4 mrad
        # defocus: -20 nm (overfocus)
        # scan step size: 0.41 Angstrom
        # pixel size in diffraction: 0.823 mrad/pixel
        # Chunks per electron: 580
        energy = 300e3
        scan_step = 0.41
        num_scan_steps = 128
        detector_shape = 128
        dalpha = 0.823e-3
        dk = dalpha / energy2wavelength(energy)

        aberrations = torch.zeros(12)
        aberrations[0] = -float(defocus_nm)
        meta = Metadata4D(
            energy=energy,
            semiconvergence_angle=21.4e-3,
            dk=torch.tensor([dk, dk]),
            rotation=0,
            aberrations=Aberrations(aberrations),
            sample_thickness_guess=210,
            scan_step=torch.tensor([scan_step, scan_step]),
            num_scan_steps=torch.tensor([num_scan_steps, num_scan_steps]),
            detector_shape=torch.tensor([detector_shape, detector_shape]),
        )

        super().__init__(
            root,
            data=self._data_3d,
            meta=meta,
            convert_to_amplitudes=convert_to_amplitudes,
            probe_index=probe_index,
            angles_index=angles_index,
            translation_index=translation_index,
            device=device,
        )

    def _check_legacy_exist(self) -> bool:
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file))
            for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self) -> torch.Tensor:
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(
            os.path.join(self.processed_folder, data_file), weights_only=True
        )

    def _load_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Load the first .mat file from resources
        mat_path = os.path.join(self.raw_folder, self.resources[self.index][0])
        # Use h5py to load the .mat file instead of scipy.io
        import h5py

        mat_path = os.path.join(self.raw_folder, self.resources[self.index][0])

        # Print HDF5 file structure
        # with h5py.File(mat_path, 'r') as f:
        #     def print_structure(name, obj):
        #         print(f"{name}: {obj.shape if hasattr(obj, 'shape') else type(obj)}")
        #     f.visititems(print_structure)

        with h5py.File(mat_path, "r") as f:
            mat_data = f["cbed"][:, :, :, :]
            mat_data = np.pad(mat_data, ((0, 0), (0, 0), (2, 2), (2, 2)))

        # Convert data to torch tensors
        data = torch.as_tensor(mat_data, device=self.device)
        targets = None  # Placeholder targets since not specified

        return data, targets

    def __getitem__(
        self, index: int
    ) -> tuple[int, int, int, list[int], int, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return (0, 0, 0, index, 0, self._data_3d[index], None)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        url = self.resources[self.index][0]

        return os.path.isfile(os.path.join(self.raw_folder, os.path.basename(url)))

    def download(self) -> None:
        """Download the Chen 2021 Science data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        filename, gdrive_id = self.resources[self.index]
        # download files
        # for filename, gdrive_id in self.resources:
        errors = []
        for mirror in self.mirrors:
            url = f"{mirror}{gdrive_id}"
            try:
                download_and_extract_archive(
                    url, download_root=self.raw_folder, filename=filename, md5=None
                )
            except URLError as e:
                errors.append(e)
                continue
            break
        else:
            s = f"Error downloading {filename}:\n"
            for mirror, err in zip(self.mirrors, errors):
                s += f"Tried {mirror}, got:\n{str(err)}\n"
            raise RuntimeError(s)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
