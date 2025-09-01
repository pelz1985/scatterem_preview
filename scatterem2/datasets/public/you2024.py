import os
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Union
from urllib.error import URLError

import torch

from scatterem2.datasets.scanning_diffraction import (
    PublicRasterScanningDiffractionDataset,
)
from scatterem2.datasets.utils import check_integrity, download_and_extract_archive
from scatterem2.utils.data import Metadata4D
from scatterem2.utils.data.abberations import Aberrations


class You2024End2EndPtychoTomography(PublicRasterScanningDiffractionDataset):
    """`You2024 End-to-End PtychoTomography Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    resources = [
        # ("167.zip", "https://zenodo.org/record/13060513/files/167.zip", "864664c3d45a87a35ca577e89c526d1a"),
        # ("169.zip", "https://zenodo.org/record/13060513/files/169.zip", "9fd9549849da24791dc53b38a1c283ac"),
        (
            "171.zip",
            "https://zenodo.org/record/13060513/files/171.zip",
            "f8e004fcd8cc86ba4e05b82080b866b9",
        ),
        (
            "173.zip",
            "https://zenodo.org/record/13060513/files/173.zip",
            "2c342de4eb6d6cb4fdc15cbc1d7a0385",
        ),
        (
            "175.zip",
            "https://zenodo.org/record/13060513/files/175.zip",
            "2ea9d7f06144f512359885d4a3f1b3f9",
        ),
        (
            "177.zip",
            "https://zenodo.org/record/13060513/files/177.zip",
            "2334e016a8f04d8444ceee04a7b4f884",
        ),
        (
            "179.zip",
            "https://zenodo.org/record/13060513/files/179.zip",
            "154b9e5aa52eec648dc439a960ad5795",
        ),
        (
            "183.zip",
            "https://zenodo.org/record/13060513/files/183.zip",
            "dbf83606189cdb65138974e1f7a51273",
        ),
        (
            "184.zip",
            "https://zenodo.org/record/13060513/files/184.zip",
            "b987fcfb0071262656af389051e0a075",
        ),
        (
            "186.zip",
            "https://zenodo.org/record/13060513/files/186.zip",
            "f8d58db4cfb541aa0fdcb68cdb28d944",
        ),
        (
            "188.zip",
            "https://zenodo.org/record/13060513/files/188.zip",
            "8092073cd1c1281f50600395f7a86cda",
        ),
        (
            "190.zip",
            "https://zenodo.org/record/13060513/files/190.zip",
            "767dec19d90aee5ef98363856b5edff3",
        ),
        (
            "192.zip",
            "https://zenodo.org/record/13060513/files/192.zip",
            "9abc4b80b743d26602feab1030dae8b6",
        ),
        (
            "194.zip",
            "https://zenodo.org/record/13060513/files/194.zip",
            "31dac4d10fc5653fcff6805ac9db429c",
        ),
        (
            "196.zip",
            "https://zenodo.org/record/13060513/files/196.zip",
            "fb849a394a3bafc3dcc8b2518cc91174",
        ),
        (
            "198.zip",
            "https://zenodo.org/record/13060513/files/198.zip",
            "9a44b18f42aad54f8c759d99e9d96df5",
        ),
        (
            "200.zip",
            "https://zenodo.org/record/13060513/files/200.zip",
            "f1d7fc503160fbd3c693f1d2146d86ba",
        ),
        (
            "202.zip",
            "https://zenodo.org/record/13060513/files/202.zip",
            "75c866b4d352bebda6c585626f166861",
        ),
        # ("204.zip", "https://zenodo.org/record/13060513/files/204.zip", "fe33ab952b288642d8810314c9086240"),
        # ("206.zip", "https://zenodo.org/record/13060513/files/206.zip", "15730aefb82e9cd044f0fba4ef49b92e"),
        # ("208.zip", "https://zenodo.org/record/13060513/files/208.zip", "5bef58818240c9919ffc7ccf62abe1d2"),
        # ("210.zip", "https://zenodo.org/record/13060513/files/210.zip", "f9d884682e797223d6a69fca6aead6cb"),
        # ("212.zip", "https://zenodo.org/record/13060513/files/212.zip", "5cd2bbc843bca679243d18ea9221c99e"),
        (
            "214.zip",
            "https://zenodo.org/record/13060513/files/214.zip",
            "967887b89764006522ee175019ea7145",
        ),
        (
            "216.zip",
            "https://zenodo.org/record/13060513/files/216.zip",
            "4fbdefc5573cc83a95493affcf4be827",
        ),
        (
            "218.zip",
            "https://zenodo.org/record/13060513/files/218.zip",
            "4fd6921ae87119e0c3370f870b9c98a8",
        ),
        (
            "220.zip",
            "https://zenodo.org/record/13060513/files/220.zip",
            "c94a2c1b7cab279e1e125a761ee40c17",
        ),
        (
            "222.zip",
            "https://zenodo.org/record/13060513/files/222.zip",
            "460b89ceb202dbbf46eb4a5179aaa9bb",
        ),
        (
            "224.zip",
            "https://zenodo.org/record/13060513/files/224.zip",
            "037a332c1075a023fbbc76764c42d38b",
        ),
        (
            "226.zip",
            "https://zenodo.org/record/13060513/files/226.zip",
            "6afa4c187d554f5b521a1d33722e36cc",
        ),
        (
            "228.zip",
            "https://zenodo.org/record/13060513/files/228.zip",
            "6662c2f7fd073ce7cf94625a75942c02",
        ),
        (
            "230.zip",
            "https://zenodo.org/record/13060513/files/230.zip",
            "7d6a4e7b38d587acf5e5b6bca1bc6084",
        ),
        (
            "232.zip",
            "https://zenodo.org/record/13060513/files/232.zip",
            "6e57219a3d1f12b2f6c8bff7120fb902",
        ),
        (
            "234.zip",
            "https://zenodo.org/record/13060513/files/234.zip",
            "e2fb59f625adf8c77c50541d44c40e76",
        ),
        # ("state.tvh5", "https://zenodo.org/record/13060513/files/state.tvh5", "d17de9a9f317d1566651ba3741116211"),
    ]

    # training_file = "ShaSCiAdv2022_1.npy"

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
        # defocus_nm: int = 20,
        device: str = "cpu",
        convert_to_amplitudes: bool = True,
        probe_index: int = 0,
        angles_index: int = 0,
        translation_index: int = 0,
    ) -> None:
        """Initialize the You2024End_to_end dataset.

        Args:
            root (Union[str, Path]): Root directory of dataset.
            transform (Optional[Callable], optional): A function/transform that takes in a PIL image
                and returns a transformed version. Defaults to None.
            target_transform (Optional[Callable], optional): A function/transform that takes in the
                target and transforms it. Defaults to None.
            download (bool, optional): If True, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again. Defaults to True.
            downloading all the datasets, including the state.

        Raises:
            ValueError: If url is not found.
        """
        self.device = device
        self.root = root
        if download:
            self.download()

        if not self._check_exists():
            missing = [
                filename
                for filename, _, _ in self.resources
                if not os.path.exists(os.path.join(self.raw_folder, filename))
            ]
            raise RuntimeError(f"Dataset is incomplete. Missing files: {missing}")

        self.data, self.targets = self._load_data()
        self._data_3d = self.data.view(
            self.data.shape[0] * self.data.shape[1],
            self.data.shape[2],
            self.data.shape[3],
        )

        # Material: Te, tilt range: -53.9 ~ 52 degrees, thickness ~6nm
        # Diffraction patterns: 220x220x66x66
        # Beam energy: 80 keV
        # Probe-forming semi-angle: 25 mrad
        # defocus spread: 6 nm
        # scan step size: 0.397 Angstrom
        # k_max: 1.259 A-1
        defocus_nm = 6
        energy = 80e3
        scan_step = 0.397
        num_scan_steps = 220
        detector_shape = 66
        dk = 1.259 / (detector_shape / 2)

        aberrations = torch.zeros(12)
        aberrations[0] = -float(defocus_nm)
        meta = Metadata4D(
            energy=energy,
            semiconvergence_angle=25e-3,
            dk=torch.tensor([dk, dk]),
            rotation=0,
            aberrations=Aberrations(aberrations),
            sample_thickness_guess=60,
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
        # self.train = train  # training set or test set

        # if download:
        #     self.download()
        #
        # if not self._check_exists():
        #     raise RuntimeError(
        #         "Dataset not found. You can use download=True to download it"
        #     )
        #
        # self.data, self.targets = self._load_data()

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
        import zipfile

        import zarr

        data_list = []
        # targets_list = []
        raw_folder = self.raw_folder

        for filename, _, _ in self.resources:
            zip_path = os.path.join(raw_folder, filename)

            with zipfile.ZipFile(zip_path, "r") as archive:
                extracted_path = os.path.join(raw_folder, filename.replace(".zip", ""))
                archive.extractall(extracted_path)

            # Open the Zarr store
            store = zarr.open(extracted_path, mode="r")
            data = store["/data"][:, :, :, :]  # 4D diffraction pattern

            # Convert to torch tensor
            tensor = torch.from_numpy(data)
            print(f"Loaded {filename} with shape {tensor.shape}")
            data_list.append(tensor)

        data = torch.cat(data_list, dim=0)
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

    def __len__(self) -> int:
        return len(self.data)

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
        return all(
            os.path.isfile(os.path.join(self.raw_folder, filename))
            for filename, _, _ in self.resources
        )

    def download(self) -> None:
        """Download the dataset files if they don't exist already."""

        os.makedirs(self.raw_folder, exist_ok=True)
        for filename, url, md5 in self.resources:
            fpath = os.path.join(self.raw_folder, filename)
            if os.path.exists(fpath):
                continue  # Skip already-downloaded files

            try:
                download_and_extract_archive(
                    url, download_root=self.raw_folder, filename=filename, md5=md5
                )
            except URLError as e:
                raise RuntimeError(f"Failed to download {filename} from {url}: {e}")

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
