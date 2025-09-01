import os
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Union
from urllib.error import URLError

import torch
from PIL import Image

from scatterem2.datasets.scanning_diffraction import (
    PublicRasterScanningDiffractionDataset,
)
from scatterem2.datasets.utils import check_integrity, download_and_extract_archive


class ShaSciAdv2022_1(PublicRasterScanningDiffractionDataset):
    """`ShaSciAdv2022 Dataset 1.

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

    mirrors = [
        "https://drive.google.com/file/d/",
    ]

    resources = [
        ("ShaSCiAdv2022_1.npy", "1T91K82EuatWtQNA1KX11DlxZOPcIthUU"),
    ]

    training_file = "ShaSCiAdv2022_1.npy"

    @property
    def train_data(self) -> torch.Tensor:
        warnings.warn("train_data has been renamed data")
        return self.data

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets = self._load_data()

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
        # image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        # data = None

        # label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        # targets = None

        return None, None

    def __getitem__(
        self, index: int
    ) -> tuple[int, int, int, list[int], int, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

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
            os.path.isfile(
                os.path.join(
                    self.raw_folder, os.path.splitext(os.path.basename(url))[0]
                )
            )
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            errors = []
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    download_and_extract_archive(
                        url, download_root=self.raw_folder, filename=filename, md5=md5
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
