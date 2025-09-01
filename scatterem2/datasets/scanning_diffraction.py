import os
import os.path
from pathlib import Path
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import torch

from scatterem2.utils.data import Metadata4D, RasterScanningDiffractionDataset

USER_AGENT = "scatterem"


class PublicRasterScanningDiffractionDataset(RasterScanningDiffractionDataset):
    """
    Base Class For making datasets which are compatible with scatterem.
    It is necessary to override the ``__getitem__`` and ``__len__`` method.

    Args:
        root (string, optional): Root directory of dataset. Only used for `__repr__`.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    .. note::

        :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.
    """

    _repr_indent = 4

    def __init__(
        self,
        root: Union[str, Path] = None,  # type: ignore[assignment]
        data: torch.Tensor = None,
        meta: Metadata4D = None,
        convert_to_amplitudes: bool = True,
        probe_index: int = 0,
        angles_index: int = 0,
        translation_index: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            data,
            meta,
            convert_to_amplitudes,
            probe_index,
            angles_index,
            translation_index,
            device,
        )

        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root

    def __getitem__(
        self, index: int
    ) -> tuple[int, int, int, list[int], int, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + [
            "{}{}".format(" " * len(head), line) for line in lines[1:]
        ]

    def extra_repr(self) -> str:
        return ""


class StandardTransform:
    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + [
            "{}{}".format(" " * len(head), line) for line in lines[1:]
        ]

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(
                self.target_transform, "Target transform: "
            )

        return "\n".join(body)
