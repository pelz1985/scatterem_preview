import torch
from torch import Tensor
from torch.nn import Module

from scatterem2.nn.functional import batch_crop, overlap_intensity


class BatchCrop(Module):
    def __init__(self, obj: Tensor) -> None:
        super().__init__()
        self.object_norm = torch.zeros(
            obj.shape[1:], device=obj.device, dtype=torch.float32, requires_grad=False
        )

    def forward(self, obj: Tensor, waves: Tensor, pos: Tensor) -> Tensor:
        patch_norm = torch.sum(torch.abs(waves.detach()) ** 2, 0).squeeze()
        # calculate normalization for object gradient
        overlap_intensity(pos.data, patch_norm.data, self.object_norm.data)
        return batch_crop(obj, waves, pos)
