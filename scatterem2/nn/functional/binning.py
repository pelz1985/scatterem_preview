from typing import Tuple

from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


class _ScaleReversible(Function):
    """
    Class that bins the object along the direction of beam propagation (z)
    inputs:
    obj_in: input object
    factor: factor at which the object will be binned
    """

    @staticmethod
    def forward(ctx: FunctionCtx, wave_in: Tensor, factor: float) -> Tensor:
        ctx.factor = factor
        if factor == 1:
            return wave_in
        return wave_in * factor

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Tensor, None]:
        factor = ctx.factor
        if factor == 1:
            return grad_output, None

        return grad_output / factor, None


def scale_reversible(wave_in: Tensor, factor: float) -> Tensor:
    return _ScaleReversible.apply(wave_in, factor)


class _BinObject(Function):
    """
    Class that bins the object along the direction of beam propagation (z)
    inputs:
    obj_in: input object
    factor: factor at which the object will be binned
    """

    @staticmethod
    def forward(ctx: FunctionCtx, obj_in: Tensor, factor: float) -> Tensor:
        assert (obj_in.shape[2] % factor) == 0
        assert len(obj_in.shape) == 3
        ctx.factor = factor
        if factor == 1:
            return obj_in
        n_z, n_y, n_x = obj_in.shape
        obj_out = obj_in.reshape(n_z // factor, factor, n_y, n_x).sum(1)
        return obj_out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Tensor, None]:
        factor = ctx.factor
        if factor == 1:
            return grad_output, None

        return grad_output.repeat_interleave(factor, dim=0), None


def bin_object(obj_in: Tensor, factor: float) -> Tensor:
    return _BinObject.apply(obj_in, factor)
