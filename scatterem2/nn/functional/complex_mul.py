from typing import Tuple

from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


class _ComplexMulMultiModeFunction(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        object_patches: Tensor,
        wave: Tensor,
    ) -> Tensor:
        """
        :param object_patches:              1 x K x M1 x M2
        :param wave:                         Nmodes x K x M1 x M2

        :return: Nmodes x K x M1 x M2
        """
        ctx.save_for_backward(object_patches, wave)
        return object_patches * wave

    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Tensor | None, ...]:
        # psi:        Nmodes x K x M1 x M2
        # S_split:    1      x K x M1 x M2
        # grad_output Nmodes x K x M1 x M2

        object_patches, wave = ctx.saved_tensors
        grad_object_patches = None
        grad_waves = None

        if object_patches.requires_grad:
            grad_object_patches = grad_output * wave.conj()
        if wave.requires_grad:
            grad_waves = grad_output * object_patches.conj()

        return grad_object_patches, grad_waves, None


def complex_mul_multi_mode_function(object_patches: Tensor, wave: Tensor) -> Tensor:
    return _ComplexMulMultiModeFunction.apply(object_patches, wave)
