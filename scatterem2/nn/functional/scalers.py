from typing import Tuple

from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


class _L2ScalingFunction(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        scaling_parameter: Tensor,
        eps: float = 1e-7,
    ) -> Tensor:
        ctx.save_for_backward(x, scaling_parameter)
        return x / (scaling_parameter + eps)

    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Tensor | None, ...]:
        x, scaling_parameter = ctx.saved_tensors
        grad_x = None
        grad_scaling_parameter = None

        if x.requires_grad:
            grad_x = scaling_parameter * grad_output
        if scaling_parameter.requires_grad:
            grad_scaling_parameter = (
                0.5 * scaling_parameter * grad_output**2 - x * grad_output
            )

        return grad_x, 1e5 * grad_scaling_parameter, None


def l2_scaling_function(
    x: Tensor, scaling_parameter: Tensor, eps: float = 1e-7
) -> Tensor:
    return _L2ScalingFunction.apply(x, scaling_parameter, eps)
