from typing import Any, Tuple

import warp as wp
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


@wp.kernel
def _amplitude_loss_kernel(
    a_model: wp.array(dtype=Any, ndim=3),
    a_target: wp.array(dtype=Any, ndim=3),
    loss: wp.array(dtype=wp.float32, ndim=1),
) -> None:
    """

    :param z:           K x My x Mx x 2
    :param z_hat:       K x My x Mx x 2
    :param a:           K x My x Mx
    :param beta:        1
    :param a_strides:   (4,)
    :return:
    """
    k, my, mx = wp.tid()
    # grad[k, my, mx] = a_target[k, my, mx]  # 1 - (a_target[k, my, mx] / (a_model[k, my, mx]+1e-6))
    wp.atomic_add(loss, k, wp.abs(a_model[k, my, mx] - a_target[k, my, mx]) ** 2.0)


def _amplitude_loss(a_model: Tensor, a_target: Tensor) -> Tensor:
    device = wp.device_from_torch(a_model.device)

    loss = wp.zeros((a_model.shape[0],), dtype=wp.float32, device=device)
    a_model = wp.from_torch(a_model.data)
    a_target = wp.from_torch(a_target.data)
    wp.launch(
        kernel=_amplitude_loss_kernel,
        dim=(a_model.shape[0], a_model.shape[1], a_model.shape[2]),
        inputs=[a_model, a_target, loss],
        device=device,
    )
    return wp.to_torch(loss)


class _AmplitudeLoss(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, a_model: Tensor, a_target: Tensor) -> Tensor:
        loss = _amplitude_loss(a_model.data, a_target.data)
        loss.requires_grad = True
        ctx.save_for_backward(a_target)
        return loss

    @staticmethod
    def backward(
        ctx: FunctionCtx, *grad_outputs: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, Tensor]:
        (grad_a_model,) = ctx.saved_tensors
        grad_a_target = None
        # print(f'AmplitudeLoss.backward any NaN: {torch.any(torch.isnan(grad_a_model))}')
        return grad_a_model, grad_a_target


def amplitude_loss(a_model: Tensor, a_target: Tensor) -> Tensor:
    return _AmplitudeLoss.apply(a_model, a_target)
