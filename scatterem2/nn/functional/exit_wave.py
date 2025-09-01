import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


class _ExitwaveMeasurementSingleMode(Function):
    """Absolute value class for autograd"""

    @staticmethod
    def forward(ctx: FunctionCtx, wave_fourier: Tensor) -> Tensor:
        """

        :param ctx:
        :param tensor_in: Nmodes, K, MY, MX
        :return: wave
        """

        amplitude_fourier = torch.abs(wave_fourier.detach()[0])
        af = amplitude_fourier
        af[af == 0.0] = np.inf
        ctx.save_for_backward(af, wave_fourier)
        return amplitude_fourier

    @staticmethod
    def backward(ctx: FunctionCtx, amplitudes_target: Tensor) -> Tensor:
        """

        :param ctx:
        :param grad_output: 2, Nmodes, K, MY, MX
        :return:
        """
        af, wave_fourier = ctx.saved_tensors
        amplitude_modification = amplitudes_target / af
        fourier_modified_overlap = amplitude_modification[None] * wave_fourier
        grad_tensor_in = wave_fourier - fourier_modified_overlap
        return grad_tensor_in


def exitwave_measurement_single_mode(wave_fourier: Tensor) -> Tensor:
    return _ExitwaveMeasurementSingleMode.apply(wave_fourier)


class _ExitwaveMeasurementMultiMode(Function):
    """Absolute value class for autograd"""

    @staticmethod
    def forward(ctx: FunctionCtx, wave_fourier: Tensor) -> Tensor:
        """

        :param ctx:
        :param tensor_in: Nmodes, K, MY, MX
        :return: wave
        """

        amplitude_fourier = torch.sqrt(torch.sum(torch.abs(wave_fourier) ** 2, 0))
        af = amplitude_fourier.detach().clone()
        af[af == 0.0] = np.inf
        ctx.save_for_backward(af, wave_fourier)
        return amplitude_fourier

    @staticmethod
    def backward(ctx: FunctionCtx, amplitudes_target: Tensor) -> Tensor:
        """

        :param ctx:
        :param grad_output: 2, Nmodes, K, MY, MX
        :return:
        """
        af, wave_fourier = ctx.saved_tensors
        amplitude_modification = amplitudes_target / af
        fourier_modified_overlap = amplitude_modification[None] * wave_fourier
        grad_tensor_in = wave_fourier - fourier_modified_overlap
        return grad_tensor_in


def exitwave_measurement_multi_mode(wave_fourier: Tensor) -> Tensor:
    return _ExitwaveMeasurementMultiMode.apply(wave_fourier)
