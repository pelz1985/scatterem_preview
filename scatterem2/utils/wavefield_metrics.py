"""
Author: Abraham Levitan
Date: April 05, 2022

This file uses pytorch to implement the error metrics defined in the paper:
    "Error Metrics for Partially Coherent Wavefields",
    doi: 10.1364/OL.455955
"""

import numpy as np
import torch as t

__all__ = [
    "square_root_fidelity",
    "partially_coherent_mean_square_error",
    "partially_coherent_frc",
]


def square_root_fidelity(fields_1, fields_2, dims=2):
    """Calculates the square-root-fidelity between two multi-mode wavefields

    The fidelity is a comparison metric between two density matrices
    (i.e. mutual coherence functions) that extends the idea of the
    overlap to incoherent light. As a reminder, the overlap between two
    fields is:

    overlap = abs(sum(field_1 * field_2))

    Whereas the square-root-fidelity is defined as:

    sqrt_fidelity = trace(sqrt(sqrt(dm_1) <dot> dm_2 <dot> sqrt(dm_1)))

    where dm_n refers to the density matrix encoded by fields_n such
    that dm_n = fields_n <dot> fields_n.conjtranspose(), sqrt
    refers to the matrix square root, and <dot> is the matrix product.

    The definition above is not practical, however, as it is not feasible
    to explicitly construct the matrices dm_1 and dm_2 in memory. Therefore,
    we take advantage of the alternate definition based directly on the
    fields_n parameter:

    sqrt_fidelity = sum(svdvals(fields_1 <dot> fields_2.conjtranspose()))

    In the definitions above, the fields_n are regarded as collections of
    wavefields, where each wavefield is by default 2-dimensional. The
    dimensionality of the wavefields can be altered via the dims argument,
    but the fields_n arguments must always have at least one more dimension
    than the dims argument. Any additional dimensions are treated as batch
    dimensions.

    Parameters
    ----------
    fields_1 : t.Tensor
        The first set of complex-valued field modes
    fields_2 : t.Tensor
        The second M2xN set of complex-valued field modes
    dims : int
        Default is 2, the number of final dimensions to reduce over.

    Returns
    -------
    fidelity : float or t.Tensor
        The fidelity, or tensor of fidelities, depending on the dim argument

    """

    # These lines generate the matrix of inner products between all the modes
    mult = fields_1.unsqueeze(-dims - 2) * fields_2.unsqueeze(-dims - 1).conj()
    sumdims = tuple(d - dims for d in range(dims))
    mat = t.sum(mult, dim=sumdims)

    # The nuclear norm is the sum of the singular values
    return t.linalg.matrix_norm(mat, ord="nuc")


def partially_coherent_mean_square_error(fields_1, fields_2, reduction="mean", dims=2):
    """Calculates the PCMSE between two complex partially coherent wavefields

    This function calculates a generalization of the RMS error which uses the
    concept of fidelity to capture the error between incoherent wavefields.
    The extension has several nice properties, in particular:

    1) For coherent wavefields, it precisely matches the magnitude of the
       RMS error.
    2) All mode decompositions of either field that correspond to the same
       density matrix / mutual coherence function will produce the same
       output
    3) The error will only be zero when comparing mode decompositions that
       correspond to the same density matrix.
    4) Due to (2), one need not worry about the ordering of the modes,
       properly orthogonalizing the modes, and it is even possible to
       compare mode decompositions with different numbers of modes.

    The formal definition of this function, with default options, is:

    output = ( sum(abs(fields_1)**2) + sum(abs(fields_2)**2)
              - 2 * sqrt_fidelity(fields_1,fields_2) ) / npix

    Where npix is the number of pixels in the wavefields. If the reduction is
    specified as 'sum', then the result is not divided by this constant.

    In the definitions above, the fields_n are regarded as collections of
    wavefields, where each wavefield is by default 2-dimensional. The
    dimensionality of the wavefields can be altered via the dims argument,
    but the fields_n arguments must always have at least one more dimension
    than the dims argument. Any additional dimensions are treated as batch
    dimensions.

    Parameters
    ----------
    fields_1 : t.Tensor
        The first set of complex-valued field modes
    fields_2 : t.Tensor
        The second set of complex-valued field modes
    normalize : bool
        Default is False, whether to normalize to the intensity of fields_1
    dims : (int or tuple of python:ints)
        Default is 2, the number of final dimensions to reduce over.

    Returns
    -------
    rms_error : float or t.Tensor
        The generalized RMS error, or tensor of generalized RMS errors,
        depending on the dim argument
    """

    sumdims = tuple(d - dims - 1 for d in range(dims + 1))
    fields_1_intensity = t.sum(t.abs(fields_1) ** 2, dim=sumdims)
    fields_2_intensity = t.sum(t.abs(fields_2) ** 2, dim=sumdims)
    sqrt_fidelity = square_root_fidelity(fields_1, fields_2, dims=dims)

    result = fields_1_intensity + fields_2_intensity - 2 * sqrt_fidelity

    if reduction.strip().lower() == "mean":
        # The number of pixels in the wavefield
        npix = t.prod(t.as_tensor(fields_1.shape[-dims:], dtype=t.int32))
        return result / npix
    elif reduction.strip().lower() == "sum":
        return result
    else:
        raise ValueError("The only valid reductions are 'mean' and 'sum'")


def partially_coherent_frc(fields_1, fields_2, bins):
    """Calculates the PCFRC between two complex partially coherent wavefields

    This function assumes that the fields are input in the form of stacked
    2D images with dimensions MxN1xN2. M is the number of coherent modes,
    and N1 and N2 are the dimensions of the images in I and J. While the
    image sizes must match, the number of modes need not be equivalent.

    The returned correlation is a function of spatial frequency, which is
    measured in units of the inverse pixel size.

    Parameters
    ----------
    im1 : t.Tensor
        The first image, a set of complex or real valued arrays
    im2 : t.Tensor
        The first image, a stack of complex or real valued arrays
    bins : int
        Number of bins to break the FRC up into

    Returns
    -------
    freqs : array
        The frequencies associated with each FRC value
    FRC : array
        The FRC values

    """
    # We Fourier transform the two wavefields
    f1 = t.fft.fftshift(t.fft.fft2(fields_1), dim=(-1, -2))
    f2 = t.fft.fftshift(t.fft.fft2(fields_2), dim=(-1, -2))

    # And we generate the associated 2d map of spatial frequencies
    i_freqs = t.fft.fftshift(t.fft.fftfreq(f1.shape[-2]))
    j_freqs = t.fft.fftshift(t.fft.fftfreq(f1.shape[-1]))
    Js, Is = t.meshgrid(j_freqs, i_freqs)
    Rs = t.sqrt(Is**2 + Js**2)

    # These lines get a set of spatial frequency bins that match the logic
    # used by np.histogram.
    n_pix, bins = np.histogram(Rs, bins=bins)
    bins = t.as_tensor(bins)

    frc = []
    for i in range(len(bins) - 1):
        # This implements the projection operator to the appropriate ring
        mask = t.logical_and(Rs < bins[i + 1], Rs >= bins[i])
        masked_f1 = f1 * mask[..., :, :]
        masked_f2 = f2 * mask[..., :, :]

        # And we calculate the sqrt_fidelity of the projected wavefields
        numerator = square_root_fidelity(masked_f1, masked_f2)

        denominator_f1 = t.sum(t.abs(masked_f1) ** 2)
        denominator_f2 = t.sum(t.abs(masked_f2) ** 2)
        frc.append(numerator / t.sqrt((denominator_f1 * denominator_f2)))

    frc = t.as_tensor(np.array(frc))

    return bins[:-1], frc
