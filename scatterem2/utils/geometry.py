from __future__ import annotations

import torch
from torch import Tensor


def get_translation_matrix_3D(yx_translation: Tensor) -> Tensor:
    """
    Create a 3D translation matrix from y,x translations.

    Parameters
    ----------
    yx_translation : Tensor
        Tensor of shape (n_translations, 2) containing [y, x] translations

    Returns
    -------
    Tensor
        Translation matrix of shape (n_translations, 3, 4)
    """
    n_translations = yx_translation.size(0)
    res = torch.eye(n=3, m=4, device=yx_translation.device).unsqueeze(0)
    res = res.repeat((n_translations, 1, 1))
    res[:, 1, -1] = yx_translation[:, 0]
    res[:, 0, -1] = yx_translation[:, 1]
    return res


def affine_matrix_3D_ZYX(
    phi: Tensor, theta: Tensor, psi: Tensor, translation: Tensor
) -> Tensor:
    """
    Create a 3D affine transformation matrix from ZYX Euler angles and translation.

    Parameters
    ----------
    phi : Tensor
        Rotation angle around Z axis
    theta : Tensor
        Rotation angle around Y axis
    psi : Tensor
        Rotation angle around X axis
    translation : Tensor
        Translation vector of shape (3,)

    Returns
    -------
    Tensor
        Affine transformation matrix of shape (3, 4)
    """
    c1 = torch.cos(phi)
    s1 = torch.sin(phi)
    c2 = torch.cos(theta)
    s2 = torch.sin(theta)
    c3 = torch.cos(psi)
    s3 = torch.sin(psi)
    line1 = torch.stack(
        [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2, translation[[0]]], 1
    )
    line2 = torch.stack(
        [
            c2 * s1,
            c1 * c3 + s1 * s2 * s3,
            c3 * s1 * s2 - c1 * s3,
            torch.zeros_like(translation[[1]]),
        ],
        1,
    )
    line3 = torch.stack([-s2, c2 * s3, c2 * c3, translation[[1]]], 1)
    R = torch.stack([line1, line2, line3], 1)
    return R


def rotate_ZYX(phi: Tensor, theta: Tensor, psi: Tensor) -> Tensor:
    """
    Create a 3D rotation matrix from ZYX Euler angles.

    Parameters
    ----------
    phi : Tensor
        Rotation angle around Z axis
    theta : Tensor
        Rotation angle around Y axis
    psi : Tensor
        Rotation angle around X axis

    Returns
    -------
    Tensor
        Rotation matrix of shape (3, 3)
    """
    c1 = torch.cos(phi)
    s1 = torch.sin(phi)
    c2 = torch.cos(theta)
    s2 = torch.sin(theta)
    c3 = torch.cos(psi)
    s3 = torch.sin(psi)
    line1 = torch.stack([c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2])
    line2 = torch.stack([c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3])
    line3 = torch.stack([-s2, c2 * s3, c2 * c3])
    R = torch.stack([line1, line2, line3], 1)
    return R


def rotate_XZY(phi: Tensor, theta: Tensor, psi: Tensor) -> Tensor:
    """
    Create a 3D rotation matrix from XZY Euler angles.

    Parameters
    ----------
    phi : Tensor
        Rotation angle around X axis
    theta : Tensor
        Rotation angle around Z axis
    psi : Tensor
        Rotation angle around Y axis

    Returns
    -------
    Tensor
        Rotation matrix of shape (3, 3)
    """
    c1 = torch.cos(phi)
    s1 = torch.sin(phi)
    c2 = torch.cos(theta)
    s2 = torch.sin(theta)
    c3 = torch.cos(psi)
    s3 = torch.sin(psi)
    line1 = torch.stack([c2 * c3, -s2, c2 * s3])
    line2 = torch.stack([s1 * s3 + c1 * c3 * s2, c1 * c2, c1 * s2 * s3 - c3 * s1])
    line3 = torch.stack([c3 * s1 * s2 - c1 * s3, c2 * s1, c1 * c3 + s1 * s2 * s3])
    R = torch.stack([line1, line2, line3], 1)
    return R


def rotate_XYZ(phi: Tensor, theta: Tensor, psi: Tensor) -> Tensor:
    """
    Create a 3D rotation matrix from XYZ Euler angles (intrinsic rotations).
    The final rotation is Rz(psi) * Ry(theta) * Rx(phi).

    Parameters
    ----------
    phi : Tensor
        Rotation angle around the X axis
    theta : Tensor
        Rotation angle around the Y axis
    psi : Tensor
        Rotation angle around the Z axis

    Returns
    -------
    Tensor
        Rotation matrix of shape (3, 3)
    """
    c1 = torch.cos(psi)
    s1 = torch.sin(psi)
    c2 = torch.cos(theta)
    s2 = torch.sin(theta)
    c3 = torch.cos(phi)
    s3 = torch.sin(phi)

    line1 = torch.stack([c2 * c3, -c2 * s3, s2])

    line2 = torch.stack([c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1])

    line3 = torch.stack([s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2])

    R = torch.stack([line1, line2, line3], dim=1)
    return R


if __name__ == "__main__":
    a, b, c = torch.rand(3)
    m1 = rotate_ZYX(a, b, c)
    m2 = rotate_XYZ(-a, -b, -c)
    res = m1 @ m2
    res[torch.abs(res) < 1e-7] = 0
    print(res)
