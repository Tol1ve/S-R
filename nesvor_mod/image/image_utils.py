from typing import Tuple, Union, Optional
import os
import nibabel as nib
import torch
import numpy as np
from ..transform import RigidTransform
from ..utils import PathType


def compare_resolution_affine(r1, a1, r2, a2, s1, s2) -> bool:
    r1 = np.array(r1)
    a1 = np.array(a1)
    r2 = np.array(r2)
    a2 = np.array(a2)
    if s1 != s2:
        return False
    if r1.shape != r2.shape:
        return False
    if np.amax(np.abs(r1 - r2)) > 1e-3:
        return False
    if a1.shape != a2.shape:
        return False
    if np.amax(np.abs(a1 - a2)) > 1e-3:
        return False
    return True


def affine2transformation(
    volume: torch.Tensor,
    mask: torch.Tensor,
    resolutions: np.ndarray,
    affine: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, RigidTransform]:
    device = volume.device
    d, h, w = volume.shape

    R = affine[:3, :3]
    negative_det = np.linalg.det(R) < 0

    T = affine[:3, -1:]  # T = R @ (-T0 + T_r)
    R = R @ np.linalg.inv(np.diag(resolutions))

    T0 = np.array([(w - 1) / 2 * resolutions[0], (h - 1) / 2 * resolutions[1], 0])
    T = np.linalg.inv(R) @ T + T0.reshape(3, 1)

    tz = (
        torch.arange(0, d, device=device, dtype=torch.float32) * resolutions[2]
        + T[2].item()
    )
    tx = torch.ones_like(tz) * T[0].item()
    ty = torch.ones_like(tz) * T[1].item()
    t = torch.stack((tx, ty, tz), -1).view(-1, 3, 1)
    R = torch.tensor(R, device=device).unsqueeze(0).repeat(d, 1, 1)

    if negative_det:
        volume = torch.flip(volume, (-1,))
        mask = torch.flip(mask, (-1,))
        t[:, 0, -1] *= -1
        R[:, :, 0] *= -1

    transformation = RigidTransform(
        torch.cat((R, t), -1).to(torch.float32), trans_first=True
    )

    return volume, mask, transformation


def transformation2affine(
    volume: torch.Tensor,
    transformation: RigidTransform,
    resolution_x: float,
    resolution_y: float,
    resolution_z: float,
) -> np.ndarray:
    mat = transformation.matrix(trans_first=True).detach().cpu().numpy()
    assert mat.shape[0] == 1
    R = mat[0, :, :-1]
    T = mat[0, :, -1:]
    d, h, w = volume.shape
    affine = np.eye(4)
    T[0] -= (w - 1) / 2 * resolution_x
    T[1] -= (h - 1) / 2 * resolution_y
    T[2] -= (d - 1) / 2 * resolution_z
    T = R @ T.reshape(3, 1)
    R = R @ np.diag([resolution_x, resolution_y, resolution_z])
    affine[:3, :] = np.concatenate((R, T), -1)
    return affine


def save_nii_volume(
    path: PathType,
    volume: Union[torch.Tensor, np.ndarray],
    affine: Optional[Union[torch.Tensor, np.ndarray]],
) -> None:
    assert len(volume.shape) == 3 or (len(volume.shape) == 4 and volume.shape[1] == 1)
    if len(volume.shape) == 4:
        volume = volume.squeeze(1)
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy().transpose(2, 1, 0)
    else:
        volume = volume.transpose(2, 1, 0)
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    if affine is None:
        affine = np.eye(4)
    if volume.dtype == bool and isinstance(
        volume, np.ndarray
    ):  # bool type is not supported
        volume = volume.astype(np.int16)
    img = nib.nifti1.Nifti1Image(volume, affine)
    img.header.set_xyzt_units(2)
    img.header.set_qform(affine, code="aligned")
    img.header.set_sform(affine, code="scanner")
    nib.save(img, os.fspath(path))


def load_nii_volume(path: PathType) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = nib.load(os.fspath(path))

    dim = img.header["dim"]
    assert dim[0] == 3 or (dim[0] > 3 and all(d == 1 for d in dim[4:])), (
        "Expect a 3D volume but the input is %dD" % dim[0]
    )

    volume = img.get_fdata().astype(np.float32)
    while volume.ndim > 3:
        volume = volume.squeeze(-1)
    volume = volume.transpose(2, 1, 0)

    resolutions = img.header["pixdim"][1:4]

    affine = img.affine
    if np.any(np.isnan(affine)):
        affine = img.get_qform()

    return volume, resolutions, affine
