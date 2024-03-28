from typing import List, Union, Optional
import os
import torch
from ..transform import transform_points, RigidTransform
from ..image import Slice, Volume, load_volume, load_mask
from .models import INR
from ..utils import resolution2sigma, meshgrid, PathType


def override_sample_mask(
    mask: Volume,
    new_mask: Union[PathType, None, Volume] = None,
    new_resolution: Optional[float] = None,
    new_orientation: Union[PathType, None, Volume, RigidTransform] = None,
) -> Volume:
    if new_mask is not None:
        if isinstance(new_mask, Volume):
            mask = new_mask
        elif isinstance(new_mask, (str, os.PathLike)):
            mask = load_mask(new_mask, device=mask.device)
        else:
            raise TypeError("unknwon type for mask")
    transformation = None
    if new_orientation is not None:
        if isinstance(new_orientation, Volume):
            transformation = new_orientation.transformation
        elif isinstance(new_orientation, RigidTransform):
            transformation = new_orientation
        elif isinstance(new_orientation, (str, os.PathLike)):
            transformation = load_volume(
                new_orientation,
                device=mask.device,
            ).transformation
    if transformation or new_resolution:
        mask = mask.resample(new_resolution, transformation)
    return mask


def sample_volume(
    model: INR,
    mask: Volume,
    psf_resolution: float,
    batch_size: int = 1024,
    n_samples: int = 128,
) -> Volume:
    model.eval()
    img = mask.clone()
    img.image[img.mask] = sample_points(
        model,
        img.xyz_masked,
        psf_resolution,
        batch_size,
        n_samples,
    )
    return img


def sample_points(
    model: INR,
    xyz: torch.Tensor,
    resolution: float = 0,
    batch_size: int = 1024,
    n_samples: int = 128,
) -> torch.Tensor:
    shape = xyz.shape[:-1]
    xyz = xyz.view(-1, 3)
    v = torch.empty(xyz.shape[0], dtype=torch.float32, device=xyz.device)
    with torch.no_grad():
        for i in range(0, xyz.shape[0], batch_size):
            xyz_batch = xyz[i : i + batch_size]
            xyz_batch = model.sample_batch(
                xyz_batch,
                None,
                resolution2sigma(resolution, isotropic=True),
                0 if resolution <= 0 else n_samples,
            )
            v_b = model(xyz_batch).mean(-1)
            v[i : i + batch_size] = v_b
    return v.view(shape)


def sample_slice(
    model: INR,
    slice: Slice,
    mask: Volume,
    output_psf_factor: float = 1.0,
    n_samples: int = 128,
) -> Slice:
    slice_sampled = slice.clone(zero=True)
    xyz = meshgrid(slice_sampled.shape_xyz, slice_sampled.resolution_xyz).view(-1, 3)
    m = mask.sample_points(transform_points(slice_sampled.transformation, xyz)) > 0
    if m.any():
        xyz_masked = model.sample_batch(
            xyz[m],
            slice_sampled.transformation,
            resolution2sigma(
                slice_sampled.resolution_xyz * output_psf_factor, isotropic=False
            ),
            0 if output_psf_factor <= 0 else n_samples,
        )
        v = model(xyz_masked).mean(-1)
        slice_sampled.mask = m.view(slice_sampled.mask.shape)
        slice_sampled.image[slice_sampled.mask] = v.to(slice_sampled.image.dtype)
    return slice_sampled


def sample_slices(
    model: INR,
    slices: List[Slice],
    mask: Volume,
    output_psf_factor: float = 1.0,
    n_samples: int = 128,
) -> List[Slice]:
    model.eval()
    with torch.no_grad():
        slices_sampled = []
        for i, slice in enumerate(slices):
            slices_sampled.append(
                sample_slice(model, slice, mask, output_psf_factor, n_samples)
            )
    return slices_sampled
