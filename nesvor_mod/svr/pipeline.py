import logging
from typing import List, Optional, Tuple, cast
import torch
import numpy as np
from .registration import SliceToVolumeRegistration
from .outlier import EM, global_ncc_exclusion, local_ssim_exclusion
from .reconstruction import (
    psf_reconstruction,
    srr_update,
    simulate_slices,
    slices_scale,
    simulated_error,
)
from ..utils import DeviceType, PathType, get_PSF
from ..image import Volume, Slice, load_volume, load_mask, Stack
from ..inr.data import PointDataset


def _initial_mask(
    slices: List[Slice],
    output_resolution: float,
    sample_mask: Optional[PathType],
    sample_orientation: Optional[PathType],
    device: DeviceType,
) -> Tuple[Volume, bool]:
    dataset = PointDataset(slices)
    mask = dataset.mask
    if sample_mask is not None:
        mask = load_mask(sample_mask, device)
    transformation = None
    if sample_orientation is not None:
        transformation = load_volume(
            sample_orientation,
            device=device,
        ).transformation
    mask = mask.resample(output_resolution, transformation)
    mask.mask = mask.image > 0
    return mask, sample_mask is None


def _check_resolution_and_shape(slices: List[Slice]) -> List[Slice]:
    res_inplane = []
    thicknesses = []
    for s in slices:
        res_inplane.append(float(s.resolution_x))
        res_inplane.append(float(s.resolution_y))
        thicknesses.append(float(s.resolution_z))

    res_s = min(res_inplane)
    s_thick = np.mean(thicknesses).item()
    slices = [s.resample((res_s, res_s, s_thick)) for s in slices]
    slices = Stack.pad_stacks(slices)

    if max(thicknesses) - min(thicknesses) > 0.001:
        logging.warning("The input data have different thicknesses!")

    return slices


def _normalize(
    stack: Stack, output_intensity_mean: float
) -> Tuple[Stack, float, float]:
    masked_v = stack.slices[stack.mask]
    mean_intensity = masked_v.mean().item()
    max_intensity = masked_v.max().item()
    min_intensity = masked_v.min().item()
    stack.slices = stack.slices * (output_intensity_mean / mean_intensity)
    max_intensity = max_intensity * (output_intensity_mean / mean_intensity)
    min_intensity = min_intensity * (output_intensity_mean / mean_intensity)
    return stack, max_intensity, min_intensity


def slice_to_volume_reconstruction(
    slices: List[Slice],
    *,
    with_background: bool = False,
    output_resolution: float = 0.8,
    output_intensity_mean: float = 700,
    delta: float = 150 / 700,
    n_iter: int = 3,
    n_iter_rec: List[int] = [7, 7, 21],
    global_ncc_threshold: float = 0.5,
    local_ssim_threshold: float = 0.4,
    no_slice_robust_statistics: bool = False,
    no_pixel_robust_statistics: bool = False,
    no_global_exclusion: bool = False,
    no_local_exclusion: bool = False,
    sample_mask: Optional[PathType] = None,
    sample_orientation: Optional[PathType] = None,
    psf: str = "gaussian",
    device: DeviceType = torch.device("cpu"),
    **unused
) -> Tuple[Volume, List[Slice], List[Slice]]:
    # check data
    slices = _check_resolution_and_shape(slices)
    stack = Stack.cat(slices)
    slices_mask_backup = stack.mask.clone()

    # init volume
    volume, is_refine_mask = _initial_mask(
        slices,
        output_resolution,
        sample_mask,
        sample_orientation,
        device,
    )

    # data normalization
    stack, max_intensity, min_intensity = _normalize(stack, output_intensity_mean)

    # define psf
    psf_tensor = get_PSF(
        res_ratio=(
            stack.resolution_x / output_resolution,
            stack.resolution_y / output_resolution,
            stack.thickness / output_resolution,
        ),
        device=volume.device,
        psf_type=psf,
    )

    # outer loop
    for i in range(n_iter):
        logging.info("outer %d", i)
        # slice-to-volume registration
        if i > 0:  # skip slice-to-volume registration for the first iteration
            svr = SliceToVolumeRegistration(
                num_levels=3,
                num_steps=5,
                step_size=2,
                max_iter=30,
            )
            slices_transform, _ = svr(
                stack,
                volume,
                use_mask=True,
            )
            stack.transformation = slices_transform

        # global structual exclusion
        if i > 0 and not no_global_exclusion:
            stack.mask = slices_mask_backup.clone()
            excluded = global_ncc_exclusion(stack, volume, global_ncc_threshold)
            stack.mask[excluded] = False
        # PSF reconstruction & volume mask
        volume = psf_reconstruction(
            stack,
            volume,
            update_mask=is_refine_mask,
            use_mask=not with_background,
            psf=psf_tensor,
        )

        # init EM
        em = EM(max_intensity, min_intensity)
        p_voxel = torch.ones_like(stack.slices)
        # super-resolution reconstruction (inner loop)
        for j in range(n_iter_rec[i]):
            logging.info("inner %d", j)
            # simulate slices
            slices_sim, slices_weight = cast(
                Tuple[Stack, Stack],
                simulate_slices(
                    stack,
                    volume,
                    return_weight=True,
                    use_mask=not with_background,
                    psf=psf_tensor,
                ),
            )
            # scale
            scale = slices_scale(stack, slices_sim, slices_weight, p_voxel, True)
            # err
            err = simulated_error(stack, slices_sim, scale)
            # EM robust statistics
            if (not no_pixel_robust_statistics) or (not no_slice_robust_statistics):
                p_voxel, p_slice = em(err, slices_weight, scale, 1)
                if no_pixel_robust_statistics:  # reset p_voxel
                    p_voxel = torch.ones_like(stack.slices)
            p = p_voxel
            if not no_slice_robust_statistics:
                p = p_voxel * p_slice.view(-1, 1, 1, 1)
            # local structural exclusion
            if not no_local_exclusion:
                p = p * local_ssim_exclusion(stack, slices_sim, local_ssim_threshold)
            # super-resolution update
            beta = max(0.01, 0.08 / (2**i))
            alpha = min(1, 0.05 / beta)
            volume = srr_update(
                err,
                volume,
                p,
                alpha,
                beta,
                delta * output_intensity_mean,
                use_mask=not with_background,
                psf=psf_tensor,
            )

    # reconstruction finished
    # prepare outputs
    slices_sim = cast(
        Stack,
        simulate_slices(
            stack, volume, return_weight=False, use_mask=True, psf=psf_tensor
        ),
    )
    simulated_slices = stack[:]
    output_slices = slices_sim[:]
    return volume, output_slices, simulated_slices
