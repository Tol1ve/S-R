from typing import List, Optional, Any, Dict
from multiprocessing import Pool
import importlib.util
import numpy as np
import torch
import logging
from ..image import Stack


def n4_bias_field_correction(
    stacks: List[Stack], n4_params: Dict[str, Any]
) -> List[Stack]:
    if importlib.util.find_spec("SimpleITK") is None:
        raise ImportError(
            "SimpleITK was not found! To use n4 bias field correction, please install SimpleITK."
        )

    pool = Pool(processes=n4_params["n_proc_n4"])
    results = []
    for i, stack in enumerate(stacks):
        logging.info(f"start to correct bias field in stack {i}")
        image_np = stack.slices.cpu().squeeze(1).numpy().astype(np.float32)
        if not torch.all(stack.mask > 0):
            mask_np = stack.mask.cpu().squeeze(1).numpy().astype(np.uint8)
        else:
            mask_np = None
        results.append(
            pool.apply_async(
                n4_bias_field_correction_single,
                args=(
                    image_np,
                    mask_np,
                    float(stack.resolution_x),
                    float(stack.resolution_y),
                    float(stack.gap),
                    n4_params,
                ),
            )
        )
    pool.close()
    pool.join()
    out_stacks = []
    for stack, result in zip(stacks, results):
        out_stack = stack.clone()
        out_stack.slices = torch.tensor(
            result.get(), dtype=stack.slices.dtype, device=stack.slices.device
        ).unsqueeze(1)
        out_stacks.append(out_stack)
    return out_stacks


def n4_bias_field_correction_single(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    res_x: float,
    res_y: float,
    res_z: float,
    n4_params: Dict[str, Any],
) -> np.ndarray:
    import SimpleITK as sitk

    sitk_img_full = sitk.GetImageFromArray(image)
    sitk_img_full.SetSpacing([res_z, res_y, res_x])
    if mask is not None:
        sitk_mask_full = sitk.GetImageFromArray(mask)
        sitk_mask_full.SetSpacing([res_z, res_y, res_x])

    shrinkFactor = n4_params.get("shrink_factor_n4", 1)
    if shrinkFactor > 1:
        sitk_img = sitk.Shrink(
            sitk_img_full, [shrinkFactor] * sitk_img_full.GetDimension()
        )
        sitk_mask = sitk.Shrink(
            sitk_mask_full, [shrinkFactor] * sitk_img_full.GetDimension()
        )
    else:
        sitk_img = sitk_img_full
        sitk_mask = sitk_mask_full

    bias_field_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # see https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1N4BiasFieldCorrectionImageFilter.html for details
    if "fwhm_n4" in n4_params:  # bias_field_fwhm
        bias_field_corrector.SetBiasFieldFullWidthAtHalfMaximum(n4_params["fwhm_n4"])
    if "tol_n4" in n4_params:  # convergence_threshold
        bias_field_corrector.SetConvergenceThreshold(n4_params["tol_n4"])
    if "spline_order_n4" in n4_params:  # spline_order
        bias_field_corrector.SetSplineOrder(n4_params["spline_order_n4"])
    if "noise_n4" in n4_params:  # wiener_filter_noise
        bias_field_corrector.SetWienerFilterNoise(n4_params["noise_n4"])
    if "n_iter_n4" in n4_params and "n_levels_n4" in n4_params:
        # number_of_iteration, number_fitting_levels
        bias_field_corrector.SetMaximumNumberOfIterations(
            [n4_params["n_iter_n4"]] * n4_params["n_levels_n4"]
        )
    if "n_control_points_n4" in n4_params:  # number of control points
        bias_field_corrector.SetNumberOfControlPoints(n4_params["n_control_points_n4"])
    if "n_bins_n4" in n4_params:  # number of histogram bins
        bias_field_corrector.SetNumberOfHistogramBins(n4_params["n_bins_n4"])

    if mask is not None:
        corrected_sitk_img = bias_field_corrector.Execute(sitk_img, sitk_mask)
    else:
        corrected_sitk_img = bias_field_corrector.Execute(sitk_img)

    if shrinkFactor > 1:
        log_bias_field_full = bias_field_corrector.GetLogBiasFieldAsImage(sitk_img_full)
        corrected_sitk_img_full = sitk_img_full / sitk.Exp(log_bias_field_full)
    else:
        corrected_sitk_img_full = corrected_sitk_img

    corrected_image = sitk.GetArrayFromImage(corrected_sitk_img_full)

    return corrected_image
