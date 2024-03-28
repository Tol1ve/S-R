import logging
import time
import math
from typing import List, Tuple, Optional, cast
import numpy as np
import torch
import torch.nn.functional as F
from ..svr.registration import stack_registration
from ..svr.reconstruction import SRR_CG, simulate_slices
from . import SVoRT, SVoRTv2
from ..transform import RigidTransform
from ..utils import get_PSF, ncc_loss, resample
from ..image import Stack, Slice, Volume
from .. import CHECKPOINT_DIR, SVORT_URL_DICT


def compute_score(ncc: torch.Tensor, ncc_weight: torch.Tensor) -> float:
    ncc_weight = ncc_weight.view(ncc.shape)
    return -((ncc * ncc_weight).sum() / ncc_weight.sum()).item()


def get_transform_diff_mean(
    transform_out: RigidTransform, transform_in: RigidTransform, mean_r: int = 3
) -> Tuple[RigidTransform, RigidTransform]:
    transform_diff = transform_out.compose(transform_in.inv())
    length = len(transform_diff)
    assert length > 0, "input is empty!"
    mid = length // 2
    left = max(0, mid - mean_r)
    right = min(length, mid + mean_r)
    transform_diff_mean = transform_diff[left:right].mean(simple_mean=False)
    return transform_diff_mean, transform_diff


def run_model(
    stacks: List[Stack],
    volume: Volume,
    model: torch.nn.Module,
    psf: torch.Tensor,
) -> Tuple[List[Stack], Volume]:
    res_r = volume.resolution_x
    res_s = stacks[0].resolution_x
    device = stacks[0].device

    # run models
    positions_ = [
        torch.arange(len(s), dtype=torch.float32, device=device) - len(s) // 2
        for s in stacks
    ]

    transforms_out: List[RigidTransform] = []
    with torch.no_grad():
        n_run = max(1, len(stacks) - 2)
        for j in range(n_run):
            idxes = [0, 1, j + 2] if j > 0 else list(range(min(3, len(stacks))))
            positions = torch.cat(
                [
                    torch.stack((positions_[i], torch.ones_like(positions_[i]) * k), -1)
                    for k, i in enumerate(idxes)
                ],
                dim=0,
            )
            data = {
                "psf_rec": psf,
                "slice_shape": stacks[0].shape[-2:],  # (128, 128)
                "resolution_slice": res_s,
                "resolution_recon": res_r,
                "volume_shape": volume.shape,  # (125, 169, 145),
                "transforms": RigidTransform.cat(
                    [stacks[idx].transformation for idx in idxes]
                ).matrix(),
                "stacks": torch.cat([stacks[idx].slices for idx in idxes], dim=0),
                "positions": positions,
            }
            t_out, v_out, _ = model(data)
            t_out = t_out[-1]

            if j == 0:
                volume = Volume(v_out[-1][0, 0], None, None, res_r)

            transforms_diff = []
            for ns in range(len(idxes)):
                idx = positions[:, -1] == ns
                if j > 0 and ns != 2:  # anchor stack
                    transform_diff_mean, _ = get_transform_diff_mean(
                        transforms_out[ns], t_out[idx]
                    )
                    transforms_diff.append(transform_diff_mean)
                    continue
                transforms_out.append(t_out[idx])  # new stack
                if j > 0:  # correct stack transformation according to anchor stacks
                    transforms_out[-1] = (
                        RigidTransform.cat(transforms_diff)
                        .mean()
                        .compose(transforms_out[-1])
                    )

    stacks_out = []
    for i in range(len(stacks)):
        stack_out = stacks[i].clone(zero=False, deep=False)
        stack_out.transformation = transforms_out[i]
        stacks_out.append(stack_out)

    volume = Volume(v_out[-1][0, 0], None, None, res_r)

    return stacks_out, volume


def run_model_all_stack(
    stacks: List[Stack],
    volume: Volume,
    model: torch.nn.Module,
    psf: torch.Tensor,
) -> Tuple[List[Stack], Volume]:
    # run models
    res_r = volume.resolution_x
    res_s = stacks[0].resolution_x
    device = stacks[0].device

    positions = torch.cat(
        [
            torch.stack(
                (
                    torch.arange(len(s), dtype=torch.float32, device=device)
                    - len(s) // 2,
                    torch.full((len(s),), i, dtype=torch.float32, device=device),
                ),
                dim=-1,
            )
            for i, s in enumerate(stacks)
        ],
        dim=0,
    )

    with torch.no_grad():
        data = {
            "psf_rec": psf,
            "slice_shape": stacks[0].shape[-2:],  # (128, 128)
            "resolution_slice": res_s,
            "resolution_recon": res_r,
            "volume_shape": volume.shape,  # (125, 169, 145),
            "transforms": RigidTransform.cat(
                [s.transformation for s in stacks]
            ).matrix(),
            "stacks": torch.cat([s.slices for s in stacks], dim=0),
            "positions": positions,
        }
        t_out, v_out, _ = model(data)
        transforms_out = [t_out[-1][positions[:, -1] == i] for i in range(len(stacks))]

    stacks_out = []
    for i in range(len(stacks)):
        stack_out = stacks[i].clone(zero=False, deep=False)
        stack_out.transformation = transforms_out[i]
        stacks_out.append(stack_out)

    volume = Volume(v_out[-1][0, 0], None, None, res_r)

    return stacks_out, volume


def parse_data(
    dataset: List[Stack], svort: bool
) -> Tuple[
    List[Stack],
    List[Stack],
    List[Stack],
    List[Stack],
    List[torch.Tensor],
    Volume,
    torch.Tensor,
]:
    stacks = []  # resampled, cropped, normalized
    stacks_ori = []  # resampled
    transforms = []  # cropped, reset (SVoRT input)
    transforms_full = []  # reset, but with original size
    transforms_ori = []  # original
    crop_idx = []  # z
    dataset_out = []

    res_s = 1.0
    res_r = 0.8

    for data in dataset:
        logging.debug("Preprocessing stack %s for registration.", data.name)
        # resample
        slices = resample(
            data.slices * data.mask,
            (data.resolution_x, data.resolution_y),
            (res_s, res_s),
        )
        slices_ori = slices.clone()
        # crop x,y
        s = slices[torch.argmax((slices > 0).sum((1, 2, 3))), 0]
        i1, i2, j1, j2 = 0, s.shape[0] - 1, 0, s.shape[1] - 1
        while i1 < s.shape[0] and s[i1, :].sum() == 0:
            i1 += 1
        while i2 and s[i2, :].sum() == 0:
            i2 -= 1
        while j1 < s.shape[1] and s[:, j1].sum() == 0:
            j1 += 1
        while j2 and s[:, j2].sum() == 0:
            j2 -= 1
        if ((i2 - i1) > 128 or (j2 - j1) > 128) and svort:
            logging.warning('ROI in input stack "%s" is too large for SVoRT', data.name)
        if (i2 - i1) <= 0:
            logging.warning(
                'Input stack "%s" is all zero after maksing and will be skipped. Please check your data!',
                data.name,
            )
            continue
        pad_margin = 64
        slices = F.pad(
            slices, (pad_margin, pad_margin, pad_margin, pad_margin), "constant", 0
        )
        i = pad_margin + (i1 + i2) // 2
        j = pad_margin + (j1 + j2) // 2
        slices = slices[:, :, i - 64 : i + 64, j - 64 : j + 64]
        # crop z
        idx = (slices > 0).float().sum((1, 2, 3)) > 0
        nz = torch.nonzero(idx)
        nnz = torch.numel(nz)
        if nnz < 7:
            logging.warning(
                'Input stack "%s" only has %d nonzero slices after masking. Consider remove this stack.',
                data.name,
                nnz,
            )
        else:
            logging.debug(
                'Input stack "%s" has %d nonzero slices after masking.', data.name, nnz
            )
        idx[int(nz[0, 0]) : int(nz[-1, 0] + 1)] = True
        crop_idx.append(idx)
        slices = slices[idx]
        # normalize
        stacks.append(slices / torch.quantile(slices[slices > 0], 0.99))
        stacks_ori.append(slices_ori)
        # transformation
        transform = data.transformation
        transforms_ori.append(transform)
        transform_full_ax = transform.axisangle().clone()
        transform_ax = transform_full_ax[idx].clone()

        transform_full_ax[:, :-1] = 0
        transform_full_ax[:, 3] = -((j1 + j2) // 2 - slices_ori.shape[-1] / 2) * res_s
        transform_full_ax[:, 4] = -((i1 + i2) // 2 - slices_ori.shape[-2] / 2) * res_s
        transform_full_ax[:, -1] -= transform_ax[:, -1].mean()

        transform_ax[:, :-1] = 0
        transform_ax[:, -1] -= transform_ax[:, -1].mean()

        transforms.append(RigidTransform(transform_ax))
        transforms_full.append(RigidTransform(transform_full_ax))

        dataset_out.append(data)

    assert len(dataset_out) > 0, "Input data is empty!"

    s_thick = np.mean([data.thickness for data in dataset_out])
    gaps = [data.gap for data in dataset_out]

    stacks_svort_in = [
        Stack(
            stacks[j],
            stacks[j] > 0,
            transforms[j],
            res_s,
            res_s,
            s_thick,
            gaps[j],
        )
        for j in range(len(dataset_out))
    ]

    stacks_resampled = [
        Stack(
            stacks_ori[j],
            stacks_ori[j] > 0,
            transforms_ori[j],
            res_s,
            res_s,
            s_thick,
            gaps[j],
        )
        for j in range(len(dataset_out))
    ]

    stacks_resampled_reset = [s.clone(zero=False, deep=False) for s in stacks_resampled]
    for j in range(len(dataset_out)):
        stacks_resampled_reset[j].transformation = transforms_full[j]

    volume = Volume.zeros((200, 200, 200), res_r, device=dataset_out[0].device)

    psf = get_PSF(
        res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r),
        device=volume.device,
    )

    return (
        dataset_out,
        stacks_svort_in,
        stacks_resampled,
        stacks_resampled_reset,
        crop_idx,
        volume,
        psf,
    )


def correct_svort(
    stacks_out: List[Stack],
    stacks_in: List[Stack],
    volume: Volume,
) -> Tuple[List[Stack], float]:
    # correct transorms
    logging.debug("Correcting SVoRT results with stack transformations ...")
    # compute stack transformation
    stacks = [s.clone(zero=False, deep=False) for s in stacks_out]
    for j in range(len(stacks)):
        transform_diff_mean, _ = get_transform_diff_mean(
            stacks_out[j].transformation, stacks_in[j].transformation
        )
        stacks[j].transformation = transform_diff_mean.compose(
            stacks_in[j].transformation
        )

    ncc_stack, weight = simulated_ncc(stacks, volume)
    ncc_svort, _ = simulated_ncc(stacks_out, volume)
    # negative NCC (the lower the better)
    logging.debug(
        "%d out of %d slices are replaced with the stack transformation",
        torch.count_nonzero(ncc_svort > ncc_stack).item(),
        ncc_svort.numel(),
    )

    idx = 0
    for j in range(len(stacks)):
        ns = len(stacks[j])
        t_out = torch.where(
            (ncc_svort[idx : idx + ns] <= ncc_stack[idx : idx + ns]).reshape(-1, 1, 1),
            stacks_out[j].transformation.matrix(),
            stacks[j].transformation.matrix(),
        )
        idx += ns
        stacks[j].transformation = RigidTransform(t_out)

    ncc_min = torch.min(ncc_svort, ncc_stack)
    score_svort = compute_score(ncc_min, weight)

    return stacks, score_svort


def get_transforms_full(
    stacks_out: List[Stack],
    stacks_in: List[Stack],
    stacks_full: List[Stack],
    crop_idx: List[torch.Tensor],
) -> Tuple[List[Stack], List[Stack]]:
    stacks_svort_full = [s.clone(zero=False, deep=False) for s in stacks_full]
    stacks_stack_full = [s.clone(zero=False, deep=False) for s in stacks_full]

    for j in range(len(stacks_in)):
        transform_diff_mean, transform_diff = get_transform_diff_mean(
            stacks_out[j].transformation, stacks_in[j].transformation
        )
        transform_stack_full = transform_diff_mean.compose(
            stacks_full[j].transformation
        )
        transform_svort_full = transform_stack_full.matrix().clone()
        transform_svort_full[crop_idx[j]] = transform_diff.compose(
            stacks_full[j].transformation[crop_idx[j]]
        ).matrix()
        stacks_svort_full[j].transformation = RigidTransform(transform_svort_full)
        stacks_stack_full[j].transformation = transform_stack_full

    return stacks_svort_full, stacks_stack_full


def reconstruct_from_stacks(
    stacks: List[Stack],
    volume: Volume,
    n_stack_recon: Optional[int],
    psf: Optional[torch.Tensor],
) -> Volume:
    stacks = Stack.pad_stacks(stacks)
    if n_stack_recon is None:
        n_stack_recon = len(stacks)
    else:
        n_stack_recon = min(len(stacks), n_stack_recon)
    stack = Stack.cat(stacks[:n_stack_recon])
    srr = SRR_CG(n_iter=1, average_init=True, use_mask=True)
    volume = srr(stack, volume, psf=psf)
    return volume


def simulated_ncc(
    stacks: List[Stack],
    volume: Volume,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ncc = []
    ncc_weight = []
    for j in range(len(stacks)):
        stack = stacks[j]
        simulated_stack = cast(
            Stack, simulate_slices(stack, volume, return_weight=False, use_mask=True)
        )
        ncc_weight.append(stack.mask.sum((1, 2, 3)))
        ncc.append(
            ncc_loss(
                simulated_stack.slices,
                stack.slices,
                stack.mask,
                win=None,
                reduction="none",
            )
        )
    ncc_all = torch.cat(ncc)
    ncc_weight_all = torch.cat(ncc_weight).view(ncc_all.shape)
    return ncc_all, ncc_weight_all


def run_svort(
    dataset: List[Stack],
    model: Optional[torch.nn.Module],
    svort: bool,
    vvr: bool,
    force_vvr: bool,
    force_scanner: bool,
) -> List[Slice]:
    if svort or vvr:
        (dataset, stacks_in, ss_ori, stacks_full, crop_idx, volume, psf) = parse_data(
            dataset, svort
        )

    if svort:
        # run SVoRT model
        time_start = time.time()
        if isinstance(model, SVoRT):
            stacks_out, volume = run_model(stacks_in, volume, model, psf)
        elif isinstance(model, SVoRTv2):
            stacks_out, volume = run_model_all_stack(stacks_in, volume, model, psf)
        else:
            raise TypeError("unkown SVoRT model!")
        logging.debug("time for running SVoRT: %f s" % (time.time() - time_start))
        # correct the prediction of SVoRT
        time_start = time.time()
        stacks_out, _ = correct_svort(stacks_out, stacks_in, volume)
        logging.debug(
            "time for stack transformation correction: %f s"
            % (time.time() - time_start)
        )
        # propagate SVoRT prediction to the entire stack
        ss_svort_full, ss_stack_full = get_transforms_full(
            stacks_out, stacks_in, stacks_full, crop_idx
        )
        # estimate NCC score for SVoRT prediction
        if vvr:
            time_start = time.time()
            volume = reconstruct_from_stacks(ss_svort_full, volume, 3, psf)
            score_svort = compute_score(
                *simulated_ncc(
                    [s.get_substack(i) for s, i in zip(ss_svort_full, crop_idx)], volume
                )
            )
            logging.debug(
                "time for evaluating SVoRT registration %f s"
                % (time.time() - time_start)
            )
        else:
            score_svort = float("inf")
    else:
        score_svort = float("-inf")

    if vvr:
        # stack-to-stack registration
        time_start = time.time()
        __ss = stack_registration([ss_stack_full, ss_ori] if svort else [ss_ori], svort)
        logging.debug("time for stack registration: %f s" % (time.time() - time_start))
        # estimate NCC score for stack-to-stack registration
        if svort:
            time_start = time.time()
            volume = reconstruct_from_stacks(__ss, volume, 3, psf)
            score_vvr = compute_score(
                *simulated_ncc(
                    [s.get_substack(i) for s, i in zip(__ss, crop_idx)], volume
                )
            )
            logging.debug(
                "time for evaluating stack registration %f s"
                % (time.time() - time_start)
            )
        else:
            score_vvr = float("inf")
    else:
        score_vvr = float("-inf")

    if svort or vvr:
        # use the result with the best NCC score
        if math.isfinite(score_svort):
            logging.info("similarity score for SVoRT = %f", score_svort)
        if math.isfinite(score_vvr):
            logging.info("similarity score for stack registration = %f", score_vvr)
        if score_svort < score_vvr or force_vvr:
            logging.info("use stack transformation")
            transforms_out = [s.transformation for s in __ss]
        else:
            logging.info("use slice transformation")
            transforms_out = [s.transformation for s in ss_svort_full]

        if force_scanner:
            # map the results back to scanner coordiante
            transform_scanner = (
                ss_ori[0].transformation.mean().compose(transforms_out[0].mean().inv())
            )
            for j in range(len(dataset)):
                transforms_out[j] = transform_scanner.compose(transforms_out[j])

        for j in range(len(dataset)):
            dataset[j].transformation = transforms_out[j]

    slices = []
    for stack in dataset:
        idx_nonempty = stack.mask.flatten(1).any(1)
        stack.slices /= torch.quantile(stack.slices[stack.mask], 0.99)  # normalize
        slices.extend(stack[idx_nonempty])

    return slices


def svort_predict(
    dataset: List[Stack],
    device,
    svort_version: str,
    svort: bool,
    vvr: bool,
    force_vvr: bool,
    force_scanner: bool,
) -> List[Slice]:
    model: Optional[torch.nn.Module] = None
    if svort:
        if svort_version not in SVORT_URL_DICT:
            raise ValueError("unknown SVoRT version!")
        svort_url = SVORT_URL_DICT[svort_version]
        cp = torch.hub.load_state_dict_from_url(
            url=svort_url,
            model_dir=CHECKPOINT_DIR,
            map_location=device,
            file_name="SVoRT_%s.pt" % svort_version,
        )
        if svort_version == "v1" or "v1." in svort_version:
            model = SVoRT(n_iter=3)
        elif svort_version == "v2" or "v2." in svort_version:
            model = SVoRTv2(n_iter=4)
        else:
            raise ValueError("unknown SVoRT version!")
        logging.debug("Loading SVoRT model")
        model.load_state_dict(cp["model"])
        model.to(device)
        model.eval()
    return run_svort(dataset, model, svort, vvr, force_vvr, force_scanner)
