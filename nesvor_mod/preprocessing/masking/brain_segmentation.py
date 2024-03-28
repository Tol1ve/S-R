"""
This is a wrapper of MONAIfbs. 
See https://github.com/gift-surg/MONAIfbs for more details.
"""

import torch
from typing import List
import os
import tarfile
import importlib
import torch.nn.functional as F
from math import ceil
import numpy as np
import logging
from skimage.morphology import dilation, disk
from skimage.measure import label
from ...image import Stack
from ... import CHECKPOINT_DIR, MONAIFBS_URL


RESOLUTION = 0.8
W_min, W_factor = 7, 64
H_min, H_factor = 4, 128


def get_monaifbs_checkpoint() -> str:
    model_dir = CHECKPOINT_DIR
    model_name = "checkpoint_dynUnet_DiceXent.pt"
    if not os.path.exists(os.path.join(model_dir, model_name)):
        logging.info(
            "monaifbs checkpoint not found. trying to download the checkpoint."
        )
        url = MONAIFBS_URL
        zip_name = "monaifbs_models.tar.gz"
        torch.hub.download_url_to_file(url, os.path.join(model_dir, zip_name))
        with tarfile.open(os.path.join(model_dir, zip_name)) as file:
            # extracting file
            file.extractall(model_dir)
        os.rename(
            os.path.join(model_dir, "models", model_name),
            os.path.join(model_dir, model_name),
        )
        os.remove(os.path.join(model_dir, zip_name))
        os.rmdir(os.path.join(model_dir, "models"))
    return os.path.join(model_dir, model_name)


def build_monaifbs_net(device):
    logging.info("building monaifbs network")

    try:
        module = importlib.import_module("monai.networks.nets")
        DynUNet = getattr(module, "DynUNet")
    except:
        raise ImportError("moani needs to be install in order to use monaifbs")

    # inference params
    nr_out_channels = 2
    spacing = [RESOLUTION, RESOLUTION, -1.0]
    patch_size = [W_min * W_factor, H_min * H_factor] + [1]

    # automatically extracts the strides and kernels based on nnU-Net empirical rules
    spacings = spacing[:2]
    sizes = patch_size[:2]
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])

    # initialise the network
    net = DynUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=nr_out_channels,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=2,
        res_block=False,
    )

    net.load_state_dict(torch.load(get_monaifbs_checkpoint(), device)["net"])
    net = net.to(device)
    net.eval()

    return net


def batch_infer(inputs: torch.Tensor, model, batch_size: int) -> torch.Tensor:
    outputs = torch.empty(
        (inputs.shape[0], 2, inputs.shape[2], inputs.shape[3]),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    i = 0
    while i < inputs.shape[0]:
        outputs[i : i + batch_size] = model(inputs[i : i + batch_size])
        i += batch_size
    return outputs


def _segment(
    img: torch.Tensor,
    res_x: float,
    res_y: float,
    net,
    batch_size: int,
    augmentation: bool,
    radius: int,
    threshold_small: float,
) -> torch.Tensor:
    # resample (bilinear)
    shape = img.shape[-2:]
    shape_resampled = (
        int(shape[-2] * res_y / RESOLUTION),
        int(shape[-1] * res_x / RESOLUTION),
    )
    img = torch.nn.functional.interpolate(
        img, size=shape_resampled, mode="bilinear", align_corners=True
    )
    # normalize
    img = (img - img.mean()) / (img.std() + 1e-8)
    seg_all: torch.Tensor
    for t in [True, False] if augmentation else [False]:
        for flip_dim in [None, (2,), (3,), (2, 3)] if augmentation else [None]:
            # augmentation
            img_aug = img.permute(0, 1, 3, 2) if t else img
            img_aug = torch.flip(img_aug, flip_dim) if flip_dim else img_aug
            shape_aug = img_aug.shape[-2:]
            # padding
            shape_padded = (
                max(int(ceil(shape_aug[-2] / W_factor)), W_min) * W_factor,
                max(int(ceil(shape_aug[-2] / H_factor)), H_min) * H_factor,
            )
            d1 = shape_padded[-1] - shape_aug[-1]
            d2 = shape_padded[-2] - shape_aug[-2]
            h1 = d1 // 2
            h2 = d2 // 2
            if d1 or d2:
                img_aug = F.pad(img_aug, (h1, d1 - h1, h2, d2 - h2))
            # inference
            with torch.no_grad():
                seg = batch_infer(img_aug, net, batch_size)
            # crop
            if d1 or d2:
                seg = seg[:, :, h2 : h2 + shape_aug[-2], h1 : h1 + shape_aug[-1]]
            # flip
            seg = torch.flip(seg, flip_dim) if flip_dim else seg
            seg = seg.permute(0, 1, 3, 2) if t else seg
            # sum
            if "seg_all" not in locals():
                seg_all = seg
            else:
                seg_all = seg_all + seg
    # resample
    seg_all = torch.nn.functional.interpolate(
        seg_all, size=shape, mode="bilinear", align_corners=True
    )
    seg_all_np = torch.argmax(seg_all, 1, keepdim=True).bool().cpu().numpy()
    for i in range(seg_all_np.shape[0]):
        seg_np = seg_all_np[i, 0]
        if seg_np.sum() > 0:
            if radius:
                seg_np = dilation(
                    seg_np, footprint=disk(ceil(2 * radius / (res_x + res_y)))
                )
            seg_np = label(seg_np)
            seg_all_np[i, 0] = seg_np == np.argmax(np.bincount(seg_np.flat)[1:]) + 1
    seg_all = torch.tensor(seg_all_np, dtype=torch.bool, device=img.device)
    nnz = seg_all.count_nonzero((1, 2, 3))
    seg_all[nnz < threshold_small * nnz.max()] = 0
    return seg_all


def brain_segmentation(
    stacks: List[Stack],
    device,
    batch_size: int,
    augmentation: bool,
    radius: int,
    threshold_small: float,
) -> List[Stack]:
    net = build_monaifbs_net(device)
    for i, stack in enumerate(stacks):
        logging.info(f"segmenting stack {i}")
        seg_mask = _segment(
            stack.slices,
            stack.resolution_x,
            stack.resolution_y,
            net,
            batch_size,
            augmentation,
            radius,
            threshold_small,
        )
        stack.mask = torch.logical_and(stack.mask, seg_mask)
        if not stack.mask.any():
            logging.warning(
                "One of the input stack is all zero after brain segmentation. Please check your data!"
            )
    return stacks
