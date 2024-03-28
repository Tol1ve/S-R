import os
from typing import List, Optional
import numpy as np
import torch
import logging
from .architectures import resnet34
from .....image import Stack
from ..... import CHECKPOINT_DIR, IQA2D_URL


def get_iqa2d_checkpoint() -> str:
    model_dir = CHECKPOINT_DIR
    model_name = "iqa2d.pt"
    if not os.path.exists(os.path.join(model_dir, model_name)):
        logging.info(
            "2D IQA CNN checkpoint not found. trying to download the checkpoint."
        )
        url = IQA2D_URL
        torch.hub.download_url_to_file(url, os.path.join(model_dir, model_name))
        checkpoint = torch.load(os.path.join(model_dir, model_name))
        state_dict = checkpoint["ema_state_dict"]
        new_state_dict = {}
        for k in state_dict.keys():
            new_state_dict[k.replace("module.", "", 1)] = state_dict[k]
        torch.save(new_state_dict, os.path.join(model_dir, model_name))
    return os.path.join(model_dir, model_name)


def iqa2d(
    stacks: List[Stack],
    device,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    batch_size: int = 64,
    augmentation: bool = True,
) -> List[float]:
    # load model
    model = resnet34(num_classes=3).to(device)
    checkpoint = torch.load(get_iqa2d_checkpoint(), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    # estimate mean and std
    if mean is None or std is None:
        v_mean = 0.0
        v2_mean = 0.0
        n = 0
        for stack in stacks:
            v_new = stack.slices[stack.slices > 0]
            m = v_new.numel()
            v_mean += m * (v_new.mean().item() - v_mean) / (n + m)
            v2_mean += m * (v_new.pow(2).mean().item() - v2_mean) / (n + m)
            n += m
        if mean is None:
            mean = v_mean
        if std is None:
            std = np.sqrt(v2_mean - v_mean * v_mean)
    # inference
    scores = []
    for stack in stacks:
        score = _iqa2d(stack, model, mean, std, batch_size, augmentation)
        scores.append(score)
    return scores


def batch_infer(
    inputs: torch.Tensor, model: torch.nn.Module, batch_size: int
) -> torch.Tensor:
    outputs = torch.empty(
        (inputs.shape[0], 3),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    i = 0
    while i < inputs.shape[0]:
        outputs[i : i + batch_size] = model(inputs[i : i + batch_size])[0]
        i += batch_size
    return outputs


def _iqa2d(
    stack: Stack,
    model: torch.nn.Module,
    mean: float,
    std: float,
    batch_size: int,
    augmentation: bool,
) -> float:
    # z_mask = stack.slices == 0
    img = (stack.slices - mean) / std
    # img[z_mask] = -mean / std
    predict_all: torch.Tensor
    n_predict = 0
    for t in [True, False] if augmentation else [False]:
        for flip_dim in [None, (2,), (3,), (2, 3)] if augmentation else [None]:
            # augmentation
            img_aug = img.permute(0, 1, 3, 2) if t else img
            img_aug = torch.flip(img_aug, flip_dim) if flip_dim else img_aug
            # inference
            with torch.no_grad():
                predict = batch_infer(img_aug, model, batch_size)
                predict = torch.softmax(predict, dim=1)
                n_predict += 1
            # sum
            if "predict_all" not in locals():
                predict_all = predict
            else:
                predict_all = predict_all + predict
    # post processing
    predict_all /= n_predict
    p_mask = (stack.slices > 0).float().sum((1, 2, 3))
    p_mask /= p_mask.max()
    p_roi = torch.minimum(1 - predict_all[:, 0], p_mask)
    p_good = predict_all[:, 1]  # / (predict_all[:, 1] + predict_all[:, 2] + 1e-8)
    score = (p_good * p_roi).sum() / (p_roi.sum() + 1e-8)
    return float(score.item())
