import os
from typing import List, Optional
import numpy as np
import torch
import logging
import multiprocessing
import importlib.util
from .architectures import model_architecture, INPUT_SHAPE
from .....image import Stack
from ..... import CHECKPOINT_DIR, IQA3D_URL


def get_iqa3d_checkpoint() -> str:
    model_dir = CHECKPOINT_DIR
    model_name = "iqa3d.hdf5"
    if not os.path.exists(os.path.join(model_dir, model_name)):
        logging.info(
            "3D IQA CNN checkpoint not found. trying to download the checkpoint."
        )
        url = IQA3D_URL
        torch.hub.download_url_to_file(url, os.path.join(model_dir, model_name))
    return os.path.join(model_dir, model_name)


def iqa3d(stacks: List[Stack], batch_size=8, augmentation=True) -> List[float]:
    if importlib.util.find_spec("tensorflow") is None:
        raise ImportError(
            "Tensorflow was not found! To use 3D IQA, please install tensorflow."
        )

    # torch -> numpy
    data: List[np.ndarray] = []
    for stack in stacks:
        d = stack.slices * stack.mask.float()
        d = d.squeeze(1).permute((2, 1, 0))
        idx = torch.nonzero(d > 0, as_tuple=True)
        x1, x2 = int(idx[0].min().item()), int(idx[0].max().item())
        y1, y2 = int(idx[1].min().item()), int(idx[1].max().item())
        z1, z2 = int(idx[2].min().item()), int(idx[2].max().item())
        d = d[x1 : x2 + 1, y1 : y2 + 1, z1 : z2 + 1]
        d = d[: INPUT_SHAPE[0], : INPUT_SHAPE[1], : INPUT_SHAPE[2]]
        d[d < 0] = 0
        d[d >= 10000] = 10000
        d = d / d.max()
        d = d[..., None]
        data.append(d.cpu().numpy())

    # augmentation
    data_all = []
    if augmentation:
        flip_dims = [None, (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    else:
        flip_dims = [None]
    for _data in data:
        data_aug = []
        for flip_dim in flip_dims:
            d_aug = np.flip(_data, flip_dim) if flip_dim else _data
            pad = np.zeros([*INPUT_SHAPE, 1], dtype=np.float32)
            pad[: d_aug.shape[0], : d_aug.shape[1], : d_aug.shape[2]] = d_aug
            data_aug.append(pad)
        data_all.append(data_aug)
    stacked_data = np.array(data_all, dtype=np.float32)
    weight_path = get_iqa3d_checkpoint()
    # run tf in another process, make sure tf release the GPU after use
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        scores = pool.apply(inference, (stacked_data, batch_size, weight_path))

    return scores


def inference(
    data: np.ndarray, batch_size: Optional[int], weight_path: str
) -> List[float]:
    # load model
    model = model_architecture()
    model.compile()
    model.load_weights(weight_path)
    # predict
    L = data.shape[0]
    C = data.shape[1]
    data = np.array(data, dtype=np.float32).reshape((L * C, *INPUT_SHAPE, 1))
    predict_all = model.predict(data, batch_size=batch_size).reshape((L, C))
    predict_all = np.flip(np.sort(predict_all, -1), -1)
    predict_all = predict_all[:, :4].mean(axis=-1)  # 2
    return [float(score) for score in predict_all]
