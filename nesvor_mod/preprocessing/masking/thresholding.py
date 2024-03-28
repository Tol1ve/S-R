from skimage.filters import threshold_multiotsu
from skimage.morphology import dilation, ball
import torch
from typing import List
from ...image import Stack, Image


def otsu_thresholding(stacks: List[Stack], nbins: int = 256) -> List[Stack]:
    for stack in stacks:
        thresholds = threshold_multiotsu(
            image=stack.slices.cpu().numpy(), classes=2, nbins=nbins
        )
        assert len(thresholds) == 1
        mask = stack.slices > thresholds[0]
        mask = torch.tensor(
            dilation(mask.squeeze().cpu().numpy(), footprint=ball(3)),
            dtype=mask.dtype,
            device=mask.device,
        ).view(mask.shape)
        stack.mask = torch.logical_and(stack.mask, mask)
    return stacks


def thresholding(inputs: List, threshold: float) -> List:
    for inp in inputs:
        if isinstance(inp, Stack):
            mask = inp.slices > threshold
        elif isinstance(inp, Image):
            mask = inp.image > threshold
        else:
            raise ValueError(f"unknown input type: {type(inp)}")
        inp.mask = torch.logical_and(inp.mask, mask)
    return inputs
