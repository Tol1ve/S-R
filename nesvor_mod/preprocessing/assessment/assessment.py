from typing import Tuple, List, Dict
import numpy as np
from ...image import Stack
from . import motion_estimation
from . import iqa
from ...utils import DeviceType


def compute_metric(
    stacks: List[Stack],
    metric: str,
    batch_size: int,
    augmentation: bool,
    device: DeviceType,
) -> Tuple[List[float], bool]:
    descending = True
    if metric == "ncc":
        scores = motion_estimation.ncc(stacks)
    elif metric == "matrix-rank":
        scores = motion_estimation.matrix_rank(stacks)
        descending = False
    elif metric == "volume":
        scores = [
            int(
                stack.mask.float().sum().item()
                * stack.resolution_x
                * stack.resolution_y
                * stack.gap
            )
            for stack in stacks
        ]
    elif metric == "iqa2d":
        scores = iqa.iqa2d(
            stacks, device, batch_size=batch_size, augmentation=augmentation
        )
    elif metric == "iqa3d":
        scores = iqa.iqa3d(stacks, batch_size=batch_size, augmentation=augmentation)
    else:
        raise ValueError("unkown metric for stack assessment")

    return scores, descending


def sort_and_filter(
    stacks: List[Stack],
    scores: List[float],
    descending: bool,
    filter_method: str,
    cutoff: float,
) -> Tuple[List[Stack], List[int], List[bool]]:
    n_total = len(scores)
    n_keep = n_total
    if filter_method == "top":
        n_keep = min(n_keep, int(cutoff))
    elif filter_method == "bottom":
        n_keep = max(0, n_total - int(cutoff))
    elif filter_method == "percentage":
        n_keep = n_total - int(n_total * min(max(0, cutoff), 1))
    elif filter_method == "threshold":
        if descending:
            n_keep = sum(score >= cutoff for score in scores)
        else:
            n_keep = sum(score <= cutoff for score in scores)
    elif filter_method == "none":
        pass
    else:
        raise ValueError("unknown filter method")

    sorter = np.argsort(-np.array(scores) if descending else scores)
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    ranks = [int(rank) for rank in inv]
    excluded = [rank >= n_keep for rank in ranks]

    output_stacks = [stacks[i] for i in sorter[:n_keep]]

    return output_stacks, ranks, excluded


def assess(
    stacks: List[Stack],
    metric: str,
    filter_method: str,
    cutoff: float,
    batch_size: int,
    augmentation: bool,
    device: DeviceType,
) -> Tuple[List[Stack], List[Dict]]:
    if metric == "none":
        return stacks, []

    scores, descending = compute_metric(
        stacks, metric, batch_size, augmentation, device
    )
    filtered_stacks, ranks, excludeds = sort_and_filter(
        stacks, scores, descending, filter_method, cutoff
    )

    results = []
    for i, (score, rank, excluded, stack) in enumerate(
        zip(scores, ranks, excludeds, stacks)
    ):
        results.append(
            dict(
                input_id=i,
                name=stack.name,
                score=score,
                rank=rank,
                excluded=excluded,
                descending=descending,
            )
        )

    return filtered_stacks, results
