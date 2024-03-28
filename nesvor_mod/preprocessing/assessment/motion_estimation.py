import torch
from typing import List
from ...image import Stack
from ...utils import ncc_loss


def matrix_rank(
    stacks: List[Stack], threshold: float = 0.1, rank_only: bool = False
) -> List[float]:
    scores: List[float] = []
    for stack in stacks:
        masked_slices = stack.slices * stack.mask.float()
        score = _matrix_rank(masked_slices.flatten(1, -1), threshold, rank_only)
        scores.append(score)
    return scores


def _matrix_rank(mat: torch.Tensor, threshold: float, rank_only: bool) -> float:
    s = torch.linalg.svdvals(mat)
    s = s / s[0]
    R = torch.count_nonzero(s > 1e-6)
    norm2 = torch.cumsum(s.pow(2), 0)
    norm2_all = norm2[-1]
    e = threshold
    for r in range(len(s) - 1):
        if norm2[r] > (1 - threshold * threshold) * norm2_all:
            e = torch.sqrt((norm2_all - norm2[r]) / norm2_all).item()
            break
    return float((r + 1) / R if rank_only else e * (r + 1) / R / threshold)


def ncc(stacks: List[Stack]) -> List[float]:
    scores: List[float] = []
    for stack in stacks:
        masked_slices = stack.slices * stack.mask.float()
        score = _ncc(masked_slices)
        scores.append(score)
    return scores


def _ncc(slices: torch.Tensor) -> float:
    slices1 = slices[:-1]
    slices2 = slices[1:]
    mask = ((slices1 + slices2) > 0).float()
    ncc = ncc_loss(slices1, slices2, mask, win=None, reduction="none")
    ncc_weight = mask.sum((1, 2, 3))
    ncc_weight = ncc_weight.view(ncc.shape)
    return float(-((ncc * ncc_weight).sum() / ncc_weight.sum()).item())
