from typing import List
import torch
from ...image import Stack, Volume


def stack_intersect(stacks: List[Stack], box: bool) -> Volume:
    return volume_intersect([stack.get_mask_volume() for stack in stacks], box)


def volume_intersect(volumes: List[Volume], box: bool) -> Volume:
    volume = volumes[0].clone()
    for i in range(1, len(volumes)):
        assign_mask = volume.mask.clone()
        volume.mask[assign_mask] = volumes[i].sample_points(volume.xyz_masked) > 0
    mask = volume.mask
    if not mask.any():
        raise ValueError("The intersection of inputs is empty!")
    if box:
        nz = torch.nonzero(mask.sum(dim=[1, 2]))
        i0 = int(nz[0, 0])
        i1 = int(nz[-1, 0] + 1)
        nz = torch.nonzero(mask.sum(dim=[0, 2]))
        j0 = int(nz[0, 0])
        j1 = int(nz[-1, 0] + 1)
        nz = torch.nonzero(mask.sum(dim=[0, 1]))
        k0 = int(nz[0, 0])
        k1 = int(nz[-1, 0] + 1)
        mask[i0:i1, j0:j1, k0:k1] = True
    volume.image = mask.float()
    return volume
