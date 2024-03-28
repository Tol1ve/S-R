from typing import Callable, Dict, Optional, Tuple, Union, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..transform import mat_update_resolution
from ..slice_acquisition import slice_acquisition, slice_acquisition_adjoint
from ..image import Volume, Stack
from ..utils import get_PSF


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.dot(x.flatten(), y.flatten())


def cg(
    A: Callable, b: torch.Tensor, x0: torch.Tensor, n_iter: int, tol: float = 0.0
) -> torch.Tensor:
    if x0 is None:
        x = 0
        r = b
    else:
        x = x0
        r = b - A(x)
    p = r
    dot_r_r = dot(r, r)
    i = 0
    while True:
        Ap = A(p)
        alpha = dot_r_r / dot(p, Ap)
        x = x + alpha * p  # alpha ~ 0.1 - 1
        i += 1
        if i == n_iter:
            return x
        r = r - alpha * Ap
        dot_r_r_new = dot(r, r)
        if dot_r_r_new <= tol:
            return x
        p = r + (dot_r_r_new / dot_r_r) * p
        dot_r_r = dot_r_r_new


def psf_reconstruction(
    slices: Stack,
    volume: Volume,
    update_mask: bool = False,
    use_mask: bool = False,
    psf: Optional[torch.Tensor] = None,
) -> Volume:
    slices_transform_mat, res_s, res_r, s_thick, psf = _parse_stack_volume(
        slices, volume, psf
    )

    if update_mask:
        m = (
            slice_acquisition_adjoint(
                slices_transform_mat,
                psf,
                slices.mask.float(),
                None,
                None,
                volume.shape[-3:],
                res_s / res_r,
                False,
                equalize=True,
            )
            > 0.3
        )[0, 0]
    else:
        m = volume.mask

    v = slice_acquisition_adjoint(
        slices_transform_mat,
        psf,
        slices.slices,
        slices.mask if use_mask else None,
        volume.mask[None, None] if use_mask else None,
        volume.shape[-3:],
        res_s / res_r,
        False,
        equalize=True,
    )[0, 0]

    return cast(Volume, Volume.like(volume, v, m, deep=False))


class SRR_CG(nn.Module):
    def __init__(
        self,
        n_iter: int = 10,
        tol: float = 0.0,
        mu: float = 0.0,
        average_init: bool = False,
        output_relu: bool = True,
        use_mask: bool = False,
    ) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.tol = tol
        self.mu = mu
        self.output_relu = output_relu
        self.use_mask = use_mask
        self.average_init = average_init

    def forward(
        self,
        stack: Stack,
        volume: Volume,
        p: Union[Stack, torch.Tensor, None] = None,
        z: Union[Stack, torch.Tensor, None] = None,
        psf: Optional[torch.Tensor] = None,
    ) -> Volume:
        transforms, res_s, res_r, s_thick, psf = _parse_stack_volume(stack, volume, psf)

        params = {
            "psf": psf,
            "slice_shape": stack.shape[-2:],
            "res_s": res_s,
            "res_r": res_r,
            "volume_shape": volume.shape[-3:],
        }

        slices_mask = stack.mask if self.use_mask else None
        vol_mask = volume.mask[None, None] if self.use_mask else None

        if isinstance(p, Stack):
            p = p.slices
        if isinstance(z, Stack):
            z = z.slices

        # A = lambda x: self.A(transforms, x, vol_mask, slices_mask, params)
        At = lambda x: self.At(transforms, x, slices_mask, vol_mask, params)
        AtA = lambda x: self.AtA(
            transforms, x, vol_mask, slices_mask, p, params, self.mu, z
        )
        y = stack.slices
        if self.average_init:
            x = slice_acquisition_adjoint(
                transforms,
                psf,
                y,
                slices_mask,
                vol_mask,
                volume.shape[-3:],
                res_s / res_r,
                False,
                equalize=True,
            )
        else:
            x = volume.image[None, None]

        b = At(y * p if p is not None else y)
        if self.mu and z is not None:
            b = b + self.mu * z
        x = cg(AtA, b, x, self.n_iter, self.tol)

        if self.output_relu:
            x = F.relu(x, True)

        return cast(Volume, Volume.like(volume, image=x[0, 0], deep=False))

    def A(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        vol_mask: Optional[torch.Tensor],
        slices_mask: Optional[torch.Tensor],
        params: Dict,
    ) -> torch.Tensor:
        return slice_acquisition(
            transforms,
            x,
            vol_mask,
            slices_mask,
            params["psf"],
            params["slice_shape"],
            params["res_s"] / params["res_r"],
            False,
            False,
        )

    def At(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        slices_mask: Optional[torch.Tensor],
        vol_mask: Optional[torch.Tensor],
        params: Dict,
    ) -> torch.Tensor:
        return slice_acquisition_adjoint(
            transforms,
            params["psf"],
            x,
            slices_mask,
            vol_mask,
            params["volume_shape"],
            params["res_s"] / params["res_r"],
            False,
            False,
        )

    def AtA(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        vol_mask: Optional[torch.Tensor],
        slices_mask: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
        params: Dict,
        mu: float,
        z: Optional[torch.Tensor],
    ) -> torch.Tensor:
        slices = self.A(transforms, x, vol_mask, slices_mask, params)
        if p is not None:
            slices = slices * p
        vol = self.At(transforms, slices, slices_mask, vol_mask, params)
        if mu and z is not None:
            vol = vol + mu * x
        return vol


def srr_update(
    e: Stack,
    v: Volume,
    p: Optional[Union[Stack, torch.Tensor]],
    alpha: float,
    beta: float,
    delta: float,
    use_mask: bool = False,
    psf: Optional[torch.Tensor] = None,
) -> Volume:
    # beta = beta * delta * delta
    err = e.slices
    volume = v.image[None, None]
    if p is not None:
        if isinstance(p, Stack):
            p = p.slices
        err = p * err

    transforms, res_s, res_r, s_thick, psf = _parse_stack_volume(e, v, psf)

    g = slice_acquisition_adjoint(
        transforms,
        psf,
        err,
        e.mask if use_mask else None,
        v.mask[None, None] if use_mask else None,
        v.shape,
        res_s / res_r,
        False,
        False,
    )

    if p is not None:
        cmap = slice_acquisition_adjoint(
            transforms,
            psf,
            p,
            e.mask if use_mask else None,
            v.mask[None, None] if use_mask else None,
            v.shape,
            res_s / res_r,
            False,
            False,
        )
        cmap_mask = cmap > 0
        g[cmap_mask] /= cmap[cmap_mask]
    reconstructed = F.relu(volume + alpha * g, True)

    g = torch.zeros_like(volume)
    D, H, W = volume.shape[-3:]
    v0 = volume[:, :, 1 : D - 1, 1 : H - 1, 1 : W - 1]
    r0 = reconstructed[:, :, 1 : D - 1, 1 : H - 1, 1 : W - 1]
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                v1 = volume[
                    :, :, 1 + dz : D - 1 + dz, 1 + dy : H - 1 + dy, 1 + dx : W - 1 + dx
                ]
                r1 = reconstructed[
                    :, :, 1 + dz : D - 1 + dz, 1 + dy : H - 1 + dy, 1 + dx : W - 1 + dx
                ]
                d2 = dx * dx + dy * dy + dz * dz
                dv2 = (v1 - v0) ** 2
                b = 1 / (d2 * torch.sqrt(1 + 1 / (d2 * delta * delta) * dv2))
                g[:, :, 1 : D - 1, 1 : H - 1, 1 : W - 1] += b * (r1 - r0)
    if p is not None:
        g *= cmap_mask
    reconstructed.add_(g, alpha=alpha * beta)
    reconstructed = F.relu(reconstructed, True)
    return cast(Volume, Volume.like(v, reconstructed[0, 0], deep=False))


def simulate_slices(
    slices: Stack,
    volume: Volume,
    return_weight: bool = False,
    use_mask: bool = False,
    psf: Optional[torch.Tensor] = None,
) -> Union[Tuple[Stack, Stack], Stack]:
    slices_transform_mat, res_s, res_r, s_thick, psf = _parse_stack_volume(
        slices, volume, psf
    )

    outputs = slice_acquisition(
        slices_transform_mat,
        volume.image[None, None],
        volume.mask[None, None] if use_mask else None,
        slices.mask if use_mask else None,
        psf,
        slices.shape[-2:],
        res_s / res_r,
        return_weight,
        False,
    )

    if return_weight:
        slices_sim, weight = outputs
        return (
            Stack.like(slices, slices=slices_sim, deep=False),
            Stack.like(slices, slices=weight, deep=False),
        )
    else:
        return Stack.like(slices, slices=outputs, deep=False)


def _parse_stack_volume(
    stack: Stack, volume: Volume, psf: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, float, float, float, torch.Tensor]:
    eps = 1e-3
    assert (
        abs(volume.resolution_x - volume.resolution_y) < eps
        and abs(volume.resolution_x - volume.resolution_z) < eps
    ), "input volume should be isotropic!"
    assert (
        abs(stack.resolution_x - stack.resolution_y) < eps
    ), "input slices should be isotropic!"

    res_s = float(stack.resolution_x)
    res_r = float(volume.resolution_x)
    s_thick = float(stack.thickness)

    slices_transform = stack.transformation
    volume_transform = volume.transformation
    slices_transform = volume_transform.inv().compose(slices_transform)

    slices_transform_mat = mat_update_resolution(slices_transform.matrix(), 1, res_r)

    if psf is None:
        psf = get_PSF(
            # r_max=5,
            res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r),
            device=volume.device,
        )

    return slices_transform_mat, res_s, res_r, s_thick, psf


def slices_scale(
    stack: Stack,
    slices_sim: Union[Stack, torch.Tensor],
    sample_weight: Optional[Union[Stack, torch.Tensor]] = None,
    pixel_weight: Optional[Union[Stack, torch.Tensor]] = None,
    use_mask: bool = False,
) -> torch.Tensor:
    if use_mask:
        scale = stack.slices * stack.mask
    else:
        scale = stack.slices
    if sample_weight is not None:
        if isinstance(sample_weight, Stack):
            sample_weight = sample_weight.slices
        scale *= sample_weight > 0.99
    if pixel_weight is not None:
        if isinstance(pixel_weight, Stack):
            pixel_weight = pixel_weight.slices
        scale *= pixel_weight
    if isinstance(slices_sim, Stack):
        slices_sim = slices_sim.slices

    scale = (scale * slices_sim).sum((1, 2, 3)) / (scale * stack.slices).sum((1, 2, 3))
    scale[~torch.isfinite(scale)] = 1.0
    return scale


def simulated_error(stack: Stack, slices_sim: Stack, scale: torch.Tensor) -> Stack:
    err = stack.slices * scale.view(-1, 1, 1, 1) - slices_sim.slices
    return Stack.like(stack, slices=err, deep=False)
