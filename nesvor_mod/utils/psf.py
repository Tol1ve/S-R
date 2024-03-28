from typing import List, Tuple, Optional, Callable, Union
import torch
from math import log, sqrt
from .types import DeviceType

GAUSSIAN_FWHM = 1 / (2 * sqrt(2 * log(2)))
SINC_FWHM = 1.206709128803223 * GAUSSIAN_FWHM


def resolution2sigma(rx, ry=None, rz=None, /, isotropic=False):
    if isotropic:
        fx = fy = fz = GAUSSIAN_FWHM
    else:
        fx = fy = SINC_FWHM
        fz = GAUSSIAN_FWHM
    assert not ((ry is None) ^ (rz is None))
    if ry is None:
        if isinstance(rx, float) or isinstance(rx, int):
            if isotropic:
                return fx * rx
            else:
                return fx * rx, fy * rx, fz * rx
        elif isinstance(rx, torch.Tensor):
            if isotropic:
                return fx * rx
            else:
                assert rx.shape[-1] == 3
                return rx * torch.tensor([fx, fy, fz], dtype=rx.dtype, device=rx.device)
        elif isinstance(rx, List) or isinstance(rx, Tuple):
            assert len(rx) == 3
            return resolution2sigma(rx[0], rx[1], rx[2], isotropic=isotropic)
        else:
            raise Exception(str(type(rx)))
    else:
        return fx * rx, fy * ry, fz * rz


def get_PSF(
    r_max: Optional[int] = None,
    res_ratio: Tuple[float, float, float] = (1, 1, 3),
    threshold: float = 1e-3,
    device: DeviceType = torch.device("cpu"),
    psf_type: str = "gaussian",
) -> torch.Tensor:
    sigma_x, sigma_y, sigma_z = resolution2sigma(res_ratio, isotropic=False)
    if r_max is None:
        r_max = max(int(2 * r + 1) for r in (sigma_x, sigma_y, sigma_z))
        r_max = max(r_max, 4)
    x = torch.linspace(-r_max, r_max, 2 * r_max + 1, dtype=torch.float32, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(x, x, x, indexing="ij")
    if psf_type == "gaussian":
        psf = torch.exp(
            -0.5
            * (
                grid_x**2 / sigma_x**2
                + grid_y**2 / sigma_y**2
                + grid_z**2 / sigma_z**2
            )
        )
    elif psf_type == "sinc":
        # psf = (
        #     torch.sinc(grid_x / res_ratio[0])
        #     * torch.sinc(grid_y / res_ratio[1])
        #     * torch.exp(-0.5 * grid_z**2 / sigma_z**2)
        # )
        psf = torch.sinc(
            torch.sqrt((grid_x / res_ratio[0]) ** 2 + (grid_y / res_ratio[1]) ** 2)
        ) ** 2 * torch.exp(-0.5 * grid_z**2 / sigma_z**2)
    else:
        raise TypeError(f"Unknown PSF type: <{psf_type}>!")
    psf[psf.abs() < threshold] = 0
    rx = int(torch.nonzero(psf.sum((0, 1)) > 0)[0, 0].item())
    ry = int(torch.nonzero(psf.sum((0, 2)) > 0)[0, 0].item())
    rz = int(torch.nonzero(psf.sum((1, 2)) > 0)[0, 0].item())
    psf = psf[
        rz : 2 * r_max + 1 - rz, ry : 2 * r_max + 1 - ry, rx : 2 * r_max + 1 - rx
    ].contiguous()
    psf = psf / psf.sum()
    return psf


# class PSF:
#     def __init__(
#         self,
#         func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
#         device: DeviceType = torch.device("cpu"),
#     ) -> None:
#         self.func = func
#         self.device = device

#     def default_r_max(self) -> int:
#         return 4

#     def to_tensor(
#         self,
#         spacing: float,
#         r_max: Optional[int],
#         threshold: float = 1e-3,
#     ) -> torch.Tensor:
#         if r_max is None:
#             r_max = self.default_r_max()
#         x = (
#             torch.linspace(
#                 -r_max, r_max, 2 * r_max + 1, dtype=torch.float32, device=self.device
#             )
#             * spacing
#         )
#         grid_z, grid_y, grid_x = torch.meshgrid(x, x, x, indexing="ij")
#         psf = self.func(grid_x, grid_y, grid_z)
#         psf[psf.abs() < threshold * psf.abs().max()] = 0
#         rx = int(torch.nonzero(psf.sum((0, 1)) > 0)[0, 0].item())
#         ry = int(torch.nonzero(psf.sum((0, 2)) > 0)[0, 0].item())
#         rz = int(torch.nonzero(psf.sum((1, 2)) > 0)[0, 0].item())
#         psf = psf[
#             rz : 2 * r_max + 1 - rz, ry : 2 * r_max + 1 - ry, rx : 2 * r_max + 1 - rx
#         ].contiguous()
#         psf = psf / psf.sum()
#         return psf


# class GaussianPSF(PSF):
#     def __init__(
#         self,
#         sigma_x: float,
#         sigma_y: float,
#         sigma_z: float,
#         device: DeviceType = torch.device("cpu"),
#     ) -> None:
#         self.sigma_x = sigma_x
#         self.sigma_y = sigma_y
#         self.sigma_z = sigma_z
#         super().__init__(self.gaussian_func, device)

#     def gaussian_func(self, x, y, z):
#         return torch.exp(
#             -0.5
#             * (
#                 (x / self.sigma_x) ** 2
#                 + (y / self.sigma_y) ** 2
#                 + (z / self.sigma_z) ** 2
#             )
#         )

#     def sample(self) -> torch.Tensor:


# class IsotropicGaussianPSF(GaussianPSF):
#     pass


# class AnisotropicGaussianPSF(GaussianPSF):
#     pass


# class SincPSF(PSF):
#     pass
