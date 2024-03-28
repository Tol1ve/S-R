import types
from typing import Dict, Any, Tuple, Callable, Union, cast, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..transform import RigidTransform, axisangle2mat, mat_update_resolution
from ..utils import ncc_loss, gaussian_blur, meshgrid, resample
from ..slice_acquisition import slice_acquisition
from ..image import Volume, Stack


class Registration(nn.Module):
    def __init__(
        self,
        num_levels: int = 3,
        num_steps: int = 4,
        step_size: float = 2,
        max_iter: int = 20,
        optimizer: Optional[Dict[str, Any]] = None,
        loss: Optional[Union[Dict[str, Any], Callable]] = None,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.current_level = self.num_levels - 1
        self.num_steps = [num_steps] * self.num_levels
        self.step_sizes = [step_size * 2**level for level in range(num_levels)]
        self.max_iter = max_iter
        self.auto_grad = False
        self._degree2rad = torch.tensor(
            [np.pi / 180, np.pi / 180, np.pi / 180, 1, 1, 1],
        ).view(1, 6)

        # init loss
        if loss is None:
            loss = {"name": "ncc", "win": None}
        if isinstance(loss, dict):
            loss_name = loss.pop("name")
            params = loss.copy()
            if loss_name == "mse":
                self.loss = types.MethodType(
                    lambda s, x, y: F.mse_loss(x, y, reduction="none", **params), self
                )
            elif loss_name == "ncc":
                self.loss = types.MethodType(
                    lambda s, x, y: ncc_loss(
                        x, y, reduction="none", level=s.current_level, **params
                    ),
                    self,
                )
            else:
                raise Exception("unknown loss")
        elif callable(loss):
            self.loss = types.MethodType(
                lambda s, x, y: cast(Callable, loss)(s, x, y), self
            )
        else:
            raise Exception("unknown loss")

        # init optimizer
        if optimizer is None:
            optimizer = {"name": "gd", "momentum": 0.1}
        if optimizer["name"] == "gd":
            if "momentum" not in optimizer:
                optimizer["momentum"] = 0
        self.optimizer = optimizer

    def degree2rad(self, theta: torch.Tensor) -> torch.Tensor:
        return theta * self._degree2rad

    def rad2degree(self, theta: torch.Tensor) -> torch.Tensor:
        return theta / self._degree2rad

    def clean_optimizer_state(self) -> None:
        if self.optimizer["name"] == "gd":
            if "buf" in self.optimizer:
                self.optimizer.pop("buf")

    def prepare(
        self,
        theta: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        params: Dict[str, Any],
    ) -> None:
        return

    def forward_tensor(
        self,
        theta: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        params: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._degree2rad = self._degree2rad.to(device=theta.device, dtype=theta.dtype)
        self.prepare(theta, source, target, params)
        theta0 = theta.clone()
        theta = self.rad2degree(theta.detach()).requires_grad_(self.auto_grad)
        with torch.set_grad_enabled(self.auto_grad):
            theta, loss = self.multilevel(theta, source, target)
        with torch.no_grad():
            dtheta = self.degree2rad(theta) - theta0
        return theta0 + dtheta, loss

    def update_level(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("")

    def multilevel(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for level in range(self.num_levels - 1, -1, -1):
            self.current_level = level
            source_new, target_new = self.update_level(theta, source, target)
            theta, loss = self.singlelevel(
                theta,
                source_new,
                target_new,
                self.num_steps[level],
                self.step_sizes[level],
            )
            self.clean_optimizer_state()

        return theta, loss

    def singlelevel(
        self,
        theta: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        num_steps: int,
        step_size: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for _ in range(num_steps):
            theta, loss = self.step(theta, source, target, step_size)
            step_size /= 2
        return theta, loss

    def step(
        self,
        theta: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        step_size: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.activate_idx = torch.ones(
            theta.shape[0], device=theta.device, dtype=torch.bool
        )
        loss_all = torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype)
        for _ in range(self.max_iter):
            theta_a, source_a, target_a = self.activate_set(theta, source, target)
            loss, grad = self.grad(theta_a, source_a, target_a, step_size)
            loss_all[self.activate_idx] = loss

            with torch.no_grad():
                step = self.optimizer_step(grad) * -step_size
                theta_a.add_(step)
                loss_new = self.evaluate(theta_a, source_a, target_a)
                idx_new = loss_new + 1e-4 < loss
                self.activate_idx[self.activate_idx.clone()] = idx_new
                if not torch.any(self.activate_idx):
                    break
                theta[self.activate_idx] += step[idx_new]

        return theta, loss_all.detach()

    def activate_set(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        theta = theta[self.activate_idx]
        if source.shape[0] > 1:
            source = source[self.activate_idx]
        if target.shape[0] > 1:
            target = target[self.activate_idx]
        return theta, source, target

    def grad(
        self,
        theta: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        step_size: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss = self.evaluate(theta, source, target)
        if self.auto_grad:
            grad = torch.autograd.grad([loss.sum()], [theta])[0]
        else:
            backup = torch.empty_like(theta[:, 0])
            grad = torch.zeros_like(theta)
            for j in range(theta.shape[1]):
                backup.copy_(theta[:, j])
                theta[:, j].copy_(backup + step_size)
                loss1 = self.evaluate(theta, source, target)
                theta[:, j].copy_(backup - step_size)
                loss2 = self.evaluate(theta, source, target)
                theta[:, j].copy_(backup)
                grad[:, j] = loss1 - loss2
        return loss, grad

    def warp(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("warp")

    def evaluate(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        warpped, target = self.warp(theta, source, target)
        loss = self.loss(warpped, target)
        loss = loss.view(loss.shape[0], -1).mean(1)
        return loss

    def optimizer_step(self, grad: torch.Tensor) -> torch.Tensor:
        if self.optimizer["name"] == "gd":
            step = self.gd_step(grad)
        else:
            raise Exception("unknown optimizer")
        step = step / (torch.linalg.norm(step, dim=-1, keepdim=True) + 1e-6)
        return step

    def gd_step(self, grad: torch.Tensor) -> torch.Tensor:
        if self.optimizer["momentum"]:
            if "buf" not in self.optimizer:
                self.optimizer["buf"] = grad.clone()
            else:
                self.optimizer["buf"][self.activate_idx] = (
                    self.optimizer["buf"][self.activate_idx]
                    * self.optimizer["momentum"]
                    + grad
                )
            return self.optimizer["buf"][self.activate_idx]
        else:
            return grad


class VolumeToVolumeRegistration(Registration):
    trans_first = False  # faster convergence

    def update_level(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_source = [
            0.5 * (2**self.current_level) / res for res in self.relative_res_source
        ]
        source = gaussian_blur(source, sigma_source, truncated=4.0)
        sigma_target = [
            0.5 * (2**self.current_level) / res for res in self.relative_res_target
        ]
        target = gaussian_blur(target, sigma_target, truncated=4.0)

        source = resample(
            source, self.relative_res_source[::-1], [2**self.current_level] * 3
        )
        target = resample(
            target, self.relative_res_target[::-1], [2**self.current_level] * 3
        )

        res_new = self.res * (2**self.current_level)
        mask = (target > 0).view(-1)

        grid = meshgrid(
            (target.shape[-1], target.shape[-2], target.shape[-3]),
            (res_new, res_new, res_new),
            device=target.device,
        )
        grid = grid.reshape(-1, 3)[mask, :]
        self._grid = grid

        self._target_flat = target.view(-1)[mask]

        scale = torch.tensor(
            [
                2.0 / (source.shape[-1] - 1),
                2.0 / (source.shape[-2] - 1),
                2.0 / (source.shape[-3] - 1),
            ],
            device=source.device,
            dtype=source.dtype,
        )
        self._grid_scale = scale / res_new

        return source, target

    def warp(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mat = (
            RigidTransform(self.degree2rad(theta), trans_first=self.trans_first)
            .inv()
            .matrix()
        )
        grid = torch.matmul(
            mat[:, :, :-1], self._grid.reshape(-1, 3, 1) + mat[:, :, -1:]
        )
        grid = grid.reshape(1, -1, 1, 1, 3)
        warpped = F.grid_sample(source, grid * self._grid_scale, align_corners=True)
        return warpped.view(1, 1, -1), self._target_flat.view(1, 1, -1)

    def prepare(
        self,
        theta: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        params: Dict[str, Any],
    ) -> None:
        assert source.ndim == 5 and target.ndim == 5
        res_source = params["res_source"]
        res_target = params["res_target"]
        self.res = min(res_source + res_target)
        self.relative_res_source = [r / self.res for r in res_source]
        self.relative_res_target = [r / self.res for r in res_target]

    def forward(
        self,
        source: Union[Stack, Volume],
        target: Union[Stack, Volume],
        use_mask: bool = False,
    ) -> Tuple[RigidTransform, torch.Tensor]:
        if isinstance(source, Stack):
            source = source.get_volume(copy=False)
        if isinstance(target, Stack):
            target = target.get_volume(copy=False)
        params = {
            "res_source": [
                source.resolution_z,
                source.resolution_y,
                source.resolution_x,
            ],
            "res_target": [
                target.resolution_z,
                target.resolution_y,
                target.resolution_x,
            ],
        }
        theta = (
            target.transformation.inv()
            .compose(source.transformation)
            .axisangle(self.trans_first)
        )

        if use_mask:
            source_input = source.image * source.mask
            target_input = target.image * target.mask
        else:
            source_input = source.image
            target_input = target.image

        theta, loss = self.forward_tensor(
            theta, source_input[None, None], target_input[None, None], params
        )

        transform_out = target.transformation.compose(
            RigidTransform(theta, trans_first=self.trans_first)
        )

        return transform_out, loss


class SliceToVolumeRegistration(Registration):
    trans_first = True  # due to slice_acquisition

    def update_level(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma = 0.5 * (2**self.current_level)
        source = gaussian_blur(source, sigma, truncated=4.0)
        target = gaussian_blur(target, sigma, truncated=4.0)
        target = resample(target, [1] * 2, [2**self.current_level] * 2)
        self.slices_mask_resampled = (
            (
                resample(
                    self.slices_mask.float(), [1] * 2, [2**self.current_level] * 2
                )
                > 0
            )
            if self.slices_mask is not None
            else None
        )
        return source, target

    def prepare(
        self,
        theta: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        params: Dict[str, Any],
    ) -> None:
        self.psf = torch.ones((1, 1, 1), device=theta.device, dtype=theta.dtype)
        self.res_s = params["res_s"]
        self.res_v = params["res_r"]
        assert len(source.shape) == 5
        assert len(target.shape) == 4

    def warp(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        transforms = axisangle2mat(self.degree2rad(theta))
        transforms = mat_update_resolution(transforms, 1, self.res_v)
        volume = source
        slices = target
        slices_mask: Optional[torch.Tensor]
        if self.slices_mask_resampled is not None:
            slices_mask = self.slices_mask_resampled[self.activate_idx]
            slices = slices * slices_mask
        else:
            slices_mask = None
        warpped = slice_acquisition(
            transforms,
            volume,
            self.volume_mask,
            slices_mask,
            self.psf,
            slices.shape[-2:],
            self.res_s * (2**self.current_level) / self.res_v,
            False,
            False,
        )
        return warpped, slices

    def forward(
        self,
        stack: Stack,
        volume: Volume,
        use_mask: bool = False,
    ) -> Tuple[RigidTransform, torch.Tensor]:
        eps = 1e-3
        assert (
            abs(volume.resolution_x - volume.resolution_y) < eps
            and abs(volume.resolution_x - volume.resolution_z) < eps
        ), "input volume should be isotropic!"
        assert (
            abs(stack.resolution_x - stack.resolution_y) < eps
        ), "input slices should be isotropic!"

        params = {"res_s": stack.resolution_x, "res_r": volume.resolution_x}

        slices_transform = stack.transformation
        volume_transform = volume.transformation

        slices_transform = volume_transform.inv().compose(slices_transform)
        theta = slices_transform.axisangle(self.trans_first)

        self.volume_mask = volume.mask[None, None] if use_mask else None
        self.slices_mask = stack.mask if use_mask else None

        theta, loss = self.forward_tensor(
            theta, volume.image[None, None], stack.slices, params
        )

        transform_out = RigidTransform(theta, trans_first=self.trans_first)
        transform_out = volume_transform.compose(transform_out)

        return transform_out, loss


def stack_registration(
    source_stacks: List[List[Stack]],
    centering: bool = False,
    args_registration: Optional[Dict] = None,
) -> List[Stack]:
    # stack registration
    vvr_args: Dict[str, Any] = {
        "num_levels": 3,
        "num_steps": 4,
        "step_size": 2,
        "max_iter": 20,
    }
    if args_registration is not None:
        vvr_args.update(args_registration)
    vvr = VolumeToVolumeRegistration(**vvr_args)

    target_stack = source_stacks[0][0]
    target = target_stack.get_volume(copy=False)
    sources = [[s.get_volume(copy=False) for s in ss] for ss in source_stacks]

    n_lists = len(sources)
    n_stacks = len(sources[0])

    ts_registered: List[RigidTransform] = []
    stacks_out: List[Stack] = []
    for j in range(n_stacks):
        if j == 0:
            ts_registered.append(target.transformation)
            stacks_out.append(target_stack)
        else:
            ncc_min: Union[float, torch.Tensor] = float("inf")
            for k in range(n_lists):
                sources[k][j].transformation = (
                    ts_registered[0]
                    .compose(sources[k][0].transformation.inv())
                    .compose(sources[k][j].transformation)
                )
                t_cur, ncc = vvr(sources[k][j], target, use_mask=True)
                if ncc < ncc_min:
                    ncc_min, t_best, s_best = ncc, t_cur, source_stacks[k][j]
            ts_registered.append(t_best)
            stacks_out.append(s_best)

    if centering:
        t_center_ax = ts_registered[0].axisangle(trans_first=False).clone()
        t_center_ax[..., :3] = 0
        t_center_ax[..., 3:] *= -1
        t_center = RigidTransform(t_center_ax)

    for s, t in zip(stacks_out, ts_registered):
        transform_init = s.init_stack_transform()
        transform_out = t.compose(transform_init)
        if centering:
            transform_out = t_center.compose(transform_out)
        s.transformation = transform_out

    return stacks_out
