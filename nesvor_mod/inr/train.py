from argparse import Namespace
from typing import List, Tuple
import time
import datetime
import torch
import torch.optim as optim
import logging
from ..utils import MovingAverage, log_params, TrainLogger
from .models import INR, NeSVoR, D_LOSS, S_LOSS, DS_LOSS, I_REG, B_REG, T_REG, D_REG
from ..transform import RigidTransform
from ..image import Volume, Slice
from .data import PointDataset


def train(slices: List[Slice], args: Namespace) -> Tuple[INR, List[Slice], Volume]:
    # create training dataset
    dataset = PointDataset(slices)
    if args.n_epochs is not None:
        args.n_iter = args.n_epochs * (dataset.v.numel() // args.batch_size)

    use_scaling = True
    use_centering = True
    # perform centering and scaling
    spatial_scaling = 30.0 if use_scaling else 1
    bb = dataset.bounding_box
    center = (bb[0] + bb[-1]) / 2 if use_centering else torch.zeros_like(bb[0])
    ax = (
        RigidTransform(torch.cat([torch.zeros_like(center), -center])[None])
        .compose(dataset.transformation)
        .axisangle()
    )
    ax[:, -3:] /= spatial_scaling
    transformation = RigidTransform(ax)
    dataset.xyz /= spatial_scaling

    model = NeSVoR(
        transformation,
        dataset.resolution / spatial_scaling,
        dataset.mean,
        (bb - center) / spatial_scaling,
        spatial_scaling,
        args,
    )
    # setup optimizer
    params_net = []
    params_encoding = []
    for name, param in model.named_parameters():
        if param.numel() > 0:
            if "_net" in name:
                params_net.append(param)
            else:
                params_encoding.append(param)
    # logging
    logging.debug(log_params(model))
    optimizer = torch.optim.AdamW(
        params=[
            {"name": "encoding", "params": params_encoding},
            {"name": "net", "params": params_net, "weight_decay": 1e-2},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        eps=1e-15,
    )
    # setup scheduler for lr decay
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=list(range(1, len(args.milestones) + 1)),
        gamma=args.gamma,
    )
    decay_milestones = [int(m * args.n_iter) for m in args.milestones]
    # setup grad scalar for mixed precision training
    fp16 = not args.single_precision
    scaler = torch.cuda.amp.GradScaler(
        init_scale=1.0,
        enabled=fp16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    )
    # training
    model.train()
    loss_weights = {
        D_LOSS: 1,
        S_LOSS: 1,
        T_REG: args.weight_transformation,
        B_REG: args.weight_bias,
        I_REG: args.weight_image,
        D_REG: args.weight_deform,
    }
    average = MovingAverage(1 - 0.001)
    # logging
    logging_header = False
    logging.info("NeSVoR training starts.")
    train_time = 0.0
    for i in range(1, args.n_iter + 1):
        train_step_start = time.time()
        # forward
        batch = dataset.get_batch(args.batch_size, args.device)
        with torch.cuda.amp.autocast(fp16):
            losses = model(**batch)
            loss = 0
            for k in losses:
                if k in loss_weights and loss_weights[k]:
                    loss = loss + loss_weights[k] * losses[k]
        # backward
        scaler.scale(loss).backward()
        if args.debug:  # check nan grad
            for _name, _p in model.named_parameters():
                if _p.grad is not None and not _p.grad.isfinite().all():
                    logging.warning("iter %d: Found NaNs in the grad of %s", i, _name)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        train_time += time.time() - train_step_start
        for k in losses:
            average(k, losses[k].item())
        if (decay_milestones and i >= decay_milestones[0]) or i == args.n_iter:
            # logging
            if not logging_header:
                train_logger = TrainLogger(
                    "time",
                    "epoch",
                    "iter",
                    *list(losses.keys()),
                    "lr",
                )
                logging_header = True
            train_logger.log(
                datetime.timedelta(seconds=int(train_time)),
                dataset.epoch,
                i,
                *[average[k] for k in losses],
                optimizer.param_groups[0]["lr"],
            )
            if i < args.n_iter:
                decay_milestones.pop(0)
                scheduler.step()
            # check scaler
            if scaler.is_enabled():
                current_scaler = scaler.get_scale()
                if current_scaler < 1 / (2**5):
                    logging.warning(
                        "Numerical instability detected! "
                        "The scale of GradScaler is %f, which is too small. "
                        "The results might be suboptimal. "
                        "Try to set --single-precision or run the command again with a different random seed."
                    )
                if i == args.n_iter:
                    logging.debug("Final scale of GradScaler = %f" % current_scaler)

    # outputs
    transformation = model.transformation

    # undo centering and scaling
    ax = transformation.axisangle()
    ax[:, -3:] *= spatial_scaling
    transformation = RigidTransform(ax)
    transformation = RigidTransform(
        torch.cat([torch.zeros_like(center), center])[None]
    ).compose(transformation)
    model.inr.bounding_box.copy_(bb)
    dataset.xyz *= spatial_scaling

    dataset.transformation = transformation
    mask = dataset.mask
    output_slices = []
    for i in range(len(slices)):
        output_slice = slices[i].clone()
        output_slice.transformation = transformation[i]
        output_slices.append(output_slice)
    return model.inr, output_slices, mask
