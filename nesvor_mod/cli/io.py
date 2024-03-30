import torch
from typing import Dict, Tuple, Any, Optional
from argparse import Namespace
import json
import os
import logging
from ..image import (
    Volume,
    save_slices,
    load_slices,
    load_stack,
    load_mask,
    load_volume,
)
from ..inr.models import INR
from ..utils import merge_args
from ..preprocessing import stack_intersect, otsu_thresholding, thresholding


def inputs(args: Namespace) -> Tuple[Dict, Namespace]:
    input_dict: Dict[str, Any] = dict()
    if getattr(args, "input_stacks", None) is not None:
        input_stacks = []
        logging.info("loading stacks")
        for i, f in enumerate(args.input_stacks):
            stack = load_stack(
                f,
                args.stack_masks[i]
                if getattr(args, "stack_masks", None) is not None
                else None,
                device=args.device,
            )
            if getattr(args, "thicknesses", None) is not None:
                stack.thickness = args.thicknesses[i]
            input_stacks.append(stack)
        # stack thresholding
        logging.info("background thresholding")
        input_stacks = thresholding(input_stacks, args.background_threshold)
        if getattr(args, "otsu_thresholding", False):
            logging.info("applying otsu thresholding")
            input_stacks = otsu_thresholding(input_stacks)
        # volume mask
        volume_mask: Optional[Volume]
        if getattr(args, "volume_mask", None):
            logging.info("loading volume mask")
            volume_mask = load_mask(args.volume_mask, device=args.device)
        elif getattr(args, "stacks_intersection", False):
            logging.info("creating volume mask using intersection of stacks")
            volume_mask = stack_intersect(input_stacks, box=True)
        else:
            volume_mask = None
        if volume_mask is not None:
            logging.info("applying volume mask")
            for stack in input_stacks:
                stack.apply_volume_mask(volume_mask)
        input_dict["input_stacks"] = input_stacks
        input_dict["volume_mask"] = volume_mask
    if getattr(args, "input_slices", None) is not None:
        logging.info("loading slices")
        input_slices = load_slices(args.input_slices, args.device)
        input_dict["input_slices"] = input_slices
    if getattr(args, "input_model", None) is not None:
        logging.info("loading model")
        cp = torch.load(args.input_model, map_location=args.device)
        input_dict["model"] = INR(cp["model"]["bounding_box"], cp["args"])
        input_dict["model"].load_state_dict(cp["model"])
        input_dict["mask"] = cp["mask"]
        args = merge_args(cp["args"], args)
    return input_dict, args


def outputs(data: Dict, args: Namespace) -> None:
    if getattr(args, "output_volume", None) and "output_volume" in data:
        if args.output_intensity_mean:
            data["output_volume"].rescale(args.output_intensity_mean)
        data["output_volume"].save(
            args.output_volume, masked=not getattr(args, "with_background", False)
        )
    if getattr(args, "output_model", None) and "output_model" in data:
        torch.save(
            {
                "model": data["output_model"].state_dict(),
                "mask": data["mask"],
                "args": args,
            },
            args.output_model,
        )
    if getattr(args, "output_slices", None) and "output_slices" in data:
        save_slices(args.output_slices, data["output_slices"], sep=True)
    if getattr(args, "simulated_slices", None) and "simulated_slices" in data:
        save_slices(args.simulated_slices, data["simulated_slices"], sep=False)
    for k in ["output_stack_masks", "output_corrected_stacks"]:
        if getattr(args, k, None) and k in data:
            for m, p in zip(data[k], getattr(args, k)):
                m.save(p)
    if getattr(args, "output_json", None):
        d = vars(args)
        d["device"] = int(str(d["device"]).split(":")[-1])
        with open(args.output_json, "w") as outfile:
            outfile.write(json.dumps(d, indent=4))


def load_model(args: Namespace) -> Tuple[INR, Volume, Namespace]:
    cp = torch.load(args.input_model, map_location=args.device)
    inr = INR(cp["model"]["bounding_box"], cp["args"])
    inr.load_state_dict(cp["model"])
    mask = cp["mask"]
    args = merge_args(cp["args"], args)
    return inr, mask, args


def inputs_dataset(args: Namespace) -> Tuple[Dict, Namespace]:
    input_dict: Dict[str, Any] = dict()
    if getattr(args, "input_dataset", None) is not None:
        
        input_volumes = []
        logging.info("loading dataset")
        if not os.path.exists(args.input_dataset):
            raise FileNotFoundError(f"no such directionary：{args.input_dataset}")
        #operation for the feta dataset
        elif 'feta' in args.input_dataset:
            subfiles=[]
            if args.input_dataset_mask=='_':
                submasks=[]
            for d in os.listdir(args.input_dataset):
                if os.path.isdir(os.path.join(args.input_dataset, d)) and 'sub-' in d:
                    _path=os.path.join(os.path.join(args.input_dataset, d),"anat")
                    for f in os.listdir(_path):
                        if f.endswith('.nii.gz') and 'seg' not in  f:
                            _imagepath=os.path.join(_path, f)
                            subfiles.append(_imagepath)
                        if args.input_dataset_mask=='_':
                            if f.endswith('.nii.gz') and 'seg' in  f:
                                _maskpath=os.path.join(_path, f)
                                submasks.append(_maskpath)
                                
        else:
            raise NotImplementedError("not able for dataset：{}".format(args.input_dataset))


        for i, f in enumerate(subfiles):
            volume = load_volume(
                f,
                submasks
                if getattr(args, "inputs_dataset_mask", None) is not None
                else None,
                device=args.device,
            )
            input_volumes.append(volume)



        input_dict["input_dataset"] = input_volumes
        return input_dict, args
 