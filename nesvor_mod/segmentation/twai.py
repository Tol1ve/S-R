import sys
import os
from argparse import Namespace
import shutil
import glob
import logging
import numpy as np
from ..image import load_mask
from ..utils import makedirs, LogIO
from .. import (
    TWAI_DIR,
    NIFTYREG_BIN_DIR,
    NNUNet_RAW_DATA_DIR,
    NNUNet_PREPROCESSED_DIR,
    NNUNET_TRAINED_MODELs_DIR,
)


def config_env():
    np.int = np.int32  # TWAI uses an older version of numpy
    twai_path = TWAI_DIR
    niftyreg_path = NIFTYREG_BIN_DIR
    os.environ["nnUNet_raw_data_base"] = NNUNet_RAW_DATA_DIR
    os.environ["nnUNet_preprocessed"] = NNUNet_PREPROCESSED_DIR
    os.environ["RESULTS_FOLDER"] = NNUNET_TRAINED_MODELs_DIR
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"] + ":/usr/local/lib/"
    os.environ["PATH"] = os.environ["PATH"] + ":" + niftyreg_path
    n_threads = 32  # make niftyreg a little bit faster
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    # os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    # os.environ["GOTO_NUM_THREADS"] = str(n_threads)
    # os.environ["MKL_NUM_THREADS"] = str(n_threads)

    # check twai module
    try:
        sys.path.append(twai_path)
        from src.utils import definitions

        # modify NIFTYREG_PATH before importing run_segment
        definitions.NIFTYREG_PATH = niftyreg_path
        from run_segment import main as entry
    except ImportError:
        logging.error(
            f"Fail to import the twai module, please make sure the twai repo is located in {twai_path}"
        )
        raise
    # check NiftyReg
    if not os.path.isfile(os.path.join(niftyreg_path, "reg_aladin")):
        raise FileNotFoundError(
            f"Cannot find NiftyReg, please make sure the niftyreg executables are located in {niftyreg_path}"
        )
    # check nnUNet
    try:
        import nnunet
    except:
        logging.error(f"nnUNet is not installed!")
        raise
    # check model weights
    model_path = os.path.join(
        NNUNET_TRAINED_MODELs_DIR, "nnUNet", "3d_fullres", "Task225_FetalBrain3dTrust"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find the weights of pretrained twai model, please download the weights to {model_path}"
        )

    return entry


def guess_ga(mask_path: str, condition: str, device) -> int:
    from src.utils.definitions import (
        ATLAS_CONTROL_HARVARD,
        ATLAS_CONTROL_CHINESE,
        ATLAS_SB,
    )

    logging.info("Try to guess GA using the volume of mask")

    if condition == "Neurotypical":
        datasets = [ATLAS_CONTROL_HARVARD, ATLAS_CONTROL_CHINESE]
    elif condition == "Spina Bifida":
        datasets = [ATLAS_SB]
    else:
        datasets = [ATLAS_CONTROL_HARVARD, ATLAS_CONTROL_CHINESE, ATLAS_SB]
    mask = load_mask(mask_path, device=device)
    v_mask = (
        (mask.image > 0).float().sum()
        * mask.resolution_x
        * mask.resolution_y
        * mask.resolution_z
    ).item()
    diff = float("inf")
    target_atlas = None
    for dataset in datasets:
        for f in glob.glob(os.path.join(dataset, "**", "mask.nii.gz")):
            atlas = load_mask(f, device=device)
            v_atlas = (
                (atlas.image > 0).float().sum()
                * atlas.resolution_x
                * atlas.resolution_y
                * atlas.resolution_z
            ).item()
            if abs(v_mask - v_atlas) < diff:
                diff = abs(v_mask - v_atlas)
                target_atlas = f
    assert target_atlas is not None, "Cannot find atlas!"
    name = target_atlas.split("/")[-2]
    ga = int("".join(c for c in name[:-1] if c.isdigit()))

    logging.info(f"Predicted GA = {ga}")
    logging.info(f"The closest atlas is {target_atlas}")

    return ga


def twai(args: Namespace) -> None:
    output_dir = args.output_folder
    workspace = "/home/.tmp/twai"
    name = args.name.replace(".nii", "").replace(".gz", "")
    tmp_input_dir = os.path.join(workspace, "inputs", name)
    tmp_output_dir = os.path.join(workspace, "outputs", name)
    makedirs(tmp_input_dir)
    makedirs(tmp_output_dir)
    makedirs(output_dir)

    if os.path.isfile(args.input_volume):
        tmp_input_file = os.path.join(
            tmp_input_dir, os.path.split(args.input_volume)[1]
        )
        shutil.copy(args.input_volume, tmp_input_file)
    else:
        raise ValueError("Input volume is not a file!")

    if args.volume_mask:
        if os.path.isfile(args.volume_mask):
            tmp_mask_file = os.path.join(
                tmp_input_dir, os.path.split(args.volume_mask)[1]
            )
            shutil.copy(args.volume_mask, tmp_mask_file)
        else:
            raise ValueError("Volume mask is not a file!")
    else:
        tmp_mask_file = os.path.join(
            tmp_input_dir, "mask_" + os.path.split(args.input_volume)[1]
        )
        shutil.copy(tmp_input_file, tmp_mask_file)

    cwd = os.getcwd()
    os.chdir(workspace)

    entry = config_env()

    if args.ga is None:
        logging.info("GA is not provided.")
        args.ga = guess_ga(tmp_mask_file, args.condition, args.device)

    args = Namespace(
        output_folder=tmp_output_dir,
        input=tmp_input_file,
        mask=tmp_mask_file,
        ga=args.ga,
        condition=args.condition,
        bfc=False,
    )

    logging.info("TWAI brain segmentation begins")
    sys.stdout = LogIO(logging.info)
    sys.stderr = LogIO(logging.error)
    entry(args)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    logging.info("TWAI brain segmentation finished")

    os.chdir(cwd)

    shutil.copytree(
        os.path.join(tmp_output_dir, ""),
        os.path.join(output_dir, ""),
        dirs_exist_ok=True,
    )
    shutil.rmtree(workspace, ignore_errors=True)

    logging.info(f"Final results are moved to {output_dir}")
