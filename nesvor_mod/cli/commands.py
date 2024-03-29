import time
import argparse
import logging
import re
import os
import torch
from typing import List, Optional, Tuple, Dict, Any, cast
from ..image import Stack, Slice, Volume
from ..svort.inference import svort_predict
from ..inr.train import train
from ..inr.models import INR
from ..inr.sample import sample_volume, sample_slices, override_sample_mask
from .io import outputs, inputs
from ..utils import makedirs, log_args, log_result
from ..preprocessing import n4_bias_field_correction, assess, brain_segmentation
from ..segmentation import twai
from ..svr import slice_to_volume_reconstruction

"base of commands"


class Command(object):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.timer: List[Tuple[Optional[str], float]] = []

    def check_args(self) -> None:
        pass

    def get_command(self) -> str:
        return "-".join(
            w.lower() for w in re.findall("[A-Z]+[^A-Z]*", self.__class__.__name__)
        )

    def new_timer(self, name: Optional[str] = None) -> None:
        t = time.time()
        if len(self.timer) > 1 and self.timer[-1][0] is not None:
            # the previous timer ends
            logging.info(
                "%s finished in %.1f s", self.timer[-1][0], t - self.timer[-1][1]
            )
        if name is None:
            if len(self.timer) == 0:  # begining of command
                pass
            else:  # end of command
                logging.info(
                    "Command 'nesvor %s' finished, overall time: %.1f s",
                    self.get_command(),
                    t - self.timer[0][1],
                )
        else:
            logging.info("%s starts ...", name)
        self.timer.append((name, t))

    def makedirs(self) -> None:
        keys = ["output_slices", "simulated_slices"]
        makedirs([getattr(self.args, k, None) for k in keys])

        keys = ["output_model", "output_volume"]
        for k in keys:
            if getattr(self.args, k, None):
                makedirs(os.path.dirname(getattr(self.args, k)))

        keys = ["output_stack_masks", "output_corrected_stacks"]
        for k in keys:
            if getattr(self.args, k, None):
                for f in getattr(self.args, k):
                    makedirs(os.path.dirname(f))

    def main(self) -> None:
        self.check_args()
        log_args(self.args)
        self.makedirs()
        self.new_timer()
        self.exec()
        self.new_timer()

        if "cuda" in str(self.args.device):
            logging.debug(
                "Max GPU memory allocated = %.3f GB",
                torch.cuda.max_memory_allocated(self.args.device) / (1024**3),
            )
            logging.debug(
                "Max GPU memory reserved = %.3f GB",
                torch.cuda.max_memory_reserved(self.args.device) / (1024**3),
            )

    def exec(self) -> None:
        raise NotImplementedError("The exec method for Command is not implemented.")


"commands"


class Reconstruct(Command):
    def check_args(self) -> None:
        # input
        check_input_stacks_slices(self.args)
        # output
        if self.args.output_volume is None and self.args.output_model is None:
            logging.warning("Both <output-volume> and <output-model> are not provided.")
        if not self.args.inference_batch_size:
            self.args.inference_batch_size = 8 * self.args.batch_size
        if not self.args.n_inference_samples:
            self.args.n_inference_samples = 2 * self.args.n_samples
        # deformable
        if self.args.deformable:
            if not self.args.single_precision:
                logging.warning(
                    "Fitting deformable model with half precision can be unstable! Try single precision instead."
                )
            if "svort" in self.args.registration:
                logging.warning(
                    "SVoRT can only be used for rigid registration in fetal brain MRI."
                )
        # assessment
        check_cutoff(self.args)
        # registration
        svort_v1_warning(self.args)
        # dtype
        self.args.dtype = torch.float32 if self.args.single_precision else torch.float16
        if "cpu" in str(self.args.device):
            self.args.dtype = torch.float32
            if not self.args.single_precision:
                logging.warning(
                    "CPU mode does not support half precision training. Will use single precision instead."
                )
                self.args.single_precision = True

    def preprocess(self) -> Dict[str, Any]:
        self.new_timer("Data loading")
        input_dict, self.args = inputs(self.args)
        if "input_stacks" in input_dict and input_dict["input_stacks"]:
            if self.args.segmentation:
                self.new_timer("Segmentation")
                input_dict["input_stacks"] = _segment_stack(
                    self.args, input_dict["input_stacks"]
                )
            if self.args.bias_field_correction:
                self.new_timer("Bias Field Correction")
                input_dict["input_stacks"] = _correct_bias_field(
                    self.args, input_dict["input_stacks"]
                )
            if self.args.metric != "none":
                self.new_timer("Assessment")
                input_dict["input_stacks"], _ = _assess(
                    self.args, input_dict["input_stacks"], False
                )
            self.new_timer("Registration")
            input_dict["input_slices"] = _register(
                self.args, input_dict["input_stacks"]
            )
        elif "input_slices" in input_dict and input_dict["input_slices"]:
            pass
        else:
            raise ValueError("No data found!")
        return input_dict

    def exec(self) -> None:
        input_dict = self.preprocess()
        self.new_timer("Reconsturction")
        model, output_slices, mask = train(input_dict["input_slices"], self.args)
        self.new_timer("Results saving")
        output_volume, simulated_slices = _sample_inr(
            self.args,
            model,
            input_dict["volume_mask"]
            if (getattr(input_dict, "volume_mask", None) is not None)
            else mask,
            output_slices,
            getattr(self.args, "output_volume", None) is not None,
            getattr(self.args, "simulated_slices", None) is not None,
        )
        outputs(
            {
                "output_volume": output_volume,
                "mask": mask,
                "output_model": model,
                "output_slices": output_slices,
                "simulated_slices": simulated_slices,
            },
            self.args,
        )


class SampleVolume(Command):
    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, self.args = inputs(self.args)
        self.new_timer("Volume sampling")
        v, _ = _sample_inr(
            self.args,
            input_dict["model"],
            input_dict["mask"],
            None,
            True,
            False,
        )
        self.new_timer("Results saving")
        outputs({"output_volume": v}, self.args)


class SampleSlices(Command):
    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, self.args = inputs(self.args)
        self.new_timer("Slices sampling")
        _, simulated_slices = _sample_inr(
            self.args,
            input_dict["model"],
            input_dict["mask"],
            input_dict["input_slices"],
            False,
            True,
        )
        self.new_timer("Results saving")
        outputs({"simulated_slices": simulated_slices}, self.args)


class Register(Command):
    def check_args(self) -> None:
        check_len(self.args, "input_stacks", "stack_masks")
        check_len(self.args, "input_stacks", "thicknesses")
        svort_v1_warning(self.args)

    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, self.args = inputs(self.args)
        if not ("input_stacks" in input_dict and input_dict["input_stacks"]):
            raise ValueError("No data found!")
        self.new_timer("Registration")
        slices = _register(self.args, input_dict["input_stacks"])
        self.new_timer("Results saving")
        outputs({"output_slices": slices}, self.args)


class SegmentStack(Command):
    def check_args(self) -> None:
        if len(self.args.output_stack_masks) == 1:
            folder = self.args.output_stack_masks[0]
            if not (folder.endswith(".nii") or folder.endswith(".nii.gz")):
                # it is a folder
                self.args.output_stack_masks = [
                    os.path.join(folder, "mask_" + os.path.basename(p))
                    for p in self.args.input_stacks
                ]
        check_len(self.args, "input_stacks", "output_stack_masks")

    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, self.args = inputs(self.args)
        if not ("input_stacks" in input_dict and input_dict["input_stacks"]):
            raise ValueError("No data found!")
        self.new_timer("Segmentation")
        seg_stacks = _segment_stack(self.args, input_dict["input_stacks"])
        self.new_timer("Results saving")
        outputs(
            {"output_stack_masks": [stack.get_mask_volume() for stack in seg_stacks]},
            self.args,
        )


class CorrectBiasField(Command):
    def check_args(self) -> None:
        if len(self.args.output_corrected_stacks) == 1:
            folder = self.args.output_corrected_stacks[0]
            if not (folder.endswith(".nii") or folder.endswith(".nii.gz")):
                # it is a folder
                self.args.output_corrected_stacks = [
                    os.path.join(folder, "corrected_" + os.path.basename(p))
                    for p in self.args.input_stacks
                ]
        check_len(self.args, "input_stacks", "output_corrected_stacks")

    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, self.args = inputs(self.args)
        if not ("input_stacks" in input_dict and input_dict["input_stacks"]):
            raise ValueError("No data found!")
        self.new_timer("Bias field correction")
        corrected_stacks = _correct_bias_field(self.args, input_dict["input_stacks"])
        self.new_timer("Results saving")
        outputs(
            {
                "output_corrected_stacks": [
                    stack.get_volume() for stack in corrected_stacks
                ]
            },
            self.args,
        )


class Assess(Command):
    def check_args(self) -> None:
        check_cutoff(self.args)
        if self.args.metric == "none":
            raise ValueError("--metric should not be none is `assess` command.")

    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, self.args = inputs(self.args)
        self.new_timer("Assessment")
        _, results = _assess(self.args, input_dict["input_stacks"], True)
        if self.args.output_json:
            self.new_timer("Results saving")
            self.args.output_assessment = results
            outputs({}, self.args)
            log_result("Assessment results saved to %s" % self.args.output_json)


class SegmentVolume(Command):
    def exec(self) -> None:
        self.new_timer("volume segmentation")
        self.args.name = "segmentation"
        twai(self.args)


class SVR(Reconstruct):
    def check_args(self) -> None:
        # input
        check_input_stacks_slices(self.args)
        # assessment
        check_cutoff(self.args)
        # registration
        svort_v1_warning(self.args)
        # optimization
        if len(self.args.n_iter_rec) == 1:
            self.args.n_iter_rec = [self.args.n_iter_rec[0]] * self.args.n_iter
            self.args.n_iter_rec[-1] *= 3
        assert (
            len(self.args.n_iter_rec) == self.args.n_iter
        ), "the length of n_iter_rec should be equal to n_iter!"

    def exec(self) -> None:
        input_dict = self.preprocess()
        self.new_timer("Reconsturction")
        output_volume, output_slices, simulated_slices = slice_to_volume_reconstruction(
            input_dict["input_slices"], **vars(self.args)
        )
        self.new_timer("Results saving")
        outputs(
            {
                "output_volume": output_volume,
                "output_slices": output_slices,
                "simulated_slices": simulated_slices,
            },
            self.args,
        )
Svr = SVR


class ConditionalRepresent(Command):
    def check_args(self) -> None:
        # input
        check_input_stacks_slices(self.args)
    def preprocess(self) -> Dict[str, Any]:
        input_dict, self.args = inputs(self.args)
        # segmentation
        if self.args.segmentation:
            self.new_timer("Volume segmentation")
    def exec(self) -> None:
        input_dict = self.preprocess()
        self.new_timer("Reconsturction")
        output_volume, output_slices, simulated_slices = slice_to_volume_reconstruction(
        
    )


class SemanticSegment(Command):
    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, self.args = inputs(self.args)
        self.new_timer("Reconsturction")
        output_volume, output_slices, simulated_slices = slice_to_volume_reconstruction(
            input_dict["input_slices"], **vars(self.args)
        )

"""helper functions"""


def _segment_stack(args: argparse.Namespace, data: List[Stack]) -> List[Stack]:
    data = brain_segmentation(
        data,
        args.device,
        args.batch_size_seg,
        not args.no_augmentation_seg,
        args.dilation_radius_seg,
        args.threshold_small_seg,
    )
    return data


def _register(args: argparse.Namespace, data: List[Stack]) -> List[Slice]:
    if args.registration == "svort":
        svort = True
        vvr = True
        force_vvr = False
    elif args.registration == "svort-stack":
        svort = True
        vvr = True
        force_vvr = True
    elif args.registration == "svort-only":
        svort = True
        vvr = False
        force_vvr = False
    elif args.registration == "stack":
        svort = False
        vvr = True
        force_vvr = False
    elif args.registration == "none":
        svort = False
        vvr = False
        force_vvr = False
    else:
        raise ValueError("Unkown registration method!")
    force_scanner = args.scanner_space
    slices = svort_predict(
        data, args.device, args.svort_version, svort, vvr, force_vvr, force_scanner
    )
    return slices


def _correct_bias_field(args: argparse.Namespace, stacks: List[Stack]) -> List[Stack]:
    n4_params = {}
    for k in vars(args):
        if k.endswith("_n4"):
            n4_params[k] = getattr(args, k)
    return n4_bias_field_correction(stacks, n4_params)


def _assess(
    args: argparse.Namespace, stacks: List[Stack], print_results=False
) -> Tuple[List[Stack], List[Dict[str, Any]]]:
    filtered_stacks, results = assess(
        stacks,
        args.metric,
        args.filter_method,
        args.cutoff,
        args.batch_size_assess,
        not args.no_augmentation_assess,
        args.device,
    )
    if results:
        descending = results[0]["descending"]
        template = "\n%15s %25s %15s %15s %15s"
        result_log = (
            "stack assessment results (metric = %s):" % args.metric
            + template
            % (
                "stack",
                "name",
                "score " + "(" + ("\u2191" if descending else "\u2193") + ")",
                "rank",
                "",
            )
        )
        for item in results:
            name = item["name"].replace(".gz", "").replace(".nii", "")
            name = name if len(name) <= 20 else ("..." + name[-17:])
            result_log += template % (
                item["input_id"],
                name,
                ("%1.4f" if isinstance(item["score"], float) else "%d") % item["score"],
                item["rank"],
                "excluded" if item["excluded"] else "",
            )
        if print_results:
            log_result(result_log)
        else:
            logging.info(result_log)

    logging.debug(
        "Input stacks after assessment and filtering: %s",
        [s.name for s in filtered_stacks],
    )

    return filtered_stacks, results


def _sample_inr(
    args: argparse.Namespace,
    model: INR,
    mask: Volume,
    slices_template: Optional[List[Slice]] = None,
    return_volume=False,
    return_slices=False,
) -> Tuple[Optional[Volume], Optional[List[Slice]]]:
    if return_slices:
        assert slices_template is not None, "slices tempalte can not be None!"

    mask = override_sample_mask(
        mask,
        getattr(args, "sample_mask", None),
        getattr(args, "output_resolution", None),
        getattr(args, "sample_orientation", None),
    )

    output_volume = (
        sample_volume(
            model,
            mask,
            args.output_resolution * args.output_psf_factor,
            args.inference_batch_size,
            args.n_inference_samples,
        )
        if return_volume
        else None
    )

    simulated_slices = (
        sample_slices(
            model,
            cast(List[Slice], slices_template),
            mask,
            args.output_psf_factor,
            args.n_inference_samples,
        )
        if return_slices
        else None
    )
    return output_volume, simulated_slices


"""warnings and checks"""


def svort_v1_warning(args: argparse.Namespace) -> None:
    if "svort" in args.registration and args.svort_version == "v1":
        logging.warning(
            "SVoRT v1 model use a different altas space. If you want to register the image to in the CRL fetal brain atlas space, try the v2 model."
        )


def check_len(args: argparse.Namespace, k1: str, k2: str) -> None:
    if getattr(args, k1, None) and getattr(args, k2, None):
        assert len(getattr(args, k1)) == len(
            getattr(args, k2)
        ), "The length of {k1} and {k2} are different!"


def check_cutoff(args: argparse.Namespace) -> None:
    if args.filter_method != "none" and args.cutoff is None:
        raise ValueError("--cutoff for filtering is not provided!")


def check_input_stacks_slices(args: argparse.Namespace) -> None:
    # input
    assert (
        args.input_slices is not None or args.input_stacks is not None
    ), "No image data provided! Use --input-slices or --input-stacks to input data."
    if args.input_slices is not None:
        # use input slices
        if (
            args.stack_masks is not None
            or args.input_stacks is not None
            or args.thicknesses is not None
        ):
            logging.warning(
                "Since <input-slices> is provided, <input-stacks>, <stack_masks> and <thicknesses> would be ignored."
            )
            args.stack_masks = None
            args.input_stacks = None
            args.thicknesses = None
    else:
        # use input stacks
        check_len(args, "input_stacks", "stack_masks")
        if args.thicknesses is not None:
            if len(args.thicknesses) == 1:
                args.thicknesses = args.thicknesses * len(args.input_stacks)
        check_len(args, "input_stacks", "thicknesses")
