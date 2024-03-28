import argparse
from typing import Union, Sequence, Optional, Tuple
from .formatters import CommandHelpFormatter, MainHelpFormatter
from .. import __version__, __url__
from .docs import rst, not_doc, doc_mode, show_link, prepare_parser_for_sphinx


# parents parsers


def build_parser_training() -> argparse.ArgumentParser:
    """arguments related to the training of INR"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("model architecture")
    # hash grid encoding
    parser.add_argument(
        "--n-features-per-level",
        default=2,
        type=int,
        help="Length of the feature vector at each level.",
    )
    parser.add_argument(
        "--log2-hashmap-size",
        default=19,
        type=int,
        help="Max log2 size of the hash grid per level.",
    )
    parser.add_argument(
        "--level-scale",
        default=1.3819,
        type=float,
        help="Scaling factor between two levels.",
    )
    parser.add_argument(
        "--coarsest-resolution",
        default=16.0,
        type=float,
        help="Resolution of the coarsest grid in millimeter.",
    )
    parser.add_argument(
        "--finest-resolution",
        default=0.5,
        type=float,
        help="Resolution of the finest grid in millimeter.",
    )
    parser.add_argument(
        "--n-levels-bias",
        default=0,
        type=int,
        help="Number of levels used for bias field estimation.",
    )
    # implicit network
    parser.add_argument(
        "--depth", default=1, type=int, help="Number of hidden layers in MLPs."
    )
    parser.add_argument(
        "--width", default=64, type=int, help="Number of neuron in each hidden layer."
    )
    parser.add_argument(
        "--n-features-z",
        default=15,
        type=int,
        help="Length of the intermediate feature vector z.",
    )
    parser.add_argument(
        "--n-features-slice",
        default=16,
        type=int,
        help="Length of the slice embedding vector e.",
    )
    parser.add_argument(
        "--no-transformation-optimization",
        action="store_true",
        help="Disable optimization for rigid slice transfromation, i.e., the slice transformations are fixed",
    )
    parser.add_argument(
        "--no-slice-scale",
        action="store_true",
        help="Disable adaptive scaling for slices.",
    )
    parser.add_argument(
        "--no-pixel-variance",
        action="store_true",
        help="Disable pixel-level variance.",
    )
    parser.add_argument(
        "--no-slice-variance",
        action="store_true",
        help="Disable slice-level variance.",
    )
    parser = _parser.add_argument_group("model architecture (deformable part)")
    # deformable net
    parser.add_argument(
        "--deformable",
        action="store_true",
        help="Enable implicit deformation field.",
    )
    parser.add_argument(
        "--n-features-deform",
        default=8,
        type=int,
        help="Length of the deformation embedding vector.",
    )
    parser.add_argument(
        "--n-features-per-level-deform",
        default=4,
        type=int,
        help="Length of the feature vector at each level (deformation field).",
    )
    parser.add_argument(
        "--level-scale-deform",
        default=1.3819,
        type=float,
        help="Scaling factor between two levels (deformation field).",
    )
    parser.add_argument(
        "--coarsest-resolution-deform",
        default=32.0,
        type=float,
        help="Resolution of the coarsest grid in millimeter (deformation field).",
    )
    parser.add_argument(
        "--finest-resolution-deform",
        default=8.0,
        type=float,
        help="Resolution of the finest grid in millimeter (deformation field).",
    )

    # loss function
    parser = _parser.add_argument_group("loss and regularization")
    # rigid transformation
    parser.add_argument(
        "--weight-transformation",
        default=0.1,
        type=float,
        help="Weight of transformation regularization.",
    )
    # bias field
    parser.add_argument(
        "--weight-bias",
        default=100.0,
        type=float,
        help="Weight of bias field regularization.",
    )
    # image regularization
    parser.add_argument(
        "--image-regularization",
        default="edge",
        type=str,
        choices=["TV", "edge", "L2", "none"],
        help=rst(
            "Type of image regularization. \n\n"
            "1. ``TV``: total variation (L1 regularization of image gradient); \n"
            "2. ``edge``: edge-preserving regularization, see `--delta <#delta>`__\ . \n"
            "3. ``L2``: L2 regularization of image gradient; \n"
            "4. ``none``: no image regularization. \n\n"
        ),
    )
    parser.add_argument(
        "--weight-image",
        default=1.0,
        type=float,
        help="Weight of image regularization.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,
        help=rst(
            "Parameter to define intensity of an edge in edge-preserving regularization. "
            "See `--image-regularization <#image-regularization>`__\ ."
            "The edge-preserving regularization becomes L1 when ``delta`` goes to 0."
        ),
    )
    parser.add_argument(
        "--img-reg-autodiff",
        action="store_true",
        help=(
            "Use auto diff to compute the image graident in the image regularization. "
            "By default, the finite difference is used."
        ),
    )
    # deformation regularization
    parser.add_argument(
        "--weight-deform",
        default=0.1,
        type=float,
        help="Weight of deformation regularization ",
    )

    # training
    parser = _parser.add_argument_group("training")
    parser.add_argument(
        "--learning-rate",
        default=5e-3,
        type=float,
        help="Learning rate of Adam optimizer.",
    )
    parser.add_argument(
        "--gamma",
        default=0.33,
        type=float,
        help="Multiplicative factor of learning rate decay.",
    )
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 0.9],
        help="List of milestones of learning rate decay. Must be in (0, 1) and increasing.",
    )
    parser.add_argument(
        "--n-iter", default=6000, type=int, help="Number of iterations for training."
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        help=rst(
            "Number of epochs for training. If provided, will ignore `--n-iter <#n-iter>`__"
        ),
    )
    parser.add_argument(
        "--batch-size", default=1024 * 4, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--n-samples",
        default=128 * 2,
        type=int,
        help="Number of sample for PSF during training.",
    )
    parser.add_argument(
        "--single-precision",
        action="store_true",
        help="use float32 training (default: float16/float32 mixed trainig)",
    )
    return _parser


def build_parser_inputs(
    *,
    input_stacks: Union[bool, str] = False,
    input_slices: Union[bool, str] = False,
    input_model: Union[bool, str] = False,
    input_volume: Union[bool, str] = False,
    no_thickness=False,
) -> argparse.ArgumentParser:
    """arguments related to input data"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("inputs")
    # stack input
    if input_stacks:
        parser.add_argument(
            "--input-stacks",
            nargs="+",
            type=str,
            required=input_stacks == "required",
            help="Paths to the input stacks (NIfTI).",
        )
        if not no_thickness:
            parser.add_argument(
                "--thicknesses",
                nargs="+",
                type=float,
                help=(
                    "Slice thickness of each input stack. "
                    "If not provided, use the slice gap of the input stack. "
                    "If only a single number is provided, Assume all input stacks have the same thickness."
                ),
            )
    # slices input
    if input_slices:
        parser.add_argument(
            "--input-slices",
            type=str,
            required=input_slices == "required",
            help="Folder of the input slices. "
            if not input_stacks
            else (
                "Folder of the input slices. "
                "i.e., the motion corrected slices generated by `nesvor register`. "
                "If input-slices are provided and input-stacks will be ignored. "
            ),
        )
    # input model
    if input_model:
        parser.add_argument(
            "--input-model",
            type=str,
            required=input_model == "required",
            help="Path to the trained NeSVoR model.",
        )
    # input volume
    if input_volume:
        parser.add_argument(
            "--input-volume",
            type=str,
            required=input_volume == "required",
            help="Path to the input 3D volume.",
        )
        parser.add_argument(
            "--volume-mask",
            type=str,
            help=(
                "Paths to a 3D mask of ROI in the volume. "
                "Will use the non-zero region of the input volume if not provided"
            ),
        )
    return _parser


def build_parser_stack_masking(*, stack_masks=True) -> argparse.ArgumentParser:
    """arguments related to ROI maksing for input stacks"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("input stacks masking")
    if stack_masks:
        parser.add_argument(
            "--stack-masks",
            nargs="+",
            type=str,
            help="Paths to masks of input stacks.",
        )
    parser.add_argument(
        "--volume-mask",
        type=str,
        help="Paths to a 3D mask which will be applied to each input stack.",
    )
    parser.add_argument(
        "--stacks-intersection",
        action="store_true",
        help=rst(
            "Only consider the region defined by the intersection of input stacks. "
            "Will be ignored if `--volume-mask <#volume-mask>`__ is provided."
        ),
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=0,
        help="pixels with value <= this threshold will be ignored.",
    )
    parser.add_argument(
        "--otsu-thresholding",
        action="store_true",
        help="Apply Otsu thresholding to each input stack.",
    )
    return _parser


def build_parser_outputs(
    *,
    output_volume: Union[bool, str] = False,
    output_slices: Union[bool, str] = False,
    simulate_slices: Union[bool, str] = False,
    output_model: Union[bool, str] = False,
    output_stack_masks: Union[bool, str] = False,
    output_corrected_stacks: Union[bool, str] = False,
    output_folder: Union[bool, str] = False,
    output_json: Union[bool, str] = True,
    **kwargs,
) -> argparse.ArgumentParser:
    """arguments related to ouptuts"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("outputs")
    # output volume
    if output_volume:
        parser.add_argument(
            "--output-volume",
            type=str,
            required=output_volume == "required",
            help="Paths to the reconstructed volume",
        )
    # output slices
    if output_slices:
        parser.add_argument(
            "--output-slices",
            required=output_slices == "required",
            type=str,
            help="Folder to save the motion corrected slices",
        )
    # simulate slices
    if simulate_slices:
        parser.add_argument(
            "--simulated-slices",
            required=simulate_slices == "required",
            type=str,
            help="Folder to save the simulated (extracted) slices from the reconstructed volume",
        )
    # output model
    if output_model:
        parser.add_argument(
            "--output-model",
            type=str,
            required=output_model == "required",
            help="Path to save the output model (.pt)",
        )
    # output seg masks
    if output_stack_masks:
        parser.add_argument(
            "--output-stack-masks",
            type=str,
            nargs="+",
            required=output_stack_masks == "required",
            help="Path to the output folder or list of pathes to the output masks",
        )
    # output bias field correction results
    if output_corrected_stacks:
        parser.add_argument(
            "--output-corrected-stacks",
            type=str,
            nargs="+",
            required=output_corrected_stacks == "required",
            help="Path to the output folder or list of pathes to the output corrected stacks",
        )
    # json
    if output_json:
        parser.add_argument(
            "--output-json",
            type=str,
            help="Path to a json file for saving the inputs and results of the command.",
        )
    # output 3D seg results
    if output_folder:
        parser.add_argument(
            "--output-folder",
            type=str,
            required=output_folder == "required",
            help="Path to save outputs.",
        )

    update_defaults(_parser, **kwargs)
    return _parser


def build_parser_outputs_sampling(
    *,
    output_volume: Union[bool, str] = False,
    simulate_slices: Union[bool, str] = False,
    use_model: bool = True,
    **kwargs,
) -> argparse.ArgumentParser:
    """arguments related to ouptuts sampling"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("outputs sampling")
    if output_volume:
        parser.add_argument(
            "--output-resolution",
            default=0.8,
            type=float,
            help="Isotropic resolution of the reconstructed volume",
        )
        parser.add_argument(
            "--output-intensity-mean",
            default=700.0,
            type=float,
            help="mean intensity of the output volume",
        )
        if use_model:
            parser.add_argument(
                "--inference-batch-size", type=int, help="batch size for inference"
            )
            parser.add_argument(
                "--n-inference-samples",
                type=int,
                help="number of sample for PSF during inference",
            )
            parser.add_argument(
                "--output-psf-factor",
                type=float,
                default=1.0,
                help="Determind the psf for generating output volume: FWHM = output-resolution * output-psf-factor",
            )
        parser.add_argument(
            "--sample-orientation",
            type=str,
            help="Path to a nii file. The sampled volume will be reoriented according to the transformatio in this file.",
        )
    if output_volume or simulate_slices:
        parser.add_argument(
            "--sample-mask",
            type=str,
            help="3D Mask for sampling INR. If not provided, will use a mask esitmated from the input data.",
        )
    update_defaults(_parser, **kwargs)
    return _parser


def build_parser_svort() -> argparse.ArgumentParser:
    """arguments related to rigid registration before reconstruction"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("rigid registration")
    parser.add_argument(
        "--registration",
        default="svort",
        type=str,
        choices=["svort", "svort-only", "svort-stack", "stack", "none"],
        help=rst(
            "The type of registration method applied before reconstruction. \n\n"
            "#. ``svort``: try SVoRT and stack-to-stack registration and choose the one with better NCC; \n"
            "#. ``svort-only``: only apply the SVoRT model; \n"
            "#. ``svort-stack``: only apply the stack transformations of SVoRT; \n"
            "#. ``stack``: stack-to-stack rigid registration; \n"
            "#. ``none``: no rigid registration. \n\n"
            "**Note**: The SVoRT model can be only applied to fetal brain data. \n"
        ),
    )
    parser.add_argument(
        "--svort-version",
        default="v2",
        type=str,
        choices=["v1", "v2"],
        help="Version of SVoRT model",
    )
    parser.add_argument(
        "--scanner-space",
        action="store_true",
        help=(
            "Perform registration in the scanner space. "
            "Default: register the data to the atlas space when svort or svort-stack are used."
        ),
    )
    return _parser


def build_parser_segmentation(optional: bool = False) -> argparse.ArgumentParser:
    """arguments related to 2D brain segmentaion/masking"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("fetal brain masking")
    if optional:
        parser.add_argument(
            "--segmentation",
            action="store_true",
            help="Perform 2D fetal brain segmentation/masking for each input stack.",
        )
    parser.add_argument(
        "--batch-size-seg",
        type=int,
        default=16,
        help="Batch size for segmentation",
    )
    parser.add_argument(
        "--no-augmentation-seg",
        action="store_true",
        help="Disable inference data augmentation in segmentation",
    )
    parser.add_argument(
        "--dilation-radius-seg",
        type=float,
        default=1.0,
        help="Dilation radius for segmentation mask in millimeter.",
    )
    parser.add_argument(
        "--threshold-small-seg",
        type=float,
        default=0.1,
        help=(
            "Threshold for removing small segmetation mask (between 0 and 1). "
            "A mask will be removed if its area < threshold * max area in the stack."
        ),
    )
    return _parser


def build_parser_bias_field_correction(
    optional: bool = False,
) -> argparse.ArgumentParser:
    """arguments related to N4 bias field correction"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("N4 bias field correction")
    if optional:
        parser.add_argument(
            "--bias-field-correction",
            action="store_true",
            help="Perform bias field correction using the N4 algorithm.",
        )
    parser.add_argument(
        "--n-proc-n4",
        type=int,
        default=8,
        help="number of workers for the N4 algorithm.",
    )
    parser.add_argument(
        "--shrink-factor-n4",
        type=int,
        default=2,
        help="The shrink factor used to reduce the size and complexity of the image.",
    )
    parser.add_argument(
        "--tol-n4",
        type=float,
        default=0.001,
        help="The convergence threshold in N4.",
    )
    parser.add_argument(
        "--spline-order-n4",
        type=int,
        default=3,
        help="The order of B-spline.",
    )
    parser.add_argument(
        "--noise-n4",
        type=float,
        default=0.01,
        help="The noise estimate defining the Wiener filter.",
    )
    parser.add_argument(
        "--n-iter-n4",
        type=int,
        default=50,
        help="The maximum number of iterations specified at each fitting level.",
    )
    parser.add_argument(
        "--n-levels-n4",
        type=int,
        default=4,
        help="The maximum number of iterations specified at each fitting level.",
    )
    parser.add_argument(
        "--n-control-points-n4",
        type=int,
        default=4,
        help=(
            "The control point grid size in each dimension. "
            "The B-spline mesh size is equal to the number of control points in that dimension minus the spline order."
        ),
    )
    parser.add_argument(
        "--n-bins-n4",
        type=int,
        default=200,
        help="The number of bins in the log input intensity histogram.",
    )
    return _parser


def build_parser_assessment(**kwargs) -> argparse.ArgumentParser:
    """arguments related to image quality and motion assessment of input data"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("stack assessment")
    parser.add_argument(
        "--metric",
        type=str,
        choices=["ncc", "matrix-rank", "volume", "iqa2d", "iqa3d", "none"],
        default="none",
        help=rst(
            "Metric for assessing input stacks. \n\n"
            "1. ``ncc`` (\u2191): cross correlaiton between adjacent slices; \n"
            "2. ``matrix-rank`` (\u2193): motion metric based on the rank of the data matrix; \n"
            "3. ``volume`` (\u2191): volume of the masked ROI; \n"
            "4. ``iqa2d`` (\u2191): image quality score generated by a `2D CNN <https://arxiv.org/abs/2006.12704>`_ (only for fetal brain), the score of a stack is the average score of the images in it; \n"
            "5. ``iqa3d`` (\u2191): image quality score generated by a `3D CNN <https://github.com/FNNDSC/pl-fetal-brain-assessment>`_ (only for fetal brain); \n"
            "6. ``none``: no metric used for assessment. \n\n"
            "**Note**: (\u2191) means a stack with a higher score will have a better rank.\n"
        ),
    )
    parser.add_argument(
        "--filter-method",
        type=str,
        choices=["top", "bottom", "threshold", "percentage", "none"],
        default="none",
        help=rst(
            "Method to remove low-quality stacks. \n\n"
            "1. ``top``: keep the top ``C`` stacks; \n"
            "2. ``bottom``: remove the bottom ``C`` stacks; \n"
            "3. ``threshold``: remove a stack if the metric is worse than ``C``; \n"
            "4. ``percentatge``: remove the bottom (``num_stack * C``) stacks; \n"
            "5. ``none``: no filtering. \n\n"
            "The value of ``C`` is specified by `--cutoff <#cutoff>`__. \n"
        ),
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        help=rst(
            "The cutoff value for filtering, i.e., the value ``C`` in `--filter-method <#filter-method>`__"
        ),
    )
    parser.add_argument(
        "--batch-size-assess", type=int, default=8, help="Batch size for IQA network"
    )
    parser.add_argument(
        "--no-augmentation-assess",
        action="store_true",
        help="Disable inference data augmentation in IQA network",
    )
    update_defaults(_parser, **kwargs)
    return _parser


def build_parser_volume_segmentation() -> argparse.ArgumentParser:
    """arguments related to 3D brain segmentation"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("TWAI brain segmentation")
    parser.add_argument(
        "--ga",
        type=float,
        help=(
            "Gestational age at the time of acquisition of the fetal brain 3D MRI to be segmented."
            "If not provided, will be estimated based on the volume of the brain ROI."
        ),
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["Neurotypical", "Spina Bifida", "Pathological"],
        default="Neurotypical",
        help="Brain condition of the fetal brain 3D MRI to be segmented.",
    )
    parser.add_argument(
        "--bias-field-correction",
        action="store_true",
        help="Perform bias field correction before segmentation.",
    )
    return _parser


def build_parser_svr() -> argparse.ArgumentParser:
    """arguments related to SVR"""
    _parser = argparse.ArgumentParser(add_help=False)
    # outlier removal
    parser = _parser.add_argument_group("outlier removal")
    parser.add_argument(
        "--no-slice-robust-statistics",
        action="store_true",
        help="Disable slice-level robust statistics for outlier removal.",
    )
    parser.add_argument(
        "--no-pixel-robust-statistics",
        action="store_true",
        help="Disable pixel-level robust statistics for outlier removal.",
    )
    parser.add_argument(
        "--no-local-exclusion ",
        action="store_true",
        help="Disable pixel-level exclusion based on SSIM.",
    )
    parser.add_argument(
        "--no-global-exclusion",
        action="store_true",
        help="Disable slice-level exclusion based on NCC.",
    )
    parser.add_argument(
        "--global-ncc-threshold",
        type=float,
        default=0.5,
        help="Threshold for global exclusion.",
    )
    parser.add_argument(
        "--local-ssim-threshold",
        type=float,
        default=0.4,
        help="Threshold for local exclusion.",
    )
    # optimization
    parser = _parser.add_argument_group("optimization")
    parser.add_argument(
        "--with-background",
        action="store_true",
        help="Reconstruct the background in the volume.",
    )
    parser.add_argument(
        "--n-iter", type=int, default=3, help="Number of iterations (outer loop)"
    )
    parser.add_argument(
        "--n-iter-rec",
        type=int,
        nargs="+",
        default=[7],
        help=(
            "Number of iterations of super-resolution reconstruction (inner loop). "
            "Should be a list of int with length = n-iter. "
            "If a single number N is provided, will use [N, N, ..., N*3]. "
        ),
    )
    parser.add_argument(
        "--psf",
        type=str,
        default="gaussian",
        choices=["gaussian", "sinc"],
        help="Type of point spread function (PSF) used in data acquisition.",
    )
    # regularization
    parser = _parser.add_argument_group("regularization")
    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,  # 150.0 / 700.0,
        help="Parameter to define intensity of an edge in edge-preserving regularization. ",
    )

    return _parser


def build_parser_common() -> argparse.ArgumentParser:
    """miscellaneous arguments"""
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("miscellaneous")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help=(
            "Id of the device to use."
            "Use GPU if it is nonnegative and use CPU if it is negative."
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Level of verbosity: (0: warning/error, 1: info, 2: debug)",
    )
    parser.add_argument("--output-log", type=str, help="Path to the output log file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    return _parser


def update_defaults(parser: argparse.ArgumentParser, **kwargs):
    # a helper function to update the default values in a parser
    parser.set_defaults(**kwargs)


# command parsers


def add_subcommand(
    subparsers: argparse._SubParsersAction,
    name: str,
    help: Optional[str],
    description: Optional[str],
    parents: Sequence,
) -> argparse.ArgumentParser:
    # a helper function to create a subcommand
    parser = subparsers.add_parser(
        name=name,
        help=help,
        description=description,
        parents=parents,
        formatter_class=CommandHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help=argparse.SUPPRESS)
    return parser


def build_command_reconstruct(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # reconstruct
    parser_reconstruct = add_subcommand(
        subparsers,
        name="reconstruct",
        help="slice-to-volume reconstruction using NeSVoR",
        description=(
            "Use the NeSVoR algorithm to reconstuct a high-quality and coherent 3D volume from multiple stacks of 2D slices. "
            "This command can be applied to both rigid (e.g., brain) and non-rigid (e.g. uterus) motion. "
            "It also includes several optional preprocessing stpes: \n\n"
            "1. ROI masking/segmentation from each input stack with a CNN (only for fetal brain); \n"
            "2. N4 bias filed correction for each stack; \n"
            "3. Quality and motion assessment of each stack, which can be used to rank and filter the data; \n"
            "4. Motion correction with SVoRT (only for fetal brain) or stack-to-stack registration. \n"
        ),
        parents=[
            build_parser_inputs(input_stacks=True, input_slices=True),
            build_parser_stack_masking(),
            build_parser_outputs(
                output_volume=True,
                output_slices=True,
                simulate_slices=True,
                output_model=True,
            ),
            build_parser_outputs_sampling(output_volume=True, simulate_slices=True),
            build_parser_segmentation(optional=True),
            build_parser_bias_field_correction(optional=True),
            build_parser_assessment(),
            build_parser_svort(),
            build_parser_training(),
            build_parser_common(),
        ],
    )
    return parser_reconstruct


def build_command_sample_volume(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # sample-volume
    parser_sample_volume = add_subcommand(
        subparsers,
        name="sample-volume",
        help="sample a volume from a trained NeSVoR model",
        description="Sample a volume from a trained NeSVoR model. ",
        parents=[
            build_parser_inputs(input_model="required"),
            build_parser_outputs(
                output_volume="required",
            ),
            build_parser_outputs_sampling(
                output_volume=True,
                inference_batch_size=1024 * 4 * 8,
                n_inference_samples=128 * 2 * 2,
            ),
            build_parser_common(),
        ],
    )
    return parser_sample_volume


def build_command_sample_slices(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # sample-slices
    parser_sample_slices = add_subcommand(
        subparsers,
        name="sample-slices",
        help="sample slices from a trained NeSVoR model",
        description="Sample slices from a trained NeSVoR model. ",
        parents=[
            build_parser_inputs(input_slices="required", input_model="required"),
            build_parser_outputs(simulate_slices="required"),
            build_parser_outputs_sampling(simulate_slices=True),
            build_parser_common(),
        ],
    )
    return parser_sample_slices


def build_command_register(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # register
    parser_register = add_subcommand(
        subparsers,
        name="register",
        help="(rigid) motion correction with SVoRT or stack-to-stack registration",
        description="Perform inital (rigid) motion correction using SVoRT (only for fetal brain) or stack-to-stack registration.",
        parents=[
            build_parser_inputs(input_stacks="required"),
            build_parser_stack_masking(),
            build_parser_outputs(output_slices="required"),
            build_parser_svort(),
            build_parser_common(),
        ],
    )
    return parser_register


def build_command_segment_stack(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # segment-stack
    parser_segment_stack = add_subcommand(
        subparsers,
        name="segment-stack",
        help="2D fetal brain masking of input stacks",
        description=(
            "Segment the fetal brain ROI from each stack using a CNN model (MONAIfbs). \n"
            f"Check out {show_link('the original repo',  'https://github.com/gift-surg/MONAIfbs')} for details.\n"
        ),
        parents=[
            build_parser_inputs(input_stacks="required", no_thickness=True),
            build_parser_stack_masking(),
            build_parser_outputs(output_stack_masks="required"),
            build_parser_segmentation(optional=False),
            build_parser_common(),
        ],
    )
    return parser_segment_stack


def build_command_correct_bias_field(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # correct-bias-field
    parser_correct_bias_field = add_subcommand(
        subparsers,
        name="correct-bias-field",
        help="bias field correction using the N4 algorithm",
        description="Perform bias field correction for each input stack with the N4 algorithm.",
        parents=[
            build_parser_inputs(input_stacks="required", no_thickness=True),
            build_parser_stack_masking(),
            build_parser_outputs(output_corrected_stacks="required"),
            build_parser_bias_field_correction(optional=False),
            build_parser_common(),
        ],
    )
    return parser_correct_bias_field


def build_command_assess(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # assess
    parser_assess = add_subcommand(
        subparsers,
        name="assess",
        help="quality assessment of input stacks",
        description=(
            "Assess the quality and motion of each input stack. "
            "The output metrics can be used for determining the template stack or removing low-quality data"
        ),
        parents=[
            build_parser_inputs(input_stacks="required", no_thickness=True),
            build_parser_stack_masking(),
            build_parser_outputs(),
            build_parser_assessment(metric="ncc"),
            build_parser_common(),
        ],
    )
    return parser_assess


def build_command_segment_volume(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # segment-volume
    parser_segment_volume = add_subcommand(
        subparsers,
        name="segment-volume",
        help="3D fetal brain segmentation",
        description=(
            "TWAI brain segmentation of reconstructed 3D volume. Segmentation labels: \n\n"
            "1. white matter (excluding corpus callosum); \n"
            "2. intra-axial cerebrospinal fluid (CSF); \n"
            "3. cerebellum; \n"
            "4. extra-axial CSF; \n"
            "5. cortical gray matter; \n"
            "6. deep gray matter; \n"
            "7. brainstem; \n"
            "8. corpus callosum. \n\n"
            f"Check out {show_link('the original repo',  'https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation')} for details.\n"
        ),
        parents=[
            build_parser_inputs(input_volume="required"),
            build_parser_volume_segmentation(),
            build_parser_outputs(output_json=False, output_folder="required"),
            build_parser_common(),
        ],
    )
    return parser_segment_volume


def build_command_svr(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    # svr
    parser_svr = add_subcommand(
        subparsers,
        name="svr",
        help="classical slice-to-volume registration/reconstruction",
        description=(
            "This command implements a classical slice-to-volume registration/reconstruction pipeline with a robust motion correction method (SVoRT). "
            "It can only be applied to data with rigid motion (e.g., brain). "
        ),
        parents=[
            build_parser_inputs(input_stacks=True, input_slices=True),
            build_parser_stack_masking(),
            build_parser_outputs(
                output_volume="required",
                output_slices=True,
                simulate_slices=True,
            ),
            build_parser_outputs_sampling(output_volume=True, simulate_slices=True),
            build_parser_segmentation(optional=True),
            build_parser_bias_field_correction(optional=True),
            build_parser_assessment(),
            build_parser_svort(),
            build_parser_svr(),
            build_parser_common(),
        ],
    )
    return parser_svr


def main_parser(
    title="commands", metavar="COMMAND", dest="command"
) -> Tuple[argparse.ArgumentParser, argparse._SubParsersAction]:
    # main parser
    parser = argparse.ArgumentParser(
        prog="nesvor",
        description="NeSVoR: a toolkit for neural slice-to-volume reconstruction"
        if not_doc()
        else (
            "`nesvor <#nesvor>`__ has a range of subcommands to perform preprocessing, slice-to-volume reconstruction, and analysis."
            " See the corresponding pages for details. "
        ),
        epilog="Run 'nesvor COMMAND --help' for more information on a command.\n\n"
        + "To learn more about NeSVoR, check out our repo at "
        + __url__,
        formatter_class=MainHelpFormatter,
    )
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s v" + __version__
    )
    # commands
    subparsers = parser.add_subparsers(title=title, metavar=metavar, dest=dest)
    build_command_reconstruct(subparsers)
    build_command_sample_volume(subparsers)
    build_command_sample_slices(subparsers)
    build_command_register(subparsers)
    build_command_segment_stack(subparsers)
    build_command_correct_bias_field(subparsers)
    build_command_assess(subparsers)
    # build_command_segment_volume(subparsers)
    build_command_svr(subparsers)
    return parser, subparsers


def get_parser_for_sphinx() -> argparse.ArgumentParser:
    """This function is used for docs."""
    doc_mode()
    parser, _ = main_parser(title="Subcommands", metavar="command", dest="command")
    parser = prepare_parser_for_sphinx(parser)
    return parser
