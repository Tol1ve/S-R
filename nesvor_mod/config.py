import os

# path to this repo
BASE_DIR = os.path.dirname(__file__)

# path to save download checkpoints
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# URLs to download checkpoints of different models
SVORT_URL_DICT = {
    "v1": "https://zenodo.org/record/7486938/files/checkpoint.pt?download=1",
    "v2": "https://zenodo.org/record/7486938/files/checkpoint_v2.pt?download=1",
}
MONAIFBS_URL = "https://zenodo.org/record/4282679/files/models.tar.gz?download=1"
IQA2D_URL = "https://zenodo.org/record/7368570/files/pytorch.ckpt?download=1"
IQA3D_URL = "https://fnndsc.childrens.harvard.edu/mri_pipeline/ivan/quality_assessment/weights_resnet_sw2_k3.hdf5"

# path for TWAI, NiftyReg, and nnUNet
# By default, use the setting in https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation/tree/2cc5ed3976a23f698cad0e1ad36c0539ce524733
# exmaple steps for setting up the TWAI module:
# 1. clone https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation to TWAI_DIR
# 2. create folders NNUNet_RAW_DATA_DIR, NNUNet_PREPROCESSED_DIR, NNUNET_TRAINED_MODELs_DIR
# 3. create NNUNET_TRAINED_MODELs_DIR/nnUNet/3d_fullres, then download and extract pretrained weights to it.
# 4. cd TWAI_DIR/docker/third-party/nnUNet and run `pip install -e .`
TWAI_DIR = "/home/trustworthy-ai-fetal-brain-segmentation"
NIFTYREG_BIN_DIR = os.path.join(
    TWAI_DIR,
    "docker",
    "third-party",
    "niftyreg",
    # "build",
    # "reg-apps",
    "install",
    "bin",
)
NNUNet_RAW_DATA_DIR = os.path.join(
    TWAI_DIR, "docker", "third-party", "nnUNet_raw_data_base"
)
NNUNet_PREPROCESSED_DIR = os.path.join(
    TWAI_DIR, "docker", "third-party", "nnUNet_preprocessed"
)
NNUNET_TRAINED_MODELs_DIR = os.path.join(
    TWAI_DIR, "docker", "third-party", "nnUNet_trained_models"
)
