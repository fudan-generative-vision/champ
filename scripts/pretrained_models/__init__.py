import os
from pathlib import Path

PRETRAIN_MODELS_DIR = os.path.join(
    Path(__file__).parent.parent.parent, "pretrained_models"
)

HMR2_MODELS_DIR = f"{PRETRAIN_MODELS_DIR}/hmr2"
HMR2_DEFAULT_CKPT = f"{HMR2_MODELS_DIR}/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"

SMPL_MODEL_DIR = f"{PRETRAIN_MODELS_DIR}/smpl"
SMPL_MODEL_PATH = f"{SMPL_MODEL_DIR}/SMPL_NEUTRAL.pkl"

DETECTRON2_MODEL_DIR = f"{PRETRAIN_MODELS_DIR}/detectron2"
DETECTRON2_MODEL_PATH = f"{DETECTRON2_MODEL_DIR}/model_final_f05665.pkl"

DWPOSE_MODEL_DIR = f"{PRETRAIN_MODELS_DIR}/DWPose"
YOLO_L_MODEL_PATH = f"{DWPOSE_MODEL_DIR}/yolox_l.onnx"
DWPOSE_MODEL_PATH = f"{DWPOSE_MODEL_DIR}/dw-ll_ucoco_384.onnx"
