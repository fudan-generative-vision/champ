import os
from pathlib import Path
from hmr2.models import CACHE_DIR_4DHUMANS

PROJECT_ROOT = Path(__file__).parent.parent.parent;

PRETRAINED_MODEL_DIR = f"{PROJECT_ROOT}/pretrained_models";

HMR2_MODELS_DIR = CACHE_DIR_4DHUMANS
HMR2_DEFAULT_CKPT = f"{HMR2_MODELS_DIR}/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"

SMPL_MODEL_DIR = f"{CACHE_DIR_4DHUMANS}/data/smpl"
SMPL_MODEL_PATH = f"{SMPL_MODEL_DIR}/SMPL_NEUTRAL.pkl"

DETECTRON2_MODEL_DIR = f"{PRETRAINED_MODEL_DIR}/detectron2"
DETECTRON2_MODEL_PATH = f"{DETECTRON2_MODEL_DIR}/model_final_f05665.pkl"