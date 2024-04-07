import os
import argparse
from pathlib import Path
from hmr2.models import download_models
from scripts.pretrained_models import (
    DETECTRON2_MODEL_DIR,
    DETECTRON2_MODEL_PATH,
    HMR2_MODELS_DIR,
    PRETRAIN_MODELS_DIR,
    SMPL_MODEL_DIR,
    SMPL_MODEL_PATH,
)

from utils.download import download


def download_hmr2_models():
    if not os.path.exists(HMR2_MODELS_DIR):
        os.makedirs(HMR2_MODELS_DIR)
    download_models(HMR2_MODELS_DIR)


def download_smpl_model():
    if not os.path.exists(SMPL_MODEL_DIR):
        os.makedirs(SMPL_MODEL_DIR)
    print(
        f"Please download smpl model from https://smplify.is.tue.mpg.de/, and place it in {SMPL_MODEL_PATH}"
    )


def download_detectron2_model():
    if not os.path.exists(DETECTRON2_MODEL_DIR):
        os.makedirs(DETECTRON2_MODEL_DIR)
    download(
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl",
        output=Path(DETECTRON2_MODEL_PATH),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model downloader")
    parser.add_argument("--all", type=bool, help="download all models")
    parser.add_argument("--hmr2", type=bool, help="download hmr2 models only")
    parser.add_argument("--smpl", type=bool, help="download smpl models only")
    parser.add_argument(
        "--detectron2", type=bool, help="download detectron2 models only"
    )

    args = parser.parse_args()

    if args.hmr2:
        download_hmr2_models()
    if args.detectron2:
        download_detectron2_model()
    if args.smpl:
        download_smpl_model()
    if args.all:
        download_hmr2_models()
        download_smpl_model()
        download_detectron2_model()
