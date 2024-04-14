import os
import argparse
from pathlib import Path
from hmr2.models import download_models
from scripts.pretrained_models import (
    DETECTRON2_MODEL_DIR,
    DETECTRON2_MODEL_PATH,
    HMR2_MODELS_DIR,
    SMPL_MODEL_DIR,
    SMPL_MODEL_PATH,
)

from utils.download import download


def download_hmr2_models():
    if not os.path.exists(HMR2_MODELS_DIR):
        os.makedirs(HMR2_MODELS_DIR, exist_ok=True)
    download_models(HMR2_MODELS_DIR)


def download_smpl_model():
    if not os.path.exists(SMPL_MODEL_DIR):
        os.makedirs(SMPL_MODEL_DIR, exist_ok=True)
    print(
        f"Please download smpl model from https://smplify.is.tue.mpg.de/, and place it in {SMPL_MODEL_PATH}"
    )


def download_detectron2_model():
    if not os.path.exists(DETECTRON2_MODEL_DIR):
        os.makedirs(DETECTRON2_MODEL_DIR, exist_ok=True)
    download(
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl",
        output=Path(DETECTRON2_MODEL_PATH),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model downloader")
    parser.add_argument("--all", action="store_true", help="download all models")
    parser.add_argument("--hmr2", action="store_true", help="download hmr2 models only")
    parser.add_argument("--smpl", action="store_true", help="download smpl models only")
    parser.add_argument(
        "--detectron2", action="store_true", help="download detectron2 models only"
    )

    args = parser.parse_args()

    if args.hmr2 or args.all:
        download_hmr2_models()
    if args.detectron2 or args.all:
        download_detectron2_model()
    if args.smpl or args.all:
        download_smpl_model()