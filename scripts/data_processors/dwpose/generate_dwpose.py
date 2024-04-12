import cv2
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import sys
import argparse
from utils.fs import traverse_folder

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(os.path.join(project_root, "DWPose/ControlNet-v1-1-nightly"))
from annotator.dwpose import DWposeDetector

def process_single_image(image_path, detector, output_dir):
    img_name = Path(image_path).name

    out_path = output_dir.joinpath(img_name)
    if os.path.exists(out_path):
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_pil = Image.open(image_path)
    image = cv2.imread(str(image_path))
    result = detector(image)
    result = cv2.resize(result, dsize=frame_pil.size, interpolation=cv2.INTER_CUBIC)

    Image.fromarray(result).save(out_path)
    print(f"save to {out_path}")


def process_batch_images(image_list, detector, output_dir):
    for i, image_path in enumerate(image_list):
        print(f"Process {i + 1}/{len(image_list)} image: {image_path}")
        process_single_image(image_path, detector, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="", help="image file path or folder include images")
    parser.add_argument("--output", type=str, default="./dwpose", help="Specify output directory")

    args = parser.parse_args()
    image_paths = []

    imgs_path = args.input
    if os.path.isdir(imgs_path):
        for file_path in traverse_folder(imgs_path):
            if os.path.isfile(file_path) and str(file_path).endswith(
                (".jpg", ".png", ".jpeg")
            ):
                image_paths.append(file_path)
    elif imgs_path.suffix in [".jpg", ".png", ".jpeg"]:
        image_paths.append(imgs_path)
    else:
        raise ValueError(
            f"--input need a image file path or a folder include images"
        )
    detector = DWposeDetector()
    output_dir = Path(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    process_batch_images(image_paths, detector, output_dir)

    print("finished")