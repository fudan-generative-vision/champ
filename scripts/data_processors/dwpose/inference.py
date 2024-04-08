import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import sys

from utils.fs import traverse_folder

project_root = Path(__file__).parent

sys.path.append(str(project_root))

from . import DWposeDetector

# Process dwpose for images
# /path/to/image_dataset/*/*.jpg -> /path/to/image_dataset_dwpose/*/*.jpg


def process_single_image(image_path, detector):
    img_name = Path(image_path).name
    root_dir = Path(image_path).parent.parent
    save_dir = root_dir.joinpath("dwpose")
    out_path = save_dir.joinpath(img_name)

    # out_path = output_dir.joinpath(img_name)
    if os.path.exists(out_path):
        return

    output_dir = Path(out_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_pil = Image.open(image_path)
    result, score = detector(frame_pil, image_resolution=min(*(frame_pil.size)))
    score = np.mean(score, axis=-1)

    result.save(out_path)
    print(f"save to {out_path}")


def process_batch_images(image_list, detector):
    for i, image_path in enumerate(image_list):
        print(f"Process {i + 1}/{len(image_list)} image")
        process_single_image(image_path, detector)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgs_path",
        type=str,
        default="",
        help="image file path or folder include images",
    )
    # parser.add_argument(
    #     "--output", type=str, default="./dwpose", help="Specify output directory"
    # )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--cpu", type=bool, default=False, help="Use CPU")
    args = parser.parse_args()
    image_paths = []

    imgs_path = Path(os.path.join(args.input, "normal"))
    if os.path.isdir(imgs_path):
        for file_path in traverse_folder(imgs_path):
            if file_path.is_file() and file_path.suffix in [".jpg", ".png", ".jpeg"]:
                image_paths.append(file_path)
    elif imgs_path.suffix in [".jpg", ".png", ".jpeg"]:
        image_paths.append(imgs_path)
    else:
        raise ValueError(
            f"--imgs_path need a image file path or a folder include images"
        )

    gpu_id = args.gpu
    cpu_enabled = args.cpu
    detector = DWposeDetector()
    if cpu_enabled:
        detector = detector.to(f"cpu")
    else:
        detector = detector.to(f"cuda:{gpu_id}")

    process_batch_images(image_paths, detector)

    print("finished")
