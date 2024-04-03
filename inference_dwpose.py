import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

project_root = Path(__file__).parent

sys.path.append(str(project_root))

from dwpose import DWposeDetector

# Process dwpose for images
# /path/to/image_dataset/*/*.jpg -> /path/to/image_dataset_dwpose/*/*.jpg


def process_single_image(image_path, detector):
    # relative_path = os.path.relpath(image_path, root_dir)
    # print(f"Processing {relative_path}")
    img_name = Path(image_path).name
    # out_path = os.path.join(save_dir, relative_path)
    root_dir = Path(image_path).parent.parent
    save_dir = root_dir.joinpath("dwpose")
    out_path = save_dir.joinpath(img_name)
    if os.path.exists(out_path):
        return

    output_dir = Path(out_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_pil = Image.open(image_path)
    result, score = detector(frame_pil, image_resolution=min(*(frame_pil.size)))
    score = np.mean(score, axis=-1)
    # result = result[np.argmax(score)]

    result.save(out_path)
    print(f"save to {out_path}")
    # Assuming save_videos_from_pil can also handle single image saving
    # If not, replace this with appropriate PIL image save function
    # save_videos_from_pil([result], out_path, fps=None)  # fps is not required for images


def process_batch_images(image_list, detector):
    for i, image_path in enumerate(image_list):
        print(f"Process {i + 1}/{len(image_list)} image")
        process_single_image(image_path, detector)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_path", type=str, default="_test")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()
    image_paths = []

    subdir = Path(args.imgs_path)
    if subdir.is_dir():
        for cond_dir in subdir.iterdir():
            if cond_dir.name == "normal":
                for image_path in cond_dir.iterdir():
                    if image_path.is_file() and image_path.suffix in [
                        ".jpg",
                        ".png",
                        ".jpeg",
                    ]:
                        image_paths.append(image_path)

    gpu_id = args.device
    detector = DWposeDetector()
    detector = detector.to(f"cuda:{gpu_id}")
    process_batch_images(image_paths, detector)

    print("finished")
