import os
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image


def save_videos_from_pil(pil_images, path, fps=24, crf=23):

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if save_fmt == ".mp4":
        with imageio.get_writer(path, fps=fps) as writer:
            for img in pil_images:
                img_array = np.array(img)  # Convert PIL Image to numpy array
                writer.append_data(img_array)

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=24):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def resize_tensor_frames(video_tensor, new_size):
    B, C, video_length, H, W = video_tensor.shape
    # Reshape video tensor to combine batch and frame dimensions: (B*F, C, H, W)
    video_tensor_reshaped = video_tensor.reshape(-1, C, H, W)
    # Resize using interpolate
    resized_frames = F.interpolate(
        video_tensor_reshaped, size=new_size, mode="bilinear", align_corners=False
    )
    resized_video = resized_frames.reshape(B, C, video_length, new_size[0], new_size[1])

    return resized_video


def pil_list_to_tensor(image_list, size=None):
    to_tensor = transforms.ToTensor()
    if size is not None:
        tensor_list = [to_tensor(img.resize(size[::-1])) for img in image_list]
    else:
        tensor_list = [to_tensor(img) for img in image_list]
    stacked_tensor = torch.stack(tensor_list, dim=0)
    tensor = stacked_tensor.permute(1, 0, 2, 3)
    return tensor
