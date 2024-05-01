import os
import json
import random
from typing import List
import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from tqdm import tqdm


def process_bbox(bbox, H, W, scale=1.):
    # transform a bbox(xmin, ymin, xmax, ymax) to (H, W) square
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    
    side_length = max(width, height)
    
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    scaled_side_length = side_length * scale
    scaled_xmin = center_x - scaled_side_length / 2
    scaled_xmax = center_x + scaled_side_length / 2
    scaled_ymin = center_y - scaled_side_length / 2
    scaled_ymax = center_y + scaled_side_length / 2

    scaled_xmin = int(max(0, scaled_xmin))
    scaled_xmax = int(min(W, scaled_xmax))
    scaled_ymin = int(max(0, scaled_ymin))
    scaled_ymax = int(min(H, scaled_ymax))
    
    return scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax

def crop_bbox(img, bbox, do_resize=False, size=512):
    
    if isinstance(img, (Path, str)):
        img = Image.open(img)
    cropped_img = img.crop(bbox)
    if do_resize:
        cropped_W, cropped_H = cropped_img.size
        ratio = size / max(cropped_W, cropped_H)
        new_W = cropped_W * ratio
        new_H = cropped_H * ratio
        cropped_img = cropped_img.resize((new_W, new_H))
    
    return cropped_img

def mask_to_bbox(mask_path):
    mask = np.array(Image.open(mask_path))[..., 0]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax

def mask_to_bkgd(img_path, mask_path):
    img = Image.open(img_path)
    img_array = np.array(img)
    
    mask = Image.open(mask_path).convert("RGB")
    mask_array = np.array(mask)
    
    img_array = np.where(mask_array > 0, img_array, 0)
    return Image.fromarray(img_array)
    