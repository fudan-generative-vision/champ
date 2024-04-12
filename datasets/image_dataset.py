import json
import random
from typing import List
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from tqdm import tqdm
from datasets.data_utils import process_bbox, crop_bbox, mask_to_bbox, mask_to_bkgd


class ImageDataset(Dataset):
    def __init__(
        self,
        video_folder: str,
        image_size: int = 768,
        sample_margin: int = 30,
        data_parts: list = ["all"],
        guids: list = ["depth", "normal", "semantic_map", "dwpose"],
        extra_region: list = [],
        bbox_crop=True,
        bbox_resize_ratio=(0.8, 1.2),
        aug_type: str = "Resize",  # "Resize" or "Padding"
        select_face=False,
    ):
        super().__init__()
        self.video_folder = video_folder
        self.image_size = image_size
        self.sample_margin = sample_margin
        self.data_parts = data_parts
        self.guids = guids
        self.extra_region = extra_region
        self.bbox_crop = bbox_crop
        self.bbox_resize_ratio = bbox_resize_ratio
        self.aug_type = aug_type
        self.select_face = select_face
        
        self.data_lst = self.generate_data_lst()
        
        self.clip_image_processor = CLIPImageProcessor()
        self.pixel_transform, self.guid_transform = self.setup_transform()
            
    def generate_data_lst(self):
        video_folder = Path(self.video_folder)
        if "all" in self.data_parts:
            data_parts = sorted(video_folder.iterdir())
        else:
            data_parts = [(video_folder / p) for p in self.data_parts]
        data_lst = []
        for data_part in data_parts:
            for video_dir in sorted(data_part.iterdir()):
                if self.is_valid(video_dir):
                    data_lst += [video_dir]
        return data_lst
    
    def is_valid(self, video_dir: Path):
        video_length = len(list((video_dir / "images").iterdir()))
        for guid in self.guids:
            guid_length = len(list((video_dir / guid).iterdir()))
            if guid_length == 0 or guid_length != video_length:
                return False
        if self.select_face:
            if not (video_dir / "face_images").is_dir():
                return False
            else:
                face_img_length = len(list((video_dir / "face_images").iterdir()))
                if face_img_length == 0:
                    return False
        return True
    
    def resize_long_edge(self, img):
        img_W, img_H = img.size
        long_edge = max(img_W, img_H)
        scale = self.image_size / long_edge
        new_W, new_H = int(img_W * scale), int(img_H * scale)
        
        img = F.resize(img, (new_H, new_W))
        return img

    def padding_short_edge(self, img):
        img_W, img_H = img.size
        width, height = self.image_size, self.image_size
        padding_left = (width - img_W) // 2
        padding_right = width - img_W - padding_left
        padding_top = (height - img_H) // 2
        padding_bottom = height - img_H - padding_top
        
        img = F.pad(img, (padding_left, padding_top, padding_right, padding_bottom), 0, "constant")
        return img
    
    def setup_transform(self):
        if self.bbox_crop:
            if self.aug_type == "Resize":
                pixel_transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose = ([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                ])
                
            elif self.aug_type == "Padding":
                pixel_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                ])
            else:
                raise NotImplementedError("Do not support this augmentation")
        
        else:
            pixel_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            guid_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
            ])
        
        return pixel_transform, guid_transform            
            
    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.cat(transformed_images, dim=0)  # (c*n, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor
    
    def set_tgt_idx(self, ref_img_idx, video_length):
        margin = self.sample_margin
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)
            
        return tgt_img_idx
    
    def __len__(self):
        return len(self.data_lst)
    
    def __getitem__(self, idx):
        video_dir = self.data_lst[idx]
        
        # reference image
        img_dir = video_dir / "images"
        if self.select_face:
            face_img_dir = video_dir / "face_images"
            face_img_lst = [img.name for img in face_img_dir.glob("*.png")]
            ref_img_name = random.choice(face_img_lst)
        else:
            ref_img_name = random.choice([img.name for img in img_dir.glob("*.png")])
        ref_img_path = img_dir / ref_img_name
        ref_img_pil = Image.open(ref_img_path)
        
        img_path_lst = sorted([img.name for img in img_dir.glob("*.png")])
        ref_img_idx = img_path_lst.index(ref_img_name)
        
        # target image
        video_length = len(img_path_lst)
        tgt_img_idx = self.set_tgt_idx(ref_img_idx, video_length)
        tgt_img_name = img_path_lst[tgt_img_idx]
        tgt_img_pil = Image.open(img_dir / img_path_lst[tgt_img_idx])
        tgt_guid_pil_lst = []
        
        # guidance images
        for guid in self.guids:
            guid_img_path = video_dir / guid / tgt_img_name
            if guid == "semantic_map":
                mask_img_path = video_dir / "mask" / tgt_img_name
                guid_img_pil = mask_to_bkgd(guid_img_path, mask_img_path)
            else:
                guid_img_pil = Image.open(guid_img_path).convert("RGB")
            tgt_guid_pil_lst += [guid_img_pil]
            
        # bbox crop
        if self.bbox_crop:
            human_bbox_json_path = video_dir / "human_bbox.json"
            with open(human_bbox_json_path) as bbox_fp:
                human_bboxes = json.load(bbox_fp)
            resize_scale = random.uniform(*self.bbox_resize_ratio)
            ref_W, ref_H = ref_img_pil.size
            ref_bbox = process_bbox(human_bboxes[ref_img_idx], ref_H, ref_W, resize_scale)
            ref_img_pil = crop_bbox(ref_img_pil, ref_bbox)
            tgt_bbox = process_bbox(human_bboxes[tgt_img_idx], ref_H, ref_W, resize_scale)
            tgt_img_pil = crop_bbox(tgt_img_pil, tgt_bbox)
            tgt_guid_pil_lst = [crop_bbox(guid_pil, tgt_bbox) for guid_pil in tgt_guid_pil_lst]
        
        # augmentation
        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.pixel_transform, state)
        tgt_guid = self.augmentation(tgt_guid_pil_lst, self.guid_transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.pixel_transform, state)
        clip_img = self.clip_image_processor(
            images=ref_img_pil, return_tensor="pt"
        ).pixel_values[0]
        
        sample = dict(
            tgt_img=tgt_img,
            tgt_guid=tgt_guid,
            ref_img=ref_img_vae,
            clip_img=clip_img,
        )
        
        return sample
