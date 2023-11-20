import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import os
import hydra
import torch
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger
from phalp.configs.base import CACHE_DIR

from hmr2.datasets.utils import expand_bbox_to_aspect_ratio

warnings.filterwarnings('ignore')

log = get_pylogger(__name__)

class HMR2Predictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from hmr2.models import download_models, load_hmr2

        # Download and load checkpoints
        download_models()
        model, _ = load_hmr2()

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out
    
class HMR2023TextureSampler(HMR2Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # Model's all set up. Now, load tex_bmap and tex_fmap
        # Texture map atlas
        bmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/bmap_256.npy'))
        fmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/fmap_256.npy'))
        self.register_buffer('tex_bmap', torch.tensor(bmap, dtype=torch.float))
        self.register_buffer('tex_fmap', torch.tensor(fmap, dtype=torch.long))

        self.img_size = 256         #self.cfg.MODEL.IMAGE_SIZE
        self.focal_length = 5000.   #self.cfg.EXTRA.FOCAL_LENGTH

        import neural_renderer as nr
        self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.img_size,
                                          image_size=self.img_size,
                                          light_intensity_ambient=1,
                                          light_intensity_directional=0,
                                          anti_aliasing=False)

    def forward(self, x):
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)

        # from hmr2.models.prohmr_texture import unproject_uvmap_to_mesh

        def unproject_uvmap_to_mesh(bmap, fmap, verts, faces):
            # bmap:  256,256,3
            # fmap:  256,256
            # verts: B,V,3
            # faces: F,3
            valid_mask = (fmap >= 0)

            fmap_flat = fmap[valid_mask]      # N
            bmap_flat = bmap[valid_mask,:]    # N,3

            face_vids = faces[fmap_flat, :]  # N,3
            face_verts = verts[:, face_vids, :] # B,N,3,3

            bs = face_verts.shape
            map_verts = torch.einsum('bnij,ni->bnj', face_verts, bmap_flat) # B,N,3

            return map_verts, valid_mask

        pred_verts = model_out['pred_vertices'] + model_out['pred_cam_t'].unsqueeze(1)
        device = pred_verts.device
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long, device=device)
        map_verts, valid_mask = unproject_uvmap_to_mesh(self.tex_bmap, self.tex_fmap, pred_verts, face_tensor) # B,N,3

        # Project map_verts to image using K,R,t
        # map_verts_view = einsum('bij,bnj->bni', R, map_verts) + t # R=I t=0
        focal = self.focal_length / (self.img_size / 2)
        map_verts_proj = focal * map_verts[:, :, :2] / map_verts[:, :, 2:3] # B,N,2
        map_verts_depth = map_verts[:, :, 2] # B,N

        # Render Depth. Annoying but we need to create this
        K = torch.eye(3, device=device)
        K[0, 0] = K[1, 1] = self.focal_length
        K[1, 2] = K[0, 2] = self.img_size / 2  # Because the neural renderer only support squared images
        K = K.unsqueeze(0)
        R = torch.eye(3, device=device).unsqueeze(0)
        t = torch.zeros(3, device=device).unsqueeze(0)
        rend_depth = self.neural_renderer(pred_verts,
                                        face_tensor[None].expand(pred_verts.shape[0], -1, -1).int(),
                                        # textures=texture_atlas_rgb,
                                        mode='depth',
                                        K=K, R=R, t=t)

        rend_depth_at_proj = torch.nn.functional.grid_sample(rend_depth[:,None,:,:], map_verts_proj[:,None,:,:]) # B,1,1,N
        rend_depth_at_proj = rend_depth_at_proj.squeeze(1).squeeze(1) # B,N

        img_rgba = torch.cat([batch['img'], batch['mask'][:,None,:,:]], dim=1) # B,4,H,W
        img_rgba_at_proj = torch.nn.functional.grid_sample(img_rgba, map_verts_proj[:,None,:,:]) # B,4,1,N
        img_rgba_at_proj = img_rgba_at_proj.squeeze(2) # B,4,N

        visibility_mask = map_verts_depth <= (rend_depth_at_proj + 1e-4) # B,N
        img_rgba_at_proj[:,3,:][~visibility_mask] = 0

        # Paste image back onto square uv_image
        uv_image = torch.zeros((batch['img'].shape[0], 4, 256, 256), dtype=torch.float, device=device)
        uv_image[:, :, valid_mask] = img_rgba_at_proj

        out = {
            'uv_image':  uv_image,
            'uv_vector' : self.hmar_old.process_uv_image(uv_image),
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam':  model_out['pred_cam'],
        }
        return out

class HMR2_4dhuman(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = HMR2023TextureSampler(self.cfg)

    def get_detections(self, image, frame_name, t_, additional_data=None, measurments=None):
        (
            pred_bbox, pred_bbox, pred_masks, pred_scores, pred_classes, 
            ground_truth_track_id, ground_truth_annotations
        ) =  super().get_detections(image, frame_name, t_, additional_data, measurments)

        # Pad bounding boxes 
        pred_bbox_padded = expand_bbox_to_aspect_ratio(pred_bbox, self.cfg.expand_bbox_shape)

        return (
            pred_bbox, pred_bbox_padded, pred_masks, pred_scores, pred_classes,
            ground_truth_track_id, ground_truth_annotations
        )
    

@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    expand_bbox_shape: Optional[Tuple[int]] = (192,256)
    pass

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function for running the PHALP tracker."""

    phalp_tracker = HMR2_4dhuman(cfg)

    phalp_tracker.track()

if __name__ == "__main__":
    main()
