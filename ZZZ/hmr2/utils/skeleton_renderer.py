import torch
import numpy as np
import trimesh
from typing import Optional
from yacs.config import CfgNode

from .geometry import perspective_projection
from .render_openpose import render_openpose

class SkeletonRenderer:

    def __init__(self, cfg: CfgNode):
        """
        Object used to render 3D keypoints. Faster for use during training.
        Args:
            cfg (CfgNode): Model config file.
        """
        self.cfg = cfg

    def __call__(self,
                 pred_keypoints_3d: torch.Tensor,
                 gt_keypoints_3d: torch.Tensor,
                 gt_keypoints_2d: torch.Tensor,
                 images: Optional[np.array] = None,
                 camera_translation: Optional[torch.Tensor] = None) -> np.array:
        """
        Render batch of 3D keypoints.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape (B, S, N, 3) containing a batch of predicted 3D keypoints, with S samples per image.
            gt_keypoints_3d (torch.Tensor): Tensor of shape (B, N, 4) containing corresponding ground truth 3D keypoints; last value is the confidence.
            gt_keypoints_2d (torch.Tensor): Tensor of shape (B, N, 3) containing corresponding ground truth 2D keypoints.
            images (torch.Tensor): Tensor of shape (B, H, W, 3) containing images with values in the [0,255] range.
            camera_translation (torch.Tensor): Tensor of shape (B, 3) containing the camera translation.
        Returns:
            np.array : Image with the following layout. Each row contains the a) input image,
                                                                              b) image with gt 2D keypoints,
                                                                              c) image with projected gt 3D keypoints,
                                                                              d_1, ... , d_S) image with projected predicted 3D keypoints,
                                                                              e) gt 3D keypoints rendered from a side view,
                                                                              f_1, ... , f_S) predicted 3D keypoints frorm a side view
        """
        batch_size = pred_keypoints_3d.shape[0]
#        num_samples = pred_keypoints_3d.shape[1]
        pred_keypoints_3d = pred_keypoints_3d.clone().cpu().float()
        gt_keypoints_3d = gt_keypoints_3d.clone().cpu().float()
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, [25+14], :-1] + pred_keypoints_3d[:, [25+14]]
        gt_keypoints_2d = gt_keypoints_2d.clone().cpu().float().numpy()
        gt_keypoints_2d[:, :, :-1] = self.cfg.MODEL.IMAGE_SIZE * (gt_keypoints_2d[:, :, :-1] + 1.0) / 2.0

        openpose_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        gt_indices = [12, 8, 7, 6, 9, 10, 11, 14, 2, 1, 0, 3, 4, 5]
        gt_indices = [25 + i for i in gt_indices]
        keypoints_to_render = torch.ones(batch_size, gt_keypoints_3d.shape[1], 1)
        rotation = torch.eye(3).unsqueeze(0)
        if camera_translation is None:
            camera_translation = torch.tensor([0.0, 0.0, 2 * self.cfg.EXTRA.FOCAL_LENGTH / (0.8 * self.cfg.MODEL.IMAGE_SIZE)]).unsqueeze(0).repeat(batch_size, 1)
        else:
            camera_translation = camera_translation.cpu()

        if images is None:
            images = np.zeros((batch_size, self.cfg.MODEL.IMAGE_SIZE, self.cfg.MODEL.IMAGE_SIZE, 3))
        focal_length = torch.tensor([self.cfg.EXTRA.FOCAL_LENGTH, self.cfg.EXTRA.FOCAL_LENGTH]).reshape(1, 2)
        camera_center = torch.tensor([self.cfg.MODEL.IMAGE_SIZE, self.cfg.MODEL.IMAGE_SIZE], dtype=torch.float).reshape(1, 2) / 2.
        gt_keypoints_3d_proj = perspective_projection(gt_keypoints_3d[:, :, :-1], rotation=rotation.repeat(batch_size, 1, 1), translation=camera_translation[:, :], focal_length=focal_length.repeat(batch_size, 1), camera_center=camera_center.repeat(batch_size, 1))
        pred_keypoints_3d_proj = perspective_projection(pred_keypoints_3d.reshape(batch_size, -1, 3), rotation=rotation.repeat(batch_size, 1, 1), translation=camera_translation.reshape(batch_size, -1), focal_length=focal_length.repeat(batch_size, 1), camera_center=camera_center.repeat(batch_size, 1)).reshape(batch_size, -1, 2)
        gt_keypoints_3d_proj = torch.cat([gt_keypoints_3d_proj, gt_keypoints_3d[:, :, [-1]]], dim=-1).cpu().numpy()
        pred_keypoints_3d_proj = torch.cat([pred_keypoints_3d_proj, keypoints_to_render.reshape(batch_size, -1, 1)], dim=-1).cpu().numpy()
        rows = []
        # Rotate keypoints to visualize side view
        R = torch.tensor(trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])[:3, :3]).float()
        gt_keypoints_3d_side = gt_keypoints_3d.clone()
        gt_keypoints_3d_side[:, :, :-1] = torch.einsum('bni,ij->bnj', gt_keypoints_3d_side[:, :, :-1], R)
        pred_keypoints_3d_side = pred_keypoints_3d.clone()
        pred_keypoints_3d_side = torch.einsum('bni,ij->bnj', pred_keypoints_3d_side, R)
        gt_keypoints_3d_proj_side = perspective_projection(gt_keypoints_3d_side[:, :, :-1], rotation=rotation.repeat(batch_size, 1, 1), translation=camera_translation[:, :], focal_length=focal_length.repeat(batch_size, 1), camera_center=camera_center.repeat(batch_size, 1))
        pred_keypoints_3d_proj_side = perspective_projection(pred_keypoints_3d_side.reshape(batch_size, -1, 3), rotation=rotation.repeat(batch_size, 1, 1), translation=camera_translation.reshape(batch_size, -1), focal_length=focal_length.repeat(batch_size, 1), camera_center=camera_center.repeat(batch_size, 1)).reshape(batch_size, -1, 2)
        gt_keypoints_3d_proj_side = torch.cat([gt_keypoints_3d_proj_side, gt_keypoints_3d_side[:, :, [-1]]], dim=-1).cpu().numpy()
        pred_keypoints_3d_proj_side = torch.cat([pred_keypoints_3d_proj_side, keypoints_to_render.reshape(batch_size, -1, 1)], dim=-1).cpu().numpy()
        for i in range(batch_size):
            img = images[i]
            side_img = np.zeros((self.cfg.MODEL.IMAGE_SIZE, self.cfg.MODEL.IMAGE_SIZE, 3))
            # gt 2D keypoints
            body_keypoints_2d = gt_keypoints_2d[i, :25].copy()
            for op, gt in zip(openpose_indices, gt_indices):
                if gt_keypoints_2d[i, gt, -1] > body_keypoints_2d[op, -1]:
                    body_keypoints_2d[op] = gt_keypoints_2d[i, gt]
            gt_keypoints_img = render_openpose(img, body_keypoints_2d) / 255.
            # gt 3D keypoints
            body_keypoints_3d_proj = gt_keypoints_3d_proj[i, :25].copy()
            for op, gt in zip(openpose_indices, gt_indices):
                if gt_keypoints_3d_proj[i, gt, -1] > body_keypoints_3d_proj[op, -1]:
                    body_keypoints_3d_proj[op] = gt_keypoints_3d_proj[i, gt]
            gt_keypoints_3d_proj_img = render_openpose(img, body_keypoints_3d_proj) / 255.
            # gt 3D keypoints from the side
            body_keypoints_3d_proj = gt_keypoints_3d_proj_side[i, :25].copy()
            for op, gt in zip(openpose_indices, gt_indices):
                if gt_keypoints_3d_proj_side[i, gt, -1] > body_keypoints_3d_proj[op, -1]:
                    body_keypoints_3d_proj[op] = gt_keypoints_3d_proj_side[i, gt]
            gt_keypoints_3d_proj_img_side = render_openpose(side_img, body_keypoints_3d_proj) / 255.
            # pred 3D keypoints
            pred_keypoints_3d_proj_imgs = []
            body_keypoints_3d_proj = pred_keypoints_3d_proj[i, :25].copy()
            for op, gt in zip(openpose_indices, gt_indices):
                if pred_keypoints_3d_proj[i, gt, -1] >= body_keypoints_3d_proj[op, -1]:
                    body_keypoints_3d_proj[op] = pred_keypoints_3d_proj[i, gt]
            pred_keypoints_3d_proj_imgs.append(render_openpose(img, body_keypoints_3d_proj) / 255.)
            pred_keypoints_3d_proj_img = np.concatenate(pred_keypoints_3d_proj_imgs, axis=1)
            # gt 3D keypoints from the side
            pred_keypoints_3d_proj_imgs_side = []
            body_keypoints_3d_proj = pred_keypoints_3d_proj_side[i, :25].copy()
            for op, gt in zip(openpose_indices, gt_indices):
                if pred_keypoints_3d_proj_side[i, gt, -1] >= body_keypoints_3d_proj[op, -1]:
                    body_keypoints_3d_proj[op] = pred_keypoints_3d_proj_side[i, gt]
            pred_keypoints_3d_proj_imgs_side.append(render_openpose(side_img, body_keypoints_3d_proj) / 255.)
            pred_keypoints_3d_proj_img_side = np.concatenate(pred_keypoints_3d_proj_imgs_side, axis=1)
            rows.append(np.concatenate((gt_keypoints_img, gt_keypoints_3d_proj_img, pred_keypoints_3d_proj_img, gt_keypoints_3d_proj_img_side, pred_keypoints_3d_proj_img_side), axis=1))
        # Concatenate images
        img = np.concatenate(rows, axis=0)
        img[:, ::self.cfg.MODEL.IMAGE_SIZE, :] = 1.0
        img[::self.cfg.MODEL.IMAGE_SIZE, :, :] = 1.0
        img[:, (1+1+1)*self.cfg.MODEL.IMAGE_SIZE, :] = 0.5
        return img
