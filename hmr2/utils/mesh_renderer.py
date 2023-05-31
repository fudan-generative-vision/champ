import os
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import cv2
import torch.nn.functional as F

from .render_openpose import render_openpose

def create_raymond_lights():
    import pyrender
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

class MeshRenderer:

    def __init__(self, cfg, faces=None):
        self.cfg = cfg
        self.focal_length = cfg.EXTRA.FOCAL_LENGTH
        self.img_res = cfg.MODEL.IMAGE_SIZE
        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.img_res,
                                       viewport_height=self.img_res,
                                       point_size=1.0)
        
        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces

    def visualize(self, vertices, camera_translation, images, focal_length=None, nrow=3, padding=2):
        images_np = np.transpose(images, (0,2,3,1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            fl = self.focal_length
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=False), (2,0,1))).float()
            rend_img_side = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=True), (2,0,1))).float()
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        return rend_imgs

    def visualize_tensorboard(self, vertices, camera_translation, images, pred_keypoints, gt_keypoints, focal_length=None, nrow=5, padding=2):
        images_np = np.transpose(images, (0,2,3,1))
        rend_imgs = []
        pred_keypoints = np.concatenate((pred_keypoints, np.ones_like(pred_keypoints)[:, :, [0]]), axis=-1)
        pred_keypoints = self.img_res * (pred_keypoints + 0.5)
        gt_keypoints[:, :, :-1] = self.img_res * (gt_keypoints[:, :, :-1] + 0.5)
        keypoint_matches = [(1, 12), (2, 8), (3, 7), (4, 6), (5, 9), (6, 10), (7, 11), (8, 14), (9, 2), (10, 1), (11, 0), (12, 3), (13, 4), (14, 5)]
        for i in range(vertices.shape[0]):
            fl = self.focal_length
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=False), (2,0,1))).float()
            rend_img_side = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=fl, side_view=True), (2,0,1))).float()
            body_keypoints = pred_keypoints[i, :25]
            extra_keypoints = pred_keypoints[i, -19:]
            for pair in keypoint_matches:
                body_keypoints[pair[0], :] = extra_keypoints[pair[1], :]
            pred_keypoints_img = render_openpose(255 * images_np[i].copy(), body_keypoints) / 255
            body_keypoints = gt_keypoints[i, :25]
            extra_keypoints = gt_keypoints[i, -19:]
            for pair in keypoint_matches:
                if extra_keypoints[pair[1], -1] > 0 and body_keypoints[pair[0], -1] == 0:
                    body_keypoints[pair[0], :] = extra_keypoints[pair[1], :]
            gt_keypoints_img = render_openpose(255*images_np[i].copy(), body_keypoints) / 255
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
            rend_imgs.append(torch.from_numpy(pred_keypoints_img).permute(2,0,1))
            rend_imgs.append(torch.from_numpy(gt_keypoints_img).permute(2,0,1))
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image, focal_length=5000, text=None, resize=None, side_view=False, baseColorFactor=(1.0, 1.0, 0.9, 1.0), rot_angle=90):
        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                              viewport_height=image.shape[0],
                                              point_size=1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=baseColorFactor)

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1])
        scene.add(camera, pose=camera_pose)


        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        if not side_view:
            output_img = (color[:, :, :3] * valid_mask +
                      (1 - valid_mask) * image)
        else:
            output_img = color[:, :, :3]
        if resize is not None:
            output_img = cv2.resize(output_img, resize)

        output_img = output_img.astype(np.float32)
        renderer.delete()
        return output_img
