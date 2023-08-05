"""
Parts of the code are taken or adapted from
https://github.com/mkocabas/EpipolarPose/blob/master/lib/utils/img_utils.py
"""
import torch
import numpy as np
from skimage.transform import rotate, resize
from skimage.filters import gaussian
import random
import cv2
from typing import List, Dict, Tuple
from yacs.config import CfgNode

def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """Increase the size of the bounding box to match the target shape."""
    if target_aspect_ratio is None:
        return input_shape

    try:
        w , h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    if h_new < h or w_new < w:
        breakpoint()
    return np.array([w_new, h_new])

def expand_bbox_to_aspect_ratio(bbox, target_aspect_ratio=None):
    # bbox: np.array: (N,4) detectron2 bbox format 
    # target_aspect_ratio: (width, height)
    if target_aspect_ratio is None:
        return bbox
    
    is_singleton = (bbox.ndim == 1)
    if is_singleton:
        bbox = bbox[None,:]

    if bbox.shape[0] > 0:
        center = np.stack(((bbox[:,0] + bbox[:,2]) / 2, (bbox[:,1] + bbox[:,3]) / 2), axis=1)
        scale_wh = np.stack((bbox[:,2] - bbox[:,0], bbox[:,3] - bbox[:,1]), axis=1)
        scale_wh = np.stack([expand_to_aspect_ratio(wh, target_aspect_ratio) for wh in scale_wh], axis=0)
        bbox = np.stack([
            center[:,0] - scale_wh[:,0] / 2,
            center[:,1] - scale_wh[:,1] / 2,
            center[:,0] + scale_wh[:,0] / 2,
            center[:,1] + scale_wh[:,1] / 2,
        ], axis=1)

    if is_singleton:
        bbox = bbox[0,:]

    return bbox

def do_augmentation(aug_config: CfgNode) -> Tuple:
    """
    Compute random augmentation parameters.
    Args:
        aug_config (CfgNode): Config containing augmentation parameters.
    Returns:
        scale (float): Box rescaling factor.
        rot (float): Random image rotation.
        do_flip (bool): Whether to flip image or not.
        do_extreme_crop (bool): Whether to apply extreme cropping (as proposed in EFT).
        color_scale (List): Color rescaling factor
        tx (float): Random translation along the x axis.
        ty (float): Random translation along the y axis. 
    """

    tx = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.TRANS_FACTOR
    ty = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.TRANS_FACTOR
    scale = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.SCALE_FACTOR + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * aug_config.ROT_FACTOR if random.random() <= aug_config.ROT_AUG_RATE else 0
    do_flip = aug_config.DO_FLIP and random.random() <= aug_config.FLIP_AUG_RATE
    do_extreme_crop = random.random() <= aug_config.EXTREME_CROP_AUG_RATE
    extreme_crop_lvl = aug_config.get('EXTREME_CROP_AUG_LEVEL', 0)
    # extreme_crop_lvl = 0
    c_up = 1.0 + aug_config.COLOR_SCALE
    c_low = 1.0 - aug_config.COLOR_SCALE
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    return scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty

def rotate_2d(pt_2d: np.array, rot_rad: float) -> np.array:
    """
    Rotate a 2D point on the x-y plane.
    Args:
        pt_2d (np.array): Input 2D point with shape (2,).
        rot_rad (float): Rotation angle
    Returns:
        np.array: Rotated 2D point.
    """
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x: float, c_y: float,
                            src_width: float, src_height: float,
                            dst_width: float, dst_height: float,
                            scale: float, rot: float) -> np.array:
    """
    Create transformation matrix for the bounding box crop.
    Args:
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        src_width (float): Bounding box width.
        src_height (float): Bounding box height.
        dst_width (float): Output box width.
        dst_height (float): Output box height.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        trans (np.array): Target geometric transformation.
    """
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def trans_point2d(pt_2d: np.array, trans: np.array):
    """
    Transform a 2D point using translation matrix trans.
    Args:
        pt_2d (np.array): Input 2D point with shape (2,).
        trans (np.array): Transformation matrix.
    Returns:
        np.array: Transformed 2D point.
    """
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    """Taken from PARE: https://github.com/mkocabas/PARE/blob/6e0caca86c6ab49ff80014b661350958e5b72fd8/pare/utils/image_utils.py"""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0, as_int=True):
    """Transform pixel location to different reference."""
    """Taken from PARE: https://github.com/mkocabas/PARE/blob/6e0caca86c6ab49ff80014b661350958e5b72fd8/pare/utils/image_utils.py"""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    if as_int:
        new_pt = new_pt.astype(int)
    return new_pt[:2] + 1

def crop_img(img, ul, br, border_mode=cv2.BORDER_CONSTANT, border_value=0):
    c_x = (ul[0] + br[0])/2
    c_y = (ul[1] + br[1])/2
    bb_width = patch_width = br[0] - ul[0]
    bb_height = patch_height = br[1] - ul[1]
    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, 1.0, 0)
    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=border_mode,
                                borderValue=border_value
                        )
    
    # Force borderValue=cv2.BORDER_CONSTANT for alpha channel
    if (img.shape[2] == 4) and (border_mode != cv2.BORDER_CONSTANT):
        img_patch[:,:,3] = cv2.warpAffine(img[:,:,3], trans, (int(patch_width), int(patch_height)), 
                                            flags=cv2.INTER_LINEAR, 
                                            borderMode=cv2.BORDER_CONSTANT,
                            )

    return img_patch

def generate_image_patch_skimage(img: np.array, c_x: float, c_y: float,
                                 bb_width: float, bb_height: float,
                                 patch_width: float, patch_height: float,
                                 do_flip: bool, scale: float, rot: float,
                                 border_mode=cv2.BORDER_CONSTANT, border_value=0) -> Tuple[np.array, np.array]:
    """
    Crop image according to the supplied bounding box.
    Args:
        img (np.array): Input image of shape (H, W, 3)
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        bb_width (float): Bounding box width.
        bb_height (float): Bounding box height.
        patch_width (float): Output box width.
        patch_height (float): Output box height.
        do_flip (bool): Whether to flip image or not.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        img_patch (np.array): Cropped image patch of shape (patch_height, patch_height, 3)
        trans (np.array): Transformation matrix.
    """
    
    img_height, img_width, img_channels = img.shape
    if do_flip:
       img = img[:, ::-1, :]
       c_x = img_width - c_x - 1

    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot)

    #img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)

    # skimage
    center = np.zeros(2)
    center[0] = c_x
    center[1] = c_y
    res = np.zeros(2)
    res[0] = patch_width
    res[1] = patch_height
    # assumes bb_width = bb_height
    # assumes patch_width = patch_height
    assert bb_width == bb_height, f'{bb_width=} != {bb_height=}'
    assert patch_width == patch_height, f'{patch_width=} != {patch_height=}'
    scale1 = scale*bb_width/200.
    
    # Upper left point
    ul = np.array(transform([1, 1], center, scale1, res, invert=1, as_int=False)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1,
                             res[1] + 1], center, scale1, res, invert=1, as_int=False)) - 1

    # Padding so that when rotated proper amount of context is included
    try:
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2) + 1
    except:
        breakpoint()
    if not rot == 0:
        ul -= pad
        br += pad


    if False:
        # Old way of cropping image
        ul_int = ul.astype(int)
        br_int = br.astype(int)
        new_shape = [br_int[1] - ul_int[1], br_int[0] - ul_int[0]]
        if len(img.shape) > 2:
            new_shape += [img.shape[2]]
        new_img = np.zeros(new_shape)

        # Range to fill new array
        new_x = max(0, -ul_int[0]), min(br_int[0], len(img[0])) - ul_int[0]
        new_y = max(0, -ul_int[1]), min(br_int[1], len(img)) - ul_int[1]
        # Range to sample from original image
        old_x = max(0, ul_int[0]), min(len(img[0]), br_int[0])
        old_y = max(0, ul_int[1]), min(len(img), br_int[1])
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                        old_x[0]:old_x[1]]

    # New way of cropping image
    new_img = crop_img(img, ul, br, border_mode=border_mode, border_value=border_value).astype(np.float32)

    # print(f'{new_img.shape=}')
    # print(f'{new_img1.shape=}')
    # print(f'{np.allclose(new_img, new_img1)=}')
    # print(f'{img.dtype=}')


    if not rot == 0:
        # Remove padding

        new_img = rotate(new_img, rot) # scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    if new_img.shape[0] < 1 or new_img.shape[1] < 1:
        print(f'{img.shape=}')
        print(f'{new_img.shape=}')
        print(f'{ul=}')
        print(f'{br=}')
        print(f'{pad=}')
        print(f'{rot=}')

        breakpoint()

    # resize image
    new_img = resize(new_img, res) # scipy.misc.imresize(new_img, res)
    
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)

    return new_img, trans


def generate_image_patch_cv2(img: np.array, c_x: float, c_y: float,
                             bb_width: float, bb_height: float,
                             patch_width: float, patch_height: float,
                             do_flip: bool, scale: float, rot: float,
                             border_mode=cv2.BORDER_CONSTANT, border_value=0) -> Tuple[np.array, np.array]:
    """
    Crop the input image and return the crop and the corresponding transformation matrix.
    Args:
        img (np.array): Input image of shape (H, W, 3)
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        bb_width (float): Bounding box width.
        bb_height (float): Bounding box height.
        patch_width (float): Output box width.
        patch_height (float): Output box height.
        do_flip (bool): Whether to flip image or not.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        img_patch (np.array): Cropped image patch of shape (patch_height, patch_height, 3)
        trans (np.array): Transformation matrix.
    """

    img_height, img_width, img_channels = img.shape
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1


    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), 
                        flags=cv2.INTER_LINEAR, 
                        borderMode=border_mode,
                        borderValue=border_value,
                )
    # Force borderValue=cv2.BORDER_CONSTANT for alpha channel
    if (img.shape[2] == 4) and (border_mode != cv2.BORDER_CONSTANT):
        img_patch[:,:,3] = cv2.warpAffine(img[:,:,3], trans, (int(patch_width), int(patch_height)), 
                                            flags=cv2.INTER_LINEAR, 
                                            borderMode=cv2.BORDER_CONSTANT,
                            )

    return img_patch, trans


def convert_cvimg_to_tensor(cvimg: np.array):
    """
    Convert image from HWC to CHW format.
    Args:
        cvimg (np.array): Image of shape (H, W, 3) as loaded by OpenCV.
    Returns:
        np.array: Output image of shape (3, H, W).
    """
    # from h,w,c(OpenCV) to c,h,w
    img = cvimg.copy()
    img = np.transpose(img, (2, 0, 1))
    # from int to float
    img = img.astype(np.float32)
    return img

def fliplr_params(smpl_params: Dict, has_smpl_params: Dict) -> Tuple[Dict, Dict]:
    """
    Flip SMPL parameters when flipping the image.
    Args:
        smpl_params (Dict): SMPL parameter annotations.
        has_smpl_params (Dict): Whether SMPL annotations are valid.
    Returns:
        Dict, Dict: Flipped SMPL parameters and valid flags.
    """
    global_orient = smpl_params['global_orient'].copy()
    body_pose = smpl_params['body_pose'].copy()
    betas = smpl_params['betas'].copy()
    has_global_orient = has_smpl_params['global_orient'].copy()
    has_body_pose = has_smpl_params['body_pose'].copy()
    has_betas = has_smpl_params['betas'].copy()

    body_pose_permutation = [6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13,
                             14 ,18, 19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33,
                             34, 35, 30, 31, 32, 36, 37, 38, 42, 43, 44, 39, 40, 41,
                             45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58, 59, 54, 55,
                             56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68]
    body_pose_permutation = body_pose_permutation[:len(body_pose)]
    body_pose_permutation = [i-3 for i in body_pose_permutation]

    body_pose = body_pose[body_pose_permutation]

    global_orient[1::3] *= -1
    global_orient[2::3] *= -1
    body_pose[1::3] *= -1
    body_pose[2::3] *= -1

    smpl_params = {'global_orient': global_orient.astype(np.float32),
                   'body_pose': body_pose.astype(np.float32),
                   'betas': betas.astype(np.float32)
                  }

    has_smpl_params = {'global_orient': has_global_orient,
                       'body_pose': has_body_pose,
                       'betas': has_betas
                      }

    return smpl_params, has_smpl_params


def fliplr_keypoints(joints: np.array, width: float, flip_permutation: List[int]) -> np.array:
    """
    Flip 2D or 3D keypoints.
    Args:
        joints (np.array): Array of shape (N, 3) or (N, 4) containing 2D or 3D keypoint locations and confidence.
        flip_permutation (List): Permutation to apply after flipping.
    Returns:
        np.array: Flipped 2D or 3D keypoints with shape (N, 3) or (N, 4) respectively.
    """
    joints = joints.copy()
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1
    joints = joints[flip_permutation, :]

    return joints

def keypoint_3d_processing(keypoints_3d: np.array, flip_permutation: List[int], rot: float, do_flip: float) -> np.array:
    """
    Process 3D keypoints (rotation/flipping).
    Args:
        keypoints_3d (np.array): Input array of shape (N, 4) containing the 3D keypoints and confidence.
        flip_permutation (List): Permutation to apply after flipping.
        rot (float): Random rotation applied to the keypoints.
        do_flip (bool): Whether to flip keypoints or not.
    Returns:
        np.array: Transformed 3D keypoints with shape (N, 4).
    """
    if do_flip:
        keypoints_3d = fliplr_keypoints(keypoints_3d, 1, flip_permutation)
    # in-plane rotation
    rot_mat = np.eye(3)
    if not rot == 0:
        rot_rad = -rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
    keypoints_3d[:, :-1] = np.einsum('ij,kj->ki', rot_mat, keypoints_3d[:, :-1])
    # flip the x coordinates
    keypoints_3d = keypoints_3d.astype('float32')
    return keypoints_3d

def rot_aa(aa: np.array, rot: float) -> np.array:
    """
    Rotate axis angle parameters.
    Args:
        aa (np.array): Axis-angle vector of shape (3,).
        rot (np.array): Rotation angle in degrees.
    Returns:
        np.array: Rotated axis-angle vector.
    """
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa.astype(np.float32)

def smpl_param_processing(smpl_params: Dict, has_smpl_params: Dict, rot: float, do_flip: bool) -> Tuple[Dict, Dict]:
    """
    Apply random augmentations to the SMPL parameters.
    Args:
        smpl_params (Dict): SMPL parameter annotations.
        has_smpl_params (Dict): Whether SMPL annotations are valid.
        rot (float): Random rotation applied to the keypoints.
        do_flip (bool): Whether to flip keypoints or not.
    Returns:
        Dict, Dict: Transformed SMPL parameters and valid flags.
    """
    if do_flip:
        smpl_params, has_smpl_params = fliplr_params(smpl_params, has_smpl_params)
    smpl_params['global_orient'] = rot_aa(smpl_params['global_orient'], rot)
    return smpl_params, has_smpl_params



def get_example(img_path: str|np.ndarray, center_x: float, center_y: float,
                width: float, height: float,
                keypoints_2d: np.array, keypoints_3d: np.array,
                smpl_params: Dict, has_smpl_params: Dict,
                flip_kp_permutation: List[int],
                patch_width: int, patch_height: int,
                mean: np.array, std: np.array,
                do_augment: bool, augm_config: CfgNode,
                is_bgr: bool = True,
                use_skimage_antialias: bool = False,
                border_mode: int = cv2.BORDER_CONSTANT,
                return_trans: bool = False) -> Tuple:
    """
    Get an example from the dataset and (possibly) apply random augmentations.
    Args:
        img_path (str): Image filename
        center_x (float): Bounding box center x coordinate in the original image.
        center_y (float): Bounding box center y coordinate in the original image.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array with shape (N,3) containing the 2D keypoints in the original image coordinates.
        keypoints_3d (np.array): Array with shape (N,4) containing the 3D keypoints.
        smpl_params (Dict): SMPL parameter annotations.
        has_smpl_params (Dict): Whether SMPL annotations are valid.
        flip_kp_permutation (List): Permutation to apply to the keypoints after flipping.
        patch_width (float): Output box width.
        patch_height (float): Output box height.
        mean (np.array): Array of shape (3,) containing the mean for normalizing the input image.
        std (np.array): Array of shape (3,) containing the std for normalizing the input image.
        do_augment (bool): Whether to apply data augmentation or not.
        aug_config (CfgNode): Config containing augmentation parameters.
    Returns:
        return img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size
        img_patch (np.array): Cropped image patch of shape (3, patch_height, patch_height)
        keypoints_2d (np.array): Array with shape (N,3) containing the transformed 2D keypoints.
        keypoints_3d (np.array): Array with shape (N,4) containing the transformed 3D keypoints.
        smpl_params (Dict): Transformed SMPL parameters.
        has_smpl_params (Dict): Valid flag for transformed SMPL parameters.
        img_size (np.array): Image size of the original image.
        """
    if isinstance(img_path, str):
        # 1. load image
        cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % img_path)
    elif isinstance(img_path, np.ndarray):
        cvimg = img_path
    else:
        raise TypeError('img_path must be either a string or a numpy array')
    img_height, img_width, img_channels = cvimg.shape

    img_size = np.array([img_height, img_width])

    # 2. get augmentation params
    if do_augment:
        scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty = do_augmentation(augm_config)
    else:
        scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty = 1.0, 0, False, False, 0, [1.0, 1.0, 1.0], 0., 0.

    if width < 1 or height < 1:
        breakpoint()

    if do_extreme_crop:
        if extreme_crop_lvl == 0:
            center_x1, center_y1, width1, height1 = extreme_cropping(center_x, center_y, width, height, keypoints_2d)
        elif extreme_crop_lvl == 1:
            center_x1, center_y1, width1, height1 = extreme_cropping_aggressive(center_x, center_y, width, height, keypoints_2d)

        THRESH = 4
        if width1 < THRESH or height1 < THRESH:
            # print(f'{do_extreme_crop=}')
            # print(f'width: {width}, height: {height}')
            # print(f'width1: {width1}, height1: {height1}')
            # print(f'center_x: {center_x}, center_y: {center_y}')
            # print(f'center_x1: {center_x1}, center_y1: {center_y1}')
            # print(f'keypoints_2d: {keypoints_2d}')
            # print(f'\n\n', flush=True)
            # breakpoint()
            pass
            # print(f'skip ==> width1: {width1}, height1: {height1}, width: {width}, height: {height}')
        else:
            center_x, center_y, width, height = center_x1, center_y1, width1, height1

    center_x += width * tx
    center_y += height * ty

    # Process 3D keypoints
    keypoints_3d = keypoint_3d_processing(keypoints_3d, flip_kp_permutation, rot, do_flip)

    # 3. generate image patch
    if use_skimage_antialias:
        # Blur image to avoid aliasing artifacts
        downsampling_factor = (patch_width / (width*scale))
        if downsampling_factor > 1.1:
            cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True, truncate=3.0)

    img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    width, height,
                                                    patch_width, patch_height,
                                                    do_flip, scale, rot, 
                                                    border_mode=border_mode)
        # img_patch_cv, trans = generate_image_patch_skimage(cvimg,
        #                                                 center_x, center_y,
        #                                                 width, height,
        #                                                 patch_width, patch_height,
        #                                                 do_flip, scale, rot, 
        #                                                 border_mode=border_mode)

    image = img_patch_cv.copy()
    if is_bgr:
        image = image[:, :, ::-1]
    img_patch_cv = image.copy()
    img_patch = convert_cvimg_to_tensor(image)


    smpl_params, has_smpl_params = smpl_param_processing(smpl_params, has_smpl_params, rot, do_flip)

    # apply normalization
    for n_c in range(min(img_channels, 3)):
        img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        if mean is not None and std is not None:
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]
    if do_flip:
        keypoints_2d = fliplr_keypoints(keypoints_2d, img_width, flip_kp_permutation)


    for n_jt in range(len(keypoints_2d)):
        keypoints_2d[n_jt, 0:2] = trans_point2d(keypoints_2d[n_jt, 0:2], trans)
    keypoints_2d[:, :-1] = keypoints_2d[:, :-1] / patch_width - 0.5

    if not return_trans:
        return img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size
    else:
        return img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size, trans

def crop_to_hips(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array) -> Tuple:
    """
    Extreme cropping: Crop the box up to the hip locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24, 25+0, 25+1, 25+4, 25+5]
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_to_shoulders(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box up to the shoulder locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16]]
    keypoints_2d[lower_body_keypoints, :] = 0
    center, scale = get_bbox(keypoints_2d)
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.2 * scale[0]
        height = 1.2 * scale[1]
    return center_x, center_y, width, height

def crop_to_head(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the head.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16]]
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.3 * scale[0]
        height = 1.3 * scale[1]
    return center_x, center_y, width, height

def crop_torso_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the torso.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nontorso_body_keypoints = [0, 3, 4, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 4, 5, 6, 7, 10, 11, 13, 17, 18]]
    keypoints_2d[nontorso_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_rightarm_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the right arm.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonrightarm_body_keypoints = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonrightarm_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_leftarm_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the left arm.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonleftarm_body_keypoints = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonleftarm_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_legs_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the legs.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonlegs_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18] + [25 + i for i in [6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]]
    keypoints_2d[nonlegs_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_rightleg_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the right leg.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonrightleg_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] + [25 + i for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonrightleg_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_leftleg_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the left leg.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonleftleg_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 22, 23, 24] + [25 + i for i in [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonleftleg_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def full_body(keypoints_2d: np.array) -> bool:
    """
    Check if all main body joints are visible.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """

    body_keypoints_openpose = [2, 3, 4, 5, 6, 7, 10, 11, 13, 14]
    body_keypoints = [25 + i for i in [8, 7, 6, 9, 10, 11, 1, 0, 4, 5]]
    return (np.maximum(keypoints_2d[body_keypoints, -1], keypoints_2d[body_keypoints_openpose, -1]) > 0).sum() == len(body_keypoints)

def upper_body(keypoints_2d: np.array):
    """
    Check if all upper body joints are visible.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """
    lower_body_keypoints_openpose = [10, 11, 13, 14]
    lower_body_keypoints = [25 + i for i in [1, 0, 4, 5]]
    upper_body_keypoints_openpose = [0, 1, 15, 16, 17, 18]
    upper_body_keypoints = [25+8, 25+9, 25+12, 25+13, 25+17, 25+18]
    return ((keypoints_2d[lower_body_keypoints + lower_body_keypoints_openpose, -1] > 0).sum() == 0)\
       and ((keypoints_2d[upper_body_keypoints + upper_body_keypoints_openpose, -1] > 0).sum() >= 2)

def get_bbox(keypoints_2d: np.array, rescale: float = 1.2) -> Tuple:
    """
    Get center and scale for bounding box from openpose detections.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center (np.array): Array of shape (2,) containing the new bounding box center.
        scale (float): New bounding box scale.
    """
    valid = keypoints_2d[:,-1] > 0
    valid_keypoints = keypoints_2d[valid][:,:-1]
    center = 0.5 * (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0))
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0))
    # adjust bounding box tightness
    scale = bbox_size
    scale *= rescale
    return center, scale

def extreme_cropping(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array) -> Tuple:
    """
    Perform extreme cropping
    Args:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
    """
    p = torch.rand(1).item()
    if full_body(keypoints_2d):
        if p < 0.7:
            center_x, center_y, width, height = crop_to_hips(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.9:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
    elif upper_body(keypoints_2d):
        if p < 0.9:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)

    return center_x, center_y, max(width, height), max(width, height)

def extreme_cropping_aggressive(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array) -> Tuple:
    """
    Perform aggressive extreme cropping
    Args:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
    """
    p = torch.rand(1).item()
    if full_body(keypoints_2d):
        if p < 0.2:
            center_x, center_y, width, height = crop_to_hips(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.3:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.4:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.5:
            center_x, center_y, width, height = crop_torso_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.6:
            center_x, center_y, width, height = crop_rightarm_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.7:
            center_x, center_y, width, height = crop_leftarm_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.8:
            center_x, center_y, width, height = crop_legs_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.9:
            center_x, center_y, width, height = crop_rightleg_only(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_leftleg_only(center_x, center_y, width, height, keypoints_2d)
    elif upper_body(keypoints_2d):
        if p < 0.2:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.4:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.6:
            center_x, center_y, width, height = crop_torso_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.8:
            center_x, center_y, width, height = crop_rightarm_only(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_leftarm_only(center_x, center_y, width, height, keypoints_2d)
    return center_x, center_y, max(width, height), max(width, height)
