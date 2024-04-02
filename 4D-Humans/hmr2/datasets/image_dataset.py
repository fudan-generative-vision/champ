import copy
import os
import numpy as np
import torch
from typing import Any, Dict, List
from yacs.config import CfgNode
import braceexpand
import cv2

from .dataset import Dataset
from .utils import get_example, expand_to_aspect_ratio
from .smplh_prob_filter import poses_check_probable, load_amass_hist_smooth

def expand(s):
    return os.path.expanduser(os.path.expandvars(s))
def expand_urls(urls: str|List[str]):
    if isinstance(urls, str):
        urls = [urls]
    urls = [u for url in urls for u in braceexpand.braceexpand(expand(url))]
    return urls

AIC_TRAIN_CORRUPT_KEYS = {
    '0a047f0124ae48f8eee15a9506ce1449ee1ba669',
    '1a703aa174450c02fbc9cfbf578a5435ef403689',
    '0394e6dc4df78042929b891dbc24f0fd7ffb6b6d',
    '5c032b9626e410441544c7669123ecc4ae077058',
    'ca018a7b4c5f53494006ebeeff9b4c0917a55f07',
    '4a77adb695bef75a5d34c04d589baf646fe2ba35',
    'a0689017b1065c664daef4ae2d14ea03d543217e',
    '39596a45cbd21bed4a5f9c2342505532f8ec5cbb',
    '3d33283b40610d87db660b62982f797d50a7366b',
}
CORRUPT_KEYS = {
    *{f'aic-train/{k}' for k in AIC_TRAIN_CORRUPT_KEYS},
    *{f'aic-train-vitpose/{k}' for k in AIC_TRAIN_CORRUPT_KEYS},
}

body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])
DEFAULT_IMG_SIZE = 256

class ImageDataset(Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 dataset_file: str,
                 img_dir: str,
                 train: bool = True,
                 prune: Dict[str, Any] = {},
                 **kwargs):
        """
        Dataset class used for loading images and corresponding annotations.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        super(ImageDataset, self).__init__()
        self.train = train
        self.cfg = cfg

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        self.img_dir = img_dir
        self.data = np.load(dataset_file, allow_pickle=True)

        self.imgname = self.data['imgname']
        self.personid = np.zeros(len(self.imgname), dtype=np.int32)
        self.extra_info = self.data.get('extra_info', [{} for _ in range(len(self.imgname))])

        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)

        num_pose = 3 * (self.cfg.SMPL.NUM_BODY_JOINTS + 1)

        # Bounding boxes are assumed to be in the center and scale format
        self.center = self.data['center']
        self.scale = self.data['scale'].reshape(len(self.center), -1) / 200.0
        if self.scale.shape[1] == 1:
            self.scale = np.tile(self.scale, (1, 2))
        assert self.scale.shape == (len(self.center), 2)

        # Get gt SMPLX parameters, if available
        try:
            self.body_pose = self.data['body_pose'].astype(np.float32)
            self.has_body_pose = self.data['has_body_pose'].astype(np.float32)
        except KeyError:
            self.body_pose = np.zeros((len(self.imgname), num_pose), dtype=np.float32)
            self.has_body_pose = np.zeros(len(self.imgname), dtype=np.float32)
        try:
            self.betas = self.data['betas'].astype(np.float32)
            self.has_betas = self.data['has_betas'].astype(np.float32)
        except KeyError:
            self.betas = np.zeros((len(self.imgname), 10), dtype=np.float32)
            self.has_betas = np.zeros(len(self.imgname), dtype=np.float32)

        # Try to get 2d keypoints, if available
        try:
            body_keypoints_2d = self.data['body_keypoints_2d']
        except KeyError:
            body_keypoints_2d = np.zeros((len(self.center), 25, 3))
        # Try to get extra 2d keypoints, if available
        try:
            extra_keypoints_2d = self.data['extra_keypoints_2d']
        except KeyError:
            extra_keypoints_2d = np.zeros((len(self.center), 19, 3))

        self.keypoints_2d = np.concatenate((body_keypoints_2d, extra_keypoints_2d), axis=1).astype(np.float32)

        # Try to get 3d keypoints, if available
        try:
            body_keypoints_3d = self.data['body_keypoints_3d'].astype(np.float32)
        except KeyError:
            body_keypoints_3d = np.zeros((len(self.center), 25, 4), dtype=np.float32)
        # Try to get extra 3d keypoints, if available
        try:
            extra_keypoints_3d = self.data['extra_keypoints_3d'].astype(np.float32)
        except KeyError:
            extra_keypoints_3d = np.zeros((len(self.center), 19, 4), dtype=np.float32)

        body_keypoints_3d[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], -1] = 0

        self.keypoints_3d = np.concatenate((body_keypoints_3d, extra_keypoints_3d), axis=1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.scale)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns an example from the dataset.
        """
        try:
            image_file_rel = self.imgname[idx].decode('utf-8')
        except AttributeError:
            image_file_rel = self.imgname[idx]
        image_file = os.path.join(self.img_dir, image_file_rel)
        keypoints_2d = self.keypoints_2d[idx].copy()
        keypoints_3d = self.keypoints_3d[idx].copy()

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]
        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()
        bbox_expand_factor = bbox_size / ((scale*200).max())
        body_pose = self.body_pose[idx].copy().astype(np.float32)
        betas = self.betas[idx].copy().astype(np.float32)

        has_body_pose = self.has_body_pose[idx].copy()
        has_betas = self.has_betas[idx].copy()

        smpl_params = {'global_orient': body_pose[:3],
                       'body_pose': body_pose[3:],
                       'betas': betas
                      }

        has_smpl_params = {'global_orient': has_body_pose,
                           'body_pose': has_body_pose,
                           'betas': has_betas
                           }

        smpl_params_is_axis_angle = {'global_orient': True,
                                     'body_pose': True,
                                     'betas': False
                                    }

        augm_config = self.cfg.DATASETS.CONFIG
        # Crop image and (possibly) perform data augmentation
        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size = get_example(image_file,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    keypoints_2d, keypoints_3d,
                                                                                                    smpl_params, has_smpl_params,
                                                                                                    self.flip_keypoint_permutation,
                                                                                                    self.img_size, self.img_size,
                                                                                                    self.mean, self.std, self.train, augm_config)

        item = {}
        # These are the keypoints in the original image coordinates (before cropping)
        orig_keypoints_2d = self.keypoints_2d[idx].copy()

        item['img'] = img_patch
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['keypoints_3d'] = keypoints_3d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = self.center[idx].copy()
        item['box_size'] = bbox_size
        item['bbox_expand_factor'] = bbox_expand_factor
        item['img_size'] = 1.0 * img_size[::-1].copy()
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['imgname'] = image_file
        item['imgname_rel'] = image_file_rel
        item['personid'] = int(self.personid[idx])
        item['extra_info'] = copy.deepcopy(self.extra_info[idx])
        item['idx'] = idx
        item['_scale'] = scale
        return item


    @staticmethod
    def load_tars_as_webdataset(cfg: CfgNode, urls: str|List[str], train: bool,
            resampled=False,
            epoch_size=None,
            cache_dir=None,
            **kwargs) -> Dataset:
        """
        Loads the dataset from a webdataset tar file.
        """

        IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        BBOX_SHAPE = cfg.MODEL.get('BBOX_SHAPE', None)
        MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        STD = 255. * np.array(cfg.MODEL.IMAGE_STD)

        def split_data(source):
            for item in source:
                datas = item['data.pyd']
                for data in datas:
                    if 'detection.npz' in item:
                        det_idx = data['extra_info']['detection_npz_idx']
                        mask = item['detection.npz']['masks'][det_idx]
                    else:
                        mask = np.ones_like(item['jpg'][:,:,0], dtype=bool)
                    yield {
                        '__key__': item['__key__'],
                        'jpg': item['jpg'],
                        'data.pyd': data,
                        'mask': mask,
                    }

        def suppress_bad_kps(item, thresh=0.0):
            if thresh > 0:
                kp2d = item['data.pyd']['keypoints_2d']
                kp2d_conf = np.where(kp2d[:, 2] < thresh, 0.0, kp2d[:, 2])
                item['data.pyd']['keypoints_2d'] = np.concatenate([kp2d[:,:2], kp2d_conf[:,None]], axis=1)
            return item

        def filter_numkp(item, numkp=4, thresh=0.0):
            kp_conf = item['data.pyd']['keypoints_2d'][:, 2]
            return (kp_conf > thresh).sum() > numkp

        def filter_reproj_error(item, thresh=10**4.5):
            losses = item['data.pyd'].get('extra_info', {}).get('fitting_loss', np.array({})).item()
            reproj_loss = losses.get('reprojection_loss', None)
            return reproj_loss is None or reproj_loss < thresh

        def filter_bbox_size(item, thresh=1):
            bbox_size_min = item['data.pyd']['scale'].min().item() * 200.
            return bbox_size_min > thresh

        def filter_no_poses(item):
            return (item['data.pyd']['has_body_pose'] > 0)

        def supress_bad_betas(item, thresh=3):
            has_betas = item['data.pyd']['has_betas']
            if thresh > 0 and has_betas:
                betas_abs = np.abs(item['data.pyd']['betas'])
                if (betas_abs > thresh).any():
                    item['data.pyd']['has_betas'] = False
            return item

        amass_poses_hist100_smooth = load_amass_hist_smooth()
        def supress_bad_poses(item):
            has_body_pose = item['data.pyd']['has_body_pose']
            if has_body_pose:
                body_pose = item['data.pyd']['body_pose']
                pose_is_probable = poses_check_probable(torch.from_numpy(body_pose)[None, 3:], amass_poses_hist100_smooth).item()
                if not pose_is_probable:
                    item['data.pyd']['has_body_pose'] = False
            return item

        def poses_betas_simultaneous(item):
            # We either have both body_pose and betas, or neither
            has_betas = item['data.pyd']['has_betas']
            has_body_pose = item['data.pyd']['has_body_pose']
            item['data.pyd']['has_betas'] = item['data.pyd']['has_body_pose'] = np.array(float((has_body_pose>0) and (has_betas>0)))
            return item

        def set_betas_for_reg(item):
            # Always have betas set to true
            has_betas = item['data.pyd']['has_betas']
            betas = item['data.pyd']['betas']

            if not (has_betas>0):
                item['data.pyd']['has_betas'] = np.array(float((True)))
                item['data.pyd']['betas'] = betas * 0
            return item

        # Load the dataset
        if epoch_size is not None:
            resampled = True
        corrupt_filter = lambda sample: (sample['__key__'] not in CORRUPT_KEYS)
        import webdataset as wds
        dataset = wds.WebDataset(expand_urls(urls),
                                nodesplitter=wds.split_by_node,
                                shardshuffle=True,
                                resampled=resampled,
                                cache_dir=cache_dir,
                              ).select(corrupt_filter)
        if train:
            dataset = dataset.shuffle(100)
        dataset = dataset.decode('rgb8').rename(jpg='jpg;jpeg;png')

        # Process the dataset
        dataset = dataset.compose(split_data)

        # Filter/clean the dataset
        SUPPRESS_KP_CONF_THRESH = cfg.DATASETS.get('SUPPRESS_KP_CONF_THRESH', 0.0)
        SUPPRESS_BETAS_THRESH = cfg.DATASETS.get('SUPPRESS_BETAS_THRESH', 0.0)
        SUPPRESS_BAD_POSES = cfg.DATASETS.get('SUPPRESS_BAD_POSES', False)
        POSES_BETAS_SIMULTANEOUS = cfg.DATASETS.get('POSES_BETAS_SIMULTANEOUS', False)
        BETAS_REG = cfg.DATASETS.get('BETAS_REG', False)
        FILTER_NO_POSES = cfg.DATASETS.get('FILTER_NO_POSES', False)
        FILTER_NUM_KP = cfg.DATASETS.get('FILTER_NUM_KP', 4)
        FILTER_NUM_KP_THRESH = cfg.DATASETS.get('FILTER_NUM_KP_THRESH', 0.0)
        FILTER_REPROJ_THRESH = cfg.DATASETS.get('FILTER_REPROJ_THRESH', 0.0)
        FILTER_MIN_BBOX_SIZE = cfg.DATASETS.get('FILTER_MIN_BBOX_SIZE', 0.0)
        if SUPPRESS_KP_CONF_THRESH > 0:
            dataset = dataset.map(lambda x: suppress_bad_kps(x, thresh=SUPPRESS_KP_CONF_THRESH))
        if SUPPRESS_BETAS_THRESH > 0:
            dataset = dataset.map(lambda x: supress_bad_betas(x, thresh=SUPPRESS_BETAS_THRESH))
        if SUPPRESS_BAD_POSES:
            dataset = dataset.map(lambda x: supress_bad_poses(x))
        if POSES_BETAS_SIMULTANEOUS:
            dataset = dataset.map(lambda x: poses_betas_simultaneous(x))
        if FILTER_NO_POSES:
            dataset = dataset.select(lambda x: filter_no_poses(x))
        if FILTER_NUM_KP > 0:
            dataset = dataset.select(lambda x: filter_numkp(x, numkp=FILTER_NUM_KP, thresh=FILTER_NUM_KP_THRESH))
        if FILTER_REPROJ_THRESH > 0:
            dataset = dataset.select(lambda x: filter_reproj_error(x, thresh=FILTER_REPROJ_THRESH))
        if FILTER_MIN_BBOX_SIZE > 0:
            dataset = dataset.select(lambda x: filter_bbox_size(x, thresh=FILTER_MIN_BBOX_SIZE))
        if BETAS_REG:
            dataset = dataset.map(lambda x: set_betas_for_reg(x))       # NOTE: Must be at the end

        use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        # Process the dataset further
        dataset = dataset.map(lambda x: ImageDataset.process_webdataset_tar_item(x, train,
                                                        augm_config=cfg.DATASETS.CONFIG,
                                                        MEAN=MEAN, STD=STD, IMG_SIZE=IMG_SIZE,
                                                        BBOX_SHAPE=BBOX_SHAPE,
                                                        use_skimage_antialias=use_skimage_antialias,
                                                        border_mode=border_mode,
                                                        ))
        if epoch_size is not None:
            dataset = dataset.with_epoch(epoch_size)

        return dataset

    @staticmethod
    def process_webdataset_tar_item(item, train, 
                                    augm_config=None, 
                                    MEAN=DEFAULT_MEAN, 
                                    STD=DEFAULT_STD, 
                                    IMG_SIZE=DEFAULT_IMG_SIZE,
                                    BBOX_SHAPE=None,
                                    use_skimage_antialias=False,
                                    border_mode=cv2.BORDER_CONSTANT,
                                    ):
        # Read data from item
        key = item['__key__']
        image = item['jpg']
        data = item['data.pyd']
        mask = item['mask']

        keypoints_2d = data['keypoints_2d']
        keypoints_3d = data['keypoints_3d']
        center = data['center']
        scale = data['scale']
        body_pose = data['body_pose']
        betas = data['betas']
        has_body_pose = data['has_body_pose']
        has_betas = data['has_betas']
        # image_file = data['image_file']

        # Process data
        orig_keypoints_2d = keypoints_2d.copy()
        center_x = center[0]
        center_y = center[1]
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()
        if bbox_size < 1:
            breakpoint()


        smpl_params = {'global_orient': body_pose[:3],
                    'body_pose': body_pose[3:],
                    'betas': betas
                    }

        has_smpl_params = {'global_orient': has_body_pose,
                        'body_pose': has_body_pose,
                        'betas': has_betas
                        }

        smpl_params_is_axis_angle = {'global_orient': True,
                                    'body_pose': True,
                                    'betas': False
                                    }

        augm_config = copy.deepcopy(augm_config)
        # Crop image and (possibly) perform data augmentation
        img_rgba = np.concatenate([image, mask.astype(np.uint8)[:,:,None]*255], axis=2)
        img_patch_rgba, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size, trans = get_example(img_rgba,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    keypoints_2d, keypoints_3d,
                                                                                                    smpl_params, has_smpl_params,
                                                                                                    FLIP_KEYPOINT_PERMUTATION,
                                                                                                    IMG_SIZE, IMG_SIZE,
                                                                                                    MEAN, STD, train, augm_config,
                                                                                                    is_bgr=False, return_trans=True,
                                                                                                    use_skimage_antialias=use_skimage_antialias,
                                                                                                    border_mode=border_mode,
                                                                                                    )
        img_patch = img_patch_rgba[:3,:,:]
        mask_patch = (img_patch_rgba[3,:,:] / 255.0).clip(0,1)
        if (mask_patch < 0.5).all():
            mask_patch = np.ones_like(mask_patch)

        item = {}

        item['img'] = img_patch
        item['mask'] = mask_patch
        # item['img_og'] = image
        # item['mask_og'] = mask
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['keypoints_3d'] = keypoints_3d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = center.copy()
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * img_size[::-1].copy()
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['_scale'] = scale
        item['_trans'] = trans
        item['imgname'] = key
        # item['idx'] = idx
        return item
