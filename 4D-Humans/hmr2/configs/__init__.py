import os
from typing import Dict
from yacs.config import CfgNode as CN

CACHE_DIR = os.path.join(os.environ.get("HOME"), ".cache")
CACHE_DIR_4DHUMANS = os.path.join(CACHE_DIR, "4DHumans")

def to_lower(x: Dict) -> Dict:
    """
    Convert all dictionary keys to lowercase
    Args:
      x (dict): Input dictionary
    Returns:
      dict: Output dictionary with all keys converted to lowercase
    """
    return {k.lower(): v for k, v in x.items()}

_C = CN(new_allowed=True)

_C.GENERAL = CN(new_allowed=True)
_C.GENERAL.RESUME = True
_C.GENERAL.TIME_TO_RUN = 3300
_C.GENERAL.VAL_STEPS = 100
_C.GENERAL.LOG_STEPS = 100
_C.GENERAL.CHECKPOINT_STEPS = 20000
_C.GENERAL.CHECKPOINT_DIR = "checkpoints"
_C.GENERAL.SUMMARY_DIR = "tensorboard"
_C.GENERAL.NUM_GPUS = 1
_C.GENERAL.NUM_WORKERS = 4
_C.GENERAL.MIXED_PRECISION = True
_C.GENERAL.ALLOW_CUDA = True
_C.GENERAL.PIN_MEMORY = False
_C.GENERAL.DISTRIBUTED = False
_C.GENERAL.LOCAL_RANK = 0
_C.GENERAL.USE_SYNCBN = False
_C.GENERAL.WORLD_SIZE = 1

_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.SHUFFLE = True
_C.TRAIN.WARMUP = False
_C.TRAIN.NORMALIZE_PER_IMAGE = False
_C.TRAIN.CLIP_GRAD = False
_C.TRAIN.CLIP_GRAD_VALUE = 1.0
_C.LOSS_WEIGHTS = CN(new_allowed=True)

_C.DATASETS = CN(new_allowed=True)

_C.MODEL = CN(new_allowed=True)
_C.MODEL.IMAGE_SIZE = 224

_C.EXTRA = CN(new_allowed=True)
_C.EXTRA.FOCAL_LENGTH = 5000

_C.DATASETS.CONFIG = CN(new_allowed=True)
_C.DATASETS.CONFIG.SCALE_FACTOR = 0.3
_C.DATASETS.CONFIG.ROT_FACTOR = 30
_C.DATASETS.CONFIG.TRANS_FACTOR = 0.02
_C.DATASETS.CONFIG.COLOR_SCALE = 0.2
_C.DATASETS.CONFIG.ROT_AUG_RATE = 0.6
_C.DATASETS.CONFIG.TRANS_AUG_RATE = 0.5
_C.DATASETS.CONFIG.DO_FLIP = True
_C.DATASETS.CONFIG.FLIP_AUG_RATE = 0.5
_C.DATASETS.CONFIG.EXTREME_CROP_AUG_RATE = 0.10

def default_config() -> CN:
    """
    Get a yacs CfgNode object with the default config values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

def dataset_config(name='datasets_tar.yaml') -> CN:
    """
    Get dataset config file
    Returns:
      CfgNode: Dataset config as a yacs CfgNode object.
    """
    cfg = CN(new_allowed=True)
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), name)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

def dataset_eval_config() -> CN:
    return dataset_config('datasets_eval.yaml')

def get_config(config_file: str, merge: bool = True, update_cachedir: bool = False) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
      merge (bool): Whether to merge with the default config or not.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    if merge:
      cfg = default_config()
    else:
      cfg = CN(new_allowed=True)
    cfg.merge_from_file(config_file)

    if update_cachedir:
      def update_path(path: str) -> str:
        if os.path.isabs(path):
          return path
        return os.path.join(CACHE_DIR_4DHUMANS, path)

      cfg.SMPL.MODEL_PATH = update_path(cfg.SMPL.MODEL_PATH)
      cfg.SMPL.JOINT_REGRESSOR_EXTRA = update_path(cfg.SMPL.JOINT_REGRESSOR_EXTRA)
      cfg.SMPL.MEAN_PARAMS = update_path(cfg.SMPL.MEAN_PARAMS)

    cfg.freeze()
    return cfg
