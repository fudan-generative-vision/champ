from typing import Dict, Optional

import torch
import numpy as np
import pytorch_lightning as pl
from yacs.config import CfgNode

import webdataset as wds
from ..configs import to_lower
from .dataset import Dataset
from .image_dataset import ImageDataset
from .mocap_dataset import MoCapDataset

def create_dataset(cfg: CfgNode, dataset_cfg: CfgNode, train: bool = True, **kwargs) -> Dataset:
    """
    Instantiate a dataset from a config file.
    Args:
        cfg (CfgNode): Model configuration file.
        dataset_cfg (CfgNode): Dataset configuration info.
        train (bool): Variable to select between train and val datasets.
    """

    dataset_type = Dataset.registry[dataset_cfg.TYPE]
    return dataset_type(cfg, **to_lower(dataset_cfg), train=train, **kwargs)

def create_webdataset(cfg: CfgNode, dataset_cfg: CfgNode, train: bool = True) -> Dataset:
    """
    Like `create_dataset` but load data from tars.
    """
    dataset_type = Dataset.registry[dataset_cfg.TYPE]
    return dataset_type.load_tars_as_webdataset(cfg, **to_lower(dataset_cfg), train=train)


class MixedWebDataset(wds.WebDataset):
    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode, train: bool = True) -> None:
        super(wds.WebDataset, self).__init__()
        dataset_list = cfg.DATASETS.TRAIN if train else cfg.DATASETS.VAL
        datasets = [create_webdataset(cfg, dataset_cfg[dataset], train=train) for dataset, v in dataset_list.items()]
        weights = np.array([v.WEIGHT for dataset, v in dataset_list.items()])
        weights = weights / weights.sum()  # normalize
        self.append(wds.RandomMix(datasets, weights))

class HMR2DataModule(pl.LightningDataModule):

    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode) -> None:
        """
        Initialize LightningDataModule for HMR2 training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.mocap_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load datasets necessary for training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
        """
        if self.train_dataset == None:
            self.train_dataset = MixedWebDataset(self.cfg, self.dataset_cfg, train=True).with_epoch(100_000).shuffle(4000)
            self.val_dataset = MixedWebDataset(self.cfg, self.dataset_cfg, train=False).shuffle(4000)
            self.mocap_dataset = MoCapDataset(**to_lower(self.dataset_cfg[self.cfg.DATASETS.MOCAP]))

    def train_dataloader(self) -> Dict:
        """
        Setup training data loader.
        Returns:
            Dict: Dictionary containing image and mocap data dataloaders
        """
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, self.cfg.TRAIN.BATCH_SIZE, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS, prefetch_factor=self.cfg.GENERAL.PREFETCH_FACTOR)
        mocap_dataloader = torch.utils.data.DataLoader(self.mocap_dataset, self.cfg.TRAIN.NUM_TRAIN_SAMPLES * self.cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=1)
        return {'img': train_dataloader, 'mocap': mocap_dataloader}

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Setup val data loader.
        Returns:
            torch.utils.data.DataLoader: Validation dataloader  
        """
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, self.cfg.TRAIN.BATCH_SIZE, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS)
        return val_dataloader
