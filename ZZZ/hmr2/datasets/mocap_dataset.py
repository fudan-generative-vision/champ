import numpy as np
from typing import Dict

class MoCapDataset:

    def __init__(self, dataset_file: str):
        """
        Dataset class used for loading a dataset of unpaired SMPL parameter annotations
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
        """
        data = np.load(dataset_file)
        self.pose = data['body_pose'].astype(np.float32)[:, 3:]
        self.betas = data['betas'].astype(np.float32)
        self.length = len(self.pose)

    def __getitem__(self, idx: int) -> Dict:
        pose = self.pose[idx].copy()
        betas = self.betas[idx].copy()
        item = {'body_pose': pose, 'betas': betas}
        return item

    def __len__(self) -> int:
        return self.length
