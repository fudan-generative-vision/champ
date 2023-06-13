import warnings
from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger

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

        # Overriding the SMPL params with the HMR2 params
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out

class PHALP_Prime_HMR2(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = HMR2Predictor(self.cfg)

@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    pass

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:

    """Main function for running the PHALP tracker."""

    phalp_tracker = PHALP_Prime_HMR2(cfg)

    phalp_tracker.track()

if __name__ == "__main__":
    main()
