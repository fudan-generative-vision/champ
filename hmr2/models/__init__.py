from .smpl_wrapper import SMPL
from .hmr2 import HMR2
from .discriminator import Discriminator

from ..utils.download import cache_url
from ..configs import CACHE_DIR_4DHUMANS


def download_models(folder=CACHE_DIR_4DHUMANS):
    """Download checkpoints and files for running inference.
    """
    import os
    os.makedirs(folder, exist_ok=True)
    download_files = {
        "hmr2_data.tar.gz"      : ["https://people.eecs.berkeley.edu/~shubham-goel/projects/4DHumans/hmr2_data.tar.gz", folder],
    }
    
    for file_name, url in download_files.items():
        if not os.path.exists(os.path.join(url[1], file_name)):
            print("Downloading file: " + file_name)
            # output = gdown.cached_download(url[0], os.path.join(url[1], file_name), fuzzy=True)
            output = cache_url(url[0], os.path.join(url[1], file_name))
            assert os.path.exists(os.path.join(url[1], file_name)), f"{output} does not exist"

            # if ends with tar.gz, tar -xzf
            if file_name.endswith(".tar.gz"):
                print("Extracting file: " + file_name)
                os.system("tar -xvf " + os.path.join(url[1], file_name) + " -C " + url[1])

DEFAULT_CHECKPOINT=f'{CACHE_DIR_4DHUMANS}/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
def load_hmr2(checkpoint_path=DEFAULT_CHECKPOINT):
    from pathlib import Path
    from ..configs import get_config
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)
    model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg
