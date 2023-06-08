from .smpl_wrapper import SMPL
from .hmr2 import HMR2
from .discriminator import Discriminator

from ..utils.download import cache_url
from ..configs import CACHE_DIR


def download_models():
    """Download checkpoints and files for running inference.
    """
    import os
    os.makedirs(os.path.join(CACHE_DIR, "4DHumans"), exist_ok=True)
    download_files = {
        "hmr2_data.tar.gz"      : ["https://people.eecs.berkeley.edu/~shubham-goel/projects/4DHumans/hmr2_data.tar.gz", os.path.join(CACHE_DIR, "4DHumans/")],
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
