# 4DHumans: Reconstructing and Tracking Humans with Transformers
Code repository for the paper:
**Humans in 4D: Reconstructing and Tracking Humans with Transformers**
[Shubham Goel](https://people.eecs.berkeley.edu/~shubham-goel/), [Georgios Pavlakos](https://geopavlakos.github.io/), [Jathushan Rajasegaran](http://people.eecs.berkeley.edu/~jathushan/), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/)<sup>\*</sup>, [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/)<sup>\*</sup>

[![arXiv](https://img.shields.io/badge/arXiv-2112.04477-00ff00.svg)](arXiv preprint 2023)       [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)]([https://people.eecs.berkeley.edu/~jathushan/PHALP/](https://shubham-goel.github.io/4dhumans/))     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/drive/1zeHvAcvflsty2p9Hgr49-AhQjkSWCXjr?usp=sharing](https://colab.research.google.com/drive/1Ex4gE5v1bPR3evfhtG7sDHxQGsWwNwby?usp=sharing))

![teaser](assets/teaser.png)

## Download dependencies
Our demo code depends on [detectron2](https://github.com/facebookresearch/detectron2) to detect humans.
To automatically download this dependency, clone this repo using `--recursive`, or run `git submodule update --init` if you've already cloned the repository. You should see the detectron2 source code at `vendor/detectron2`.
```bash
git clone https://github.com/shubham-goel/4D-Humans.git --recursive
# OR
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans
git submodule update --init
```

## Installation
We recommend creating a clean [conda](https://docs.conda.io/) environment and installing all dependencies, as follows:
```bash
conda env create -f environment.yml
```

After the installation is complete you can activate the conda environment by running:
```
conda activate 4D-humans
```

## Download checkpoints and SMPL models
To download the checkpoints and SMPL models, run
```bash
./fetch_data.sh
```

## Run demo on images
You may now run our demo to 3D reconstruct humans in images using the following command, which will run ViTDet and HMR2.0 on all images in the specified `--img_folder` and save renderings of the reconstructions in `--out_folder`. You can also use the `--side_view` flag to additionally render the side view of the reconstructed mesh. `--batch_size` batches the images together for faster processing.
```bash
python demo.py \
    --img_folder example_data/images \
    --out_folder demo_out \
    --batch_size=48 --side_view
```

## Run demo on videos
Coming soon.

## Training and evaluation
Cmoing soon.

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [ProHMR](https://github.com/nkolot/ProHMR)
- [SPIN](https://github.com/nkolot/SPIN)
- [SMPLify-X](https://github.com/vchoutas/smplify-x)
- [HMR](https://github.com/akanazawa/hmr)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [Detectron2](https://github.com/facebookresearch/detectron2)

Additionally, we thank [StabilityAI](https://stability.ai/) for a generous compute grant that enabled this work.

## Citing
If you find this code useful for your research, please consider citing the following paper:

```
@article{4DHUMANS,
    title={Humans in 4{D}: Reconstructing and Tracking Humans with Transformers},
    author={Goel, Shubham and Pavlakos, Georgios and Rajasegaran, Jathushan and Kanazawa, Angjoo and Malik, Jitendra},
    journal={arXiv preprint},
    year={2023}
}
```
