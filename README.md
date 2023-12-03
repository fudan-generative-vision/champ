# 4DHumans: Reconstructing and Tracking Humans with Transformers
Code repository for the paper:
**Humans in 4D: Reconstructing and Tracking Humans with Transformers**
[Shubham Goel](https://people.eecs.berkeley.edu/~shubham-goel/), [Georgios Pavlakos](https://geopavlakos.github.io/), [Jathushan Rajasegaran](http://people.eecs.berkeley.edu/~jathushan/), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/)<sup>\*</sup>, [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/)<sup>\*</sup>

[![arXiv](https://img.shields.io/badge/arXiv-2305.20091-00ff00.svg)](https://arxiv.org/pdf/2305.20091.pdf)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://shubham-goel.github.io/4dhumans/)     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ex4gE5v1bPR3evfhtG7sDHxQGsWwNwby?usp=sharing)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/brjathu/HMR2.0)


![teaser](assets/teaser.png)

## Installation and Setup
First, clone the repo. Then, we recommend creating a clean [conda](https://docs.conda.io/) environment, installing all dependencies, and finally activating the environment, as follows:
```bash
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans
conda env create -f environment.yml
conda activate 4D-humans
```

If conda is too slow, you can use pip:
```bash
conda create --name 4D-humans python=3.10
conda activate 4D-humans
pip install torch
pip install -e .[all]
```

All checkpoints and data will automatically be downloaded to `$HOME/.cache/4DHumans` the first time you run the demo code.

Besides these files, you also need to download the *SMPL* model. You will need the [neutral model](http://smplify.is.tue.mpg.de) for training and running the demo code. Please go to the corresponding website and register to get access to the downloads section. Download the model and place `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` in `./data/`.

## Run demo on images
The following command will run ViTDet and HMR2.0 on all images in the specified `--img_folder`, and save renderings of the reconstructions in `--out_folder`. `--batch_size` batches the images together for faster processing. The `--side_view` flags additionally renders the side view of the reconstructed mesh, `--full_frame` renders all people together in front view, `--save_mesh` saves meshes as `.obj`s.
```bash
python demo.py \
    --img_folder example_data/images \
    --out_folder demo_out \
    --batch_size=48 --side_view --save_mesh --full_frame
```

## Run tracking demo on videos
Our tracker builds on PHALP, please install that first:
```bash
pip install git+https://github.com/brjathu/PHALP.git
```

Now, run `track.py` to reconstruct and track humans in any video. Input video source may be a video file, a folder of frames, or a youtube link:
```bash
# Run on video file
python track.py video.source="example_data/videos/gymnasts.mp4"

# Run on extracted frames
python track.py video.source="/path/to/frames_folder/"

# Run on a youtube link (depends on pytube working properly)
python track.py video.source=\'"https://www.youtube.com/watch?v=xEH_5T9jMVU"\'
```
The output directory (`./outputs` by default) will contain a video rendering of the tracklets and a `.pkl` file containing the tracklets with 3D pose and shape. Please see the [PHALP](https://github.com/brjathu/PHALP) repository for details.

## Training
Download the [training data](https://www.dropbox.com/sh/mjdwu59fxuhls5h/AACQ6FCGSrggUXmRzuubRHXIa) to `./hmr2_training_data/`, then start training using the following command:
```
bash fetch_training_data.sh
python train.py exp_name=hmr2 data=mix_all experiment=hmr_vit_transformer trainer=gpu launcher=local
```
Checkpoints and logs will be saved to `./logs/`. We trained on 8 A100 GPUs for 7 days using PyTorch 1.13.1 and PyTorch-Lightning 1.8.1 with CUDA 11.6 on a Linux system. You may adjust batch size and number of GPUs per your convenience.

## Evaluation
Download the [evaluation metadata](https://www.dropbox.com/scl/fi/kl79djemdgqcl6d691er7/hmr2_evaluation_data.tar.gz?rlkey=ttmbdu3x5etxwqqyzwk581zjl) to `./hmr2_evaluation_data/`. Additionally, download the Human3.6M, 3DPW, LSP-Extended, COCO, and PoseTrack dataset images and update the corresponding paths in  `hmr2/configs/datasets_eval.yaml`.

Run evaluation on multiple datasets as follows, results are stored in `results/eval_regression.csv`. 
```bash
python eval.py --dataset 'H36M-VAL-P2,3DPW-TEST,LSP-EXTENDED,POSETRACK-VAL,COCO-VAL' 
```

By default, our code uses the released checkpoint (mentioned as HMR2.0b in the paper). To use the HMR2.0a checkpoint, you may download and untar from [here](https://people.eecs.berkeley.edu/~jathushan/projects/4dhumans/hmr2a_model.tar.gz)

## Preprocess code
To preprocess LSP Extended and Posetrack into metadata zip files for evaluation, see `hmr2/datasets/preprocess`.

Training data preprocessing coming soon.

## Open Source Contributions
[carlosedubarreto](https://github.com/carlosedubarreto/) has created a tutorial to import 4D Humans in Blender: https://www.patreon.com/posts/86992009

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

```bibtex
@inproceedings{goel2023humans,
    title={Humans in 4{D}: Reconstructing and Tracking Humans with Transformers},
    author={Goel, Shubham and Pavlakos, Georgios and Rajasegaran, Jathushan and Kanazawa, Angjoo and Malik, Jitendra},
    booktitle={ICCV},
    year={2023}
}
```
