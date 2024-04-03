<h1 align='Center'>Champ: Controllable and Consistent Human Image Animation with 3D Parametric Guidance</h1>

<div align='Center'>
    <a href='https://github.com/ShenhaoZhu' target='_blank'>Shenhao Zhu</a><sup>*1</sup>&emsp;
    <a href='https://github.com/Leoooo333' target='_blank'>Junming Leo Chen</a><sup>*2</sup>&emsp;
    <a href='https://github.com/daizuozhuo' target='_blank'>Zuozhuo Dai</a><sup>3</sup>&emsp;
    <a href='https://ai3.fudan.edu.cn/info/1088/1266.htm' target='_blank'>Yinghui Xu</a><sup>2</sup>&emsp;
    <a href='https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html' target='_blank'>Xun Cao</a><sup>1</sup>&emsp;
    <a href='https://yoyo000.github.io/' target='_blank'>Yao Yao</a><sup>1</sup>&emsp;
    <a href='http://zhuhao.cc/home/' target='_blank'>Hao Zhu</a><sup>+1</sup>&emsp;
    <a href='https://sites.google.com/site/zhusiyucs/home' target='_blank'>Siyu Zhu</a><sup>+2</sup>
</div>
<div align='Center'>
    <sup>1</sup>Nanjing University <sup>2</sup>Fudan University <sup>3</sup>Alibaba Group
</div>
<div align='Center'>
    <sup>*</sup>Equal Contribution
    <sup>+</sup>Corresponding Author
</div>

<div align='Center'>
    <a href='https://fudan-generative-vision.github.io/champ/#/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='https://arxiv.org/abs/2403.14781'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://youtu.be/2XVsy9tQRAY'><img src='https://badges.aleen42.com/src/youtube.svg'></a>
</div>

https://github.com/fudan-generative-vision/champ/assets/82803297/b4571be6-dfb0-4926-8440-3db229ebd4aa

# 🏆 Framework

![framework](assets/framework.jpg)


# 🔥 News

- **`2024/04/02`**: ✨✨✨SMPL & Rendering scripts released! Champ your dance videos now.💃🤸‍♂️🕺
  
- **`2024/03/30`**: 🚀🚀🚀Watch this amazing [video tutorial](https://www.youtube.com/watch?app=desktop&v=cbElsTBv2-A) from [Toy](https://twitter.com/toyxyz3). It's based on the unofficial easy [Champ ComfyUI ](https://github.com/kijai/ComfyUI-champWrapper?tab=readme-ov-file)without SMPL from [Kijai](https://github.com/kijai)🥳.
  
- **`2024/03/27`**: Cool Demo on [replicate](https://replicate.com/camenduru/champ)🌟, Thanks [camenduru](https://github.com/camenduru)!👏


# 🐟 Installation

- System requirement: Ubuntu20.04/Windows 11, Cuda 12.1
- Tested GPUs: A100, RTX3090

Git clone Champ with following command:
```bash
  git clone --recurse-submodules https://github.com/fudan-generative-vision/champ
```


Create conda environment:

```bash
  conda create -n champ python=3.10
  conda activate champ
```

## Install packages with `pip`

```bash
  pip install -r requirements.txt
```

## Install packages with [poetry](https://python-poetry.org/)
> If you want to run this project on a Windows device, we strongly recommend to use `poetry`.
```shell
poetry install --no-root
```
## Install 4D-Humans

Champ use the great work [4D-Humans](https://github.com/shubham-goel/4D-Humans) to fit SMPL on inputs. Please follow their instructions `Installation` to set it up and `Run demo on images` to download checkpoints. Note that we have a fork in `Champ/4D-Humans`, so you don't need to clone the original repository.

# 💾 Download pretrained models

1. Download pretrained weight of base models:

   - [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)
   - [DWPose](https://github.com/IDEA-Research/DWPose?tab=readme-ov-file#-dwpose-for-controlnet)
2. Download our checkpoints: \

Our [checkpoints](https://huggingface.co/fudan-generative-ai/champ/tree/main) consist of denoising UNet, guidance encoders, Reference UNet, and motion module.

Finally, these pretrained models should be organized as follows:

```text
./pretrained_models/
|-- champ
|   |-- denoising_unet.pth
|   |-- guidance_encoder_depth.pth
|   |-- guidance_encoder_dwpose.pth
|   |-- guidance_encoder_normal.pth
|   |-- guidance_encoder_semantic_map.pth
|   |-- reference_unet.pth
|   `-- motion_module.pth
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
|-- DWPose
|   |-- dw-ll_ucoco_384.onnx
|   `-- yolox_l.onnx
|-- sd-vae-ft-mse
|   |-- config.json
|   |-- diffusion_pytorch_model.bin
|   `-- diffusion_pytorch_model.safetensors
`-- stable-diffusion-v1-5
    |-- feature_extractor
    |   `-- preprocessor_config.json
    |-- model_index.json
    |-- unet
    |   |-- config.json
    |   `-- diffusion_pytorch_model.bin
    `-- v1-inference.yaml
```

# 🐳 Inference

We have provided several sets of [example data](https://huggingface.co/fudan-generative-ai/champ/tree/main) for inference. Please first download and place them in the `example_data` folder. 

Here is the command for inference:

```bash
  python inference.py --config configs/inference.yaml
```

If using `poetry`, command is 
```shell
poetry run python inference.py --config configs/inference.yaml
```

Animation results will be saved in `results` folder. You can change the reference image or the guidance motion by modifying `inference.yaml`.

You can also extract the driving motion from any videos and then render with Blender. We will later provide the instructions and scripts for this.

Note: The default motion-01 in `inference.yaml` has more than 500 frames and takes about 36GB VRAM. If you encounter VRAM issues, consider switching to other example data with less frames.

# 💃 SMPL & Rendering

Try Champ with your dance videos! It may take time to setup the environment, follow the instruction step by step🐢, report issue when necessary.

## Preprocess
Use [ffmpeg](https://ffmpeg.org/) to extract frames from video. For example:

```
ffmpeg -i driving_videos/Video_1/Your_Video.mp4 -c:v png driving_videos/Video_1/images/%04d.png 
```

Please organize your driving videos and reference images like this:

```shell
|-- driving_videos
    |-- Video_1
        |-- images
        	|-- 0000.png
        		 ...
        	|-- 0020.png
        		 ...
    |-- Video_2
        |-- images
        	|-- 0000.png
        		 ...
    ...
    |-- Video_n

|-- reference_imgs
    |-- images
    	|-- your_ref_img_A.png
    	|-- your_ref_img_B.png
                ...
```

## SMPL

### Fit SMPL

Make sure you have organized directory as above. Substitute your path as **absolute path** in following command:

``` 
python inference_smpl.py  --reference_imgs_folder test_smpl/reference_imgs --driving_videos_folder test_smpl/driving_videos --device YOUR_GPU_ID
```
Once finished, you can check `reference_imgs/visualized_imgs` to see the overlay results. To better fit some extreme figures, you may also append `--figure_scale ` to manually change the figure(or shape) of predicted SMPL, from `-10`(extreme fat) to `10`(extreme slim).

### Smooth SMPL (optional)

**TODO**: Coming Soon.

### Transfer SMPL

Replace with **absolute path** in following command:

```shell
python transfer_smpl.py --reference_path test_smpl/reference_imgs/smpl_results/ref.npy --driving_path test_smpl/driving_videos/Video_1 --output_folder test_smpl/transfer_result --figure_transfer --view_transfer
```

Append `--figure_transfer` when you want the result matches the reference SMPL's figure, and `--view_transfer` to transform the driving SMPL onto reference image's camera space.

## Rendering

First of all, install [Blender](https://www.blender.org/download/) in your Server or PC.

Replace with **absolute path** in following command:

```shell
blender smpl_rendering.blend --background --python rendering.py --driving_path test_smpl/transfer_result/smpl_results --reference_path test_smpl/reference_imgs/images/ref.png
```

This will rendering in CPU on default. Append `--device YOUR_GPU_ID` to select a GPU for rendering. 

### Rendering DWPose

Make sure you have finished SMPL rendering. Replace with **absolute path** in following command:

```
python inference_dwpose.py --imgs_path test_smpl/transfer_result --device YOUR_GPU_ID
```



# 👏 Acknowledgements

We thank the authors of [MagicAnimate](https://github.com/magic-research/magic-animate), [Animate Anyone](https://github.com/HumanAIGC/AnimateAnyone), and [AnimateDiff](https://github.com/guoyww/AnimateDiff) for their excellent work. Our project is built upon [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), [4D-Humans](https://github.com/shubham-goel/4D-Humans), [DWPose](https://github.com/IDEA-Research/DWPose) and we are grateful for their open-source contributions.

# 🕒 Roadmap

Visit [our roadmap](https://github.com/fudan-generative-vision/champ/blob/master/docs/ROADMAP.md) to preview the future of Champ.

# 🌟 Citation

If you find our work useful for your research, please consider citing the paper:

```
@misc{zhu2024champ,
      title={Champ: Controllable and Consistent Human Image Animation with 3D Parametric Guidance},
      author={Shenhao Zhu and Junming Leo Chen and Zuozhuo Dai and Yinghui Xu and Xun Cao and Yao Yao and Hao Zhu and Siyu Zhu},
      year={2024},
      eprint={2403.14781},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# 👋 Opportunities available

Multiple research positions are open at the **Generative Vision Lab, Fudan University**! Include:

- Research assistant
- Postdoctoral researcher
- PhD candidate
- Master students

Interested individuals are encouraged to contact us at [siyuzhu@fudan.edu.cn](mailto://siyuzhu@fudan.edu.cn) for further information.
