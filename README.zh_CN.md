# Champ: 可控制和一致性的3D参数引导的人像动画

<div align='Center'>
    <a href='https://github.com/ShenhaoZhu' target='_blank'>Shenhao Zhu</a><sup>*1</sup>&emsp;
    <a href='https://github.com/Leoooo333' target='_blank'>Junming Leo Chen</a><sup>*2</sup>&emsp;
    <a href='https://github.com/daizuozhuo' target='_blank'>Zuozhuo Dai</a><sup>3</sup>&emsp;
    <a href='https://ai3.fudan.edu.cn/info/1088/1266.htm' target='_blank'>徐盈辉</a><sup>2</sup>&emsp;
    <a href='https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html' target='_blank'>Xun Cao</a><sup>1</sup>&emsp;
    <a href='https://yoyo000.github.io/' target='_blank'>Yao Yao</a><sup>1</sup>&emsp;
    <a href='http://zhuhao.cc/home/' target='_blank'>朱昊</a><sup>+1</sup>&emsp;
    <a href='https://sites.google.com/site/zhusiyucs/home' target='_blank'>Siyu Zhu</a><sup>+2</sup>
</div>
<div align='center'>
    <sup>1</sup>南京大学 <sup>2</sup>复旦大学 <sup>3</sup>阿里巴巴集团
</div>
<div align='center'>
    <sup>*</sup>平等贡献
    <sup>+</sup>通讯作者
</div>

<div align='center'>
    <a href='https://fudan-generative-vision.github.io/champ/#/'><img src='https://img.shields.io/badge/项目-页面-Green'></a>
    <a href='https://arxiv.org/abs/2403.14781'><img src='https://img.shields.io/badge/论文-Arxiv-red'></a>
    <a href='https://youtu.be/2XVsy9tQRAY'><img src='https://badges.aleen42.com/src/youtube.svg'></a>
    <a href="README.md"><img src="https://img.shields.io/badge/english-document-white.svg" alt="EN doc"></a>
</div>

https://github.com/fudan-generative-vision/champ/assets/82803297/b4571be6-dfb0-4926-8440-3db229ebd4aa

# 框架

![框架](assets/framework.jpg)

# 安装

- 系统要求：Ubuntu20.04
- 测试GPU：A100、RTX3090

创建conda环境：

```bash
  conda create -n champ python=3.10
  conda activate champ
```

使用 `pip` 安装包：

```bash
  pip install -r requirements.txt
```

# 下载预训练模型

1. 下载基础模型的预训练权重：

   - [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)

2. 下载我们的检查点：\

我们的[检查点](https://huggingface.co/fudan-generative-ai/champ/tree/main)包括降噪UNet、引导编码器、参考UNet和运动模块。

最后，这些预训练模型应该如下所示组织：

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

# 推理

我们已经提供了几组[示例数据](https://huggingface.co/fudan-generative-ai/champ/tree/main)供推理使用。请先下载并将它们放置在 `example_data` 文件夹中。

这是推理的命令：

```bash
  python inference.py --config configs/inference.yaml
```

动画结果将保存在 `results` 文件夹中。您可以通过修改 `inference.yaml` 来更改参考图像或指导运动。

您还可以从任何视频中提取驱动运动，然后使用Blender渲染。我们稍后将提供指导和脚本。

注意：`inference.yaml` 中的默认motion-01有500多帧，需要大约36GB的显存。如果遇到显存问题，请考虑切换到其他帧数较少的示例数据。

# 致谢

我们感谢[MagicAnimate](https://github.com/magic-research/magic-animate)、[Animate Anybody](https://github.com/HumanAIGC/AnimateAnyone)和[AnimateDiff](https://github.com/guoyww/AnimateDiff)的作者们的杰出工作。我们的项目是基于[Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone)建立的，我们非常感谢他们的开源贡献。

# 路线图

访问[路线图](docs/ROADMAP.zh_CN.md)以预览Champ的未来。

# 引用

如果您发现我们的工作对您的研究有帮助，请考虑引用这篇论文：

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

# 可用机会

复旦大学**生成式视觉实验室**有多个研究职位开放！包括：

- 研究助理
- 博士后研究员
- 博士候选人
- 硕士研究生

感兴趣的个人欢迎通过[siyuzhu@fudan.edu.cn](mailto://siyuzhu@fudan.edu.cn)与我们联系以获取更多信息。