<h1 align='center'>Champ：具有3D參數引導的可控且一致的人像動畫框架</h1>

<div align='Center'>
    <a href='https://github.com/ShenhaoZhu' target='_blank'>Shenhao Zhu</a><sup>*1</sup>&emsp;
    <a href='https://github.com/Leoooo333' target='_blank'>Junming Leo Chen</a><sup>*2</sup>&emsp;
    <a href='https://github.com/daizuozhuo' target='_blank'>Zuozhuo Dai</a><sup>3</sup>&emsp;
    <a href='https://ai3.fudan.edu.cn/info/1088/1266.htm' target='_blank'>徐盈煇</a><sup>2</sup>&emsp;
    <a href='https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html' target='_blank'>Xun Cao</a><sup>1</sup>&emsp;
    <a href='https://yoyo000.github.io/' target='_blank'>姚遙</a><sup>1</sup>&emsp;
    <a href='http://zhuhao.cc/home/' target='_blank'>朱昊</a><sup>+1</sup>&emsp;
    <a href='https://sites.google.com/site/zhusiyucs/home' target='_blank'>Siyu Zhu</a><sup>+2</sup>
</div>
<div align='center'>
    <sup>1</sup>南京大學 <sup>2</sup>復旦大學 <sup>3</sup>阿里巴巴集團
</div>
<div align='center'>
    <sup>*</sup>平等貢獻
    <sup>+</sup>通訊作者
</div>

<div align='center'>
    <a href='https://fudan-generative-vision.github.io/champ/#/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='https://arxiv.org/abs/2403.14781'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://youtu.be/2XVsy9tQRAY'><img src='https://badges.aleen42.com/src/youtube.svg'></a>
    <a href="README.md"><img src="https://img.shields.io/badge/english-document-white.svg" alt="EN doc"></a>
</div>

https://github.com/fudan-generative-vision/champ/assets/82803297/b4571be6-dfb0-4926-8440-3db229ebd4aa

# 框架

![framework](assets/framework.jpg)

# 安裝

- 系統要求：Ubuntu20.04
- 測試過的GPU：A100、RTX3090

創建`conda`環境：

```bash
  conda create -n champ python=3.10
  conda activate champ
```

用`pip`安裝套件：
```bash
  pip install -r requirements.txt
```

# 下載預訓練模型

1. 下載基本模型的預訓練權重：

   - [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)

2. 下載我們的檢查點： \

我們的[檢查點](https://huggingface.co/fudan-generative-ai/champ/tree/main)包括去噪UNet、引導編碼器、參考UNet和運動模組。

最終，這些預訓練模型應該按如下方式組織：

```
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

# 推斷

我們提供了幾組[示例數據](https://huggingface.co/fudan-generative-ai/champ/tree/main)用於推斷。請首先下載並將它們放在 `example_data` 文件夾中。

這是推斷的命令：

```bash
  python inference.py --config configs/inference.yaml
```

動畫結果將保存在 `results` 文件夾中。您可以通過修改 `inference.yaml` 更改參考圖像或引導運動。

您還可以從任何視頻中提取驅動運動，然後使用Blender渲染。我們稍後會提供相關指南和腳本。

注意：`inference.yaml` 中的默認動作-01有500多幀，需要約36GB的VRAM。如果您遇到VRAM問題，請考慮切換到其他帶有較少幀的示例數據。

# 鳴謝

我們感謝 [MagicAnimate](https://github.com/magic-research/magic-animate)、[Animate Anyone](https://github.com/HumanAIGC/AnimateAnyone) 和 [AnimateDiff](https://github.com/guoyww/AnimateDiff) 的作者們的出色工作。我們的項目是建立在 [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) 的基礎上的，我們感謝他們的開源貢獻。

# 路線圖

訪問[路線圖](docs/ROADMAP.zh_TW.md)以預覽Champ的未來。

# 引用

如果您認為我們的工作對您的研究有用，請考慮引用以下論文：

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

# 機會

**復旦大學生成視覺實驗室**開放多個研究職位！包括：

- 研究助理
- 博士後研究員
- 博士候選人
- 碩士生

有興趣的人士可通過 [siyuzhu@fudan.edu.cn](mailto://siyuzhu@fudan.edu.cn) 聯繫我們以獲取更多信息。