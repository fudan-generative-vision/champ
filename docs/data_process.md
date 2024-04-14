# 💃 SMPL & Rendering

Try Champ with your dance videos! It may take time to setup the environment, follow the instruction step by step🐢, report issue when necessary. 
> Notice that it has been tested only on Linux. Windows user may encounter some environment issues for pyrender.


## Install dependencies

1. Install [4D-Humans](https://github.com/shubham-goel/4D-Humans)
    ```shell
    git clone https://github.com/shubham-goel/4D-Humans.git
    conda create --name 4D-humans python=3.10
    conda activate 4D-humans
    pip install -e 4D-Humans
    ```

    or you can install via pip by a simple command
    ```shell
    pip install git+https://github.com/shubham-goel/4D-Humans
    ```

2. Install [detectron2](https://github.com/facebookresearch/detectron2)
    
    gcc and g++ 12 is necessary to build detectron2

    ```shell
    conda install -c conda-forge gcc=12 gxx=12
    ```
    Then
    ```shell
    git clone https://github.com/facebookresearch/detectron2

    pip install -e detectron2
    ```
    or you can install via pip by a simple command
    ```shell
    pip install git+https://github.com/facebookresearch/detectron2
    ```

3. Install [Blender](https://www.blender.org/)

    You can download Blender 3.x version for your operation system from this url [https://download.blender.org/release/Blender3.6](https://download.blender.org/release/Blender3.6/).

## Download models

1. [DWPose for controlnet](https://github.com/IDEA-Research/DWPose?tab=readme-ov-file#-dwpose-for-controlnet)

    First, you need to download our Pose model dw-ll_ucoco_384.onnx ([baidu](https://pan.baidu.com/s/1nuBjw-KKSxD_BkpmwXUJiw?pwd=28d7), [google](https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing)) and Det model yolox_l.onnx ([baidu](https://pan.baidu.com/s/1fpfIVpv5ypo4c1bUlzkMYQ?pwd=mjdn), [google](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view)), then put them into `${PROJECT_ROOT}/annotator/ckpts/`.


2. HMR2 checkpoints

    ```shell
    python -m scripts.pretrained_models.download --hmr2
    ```
3. Detectron2 model

    ```shell
    python -m scripts.pretrained_models.download --detectron2
    ```
4. SMPL model
    You also need to download the SMPL model.
    ```shell
    wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
    mkdir -p 4D-Humans/data/
    mv basicModel_neutral_lbs_10_207_0_v1.0.0.pkl 4D-Humans/data/
    ```


## Produce motion data


1. Prepare video

    Prepare a "dancing" video, and use `ffmpeg` to split it into frame images:
    ```shell
    ffmpeg -i your_video_file.mp4 -c:v png driving_videos/Video_1/images/%04d.png
    ```

2. Fit SMPL

    Make sure you have splitted the video into frames and organized the image files as below:
    ```shell
    |-- driving_videos
        |-- your_video_1
            |-- images
                |-- 0000.png
                    ...
                |-- 0020.png
                    ...
        |-- your_video_2
            |-- images
                |-- 0000.png
                    ...
        ...

    |-- reference_imgs
        |-- images
            |-- your_ref_img_A.png
            |-- your_ref_img_B.png
                    ...
    ```

    Then run script below to fit SMPL on reference images and driving videos:

    ```shell
    python -m scripts.data_processors.smpl.generate_smpls --reference_imgs_folder reference_imgs --driving_video_path driving_videos/your_video_1 --device YOUR_GPU_ID
    ```

    Once finished, you can check `reference_imgs/visualized_imgs` to see the overlay results. To better fit some extreme figures, you may also append `--figure_scale ` to manually change the figure(or shape) of predicted SMPL, from `-10`(extreme fat) to `10`(extreme slim).


3. Smooth SMPL (optional)

    **TODO:** Coming Soon.

4. Transfer SMPL

    ```shell
    python -m scripts.data_processors.smpl.smpl_transfer --reference_path reference_imgs/smpl_results/your_ref_img_A.npy --driving_path driving_videos/your_video_1 --output_folder transferd_result --figure_transfer --view_transfer
    ```

    Append `--figure_transfer` when you want the result matches the reference SMPL's figure, and `--view_transfer` to transform the driving SMPL onto reference image's camera space.


5. Render SMPL via Blender

    ```shell
    blender scripts/data_processors/smpl/blend/smpl_rendering.blend --background --python scripts/data_processors/smpl/render_condition_maps.py --driving_path transferd_result/smpl_results --reference_path reference_imgs/images/your_ref_img_A.npy
    ```

    This will rendering in CPU on default. Append `--device YOUR_GPU_ID` to select a GPU for rendering. It will skip the exsiting rendered frames under the `transferd_result`. Keep it in mind when you want to overwrite with new rendering results. Ignore the warning message like `unknown argument` printed by Blender.

6. Render DWPose
    Clone [DWPose](https://github.com/IDEA-Research/DWPose)

    DWPose is required by `scripts/data_processors/dwpose/generate_dwpose.py`. You need clone this repo to the specific directory `DWPose` by command below:

    ```shell
    git clone https://github.com/IDEA-Research/DWPose.git DWPose
    conda activate champ
    ```
    Then 
    ```shell
    python -m scripts.data_processors.dwpose.generate_dwpose --input transferd_result/normal --output transferd_result/dwpose
    ```

Now, the `transferd_result` is prepared to be used in Champ🥳!