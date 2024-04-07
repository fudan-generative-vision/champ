
1. pip install torch==2.2.2

2. conda install -c conda-forge gcc gxx

3. pip install -r requirements.txt

4. download models
```shell
python -m scripts.pretrained_models.download --all true
```
> you need to download smpl model manually from https://smplify.is.tue.mpg.de/ and place it at pretrain_models/smpl/SMPL_NEUTRAL.pkl



inference_smpl
```shell
python -m scripts.data_processors.smpl.inference --reference_imgs_folder example_data/test_imgs --driving_videos_folder driving_videos --device 0

```

transfer_smpl
```shell
python -m scripts.data_processors.smpl.transfer --reference_path example_data/test_imgs/smpl_results/ref-01.npy --driving_path driving_videos/Video_1 --output_folder results/smpl/transfered --figure_transfer --view_transfer
```

blender render
```shell
blender scripts/data_processors/smpl/blend/smpl_rendering.blend --background --python scripts/data_processors/smpl/render.py --driving_path driving_videos/Video_1/smpl_results --reference_path example_data/test_imgs/images/ref-01.png
```

dwpose
```shell
python -m scripts.data_processors.dwpose.inference --input /home/leeway/workspace/champ/github/driving_videos/Video_1/images --cpu True --output ./results/dwpose
```