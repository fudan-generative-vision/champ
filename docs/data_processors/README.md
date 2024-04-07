
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
python -m scripts.data_processors.inference_smpl --reference_imgs_folder example_data/test_imgs --driving_videos_folder driving_videos --device 0

```

transfer_smpl
```shell
python -m scripts.data_processors.transfer_smpl --reference_path example_data/test_imgs/smpl_results/ref-01.npy --driving_path driving_videos/Video_1 --output_folder results/smpl/transfered --figure_transfer --view_transfer
```