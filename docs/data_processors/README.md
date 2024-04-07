
1. pip install torch==2.2.2

2. conda install -c conda-forge gcc gxx

3. pip install -r requirements.txt

4. download models
```shell
python -m scripts.pretrained_models.download --all true
```
> you need to download smpl model manually from https://smplify.is.tue.mpg.de/ and place it at pretrain_models/smpl/SMPL_NEUTRAL.pkl
