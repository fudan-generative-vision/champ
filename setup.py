from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='HMR2 as a package',
    name='hmr2',
    packages=find_packages(),
    install_requires=[
        'numpy==1.23.0',
        'torch',
        'torchvision',
        'pytorch-lightning',
        'smplx==0.1.28',
        'pyrender',
        'opencv-python',
        'yacs',
        'scikit-image',
        'einops',
        'timm',
        'webdataset',
        'dill',
        'chumpy',
        'pandas',
    ],
    extras_require={
        'all': [
            'detectron2 @ git+https://github.com/facebookresearch/detectron2',
        ],
    },
)
