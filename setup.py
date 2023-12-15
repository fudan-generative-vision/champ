from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='HMR2 as a package',
    name='hmr2',
    packages=find_packages(),
    install_requires=[
        'gdown',
        'numpy',
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
        'pandas',
        'chumpy @ git+https://github.com/mattloper/chumpy',
    ],
    extras_require={
        'all': [
            'detectron2 @ git+https://github.com/facebookresearch/detectron2',
        ],
    },
)
