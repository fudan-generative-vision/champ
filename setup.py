from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='HMR2 as a package',
    name='hmr2',
    packages=find_packages()
)
