from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = (ROOT / 'README.md').read_text()

setup(
    name='neureps',
    packages=find_packages(include=['neureps']),
    version='0.0.1',
    license='MIT',
    description='A PyTorch toolkit for Neural Representations',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Kristian Klemon',
    author_email='kristian.klemon@gmail.com',
    url='https://github.com/kklemon/neureps',
    keywords=['artificial intelligence', 'deep learning', 'neural representations'],
    install_requires=['torch']
)
