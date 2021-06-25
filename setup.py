#!/usr/bin/env python

from setuptools import find_packages, setup

from dodo_detector import __version__

setup(
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Environment :: GPU :: NVIDIA CUDA :: 9.0",
        "Environment :: GPU :: NVIDIA CUDA :: 9.2",
        "Environment :: GPU :: NVIDIA CUDA :: 10.0",
        "Environment :: GPU :: NVIDIA CUDA :: 10.1",
        "Environment :: GPU :: NVIDIA CUDA :: 10.2",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    name="dodo_detector",
    version=__version__,
    description="Object detection package",
    author="Douglas De Rizzo Meneghetti",
    author_email="douglasrizzom@gmail.com",
    packages=find_packages(exclude=["contrib", "docs"]),
    install_requires=["numpy", "tqdm", "imutils", "opencv-python"],
    extras_require={
        "dev": ["flake8", "yapf"],
        "testing": ["nose", "pillow", "matplotlib"],
        "docs": ["Sphinx", "numpydoc", "sphinx_autodoc_annotation", "sphinx_rtd_theme"],
        "tf1-cpu": ["tensorflow==1.15"],
        "tf2-cpu": ["tensorflow>=2.2.0"],
        "tf1-gpu": ["tensorflow-gpu==1.15"],
        "tf2-gpu": ["tensorflow-gpu>=2.2.0"],
    },
    license="GPLv3",
)
