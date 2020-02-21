#!/usr/bin/env python

from setuptools import setup, find_packages
from dodo_detector import __version__
from pkg_resources import DistributionNotFound, get_distribution

def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

install_deps = ['numpy', 'tqdm', 'imutils', 'opencv-python', 'tensorflow>=1.13, <=1.15.2']
if get_dist('tensorflow') is None and get_dist('tensorflow-gpu') is not None:
    del install_deps[-1]
    install_deps.append('tensorflow-gpu>=1.13, <=1.15.2')

setup(
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ],
    name='dodo_detector',
    version=__version__,
    description='Object detection package',
    author='Douglas De Rizzo Meneghetti',
    author_email='douglasrizzom@gmail.com',
    packages=find_packages(exclude=['contrib', 'docs']),
    install_requires=install_deps,
    extras_require={
        'testing': ['nose', 'pillow', 'matplotlib'],
        'docs': ['Sphinx', 'numpydoc', 'sphinx_autodoc_annotation', 'sphinx_rtd_theme']
    },
    license='GPLv3'
)
