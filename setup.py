#!/usr/bin/env python3

from setuptools import setup, find_packages
from dodo_detector import __version__

setup(
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Topic :: Education :: Testing',
        'License :: OSI Approved :: GNU General Public License v3 (LGPLv3)'
    ],
    name='dodo_detector',
    version=__version__,
    description='Object detection package',
    author='Douglas De Rizzo Meneghetti',
    author_email='douglasrizzom@gmail.com',
    packages=find_packages(exclude=['contrib', 'docs']),
    install_requires=['numpy', 'tqdm', 'imutils', 'opencv-python'],
    extras_require={
        'tf-cpu': [
            'tensorflow',
        ],
        'tf-gpu': [
            'tensorflow-gpu',
        ],
        'testing': ['nose'],
        'docs': ['Sphinx', 'numpydoc', 'sphinx_autodoc_annotation', 'sphinx_rtd_theme']
    },
    license='GPLv3'
)
