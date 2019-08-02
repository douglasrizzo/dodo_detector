#!/usr/bin/env python
# coding: utf-8

import tarfile
import zipfile
import unittest
from urllib.request import urlretrieve
import numpy as np
from os import listdir, remove
from PIL import Image
from tqdm import tqdm
from os.path import isdir, exists, isfile
from dodo_detector.detection import SingleShotDetector


class TheOnlyTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__model = 'faster_rcnn_resnet50_coco_2018_01_28'
        self.__tarname = 'model.tar.gz'
        self.__modeldir = 'model'
        self.__zipname = 'images.zip'
        self.__imagedir = 'images'
        self.__labelmap = 'labelmap.pbtxt'
        self.__pbar = None

    def download_progress_hook(self, selfcount, blockSize, totalSize):
        """Report download progress.
        """
        if self.__pbar is None:
            self.__pbar = tqdm(total=totalSize)
        self.__pbar.update(blockSize)

    def maybe_download_labelmap(self):
        # download label map
        if not exists(self.__labelmap):
            self.__pbar = None
            urlretrieve(
                'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt', self.__labelmap,
                self.download_progress_hook
            )

    def maybe_download_model(self):
        # download pre-trained model
        if not exists(self.__modeldir) or not isdir(self.__modeldir):
            if not exists(self.__tarname) or not isfile(self.__tarname):
                self.__pbar = None
                urlretrieve('http://download.tensorflow.org/models/object_detection/' + self.__model + '.tar.gz', self.__tarname, self.download_progress_hook)
            
            tar = tarfile.open(self.__tarname)
            for t in tar: 
                if 'frozen_inference_graph.pb' in t.name: 
                    tar.extract(t) 
            tar.close()

    def maybe_download_image_dataset(self):
        # download images
        if not exists(self.__imagedir) or not isdir(self.__imagedir):
            if not exists(self.__zipname) or not isfile(self.__zipname):
                self.__pbar = None
                urlretrieve('http://images.cocodataset.org/zips/test2017.zip', self.__zipname, self.download_progress_hook)
            zip_ref = zipfile.ZipFile(self.__zipname)
            zip_ref.extractall(self.__imagedir)
            zip_ref.close()

    def setUp(self):
        self.maybe_download_labelmap()
        self.maybe_download_model()
        self.maybe_download_image_dataset()

    def test_singleshotdetector(self):
        # create detector
        detector = SingleShotDetector(self.__model + '/frozen_inference_graph.pb', self.__labelmap)

        # load image locations into list
        ims = [self.__imagedir + '/' + im for im in listdir(self.__imagedir)]
        ims.sort()

        # run detector on every image
        for im in tqdm(ims):
            img = np.array(Image.open(im))
            marked_image, objects = detector.from_image(img)


if __name__ == '__main__':
    unittest.main()