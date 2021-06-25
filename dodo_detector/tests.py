#!/usr/bin/env python
# coding: utf-8

import os
import tarfile
import unittest
import zipfile
from os.path import exists, isdir, isfile
from unittest.suite import TestSuite
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from detection import ObjectDetector, TFObjectDetectorV1, TFObjectDetectorV2


class TheOnlyTestCase(unittest.TestCase):

    def __init__(self, model, tarname, modeldir, download_link, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__model = model
        self.__tarname = tarname
        self.__modeldir = modeldir
        self.__download_link = download_link
        self.__zipname = "images.zip"
        self.__imagedir = "images"
        self.__labelmap = "labelmap.pbtxt"
        self.__pbar = None

    def download_progress_hook(self, selfcount, blockSize, totalSize):
        """Report download progress."""
        if self.__pbar is None:
            self.__pbar = tqdm(total=totalSize)
        self.__pbar.update(blockSize)

    def maybe_download_labelmap(self):
        # download label map
        if not exists(self.__labelmap):
            self.__pbar = None
            urlretrieve(
                "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt",
                self.__labelmap,
                self.download_progress_hook,
            )

    def maybe_download_model(self):
        # download pre-trained model
        if not exists(self.__modeldir) or not isdir(self.__modeldir):
            if not exists(self.__tarname) or not isfile(self.__tarname):
                self.__pbar = None
                urlretrieve(self.__download_link, self.__tarname, self.download_progress_hook)

            tar = tarfile.open(self.__tarname)
            for t in tar:
                if "frozen_inference_graph.pb" in t.name or "saved_model" in t.name:
                    tar.extract(t)
            tar.close()

    def maybe_download_image_dataset(self):
        # download images
        if not exists(self.__imagedir) or not isdir(self.__imagedir):
            if not exists(self.__zipname) or not isfile(self.__zipname):
                self.__pbar = None
                urlretrieve(
                    "http://images.cocodataset.org/zips/val2017.zip",
                    self.__zipname,
                    self.download_progress_hook,
                )

            with zipfile.ZipFile(self.__zipname) as theZip:
                fileNames = theZip.namelist()
                for fileName in fileNames:
                    if fileName.endswith("jpg"):
                        if not os.path.isdir(self.__imagedir):
                            os.mkdir(self.__imagedir)
                        content = theZip.open(fileName).read()
                        image_file = open(
                            os.path.join(self.__imagedir, os.path.basename(fileName)),
                            "wb",
                        )
                        image_file.write(content)
                        image_file.close()

    def _load_images_and_run_detector(self, detector: ObjectDetector):
        # load image locations into list
        ims = [self.__imagedir + "/" + im for im in os.listdir(self.__imagedir)]
        ims.sort()

        img = np.array(Image.open(ims[0]).convert("RGB"))
        marked_image, objects = detector.from_image(img)

        # run detector on every image
        for im in tqdm(ims):
            img = np.array(Image.open(im).convert("RGB"))
            marked_image, objects = detector.from_image(img)

    def _load_image_and_run_detector(self, detector: ObjectDetector):
        img = np.array(Image.open('img.jpg').convert("RGB"))
        marked_image, objects = detector.from_image(img)

    def setUp(self):
        self.maybe_download_labelmap()
        self.maybe_download_image_dataset()
        self.maybe_download_model()

        # solves Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
        # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


class TF1TestCase(TheOnlyTestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(
            "faster_rcnn_resnet50_coco_2018_01_28", "v1_model.tar.gz", "v1_model",
            "http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz", *args, **kwargs
        )
        self.__model = "faster_rcnn_resnet50_coco_2018_01_28"
        self.__labelmap = "labelmap.pbtxt"
        self.__imagedir = "images"

    def runTest(self):
        # create detector
        detector = TFObjectDetectorV1(self.__model + "/frozen_inference_graph.pb", self.__labelmap)
        self._load_image_and_run_detector(detector)


class TF2TestCase(TheOnlyTestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(
            "efficientdet_d0_coco17_tpu-32", "v2_model.tar.gz", "v2_model",
            "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz", *args, **kwargs
        )
        self.__model = "efficientdet_d0_coco17_tpu-32"
        self.__labelmap = "labelmap.pbtxt"
        self.__imagedir = "images"

    def runTest(self):
        # create detector
        detector = TFObjectDetectorV2(self.__model + "/saved_model", self.__labelmap)
        self._load_image_and_run_detector(detector)


if __name__ == "__main__":
    test_case = TF2TestCase() if tf.python.tf2.enabled() else TF1TestCase
    unittest.TextTestRunner().run(test_case)
    # unittest.main()
