#!/usr/bin/env python

import os
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from tqdm import tqdm

from dodo_detector.ObjectDetector import ObjectDetector


class KeypointObjectDetector(ObjectDetector):

    def __init__(self, database_path, detector_type='RootSIFT', matcher_type='BF', min_points=10):
        """
        Object detector based on keypoints. This class depends on OpenCV SIFT and SURF feature detection algorithms,
        as well as the brute-force and FLANN-based feature matchers.

        :param database_path: Path to the top-level directory containing subdirectories, each subdirectory named after a class of objects and containing images of that object.
        :param detector_type: either `SURF`, `SIFT` or `RootSIFT`
        :param matcher_type: either `BF` for brute-force matcher or `FLANN` for flann-based matcher
        :param min_points: minimum number of keypoints necessary for an object to be considered detected in an image
        """
        self.current_frame = 0
        self.detector_type = detector_type

        # get the directory where object textures are stored
        self.database_path = database_path

        if self.database_path[-1] != '/':
            self.database_path += '/'

        # minimum number of features for a KNN match to consider that an object has been found
        self.min_points = min_points

        # create the detector
        if self.detector_type in ['SIFT', 'RootSIFT']:
            self.detector = cv2.xfeatures2d.SIFT_create()
        elif self.detector_type == 'SURF':
            self.detector = cv2.xfeatures2d.SURF_create()

        # get which OpenCV feature matcher the user wants
        if matcher_type == 'BF':
            self.matcher = cv2.BFMatcher()
        elif matcher_type == 'FLANN':
            flann_index_kdtree = 0
            index_params = dict(algorithm=flann_index_kdtree, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # store object classes in a list
        # each directory in the object database corresponds to a class
        self.objects = [os.path.basename(d) for d in os.listdir(self.database_path)]

        # minimum object dimensions in pixels
        self.min_object_height = 10
        self.min_object_width = 10
        self.min_object_area = self.min_object_height * self.min_object_width

        # initialize the frame counter for each object class at 0
        self.object_counters = {}
        for ob in self.objects:
            self.object_counters[ob] = 0

        # load features for each texture and store the image,
        # its keypoints and corresponding descriptor
        self.object_features = {}

        for obj in self.objects:
            self.object_features[obj] = self._load_features(obj)

    def _load_features(self, object_name):
        """
        Given the name of an object class from the image database directory, this method iterates through all the images contained in that directory and extracts their keypoints and descriptors using the desired feature detector

        :param object_name: the name of an object class, whose image directory is contained inside the image database directory
        :return: a list of tuples, each tuple containing the processed image as a grayscale numpy.ndarray, its keypoints and desciptors
        """
        img_files = [
            join(self.database_path + object_name + '/', f) for f in listdir(self.database_path + object_name + '/')
            if isfile(join(self.database_path + object_name + '/', f))
        ]

        pbar = tqdm(desc=object_name, total=len(img_files))

        # extract the keypoints from all images in the database
        features = []
        for img_file in img_files:
            pbar.update()
            img = cv2.imread(img_file)

            # scaling_factor = 640 / img.shape[0]
            if img.shape[0] > 1000:
                img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)

            # find keypoints and descriptors with the selected feature detector
            keypoints, descriptors = self._detectAndCompute(img)

            features.append((img, keypoints, descriptors))

        return features

    def _detect_object(self, name, img_features, scene):
        """

        :param name: name of the object class
        :param img_features: a list of tuples, each tuple containing three elements: an image, its keypoints and its descriptors.
        :param scene: the image where the object `name` will be detected
        :return: a tuple containing two elements: the homography matrix and the coordinates of a rectangle containing the object in a list `[xmin, ymin, xmax, ymax]`
        """
        scene_kp, scene_descs = self._detectAndCompute(scene)

        for img_feature in img_features:
            obj_image, obj_keypoints, obj_descriptors = img_feature

            if obj_descriptors is not None and len(obj_descriptors) > 0 and scene_descs is not None and len(scene_descs) > 0:
                matches = self.matcher.knnMatch(obj_descriptors, scene_descs, k=2)

                # store all the good matches as per Lowe's ratio test
                good = []
                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        if m.distance < 0.7 * n.distance:
                            good.append(m)

                # an object was detected
                if len(good) > self.min_points:
                    self.object_counters[name] += 1
                    source_points = np.float32([obj_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    destination_points = np.float32([scene_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)
                    h, w, c = obj_image.shape

                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                    # dst contains the coordinates of the vertices of the drawn rectangle
                    dst = np.int32(cv2.perspectiveTransform(pts, M))

                    # get the min and max x and y coordinates of the object
                    xmin = min(dst[x, 0][0] for x in range(4))
                    xmax = max(dst[x, 0][0] for x in range(4))
                    ymin = min(dst[x, 0][1] for x in range(4))
                    ymax = max(dst[x, 0][1] for x in range(4))

                    # transform homography into a simpler data structure
                    dst = np.array([point[0] for point in dst], dtype=np.int32)

                    # check the object's height, width and area according to the parameters in the config file
                    if w < self.min_object_width or h < self.min_object_height or w * h < self.min_object_area:
                        break

                    # returns the homography and a rectangle containing o object
                    return dst, [xmin, ymin, xmax, ymax]

        return None, None

    def _detectAndCompute(self, image):
        """
        Detects keypoints and generates descriptors according to the desired algorithm
        
        :param image: a numpy.ndarray containing the image whose keypoints and descriptors will be processed
        :return: a tuple containing keypoints and descriptors
        """
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        if self.detector_type == 'RootSIFT' and len(keypoints) > 0:
            # Transforms SIFT descriptors into RootSIFT descriptors
            # apply the Hellinger kernel by first L1-normalizing
            # and taking the square-root
            eps = 1e-7
            descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
            descriptors = np.sqrt(descriptors)

        return keypoints, descriptors

    def from_image(self, frame):
        self.current_frame += 1
        # Our operations on the frame come here

        detected_objects = {}

        for object_name in self.object_features:
            homography, rct = self._detect_object(object_name, self.object_features[object_name], frame)

            if rct is not None:
                xmin = rct[0]
                ymin = rct[1]
                xmax = rct[2]
                ymax = rct[3]

                if object_name not in detected_objects:
                    detected_objects[object_name] = []

                detected_objects[object_name].append((ymin, xmin, ymax, xmax))

                text_point = (homography[0][0], homography[1][1])
                homography = homography.reshape((-1, 1, 2))
                cv2.polylines(frame, [homography], True, (0, 255, 255), 10)

                cv2.putText(frame, object_name + ': ' + str(self.object_counters[object_name]), text_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 0), 2)

        return frame, detected_objects
